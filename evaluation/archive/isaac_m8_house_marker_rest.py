#!/usr/bin/env python3
"""住宅場景：代理物靜置的最小診斷（Codex 20261112；無機器人、不施力）。同配置（地面＋偏航抽屜櫃＋代理物），連續 3 s，**逐物理步**記錄模擬時間、位置、姿態、線／角速度讀回。
比較位姿差分速度與讀回速度。原 2 s 判準保留並在 2 s 時刻照原規則判一次；不改摩擦、阻尼、門檻；原失敗（drawer_check r1 的 B 項）不被取代。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_marker_rest.py --out <dir>"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json'))
ap.add_argument('--out', required=True)
a = ap.parse_args()
from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
import yaml                                                      # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import GroundPlane                # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view        # noqa: E402
from pxr import UsdLux                                           # noqa: E402
import drawer_asset as DRA                                       # noqa: E402
import house_assets as HA                                        # noqa: E402
import marker_proxy_asset as MK                                  # noqa: E402

RC = 3


def main():
    global RC
    reg = json.load(open(a.reg))
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.02)
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    dpose = reg['drawer']['pose']
    HA.build_drawer_yaw(world.stage, DRA, DRA.load(os.path.join(WS, reg['drawer']['asset'])), dpose)
    M = yaml.safe_load(open(os.path.join(WS, reg['marker']['asset'])))
    mobj, _ = HA.build_marker_yaw(world, MK, M, reg['marker']['center_world'], reg['marker']['axis_yaw_deg'])
    UsdLux.DomeLight.Define(world.stage, '/World/dome').CreateIntensityAttr(800.0)
    set_camera_view(eye=[dpose[0] + 1.4, dpose[1] - 0.8, 1.5], target=[dpose[0] + 0.2, dpose[1], 0.45])
    world.reset()
    t0 = float(world.current_time)
    rows, k = [], 0
    while float(world.current_time) - t0 < 3.0 - 1e-9:
        world.step(render=(k % 4 == 0))
        k += 1
        p, q = mobj.get_world_pose()
        rows.append([float(world.current_time) - t0] + [float(v) for v in p] + [float(v) for v in q] + [float(v) for v in mobj.get_linear_velocity()] + [float(v) for v in mobj.get_angular_velocity()])
    R = np.array(rows)
    t, P, Q, V, Wv = R[:, 0], R[:, 1:4], R[:, 4:8], R[:, 8:11], R[:, 11:14]
    dt = np.diff(t)
    vfd = np.linalg.norm(np.diff(P, axis=0), axis=1) / dt
    dq = np.abs(np.sum(Q[1:] * Q[:-1], axis=1)).clip(0, 1)
    wfd = 2 * np.arccos(dq) / dt
    vr, wr = np.linalg.norm(V, axis=1), np.linalg.norm(Wv, axis=1)
    m0 = np.array(reg['marker']['center_world'])
    z_expect = M['drawer_interior']['bottom_top_z'] + M['half_extent_z']
    i2 = int(np.searchsorted(t, 2.0 - 1e-9))

    def seg(a_, b_):
        m = (t[1:] >= a_) & (t[1:] < b_)
        return dict(v_read_max=round(float(vr[1:][m].max()) * 1e3, 3), v_read_median=round(float(np.median(vr[1:][m])) * 1e3, 3), v_fd_max=round(float(vfd[m].max()) * 1e3, 3), v_fd_median=round(float(np.median(vfd[m])) * 1e3, 3),
                    w_read_max=round(float(wr[1:][m].max()), 4), w_fd_max=round(float(wfd[m].max()), 4), path_len_mm=round(float((vfd[m] * dt[m]).sum()) * 1e3, 4), z_ptp_mm=round(float(np.ptp(P[1:][m][:, 2])) * 1e3, 4))
    at2 = dict(dxy_mm=[round(float(P[i2, 0] - m0[0]) * 1e3, 3), round(float(P[i2, 1] - m0[1]) * 1e3, 3)], dz_mm=round(float(P[i2, 2] - z_expect) * 1e3, 3), v_read_mm_s=round(float(vr[i2]) * 1e3, 3), v_fd_mm_s=round(float(vfd[i2 - 1]) * 1e3, 3))
    at2['orig_rule_ok'] = bool(max(abs(v) for v in at2['dxy_mm']) < 5.0 and abs(at2['dz_mm']) < 2.0 and at2['v_read_mm_s'] < 2.0)
    out = dict(reg=reg['name'], n_steps=len(rows), dt=round(float(np.median(dt)), 5), at_2s_original_rule=at2, segments={'0-0.5': seg(0, 0.5), '0.5-1': seg(0.5, 1.0), '1-2': seg(1.0, 2.0), '2-3': seg(2.0, 3.0)},
               net_disp_3s_mm=[round(float(v) * 1e3, 4) for v in (P[-1] - P[0])], cols=['t', 'px', 'py', 'pz', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
               note='診斷，不取代 drawer_check r1 的 B 項失敗；v_fd＝位置逐步差分、v_read＝速度讀回')
    np.savez_compressed(os.path.join(a.out, 'marker_rest_steps.npz'), rows=R)
    json.dump(out, open(os.path.join(a.out, 'm8_house_marker_rest.json'), 'w'), ensure_ascii=False, indent=1)
    print('[mrest]', json.dumps(out, ensure_ascii=False)[:1800], flush=True)
    RC = 0
    t1 = time.time()
    while sim_app.is_running() and time.time() - t1 < 2.0:
        world.step(render=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        print('[mrest] 例外', traceback.format_exc(), flush=True)
    print(f'[mrest] 離開碼 {RC}', flush=True)
    sim_app.close()
    os._exit(RC)
