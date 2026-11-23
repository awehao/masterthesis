#!/usr/bin/env python3
"""第八次場景：代理物物理接線＋靜置核對（Codex 20261111_180841 第 2 項）。無機器人、無任何命令。
場景只有地面、抽屜櫃（drawer_asset，同登錄位姿）、代理物（marker_proxy_asset，獨立動態剛體）。步進 --sim-s 秒後讀回：
質量／碰撞／材料、代理物位移與末速度、抽屜開度。再做一個「沒有黏住」核對：把抽屜剛體瞬移拉出 0.10 m（顯示用瞬移，不是物理拉開）後再步進，
代理物若是獨立物件就不會跟著瞬移（留在原世界位置附近、落在仍在其下方的底板上）。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_marker_settle.py --out <dir>
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261111_m8_full_scene.json'))
ap.add_argument('--out', required=True)
ap.add_argument('--sim-s', type=float, default=3.0)
ap.add_argument('--hold-s', type=float, default=4.0)
ap.add_argument('--headless', default='false')
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true', 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import GroundPlane                # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view        # noqa: E402
from pxr import Gf, UsdGeom, UsdLux                              # noqa: E402

import drawer_asset as DRA                                       # noqa: E402
import marker_proxy_asset as MK                                  # noqa: E402


RC = 3


def main():
    reg = json.load(open(a.reg))
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.02)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    dpose = tuple(reg['drawer']['pose'])
    drb = DRA.build_usd(stage, DRA.load(os.path.join(WS, reg['drawer']['asset'])), dpose, root='/World/drawer_unit')
    spec = MK.load(os.path.join(WS, reg['marker']['asset']))
    obj, mat = MK.build(world, spec, dpose)
    k = UsdLux.DistantLight.Define(stage, '/World/key')
    k.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(k).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    set_camera_view(eye=[dpose[0] + 0.9, dpose[1] - 1.4, 1.5], target=[dpose[0], dpose[1] - 0.1, 0.45])
    world.reset()
    rb = MK.readback(stage, obj, mat)
    dprim = UsdGeom.Xformable(stage.GetPrimAtPath('/World/drawer_unit/drawer'))

    from isaacsim.core.prims import SingleRigidPrim
    drw = SingleRigidPrim('/World/drawer_unit/drawer', name='drawer_rb')
    drw.initialize()

    def drawer_y():
        return float(drw.get_world_pose()[0][1])                    # 物理讀回（r1 用 USD 變換，瞬移後讀到的是自己寫的值）

    def state():
        p, q = obj.get_world_pose()
        return dict(p=[float(v) for v in p], q_wxyz=[float(v) for v in q], v=[float(v) for v in obj.get_linear_velocity()], w=[float(v) for v in obj.get_angular_velocity()],
                    drawer_opening=round(dpose[1] - drawer_y(), 5), drawer_p=[round(float(v), 6) for v in drw.get_world_pose()[0]],
                    drawer_q_wxyz=[round(float(v), 7) for v in drw.get_world_pose()[1]])
    p0 = MK.world_center(spec, dpose)
    s0 = state()
    t_start = float(world.current_time)                               # 以模擬時鐘記錄與終止（r1／r2 用迴圈次數×0.01，時間軸不可信）
    trace, k = [], 0
    while float(world.current_time) - t_start < a.sim_s - 1e-9 and k < 100000:
        world.step(render=(k % 2 == 0))
        k += 1
        if k % 10 == 0:
            st = state()
            trace.append([round(float(world.current_time) - t_start, 4)] + [round(v, 5) for v in st['p']] + [round(float(np.linalg.norm(st['v'])), 5), round(float(np.linalg.norm(st['w'])), 5)])
    s1 = state()
    t_sim = float(world.current_time) - t_start
    z_expect = spec['drawer_interior']['bottom_top_z'] + spec['half_extent_z']
    drift = np.array(s1['p']) - p0
    last = [r for r in trace if r[0] >= t_sim - 1.0]
    fd = (np.array(last[-1][1:4]) - np.array(last[0][1:4])) / max(last[-1][0] - last[0][0], 1e-9)       # 末 1 s 位移／時間，與讀回速度互相核對
    settle = dict(sim_time_s=round(t_sim, 4), steps=k, dt_per_step=round(t_sim / max(k, 1), 6), z_final=round(s1['p'][2], 5), z_expect=round(z_expect, 5),
                  dz_mm=round((s1['p'][2] - z_expect) * 1000, 3), dxy_mm=[round(drift[0] * 1000, 3), round(drift[1] * 1000, 3)],
                  speed_mps=round(float(np.linalg.norm(s1['v'])), 6), speed_from_position_last1s_mps=round(float(np.linalg.norm(fd)), 6), ang_speed=round(float(np.linalg.norm(s1['w'])), 6),
                  drawer_opening_m=s1['drawer_opening'], drawer_p=s1['drawer_p'], drawer_q_wxyz=s1['drawer_q_wxyz'],
                  criteria='|dz|<2 mm、|dx|,|dy|<5 mm、讀回速度與末 1 s 位移速度皆 <2 mm/s、抽屜開度 <2 mm（門檻同 r1／r2，另加位移速度）')
    settle['ok'] = bool(abs(settle['dz_mm']) < 2.0 and max(abs(drift[0]), abs(drift[1])) < 0.005 and settle['speed_mps'] < 0.002 and settle['speed_from_position_last1s_mps'] < 0.002
                        and abs(s1['drawer_opening']) < 0.002)
    # 沒有黏住：抽屜瞬移拉出 0.10 m，代理物不應跟著瞬移
    drw.set_world_pose(position=np.array([dpose[0], dpose[1] - 0.10, 0.0]))
    t1 = float(world.current_time)
    while float(world.current_time) - t1 < 1.0 - 1e-9:
        world.step(render=True)
    s2 = state()
    moved = float(s2['p'][1] - s1['p'][1])
    free = dict(drawer_teleport_m=0.10, drawer_opening_after=s2['drawer_opening'], marker_dy_m=round(moved, 5), marker_z=round(s2['p'][2], 5),
                note='代理物若掛在抽屜底下會跟著移 −0.10 m；獨立物件只受接觸影響（此核對是瞬移，不是物理拉開，位移量不當成拉抽屜時的滑移結論）')
    free['ok'] = bool(abs(moved) < 0.05 and abs(s2['drawer_opening'] - 0.10) < 0.02)
    sz = rb['size_xyz']
    rb_ok = bool(rb['parent'] == '/World' and rb['rigid_body'] and rb['rigid_enabled'] and not rb['kinematic'] and rb['collision'] and rb['collision_enabled']
                 and rb['collision_approx'] == 'convexHull' and not rb['angular_damping_authored'] and abs(rb['mass_sim'] - spec['mass_kg']) < 1e-6
                 and rb['n_points'] == 12 and max(abs(v - 1.0) for v in rb['scale']) < 1e-6
                 and abs(sz[0] - spec['length']) < 1e-5 and abs(sz[1] - spec['across_corners']) < 1e-5 and abs(sz[2] - spec['across_flats']) < 1e-5 and len(rb['material_bound']) == 1
                 and abs(rb['static_friction'] - spec['material']['static_friction']) < 1e-6 and abs(rb['dynamic_friction'] - spec['material']['dynamic_friction']) < 1e-6)
    out = dict(reg=reg['name'], spec={kk: spec[kk] for kk in ('shape', 'across_flats', 'across_corners', 'length', 'mass_kg', 'material')}, center_drawer_frame=spec['placement']['center'],
               drawer_readback=drb, marker_readback=rb, readback_ok=rb_ok, p_init_world=[float(v) for v in p0], state_after_reset=s0, settle=settle, not_glued=free,
               trace_cols=['t_sim', 'x', 'y', 'z', 'speed', 'ang_speed'], trace=trace, verdict=dict(ok=bool(rb_ok and settle['ok'] and free['ok'])),
               scope='無機器人、無命令的靜置核對；不含夾取、不含物理拉抽屜時代理物是否滑動')
    global RC
    RC = 0 if out['verdict']['ok'] else 2
    json.dump(out, open(os.path.join(a.out, 'm8_marker_settle.json'), 'w'), ensure_ascii=False, indent=1)
    print('[settle] 讀回', json.dumps(rb, ensure_ascii=False), flush=True)
    print('[settle] 靜置', json.dumps(settle, ensure_ascii=False), flush=True)
    print('[settle] 未黏住', json.dumps(free, ensure_ascii=False), flush=True)
    print('[settle] 判定', out['verdict'], flush=True)
    t0 = time.time()
    while sim_app.is_running() and time.time() - t0 < a.hold_s:
        world.step(render=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        RC = 3
        print('[settle] 例外', traceback.format_exc(), flush=True)
    print(f'[settle] 離開碼 {RC}', flush=True)
    sim_app.close()
    os._exit(RC)                                                      # verdict=false ⇒ 2、例外 ⇒ 3（不能只靠程序正常結束判通過）
