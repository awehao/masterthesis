#!/usr/bin/env python3
"""住宅場景：偏航抽屜的資產接線核對（Codex 20261112 核可；無機器人）。場景只有地面、抽屜櫃（house_assets.build_drawer_yaw，登錄位姿）、代理物。
  A 讀回：滑動關節兩側框架（localPos／localRot）、軸、限位、驅動；換算世界滑動方向，對登錄 pull_dir。
  B 零位靜置 2 s：抽屜位置、姿態；代理物獨立剛體讀回與靜置（位移、速度）。
  C 有界外力：對抽屜質心施加沿登錄拉出方向的力 F（N，預設 3），到開度不再增加；量開度、橫向位移、z、轉角；到限位後應停在 upper。
  D 反向力推回：應回到 0 附近（下限）。
  E 橫向力（垂直拉出方向，同大小）：開度與橫向位移都應 ≈ 0（只允許沿滑動軸）。
不用瞬移代替滑動。判定不過或例外 ⇒ 非零離開。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_drawer_check.py --out <dir>"""
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
ap.add_argument('--force', type=float, default=3.0)
ap.add_argument('--hold-s', type=float, default=3.0)
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
import yaml                                                      # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import GroundPlane                # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleRigidPrim       # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view        # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics                  # noqa: E402

import drawer_asset as DRA                                       # noqa: E402
import house_assets as HA                                        # noqa: E402
import marker_proxy_asset as MK                                  # noqa: E402

RC = 3


def main():
    global RC
    reg = json.load(open(a.reg))
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.02)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    dpose = reg['drawer']['pose']
    spec = DRA.load(os.path.join(WS, reg['drawer']['asset']))
    drb = HA.build_drawer_yaw(stage, DRA, spec, dpose)
    M = yaml.safe_load(open(os.path.join(WS, reg['marker']['asset'])))
    mobj, mmat = HA.build_marker_yaw(world, MK, M, reg['marker']['center_world'], reg['marker']['axis_yaw_deg'])
    k = UsdLux.DistantLight.Define(stage, '/World/key')
    k.CreateIntensityAttr(1500.0)
    UsdGeom.Xformable(k).AddRotateXYZOp().Set(Gf.Vec3f(-30.0, 20.0, 0.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(500.0)
    set_camera_view(eye=[dpose[0] + 1.6, dpose[1] - 1.0, 1.6], target=[dpose[0] + 0.2, dpose[1], 0.45])
    world.reset()
    pull = np.array(reg['drawer']['pull_dir'] + [0.0], float)
    lat = np.array([-pull[1], pull[0], 0.0])
    # ---- A 讀回
    j = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath('/World/drawer_unit/drawer_slide'))
    r0, r1 = j.GetLocalRot0Attr().Get(), j.GetLocalRot1Attr().Get()

    def qR(q):
        w, (x, y, z) = q.GetReal(), q.GetImaginary()
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    ax = dict(X=0, Y=1, Z=2)[str(j.GetAxisAttr().Get())]
    axis_w0 = qR(r0)[:, ax]                                         # body0＝世界：關節 + 軸在世界的方向
    dprim = stage.GetPrimAtPath('/World/drawer_unit/drawer')
    Mw = UsdGeom.XformCache().GetLocalToWorldTransform(dprim)
    Rd = np.array([[Mw[i][jj] for i in range(3)] for jj in range(3)])  # USD 列向量 → 欄向量
    axis_w1 = Rd @ qR(r1)[:, ax]                                    # 抽屜側框架的同一軸在世界的方向
    A = dict(axis=str(j.GetAxisAttr().Get()), localPos0=[float(v) for v in j.GetLocalPos0Attr().Get()], localPos1=[float(v) for v in j.GetLocalPos1Attr().Get()],
             localRot0_wxyz=[float(r0.GetReal())] + [float(v) for v in r0.GetImaginary()], localRot1_wxyz=[float(r1.GetReal())] + [float(v) for v in r1.GetImaginary()],
             lower=float(j.GetLowerLimitAttr().Get()), upper=float(j.GetUpperLimitAttr().Get()), axis_world_from_body0=axis_w0.round(5).tolist(), axis_world_from_body1=axis_w1.round(5).tolist(),
             pull_dir_registered=pull.round(5).tolist(), frames_agree=bool(np.linalg.norm(axis_w0 - axis_w1) < 1e-3), axis_matches_pull=bool(np.linalg.norm(axis_w0 - pull) < 1e-3), drawer_readback=drb)
    drw = SingleRigidPrim('/World/drawer_unit/drawer', name='drawer_rb')
    drw.initialize()
    view = RigidPrim('/World/drawer_unit/drawer', name='drawer_view')
    view.initialize()
    p_ref = np.array([dpose[0], dpose[1], 0.0])

    def dstate():
        p, q = drw.get_world_pose()
        p = np.asarray(p, float)
        yaw = math.degrees(2 * math.atan2(float(q[3]), float(q[0])))
        tilt = math.degrees(2 * math.asin(min(1.0, math.hypot(float(q[1]), float(q[2])))))
        d = p - p_ref
        return dict(opening=float(d @ pull), lateral=float(d @ lat), z=float(d[2]), yaw_deg=yaw, tilt_deg=tilt, v=float(np.linalg.norm(drw.get_linear_velocity())))

    def mstate():
        p, q = mobj.get_world_pose()
        return dict(p=[float(v) for v in p], v=float(np.linalg.norm(mobj.get_linear_velocity())), w=float(np.linalg.norm(mobj.get_angular_velocity())))

    def run(T, F=None):
        t0 = float(world.current_time)
        tr, k_ = [], 0
        while float(world.current_time) - t0 < T - 1e-9:
            if F is not None:
                view.apply_forces(np.asarray([F], float), is_global=True)
            world.step(render=(k_ % 2 == 0))
            k_ += 1
            if k_ % 10 == 0:
                s = dstate()
                tr.append([round(float(world.current_time) - t0, 3), round(s['opening'], 5), round(s['lateral'], 5), round(s['z'], 5), round(s['yaw_deg'], 3), round(s['tilt_deg'], 3)])
        return tr
    # ---- B 靜置
    mb = MK.readback(stage, mobj, mmat)
    m0 = np.array(reg['marker']['center_world'])
    run(2.0)
    d_rest, m_rest = dstate(), mstate()
    z_expect = M['drawer_interior']['bottom_top_z'] + M['half_extent_z']
    B = dict(drawer=d_rest, marker=m_rest, marker_dxy_mm=[round((m_rest['p'][0] - m0[0]) * 1e3, 3), round((m_rest['p'][1] - m0[1]) * 1e3, 3)], marker_dz_mm=round((m_rest['p'][2] - z_expect) * 1e3, 3), marker_readback=mb)
    B['ok'] = bool(abs(d_rest['opening']) < 0.002 and abs(d_rest['lateral']) < 0.001 and abs(d_rest['yaw_deg'] - dpose[2]) < 0.2 and d_rest['tilt_deg'] < 0.2 and max(abs(v) for v in B['marker_dxy_mm']) < 5.0
                   and abs(B['marker_dz_mm']) < 2.0 and m_rest['v'] < 0.002 and mb['parent'] == '/World' and mb['rigid_body'] and not mb['kinematic'] and mb['collision_enabled'])
    # ---- C 拉出
    trC = run(4.0, a.force * pull)
    dC = dstate()
    mC = mstate()
    run(1.0)
    dC2 = dstate()
    C = dict(force_N=a.force, end=dC, after_release_1s=dC2, trace=trC[::4], marker_moved_along_pull_mm=round(float((np.array(mC['p']) - np.array(m_rest['p'])) @ pull) * 1e3, 2), marker_z=mC['p'][2])
    C['ok'] = bool(abs(dC['opening'] - A['upper']) < 0.005 and abs(dC['lateral']) < 0.002 and abs(dC['z']) < 0.002 and abs(dC['yaw_deg'] - dpose[2]) < 0.3 and dC['tilt_deg'] < 0.3)
    # ---- D 推回
    trD = run(4.0, -a.force * pull)
    dD = dstate()
    D = dict(end=dD, trace=trD[::4])
    D['ok'] = bool(abs(dD['opening'] - A['lower']) < 0.005 and abs(dD['lateral']) < 0.002 and dD['tilt_deg'] < 0.3)
    # ---- E 橫向力
    run(1.0)
    e0 = dstate()
    trE = run(2.0, a.force * lat)
    dE = dstate()
    E = dict(before=e0, end=dE, trace=trE[::4])
    E['ok'] = bool(abs(dE['opening'] - e0['opening']) < 0.003 and abs(dE['lateral']) < 0.002 and abs(dE['yaw_deg'] - dpose[2]) < 0.3)
    A['ok'] = bool(A['frames_agree'] and A['axis_matches_pull'] and abs(A['upper'] - reg['drawer']['travel']) < 1e-6 and drb['drive_max_force'] == 0.0)
    out = dict(reg=reg['name'], A_readback=A, B_rest=B, C_pull=C, D_push_back=D, E_lateral=E, verdict=dict(ok=bool(A['ok'] and B['ok'] and C['ok'] and D['ok'] and E['ok']), failed=[n for n, x in (('A', A), ('B', B), ('C', C), ('D', D), ('E', E)) if not x['ok']]),
               scope='無機器人；有界外力作用在抽屜質心；不含夾爪抓拉、不含代理物夾取')
    json.dump(out, open(os.path.join(a.out, 'm8_house_drawer_check.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    for n, x in (('A', A), ('B', B), ('C', C), ('D', D), ('E', E)):
        print(f'[dchk] {n}', json.dumps({kk: vv for kk, vv in x.items() if kk not in ('trace', 'marker_readback', 'drawer_readback')}, ensure_ascii=False, default=float)[:700], flush=True)
    print('[dchk] 判定', out['verdict'], flush=True)
    RC = 0 if out['verdict']['ok'] else 2
    t0 = time.time()
    while sim_app.is_running() and time.time() - t0 < a.hold_s:
        world.step(render=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        RC = 3
        print('[dchk] 例外', traceback.format_exc(), flush=True)
    print(f'[dchk] 離開碼 {RC}', flush=True)
    sim_app.close()
    os._exit(RC)
