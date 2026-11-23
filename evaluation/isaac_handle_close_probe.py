#!/usr/bin/env python3
"""抽屜把手閉爪受控測試（Isaac 86／94 受力過大、95 兩指停在 6.6 mm 的原因調查）。不是任務趟、不跑控制器。
場景＝地面＋抽屜櫃（house_assets.build_drawer_yaw，實跑同登錄位姿、同動態滑軌）＋機器人（固定底盤，位姿取自指定實跑閉爪起點）。
手臂：初始關節＝該時刻實測值，驅動目標＝該時刻設定點（同實跑增益 kp 600／kd 60），讓手臂自然下垂到與實跑相同狀態；
手指：同實跑增益（kp 1e4／kd 1e3），自開到 0 線性斜降 T_CLOSE（0.6 s），再保持 1.0 s。
垂直偏移 dz：整台機器人上下平移 dz（m），等同夾爪相對橫桿的高度差。逐步記兩指開度、淨接觸力、對抽屜各零件的接觸力。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_handle_close_probe.py --run 92 --t-close 159.31 --dz -0.004 --out <json>"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
ap = argparse.ArgumentParser()
ap.add_argument('--run', type=int, required=True)
ap.add_argument('--t-close', type=float, required=True)
ap.add_argument('--dz', type=float, default=0.0)
ap.add_argument('--t-close-s', type=float, default=0.6)
ap.add_argument('--dapp', type=float, default=0.0, help='沿底盤車頭方向平移（m，正＝往前＝夾爪插得更深）')
ap.add_argument('--out', required=True)
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261121_m8_house_r2w3_scene.json'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
a = ap.parse_args()
os.environ['M8_SCENE_REG'] = os.path.basename(a.reg)

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import GroundPlane                # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation   # noqa: E402
from isaacsim.core.utils.types import ArticulationAction         # noqa: E402
from pxr import Gf, UsdGeom                                      # noqa: E402

from isaac_common import import_urdf, physics_parts             # noqa: E402

DT = 0.01
ARM = [f'joint{i}' for i in range(1, 7)]
FINGER_OPEN = 0.0089


def main():
    reg = json.load(open(a.reg))
    RD = os.path.join(HERE, 'runs', f'm8house_d2_unified_{a.run}')
    Dz = np.load(os.path.join(RD, 'log.npz')); Dc = [str(x) for x in Dz['cols']]; DL = Dz['L']
    k = int(np.searchsorted(DL[:, Dc.index('t')], a.t_close))
    b = [float(DL[k, Dc.index(f'q{i}')]) for i in range(3)]
    q_arm = [float(DL[k, Dc.index(f'q{i}')]) for i in range(3, 9)]
    s_arm = [float(DL[k, Dc.index(f's{i}')]) for i in range(6)]
    fq0 = [float(DL[k, Dc.index('finger1')]), float(DL[k, Dc.index('finger2')])]
    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    import drawer_asset as DRA_
    import house_assets as HA_
    dspec_ = DRA_.load(os.path.join(WS, reg['drawer']['asset']))
    pose_true = json.load(open(os.path.join(RD, 'result.json')))['drawer']['pose_true']   # 實跑真實擺放（登錄＋D2_DRAWER_TRUE 偏移）
    HA_.build_drawer_yaw(stage, DRA_, dspec_, list(pose_true))
    parts = ['handle_bar', 'post_l', 'post_r', 'front_panel', 'bottom', 'side_l', 'side_r', 'back']
    filt = [f'/World/drawer_unit/drawer/{p}' for p in parts] + ['/World/drawer_unit/cabinet']
    filt = [p for p in filt if stage.GetPrimAtPath(p).IsValid()]
    ROBOT = '/World/omni_bot'
    import_urdf(a.urdf, ROBOT, fix_base=True)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(b[0] + float(a.dapp) * math.cos(b[2]), b[1] + float(a.dapp) * math.sin(b[2]), float(a.dz)))
    xf.AddRotateZOp().Set(float(math.degrees(b[2])))
    bodies, arts = physics_parts(stage, ROBOT)
    name_of = lambda x: str(x).rstrip('/').split('/')[-1]            # noqa: E731
    fb = {name_of(x): x for x in bodies if name_of(x) in ('uflite_finger1', 'uflite_finger2', 'uflite_gripper_link')}
    vf = {n: RigidPrim(prim_paths_expr=str(p), name=f'v_{n}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=filt) for n, p in fb.items()}
    vdr = RigidPrim(prim_paths_expr='/World/drawer_unit/drawer', name='v_dr')
    world.reset()
    for v in list(vf.values()) + [vdr]:
        v.initialize()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i in idx:
        kp[i], kd[i] = 600.0, 60.0
    kp[f1], kd[f1], kp[f2], kd[f2] = 1.0e4, 1.0e3, 1.0e4, 1.0e3
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    q = np.array(robot.get_joint_positions(), np.float32)
    for i, v in zip(idx, q_arm):
        q[i] = v
    q[f1], q[f2] = fq0[0], fq0[1]
    robot.set_joint_positions(q)
    robot.set_joint_velocities(np.zeros_like(q))
    act = np.array(idx + [f1, f2])

    def cmd(fc):
        tgt = np.array(s_arm + [fc, fc], np.float32)
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=tgt, joint_indices=act))
    rows = []
    n_settle, n_close, n_hold = int(1.0 / DT), int(a.t_close_s / DT), int(1.0 / DT)
    dp0 = None
    for n in range(n_settle + n_close + n_hold):
        tt = n * DT
        fc = FINGER_OPEN if n < n_settle else max(0.0, FINGER_OPEN * (1.0 - (n - n_settle) / n_close))
        cmd(fc)
        world.step(render=(n % 5 == 0))
        jp = np.asarray(robot.get_joint_positions(), float)
        rec = dict(t=round(tt, 3), fcmd=round(fc, 5), f1=round(float(jp[f1]), 5), f2=round(float(jp[f2]), 5))
        for nm, v in vf.items():
            net = float(np.linalg.norm(np.asarray(v.get_net_contact_forces(dt=DT), float).reshape(-1)[:3]))
            M = np.asarray(v.get_contact_force_matrix(dt=DT), float).reshape(-1, len(filt), 3)
            per = {name_of(p): round(float(np.linalg.norm(M[0][j])), 2) for j, p in enumerate(filt) if float(np.linalg.norm(M[0][j])) > 0.05}
            rec[nm[7:] if nm.startswith('uflite_') else nm] = dict(net=round(net, 2), per=per)
        pdr = np.asarray(vdr.get_world_poses()[0], float)[0]
        dp0 = pdr if dp0 is None else dp0
        rec['drawer_dpos_mm'] = [round(float(x) * 1e3, 2) for x in (pdr - dp0)]
        rows.append(rec)
        if n == n_settle - 1:                                             # 診斷：靜置後夾爪與橫桿實際世界位置
            cache = UsdGeom.XformCache()
            def wpos(pth):
                M = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(pth)); return [round(float(M[3][i]), 4) for i in range(3)]
            gp = [p_ for p_ in [str(x) for x in bodies] if p_.endswith('uflite_gripper_link')][0]
            tcp_ = [str(x) for x in stage.Traverse() if str(x.GetPath()).endswith('/link_tcp')]
            pq_ = vdr.get_world_poses(); print('[probe] drawer body physics pos', [round(float(x), 4) for x in np.asarray(pq_[0])[0]], 'quat', [round(float(x), 4) for x in np.asarray(pq_[1])[0]], 'true pose', pose_true, flush=True)
            print('[probe] settle gripper', wpos(gp), 'tcp', wpos(tcp_[0]) if tcp_ else None, 'bar', wpos('/World/drawer_unit/drawer/handle_bar'), 'arm q', [round(float(jp[i]), 3) for i in idx], 'q_log', [round(x, 3) for x in q_arm], flush=True)
    cl = [r for r in rows if r['t'] >= 1.0]
    summ = dict(run=a.run, t_close=a.t_close, dz_mm=round(a.dz * 1e3, 2), dapp_mm=round(a.dapp * 1e3, 2), final_open=[cl[-1]['f1'], cl[-1]['f2']],
                max_net=max(max(r['finger1']['net'], r['finger2']['net']) for r in cl),
                first_contact=next(((r['t'], r['f1'], r['finger1']['per'], r['finger2']['per']) for r in cl if r['finger1']['per'] or r['finger2']['per']), None),
                drawer_dpos_end_mm=cl[-1]['drawer_dpos_mm'])
    json.dump(dict(summary=summ, rows=rows, filt=filt, base=b, q_arm=q_arm, s_arm=s_arm), open(a.out, 'w'), indent=0)
    print('[probe] ' + json.dumps(summ, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    rc = 99
    try:
        rc = main()
    except Exception:
        import traceback
        print('[probe] 例外', traceback.format_exc(), flush=True)
    sim_app.close()
    sys.exit(rc)
