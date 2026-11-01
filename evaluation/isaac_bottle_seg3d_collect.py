#!/usr/bin/env python3
"""三維定位採集（Codex 20261101_220915；不抓取、估計不進控制）：一個佈置一趟。動作與佈置見 bottle_seg3d_collect_design.py（凍結），本腳本只執行。
場景：第一桌桌面頂 --table-z（厚度、尺寸、桌腳內縮同資產；桌腳由地面到桌板底，不懸空）、瓶 --bottle-x/--bottle-y（瓶底在桌面）；
  第二桌（資產高 0.60）與紅色方塊障礙同 W1 場景、六個佈置一致；不放移動物。
動作：
  1 接近：底盤以紀錄趟（design src_run）0 → t_obs 的速度前饋＋位置回授（2.0/s）跟隨，手臂設定點照紀錄的 s1–s6；t_obs 後底盤速度 0。
  2 K2_READY → Q_LOOK（關節 min-jerk，design T_tr）→ 保持 --hold-s → 回 K2_READY（同 T_tr）→ 回收 q_tuck（時長依 Δq/v_arm）。
相機：V1（底盤）、W1（手腕）同實跑擷取類別（發布 /v1/*、/w1/*，量測位姿＝量測關節 FK×外參）；觀察保持段每個新影格都存檔（其餘照原規則）。
旁路：同網域的 bottle_seg3d_node（v1、w1，runner 啟動）輸出 /{ns}/bottle_seg3d，本程序只收下存檔（seg3d_{ns}.npz：收到時的模擬時刻＋原始欄位＋layout 標籤）。
安全：每 10 物理步以量測關節算手臂連桿點雲對第一桌（桌面、桌腳）與瓶身的距離；< 0.02 m ⇒ 中止（手臂停在目前設定點、底盤速度 0），記 abort。
  瓶真值位移 > 5 mm（被碰到）⇒ 同樣中止。桌高只用於生成場景、安全核對與評估，不傳給估計節點。
    RUN_ID=seg3d_col_z045_p1 ROS_DOMAIN_ID=95 COLLECT_ARGS="--table-z 0.45 --bottle-x 0.0 --bottle-y 0.42" RUNNER=evaluation/run_seg3d_collect.sh bash evaluation/run_guarded_v3.sh
"""
import argparse
import json
import math
import os
import sys
import time

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--asset', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--design', default=os.path.join(WS, 'evaluation/results/vision/seg3d_collect_design.json'))
ap.add_argument('--table-z', type=float, required=True)
ap.add_argument('--bottle-x', type=float, required=True)
ap.add_argument('--bottle-y', type=float, required=True)
ap.add_argument('--hold-s', type=float, default=3.0)
ap.add_argument('--hz', type=float, default=5.0)
ap.add_argument('--obst-x', type=float, default=-2.0)
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--settle-s', type=float, default=1.0)
ap.add_argument('--headless', default='false')
a = ap.parse_args()

from isaacsim import SimulationApp                                   # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                                   # noqa: E402
import yaml                                                          # noqa: E402
from isaacsim.core.api import World                                  # noqa: E402
from isaacsim.core.api.objects import DynamicCylinder, FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import SingleArticulation                   # noqa: E402
from isaacsim.core.utils.types import ArticulationAction             # noqa: E402

HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from isaac_common import bind_friction, bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402
from finger_collision import apply_split                             # noqa: E402
import bottle_ros_chain as RC                                        # noqa: E402
PCm, V1C, W1C = RC.import_with_real_ros('bottle_place_ctrl', 'bottle_v1_cam', 'bottle_w1_cam')
import bottle_ref as BRF                                             # noqa: E402
import bottle_s0_ctrl as C                                           # noqa: E402
import bottle_s0b_ctrl as SB                                         # noqa: E402
import bottle_s1_geom as G                                           # noqa: E402

ROBOT = '/World/omni_bot'
ARM = [f'joint{i}' for i in range(1, 7)]
PHASES = ('approach', 'to_look', 'hold', 'back', 'tuck', 'end')
CLEAR_ABORT, BOTTLE_MOVE_ABORT = 0.02, 0.005


def yaw_of(q):
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class NoChain:                                                       # Bridge 需要的回呼；採集不用障礙／指令鏈
    def cb_obs(self, d):
        pass

    def cb_cmd(self, u):
        pass

    def cb_diag(self, d):
        pass


def main():
    S = yaml.safe_load(open(a.asset))
    T, B, PH = S['table'], S['bottle'], S['physics']
    D = json.load(open(a.design))
    if D.get('q_look') is None:
        print('[col] 設計沒有 Q_LOOK ⇒ 拒跑', flush=True)
        return 80
    dt = float(PH['dt'])
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics
    scn = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)]
    sapi = PhysxSchema.PhysxSceneAPI(scn[0]) if scn[0].HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(scn[0])
    sapi.CreateEnableExternalForcesEveryIterationAttr().Set(bool(PH.get('ext_forces_every_iter')))
    TZ = float(a.table_z)
    T1 = dict(T, top_z=TZ)

    def make_table(root, Tt, off, color, leg_color):
        FixedCuboid(prim_path=f'{root}/top', name=f'{root[7:]}_top', position=np.array([0.0, Tt['near_y'] + Tt['size_y'] / 2, Tt['top_z'] - Tt['thick'] / 2]) + off,
                    scale=np.array([Tt['size_x'], Tt['size_y'], Tt['thick']]), color=color)
        hz = Tt['top_z'] - Tt['thick']
        paths = [f'{root}/top']
        for sx in (-1, 1):
            lx = sx * (Tt['size_x'] / 2 - Tt['leg_inset_x'] - Tt['leg'] / 2)
            for nm, ly in (('near', Tt['near_y'] + Tt['leg_inset_near'] + Tt['leg'] / 2), ('far', Tt['near_y'] + Tt['size_y'] - Tt['leg_inset_far'] - Tt['leg'] / 2)):
                pth = f'{root}/leg_{nm}_{"p" if sx > 0 else "m"}'
                FixedCuboid(prim_path=pth, name=f'{root[7:]}_leg_{nm}_{sx}', position=np.array([lx, ly, hz / 2]) + off, scale=np.array([Tt['leg'], Tt['leg'], hz]), color=leg_color)
                paths.append(pth)
        return paths
    t1_paths = make_table('/World/table', T1, np.zeros(3), np.array([0.55, 0.42, 0.30]), np.array([0.4, 0.3, 0.2]))
    T2D = np.array([PCm.PLACE['target'][0] - float(B['center_xy'][0]), PCm.PLACE['lane_y'], 0.0])
    t2_paths = make_table('/World/table2', T, T2D, np.array([0.45, 0.50, 0.62]), np.array([0.35, 0.38, 0.45]))
    OB_C, OB_S = np.array([a.obst_x, 0.0, 0.25]), np.array([0.40, 0.40, 0.50])
    FixedCuboid(prim_path='/World/obstacle', name='obstacle', position=OB_C, scale=OB_S, color=np.array([0.7, 0.25, 0.2]))
    BX, BY = float(a.bottle_x), float(a.bottle_y)
    bottle = DynamicCylinder(prim_path='/World/bottle', name='bottle', position=np.array([BX, BY, TZ + B['height'] / 2]), radius=B['radius'], height=B['height'], mass=B['mass'],
                             color=np.array([0.2, 0.6, 0.9]))
    bind_friction(stage, [stage.GetPrimAtPath(p) for p in t1_paths + t2_paths], T['material']['static_friction'], T['material']['dynamic_friction'], T['material']['restitution'],
                  mat_path='/World/PhysicsMaterials/table')
    bind_friction(stage, [stage.GetPrimAtPath('/World/bottle')], B['material']['static_friction'], B['material']['dynamic_friction'], B['material']['restitution'],
                  mat_path='/World/PhysicsMaterials/bottle')
    # 紀錄路徑
    Z = np.load(os.path.join(HERE, 'runs', D['src_run'], 's1_steps.npz'))
    rc = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    m_ = rc['t'] <= float(D['t_obs']) + 1e-9
    R_t, R_b = rc['t'][m_], np.column_stack([rc['base_x'][m_], rc['base_y'][m_], rc['base_yaw'][m_]])
    R_s = np.column_stack([rc[f's{j}'][m_] for j in range(1, 7)])
    R_v = np.gradient(R_b, R_t, axis=0)
    base0 = R_b[0]
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(base0[0]), float(base0[1]), 0.0))
    xf.AddRotateZOp().Set(float(math.degrees(base0[2])))
    apply_split(stage, ROBOT)
    if bind_frictionless(stage, collision_prims(stage, ROBOT, lambda p: 'support_' in p) + collision_prims(stage, '/World/ground')) == 0:
        print('[col] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
        return 19
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(600.0)
    bodies, arts = physics_parts(stage, ROBOT)
    world.reset()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    q_tuck = np.array(R_s[0], float)
    q0 = robot.get_joint_positions()
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i, j in zip(idx, range(6)):
        q0[i] = float(q_tuck[j])
        kp[i], kd[i] = a.kp, a.kd
    q0[f1], q0[f2] = C.FINGER_OPEN, C.FINGER_OPEN
    kp[f1], kd[f1] = a.finger_kp, a.finger_kd
    act = np.array(sorted(idx + [f1]), np.int32)
    robot.set_joint_positions(q0)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    tgt = np.array(q0, float)

    def apply(q_arm):
        tgt[idx] = q_arm
        tgt[f1] = C.FINGER_OPEN
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.asarray(tgt, np.float32)[act], joint_indices=act))

    def base_world(v):
        robot.set_linear_velocity(np.array([v[0], v[1], 0.0], np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, v[2]], np.float32))

    def q9():
        p, qt = robot.get_world_pose()
        jp = robot.get_joint_positions()
        return np.r_[float(p[0]), float(p[1]), yaw_of(qt), [float(jp[i]) for i in idx]], p, qt

    def bottle_T():
        pb, qb = bottle.get_world_pose()
        Rb = quat_R(qb)
        Tb = np.eye(4)
        Tb[:3, :3] = Rb
        Tb[:3, 3] = np.asarray(pb, float) - Rb[:, 2] * float(B['height']) / 2
        return Tb
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([-1.6, -2.2, 1.8]), target=np.array([-0.2, 0.3, TZ]))
    except Exception as e:
        print(f'[col] 視角設定失敗：{e!r}', flush=True)
    node = RC.make_bridge(NoChain())
    from isaacsim.sensors.camera import Camera
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Float64MultiArray
    vis_boxes = [(c_, s_) for _, c_, s_ in G.table_boxes(T1)] + [(OB_C, OB_S)] + [(c_ + T2D, s_) for _, c_, s_ in G.table_boxes()]
    msgs = dict(Image=Image, CameraInfo=CameraInfo, PoseStamped=PoseStamped, Float64MultiArray=Float64MultiArray)
    b_base = [b for b in bodies if str(b).rstrip('/').split('/')[-1] == 'base_link']
    b_eef = [b for b in bodies if str(b).rstrip('/').split('/')[-1] == 'link_eef']
    v1 = V1C.V1Capture(Camera, str(b_base[0]), a.urdf, node, a.hz, dt, vis_boxes, msgs)
    w1 = W1C.CamCapture('w1', Camera, str(b_eef[0]), a.urdf, node, a.hz, dt, vis_boxes, msgs)
    seg = {ns: [] for ns in ('v1', 'w1')}
    label = {}

    def mk(ns):
        def cb(m):
            if m.layout.dim:
                label[ns] = m.layout.dim[0].label
            seg[ns].append([float(node_t[0])] + [float(v) for v in m.data])
        return cb
    node_t = [0.0]
    for ns in ('v1', 'w1'):
        node.create_subscription(Float64MultiArray, f'/{ns}/bottle_seg3d', mk(ns), 50)
    # 靜置
    for _ in range(int(round(a.settle_s / dt))):
        apply(q_tuck)
        base_world(np.zeros(3))
        world.step(render=True)
    Tb0 = bottle_T()
    # 時程
    q_ready, q_look = np.array(D['q_ready'], float), np.array(D['q_look'], float)
    T_tr = float(D['transition']['T_tr'])
    T_tk = math.ceil(1.875 * float(np.abs(q_ready - q_tuck).max()) / SB.W_ARM['v_arm'] / 0.05) * 0.05
    t_obs = float(D['t_obs'])
    sched = [('to_look', t_obs, t_obs + T_tr), ('hold', t_obs + T_tr, t_obs + T_tr + a.hold_s), ('back', t_obs + T_tr + a.hold_s, t_obs + 2 * T_tr + a.hold_s),
             ('tuck', t_obs + 2 * T_tr + a.hold_s, t_obs + 2 * T_tr + a.hold_s + T_tk)]
    t_end = sched[-1][2] + 1.0
    abort, min_clear, n_step = None, 9.9, 0
    t0w = float(world.current_time)
    log = []
    w0 = time.perf_counter()
    while True:
        t = float(world.current_time) - t0w
        q, p, qt = q9()
        ph = 'approach' if t < t_obs else next((nm for nm, s0, s1 in sched if s0 <= t < s1), 'end')
        if abort is None:
            if ph == 'approach':
                k = min(int(np.searchsorted(R_t, t)), len(R_t) - 1)
                pr, vr = R_b[k], R_v[k]
                e = pr - q[:3]
                e[2] = (e[2] + math.pi) % (2 * math.pi) - math.pi
                base_world(vr + 2.0 * e)
                apply(R_s[k])
            else:
                base_world(np.zeros(3))
                if ph == 'to_look':
                    s_ = BRF.min_jerk((t - sched[0][1]) / T_tr)
                    apply(q_ready + (q_look - q_ready) * s_)
                elif ph == 'hold':
                    apply(q_look)
                elif ph == 'back':
                    s_ = BRF.min_jerk((t - sched[2][1]) / T_tr)
                    apply(q_look + (q_ready - q_look) * s_)
                elif ph == 'tuck':
                    s_ = BRF.min_jerk(min((t - sched[3][1]) / T_tk, 1.0))
                    apply(q_ready + (q_tuck - q_ready) * s_)
                else:
                    apply(q_tuck)
        else:
            base_world(np.zeros(3))
        Tb = bottle_T()
        node.publish_state(float(world.current_time), p, qt, q[3:])
        node_t[0] = float(world.current_time)
        for cap in (v1, w1):
            cap.record_step(float(world.current_time), float(world.current_time), t, q, Tb, PHASES.index(ph))
            if ph == 'hold':
                cap.t_sparse = -1e9                                   # 觀察保持段：每個新影格都存檔
            cap.poll()
        node.drain()
        if n_step % 10 == 0:
            P = np.vstack(list(G.link_world(q, C.FINGER_OPEN).values()))
            d = min(float(G.box_dist(P, c_, s_).min()) for _, c_, s_ in G.table_boxes(T1))
            d = min(d, float(G.cyl_dist((np.linalg.inv(Tb0) @ np.c_[P, np.ones(len(P))].T).T[:, :3], 0.0, 0.0, 0.0, G.BOTTLE['h'], G.BOTTLE['r']).min()))
            min_clear = min(min_clear, d)
            mv = float(np.linalg.norm(Tb[:3, 3] - Tb0[:3, 3]))
            log.append((t, PHASES.index(ph), d, mv, *q))
            if abort is None and (d < CLEAR_ABORT or mv > BOTTLE_MOVE_ABORT):
                abort = dict(t=t, phase=ph, clear=d, bottle_move=mv)
                print(f'[col] 中止 {abort}', flush=True)
                apply(q[3:])
        n_step += 1
        world.step(render=True)
        if t >= t_end or (abort is not None and t >= abort['t'] + 2.0):
            break
    rep = dict(layout=dict(table_z=TZ, bottle=[BX, BY]), design=a.design, design_q_look=D['q_look'], src_run=D['src_run'], t_obs=t_obs, T_tr=T_tr, T_tuck=T_tk, hold_s=a.hold_s,
               sched=sched, abort=abort, min_clear_m=min_clear, bottle_move_m=float(np.linalg.norm(bottle_T()[:3, 3] - Tb0[:3, 3])), wall_s=time.perf_counter() - w0, n_step=n_step,
               seg3d=dict(n={ns: len(v) for ns, v in seg.items()}, label=label))
    rep['v1'] = v1.save(a.out)
    rep['w1_cap'] = w1.save(a.out)
    for ns, rows in seg.items():
        np.savez_compressed(os.path.join(a.out, f'seg3d_{ns}.npz'), rows=np.array(rows, float) if rows else np.zeros((0, 1)), label=np.array(label.get(ns, '')))
    np.savez_compressed(os.path.join(a.out, 'collect_log.npz'), rows=np.array(log, float), cols=np.array(['t', 'phase', 'clear', 'bottle_move', 'x', 'y', 'yaw'] + [f'q{j}' for j in range(1, 7)]))
    json.dump(rep, open(os.path.join(a.out, 'collect_report.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"[col] 結果 中止 {abort} 最小淨空 {min_clear:.3f} m 瓶位移 {rep['bottle_move_m'] * 1e3:.1f} mm seg3d {rep['seg3d']['n']} 影格 v1 {rep['v1'].get('n_frames') if isinstance(rep['v1'], dict) else rep['v1']} w1 {rep['w1_cap'].get('n_frames') if isinstance(rep['w1_cap'], dict) else rep['w1_cap']}", flush=True)
    with open(os.path.join(a.out, 'exit_rc.txt'), 'w') as fh:
        fh.write('0' if abort is None else '3')
    return 0 if abort is None else 3


if __name__ == '__main__':
    rc_ = 1
    try:
        rc_ = main()
    finally:
        sim_app.close()
    sys.exit(rc_)
