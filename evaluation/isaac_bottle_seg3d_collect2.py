#!/usr/bin/env python3
"""三維定位採集 v2：有界自適應觀察（Codex 20261101_230248 (b)；isaac_bottle_seg3d_collect.py 的版本化複本；不抓取、估計不進控制）。
接近與 V0（Q_LOOK）同 v1；到 V0 後依手腕旁路的即時輸出（schema 3）用 bottle_seg3d_view_policy 決定：接受、移到視點庫（seg3d_view_library.json）的 V_LOW／V_HIGH，
或拒絕；最多兩次調整，已用過的視點不回去；決策只讀有效旗標與影像截斷／外緣分類，不讀桌高或佈置編號。視點間過渡只用視點庫已核可的關節直線路徑（含停止路徑）；
結束時經 V0 回 K2_READY 再回收。每個觀察段等新鮮輸出（擷取戳 ≥ 到達＋0.4 s、收到時年齡 ≤ 0.5 s）≥ 5 列，逾時 --obs-timeout 視同沒有有效實例。
以下為 v1 說明。
三維定位採集（Codex 20261101_220915；不抓取、估計不進控制）：一個佈置一趟。動作與佈置見 bottle_seg3d_collect_design.py（凍結），本腳本只執行。
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
ap.add_argument('--library', default=os.path.join(WS, 'evaluation/results/vision/seg3d_view_library.json'))
ap.add_argument('--obs-timeout', type=float, default=4.0)
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
import bottle_seg3d_adapt_sm as ASM                                  # noqa: E402

ROBOT = '/World/omni_bot'
ARM = [f'joint{i}' for i in range(1, 7)]
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
    import rclpy
    if not rclpy.ok():
        rclpy.init()
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
    # 時程（有界自適應觀察；狀態機與決策紀錄在 bottle_seg3d_adapt_sm，腳本只負責接近段與下指令）
    LIB = json.load(open(a.library))
    q_ready = np.array(D['q_ready'], float)
    T_tr = float(D['transition']['T_tr'])
    T_tk = math.ceil(1.875 * float(np.abs(q_ready - q_tuck).max()) / SB.W_ARM['v_arm'] / 0.05) * 0.05
    t_obs = float(D['t_obs'])

    def T_between(x, y):
        tr = LIB['transitions'].get(f'{x}-{y}') or LIB['transitions'].get(f'{y}-{x}')
        if tr is None or not tr['ok']:
            raise RuntimeError(f'視點庫沒有核可的過渡 {x}→{y}')
        return float(tr['T'])
    sm = ASM.AdaptiveSM(q_ready, q_tuck, {k: v['q'] for k, v in LIB['views'].items()}, T_tr, T_tk, T_between, t_obs, a.obs_timeout, os.path.join(a.out, 'decisions.jsonl'))
    st_ = dict(abort=None, min_clear=9.9, n_step=0, t0w=float(world.current_time), w0=time.perf_counter())
    log = []

    def w1_rows():
        lb = label.get('w1', '')
        fs = lb.split('fields=')[1].split(',') if 'fields=' in lb else []
        out_ = []
        for r in seg['w1']:
            if len(r) != len(fs) + 1:
                continue
            d_ = dict(zip(fs, r[1:]))
            d_['recv_age_s'] = r[0] - d_['stamp']
            d_['stamp'] = d_['stamp'] - st_['t0w']                         # 擷取戳換成任務時刻
            out_.append(d_)
        return out_

    def loop():
        while True:
            t = float(world.current_time) - st_['t0w']
            q, p, qt = q9()
            if st_['abort'] is None:
                q_cmd = sm.step(t, w1_rows)
                if q_cmd is None:                                          # 接近段：照紀錄
                    k = min(int(np.searchsorted(R_t, t)), len(R_t) - 1)
                    e = R_b[k] - q[:3]
                    e[2] = (e[2] + math.pi) % (2 * math.pi) - math.pi
                    base_world(R_v[k] + 2.0 * e)
                    apply(R_s[k])
                else:
                    base_world(np.zeros(3))
                    apply(q_cmd)
            else:
                base_world(np.zeros(3))
            Tb = bottle_T()
            node.publish_state(float(world.current_time), p, qt, q[3:])
            node_t[0] = float(world.current_time)
            for cap in (v1, w1):
                cap.record_step(float(world.current_time), float(world.current_time), t, q, Tb, ASM.STATES.index(sm.state))
                if sm.state == 'observe':
                    cap.t_sparse = -1e9                                   # 觀察段：每個新影格都存檔
                cap.poll()
            node.drain()
            if st_['n_step'] % 10 == 0:
                P = np.vstack(list(G.link_world(q, C.FINGER_OPEN).values()))
                d = min(float(G.box_dist(P, c_, s_).min()) for _, c_, s_ in G.table_boxes(T1))
                d = min(d, float(G.cyl_dist((np.linalg.inv(Tb0) @ np.c_[P, np.ones(len(P))].T).T[:, :3], 0.0, 0.0, 0.0, G.BOTTLE['h'], G.BOTTLE['r']).min()))
                st_['min_clear'] = min(st_['min_clear'], d)
                mv = float(np.linalg.norm(Tb[:3, 3] - Tb0[:3, 3]))
                log.append((t, ASM.STATES.index(sm.state), (['V0', 'V_LOW', 'V_HIGH'].index(sm.view) if sm.view else -1), d, mv, *q))
                if st_['abort'] is None and (d < CLEAR_ABORT or mv > BOTTLE_MOVE_ABORT):
                    st_['abort'] = dict(t=t, phase=sm.state, view=sm.view, clear=d, bottle_move=mv)
                    print(f"[col] 中止 {st_['abort']}", flush=True)
                    apply(q[3:])
            st_['n_step'] += 1
            world.step(render=True)
            if (sm.t_done is not None and t >= sm.t_done + 1.0) or (st_['abort'] is not None and t >= st_['abort']['t'] + 2.0) or t > t_obs + 60.0:
                return

    def save_all(err=None):
        """正常結束或例外時都呼叫；每一項各自 try，前一項失敗不擋後一項。"""
        rep = dict(layout=dict(table_z=TZ, bottle=[BX, BY]), design=a.design, design_q_look=D['q_look'], src_run=D['src_run'], t_obs=t_obs, T_tr=T_tr, T_tuck=T_tk,
                   library=a.library, trace=sm.trace, final=sm.final, views_tried=sm.tried, t_done=sm.t_done, abort=st_['abort'], min_clear_m=st_['min_clear'],
                   wall_s=time.perf_counter() - st_['w0'], n_step=st_['n_step'], seg3d=dict(n={ns: len(v) for ns, v in seg.items()}, label=label), error=None if err is None else repr(err))
        errs = []
        for nm, fn in (('bottle_move', lambda: rep.update(bottle_move_m=float(np.linalg.norm(bottle_T()[:3, 3] - Tb0[:3, 3])))),
                       ('seg3d', lambda: [np.savez_compressed(os.path.join(a.out, f'seg3d_{ns}.npz'), rows=np.array(rows, float) if rows else np.zeros((0, 1)), label=np.array(label.get(ns, '')))
                                          for ns, rows in seg.items()]),
                       ('collect_log', lambda: np.savez_compressed(os.path.join(a.out, 'collect_log.npz'), rows=np.array(log, float),
                                                                    cols=np.array(['t', 'phase', 'view', 'clear', 'bottle_move', 'x', 'y', 'yaw'] + [f'q{j}' for j in range(1, 7)]))),
                       ('v1', lambda: rep.update(v1=v1.save(a.out))), ('w1', lambda: rep.update(w1_cap=w1.save(a.out)))):
            try:
                fn()
            except Exception as e2:                                        # noqa: BLE001
                errs.append(f'{nm}: {e2!r}')
        rep['save_errors'] = errs
        json.dump(rep, open(os.path.join(a.out, 'collect_report.json'), 'w'), ensure_ascii=False, indent=1, default=float)
        return rep
    ASM.run_with_partial_save(loop, save_all)
    rep = save_all()
    abort = st_['abort']
    print(f"[col] 決策 {sm.final}；視點 {sm.tried}", flush=True)
    print(f"[col] 結果 中止 {abort} 最小淨空 {st_['min_clear']:.3f} m 瓶位移 {rep.get('bottle_move_m', float('nan')) * 1e3:.1f} mm seg3d {rep['seg3d']['n']} 存檔錯誤 {rep['save_errors']}", flush=True)
    rc = 0 if (abort is None and not rep['save_errors']) else 3
    with open(os.path.join(a.out, 'exit_rc.txt'), 'w') as fh:
        fh.write(str(rc))
    return rc


if __name__ == '__main__':
    rc_ = 1
    try:
        rc_ = main()
    except BaseException:                                             # noqa: BLE001  先印出再關 Isaac（close 會直接結束程序，例外訊息會遺失）
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            with open(os.path.join(a.out, 'exit_rc.txt'), 'w') as fh:
                fh.write('1')
        except OSError:
            pass
    finally:
        sim_app.close()
    sys.exit(rc_)
