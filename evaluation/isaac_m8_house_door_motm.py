#!/usr/bin/env python3
"""住宅門 MotM 名目展示（Howard 20261113 13:49；Codex 20261113_141451）。單一 Isaac 程序；控制在主迴圈同步計算（非即時：控制計算期間物理不步進）。
**理想感知＋理想抓持**：把手估計＝把手 prim 真值以 30 Hz 取樣（無偏差、無延遲；腕部相機在抓持距離看不到把手中心，見 Codex 裁定 (a)）；抓持成立後以 FixedJoint 理想綁定。
起始條件（記錄於結果）：底盤瞬移到通過線（full_2 綁定底盤位姿的 y、偏航）上、可抓窗口起點前 L_LEAD＝0.01 m，手臂在預抓構型，靜置後**從靜止起步**；
  前導接近 0.2 s（前導長度受此通過線上 x≈0.80 m 的腕部奇異限制）（移動抓持控制器 HOLD 段，未接觸）→ 進窗口 → 插入 → 閉爪 → 確認 → READY；起跑後不瞬移、不回填時間、不暫停等規劃。
P2 首段：前導第一拍以當時估計與「預測 READY 狀態」送背景規劃程序（啟動模擬器前 fork 的獨立程序、完整快照）；可用時刻＝模擬迴圈實際收到結果的模擬時間；
  閉爪閘門：首段可用且可行才閉爪。
交接（READY 那拍首）：綁定條件（BindGate：TCP 對把手抓取位姿、雙指開度、雙指—橫桿接觸力、相對靜止 0.3 s、無禁止接觸、門角未動）＋
  開門執行層 r14（首段參考以本趟實測 T_HE 重投影手臂、移動承接）複本試跑第一拍，全過才 accept_handover → FixedJoint。
開門：P2 續接（每 0.4 s 請求；σ_s＝σ＋1.2；暫存＋首次影響那拍試跑；末段模式），規劃快照固定用本趟實測 T_HE。期限 50 ms（計算池 6、預熱）。
保護：40 N 工程中止（手腕原始傳遞力）、禁止接觸鎖存、閉鏈／淨空監看、期限失效處置。範圍：開到 85° 停穩為止，不解綁、不退出、不穿門。
--smoke：靜置＋前導＋插入前段到 1.2 s（未接觸；送首段並收結果；讀回與時序）後受控停止。不閉爪、不綁、不開門。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_door_motm.py --out <dir> [--smoke]"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
ap = argparse.ArgumentParser()
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--settle-s', type=float, default=1.0)
ap.add_argument('--smoke', action='store_true')
a = ap.parse_args()

# ---- 啟動模擬器前：只匯入輕量的計算池／規劃程序模組並 fork（子程序自己匯入專案模組）；專案模組在 SimulationApp 之後才匯入
#      （專案模組會放入離線用的 rclpy 替身，若早於 Kit 載入會使 Kit 擴充套件匯入失敗——冒煙 1 實測崩潰 139）
import numpy as np                                               # noqa: E402
os.environ.setdefault('P2_TERM', '1')
os.environ.setdefault('P2_FINAL', '1')
import door_sweep_pool as DSP                                    # noqa: E402
import door_p2_planner_server as PS                              # noqa: E402
POOL = DSP.SweepPool(6)
PLANNER = PS.PlannerServer()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import door_wbopt as WB                                          # noqa: E402
import door_clear_door_fast as _F                                # noqa: E402
WB.clear_door = _F.clear_door_fast
import door_motm_grasp_ctrl as G                                 # noqa: E402
import door_open_exec_ctrl as XO                                 # noqa: E402
import door_body_model as BM                                     # noqa: E402
import door_bind_logic as BL                                     # noqa: E402
import door_full_logic as FL                                     # noqa: E402
import door_motm_p2 as P2                                        # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation    # noqa: E402
from isaacsim.core.utils.types import ArticulationAction         # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics   # noqa: E402
import door_asset_v3 as DA3                                      # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402
from finger_collision import apply_split                         # noqa: E402
from isaac_common import bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402

K, PG = G.K, WB.PG
ROBOT, DOOR = '/World/omni_bot', '/World/door'
JOINT = '/World/door/ideal_grasp_attach'
ARM = [f'joint{i}' for i in range(1, 7)]
DT = 0.01
NPH = G.NPH
V, H_SEG, L_LAT = 0.05, 2.0, 1.2
L_LEAD, T_LEAD = 0.01, 0.2                                         # 冒煙 2：前導 0.15 m 起步會穿過 x≈0.80 的腕部奇異（關節 5 過零）⇒ 前導縮到 1 cm（σ 全程 ≥0.031）
GOAL = math.radians(85.0)
COND = dict(pos_m=0.005, rot_deg=3.0, rel_lin_mps=0.005, rel_ang_rps=0.02, rel_still_s=0.3, finger_q_m=(0.0005, 0.004), finger_contact_n=0.5, door_pre_deg=0.5, forbidden_n=0.05)
GRASP_LIMIT_S, OPEN_LIMIT_S = 12.0, 40.0
PH = ['L', 'I', 'C', 'F', 'R', 'O', 'S']                            # 前導(HOLD)／插入／閉爪／確認／READY／開門／停止
PH_ZH = dict(L='前導接近（預抓保持、移動）', I='插入（移動中）', C='閉爪（移動中）', F='確認（移動中）', R='READY（移動中）', O='開門（P2 續接）', S='停止')


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return clean(o.item())
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot_deg(A, B):
    return math.degrees(math.acos(float(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1))))


def ik(b, T, seed):
    q = np.asarray(seed, float).copy()
    for _ in range(60):
        q9 = np.r_[b, q]; Tc = K.fk(q9, 'link_tcp'); dR = T[:3, :3] @ Tc[:3, :3].T
        e = np.r_[T[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
        if np.linalg.norm(e) < 1e-11:
            break
        J = K.jacobian(q9, 'link_tcp')[:, 3:9]; q = q + np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
    return q


def main():
    os.makedirs(a.out, exist_ok=True)
    reg = json.load(open(a.reg))
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    al = np.asarray(ident['alpha'], float)
    Rg, T_HG, a_H = PG.baseline_reference()
    T_at = lambda T, s: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
    Lf2 = np.load(os.path.join(HERE, 'runs', 'm8house_door_full_2', 'log.npz'))['rows']
    E4 = Lf2[Lf2[:, 1] == 4][-1]
    q_pass = np.r_[E4[5:8]]                                          # 通過線：full_2 綁定底盤位姿（y、偏航；x 為窗口原點 −0.03）
    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    scn = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)]
    sapi = PhysxSchema.PhysxSceneAPI(scn[0]) if scn[0].HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(scn[0])
    sapi.CreateEnableExternalForcesEveryIterationAttr().Set(True)
    env_paths = []
    for o in reg['walls'] + reg['furniture']:
        FixedCuboid(prim_path=f"/World/{o['name']}", name=o['name'], position=np.array(o['c']), scale=np.array(o['s']), color=np.array(o['rgb']))
        env_paths.append(f"/World/{o['name']}")
    dauth = DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root=DOOR)
    ob = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
    for o in ob['static_unknown']:
        y2 = math.radians(o['yaw_deg']) / 2
        FixedCuboid(prim_path=f"/World/unknown/{o['name']}", name=o['name'], position=np.array([o['x'], o['y'], o['size'][2] / 2]), scale=np.array(o['size']),
                    orientation=np.array([math.cos(y2), 0.0, 0.0, math.sin(y2)]), color=np.array([0.72, 0.52, 0.28]))
        env_paths.append(f"/World/unknown/{o['name']}")
    b_start = np.r_[q_pass[0] + 0.03 + L_LEAD, q_pass[1], q_pass[2]]
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(b_start[0]), float(b_start[1]), 0.0))
    xf.AddRotateZOp().Set(float(math.degrees(b_start[2])))
    apply_split(stage, ROBOT)
    if bind_frictionless(stage, collision_prims(stage, ROBOT, lambda p: 'support_' in p) + collision_prims(stage, '/World/ground')) == 0:
        print('[motm] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
        return 19
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(1300.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 20.0, 0.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(450.0)
    bodies, arts = physics_parts(stage, ROBOT)
    nm = lambda b: str(b).rstrip('/').split('/')[-1]              # noqa: E731
    grip = [b for b in bodies if nm(b) == 'uflite_gripper_link']
    fingers = sorted(b for b in bodies if nm(b) in ('uflite_finger1', 'uflite_finger2'))
    others = [b for b in bodies if nm(b) in tuple(f'link{i}' for i in range(1, 7)) + ('link_eef',)]
    base_b = [b for b in bodies if nm(b) == 'base_link']
    if len(grip) != 1 or len(fingers) != 2 or len(others) != 7 or len(base_b) != 1:
        print('[motm] 剛體不符 ⇒ 拒跑', flush=True)
        return 16
    leaf = dauth['leaf_prim']
    v_f = [RigidPrim(prim_paths_expr=str(f), name=f'v_{nm(f)}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=[leaf]) for f in fingers]
    v_o = [RigidPrim(prim_paths_expr=str(b), name=f'v_{nm(b)}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=[leaf] + env_paths) for b in others + grip]
    v_base = RigidPrim(prim_paths_expr=str(base_b[0]), name='v_base', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=[leaf] + env_paths)
    v_grip = RigidPrim(prim_paths_expr=str(grip[0]), name='v_grip_pose')
    v_leaf = RigidPrim(prim_paths_expr=leaf, name='v_leaf_pose')
    world.reset()
    for v in v_f + v_o + [v_base, v_grip, v_leaf]:
        v.initialize()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    hprim = stage.GetPrimAtPath(f'{DOOR}/leaf/handle_bar')

    def truth_handle():                                            # 理想感知的來源（也供評估）
        M = UsdGeom.XformCache().GetLocalToWorldTransform(hprim)
        return np.array([[M[i][j] for j in range(4)] for i in range(4)]).T
    hinge = DA3.readback(stage, DOOR)
    if abs(hinge['drive_stiffness']) > 1e-12 or abs(hinge['drive_damping']) > 1e-12 or not hinge['is_rigid_body']:
        print('[motm] 鉸鏈不是被動 ⇒ 拒跑', flush=True)
        return 64
    bar_axis = 'XYZ'.index(str(hinge['handle_bar']['axis']))
    bar_r, bar_hl = float(hinge['handle_bar']['radius']), 0.5 * float(hinge['handle_bar']['length'])
    T_true0 = truth_handle()
    qa = ik(b_start, T_at(T_true0, PG.S_PRE), E4[8:14])
    q0 = np.array(robot.get_joint_positions(), float)
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i, j in zip(idx, range(6)):
        q0[i] = float(qa[j])
        kp[i], kd[i] = a.kp, a.kd
    q0[f1], q0[f2] = XO.FINGER_OPEN, XO.FINGER_OPEN
    kp[f1], kd[f1] = a.finger_kp, a.finger_kd
    act = np.array(sorted(idx + [f1]), np.int32)
    robot.set_joint_positions(q0.astype(np.float32))
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    kps_rb, _ = robot.get_articulation_controller().get_gains()
    if not all(abs(float(np.asarray(kps_rb)[i]) - a.kp) <= 1e-3 * a.kp for i in idx):
        print('[motm] 增益讀回不符 ⇒ 拒跑', flush=True)
        return 66
    bodies_arm = BM.arm_bodies(K)
    q9a = np.r_[b_start, qa]
    s0 = qa - BM.arm_gravity(K, q9a, bodies_arm) / a.kp               # 靜置設定點：q−s＝G/kp
    tgt = q0.copy()
    for i, j in zip(idx, range(6)):
        tgt[i] = float(s0[j])
    tgt[f1] = XO.FINGER_OPEN

    def apply():
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.asarray(tgt, np.float32)[act], joint_indices=act))

    def base_cmd(u, yw):
        robot.set_linear_velocity(np.array([u[0] * math.cos(yw) - u[1] * math.sin(yw), u[0] * math.sin(yw) + u[1] * math.cos(yw), 0.0], np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, u[2]], np.float32))

    def q9():
        p, qt = robot.get_world_pose()
        jp = np.asarray(robot.get_joint_positions(), float)
        return np.r_[float(p[0]), float(p[1]), yaw_of(qt), jp[idx]], jp

    def door_q():
        _, o = v_leaf.get_world_poses()
        w, x, y, z = (float(v) for v in np.asarray(o)[0])
        return -math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))
    meta = robot._articulation_view._metadata
    j6 = int(meta.joint_indices['joint6']) + 1

    def wrist_F6():
        return np.asarray(robot.get_measured_joint_forces(), float)[j6][:6].copy()

    def rel_T():
        pg, qg = v_grip.get_world_poses()
        pl, ql = v_leaf.get_world_poses()
        Tg_, Tl_ = BL.pose_from_pq(pg[0], qg[0]), BL.pose_from_pq(pl[0], ql[0])
        if Tg_ is None or Tl_ is None:
            return None, Tg_, Tl_
        return np.linalg.inv(Tl_) @ Tg_, Tg_, Tl_

    def finger_contacts():
        Th = truth_handle()
        out_ = []
        for v in v_f:
            d_ = v.get_contact_force_data(dt=DT)
            fn, pts, cnt, st = np.asarray(d_[0], float).reshape(-1), np.asarray(d_[1], float).reshape(-1, 3), np.asarray(d_[4]).reshape(-1), np.asarray(d_[5]).reshape(-1)
            sel = np.concatenate([np.arange(int(s_), int(s_) + int(c_)) for s_, c_ in zip(st, cnt)]) if int(cnt.sum()) else np.zeros(0, int)
            fb, fo, _ = BL.classify_points(pts[sel], fn[sel], Th[:3, 3], Th[:3, bar_axis] / np.linalg.norm(Th[:3, bar_axis]), bar_hl, bar_r)
            out_.append((fb, fo))
        return out_

    def other_contacts():
        mx, who = 0.0, None
        for v, b in zip(v_o + [v_base], others + grip + base_b):
            M = np.asarray(v.get_contact_force_matrix(dt=DT), float).reshape(-1, 3)
            n_ = np.linalg.norm(M, axis=1)
            if len(n_) and float(n_.max()) > mx:
                mx, who = float(n_.max()), nm(b)
        return mx, who
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([2.4, -2.2, 2.2]), target=np.array([float(b_start[0]) - 0.3, float(b_start[1]) + 0.3, 0.8]))
    except Exception:                                              # noqa: BLE001
        pass
    res = dict(run=os.path.basename(a.out.rstrip('/')), mode='smoke' if a.smoke else 'nominal',
               scope='理想感知（把手 prim 真值 30 Hz）＋理想抓持（FixedJoint）；非即時（控制計算期間物理不步進）；範圍：開到 85° 停穩為止（不解綁、不退出、不穿門）',
               init=dict(pass_line_from='full_2 綁定底盤位姿 y／偏航（場景初始條件）', b_start=b_start.tolist(), lead_m=L_LEAD, lead_s=T_LEAD, v_pass=V, q_arm_pregrasp=qa.tolist(),
                         note='起點瞬移到通過線上窗口前 0.01 m、手臂預抓構型（抓取同側分支），靜置後從靜止起步；起跑後不瞬移。前導受 x≈0.80 m 腕部奇異限制（冒煙 2：0.15 m 前導在 1.9 s 手臂速率 1.61 rad/s 超限停止）'),
               gains=dict(kp=a.kp, kd=a.kd))
    # ---- 靜置
    quat0 = np.array([math.cos(b_start[2] / 2), 0.0, 0.0, math.sin(b_start[2] / 2)], np.float32)
    for k in range(int(round(a.settle_s / DT))):
        if k < 5:
            robot.set_world_pose(np.array([b_start[0], b_start[1], 0.0], np.float32), quat0)
        apply()
        base_cmd(np.zeros(3), 0.0)
        world.step(render=True)
    qs, jps = q9()
    off_static = qs[3:] - s0                                          # 靜置實測靜態偏置（含未建模）
    door0 = door_q()
    res['settle'] = dict(base=qs[:3].tolist(), arm_err_vs_pregrasp=[round(float(x), 5) for x in (qs[3:] - qa)], off_static=off_static.tolist(), door_deg=round(math.degrees(door0), 5))
    # ---- 理想感知影格
    frames = []

    def capture(t):
        if not frames or t >= frames[-1][0] + 1.0 / 30.0 - 1e-9:
            frames.append((round(t, 4), truth_handle()))
            if len(frames) > 200:
                del frames[:100]

    def est_fn(t):
        for tc, T in reversed(frames):
            if tc <= t + 1e-9:
                return T, tc, tc
        return None
    gate = BL.BindGate(COND, DT)
    t, t_wall0 = 0.0, time.perf_counter()
    capture(0.0)
    gc = G.MovingGraspCtrl(ident, est_fn, np.array([-1.0, 0.0]), V, (L_LEAD, L_LEAD + 0.22), s0.copy(), [np.zeros(9) for _ in range(G.D_STEPS)], np.zeros(9), np.zeros(9), off_static,
                           sched=dict(hold_s=0.3 + T_LEAD), pool=POOL, fastkin=True, deadline_ms=50.0)
    first = dict(rid=None, t_req=None, wall_req=None, t_avail=None, wall_avail=None, ok=False, res=None)
    gc.close_gate = lambda cc: first['ok']
    ev = []
    rows, ts = [], []
    STATE.update(res=res, rows=rows, ts=ts, ev=ev)                    # 例外時存部分紀錄（名目趟 1 例外未存檔）
    box = dict(mode='grasp', ex=None, aborted=None, t_bind=None, T_WH_b=None, T_HE=None, phi_b=None, helper=None, pend=None, t_next=None, k=1, decisions=[], reqs=[])

    def log_event(what, **kw):
        ev.append(dict(t=round(t, 3), wall=round(time.perf_counter() - t_wall0, 3), what=what, **kw))
        print(f"[motm] t={t:.2f} {what} {json.dumps(clean(kw), ensure_ascii=False)[:300]}", flush=True)
    phase_i = lambda: PH.index(dict(HOLD='L', INSERT='I', CLOSE='C', CONFIRM='F', READY='R')[gc.phase]) if box['mode'] == 'grasp' else 5   # noqa: E731
    t_end_lim = 1.2 if a.smoke else GRASP_LIMIT_S
    prev_b = qs[:3].copy()
    # ================= 移動抓持 =================
    while t < t_end_lim and box['mode'] == 'grasp':
        q, jp = q9()
        tick = gc.n_step % NPH == 0
        # P2 首段：前導第一拍送出（預測 READY 狀態）
        if tick and first['rid'] is None and gc.t0 is not None and gc.T_est is not None:
            T_plan = gc.T_est.copy()
            t_rel = t - gc.t0
            t_ready_rel = gc.sched['hold_s'] + gc.sched['ins_s'] + gc.sched['close_s'] + gc.sched['confirm_s'] - G.DT
            b_pred = q[:3] + np.r_[np.array([-V, 0.0]) * (t_ready_rel - t_rel), 0.0]
            qa_pred = ik(b_pred, T_at(T_plan, 0.0), q[3:])
            phi_now = door_q()
            x_pred = np.r_[b_pred, qa_pred, phi_now]
            T_HE_pred = np.linalg.inv(T_plan) @ T_at(T_plan, 0.0)
            WB.set_model(phi0=phi_now, finger_q=G.FINGER_CLOSED)
            hlp = WB.DoorWBOpt(T_plan, np.eye(4), x_pred, GOAL, N=2, dt=0.2, T_HE=T_HE_pred, project_start=False)
            v_pred = np.r_[-V, 0.0, 0.0, hlp.closure_velocity(x_pred, np.r_[-V, 0.0, 0.0], 0.0), 0.0]
            first.update(rid=1, t_req=t, wall_req=time.perf_counter() - t_wall0, T_plan=T_plan, T_HE_pred=T_HE_pred, x_pred=x_pred, t_ready_pred=gc.t0 + t_ready_rel, phi_req=phi_now)
            PLANNER.submit(1, dict(T_WH0=T_plan, T_HE=T_HE_pred, x0=x_pred, v0=v_pred, H=H_SEG, goal=GOAL, fq=G.FINGER_CLOSED, phi_model=phi_now, try_final=True))
            log_event('first_segment_request', t_ready_pred=round(gc.t0 + t_ready_rel, 3), b_pred=[round(float(x), 4) for x in b_pred])
        r_ = PLANNER.poll() if first['rid'] is not None and first['res'] is None else None
        if r_ is not None:
            first.update(res=r_, t_avail=t, wall_avail=time.perf_counter() - t_wall0, ok=bool(r_[2] and r_[1] is not None))
            log_event('first_segment_available', feasible=r_[2], failed=r_[3], final=r_[4], plan_wall_s=round(r_[5], 3), info=r_[6])
        # 交接（READY 那拍首）
        if tick and gc.ready_now and gc.ready_ver is not None and gc.stop is None and not a.smoke:
            w_ho = time.perf_counter()                                   # 交接計算（綁定條件、建執行層、預熱、試跑）的牆鐘：在本體 50 ms 計時之外，另記
            Th = truth_handle()
            Ttcp = K.fk(q, 'link_tcp')
            Tg_des = T_at(Th, 0.0)
            fc = finger_contacts()
            c = dict(pos_m=float(np.linalg.norm(Ttcp[:3, 3] - Tg_des[:3, 3])), rot_deg=rot_deg(Ttcp, Tg_des), finger_q=[float(jp[f1]), float(jp[f2])], finger_bar_n=[float(fc[0][0]), float(fc[1][0])],
                     door_pre_deg=math.degrees(abs(door_q() - door0)))
            okb, okd = gate.decide(c['pos_m'], c['rot_deg'], c['finger_q'], c['finger_bar_n'], c['door_pre_deg'])
            phi_b = door_q()
            T_WH_b = gc.handle_ref(phi_b)
            T_HE_act = np.linalg.inv(T_WH_b) @ Ttcp
            fq_b = 0.5 * (float(jp[f1]) + float(jp[f2]))
            Xd0 = first['res'][1]
            WB.set_model(phi0=phi_b, finger_q=fq_b)
            hlp2 = WB.DoorWBOpt(T_WH_b, np.eye(4), np.r_[q, phi_b], GOAL, N=2, dt=0.2, T_HE=T_HE_act, project_start=True)
            Xd0p = np.array([hlp2.project(x, iters=30) for x in Xd0])     # 首段參考以本趟實測 T_HE 重投影手臂（底盤、門角不變）；以試跑重核
            ho = dict(s=gc.s.copy(), pipe=[u_.copy() for u_ in gc.pipe], u_last=gc.u_last.copy(), u_prev=gc.u_prev.copy(), bias=np.asarray(gc.cfg.arm_model.bias, float).copy(), moving=True, off_static=gc.off(q))
            exn = XO.DoorOpenExec(ident, Xd0p, 0.05, T_WH_b, phi_b, T_HE_act, GOAL, ho, fq_b, arm_mode='closure', joint_kp=a.kp,
                                  g2=dict(gravity2=True, ik_fc=True, pool=POOL, prep_fc=True, deadline_ms=50.0))
            exn.p2_open = True
            wu = exn.warmup(q, phi_b)                                     # 交接前預熱（複本空跑，本體不動）；試跑前做，避免試跑拍含冷啟成本
            w_dry = time.perf_counter()
            tmp = exn._light_copy()
            tmp.step(0.0, q.copy(), phi_b, float(np.linalg.norm(wrist_F6()[:3])))
            dry_ms = (time.perf_counter() - w_dry) * 1e3
            dry = dict(stop=None if tmp.stop is None else tmp.stop.get('why'), abort=None if tmp.abort is None else tmp.abort.get('why'), cl_fail=tmp._cl_fail, fb_ok=bool(tmp._fb_ok),
                       stop_closure_pos_mm=round(tmp.mins['stop_closure_pos'] * 1e3, 3), reproj_arm_max_rad=round(float(np.abs(Xd0p[:, 3:9] - Xd0[:, 3:9]).max()), 5),
                       start_base_vs_meas_mm=round(float(np.linalg.norm(Xd0p[0, :2] - q[:2])) * 1e3, 3))
            valid = tmp.stop is None and tmp.abort is None and tmp._cl_fail is None and tmp.mins['stop_closure_pos'] <= XO.CL_TOL['pos_m'] and tmp._fb_ok
            log_event('handover_attempt', bind_ok=okb, bind=okd, bind_values=c, dry_ok=valid, dry=dry, ready_ver=gc.ready_ver, warmup_ms=round(wu, 1), dry_run_ms=round(dry_ms, 1),
                      handover_compute_ms=round((time.perf_counter() - w_ho) * 1e3, 1), note='交接計算在本體控制拍計時之外；計算期間物理不步進')
            if okb and valid and gc.accept_handover(gc.ready_ver):
                Rr0, Tg_b, _ = rel_T()
                j = UsdPhysics.FixedJoint.Define(stage, JOINT)
                j.CreateBody0Rel().SetTargets([Sdf.Path(str(grip[0]))])
                j.CreateBody1Rel().SetTargets([Sdf.Path(leaf)])
                j.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                qr = W0.R_to_wxyz(Rr0[:3, :3])
                j.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in Rr0[:3, 3]]))
                j.CreateLocalRot1Attr().Set(Gf.Quatf(float(qr[0]), float(qr[1]), float(qr[2]), float(qr[3])))
                j.CreateJointEnabledAttr().Set(True)
                j.CreateBreakForceAttr().Set(float('inf'))
                j.CreateBreakTorqueAttr().Set(float('inf'))
                box.update(mode='open', ex=exn, t_bind=t, T_WH_b=T_WH_b, T_HE=T_HE_act, phi_b=phi_b, fq=fq_b, helper=hlp2, t_next=t + P2.T_RE)
                np.savez(os.path.join(a.out, 'first_segment.npz'), Xd0=Xd0, Xd0p=Xd0p, T_plan=first['T_plan'], T_HE_pred=first['T_HE_pred'], T_WH_b=T_WH_b, T_HE_act=T_HE_act, phi_b=phi_b)
                log_event('bind_and_handover', base_speed_cmd=round(float(np.linalg.norm(gc.u_last[:2])), 4), joint_enabled=bool(UsdPhysics.Joint(stage.GetPrimAtPath(JOINT)).GetJointEnabledAttr().Get()), warmup_ms=round(wu, 1))
                break
        o = gc.step(t, q.copy(), door_q())
        if o.get('abort'):
            box['aborted'] = dict(t=round(t, 3), controller=gc.abort)
            log_event('grasp_abort', abort=gc.abort)
            break
        for i, j_ in zip(idx, range(6)):
            tgt[i] = float(o['arm_target'][j_])
        tgt[f1] = float(o['finger'])
        apply()
        base_cmd(o['base_twist'], float(q[2]))
        world.step(render=True)
        t += DT
        capture(t)
        q2, jp2 = q9()
        Rr, _, _ = rel_T()
        fc = finger_contacts()
        oc, ow = other_contacts()
        gate.step_seq(float(world.current_time), Rr, oc, max(fc[0][1], fc[1][1]))
        F6 = wrist_F6()
        rows.append([t, phase_i() if gc.stop is None else 6, door_q(), float(jp2[f1]), float(jp2[f2]), *q2, *F6, time.perf_counter() - t_wall0])
        if gc.n_step % NPH == 0:
            bsp = float(np.linalg.norm(q2[:2] - prev_b[:2])) / G.DT
            prev_b = q2[:3].copy()
            Tt = T_at(truth_handle(), 0.0)
            ts.append([t, time.perf_counter() - t_wall0, 0, phase_i(), bsp, door_q(), float(np.linalg.norm(K.fk(q2, 'link_tcp')[:3, 3] - Tt[:3, 3])), np.nan, np.nan])
        if gate.latched is not None and gc.stop is None:
            log_event('forbidden_contact_latched', latch=gate.latched, who=ow)
            gc._stop('exec_forbidden_contact', contact_info=dict(gate.latched, who=ow))
        if gc.stop is not None and gc.stop_ok is not None:
            log_event('grasp_stopped', stop=gc.stop, stop_ok=gc.stop_ok)
            break
        if a.smoke and t >= 1.2 - 1e-9 and gc.stop is None:              # 冒煙：INSERT 中途（未接觸、未閉爪）即停
            gc._stop('smoke_end')
            log_event('smoke_stop_requested')
            t_end_lim = t + 2.0
    rep_g = gc.report()
    res['grasp'] = {k_: rep_g.get(k_) for k_ in ('phases', 'stop', 'abort', 'stop_effect', 'ready_t_rel', 'accepted', 'accept_t_rel', 'n_ready_revoked', 'min_clear', 'limits', 'contact_lock', 'lock_inconsistency_max', 'contact_stop', 'fast_slip')}
    res['grasp']['n_deadline_miss'] = gc.n_deadline_miss
    res['grasp']['tick_log'] = gc.tick_log
    res['first_segment'] = {k_: first.get(k_) for k_ in ('t_req', 'wall_req', 't_avail', 'wall_avail', 'ok', 't_ready_pred', 'phi_req')}
    if first['res'] is not None:
        res['first_segment'].update(feasible=first['res'][2], failed=first['res'][3], final=first['res'][4], plan_wall_s=round(first['res'][5], 3), info=first['res'][6])
    # ================= 開門（P2 續接）=================
    if box['mode'] == 'open':
        ex = box['ex']
        olog = []
        while t - box['t_bind'] < OPEN_LIMIT_S:
            q, jp = q9()
            tr = t - box['t_bind']
            phm = door_q()
            tick = ex.n_step % NPH == 0
            if tick and ex.staged is not None and ex.stop is None:
                dec = ex.staged_step(tr, q.copy(), phm)
                if isinstance(dec, dict):
                    box['decisions'].append(dec)
                    log_event('p2_decision', ok=dec.get('ok'), ver=dec.get('ver'), why=dec.get('why'), dry=dec.get('dry'))
            r_ = PLANNER.poll() if box['pend'] is not None else None
            if r_ is not None:
                pend = box['pend']; box['pend'] = None
                rec = pend['rec']
                rec.update(t_avail=round(t, 3), wall_avail=round(time.perf_counter() - t_wall0, 3), plan_wall_s=round(r_[5], 3), feasible=r_[2], final=r_[4], failed=r_[3])
                xr = ex.ref(ex.sigma)
                ey, ed = abs(math.degrees(q[2] - xr[2])), abs(math.degrees(phm - xr[9]))
                rec['track_err'] = dict(base_mm=round(float(np.linalg.norm(q[:2] - xr[:2])) * 1e3, 3), yaw_deg=round(ey, 3), door_deg=round(ed, 3))
                if not r_[2] or r_[1] is None:
                    rec['result'] = 'infeasible'
                elif ex.sigma > pend['sigma_s'] + 1e-9 and not pend['rest']:
                    rec['result'] = 'missed_deadline'
                elif ey > 1.0 or ed > 1.0:
                    rec['result'] = 'precheck_rejected'
                elif ex.staged is not None or ex.stop is not None:
                    rec['result'] = 'staged_busy_or_stopped'
                else:
                    okS, info = ex.stage_reference(r_[1], pend['sigma_s'], final=r_[4], ver=rec['k'])
                    rec['result'] = 'staged' if okS else 'stage_rejected'
                    if not okS:
                        rec['why'] = info.get('why')
                log_event('p2_plan_result', k=rec['k'], result=rec['result'], plan_wall_s=rec['plan_wall_s'], final=rec['final'])
            if box['pend'] is None and ex.stop is None and ex.p2_open and t >= box['t_next'] - 1e-9:
                sig_s = math.ceil((ex.sigma + L_LAT) / ex.h - 1e-9) * ex.h
                rest = sig_s >= ex.sig_end - 1e-9
                still_ok = ex.rate['arm'] <= XO.END['arm_rps'] and ex.rate['base'] <= XO.END['base_mps'] and ex.rate['door'] <= XO.END['door_rps']
                if rest and not still_ok:
                    box['t_next'] = t + 0.05
                else:
                    if rest:
                        sig_s = ex.sig_end
                    i_ = min(int(round(sig_s / ex.h)), len(ex.Xd) - 1)
                    xs = ex.Xd[i_].copy()
                    if rest or i_ >= len(ex.Xd) - 1:
                        vs = np.zeros(10)
                    else:
                        vv = (ex.Xd[min(i_ + 1, len(ex.Xd) - 1)] - ex.Xd[max(i_ - 1, 0)]) / (2 * ex.h)
                        WB.set_model(phi0=box['phi_b'], finger_q=box['fq'])
                        vs = np.r_[vv[:3], box['helper'].closure_velocity(xs, vv[:3], vv[9]), vv[9]]
                    box['k'] += 1
                    rec = dict(k=box['k'], t_req=round(t, 3), wall_req=round(time.perf_counter() - t_wall0, 3), sigma=round(ex.sigma, 3), sigma_s=round(sig_s, 3), rest=rest)
                    box['reqs'].append(rec)
                    PLANNER.submit(box['k'], dict(T_WH0=box['T_WH_b'], T_HE=box['T_HE'], x0=xs, v0=vs, H=H_SEG, goal=GOAL, fq=box['fq'], phi_model=box['phi_b'], try_final=True))
                    box['pend'] = dict(sigma_s=sig_s, rest=rest, rec=rec)
                    box['t_next'] = t + P2.T_RE
            F6 = wrist_F6()
            fn_ = float(np.linalg.norm(F6[:3]))
            out = ex.step(tr, q, phm, fn_)
            if out['abort']:
                box['aborted'] = dict(t=round(t, 3), controller=ex.abort)
                log_event('open_abort', abort=ex.abort)
                break
            for i, j_ in zip(idx, range(6)):
                tgt[i] = float(out['arm_target'][j_])
            apply()
            base_cmd(out['base_twist'], float(q[2]))
            world.step(render=True)
            t += DT
            q2, jp2 = q9()
            Rr, _, _ = rel_T()
            fc = finger_contacts()
            oc, ow = other_contacts()
            gate.step_seq(float(world.current_time), Rr, oc, max(fc[0][1], fc[1][1]))
            F6 = wrist_F6()
            rows.append([t, 5 if ex.stop is None else 6, door_q(), float(jp2[f1]), float(jp2[f2]), *q2, *F6, time.perf_counter() - t_wall0])
            olog.append(np.r_[tr, q2, door_q(), fn_, ex.sigma, ex.ref(ex.sigma)[9]])
            if ex.n_step % NPH == 0:
                bsp = float(np.linalg.norm(q2[:2] - prev_b[:2])) / G.DT
                prev_b = q2[:3].copy()
                xr = ex.ref(ex.sigma)
                ts.append([t, time.perf_counter() - t_wall0, 1, 5 if ex.stop is None else 6, bsp, door_q(), np.nan, ex.sigma, float(xr[9])])
            if gate.latched is not None and ex.stop is None:
                log_event('forbidden_contact_latched', latch=gate.latched, who=ow)
                BL.forward_forbidden(gate.latched, ow, ex=ex)
            if ex.stop is not None and ex.stop_ok is not None and t - box['t_bind'] - ex.stop_ok['t'] >= 0.3:
                log_event('open_end', stop=ex.stop, done=ex.done, stop_ok=ex.stop_ok)
                break
        if box['aborted'] is None and ex.stop_ok is None:
            ex._stop('open_time_limit')
        rep_o = ex.report()
        res['open'] = {k_: rep_o.get(k_) for k_ in ('arm_mode', 'closure_anchor', 'closure_stats', 'stop', 'done', 'stop_effect', 'phi_end_deg', 't_end', 'closure_measured', 'min_clear_actual', 'force_max_n', 'timing')}
        res['open'].update(n_deadline_miss=ex.n_deadline_miss, p2_log=ex.p2_log, reqs=box['reqs'])
        np.savez_compressed(os.path.join(a.out, 'open_steps.npz'), meas=np.array(olog), Xd_final=ex.Xd, h=ex.h, ctrl=np.array(ex.step_log))
    res['events'] = ev
    res['forbidden_contact'] = dict(latched=gate.latched)
    res['aborted'] = box['aborted']
    res['wall_total_s'] = round(time.perf_counter() - t_wall0, 1)
    res['sim_total_s'] = round(t, 2)
    res['rtf'] = round(t / max(res['wall_total_s'], 1e-9), 3)
    o_ = res.get('open') or {}
    if a.smoke:
        sw = ((res['grasp'] or {}).get('stop') or {}).get('why')
        res['result'] = 'smoke_done' if (res['first_segment'].get('t_avail') is not None and box['aborted'] is None and sw in (None, 'smoke_end')) else 'smoke_blocked：' + str(sw)
    elif box['aborted'] is not None:
        res['result'] = 'aborted：模擬中止（停止有效性未驗證；綁定中則未解綁）'
    elif o_.get('done') and (o_.get('stop_effect') or {}).get('ok'):
        res['result'] = 'open_done_85'
    elif box['mode'] == 'grasp':
        res['result'] = 'grasp_not_handed_over：' + str((res['grasp'] or {}).get('stop'))
    else:
        res['result'] = 'open_not_done：' + str(o_.get('stop'))
    np.savez_compressed(os.path.join(a.out, 'log.npz'), rows=np.array(rows), cols=np.array(['t', 'phase', 'door_q', 'f1', 'f2', 'bx', 'by', 'byaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz', 'wall']),
                        phases=np.array(PH), ts=np.array(ts), ts_cols=np.array(['t', 'wall', 'ctrl', 'phase', 'base_speed', 'door_q', 'tcp_err_grasp', 'sigma', 'phi_ref']))
    json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[motm] 完成 result={res['result']} sim={res['sim_total_s']} s wall={res['wall_total_s']} s RTF={res['rtf']}", flush=True)
    return 0 if res['result'] in ('open_done_85', 'smoke_done') else 5


STATE = {}


def save_partial(tb):
    """例外中斷時存部分紀錄：log_partial.npz（逐步列、時間序列）＋result_partial.json（事件、例外）。"""
    try:
        if STATE.get('rows'):
            np.savez_compressed(os.path.join(a.out, 'log_partial.npz'), rows=np.array(STATE['rows']), ts=np.array(STATE['ts']),
                                cols=np.array(['t', 'phase', 'door_q', 'f1', 'f2', 'bx', 'by', 'byaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz', 'wall']), phases=np.array(PH))
        r = dict(STATE.get('res') or {}, events=STATE.get('ev'), exception=tb, result='exception：程式例外中斷（非受控停止；停止有效性未驗證）')
        json.dump(clean(r), open(os.path.join(a.out, 'result_partial.json'), 'w'), ensure_ascii=False, indent=1)
    except Exception as e_:                                          # noqa: BLE001
        print('[motm] 部分紀錄存檔失敗', repr(e_), flush=True)


if __name__ == '__main__':
    rc = 99
    try:
        rc = main()
    except Exception:
        import traceback
        tb = traceback.format_exc()
        print('[motm] 例外', tb, flush=True)
        save_partial(tb)
    try:
        POOL.close()
        PLANNER.close()
    except Exception:                                              # noqa: BLE001
        pass
    sim_app.close()
    sys.exit(rc)
