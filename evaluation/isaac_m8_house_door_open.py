#!/usr/bin/env python3
"""住宅場景門把手：**門機構約束全身最佳化開門**，Isaac 正式一趟（理想抓持綁定；Codex 20261112_223736 條件授權）。
定位：理想抓持下，新機構約束規劃能否實際帶門開啟的**名目功能驗證**。異常處置依賴模擬中止，不宣稱完整安全停止或實機可用。不接導航、不沿用舊開門控制、不含退出。
流程（承 smoke_3，不改）：P 預抓擺位靜置 → I door_insert_ctrl 插入 → B 閉爪 → C BindGate 條件 → D 綁定（FixedJoint，當下相對位姿，理想承載）→ E 綁定靜置 0.5 s
接著（新）：
  規劃  暫停（不呼叫 world.step）以綁定瞬間**實測**狀態跑 door_wbopt：φ_bind＝鉸鏈讀值（模擬量測）；T_WH_bind＝Rz[−φ_bind]·把手估計（online_1 起點影格，φ＝0 時）；
        T_HE＝T_WH_bind⁻¹·T_WE(實測構型 FK)（之後固定）；淨空模型手指開度＝兩指實測閉合開度平均；起點不投影（殘差 >1e-6 拒絕）。非即時。不可行 ⇒ 不動作、結束。
  O 開門 door_open_exec_ctrl（同一全身求解＋進度調節＋提交前停止掃掠＋path 停止）；承接插入控制器的設定點、8 步管線、上一命令、偏置。
        門角輸入＝鉸鏈讀值（模擬量測；實機無此感測）；力輸入＝手腕傳遞力範數（列＝metadata joint6＋1，未校準，40 N 為模擬工程中止值）。
        禁止接觸（手臂六連桿／link_eef／夾爪殼／底盤 對 門扇與牆家具；手指對橫桿以外）出現 ⇒ 控制器 abort。
        **abort ⇒ 在下一次 world.step 前停止物理步進**，鎖存、保存狀態，不解綁、不 hold、不退出；停止有效性標未驗證。
  U 正常完成（open_done 或 path→settled）且停止有效後：解除連接並讀回 jointEnabled＝False。
  R 張爪斜坡 0.5 s＋觀察 2 s：底盤零、手臂保持；每步以兩指**實際開度**算淨空；記門角、接觸。
底盤：逐物理步以控制器輸出設定底盤速度＝理想底盤介面。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_door_open.py --out <dir> [--wiring]"""
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
ap.add_argument('--hold-s', type=float, default=3.0)
ap.add_argument('--wiring', action='store_true', help='不動作接線核對：只擺位靜置、讀回、以合成綁定狀態試規劃與控制器一步，不插入、不閉爪、不綁、不開門')
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation    # noqa: E402
from isaacsim.core.utils.types import ArticulationAction         # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics   # noqa: E402

import door_bind_logic as BL                                     # noqa: E402
import door_insert_ctrl as DA                                    # noqa: E402
import door_open_exec_ctrl as XO                                 # noqa: E402
import door_wbopt as WB                                          # noqa: E402
import door_asset_v3 as DA3                                      # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402
from finger_collision import apply_split                         # noqa: E402
from isaac_common import bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402

PG, E, K = DA.PG, DA.R4.E, DA.K
ROBOT, DOOR = '/World/omni_bot', '/World/door'
JOINT = '/World/door/ideal_grasp_attach'
ARM = [f'joint{i}' for i in range(1, 7)]
DT = 0.01
FINGER_OPEN = 0.0089
COND = dict(pos_m=0.005, rot_deg=3.0, rel_lin_mps=0.005, rel_ang_rps=0.02, rel_still_s=0.3, finger_q_m=(0.0005, 0.004), finger_contact_n=0.5, door_pre_deg=0.5, forbidden_n=0.05)
CTRL_LIMIT_S = 12.0
PHI_GOAL = math.radians(85.0)
OPEN_LIMIT_S = 40.0                                              # 開門段模擬時間上限（逾時＝abort）
PLAN_WALL_LIMIT_S = 600.0


def run_responsive(world, fn):
    """在背景執行緒跑 fn（規劃）；主執行緒只呼叫 world.render()（算圖、不步進物理）讓視窗保持回應。
    m8house_door_open_wiring_1 在主執行緒阻塞約 2 分鐘時被 SIGKILL（原因未定；視窗無回應是候選），r2 起改此作法。回 (結果, 例外)。"""
    import threading
    box = {}

    def _t():
        try:
            box['r'] = fn()
        except Exception as e_:                                    # noqa: BLE001
            box['e'] = e_
    th = threading.Thread(target=_t, daemon=True)
    th.start()
    while th.is_alive():
        world.render()
        time.sleep(0.05)
    return box.get('r'), box.get('e')


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


def T_of(p, q):
    T = np.eye(4)
    T[:3, :3] = W0.wxyz_to_R(np.asarray(q, float))
    T[:3, 3] = np.asarray(p, float)
    return T


def rot_deg(A, B):
    return math.degrees(math.acos(float(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1))))


def main():
    os.makedirs(a.out, exist_ok=True)
    reg = json.load(open(a.reg))
    r1 = json.load(open(os.path.join(HERE, 'runs', 'm8house_da_online_1', 'result.json')))
    T_far, b1 = np.array(r1['online']['T_far']), np.array(r1['online']['b1'])
    q_seed = np.array(r1['controller']['replan']['q_pre'])
    Rg, T_HG, a_H = PG.baseline_reference()
    T_at = lambda T, s: np.asarray(E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
    qg9 = PG.ik(b1, q_seed, T_at(T_far, PG.S_PRE), extra_seeds=True)
    if qg9 is None:
        print('[bind] 預抓 IK 無解 ⇒ 拒跑', flush=True)
        return 3
    q_grasp = qg9[3:9]
    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    scn = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)][0]
    sapi = PhysxSchema.PhysxSceneAPI(scn) if scn.HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(scn)
    sapi.CreateEnableExternalForcesEveryIterationAttr().Set(True)
    env_paths = []
    for o in reg['walls'] + reg['furniture']:
        FixedCuboid(prim_path=f"/World/{o['name']}", name=o['name'], position=np.array(o['c']), scale=np.array(o['s']), color=np.array(o['rgb']))
        env_paths.append(f"/World/{o['name']}")
    dauth = DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root=DOOR)
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(b1[0]), float(b1[1]), 0.0))
    xf.AddRotateZOp().Set(float(math.degrees(b1[2])))
    apply_split(stage, ROBOT)
    if bind_frictionless(stage, collision_prims(stage, ROBOT, lambda p: 'support_' in p) + collision_prims(stage, '/World/ground')) == 0:
        print('[bind] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
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
        print(f'[bind] 剛體不符 ⇒ 拒跑 grip {grip} fingers {fingers} others {len(others)}', flush=True)
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
    q0 = np.array(robot.get_joint_positions(), float)
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i, j in zip(idx, range(6)):
        q0[i] = float(q_grasp[j])
        kp[i], kd[i] = a.kp, a.kd
    q0[f1], q0[f2] = FINGER_OPEN, FINGER_OPEN
    kp[f1], kd[f1] = a.finger_kp, a.finger_kd
    act = np.array(sorted(idx + [f1]), np.int32)
    robot.set_joint_positions(q0.astype(np.float32))
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    tgt = q0.copy()

    def apply():
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.asarray(tgt, np.float32)[act], joint_indices=act))

    hinge = DA3.readback(stage, DOOR)
    rb = dict(physics_dt=float(world.get_physics_dt()), hinge={k: hinge.get(k) for k in ('joint_type', 'joint_axis', 'limit_lower_deg', 'limit_upper_deg', 'drive_stiffness', 'drive_damping', 'drive_max_force', 'mass_kg', 'angular_damping', 'joint_friction', 'is_rigid_body')},
              damping_note='PhysX 剛體角阻尼（angular_damping），不是鉸鏈黏滯係數', base=b1.tolist(), q_pregrasp=q_grasp.tolist(), handle_source='online_1 起點影格估計', cond=COND)
    print(f'[bind] 讀回 {rb}', flush=True)
    if abs(rb['hinge']['drive_stiffness']) > 1e-12 or abs(rb['hinge']['drive_damping']) > 1e-12 or not rb['hinge']['is_rigid_body']:
        print('[bind] 鉸鏈不是被動（有驅動）或門扇非剛體 ⇒ 拒跑', flush=True)
        return 64
    hprim = stage.GetPrimAtPath(f'{DOOR}/leaf/handle_bar')

    def handle_T():
        M = UsdGeom.XformCache().GetLocalToWorldTransform(hprim)
        return np.array([[M[i][j] for j in range(4)] for i in range(4)]).T

    def door_q():
        _, o = v_leaf.get_world_poses()
        w, x, y, z = (float(v) for v in np.asarray(o)[0])
        return -math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))

    raw = [[float('nan')] * 14]

    def rel_T():
        pg, qg = v_grip.get_world_poses()
        pl, ql = v_leaf.get_world_poses()
        raw[0] = [*np.asarray(pg[0], float), *np.asarray(qg[0], float), *np.asarray(pl[0], float), *np.asarray(ql[0], float)]
        Tg_, Tl_ = BL.pose_from_pq(pg[0], qg[0]), BL.pose_from_pq(pl[0], ql[0])
        if Tg_ is None or Tl_ is None:
            return None, Tg_, Tl_
        return np.linalg.inv(Tl_) @ Tg_, Tg_, Tl_

    def rel_vel():
        lv = np.asarray(v_grip.get_linear_velocities())[0] - np.asarray(v_leaf.get_linear_velocities())[0]
        av = np.asarray(v_grip.get_angular_velocities())[0] - np.asarray(v_leaf.get_angular_velocities())[0]
        return float(np.linalg.norm(lv)), float(np.linalg.norm(av))

    bar_axis = 'XYZ'.index(str(hinge['handle_bar']['axis']))
    bar_r, bar_hl = float(hinge['handle_bar']['radius']), 0.5 * float(hinge['handle_bar']['length'])

    def finger_contacts():
        """各指：回 (對橫桿合力, 對橫桿以外合力)。接觸點到橫桿軸線段距離 ≤ 半徑＋3 mm 算橫桿。"""
        Th = handle_T()
        out = []
        for v in v_f:
            d_ = v.get_contact_force_data(dt=DT)
            fn, pts, cnt, st = np.asarray(d_[0], float).reshape(-1), np.asarray(d_[1], float).reshape(-1, 3), np.asarray(d_[4]).reshape(-1), np.asarray(d_[5]).reshape(-1)
            sel = np.concatenate([np.arange(int(s_), int(s_) + int(c_)) for s_, c_ in zip(st, cnt)]) if int(cnt.sum()) else np.zeros(0, int)
            fb, fo, _ = BL.classify_points(pts[sel], fn[sel], Th[:3, 3], Th[:3, bar_axis] / np.linalg.norm(Th[:3, bar_axis]), bar_hl, bar_r)
            out.append((fb, fo))
        return out

    def other_contacts():
        mx, who = 0.0, None
        for v, b in zip(v_o + [v_base], others + grip + base_b):
            M = np.asarray(v.get_contact_force_matrix(dt=DT), float).reshape(-1, 3)
            n_ = np.linalg.norm(M, axis=1)
            if len(n_) and float(n_.max()) > mx:
                mx, who = float(n_.max()), nm(b)
        return mx, who

    meta = robot._articulation_view._metadata
    j6 = int(meta.joint_indices['joint6']) + 1                      # 安裝版 API：完整關節索引＋1＝該關節子連桿的列（不是 dof 索引）

    def wrist_F():
        F = np.asarray(robot.get_measured_joint_forces(), float)
        return F[j6][:3].copy(), F.shape

    rows, events = [], []
    t = 0.0
    gate = BL.BindGate(COND, DT)
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    ctrl, prev_rel = None, [None]
    PH = ['P', 'I', 'B', 'D', 'E', 'O', 'U', 'R']
    ex_box = dict(ex=None, t0=0.0, aborted=None, force=None)

    def q9():
        p, qt = robot.get_world_pose()
        w, x, y, z = (float(v) for v in qt)
        jp = np.asarray(robot.get_joint_positions(), float)
        return np.r_[float(p[0]), float(p[1]), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)), jp[idx]], jp

    def step(phase, n):
        nonlocal t
        for _ in range(n):
            q, jp = q9()
            ex = ex_box['ex']
            if ex is not None:
                Fw, _ = wrist_F()
                ex_box['force'] = float(np.linalg.norm(Fw))
                out = ex.step(t - ex_box['t0'], q, door_q(), ex_box['force'])
                if out['abort']:                                       # 模擬中止：不再步進物理
                    ex_box['aborted'] = dict(t=round(t, 3), controller=ex.abort)
                    print(f"[open] **模擬中止** {ex_box['aborted']}", flush=True)
                    return False
                for i, j_ in zip(idx, range(6)):
                    tgt[i] = float(out['arm_target'][j_])
                u = out['base_twist']
            elif ctrl is not None:
                out = ctrl.step(t - t_ctrl0, q)
                for i, j_ in zip(idx, range(6)):
                    tgt[i] = float(out['arm_target'][j_])
                u = out['base_twist']
            else:
                u = np.zeros(3)
            apply()
            yw = float(q[2])
            robot.set_linear_velocity(np.array([u[0] * math.cos(yw) - u[1] * math.sin(yw), u[0] * math.sin(yw) + u[1] * math.cos(yw), 0.0], np.float32))   # 理想底盤介面
            robot.set_angular_velocity(np.array([0.0, 0.0, u[2]], np.float32))
            world.step(render=True)
            t += DT
            q, jp = q9()
            Rr, _, _ = rel_T()
            fc = finger_contacts()
            oc, ow = other_contacts()
            gate.step_seq(float(world.current_time), Rr, oc, max(fc[0][1], fc[1][1]))
            lin, ang = gate.last if gate.last is not None else (float('nan'), float('nan'))
            if Rr is None:
                Rr = np.full((4, 4), np.nan)
            if ctrl is not None and (fc[0][0] + fc[1][0] + fc[0][1] + fc[1][1]) > 0.05:
                ctrl.freeze_offset('first_finger_contact')
            F, _ = wrist_F()
            rows.append([t, PH.index(phase), door_q(), float(jp[f1]), float(jp[f2]), fc[0][0], fc[1][0], lin, ang, oc, *F, *Rr[:3, 3], *jp[idx], fc[0][1], fc[1][1], *q[:3], float(world.current_time), gate.still_s, *(gate.win if gate.win is not None else (float('nan'), float('nan'))), *raw[0]])
            if gate.latched is not None and not events:
                events.append(dict(gate.latched, who=ow))
                print(f'[bind] **禁止接觸鎖存** {events[-1]}', flush=True)
                if ctrl is not None:
                    ctrl._stop('exec_forbidden_contact')
                if ex_box['ex'] is not None:
                    ex_box['ex']._stop('forbidden_contact', **gate.latched)
        return True
    quat = np.array([math.cos(b1[2] / 2), 0, 0, math.sin(b1[2] / 2)], np.float32)
    for _ in range(5):
        robot.set_world_pose(np.array([b1[0], b1[1], 0.0], np.float32), quat)
        robot.set_joint_positions(q0.astype(np.float32))
        robot.set_joint_velocities(np.zeros(robot.num_dof, np.float32))
        apply()
        world.step(render=True)
    door0 = door_q()
    _, Fshape = wrist_F()
    t_ctrl0 = 0.0
    step('P', 100)
    A = np.array(rows)
    dist_mass = 0.4126
    F_all = np.asarray(robot.get_measured_joint_forces(), float)
    res = dict(readback=rb, wrist=dict(row=j6, joint_names=list(meta.joint_names), joint_indices={k_: int(v_) for k_, v_ in meta.joint_indices.items()}, shape=list(Fshape), raw_all_rows_P_end=F_all.tolist(),
                                       P_mean_norm_n=round(float(np.linalg.norm(A[:, 10:13].mean(axis=0))), 3), downstream_weight_n=round(dist_mass * 9.81, 3), status='未校準；只作合理性檢查，不稱把手拉力', frame='子連桿座標（API 說明）'))
    res['P'] = dict(door_deg_change=round(math.degrees(float(np.abs(A[:, 2] - door0).max())), 4), finger_contact_max_n=round(float(max(A[:, 5:7].max(), A[:, 22:24].max())), 4), other_contact_max_n=round(float(A[:, 9].max()), 4),
                    static_offset_rad=[round(float(v), 5) for v in (A[-1, 16:22] - q_grasp)], note='預抓位姿、無接觸靜置；偏置＝實測關節−設定點')
    if a.wiring:
        qm, jpm = q9()
        w0 = time.perf_counter()
        qg = PG.ik(qm[:3], qm[3:], T_at(T_far, 0.0), extra_seeds=True)
        wr = dict(meta_row=j6, fixedjoint_api=hasattr(UsdPhysics, 'FixedJoint'), finger_readback=[float(jpm[f1]), float(jpm[f2])])
        if qg is None:
            wr['plan'] = 'grasp_ik_fail'
        else:
            WB.set_model(phi0=0.0, finger_q=0.00173)
            T_HE = np.linalg.inv(T_far) @ K.fk(qg, 'link_tcp')
            b = qg[:3]
            T_ramp = 1.875 * PHI_GOAL * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM)
            st_w = float(world.current_time)

            def _plan_w():
                o_ = WB.DoorWBOpt(T_far, np.eye(4), np.r_[qg, 0.0], PHI_GOAL, N=int(math.ceil((T_ramp + 2.0) / 0.2)), dt=0.2, T_HE=T_HE, project_start=False)
                o_.init_corotate(T_ramp)
                i_ = o_.solve(max_iter=45)
                return o_, i_, o_.verify(), o_.dense_reference(4)
            rr, ee = run_responsive(world, _plan_w)
            if ee is not None:
                raise ee
            opt, info, v, (Xd, h, _) = rr
            wr_clock = float(world.current_time) - st_w
            ex = XO.DoorOpenExec(ident, Xd, h, T_far, 0.0, T_HE, PHI_GOAL, dict(s=np.array([tgt[i] for i in idx], float), pipe=[np.zeros(9)] * DA.D_STEPS, u_last=np.zeros(9), u_prev=np.zeros(9), bias=np.zeros(6)), 0.00173)
            Fw, _ = wrist_F()
            o1 = ex.step(0.0, qg, 0.0, float(np.linalg.norm(Fw)))      # 合成狀態一步：不套用命令、不步進物理
            wr.update(plan=dict(solve=info, feasible=v['feasible'], failed=v['failed'], t_phi_first_goal_s=v['t_phi_first_goal_s']), plan_wall_s=round(time.perf_counter() - w0, 1), plan_sim_clock_advance_s=wr_clock,
                      exec_one_step=dict(abort=o1['abort'], finite=bool(np.isfinite(o1['arm_target']).all() and np.isfinite(o1['base_twist']).all()), stop=ex.stop, solve_ms=ex.solve_ms[-1:] if ex.solve_ms else None),
                      wrist_F_norm_static=round(float(np.linalg.norm(Fw)), 3))
        res['wiring'] = wr
        res['result'] = 'wiring_only'
        print(f"[open] 接線核對 {json.dumps(clean(wr), ensure_ascii=False)[:1500]}", flush=True)
        ok_w = bool(isinstance(wr.get('plan'), dict) and wr['plan']['feasible'] and wr['exec_one_step']['finite'] and not wr['exec_one_step']['abort'] and wr['fixedjoint_api'] and wr.get('plan_sim_clock_advance_s') == 0.0)
        res['checks'], res['ok'] = dict(wiring=ok_w, P_no_contact=res['P']['finger_contact_max_n'] <= COND['forbidden_n'] and res['P']['other_contact_max_n'] <= COND['forbidden_n']), None
        res['ok'] = bool(all(res['checks'].values()))
        json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
        print(f"[open] 完成 ok={res['ok']} result=wiring_only", flush=True)
        return 0 if res['ok'] else 5
    # I：控制器插入（起點＝實測；設定點不重置）
    qm, _ = q9()
    ctrl_obj = DA.DoorApproachCtrl(ident, qm[:3], qm[3:], qm[:3], qm[3:], T_far, T_at, joint_nominal=True)
    ctrl_obj.s = np.array([tgt[i] for i in idx], float)
    ctrl, t_ctrl0, sent = ctrl_obj, t, False
    while t - t_ctrl0 < CTRL_LIMIT_S and ctrl.stop_ok is None:
        if ctrl.phase == 'LOOK' and not sent:
            ctrl.obs_intake((t - t_ctrl0) - DT, T_far)                 # 既有影格估計（非本趟重拍），標示於結果
            sent = True
        step('I', 1)
    rep = ctrl.report()
    I = np.array([r for r in rows if r[1] == 1])
    res['I_insert'] = dict(stop=rep['stop'], done=rep['done'], stop_effect=rep['stop_effect'], replan=None if rep['replan'] is None else {k_: v_ for k_, v_ in rep['replan'].items() if k_ != 'T_pre_world'}, tracking=rep['tracking']['TO_PRE'], d_hat=rep['d_hat'],
                           events=[e for e in rep['events'] if e['what'] in ('offset_frozen', 'replan', 'stop', 'arrived_look')], hold_dev=rep['hold_dev_max'], min_clear_stop_now=rep['min_clear_stop_now'], min_clear_candidate=rep['min_clear_candidate'],
                           finger_bar_max_n=[round(float(I[:, 5].max()), 3), round(float(I[:, 6].max()), 3)], door_deg_change=round(math.degrees(float(np.abs(I[:, 2] - door0).max())), 4), timing=rep['timing']['whole_tick_ms'], near_obs='既有影格估計（online_1 起點影格），不是本趟重拍',
                           base_drift_mm=round(float(np.linalg.norm(I[:, 24:26] - I[0, 24:26], axis=1).max()) * 1e3, 4))
    print(f"[bind] 插入 {json.dumps(clean({k_: res['I_insert'][k_] for k_ in ('stop', 'done', 'tracking', 'd_hat', 'finger_bar_max_n', 'door_deg_change', 'base_drift_mm')}), ensure_ascii=False)[:900]}", flush=True)
    proceed = bool(rep['done'] and rep['stop_effect'] and rep['stop_effect']['ok'] and gate.latched is None)
    if not proceed:
        res['result'] = 'insert_not_completed_or_forbidden_contact：不閉爪、不綁'
    else:
        for i in range(50):
            tgt[f1] = FINGER_OPEN * (1 - (i + 1) / 50)
            step('B', 1)
            if gate.latched is not None:
                break
        if gate.latched is None:
            step('B', 100)
        B = np.array([r for r in rows if r[1] == 2])
        qm, jpm = q9()
        Ttcp, Tg_des = K.fk(qm, 'link_tcp'), T_at(handle_T(), 0.0)
        c = dict(pos_m=float(np.linalg.norm(Ttcp[:3, 3] - Tg_des[:3, 3])), rot_deg=rot_deg(Ttcp, Tg_des), finger_q=[float(jpm[f1]), float(jpm[f2])], finger_bar_n=[float(B[-1, 5]), float(B[-1, 6])], door_pre_deg=math.degrees(float(np.abs(B[:, 2] - door0).max())))
        okb, okd = gate.decide(c['pos_m'], c['rot_deg'], c['finger_q'], c['finger_bar_n'], c['door_pre_deg'])
        res['C_bind_conditions'] = dict(values={k_: (round(v_, 5) if isinstance(v_, float) else [round(x, 5) for x in v_]) for k_, v_ in c.items()}, rel_still_s=round(gate.still_s, 3), last_step_motion=gate.last, window_diag=gate.win, n_invalid_samples=gate.n_invalid, B_tail_0p3s=dict(lin_max_mmps=round(float(np.nanmax(B[-30:, 7])) * 1e3, 4), ang_max_radps=round(float(np.nanmax(B[-30:, 8])), 5)), ok=okd, all_ok=okb, source='模擬判據（把手 prim 真值、逐步相對變換差分〔正規化四元數、atan2〕連續 0.3 s、接觸點分類）；TCP＝實測底盤＋關節 FK')
        print(f"[bind] 綁定條件 {res['C_bind_conditions']}", flush=True)
        if not okb:
            res['result'] = 'bind_conditions_not_met：不綁，結束'
        else:
            Rr0, Tg_b, Tl_b = rel_T()
            door_b = door_q()
            j = UsdPhysics.FixedJoint.Define(stage, JOINT)
            j.CreateBody0Rel().SetTargets([Sdf.Path(str(grip[0]))])
            j.CreateBody1Rel().SetTargets([Sdf.Path(leaf)])
            j.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            Rl = np.linalg.inv(Rr0)                                   # 夾爪殼座標下的門扇位姿：joint 的 local1 是 body1（門扇）座標裡的連接框架＝夾爪殼位姿（Rr0）
            qr = W0.R_to_wxyz(Rr0[:3, :3])
            j.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in Rr0[:3, 3]]))
            j.CreateLocalRot1Attr().Set(Gf.Quatf(float(qr[0]), float(qr[1]), float(qr[2]), float(qr[3])))
            j.CreateJointEnabledAttr().Set(True)
            j.CreateBreakForceAttr().Set(float('inf'))
            j.CreateBreakTorqueAttr().Set(float('inf'))
            step('D', 1)
            Rr1, Tg_a, _ = rel_T()
            res['D_bind'] = dict(t=round(t, 3), joint=JOINT, body0=str(grip[0]), body1=leaf, local_pos1=Rr0[:3, 3].tolist(), local_rot1_wxyz=list(map(float, qr)), break_force='inf（理想承載）',
                                 T_HE_actual=(np.linalg.inv(handle_T()) @ K.fk(q9()[0], 'link_tcp')).tolist(), door_deg_at_bind=round(math.degrees(door_b), 5),
                                 jump_gripper_mm=round(float(np.linalg.norm(Tg_a[:3, 3] - Tg_b[:3, 3])) * 1e3, 4), jump_door_deg=round(math.degrees(door_q() - door_b), 5),
                                 rel_err_after_mm=round(float(np.linalg.norm(Rr1[:3, 3] - Rr0[:3, 3])) * 1e3, 4), rel_err_after_deg=round(rot_deg(Rr1, Rr0), 5))
            print(f"[bind] 綁定 {json.dumps(clean({k_: v_ for k_, v_ in res['D_bind'].items() if k_ != 'T_HE_actual'}), ensure_ascii=False)}", flush=True)
            step('E', 50)
            Ev = np.array([r for r in rows if r[1] == 4])
            pe = np.linalg.norm(Ev[:, 13:16] - Rr0[:3, 3], axis=1)
            res['E_bound_hold'] = dict(rel_err_max_mm=round(float(pe.max()) * 1e3, 4), door_drift_deg=round(math.degrees(float(Ev[-1, 2] - door_b)), 5), wrist_F_max_norm=round(float(np.linalg.norm(Ev[:, 10:13], axis=1).max()), 3))
            # ---- 規劃（暫停；以綁定瞬間實測狀態）
            qm, jpm = q9()
            phi_b = door_q()
            fq = 0.5 * (float(jpm[f1]) + float(jpm[f2]))
            T_WH_b = WB.Rz4(-phi_b) @ T_far
            T_HE = np.linalg.inv(T_WH_b) @ K.fk(qm, 'link_tcp')
            WB.set_model(phi0=phi_b, finger_q=fq)
            b = qm[:3]
            T_ramp = 1.875 * (PHI_GOAL - phi_b) * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM)
            w0, st0 = time.perf_counter(), float(world.current_time)
            plan_ok, why_p = False, None
            def _plan_f():
                o_ = WB.DoorWBOpt(T_WH_b, np.eye(4), np.r_[qm, phi_b], PHI_GOAL, N=int(math.ceil((T_ramp + 2.0) / 0.2)), dt=0.2, T_HE=T_HE, project_start=False)
                o_.init_corotate(T_ramp)
                i_ = o_.solve(max_iter=45)
                return o_, i_, o_.verify(), o_.dense_reference(4)
            rr, ee = run_responsive(world, _plan_f)
            if ee is None:
                opt, info, v, (Xd, h, adj) = rr
                plan_ok = bool(v['feasible'])
                why_p = None if plan_ok else v['failed']
            else:
                info, v, why_p = None, None, repr(ee)
            res['plan'] = dict(ok=plan_ok, why=why_p, wall_s=round(time.perf_counter() - w0, 1), sim_clock_advance_s=float(world.current_time) - st0, phi_bind_deg=round(math.degrees(phi_b), 5), finger_q_model=fq, T_HE=T_HE.tolist(),
                               solve=info, verify=None if v is None else {k_: v[k_] for k_ in ('feasible', 'failed', 't_phi_95pct_s', 't_phi_first_goal_s', 't_ref_end_s', 'min_clear_m', 'sigma_min', 'sigma_start', 'setpoint_rate_needed_max', 'wheel_max', 'body_acc_max', 'closure_pos_mm', 'base_travel_m', 'base_yaw_change_deg')},
                               handle_source='online_1 起點影格估計（φ＝0），以鉸鏈讀值轉到 φ_bind', note='非即時：規劃期間不步進物理')
            print(f"[open] 規劃 {json.dumps(clean({k_: res['plan'][k_] for k_ in ('ok', 'why', 'wall_s', 'sim_clock_advance_s', 'phi_bind_deg', 'finger_q_model', 'verify')}), ensure_ascii=False)[:1500]}", flush=True)
            if not plan_ok or res['plan']['wall_s'] > PLAN_WALL_LIMIT_S or res['plan']['sim_clock_advance_s'] != 0.0:
                res['result'] = 'plan_failed：不動作'
            else:
                np.savez(os.path.join(a.out, 'plan.npz'), Xd=Xd, h=h, X=opt.X, T_WH_bind=T_WH_b, phi_bind=phi_b, T_HE=T_HE)
                ho = dict(s=ctrl.s.copy(), pipe=[u_.copy() for u_ in ctrl.pipe], u_last=ctrl.u_last.copy(), u_prev=ctrl.u_prev.copy(), bias=np.asarray(ctrl.cfg.arm_model.bias, float).copy())
                ex = XO.DoorOpenExec(ident, Xd, h, T_WH_b, phi_b, T_HE, PHI_GOAL, ho, fq)
                res['handover'] = dict(pipe_nonzero=int(sum(bool(np.any(u_)) for u_ in ho['pipe'])), setpoint_minus_measured=[round(float(x), 5) for x in (ho['s'] - qm[3:])], bias=[round(float(x), 6) for x in ho['bias']])
                ctrl = None
                ex_box.update(ex=ex, t0=t)
                t_o0 = t
                while t - t_o0 < OPEN_LIMIT_S and ex.stop_ok is None:
                    if not step('O', 1):
                        break
                if ex_box['aborted'] is None and ex.stop_ok is None:
                    ex._stop('open_time_limit')
                    step('O', 1)
                rep_o = ex.report()
                res['O_open'] = rep_o
                res['O_open']['force_note'] = '手腕傳遞力範數：列＝metadata joint6＋1、未校準；40 N＝模擬工程中止值，不是安全上限'
                res['O_open']['door_angle_note'] = '門角＝鉸鏈讀值（模擬量測；實機無此感測）'
                print(f"[open] 開門 {json.dumps(clean({k_: rep_o[k_] for k_ in ('stop', 'done', 'stop_effect', 'phi_end_deg', 't_end', 'closure_measured', 'setpoint_model_deviation', 'lead_deg_max', 'min_clear_actual', 'force_max_n', 'arm_sat_s')}), ensure_ascii=False)[:2200]}", flush=True)
                ex_box['ex'] = None
                if ex_box['aborted'] is not None:
                    res['result'] = 'aborted：模擬中止（不是受控停止；停止有效性未驗證；未解綁）'
                elif rep_o['done'] and rep_o['stop_effect'] and rep_o['stop_effect'].get('ok'):
                    # U：解綁並讀回
                    door_u = door_q()
                    UsdPhysics.Joint(stage.GetPrimAtPath(JOINT)).GetJointEnabledAttr().Set(False)
                    en = bool(UsdPhysics.Joint(stage.GetPrimAtPath(JOINT)).GetJointEnabledAttr().Get())
                    res['U_unbind'] = dict(t=round(t, 3), joint_enabled_readback=en, door_deg=round(math.degrees(door_u), 4))
                    step('U', 1)
                    # R：張爪＋觀察（兩指實際開度核淨空）
                    rmin = {k_: np.inf for k_ in XO.KEYS}
                    for i in range(50):
                        tgt[f1] = fq + (FINGER_OPEN - fq) * (i + 1) / 50
                        step('R', 1)
                        qm2, jp2 = q9()
                        WB.set_model(phi0=phi_b, finger_q=0.5 * (float(jp2[f1]) + float(jp2[f2])))
                        cl = WB.clear_door(np.r_[qm2, door_q()], T_WH_b)
                        for k_ in XO.KEYS:
                            rmin[k_] = min(rmin[k_], cl[k_])
                    for i in range(200):
                        step('R', 1)
                        if i % 5 == 0:
                            qm2, jp2 = q9()
                            WB.set_model(phi0=phi_b, finger_q=0.5 * (float(jp2[f1]) + float(jp2[f2])))
                            cl = WB.clear_door(np.r_[qm2, door_q()], T_WH_b)
                            for k_ in XO.KEYS:
                                rmin[k_] = min(rmin[k_], cl[k_])
                    Rv = np.array([r for r in rows if r[1] == 7])
                    dq_ = np.diff(Rv[:, 2]) / DT
                    res['R_release_observe'] = dict(finger_q_end_mm=[round(float(Rv[-1, 3]) * 1e3, 3), round(float(Rv[-1, 4]) * 1e3, 3)], finger_bar_end=[round(float(Rv[-1, 5]), 3), round(float(Rv[-1, 6]), 3)], door_deg_change=round(math.degrees(float(Rv[-1, 2] - door_u)), 4),
                                                    door_rate_max_dps=round(math.degrees(float(np.abs(dq_).max())), 4), door_rate_end_dps=round(math.degrees(float(np.abs(dq_[-10:]).mean())), 4), min_clear_actual_fingers={k_: round(float(v_), 4) for k_, v_ in rmin.items()},
                                                    note='淨空以兩指實際開度（逐步讀回）計算')
                    res['result'] = 'completed'
                else:
                    res['result'] = 'open_not_done：' + str(rep_o['stop'])
    R = np.array(rows)
    res['forbidden_contact'] = dict(latched=gate.latched, events=events, threshold_n=COND['forbidden_n'], scope='手臂六連桿／link_eef／夾爪殼／底盤 對 門扇＋牆／家具；手指對橫桿以外部位。只豁免兩指—橫桿')
    res['base_interface_note'] = '逐物理步以控制器輸出設定底盤速度＝理想底盤介面／支撐假設，不是正式執行層底盤保持的證據'
    res['scope'] = '理想抓持綁定下的名目功能驗證；非即時；異常處置＝模擬中止；不含退出；不接導航；不沿用舊開門控制'
    checks = dict(passive_hinge=True, P_no_contact=res['P']['finger_contact_max_n'] <= COND['forbidden_n'] and res['P']['other_contact_max_n'] <= COND['forbidden_n'], P_door_still=res['P']['door_deg_change'] <= 0.05,
                  insert_done=bool(res['I_insert']['done']), no_forbidden_contact=gate.latched is None)
    if 'D_bind' in res:
        checks.update(bind_conditions=True, bind_no_jump=res['D_bind']['jump_gripper_mm'] < 1.0 and abs(res['D_bind']['jump_door_deg']) < 0.05)
    else:
        checks.update(bind_conditions=False)
    if 'plan' in res:
        checks.update(plan_ok=bool(res['plan']['ok']), plan_clock_paused=bool(res['plan']['sim_clock_advance_s'] == 0.0))
    if 'O_open' in res:
        o_ = res['O_open']
        checks.update(not_aborted=ex_box['aborted'] is None, open_done=bool(o_['done']), stop_effective=bool(o_['stop_effect'] and o_['stop_effect'].get('ok')), phi_reached=bool(o_['phi_end_deg'] >= 84.5),
                      clear_during_open=bool(all(o_['min_clear_actual'][k_] >= WB.THR[k_] for k_ in XO.KEYS)))
    if res['result'] == 'completed':
        checks.update(unbound=not res['U_unbind']['joint_enabled_readback'], clear_during_finger_open=bool(all(res['R_release_observe']['min_clear_actual_fingers'][k_] >= WB.THR[k_] for k_ in XO.KEYS)))
    else:
        checks.update(completed=False)
    res['checks'], res['ok'] = checks, bool(all(checks.values()))
    np.savez_compressed(os.path.join(a.out, 'log.npz'), rows=R, cols=np.array(['t', 'phase', 'door_q', 'f1', 'f2', 'fbar1', 'fbar2', 'rel_lin', 'rel_ang', 'other_contact', 'Fx', 'Fy', 'Fz', 'rel_x', 'rel_y', 'rel_z', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'fnonbar1', 'fnonbar2', 'bx', 'by', 'byaw', 't_sim', 'still_s', 'win_lin_diag', 'win_ang_diag', 'gp_x', 'gp_y', 'gp_z', 'gq_w', 'gq_x', 'gq_y', 'gq_z', 'lp_x', 'lp_y', 'lp_z', 'lq_w', 'lq_x', 'lq_y', 'lq_z']), plan_note='開門段控制器紀錄另見 result.json O_open')
    json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[open] 完成 ok={res['ok']} checks={checks} result={res['result']}", flush=True)
    print(f"[open] 摘要 {json.dumps(clean({k: res.get(k) for k in ('P', 'E_bound_hold', 'U_unbind', 'R_release_observe', 'forbidden_contact')}), ensure_ascii=False)[:2500]}", flush=True)
    t_h = time.time()
    while sim_app.is_running() and time.time() - t_h < a.hold_s:
        world.render()
    return 0 if res['ok'] else 5


if __name__ == '__main__':
    rc = 99
    try:
        rc = main()
    except Exception:
        import traceback
        print('[bind] 例外', traceback.format_exc(), flush=True)
    sim_app.close()
    sys.exit(rc)
