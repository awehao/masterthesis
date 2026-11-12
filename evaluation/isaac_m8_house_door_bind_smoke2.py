#!/usr/bin/env python3
"""住宅場景門把手：**局部插入＋理想抓持綁定介面**冒煙 r2（Codex 20261112 綁定冒煙 1 裁定）。不是純靜態測試。
範圍：不導航、不做遠距視覺與在線停位、不執行開門軌跡。把手估計＝既有影格估計（m8house_da_online_1 起點影格）。
流程：
  P 擺位在**預抓位姿**（沿接近軸退 30 mm；張爪；只在此擺位一次）並靜置 1.0 s。
  I 插入：door_insert_ctrl（jnom 局部鏈的版本化複本；偏置觀測器在無接觸段估靜態偏置；重算終點＝抓取位姿；兩指—橫桿為指定接觸）。
    控制器起點＝實測底盤／手臂、設定點不重置。首次手指接觸即凍結偏置觀測（不把接觸負載學成偏置）。
  B 控制器到位並停穩後，閉爪斜坡 0.5 s＋靜置 1.0 s。
  C 綁定條件（door_bind_logic.BindGate；模擬判據）：禁止接觸從未出現；TCP（**實測底盤＋關節** FK）對把手抓取位姿 ≤5 mm／3°；
    夾爪殼相對門扇的**相對變換差分** ≤5 mm/s、≤0.02 rad/s 連續 0.3 s；兩指開度各自在 [0.5, 4.0] mm；兩指各自對**橫桿**（接觸點分類）>0.5 N；綁前門角變化 ≤0.5°。
  D 綁定 FixedJoint（uflite_gripper_link↔leaf；當下相對位姿；不設斷裂力＝理想承載）→ E 靜置 2 s → F 解除並觀察 2 s。
禁止接觸（任何時刻出現即鎖存，之後不插入、不閉爪、不綁定）：手臂六連桿／link_eef／夾爪殼／底盤 對 門扇或牆／家具；手指對門扇上**橫桿以外**的部位（門板、立柱）。只豁免兩指—橫桿。
底盤：每物理步以控制器輸出設定底盤速度（保持相位為零）——**理想底盤介面／支撐假設**，不是正式執行層底盤保持的證據。
手腕讀值只稱「傳遞力」：列＝articulation metadata 的 joint_indices['joint6']＋1（安裝版 API 說明）；存名稱、索引與全部原始列；P 段無接觸靜置對下游重量只作合理性檢查，未校準。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_door_bind_smoke2.py --out <dir>"""
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

    def rel_T():
        pg, qg = v_grip.get_world_poses()
        pl, ql = v_leaf.get_world_poses()
        return np.linalg.inv(T_of(pl[0], ql[0])) @ T_of(pg[0], qg[0]), T_of(pg[0], qg[0]), T_of(pl[0], ql[0])

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
    PH = ['P', 'I', 'B', 'D', 'E', 'F']

    def q9():
        p, qt = robot.get_world_pose()
        w, x, y, z = (float(v) for v in qt)
        jp = np.asarray(robot.get_joint_positions(), float)
        return np.r_[float(p[0]), float(p[1]), math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)), jp[idx]], jp

    def step(phase, n):
        nonlocal t
        for _ in range(n):
            q, jp = q9()
            if ctrl is not None:
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
            lin, ang = (0.0, 0.0) if prev_rel[0] is None else BL.rel_motion(prev_rel[0], Rr, DT)
            prev_rel[0] = Rr
            fc = finger_contacts()
            oc, ow = other_contacts()
            gate.step(t, lin, ang, oc, max(fc[0][1], fc[1][1]))
            if ctrl is not None and (fc[0][0] + fc[1][0] + fc[0][1] + fc[1][1]) > 0.05:
                ctrl.freeze_offset('first_finger_contact')
            F, _ = wrist_F()
            rows.append([t, PH.index(phase), door_q(), float(jp[f1]), float(jp[f2]), fc[0][0], fc[1][0], lin, ang, oc, *F, *Rr[:3, 3], *jp[idx], fc[0][1], fc[1][1], *q[:3]])
            if gate.latched is not None and not events:
                events.append(dict(gate.latched, who=ow))
                print(f'[bind] **禁止接觸鎖存** {events[-1]}', flush=True)
                if ctrl is not None:
                    ctrl._stop('exec_forbidden_contact')
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
        res['C_bind_conditions'] = dict(values={k_: (round(v_, 5) if isinstance(v_, float) else [round(x, 5) for x in v_]) for k_, v_ in c.items()}, rel_still_s=round(gate.n_still * DT, 2), ok=okd, all_ok=okb, source='模擬判據（把手 prim 真值、相對變換差分、接觸點分類）；TCP＝實測底盤＋關節 FK')
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
            step('E', 200)
            Ev = np.array([r for r in rows if r[1] == 4])
            pe = np.linalg.norm(Ev[:, 13:16] - Rr0[:3, 3], axis=1)
            res['E_bound_hold'] = dict(rel_err_max_mm=round(float(pe.max()) * 1e3, 4), rel_err_end_mm=round(float(pe[-1]) * 1e3, 4), rel_lin_max_mmps=round(float(Ev[:, 7].max()) * 1e3, 3), door_drift_deg=round(math.degrees(float(Ev[-1, 2] - door_b)), 5),
                                       door_deg_range=[round(math.degrees(float(Ev[:, 2].min())), 4), round(math.degrees(float(Ev[:, 2].max())), 4)], wrist_F_mean=[round(float(v), 3) for v in Ev[:, 10:13].mean(axis=0)],
                                       wrist_F_max_norm=round(float(np.linalg.norm(Ev[:, 10:13], axis=1).max()), 3), arm_joint_drift_max_rad=round(float(np.abs(Ev[:, 16:22] - Ev[0, 16:22]).max()), 6), finger_bar_end=[round(float(Ev[-1, 5]), 3), round(float(Ev[-1, 6]), 3)])
            UsdPhysics.Joint(stage.GetPrimAtPath(JOINT)).GetJointEnabledAttr().Set(False)
            t_rel, door_rel = t, door_q()
            step('F', 200)
            Fv = np.array([r for r in rows if r[1] == 5])
            dq = np.diff(Fv[:, 2]) / DT
            res['F_released'] = dict(t=round(t_rel, 3), joint_enabled_after=bool(UsdPhysics.Joint(stage.GetPrimAtPath(JOINT)).GetJointEnabledAttr().Get()), door_deg_change=round(math.degrees(float(Fv[-1, 2] - door_rel)), 5),
                                     door_rate_max_dps=round(math.degrees(float(np.abs(dq).max())), 5), door_rate_end_dps=round(math.degrees(float(np.abs(dq[-10:]).mean())), 5),
                                     rel_err_end_mm=round(float(np.linalg.norm(Fv[-1, 13:16] - Rr0[:3, 3])) * 1e3, 4), finger_bar_end=[round(float(Fv[-1, 5]), 3), round(float(Fv[-1, 6]), 3)])
            res['result'] = 'completed'
    R = np.array(rows)
    res['forbidden_contact'] = dict(latched=gate.latched, events=events, threshold_n=COND['forbidden_n'], scope='手臂六連桿／link_eef／夾爪殼／底盤 對 門扇＋牆／家具；手指對橫桿以外部位。只豁免兩指—橫桿')
    res['base_interface_note'] = '逐物理步以控制器輸出設定底盤速度（保持相位為零）＝理想底盤介面／支撐假設，不是正式執行層底盤保持的證據'
    checks = dict(passive_hinge=True, P_no_contact=res['P']['finger_contact_max_n'] <= COND['forbidden_n'] and res['P']['other_contact_max_n'] <= COND['forbidden_n'], P_door_still=res['P']['door_deg_change'] <= 0.05,
                  insert_done=bool(res['I_insert']['done']), no_forbidden_contact=gate.latched is None)
    if res['result'] == 'completed':
        checks.update(bind_conditions=True, bind_no_jump=res['D_bind']['jump_gripper_mm'] < 1.0 and abs(res['D_bind']['jump_door_deg']) < 0.05, bound_rel_err_small=res['E_bound_hold']['rel_err_max_mm'] < 1.0, released=not res['F_released']['joint_enabled_after'])
    else:
        checks.update(bind_conditions=False)
    res['checks'], res['ok'] = checks, bool(all(checks.values()))
    np.savez_compressed(os.path.join(a.out, 'log.npz'), rows=R, cols=np.array(['t', 'phase', 'door_q', 'f1', 'f2', 'fbar1', 'fbar2', 'rel_lin', 'rel_ang', 'other_contact', 'Fx', 'Fy', 'Fz', 'rel_x', 'rel_y', 'rel_z', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'fnonbar1', 'fnonbar2', 'bx', 'by', 'byaw']))
    json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[bind] 完成 ok={res['ok']} checks={checks} result={res['result']}", flush=True)
    print(f"[bind] 摘要 {json.dumps(clean({k: res.get(k) for k in ('P', 'E_bound_hold', 'F_released', 'forbidden_contact')}), ensure_ascii=False)[:2500]}", flush=True)
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
