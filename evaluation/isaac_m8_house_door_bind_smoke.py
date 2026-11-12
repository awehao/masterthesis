#!/usr/bin/env python3
"""住宅場景門把手：**理想抓持綁定**的靜態介面冒煙（Codex 20261112 r2 裁定授權一趟）。
範圍：不導航、不執行開門軌跡、不驗追蹤或重規劃。機器人直接擺在抓取構型（底盤＝在線停位、手臂＝抓取 IK；把手位姿＝online_1 起點影格估計）。
流程（模擬時間）：
  A 張爪靜置 1.0 s（未綁）——核手指不碰橫桿、門不動。
  B 閉爪斜坡 0.5 s＋靜置 1.0 s（未綁）——讀回各指實際開度、手指對門扇的接觸力、門角變化。
  C 綁定條件（事前登錄；模擬判據）：TCP 對把手抓取位姿（把手 prim 真值）位置 ≤5 mm、姿態 ≤3°；夾爪殼對門扇相對線速度 ≤5 mm/s、角速度 ≤0.02 rad/s 連續 0.3 s；
    兩指開度都在 [0.5, 4.0] mm（橫桿直徑 26 mm 擋住、到不了空夾終點 0）且各自對門扇接觸力 >0.5 N；綁定前門角變化 ≤0.5°。任一不過 ⇒ 不綁，記錄後結束。
  D 綁定：UsdPhysics.FixedJoint，body0＝uflite_gripper_link、body1＝門扇 leaf；兩端框架取**當下**實測相對位姿（不瞬移）。不設斷裂力（理想承載）。
  E 綁定靜置 2.0 s——連接誤差（夾爪殼相對門扇的位姿對綁定瞬間的偏差）、綁定前後跳動、門角、手臂關節、手腕傳遞力。
  F 解除（jointEnabled＝False）並觀察 2.0 s——門角與角速度、相對位姿、非預期接觸。
鉸鏈保持被動（讀回驅動 0、質量、角阻尼、限位）；本程式不發門角命令、不寫門或夾爪位姿（擺位只在 A 之前）。
手腕讀值只稱「傳遞力」：get_measured_joint_forces 的列號用 dof 索引＋1（9/13 抽屜模擬以已知外力校正的規則），本趟再以 A 段靜置對 joint6 以下重量自我核對。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_door_bind_smoke.py --out <dir>"""
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

import door_approach_jnom_ctrl as DA                             # noqa: E402
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
COND = dict(pos_m=0.005, rot_deg=3.0, rel_lin_mps=0.005, rel_ang_rps=0.02, rel_still_s=0.3, finger_q_m=(0.0005, 0.004), finger_contact_n=0.5, door_pre_deg=0.5)
LINK_TO_ENV_N = 0.05


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
    qg9 = PG.ik(b1, q_seed, T_at(T_far, 0.0), extra_seeds=True)
    if qg9 is None:
        print('[bind] 抓取 IK 無解 ⇒ 拒跑', flush=True)
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
              damping_note='PhysX 剛體角阻尼（angular_damping），不是鉸鏈黏滯係數', base=b1.tolist(), q_grasp=q_grasp.tolist(), handle_source='online_1 起點影格估計', cond=COND)
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

    def finger_contacts():
        out = []
        for v in v_f:
            M = np.asarray(v.get_contact_force_matrix(dt=DT), float).reshape(-1, 3)
            out.append(float(np.linalg.norm(M.sum(axis=0))))
        return out

    def other_contacts():
        mx, who = 0.0, None
        for v, b in zip(v_o + [v_base], others + grip + base_b):
            M = np.asarray(v.get_contact_force_matrix(dt=DT), float).reshape(-1, 3)
            n_ = np.linalg.norm(M, axis=1)
            if len(n_) and float(n_.max()) > mx:
                mx, who = float(n_.max()), nm(b)
        return mx, who

    j6 = names.index('joint6') + 1                                  # 列號規則：dof 索引＋1（9/13 抽屜模擬已知外力校正）

    def wrist_F():
        F = np.asarray(robot.get_measured_joint_forces(), float)
        return F[j6][:3].copy(), F.shape

    rows, phase_t, events = [], {}, []
    t = 0.0
    door0 = None

    def step(phase, n):
        nonlocal t
        for _ in range(n):
            apply()
            robot.set_linear_velocity(np.zeros(3, np.float32))
            robot.set_angular_velocity(np.zeros(3, np.float32))
            world.step(render=True)
            t += DT
            jp = np.asarray(robot.get_joint_positions(), float)
            Rr, Tg, Tl = rel_T()
            lv, av = rel_vel()
            fc = finger_contacts()
            oc, ow = other_contacts()
            F, _ = wrist_F()
            rows.append([t, ['A', 'B', 'D', 'E', 'F'].index(phase), door_q(), float(jp[f1]), float(jp[f2]), fc[0], fc[1], lv, av, oc, *F, *Rr[:3, 3], *jp[idx]])
            if ow is not None and oc > LINK_TO_ENV_N:
                events.append(dict(t=round(t, 3), phase=phase, unexpected_contact_n=round(oc, 4), who=ow))
    # 擺位（只在這裡）：底盤與手臂到抓取構型，手指張開
    quat = np.array([math.cos(b1[2] / 2), 0, 0, math.sin(b1[2] / 2)], np.float32)
    for _ in range(5):
        robot.set_world_pose(np.array([b1[0], b1[1], 0.0], np.float32), quat)
        robot.set_joint_positions(q0.astype(np.float32))
        robot.set_joint_velocities(np.zeros(robot.num_dof, np.float32))
        apply()
        world.step(render=True)
    door0 = door_q()
    Fg0, Fshape = wrist_F()
    step('A', 100)
    A = np.array(rows)
    fA = A[:, 10:13]
    dist_mass = 0.4126
    res = dict(readback=rb, wrist_row=j6, wrist_shape=list(Fshape), wrist_static_check=dict(mean_norm_n=round(float(np.linalg.norm(fA.mean(axis=0))), 3), downstream_weight_n=round(dist_mass * 9.81, 3), note='A 段無接觸靜置的傳遞力應約等於 joint6 以下重量（9/13 曾量到約 1.1 N 常數偏置）'))
    res['A'] = dict(door_deg_change=round(math.degrees(float(np.abs(A[:, 2] - door0).max())), 4), finger_contact_max_n=round(float(A[:, 5:7].max()), 4), finger_q=[round(float(A[-1, 3]) * 1e3, 3), round(float(A[-1, 4]) * 1e3, 3)])
    # B：閉爪
    for i in range(50):
        tgt[f1] = FINGER_OPEN * (1 - (i + 1) / 50)
        step('B', 1)
    step('B', 100)
    B = np.array([r for r in rows if r[1] == 1])
    Th = handle_T()
    Tg_des = T_at(Th, 0.0)
    _, Tg_now, _ = rel_T()
    Ttcp = K.fk(np.r_[b1, np.asarray(robot.get_joint_positions(), float)[idx]], 'link_tcp')
    still = B[-int(COND['rel_still_s'] / DT):]
    c = dict(pos_m=float(np.linalg.norm(Ttcp[:3, 3] - Tg_des[:3, 3])), rot_deg=rot_deg(Ttcp, Tg_des), rel_lin_max=float(still[:, 7].max()), rel_ang_max=float(still[:, 8].max()),
             finger_q=[float(B[-1, 3]), float(B[-1, 4])], finger_contact=[float(B[-1, 5]), float(B[-1, 6])], door_pre_deg=math.degrees(float(np.abs(B[:, 2] - door0).max())))
    ok = dict(pos=c['pos_m'] <= COND['pos_m'], rot=c['rot_deg'] <= COND['rot_deg'], rel_still=c['rel_lin_max'] <= COND['rel_lin_mps'] and c['rel_ang_max'] <= COND['rel_ang_rps'],
              finger_q=all(COND['finger_q_m'][0] <= q <= COND['finger_q_m'][1] for q in c['finger_q']), finger_contact=all(f > COND['finger_contact_n'] for f in c['finger_contact']), door_pre=c['door_pre_deg'] <= COND['door_pre_deg'])
    res['C_bind_conditions'] = dict(values={k: (round(v, 5) if isinstance(v, float) else [round(x, 5) for x in v]) for k, v in c.items()}, ok=ok, all_ok=bool(all(ok.values())), source='模擬判據（把手 prim 真值、剛體速度、接觸力）；TCP 由關節讀值 FK')
    print(f"[bind] 綁定條件 {res['C_bind_conditions']}", flush=True)
    if not res['C_bind_conditions']['all_ok']:
        res['result'] = 'bind_conditions_not_met：不綁，結束'
    else:
        # D：綁定（兩端框架＝當下實測相對位姿）
        Rr0, Tg_b, Tl_b = rel_T()
        door_b = door_q()
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
        step('D', 1)
        Rr1, Tg_a, _ = rel_T()
        res['D_bind'] = dict(t=round(t, 3), joint=JOINT, body0=str(grip[0]), body1=leaf, local_pos1=Rr0[:3, 3].tolist(), local_rot1_wxyz=list(map(float, qr)), break_force='inf（理想承載）',
                             jump_gripper_mm=round(float(np.linalg.norm(Tg_a[:3, 3] - Tg_b[:3, 3])) * 1e3, 4), jump_door_deg=round(math.degrees(door_q() - door_b), 5),
                             rel_err_after_mm=round(float(np.linalg.norm(Rr1[:3, 3] - Rr0[:3, 3])) * 1e3, 4), rel_err_after_deg=round(rot_deg(Rr1, Rr0), 5))
        print(f"[bind] 綁定 {res['D_bind']}", flush=True)
        step('E', 200)
        Ev = np.array([r for r in rows if r[1] == 3])
        pe = np.linalg.norm(Ev[:, 13:16] - Rr0[:3, 3], axis=1)
        res['E_bound_hold'] = dict(rel_err_max_mm=round(float(pe.max()) * 1e3, 4), rel_err_end_mm=round(float(pe[-1]) * 1e3, 4), door_deg_range=[round(math.degrees(float(Ev[:, 2].min())), 4), round(math.degrees(float(Ev[:, 2].max())), 4)],
                                   door_drift_deg=round(math.degrees(float(Ev[-1, 2] - door_b)), 5), wrist_F_mean=[round(float(v), 3) for v in Ev[:, 10:13].mean(axis=0)], wrist_F_max_norm=round(float(np.linalg.norm(Ev[:, 10:13], axis=1).max()), 3),
                                   arm_joint_drift_max_rad=round(float(np.abs(Ev[:, 16:22] - Ev[0, 16:22]).max()), 6), finger_contact_end=[round(float(Ev[-1, 5]), 3), round(float(Ev[-1, 6]), 3)])
        # F：解除
        UsdPhysics.Joint(stage.GetPrimAtPath(JOINT)).GetJointEnabledAttr().Set(False)
        t_rel, door_rel = t, door_q()
        step('F', 200)
        Fv = np.array([r for r in rows if r[1] == 4])
        dq = np.diff(Fv[:, 2]) / DT
        res['F_released'] = dict(t=round(t_rel, 3), joint_enabled_after=bool(UsdPhysics.Joint(stage.GetPrimAtPath(JOINT)).GetJointEnabledAttr().Get()), door_deg_change=round(math.degrees(float(Fv[-1, 2] - door_rel)), 5),
                                 door_rate_max_dps=round(math.degrees(float(np.abs(dq).max())), 5), door_rate_end_dps=round(math.degrees(float(np.abs(dq[-10:]).mean())), 5),
                                 rel_err_end_mm=round(float(np.linalg.norm(Fv[-1, 13:16] - Rr0[:3, 3])) * 1e3, 4), finger_contact_end=[round(float(Fv[-1, 5]), 3), round(float(Fv[-1, 6]), 3)])
        res['result'] = 'completed'
    R = np.array(rows)
    res['unexpected_contacts'] = dict(n_events=len(events), first=events[:5], threshold_n=LINK_TO_ENV_N, scope='手臂六連桿、link_eef、夾爪殼、底盤 對 門扇＋牆／家具（手指對門扇不在此列，屬指定接觸）')
    res['finger_model_note'] = '張開手指幾何模型 8.9 mm 只用於擺位前的淨空；閉爪後實際開度見 C 段 finger_q（各指）'
    checks = dict(passive_hinge=True, A_no_finger_contact=res['A']['finger_contact_max_n'] <= COND['finger_contact_n'], A_door_still=res['A']['door_deg_change'] <= 0.05)
    if res['result'] == 'completed':
        checks.update(bind_conditions=True, bind_no_jump=res['D_bind']['jump_gripper_mm'] < 1.0 and abs(res['D_bind']['jump_door_deg']) < 0.05, bound_rel_err_small=res['E_bound_hold']['rel_err_max_mm'] < 1.0,
                      released=not res['F_released']['joint_enabled_after'], no_unexpected_contact=len(events) == 0)
    else:
        checks.update(bind_conditions=False)
    res['checks'], res['ok'] = checks, bool(all(checks.values()))
    np.savez_compressed(os.path.join(a.out, 'log.npz'), rows=R, cols=np.array(['t', 'phase', 'door_q', 'f1', 'f2', 'fc1', 'fc2', 'rel_lin', 'rel_ang', 'other_contact', 'Fx', 'Fy', 'Fz', 'rel_x', 'rel_y', 'rel_z', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6']))
    json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[bind] 完成 ok={res['ok']} checks={checks} result={res['result']}", flush=True)
    print(f"[bind] 摘要 {json.dumps(clean({k: res.get(k) for k in ('A', 'wrist_static_check', 'E_bound_hold', 'F_released', 'unexpected_contacts')}), ensure_ascii=False)[:2500]}", flush=True)
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
