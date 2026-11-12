#!/usr/bin/env python3
"""住宅場景門：**整合執行器**（Codex 20261113_013230：首版物理範圍到 X2）。單一 Isaac 程序、單一執行緒同步迴圈（非即時：控制計算期間模擬暫停）。
  A0 本趟視覺接近（同 isaac_m8_house_door_approach.py --mode approach --online-plan --joint-nominal 的鏈，不改）：
     起點靜置 → 本趟遠距取像 → 在線停位與重瞄規劃 → 到位停穩 → 近距新影格 → 偵測＋幾何估計 T_near → 重算 → 非接觸到預抓 → 停止有效
  A1 插入（door_insert_ctrl；**目標＝本趟 T_near**，不讀 m8house_da_online_1）→ 閉爪 → BindGate → 理想抓持綁定 →
     規劃（暫停；把手＝T_near 以角度增量轉到綁定門角：T_WH_b＝Rz[−(φ_b−φ_cap)]·T_near）→ 開門（door_open_exec_ctrl arm_mode='closure'）→ 停穩 → 解綁讀回 → 張爪＋觀察
  X  退出（door_exit_ctrl）：X1 沿當下接近軸反向退出 → X1b 保持關節構型底盤後退 → X2 收臂到本趟起始手臂姿態 → 進入 X3X4 即範圍終點停止（不跑運動學跟隨、不導航）
承接：每段由前一段控制器當下的設定點、管線、上一命令、偏置建立（不重置、不回填舊紀錄；逐值核對寫入結果）。
監看：A0 用接近執行器的非預期接觸（工具連桿淨力、底盤配對力）＋門角 0.05°＋手指模型；A1 起用開門執行器的禁止接觸（BindGate；兩指—橫桿以外）；
      退出段另加控制器自身幾何監看（門角帶、淨空、兩指規則、停止掃掠）。
--smoke-nobind（接線冒煙）：A0 → 插入 → 閉爪 → 綁定條件只評估**不綁** → 張爪 → X1–X2（門角約 0°）。不開門。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_door_full.py --out <dir> [--smoke-nobind]"""
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
ap.add_argument('--viewpoints', default=os.path.join(HERE, 'results', 'vision', 'm8_door_reaim_viewpoints_r1.json'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--settle-s', type=float, default=1.0)
ap.add_argument('--hold-s', type=float, default=3.0)
ap.add_argument('--smoke-nobind', action='store_true')
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation    # noqa: E402
from isaacsim.core.utils.types import ArticulationAction         # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics   # noqa: E402

import d1_handle_detect_v2 as D1                                 # noqa: E402
import door_approach_jnom_ctrl as DJ                             # noqa: E402
import door_insert_ctrl as DI                                    # noqa: E402
import door_open_exec_ctrl as XO                                 # noqa: E402
import door_exit_ctrl as XE                                      # noqa: E402
import door_bind_logic as BL                                     # noqa: E402
import door_online_plan as OP                                    # noqa: E402
import door_asset_v3 as DA3                                      # noqa: E402
import m8_door_detect as MD                                      # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402
from finger_collision import apply_split                         # noqa: E402
from isaac_common import bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402

WB = XO.W
PG, E, K = DJ.PG, DJ.R4.E, DJ.K
ROBOT, DOOR = '/World/omni_bot', '/World/door'
JOINT = '/World/door/ideal_grasp_attach'
ARM = [f'joint{i}' for i in range(1, 7)]
DT = 0.01
FINGER_OPEN = DJ.FINGER_Q
FINGER_TOL_M = 0.0005
CONTACT_N, DOOR_DEG = 0.05, 0.05
COND = dict(pos_m=0.005, rot_deg=3.0, rel_lin_mps=0.005, rel_ang_rps=0.02, rel_still_s=0.3, finger_q_m=(0.0005, 0.004), finger_contact_n=0.5, door_pre_deg=0.5, forbidden_n=0.05)
A0_LIMIT_S, CTRL_LIMIT_S, OPEN_LIMIT_S, EXIT_LIMIT_S = 40.0, 12.0, 40.0, 30.0
PHI_GOAL = math.radians(85.0)
PLAN_WALL_LIMIT_S, OPLAN_WALL_LIMIT_S = 600.0, 120.0
CAM = dict(res=(640, 480), hfov_deg=69.0, aperture=0.020, clip=(0.05, 10.0), hz=30)
SEED = 20261112


def run_responsive(world, fn):
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


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def T_of(pos, qwxyz):
    T = np.eye(4)
    T[:3, :3] = W0.wxyz_to_R(np.asarray(qwxyz, float))
    T[:3, 3] = np.asarray(pos, float)
    return T


def rot_deg(A, B):
    return math.degrees(math.acos(float(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1))))


def main():
    os.makedirs(a.out, exist_ok=True)
    reg = json.load(open(a.reg))
    vreg = json.load(open(a.viewpoints))
    base0 = np.array(vreg['standoff']['base0'], float)               # 場景初始條件（同 da_online_1）
    src = json.load(open(os.path.join(WS, json.load(open(os.path.join(HERE, 'runs', 'm8doorcap_1', 'wrist_door', 'meta.json')))['registration'])))
    q_obs = np.array({x['id']: x for x in src['viewpoints']}[vreg['source']['viewpoint']]['q_arm'], float)
    Rg, T_HG, a_H = PG.baseline_reference()
    T_at = lambda T, s: np.asarray(E.OT.object_target(T, T_HG, a_H, float(s))['T_WE'], float)   # noqa: E731
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
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
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(base0[0]), float(base0[1]), 0.0))
    xf.AddRotateZOp().Set(float(math.degrees(base0[2])))
    apply_split(stage, ROBOT)
    if bind_frictionless(stage, collision_prims(stage, ROBOT, lambda p: 'support_' in p) + collision_prims(stage, '/World/ground')) == 0:
        print('[full] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
        return 19
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(1300.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 20.0, 0.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(450.0)
    bodies, arts = physics_parts(stage, ROBOT)
    nm = lambda b: str(b).rstrip('/').split('/')[-1]              # noqa: E731
    tool_names = tuple(f'link{i}' for i in range(1, 7)) + ('link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')
    tool_bodies = [b for b in bodies if nm(b) in tool_names]
    grip = [b for b in bodies if nm(b) == 'uflite_gripper_link']
    fingers = sorted(b for b in bodies if nm(b) in ('uflite_finger1', 'uflite_finger2'))
    others = [b for b in bodies if nm(b) in tuple(f'link{i}' for i in range(1, 7)) + ('link_eef',)]
    base_b = [b for b in bodies if nm(b) == 'base_link']
    if len(grip) != 1 or len(fingers) != 2 or len(others) != 7 or len(base_b) != 1 or len(tool_bodies) != len(tool_names):
        print('[full] 剛體不符 ⇒ 拒跑', flush=True)
        return 16
    leaf = dauth['leaf_prim']
    base_filters = env_paths + [leaf]
    # A0 監看用（同接近執行器）
    v_tool = [RigidPrim(prim_paths_expr=str(b), name=f'vt_{nm(b)}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True) for b in tool_bodies]
    v_base0 = RigidPrim(prim_paths_expr=str(base_b[0]), name='v_base0', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=base_filters)
    # A1 起監看用（同開門執行器）
    v_f = [RigidPrim(prim_paths_expr=str(f), name=f'v_{nm(f)}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=[leaf]) for f in fingers]
    v_o = [RigidPrim(prim_paths_expr=str(b), name=f'v_{nm(b)}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=[leaf] + env_paths) for b in others + grip]
    v_base = RigidPrim(prim_paths_expr=str(base_b[0]), name='v_base', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=[leaf] + env_paths)
    v_grip = RigidPrim(prim_paths_expr=str(grip[0]), name='v_grip_pose')
    v_leaf = RigidPrim(prim_paths_expr=leaf, name='v_leaf_pose')
    world.reset()
    for v in v_tool + [v_base0] + v_f + v_o + [v_base, v_grip, v_leaf]:
        v.initialize()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    q0 = np.array(robot.get_joint_positions(), float)
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i, j in zip(idx, range(6)):
        q0[i] = float(q_obs[j])
        kp[i], kd[i] = a.kp, a.kd
    q0[f1], q0[f2] = FINGER_OPEN, FINGER_OPEN
    kp[f1], kd[f1] = a.finger_kp, a.finger_kd
    act = np.array(sorted(idx + [f1]), np.int32)
    robot.set_joint_positions(q0.astype(np.float32))
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    tgt = q0.copy()
    kps_rb, kds_rb = robot.get_articulation_controller().get_gains()
    gr = dict(api_kp=[float(np.asarray(kps_rb)[i]) for i in idx], api_kd=[float(np.asarray(kds_rb)[i]) for i in idx])
    if not all(abs(v - a.kp) <= 1e-3 * a.kp for v in gr['api_kp']):
        print(f'[full] 增益讀回不符 {gr} ⇒ 拒跑', flush=True)
        return 66
    joint_kp = float(a.kp)

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
    hprim = stage.GetPrimAtPath(f'{DOOR}/leaf/handle_bar')

    def truth_handle():                                            # 只供事後評估
        M = UsdGeom.XformCache().GetLocalToWorldTransform(hprim)
        return np.array([[M[i][j] for j in range(4)] for i in range(4)]).T
    hinge = DA3.readback(stage, DOOR)
    if abs(hinge['drive_stiffness']) > 1e-12 or abs(hinge['drive_damping']) > 1e-12 or not hinge['is_rigid_body']:
        print('[full] 鉸鏈不是被動 ⇒ 拒跑', flush=True)
        return 64
    bar_axis = 'XYZ'.index(str(hinge['handle_bar']['axis']))
    bar_r, bar_hl = float(hinge['handle_bar']['radius']), 0.5 * float(hinge['handle_bar']['length'])
    meta = robot._articulation_view._metadata
    j6 = int(meta.joint_indices['joint6']) + 1

    def wrist_F6():
        return np.asarray(robot.get_measured_joint_forces(), float)[j6][:6].copy()
    # 手腕相機
    eef = [b for b in bodies if nm(b) == 'link_eef'][0]
    T_mount = W0.urdf_chain_T(a.urdf, 'link_eef', W0.OPT)
    cam = Camera(prim_path=f'{eef}/wrist_cam', resolution=CAM['res'], frequency=CAM['hz'])
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
    cam.set_horizontal_aperture(CAM['aperture'])
    cam.set_focal_length(CAM['aperture'] / (2.0 * math.tan(math.radians(CAM['hfov_deg']) / 2.0)))
    cam.set_clipping_range(*CAM['clip'])
    cam.set_local_pose(translation=T_mount[:3, 3], orientation=W0.R_to_wxyz(T_mount[:3, :3]), camera_axes='ros')
    try:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([2.6, -2.6, 2.4]), target=np.array([0.6, -0.2, 0.8]))
    except Exception:                                              # noqa: BLE001
        pass
    res = dict(run=os.path.basename(a.out.rstrip('/')), mode='smoke_nobind' if a.smoke_nobind else 'full', gains_readback=gr,
               scope='理想抓持綁定；非即時（計算時模擬暫停）；範圍到 X2（不導航、不穿門）；門角＝鉸鏈讀值（模擬量測）；手腕力未校準',
               init=dict(base0=base0.tolist(), q_obs=q_obs.tolist(), note='起點底盤位姿與起始手臂姿態為場景初始條件（同 m8house_da_online_1）'))
    # 靜置
    quat0 = np.array([math.cos(base0[2] / 2), 0.0, 0.0, math.sin(base0[2] / 2)], np.float32)
    for k in range(int(round(a.settle_s / DT))):
        if k < 5:
            robot.set_world_pose(np.array([base0[0], base0[1], 0.0], np.float32), quat0)
        apply()
        base_cmd(np.zeros(3), 0.0)
        world.step(render=True)
    Kr = np.asarray(cam.get_intrinsics_matrix(), float)
    door_q0 = door_q()
    rows, t = [], 0.0
    PH = ['A0', 'I', 'B', 'D', 'E', 'O', 'U', 'R', 'X']

    # ================= A0：本趟視覺接近 =================
    hist_p, f_far, n_read = {}, None, 0
    for _ in range(120):
        t_s = float(world.current_time)
        qv, _ = q9()
        hist_p[int(round(t_s / DT))] = (t_s, qv.copy())
        fr = cam.get_current_frame()
        n_read += 1
        t_r, dep_ = fr.get('rendering_time'), fr.get('distance_to_image_plane')
        hp = None if t_r is None else hist_p.get(int(round(float(t_r) / DT)))
        if hp is not None and dep_ is not None and np.asarray(dep_).size and len(hist_p) > 3 and float(t_r) >= t_s - 3 * DT:
            dep_ = np.asarray(dep_, dtype=np.float32)
            f_far = dict(t_r=float(t_r), t_q=t_s, dep=dep_[:, :, 0] if dep_.ndim == 3 else dep_, T_cam=K.fk(hp[1], W0.OPT), q=hp[1])
            break
        apply()
        base_cmd(np.zeros(3), 0.0)
        world.step(render=True)

    def est_from(dep, T_cam, t_obs, t_q, tag, seed):
        det, ms = D1.detect(dep, Kr, T_cam, np.random.default_rng(seed))
        cls = MD.classify(det)
        info = dict(tag=tag, cls=cls, reject=det['reject'], detect_ms=round(float(ms), 1), t_obs=t_obs, t_query=t_q)
        if cls != 'center_valid':
            return None, info
        est = E.estimate(dict(path=tag, L1=det['L1'], L2=det['L2'], reject=None, reject_L2=None, depth=dep, K=Kr, T_cam=T_cam, t_obs=t_obs), 'N-obs', Rg, T_HG, a_H, query_t=t_q)
        info['est_ok'], info['est_why'] = bool(est['ok']), est.get('why')
        return (np.asarray(est['T_WH'], float) if est['ok'] else None), info
    online = dict(frame_reads=n_read, got_frame=f_far is not None)
    if f_far is None:
        res['result'] = 'A0_no_far_frame：不運動'
        json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
        return 7
    T_far, online['far_estimate'] = est_from(f_far['dep'], f_far['T_cam'], f_far['t_r'], f_far['t_q'], 'far_plan', SEED)
    qn, _ = q9()
    w_p, st_a = time.perf_counter(), float(world.current_time)
    plan_ = OP.plan(T_far, qn[:3], qn[3:], T_mount, Kr, T_at)
    online['plan_wall_s'], online['sim_clock_advance_during_plan_s'] = round(time.perf_counter() - w_p, 2), float(world.current_time) - st_a
    online['plan'] = {k_: v_ for k_, v_ in plan_.items() if k_ not in ('b1', 'q_look', 'q_pre_far')}
    if not plan_['ok'] or online['plan_wall_s'] > OPLAN_WALL_LIMIT_S or online['sim_clock_advance_during_plan_s'] != 0.0:
        res['A0'] = dict(online=online)
        res['result'] = 'A0_online_plan_failed：不運動'
        json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
        print('[full] 在線規劃不成立 ⇒ 不運動', flush=True)
        return 7
    b1, q_look = plan_['b1'].copy(), plan_['q_look'].copy()
    online.update(b1=b1.tolist(), q_look=q_look.tolist(), T_far=T_far.tolist(), far_est_vs_truth_mm_eval=round(float(np.linalg.norm(T_far[:3, 3] - truth_handle()[:3, 3])) * 1e3, 2))
    print(f"[full] A0 在線規劃 b1={np.round(b1, 3).tolist()} 規劃牆鐘 {online['plan_wall_s']} s", flush=True)
    qm, _ = q9()
    s_now = np.array([tgt[i] for i in idx], float)
    ctrl = DJ.DoorApproachCtrl(ident, qm[:3], qm[3:], b1, q_look, T_far, T_at, joint_nominal=True)
    ctrl.s = s_now.copy()
    hist, last_rf, near = {}, None, None
    latch, cmax = None, dict(n=0.0, who=None)
    T0 = float(world.current_time)

    def contacts_a0():
        mx, who = 0.0, None
        for v, b in zip(v_tool, tool_bodies):
            n_ = float(np.linalg.norm(np.asarray(v.get_net_contact_forces(dt=DT), float).reshape(-1)[:3]))
            if n_ > mx:
                mx, who = n_, nm(b)
        Mb = np.asarray(v_base0.get_contact_force_matrix(dt=DT), float).reshape(-1, len(base_filters), 3)
        nb = np.linalg.norm(Mb[0], axis=1)
        if float(nb.max()) > mx:
            mx, who = float(nb.max()), 'base_link|' + base_filters[int(nb.argmax())]
        return mx, who
    rngn = [0]
    k = 0
    while k * DT < A0_LIMIT_S:
        tc = k * DT
        t_sim = float(world.current_time)
        q, jp = q9()
        pos_r, q_r = cam.get_world_pose(camera_axes='ros')
        hist[int(round(t_sim / DT))] = dict(t_sim=t_sim, q=q.copy(), door=door_q())
        if len(hist) > 300:
            for kk in sorted(hist)[:len(hist) - 300]:
                hist.pop(kk, None)
        cn, cwho = contacts_a0()
        cmax = dict(n=cn, who=cwho) if cn > cmax['n'] else cmax
        if cn > CONTACT_N and latch is None:
            latch = dict(t=round(tc, 3), why='unexpected_contact', force_n=round(cn, 4), who=cwho)
            ctrl._stop('exec_unexpected_contact')
        if abs(math.degrees(door_q() - door_q0)) > DOOR_DEG and latch is None:
            latch = dict(t=round(tc, 3), why='door_moved')
            ctrl._stop('exec_door_moved')
        fr = cam.get_current_frame()
        rfd = fr.get('rendering_frame')
        rf = (rfd.get('referenceTimeNumerator'), rfd.get('referenceTimeDenominator')) if isinstance(rfd, dict) else rfd
        if rf is not None and rf != last_rf and rf != (0, 0) and fr.get('distance_to_image_plane') is not None and fr.get('rendering_time') is not None:
            last_rf = rf
            t_r = float(fr['rendering_time'])
            hp = hist.get(int(round(t_r / DT)))
            if hp is not None and abs(hp['t_sim'] - t_r) <= 0.25 * DT and ctrl.look is not None and (t_r - T0) >= ctrl.look['t_arrive'] + DT - 1e-9 and ctrl.phase == 'LOOK' and ctrl.stop is None and near is None:
                dep = np.asarray(fr['distance_to_image_plane'], dtype=np.float32)
                dep = dep[:, :, 0] if dep.ndim == 3 else dep
                rngn[0] += 1
                Tn, info = est_from(dep, K.fk(hp['q'], W0.OPT), t_r - T0, tc, 'near', SEED + rngn[0])
                if Tn is not None:
                    near = dict(info, T_WH=Tn.tolist(), t_render_sim=t_r, door_at_capture_rad=hp['door'], frame='world（門局部座標＝世界）', age_at_intake_s=round(tc - (t_r - T0), 4),
                                vs_truth_mm_eval=round(float(np.linalg.norm(Tn[:3, 3] - truth_handle()[:3, 3])) * 1e3, 2))
                    ctrl.obs_intake(t_r - T0, Tn)
                    print(f"[full] A0 近距估計送入 t_obs={near['t_obs']:.3f} 門角 {math.degrees(hp['door']):.4f}°", flush=True)
        out = ctrl.step(tc, q)
        for i, j_ in zip(idx, range(6)):
            tgt[i] = float(out['arm_target'][j_])
        tgt[f1] = FINGER_OPEN
        apply()
        base_cmd(out['base_twist'], float(q[2]))
        world.step(render=True)
        t += DT
        Fw = wrist_F6()
        rows.append([t, 0, door_q(), float(jp[f1]), float(jp[f2]), *q, *Fw])
        k += 1
        if ctrl.stop_ok is not None and tc - ctrl.stop_ok['t'] >= 0.5:
            break
    rep_a0 = ctrl.report()
    res['A0'] = dict(online=online, near=near, stop=rep_a0['stop'], done=rep_a0['done'], stop_effect=rep_a0['stop_effect'], target_source=rep_a0['target_source'], latch=latch, contact_max=cmax, replan=None if rep_a0['replan'] is None else {k_: v_ for k_, v_ in rep_a0['replan'].items() if k_ != 'T_pre_world'})
    print(f"[full] A0 完成 done={rep_a0['done']} stop={rep_a0['stop']} latch={latch}", flush=True)
    a0_ok = bool(rep_a0['done'] and rep_a0['stop_effect'] and rep_a0['stop_effect'].get('ok') and latch is None and near is not None and rep_a0['target_source'] == 'near')
    if not a0_ok:
        res['result'] = 'A0_not_completed：不插入'
        return finish(res, rows, None)
    T_near = np.array(near['T_WH'])
    phi_cap = float(near['door_at_capture_rad'])
    # 承接紀錄：A0 → 插入
    ho_a0 = dict(s=ctrl.s.copy(), pipe=[u_.copy() for u_ in ctrl.pipe], u_last=ctrl.u_last.copy(), u_prev=ctrl.u_prev.copy(), bias=np.asarray(ctrl.cfg.arm_model.bias, float).copy())

    # ================= A1 起：開門執行器同一套監看 =================
    raw = [[float('nan')] * 14]

    def rel_T():
        pg, qg = v_grip.get_world_poses()
        pl, ql = v_leaf.get_world_poses()
        raw[0] = [*np.asarray(pg[0], float), *np.asarray(qg[0], float), *np.asarray(pl[0], float), *np.asarray(ql[0], float)]
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
    gate = BL.BindGate(COND, DT)
    events = []
    box = dict(kind=None, obj=None, t0=0.0, aborted=None)
    olog = []

    def step(phase, n):
        nonlocal t
        for _ in range(n):
            q, jp = q9()
            obj = box['obj']
            if box['kind'] == 'open':
                F6 = wrist_F6()
                fn_ = float(np.linalg.norm(F6[:3]))
                phm = door_q()
                out = obj.step(t - box['t0'], q, phm, fn_)
                olog.append(np.r_[t - box['t0'], q, phm, fn_, F6, float(out['abort'])])
                if out['abort']:
                    box['aborted'] = dict(t=round(t, 3), controller=obj.abort)
                    return False
            elif box['kind'] == 'exit':
                out = obj.step(t - box['t0'], q, door_q())
                if out.get('abort'):
                    box['aborted'] = dict(t=round(t, 3), controller=obj.abort)
                    return False
            elif box['kind'] == 'insert':
                out = obj.step(t - box['t0'], q)
            else:
                out = None
            if out is not None:
                for i, j_ in zip(idx, range(6)):
                    tgt[i] = float(out['arm_target'][j_])
                u = out['base_twist']
            else:
                u = np.zeros(3)
            apply()
            base_cmd(u, float(q[2]))
            world.step(render=True)
            t += DT
            q, jp = q9()
            Rr, _, _ = rel_T()
            fc = finger_contacts()
            oc, ow = other_contacts()
            gate.step_seq(float(world.current_time), Rr, oc, max(fc[0][1], fc[1][1]))
            if box['kind'] == 'insert' and (fc[0][0] + fc[1][0] + fc[0][1] + fc[1][1]) > 0.05:
                obj.freeze_offset('first_finger_contact')
            rows.append([t, PH.index(phase), door_q(), float(jp[f1]), float(jp[f2]), *q, *wrist_F6()])
            if gate.latched is not None and not events:
                events.append(dict(gate.latched, who=ow))
                print(f'[full] **禁止接觸鎖存** {events[-1]}', flush=True)
                if box['kind'] == 'insert':
                    BL.forward_forbidden(gate.latched, ow, ctrl=obj)
                elif box['kind'] == 'open':
                    BL.forward_forbidden(gate.latched, ow, ex=obj)
                elif box['kind'] == 'exit':
                    obj._stop('exec_forbidden_contact', contact_info=dict(gate.latched, who=ow))
        return True
    # I：插入（目標＝本趟 T_near；設定點承接 A0；管線／上一命令承接 A0 控制器當下值）
    qm, _ = q9()
    ins = DI.DoorApproachCtrl(ident, qm[:3], qm[3:], qm[:3], qm[3:], T_near, T_at, joint_nominal=True)
    ins.s = ho_a0['s'].copy()
    ins.pipe = [u_.copy() for u_ in ho_a0['pipe']]
    ins.u_last, ins.u_prev = ho_a0['u_last'].copy(), ho_a0['u_prev'].copy()
    # 偏置：插入控制器自帶區間偏置觀測（bias＝bias0＋α·d̂，自當下狀態重新估計），外部覆寫會在下一次觀測更新被蓋掉 ⇒ 不承接 A0 的偏置值，只記錄兩者
    res['handover_A0_to_I'] = dict(setpoint_equal=bool(np.array_equal(ins.s, ho_a0['s'])), pipe_equal=bool(all(np.array_equal(x_, y_) for x_, y_ in zip(ins.pipe, ho_a0['pipe']))), u_last_equal=bool(np.array_equal(ins.u_last, ho_a0['u_last'])),
                                   setpoint_minus_measured=[round(float(x_), 5) for x_ in (ho_a0['s'] - qm[3:])], bias_A0=ho_a0['bias'].tolist(), bias_insert_init=np.asarray(ins.cfg.arm_model.bias, float).tolist(),
                                   bias_note='插入控制器以自帶觀測重新估計偏置（不承接 A0 值）')
    box.update(kind='insert', obj=ins, t0=t)
    sent = False
    while t - box['t0'] < CTRL_LIMIT_S and ins.stop_ok is None:
        if ins.phase == 'LOOK' and not sent:
            ins.obs_intake((t - box['t0']) - DT, T_near)                 # 本趟近距估計（A0 已送過接近控制器；插入控制器以同一估計為目標）
            sent = True
        step('I', 1)
    rep_i = ins.report()
    res['I_insert'] = dict(stop=rep_i['stop'], done=rep_i['done'], stop_effect=rep_i['stop_effect'], tracking=rep_i['tracking']['TO_PRE'], d_hat=rep_i['d_hat'], target='本趟 T_near')
    print(f"[full] 插入 done={rep_i['done']} stop={rep_i['stop']}", flush=True)
    if not (rep_i['done'] and rep_i['stop_effect'] and rep_i['stop_effect']['ok'] and gate.latched is None):
        res['result'] = 'insert_not_completed：不閉爪'
        return finish(res, rows, None, events=events, gate=gate)
    for i in range(50):
        tgt[f1] = FINGER_OPEN * (1 - (i + 1) / 50)
        step('B', 1)
        if gate.latched is not None:
            break
    if gate.latched is None:
        step('B', 100)
    Bv = np.array([r for r in rows if r[1] == 2])
    qm, jpm = q9()
    Ttcp, Tg_des = K.fk(qm, 'link_tcp'), T_at(truth_handle(), 0.0)
    fcB = finger_contacts()
    c = dict(pos_m=float(np.linalg.norm(Ttcp[:3, 3] - Tg_des[:3, 3])), rot_deg=rot_deg(Ttcp, Tg_des), finger_q=[float(jpm[f1]), float(jpm[f2])], finger_bar_n=[float(fcB[0][0]), float(fcB[1][0])],
             door_pre_deg=math.degrees(float(np.abs(Bv[:, 2] - door_q0).max())))
    okb, okd = gate.decide(c['pos_m'], c['rot_deg'], c['finger_q'], c['finger_bar_n'], c['door_pre_deg'])
    res['C_bind_conditions'] = dict(values=c, rel_still_s=round(gate.still_s, 3), ok=okd, all_ok=okb, source='模擬判據（把手 prim 真值、逐步相對變換差分）')
    print(f'[full] 綁定條件 all_ok={okb} {okd}', flush=True)
    fq = 0.5 * (float(jpm[f1]) + float(jpm[f2]))
    if a.smoke_nobind:
        res['note_smoke'] = '接線冒煙：綁定條件只評估、不綁、不開門；直接張爪後退出'
        exit_ho_src = ins
        phi_b, T_WH_b = door_q(), WB.Rz4(-(door_q() - phi_cap)) @ T_near
    else:
        if not okb:
            res['result'] = 'bind_conditions_not_met：不綁'
            return finish(res, rows, None, events=events, gate=gate)
        Rr0, Tg_b, Tl_b = rel_T()
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
        res['D_bind'] = dict(t=round(t, 3), jump_gripper_mm=round(float(np.linalg.norm(Tg_a[:3, 3] - Tg_b[:3, 3])) * 1e3, 4))
        step('E', 50)
        qm, jpm = q9()
        phi_b = door_q()
        fq = 0.5 * (float(jpm[f1]) + float(jpm[f2]))
        T_WH_b = WB.Rz4(-(phi_b - phi_cap)) @ T_near                     # 近距估計（擷取時門角 φ_cap）以角度增量轉到綁定門角
        T_HE = np.linalg.inv(T_WH_b) @ K.fk(qm, 'link_tcp')
        WB.set_model(phi0=phi_b, finger_q=fq)
        b = qm[:3]
        T_ramp = 1.875 * (PHI_GOAL - phi_b) * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM)
        w0, st0 = time.perf_counter(), float(world.current_time)

        def _plan_f():
            o_ = WB.DoorWBOpt(T_WH_b, np.eye(4), np.r_[qm, phi_b], PHI_GOAL, N=int(math.ceil((T_ramp + 2.0) / 0.2)), dt=0.2, T_HE=T_HE, project_start=False)
            o_.init_corotate(T_ramp)
            i_ = o_.solve(max_iter=45)
            return o_, i_, o_.verify(), o_.dense_reference(4)
        rr, ee = run_responsive(world, _plan_f)
        plan_ok = ee is None and bool(rr[2]['feasible'])
        res['plan'] = dict(ok=plan_ok, why=None if plan_ok else (repr(ee) if ee is not None else rr[2]['failed']), wall_s=round(time.perf_counter() - w0, 1), sim_clock_advance_s=float(world.current_time) - st0,
                           phi_bind_deg=round(math.degrees(phi_b), 5), phi_capture_deg=round(math.degrees(phi_cap), 5), handle_source='本趟近距估計 T_near（擷取時門角 φ_cap）以角度增量 φ_b−φ_cap 轉到綁定門角',
                           verify=None if ee is not None else {k_: rr[2][k_] for k_ in ('feasible', 'failed', 't_phi_first_goal_s', 'min_clear_m', 'sigma_min')})
        print(f"[full] 規劃 {json.dumps(clean(res['plan']), ensure_ascii=False)[:800]}", flush=True)
        if not plan_ok or res['plan']['wall_s'] > PLAN_WALL_LIMIT_S or res['plan']['sim_clock_advance_s'] != 0.0:
            res['result'] = 'plan_failed：不動作'
            return finish(res, rows, None, events=events, gate=gate)
        opt, info, v, (Xd, h, _) = rr
        np.savez(os.path.join(a.out, 'plan.npz'), Xd=Xd, h=h, X=opt.X, T_WH_bind=T_WH_b, phi_bind=phi_b, T_HE=T_HE, T_near=T_near, phi_capture=phi_cap)
        ho = dict(s=ins.s.copy(), pipe=[u_.copy() for u_ in ins.pipe], u_last=ins.u_last.copy(), u_prev=ins.u_prev.copy(), bias=np.asarray(ins.cfg.arm_model.bias, float).copy())
        ex = XO.DoorOpenExec(ident, Xd, h, T_WH_b, phi_b, T_HE, PHI_GOAL, ho, fq, arm_mode='closure', joint_kp=joint_kp)
        res['exec_config'] = ex.config_readback()
        box.update(kind='open', obj=ex, t0=t)
        t_o0 = t
        while t - t_o0 < OPEN_LIMIT_S and ex.stop_ok is None:
            if not step('O', 1):
                break
        if box['aborted'] is None and ex.stop_ok is None:
            ex._stop('open_time_limit')
            step('O', 1)
        rep_o = ex.report()
        res['O_open'] = {k_: rep_o[k_] for k_ in ('arm_mode', 'closure_anchor', 'closure_stats', 'stop', 'done', 'stop_effect', 'phi_end_deg', 't_end', 'closure_measured', 'min_clear_actual', 'force_max_n', 'timing')}
        np.savez_compressed(os.path.join(a.out, 'open_steps.npz'), meas=np.array(olog), ctrl=np.array(ex.step_log), ctrl_cols=np.array(XO.DoorOpenExec.STEP_LOG_COLS))
        print(f"[full] 開門 done={rep_o['done']} stop={rep_o['stop']} 門 {rep_o['phi_end_deg']}° 力最大 {rep_o['force_max_n']}", flush=True)
        box.update(kind=None, obj=None)
        if box['aborted'] is not None:
            res['result'] = 'aborted_in_open：模擬中止（停止有效性未驗證；未解綁）'
            return finish(res, rows, box['aborted'], events=events, gate=gate)
        if not (rep_o['done'] and rep_o['stop_effect'] and rep_o['stop_effect'].get('ok')):
            res['result'] = 'open_not_done：' + str(rep_o['stop'])
            return finish(res, rows, None, events=events, gate=gate)
        UsdPhysics.Joint(stage.GetPrimAtPath(JOINT)).GetJointEnabledAttr().Set(False)
        en = bool(UsdPhysics.Joint(stage.GetPrimAtPath(JOINT)).GetJointEnabledAttr().Get())
        res['U_unbind'] = dict(t=round(t, 3), joint_enabled_readback=en, door_deg=round(math.degrees(door_q()), 4))
        step('U', 1)
        exit_ho_src = ex
    # 張爪（兩指實際開度；觀察 0.5 s）
    box.update(kind=None, obj=None)
    for i in range(50):
        tgt[f1] = fq + (FINGER_OPEN - fq) * (i + 1) / 50
        step('R', 1)
    step('R', 50)
    qm, jpm = q9()
    Rd = np.array([r for r in rows if r[1] == 7])
    omega0 = float((Rd[-1, 2] - Rd[-11, 2]) / 0.1)
    fq_open = 0.5 * (float(jpm[f1]) + float(jpm[f2]))
    # ================= X：退出（承接當下設定點、管線、上一命令、偏置）=================
    src_ = exit_ho_src
    ho_x = dict(s=src_.s.copy(), pipe=[u_.copy() for u_ in src_.pipe], u_last=src_.u_last.copy(), u_prev=src_.u_prev.copy(), bias=np.asarray(src_.cfg.arm_model.bias, float).copy())
    xc = XE.DoorExitCtrl(ident, T_WH_b, phi_b, ho_x['s'], ho_x['pipe'], ho_x['u_last'], ho_x['u_prev'], ho_x['bias'], fq_open, door_q(), omega0, np.zeros((1, 2)), q_obs, joint_kp=joint_kp)
    res['handover_to_X'] = dict(source='door_open_exec_ctrl' if src_ is not ins else 'door_insert_ctrl（冒煙）', setpoint_equal=bool(np.array_equal(xc.s, ho_x['s'])), pipe_equal=bool(all(np.array_equal(x_, y_) for x_, y_ in zip(xc.pipe, ho_x['pipe']))),
                                u_last_equal=bool(np.array_equal(xc.u_last, ho_x['u_last'])), setpoint_vs_actuator_target=float(np.abs(ho_x['s'] - np.array([tgt[i] for i in idx], float)).max()),
                                phi0_deg=round(math.degrees(door_q()), 4), omega0_dps=round(math.degrees(omega0), 4), finger_q_open=fq_open, q_stow='本趟起始手臂姿態（同 da_online_1 起點）')
    box.update(kind='exit', obj=xc, t0=t)
    t_x0 = t
    while t - t_x0 < EXIT_LIMIT_S:
        if not step('X', 1):
            break
        if xc.phase == 'X3X4' and xc.stop is None:
            xc._stop('scope_end_after_X2')
        if xc.stop is not None and xc.stop_ok is not None:
            break
    rep_x = xc.report()
    res['X_exit'] = rep_x
    print(f"[full] 退出 stop={rep_x['stop']} abort={rep_x['abort']} 相位 {rep_x['phases']} 淨空 {rep_x['min_clear']}", flush=True)
    if box['aborted'] is not None:
        res['result'] = 'aborted_in_exit：模擬中止（停止有效性未驗證）'
    elif rep_x['stop'] and rep_x['stop']['why'] == 'scope_end_after_X2' and rep_x['stop_effect'] and rep_x['stop_effect'].get('ok'):
        res['result'] = 'completed_to_X2'
    else:
        res['result'] = 'exit_not_completed：' + str(rep_x['stop'])
    return finish(res, rows, box['aborted'], events=events, gate=gate)


def finish(res, rows, aborted, events=None, gate=None):
    R = np.array(rows) if rows else np.zeros((1, 20))
    res['forbidden_contact'] = None if gate is None else dict(latched=gate.latched, events=events)
    ck = dict(A0=bool(res.get('A0', {}).get('done')), insert=bool(res.get('I_insert', {}).get('done')), no_forbidden=bool(gate is None or gate.latched is None))
    if 'O_open' in res:
        ck.update(open_done=bool(res['O_open']['done']), unbound=bool(res.get('U_unbind') and not res['U_unbind']['joint_enabled_readback']))
    if 'X_exit' in res:
        x = res['X_exit']
        ck.update(exit_to_X2=bool(x['stop'] and x['stop']['why'] == 'scope_end_after_X2'), exit_stop_effective=bool(x['stop_effect'] and x['stop_effect'].get('ok')), exit_no_abort=x['abort'] is None,
                  exit_clear=bool(all(x['min_clear'][k_] >= XE.W.THR[k_] for k_ in XE.KEYS)), handover_X=bool(res['handover_to_X']['setpoint_equal'] and res['handover_to_X']['pipe_equal'] and res['handover_to_X']['u_last_equal']))
    if 'handover_A0_to_I' in res:
        h = res['handover_A0_to_I']
        ck['handover_A0_I'] = bool(h['setpoint_equal'] and h['pipe_equal'] and h['u_last_equal'])
    res['checks'], res['aborted'] = ck, aborted
    res['ok'] = bool(all(ck.values()) and res.get('result') in ('completed_to_X2',))
    np.savez_compressed(os.path.join(a.out, 'log.npz'), rows=R, cols=np.array(['t', 'phase', 'door_q', 'f1', 'f2', 'bx', 'by', 'byaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']),
                        phases=np.array(['A0', 'I', 'B', 'D', 'E', 'O', 'U', 'R', 'X']))
    json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[full] 完成 ok={res['ok']} result={res.get('result')} checks={ck}", flush=True)
    return 0 if res['ok'] else 5


if __name__ == '__main__':
    rc = 99
    try:
        rc = main()
    except Exception:
        import traceback
        print('[full] 例外', traceback.format_exc(), flush=True)
    sim_app.close()
    sys.exit(rc)
