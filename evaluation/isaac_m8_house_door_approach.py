#!/usr/bin/env python3
"""住宅場景 m8_house_r1：門把手**非接觸**接近的 Isaac 執行器（Codex 20261112_120637 核可接線與冒煙；正式趟另行申請）。
控制器＝door_approach_ctrl.DoorApproachCtrl（r3 候選凍結，不改）。不插入、不閉爪、不開門、不放行人。
**非即時閉迴路測試**：單一執行緒同步迴圈——每個物理步「讀狀態 →（相機／估計）→ ctrl.step → 套用命令 → world.step 一步」。
  控制計算期間不呼叫 world.step ⇒ 物理世界與模擬時鐘不前進；每步核對模擬時鐘只前進一個 physics_dt（不符計數並拒絕成功判定）。
  8 步在途命令管線在控制器內，每個物理步推進一格；等待不清空、不跳步、不補送。
  另設牆鐘上限（整趟 WALL_LIMIT_S、單步 STEP_WALL_LIMIT_S），超過 ⇒ 鎖存失敗並走受控停止。
相機：手腕相機掛 link_eef（同靜態擷取的外參、內參設定）。影格時間＝渲染器回報的 rendering_time（換成控制時間 t＝模擬時間−T0）；
  相機位姿＝該渲染時刻的關節／底盤讀值 FK（姿態歷史逐步保存；找不到同一步 ⇒ 拒收）。不把舊影格改蓋新時間戳。
  到位（控制器 look.t_arrive）之後才渲染的影格才送偵測（d1_handle_detect_v2.detect，凍結版）→ dl2 幾何估計 → ctrl.obs_intake(t_obs, T_WH)。
遠距估計：登錄檔 results/vision/m8_door_reaim_viewpoints_r1.json 的 source.T_WH_est（既有影格的估計）＝**初始化條件**，不是本趟在線視覺。
  本趟起點另拍一格做同一偵測估計，只記錄比較，不進控制。
真值（把手 prim 世界位姿、門角）只在 truth／評估欄，控制器與估計都不讀。
非預期接觸：手臂各連桿、link_eef、夾爪殼、兩指的淨接觸力，底盤對牆／家具／門扇的配對力，門角變化——任一超門檻 ⇒ 鎖存失敗＋受控停止。
模式：
  --mode approach：完整鏈（起點→到位停穩→新影格→近距重算→非接觸終點→停止）。
  --mode smoke   ：機器人直接擺在登錄停位與重瞄姿態（不做接近運動）。控制器以零長度 TO_LOOK 起步 → SETTLE → LOOK；
                   主控制器**不收量測** ⇒ near_obs_timeout 走真停止流程（量 Isaac 底盤零命令漂移與停止有效性）。
                   影格三條路徑在控制器複本上核：到位前影格（拒收）、到位後新影格（偵測→估計→重算，不下命令）、無有效估計（故意送相機看不到把手的影格）。
                   停止後把一個 50 g 小方塊移到前臂上方落下 ⇒ 非預期接觸必須鎖存。另做管線延遲與等待不偷跑核對（每拍故意睡 SMOKE_SLEEP_S）。
    ~/venvs/isaacsim-6.0.1/bin/python -u evaluation/isaac_m8_house_door_approach.py --mode smoke --out <dir>"""
import argparse
import copy
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
ap.add_argument('--mode', choices=('smoke', 'approach'), required=True)
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json'))
ap.add_argument('--viewpoints', default=os.path.join(HERE, 'results', 'vision', 'm8_door_reaim_viewpoints_r1.json'))
ap.add_argument('--viewpoint-id', default='reaim_d0.35_az+0_el0_lam1.0', help='r5 通過的底盤到位視點中總時間最短者（test_door_approach.load 同一規則；起跑前核對）')
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--settle-s', type=float, default=1.0)
ap.add_argument('--sim-limit-s', type=float, default=40.0)
ap.add_argument('--hold-s', type=float, default=3.0)
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation    # noqa: E402
from isaacsim.core.utils.types import ArticulationAction         # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics     # noqa: E402

import d1_handle_detect_v2 as D1                                 # noqa: E402
import door_approach_ctrl as DA                                  # noqa: E402
import door_asset_v3 as DA3                                      # noqa: E402
import m8_door_detect as MD                                      # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402
from finger_collision import apply_split                         # noqa: E402
from isaac_common import bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402

R4 = DA.R4
PG, E, K, IDX = DA.PG, R4.E, DA.K, DA.IDX
ROBOT, DOOR = '/World/omni_bot', '/World/door'
ARM = [f'joint{i}' for i in range(1, 7)]
FINGER_OPEN = DA.FINGER_Q                                       # 與控制器淨空模型同一張開量
FINGER_TOL_M, FINGER_POSE_TOL_M = 0.0005, 0.0005                # 兩指關節讀回／連桿位姿對模型的容許差；超出 ⇒ 拒跑或鎖存
HARD_WALL_EXTRA_S = 60.0                                        # 牆鐘逾時後仍讓受控停止走完的額外上限；再超過才中止（停止有效性記為未驗證）
DT = 0.01
WALL_LIMIT_S, STEP_WALL_LIMIT_S = 900.0, 20.0                    # 兩者都是迴圈內事後檢查；同步呼叫本身卡死時不會觸發（不是硬性看門狗）
CONTACT_N, DOOR_DEG = 0.05, 0.05                                 # 非預期接觸門檻：淨力 N；門角變化 °
SMOKE_SLEEP_S = 0.15                                             # 冒煙：每控制拍故意多等（核對等待時物理不偷跑）
CAM = dict(res=(640, 480), hfov_deg=69.0, aperture=0.020, clip=(0.05, 10.0), hz=30)
SEED = 20261112


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def T_of(pos, qwxyz):
    T = np.eye(4)
    T[:3, :3] = W0.wxyz_to_R(np.asarray(qwxyz, float))
    T[:3, 3] = np.asarray(pos, float)
    return T


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


def main():
    os.makedirs(a.out, exist_ok=True)
    reg = json.load(open(a.reg))
    vreg = json.load(open(a.viewpoints))
    vp = {v['id']: v for v in vreg['viewpoints']}[a.viewpoint_id]
    assert vp['status'] == 'ok' and vp['lam'] == 1.0, vp
    r5 = json.load(open(os.path.join(HERE, 'runs', 'm8doorcap_reaim_1', 'wrist_door', 'door_reaim_arrived_r5.json')))
    cand = sorted([r for r in r5['rows'] if r.get('ok') and r['lam'] == 1.0], key=lambda r: (r['T_total'], r['id']))
    if cand[0]['id'] != a.viewpoint_id:
        print(f"[da] 視點 {a.viewpoint_id} 不是登錄規則選出的 {cand[0]['id']} ⇒ 拒跑", flush=True)
        return 2
    base0, b1 = np.array(vreg['standoff']['base0'], float), np.array(vp['base'], float)
    src = json.load(open(os.path.join(WS, json.load(open(os.path.join(HERE, 'runs', 'm8doorcap_1', 'wrist_door', 'meta.json')))['registration'])))
    q_obs = np.array({x['id']: x for x in src['viewpoints']}[vreg['source']['viewpoint']]['q_arm'], float)
    q_look = np.array(vp['q_arm'], float)
    T_far = np.array(vreg['source']['T_WH_est'], float)
    Rg, T_HG, a_H = PG.baseline_reference()

    def T_at(T_WH, s):
        return np.asarray(E.OT.object_target(T_WH, T_HG, a_H, float(s))['T_WE'], float)
    smoke = a.mode == 'smoke'
    start_b, start_q = (b1, q_look) if smoke else (base0, q_obs)

    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    scn = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)]
    if len(scn) != 1:
        print(f'[da] 物理場景數 {len(scn)} ≠ 1 ⇒ 拒跑', flush=True)
        return 67
    sapi = PhysxSchema.PhysxSceneAPI(scn[0]) if scn[0].HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(scn[0])
    sapi.CreateEnableExternalForcesEveryIterationAttr().Set(True)     # 與手臂設定點模型辨識時相同（bottle_s0.yaml physics）
    env_paths = []
    for o in reg['walls'] + reg['furniture']:
        FixedCuboid(prim_path=f"/World/{o['name']}", name=o['name'], position=np.array(o['c']), scale=np.array(o['s']), color=np.array(o['rgb']))
        env_paths.append(f"/World/{o['name']}")
    dauth = DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root=DOOR)
    ob = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
    for o in ob['static_unknown']:                                    # 未知靜物照場景登錄放（控制器幾何模型不含它們；起跑前核距離）
        y2 = math.radians(o['yaw_deg']) / 2
        FixedCuboid(prim_path=f"/World/unknown/{o['name']}", name=o['name'], position=np.array([o['x'], o['y'], o['size'][2] / 2]), scale=np.array(o['size']),
                    orientation=np.array([math.cos(y2), 0.0, 0.0, math.sin(y2)]), color=np.array([0.72, 0.52, 0.28]))
        env_paths.append(f"/World/unknown/{o['name']}")
        dmin = min(float(np.hypot(o['x'] - p[0], o['y'] - p[1])) for p in (base0, b1)) - 0.5 * float(np.hypot(o['size'][0], o['size'][1]))
        if dmin < 1.0:
            print(f"[da] 未知靜物 {o['name']} 離門前作業路徑 {dmin:.2f} m < 1.0 m；本包不含 LiDAR 避障 ⇒ 拒跑", flush=True)
            return 3
    probe = None
    if smoke:
        probe = DynamicCuboid(prim_path='/World/contact_probe', name='contact_probe', position=np.array([2.0, -3.0, 0.03]), scale=np.array([0.05, 0.05, 0.05]), mass=0.05, color=np.array([0.9, 0.2, 0.2]))
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(start_b[0]), float(start_b[1]), 0.0))
    xf.AddRotateZOp().Set(float(math.degrees(start_b[2])))
    apply_split(stage, ROBOT)
    sph = collision_prims(stage, ROBOT, lambda p: 'support_' in p)
    if bind_frictionless(stage, sph + collision_prims(stage, '/World/ground')) == 0:
        print('[da] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
        return 19
    k_ = UsdLux.DistantLight.Define(stage, '/World/key')
    k_.CreateIntensityAttr(1300.0)
    UsdGeom.Xformable(k_).AddRotateXYZOp().Set(Gf.Vec3f(-25.0, 20.0, 0.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(450.0)
    bodies, arts = physics_parts(stage, ROBOT)
    name_of = lambda b: str(b).rstrip('/').split('/')[-1]            # noqa: E731
    tool_names = tuple(f'link{i}' for i in range(1, 7)) + ('link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')
    tool_bodies = [b for b in bodies if name_of(b) in tool_names]
    b_base = [b for b in bodies if name_of(b) == 'base_link']
    if not arts or len(tool_bodies) != len(tool_names) or len(b_base) != 1:
        print(f'[da] 剛體不符：{[name_of(b) for b in tool_bodies]}／base {b_base}／arts {arts} ⇒ 拒跑', flush=True)
        return 16
    v_tool = [RigidPrim(prim_paths_expr=str(b), name=f'v_{name_of(b)}', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True) for b in tool_bodies]
    base_filters = env_paths + [dauth['leaf_prim']]
    v_base = RigidPrim(prim_paths_expr=str(b_base[0]), name='v_base', track_contact_forces=True, max_contact_count=64, prepare_contact_sensors=True, contact_filter_prim_paths_expr=base_filters)
    leaf_v = RigidPrim(prim_paths_expr=dauth['leaf_prim'], name='leaf_state')
    world.reset()
    for v in v_tool + [v_base, leaf_v]:
        v.initialize()
    robot = SingleArticulation(prim_path=arts[0], name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    if any(j not in names for j in ARM + ['finger_joint1', 'finger_joint2']):
        print(f'[da] 缺關節 {names} ⇒ 拒跑', flush=True)
        return 16
    idx = [names.index(j) for j in ARM]
    f1, f2 = names.index('finger_joint1'), names.index('finger_joint2')
    q0 = np.array(robot.get_joint_positions(), float)
    kp, kd = np.zeros(robot.num_dof, np.float32), np.zeros(robot.num_dof, np.float32)
    for i, j in zip(idx, range(6)):
        q0[i] = float(start_q[j])
        kp[i], kd[i] = a.kp, a.kd
    q0[f1], q0[f2] = FINGER_OPEN, FINGER_OPEN
    kp[f1], kd[f1] = a.finger_kp, a.finger_kd
    act = np.array(sorted(idx + [f1]), np.int32)
    robot.set_joint_positions(q0.astype(np.float32))
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)

    def apply(tgt_full):
        robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=np.asarray(tgt_full, np.float32)[act], joint_indices=act))

    def base_cmd(u, yw):
        vx, vy, wz = (float(v) for v in u)
        robot.set_linear_velocity(np.array([vx * math.cos(yw) - vy * math.sin(yw), vx * math.sin(yw) + vy * math.cos(yw), 0.0], dtype=np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, wz], dtype=np.float32))

    def q9():
        p, qt = robot.get_world_pose()
        jp = np.asarray(robot.get_joint_positions(), float)
        return np.r_[float(p[0]), float(p[1]), yaw_of(qt), [float(jp[i]) for i in idx]], jp

    def door_q():
        _, o = leaf_v.get_world_poses()
        w, x, y, z = (float(v) for v in np.asarray(o)[0])
        return -math.atan2(2 * (x * y + z * w), 1 - 2 * (y * y + z * z))
    hprims = [pr for pr in collision_prims(stage, DOOR) if str(pr.GetPath()) == f'{DOOR}/leaf/handle_bar']
    if len(hprims) != 1:
        print(f'[da] 把手 prim 數 {len(hprims)} ≠ 1 ⇒ 拒跑', flush=True)
        return 22

    def truth_handle():                                               # 只供事後評估
        M = UsdGeom.XformCache().GetLocalToWorldTransform(hprims[0])
        return np.array([[M[i][j] for j in range(4)] for i in range(4)]).T
    # 手腕相機（同靜態擷取）
    eef = [b for b in bodies if name_of(b) == 'link_eef'][0]
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
    except Exception as e:                                            # noqa: BLE001
        print(f'[da] 視角設定失敗（不影響物理）：{e!r}', flush=True)
    readback = dict(physics_dt=float(world.get_physics_dt()), ext_forces_every_iter=bool(sapi.GetEnableExternalForcesEveryIterationAttr().Get()), kp=a.kp, kd=a.kd, start_base=start_b.tolist(), start_q=start_q.tolist(),
                    door_limits_deg=[dauth.get('limit_lower_deg'), dauth.get('limit_upper_deg')], finger_open=FINGER_OPEN, mode=a.mode, viewpoint=a.viewpoint_id)
    print(f'[da] 讀回 {readback}', flush=True)
    if abs(readback['physics_dt'] - DT) > 1e-12 or not readback['ext_forces_every_iter']:
        print('[da] 物理設定讀回不符 ⇒ 拒跑', flush=True)
        return 64
    # 靜態保持
    tgt = np.array(robot.get_joint_positions(), float)
    quat0 = np.array([math.cos(start_b[2] / 2), 0.0, 0.0, math.sin(start_b[2] / 2)], np.float32)
    for k in range(int(round(a.settle_s / DT))):
        if k < 5:
            robot.set_world_pose(np.array([start_b[0], start_b[1], 0.0], np.float32), quat0)
        apply(tgt)
        base_cmd(np.zeros(3), 0.0)
        world.step(render=True)
    Kr = np.asarray(cam.get_intrinsics_matrix(), float)
    q_s, jp_s = q9()
    fbody = {name_of(b): b for b in bodies if name_of(b) in ('uflite_finger1', 'uflite_finger2')}

    def finger_model_vs_isaac(qv, jpv):
        """兩指連桿原點：控制器淨空模型（實際張開量）對 Isaac 連桿世界位姿。機器人自身幾何，不是場景真值。"""
        out, xc_ = {}, UsdGeom.XformCache()
        for n_, sg in (('uflite_finger1', 1.0), ('uflite_finger2', -1.0)):
            Tm = K.fk(qv, n_)
            pm = Tm[:3, 3] + sg * FINGER_OPEN * Tm[:3, 1]
            M = xc_.GetLocalToWorldTransform(stage.GetPrimAtPath(str(fbody[n_])))
            pi = np.array([M[3][0], M[3][1], M[3][2]], float)
            out[n_] = float(np.linalg.norm(pm - pi))
        return out
    fing = dict(q1=float(jp_s[f1]), q2=float(jp_s[f2]), model=FINGER_OPEN, pose_err_m=finger_model_vs_isaac(q_s, jp_s))
    readback['finger_readback'] = fing
    print(f'[da] 手指讀回 {fing}', flush=True)
    if abs(fing['q1'] - FINGER_OPEN) > FINGER_TOL_M or abs(fing['q2'] - FINGER_OPEN) > FINGER_TOL_M or max(fing['pose_err_m'].values()) > FINGER_POSE_TOL_M:
        print('[da] 手指張開量或連桿位姿與淨空模型不符 ⇒ 拒跑', flush=True)
        json.dump(clean(dict(refused='finger_model_mismatch', readback=readback)), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
        return 65
    door_q0 = door_q()
    truth = dict(note='只供事後評估；控制、估計、視點選擇都不讀', T_handle_start=truth_handle().tolist(), door_q0_rad=door_q0)
    print(f'[da] 起點讀回 底盤 {np.round(q_s[:3], 4).tolist()} 手臂靜態偏移 {np.round(q_s[3:] - start_q, 4).tolist()} 門角 {math.degrees(door_q0):.4f}°', flush=True)
    ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    ctrl = DA.DoorApproachCtrl(ident, start_b, start_q, b1, q_look, T_far, T_at)
    T0 = float(world.current_time)
    hist, last_rf, frames = {}, None, []
    cam_stat = dict(read=0, new=0, rejects={}, detect_ms=[], est_fail={}, delivered=0)
    latch = None
    rows, step_wall, clock_bad, n_latch_check = [], [], 0, 0
    obs_delivered, far_live, smoke_res = False, None, dict(paths={}, pipe=None, contact_probe=None)
    w_all0 = time.perf_counter()
    fdir = os.path.join(a.out, 'frames')
    os.makedirs(fdir, exist_ok=True)
    rng_n = [0]

    def fail(why, **kw):
        nonlocal latch
        if latch is None:
            latch = dict(t=round(float(ctrl.t), 3), why=why, **kw)
            print(f'[da] **鎖存失敗** {latch}', flush=True)
            ctrl._stop('exec_' + why)

    def grab(t_now):
        """取目前影格：回 None 或 dict(t_obs（控制時間）, dep, rgb, T_cam_fk, q, rf)；拒收原因計數。"""
        nonlocal last_rf
        fr = cam.get_current_frame()
        cam_stat['read'] += 1
        rfd = fr.get('rendering_frame')
        rf = (rfd.get('referenceTimeNumerator'), rfd.get('referenceTimeDenominator')) if isinstance(rfd, dict) else rfd
        col = fr.get('rgb') if fr.get('rgb') is not None else fr.get('rgba')
        if rf is None or rf == last_rf or rf == (0, 0) or col is None or fr.get('distance_to_image_plane') is None:
            return None
        last_rf = rf
        cam_stat['new'] += 1

        def rej(why):
            cam_stat['rejects'][why] = cam_stat['rejects'].get(why, 0) + 1
        t_r = fr.get('rendering_time')
        if t_r is None:
            return rej('no_rendering_time')
        hp = hist.get(int(round(float(t_r) / DT)))
        if hp is None or abs(hp['t_sim'] - float(t_r)) > 0.25 * DT:
            return rej('no_pose_at_render_time')
        dep = np.asarray(fr['distance_to_image_plane'], dtype=np.float32)
        rgba = np.asarray(col)
        if dep.size == 0 or rgba.size == 0:
            return rej('empty_image')
        T_fk = K.fk(hp['q'], W0.OPT)
        dp = float(np.linalg.norm(T_fk[:3, 3] - hp['T_render'][:3, 3]))
        return dict(t_obs=float(t_r) - T0, dep=dep[:, :, 0] if dep.ndim == 3 else dep, rgb=rgba[:, :, :3].astype(np.uint8), T_cam=T_fk, q=hp['q'], rf=rf, fk_vs_render_m=dp, age_at_grab=t_now - (float(t_r) - T0))

    def estimate(f, t_now, tag):
        """凍結偵測＋幾何估計。回 (T_WH 或 None, 資訊)。"""
        rng_n[0] += 1
        det, ms = D1.detect(f['dep'], Kr, f['T_cam'], np.random.default_rng(SEED + rng_n[0]))
        cam_stat['detect_ms'].append(float(ms))
        cls = MD.classify(det)
        info = dict(tag=tag, t_obs=round(f['t_obs'], 4), t_query=round(t_now, 4), cls=cls, reject=det['reject'], reject_L2=det['reject_L2'], detect_ms=round(float(ms), 1), fk_vs_render_m=f['fk_vs_render_m'])
        if cls != 'center_valid':
            return None, info
        est = E.estimate(dict(path=tag, L1=det['L1'], L2=det['L2'], reject=None, reject_L2=None, depth=f['dep'], K=Kr, T_cam=f['T_cam'], t_obs=f['t_obs']), 'N-obs', Rg, T_HG, a_H, query_t=t_now)
        info['est_ok'], info['est_why'] = bool(est['ok']), est.get('why')
        if not est['ok']:
            return None, info
        return np.asarray(est['T_WH'], float), info

    def save_frame(f, stem):
        import imageio.v2 as imageio
        imageio.imwrite(os.path.join(fdir, f'{stem}_rgb.png'), f['rgb'])
        np.save(os.path.join(fdir, f'{stem}_depth_m.npy'), f['dep'])
        frames.append(dict(stem=stem, t_obs=f['t_obs'], cam_T_fk=f['T_cam'].tolist(), q=[float(v) for v in f['q']], rendering_frame=[int(v) if v is not None else None for v in f['rf']] if isinstance(f['rf'], tuple) else f['rf']))

    def contacts():
        mx, who = 0.0, None
        for v, b in zip(v_tool, tool_bodies):
            try:
                n_ = float(np.linalg.norm(np.asarray(v.get_net_contact_forces(dt=DT), float).reshape(-1)[:3]))
            except Exception as e:                                    # noqa: BLE001
                return None, f'read_fail:{e!r}'
            if n_ > mx:
                mx, who = n_, name_of(b)
        try:
            Mb = np.asarray(v_base.get_contact_force_matrix(dt=DT), float).reshape(-1, len(base_filters), 3)
            nb = np.linalg.norm(Mb[0], axis=1)
            if float(nb.max()) > mx:
                mx, who = float(nb.max()), 'base_link|' + base_filters[int(nb.argmax())]
        except Exception as e:                                        # noqa: BLE001
            return None, f'read_fail:{e!r}'
        return mx, who

    n_steps = int(round(a.sim_limit_s / DT))
    pre_frame, phase_prev = None, None
    pipe_probe = dict(first_cmd_step=None, first_target_step=None, first_motion_step=None, s0=None)
    probe_state = dict(t_drop=None, done=False)
    contact_max = dict(n=0.0, who=None)
    finger_rng = dict(lo=FINGER_OPEN, hi=FINGER_OPEN)
    wall_over, aborted = False, None
    for k in range(n_steps):
        t = k * DT
        w_step0 = time.perf_counter()
        t_sim = float(world.current_time)
        q, jp = q9()
        pos_r, q_r = cam.get_world_pose(camera_axes='ros')
        hist[int(round(t_sim / DT))] = dict(t_sim=t_sim, q=q.copy(), T_render=T_of(pos_r, q_r))
        if len(hist) > 300:
            for kk in sorted(hist)[:len(hist) - 300]:
                hist.pop(kk, None)
        # 非預期接觸與門角（冒煙的接觸探針階段：鎖存即為預期結果）
        cn, cwho = contacts()
        dq_deg = abs(math.degrees(door_q() - door_q0))
        if cn is None:
            fail('contact_read_fail', detail=cwho)
        else:
            if cn > contact_max['n']:
                contact_max.update(n=cn, who=cwho)
            if cn > CONTACT_N:
                if probe_state['t_drop'] is not None and smoke_res['contact_probe'] is None:
                    smoke_res['contact_probe'] = dict(latched=True, t=round(t, 3), dt_after_drop_s=round(t - probe_state['t_drop'], 3), force_n=round(cn, 4), who=cwho)
                    print(f"[da] 冒煙：接觸探針鎖存 {smoke_res['contact_probe']}", flush=True)
                fail('unexpected_contact', force_n=round(cn, 4), who=cwho)
        if dq_deg > DOOR_DEG:
            fail('door_moved', door_deg=round(dq_deg, 4))
        for fv in (float(jp[f1]), float(jp[f2])):
            finger_rng['lo'], finger_rng['hi'] = min(finger_rng['lo'], fv), max(finger_rng['hi'], fv)
            if abs(fv - FINGER_OPEN) > FINGER_TOL_M and (probe_state['t_drop'] is None):
                fail('finger_opening_out_of_model', finger_q=fv)
        # 相機
        f = grab(t)
        if f is not None:
            if k < 50 and far_live is None and not smoke:              # 起點另拍一格：只記錄比較
                Tl, info = estimate(f, t, 'far_live')
                far_live = dict(info, T_WH=None if Tl is None else Tl.tolist(), d_vs_registered_far_mm=None if Tl is None else round(float(np.linalg.norm(Tl[:3, 3] - T_far[:3, 3])) * 1e3, 2))
                save_frame(f, 'far_live')
            if ctrl.look is None:
                pre_frame = f                                         # 到位前最後一格（冒煙：拿來核拒收）
            elif f['t_obs'] < ctrl.look['t_arrive'] + DT - 1e-9:
                cam_stat['rejects']['rendered_before_arrival'] = cam_stat['rejects'].get('rendered_before_arrival', 0) + 1
            elif ctrl.phase == 'LOOK' and ctrl.stop is None and not obs_delivered:
                if smoke:
                    c2 = copy.deepcopy(ctrl)
                    Tn, info = estimate(f, t, 'near_new')
                    save_frame(f, 'near_new')
                    r_new = dict(info, T_WH=None if Tn is None else Tn.tolist())
                    if Tn is not None:
                        c2.obs_intake(f['t_obs'], Tn)
                        c2._try_replan(q)
                        r_new.update(copy_phase_after=c2.phase, copy_replan=None if c2.replan is None else {kk: vv for kk, vv in c2.replan.items() if kk != 'T_pre_world'}, copy_stop=c2.stop, copy_rejected=dict(c2.obs_rej))
                    smoke_res['paths']['new_frame'] = r_new
                    c3 = copy.deepcopy(ctrl)                          # 到位前影格：控制器必須拒收
                    if pre_frame is not None:
                        Tp_, info_p = estimate(pre_frame, t, 'pre_arrival')
                        c3.obs_intake(pre_frame['t_obs'], Tp_ if Tp_ is not None else T_far)
                        c3._try_replan(q)
                        smoke_res['paths']['stale_pre_arrival'] = dict(info_p, t_arrive=round(ctrl.look['t_arrive'], 4), copy_phase_after=c3.phase, copy_rejected=dict(c3.obs_rej))
                    c4 = copy.deepcopy(ctrl)                          # 無有效估計：把深度整張設為無效（看不到把手）送同一偵測
                    f_bad = dict(f, dep=np.full_like(f['dep'], np.inf))
                    Tb_, info_b = estimate(f_bad, t, 'no_valid_estimate')
                    c4.obs_intake(f['t_obs'], Tb_, ok=Tb_ is not None)
                    c4._try_replan(q)
                    smoke_res['paths']['no_valid_estimate'] = dict(info_b, copy_phase_after=c4.phase, copy_rejected=dict(c4.obs_rej))
                    obs_delivered = True                              # 主控制器不收量測 ⇒ near_obs_timeout
                    print(f"[da] 冒煙影格三路徑 {json.dumps(clean(smoke_res['paths']), ensure_ascii=False)[:900]}", flush=True)
                else:
                    Tn, info = estimate(f, t, f'near_{cam_stat["new"]}')
                    if Tn is None:
                        key = info.get('est_why') or info['cls']
                        cam_stat['est_fail'][str(key)] = cam_stat['est_fail'].get(str(key), 0) + 1
                    else:
                        save_frame(f, 'near_used')
                        truth['T_handle_at_near'] = truth_handle().tolist()
                        ctrl.obs_intake(f['t_obs'], Tn)
                        obs_delivered = True
                        cam_stat['delivered'] += 1
                        cam_stat['near'] = dict(info, age_at_intake_s=round(t - f['t_obs'], 4), T_WH=Tn.tolist())
                        print(f"[da] 近距估計送入 {cam_stat['near']['t_obs']} 年齡 {cam_stat['near']['age_at_intake_s']} s", flush=True)
        # 控制
        tick = k % DA.NPH == 0
        w0 = time.perf_counter()
        out = ctrl.step(t, q)
        if smoke and tick:
            time.sleep(SMOKE_SLEEP_S)
        w_ctrl = time.perf_counter() - w0
        if ctrl.phase != phase_prev:
            print(f'[da] t={t:.2f} 相位 {phase_prev} → {ctrl.phase}', flush=True)
            phase_prev = ctrl.phase
        if pipe_probe['s0'] is None:
            pipe_probe['s0'] = out['arm_target'].copy()
        if pipe_probe['first_cmd_step'] is None and np.any(np.abs(ctrl.u_last[3:]) > 1e-9):
            pipe_probe['first_cmd_step'] = k
        if pipe_probe['first_target_step'] is None and np.any(np.abs(out['arm_target'] - pipe_probe['s0']) > 1e-12):
            pipe_probe['first_target_step'] = k
        for i, j in zip(idx, range(6)):
            tgt[i] = float(out['arm_target'][j])
        tgt[f1] = FINGER_OPEN
        apply(tgt)
        base_cmd(out['base_twist'], float(q[2]))
        # 冒煙：停止有效性判定後丟接觸探針
        if smoke and ctrl.stop_ok is not None and probe_state['t_drop'] is None:
            p3 = K.fk(q, 'link3')[:3, 3]
            probe.set_world_pose(np.array([p3[0], p3[1], p3[2] + 0.12], np.float32), np.array([1.0, 0, 0, 0], np.float32))
            probe.set_linear_velocity(np.zeros(3, np.float32))
            probe_state['t_drop'] = t
            print(f'[da] 冒煙：t={t:.2f} 接觸探針移到 link3 上方 0.12 m 落下', flush=True)
        st0 = float(world.current_time)
        world.step(render=True)
        st1 = float(world.current_time)
        if abs((st1 - st0) - DT) > 1e-9:
            clock_bad += 1
        sw = time.perf_counter() - w_step0
        step_wall.append(sw)
        rows.append([t, st0 - T0, *q, *out['arm_target'], *out['base_twist'], DA.HOLD.index(ctrl.phase) + 1 if ctrl.phase in DA.HOLD else 0, float(out['stopped']), w_ctrl, sw, 0.0 if cn is None else cn, dq_deg])
        if sw > STEP_WALL_LIMIT_S:
            fail('step_wall_timeout', wall_s=round(sw, 2))
        if time.perf_counter() - w_all0 > WALL_LIMIT_S and not wall_over:
            wall_over = True
            fail('wall_timeout')                                      # 可繼續執行：鎖存失敗並走受控停止，迴圈續跑到停止有效性判定
        if time.perf_counter() - w_all0 > WALL_LIMIT_S + HARD_WALL_EXTRA_S:
            aborted = 'hard_wall_abort：受控停止未在額外上限內完成；停止有效性未驗證'
            print(f'[da] **中止** {aborted}', flush=True)
            break
        if ctrl.stop_ok is not None:
            if not smoke and t - ctrl.stop_ok['t'] >= 1.0:
                break
            if smoke and (smoke_res['contact_probe'] is not None and t - smoke_res['contact_probe']['t'] >= 0.5 or (probe_state['t_drop'] is not None and t - probe_state['t_drop'] > 3.0)):
                break
    else:
        fail('sim_limit')
        aborted = 'sim_limit：模擬時間上限到；停止流程未跑完，停止有效性未驗證'
    wall_total = time.perf_counter() - w_all0
    R = np.array(rows)
    q_end, _ = q9()
    # 管線延遲（執行器側可觀察）：第一個非零手臂命令提交的步 → 設定點第一次改變的步，應相差 D_STEPS
    pipe_probe.pop('s0')
    pipe_probe['delay_steps'] = None if pipe_probe['first_cmd_step'] is None or pipe_probe['first_target_step'] is None else pipe_probe['first_target_step'] - pipe_probe['first_cmd_step']
    pipe_probe['expected'] = DA.D_STEPS
    rep = ctrl.report()
    # 事後評估（真值只在此）
    T_true = truth_handle()
    truth['T_handle_end'] = T_true.tolist()
    truth['door_q_end_rad'] = door_q()
    Tt = K.fk(q_end, 'link_tcp')
    Tp, Tg = T_at(T_true, PG.S_PRE), T_at(T_true, 0.0)
    z = Tg[:3, 2]
    e = Tt[:3, 3] - Tp[:3, 3]
    # 事後沿途真值幾何淨空：每控制拍的實測構型對真值把手，手指用本趟張開量（與 Isaac 讀回核對過的同一模型）
    th = dict(env=DA.R1.ENV_M, base=DA.R1.BASE_M, self=DA.R1.SELF_M, bar_tool=DA.R1.ENV_M)
    t_probe = probe_state['t_drop'] if probe_state['t_drop'] is not None else np.inf
    tc = dict(status='unverified', why=None)
    try:
        if abs(math.degrees(door_q() - door_q0)) > DOOR_DEG:
            tc['why'] = 'door_moved：把手真值非定值，本評估不適用'
        else:
            mn, at = {k_: np.inf for k_ in th}, {}
            n_cfg = 0
            for row in R[::DA.NPH]:
                if row[0] >= t_probe:
                    break
                cc = DA.FAST.clear_full(row[2:11], T_true, finger_q=FINGER_OPEN)
                n_cfg += 1
                for k_ in th:
                    if cc[k_] < mn[k_]:
                        mn[k_], at[k_] = cc[k_], round(float(row[0]), 2)
            tc = dict(status='computed', n_cfg=n_cfg, min_m={k_: round(float(v), 4) for k_, v in mn.items()}, at_t=at, thresholds=th, ok=bool(all(mn[k_] >= th[k_] for k_ in th)),
                      finger_q_model=FINGER_OPEN, finger_q_range_readback=[round(finger_rng['lo'], 5), round(finger_rng['hi'], 5)], note='幾何模型（點雲對方盒／圓柱）對真值把手位姿；取樣＝每控制拍')
    except Exception as e_:                                           # noqa: BLE001
        tc['why'] = repr(e_)
    hold = R[R[:, 20] > 0]
    stop_rows = R[R[:, 21] > 0]
    ev = dict(final_tcp_vs_true_pregrasp_mm=dict(total=round(float(np.linalg.norm(e)) * 1e3, 2), along_approach=round(float(e @ z) * 1e3, 2), perp=round(float(np.linalg.norm(e - (e @ z) * z)) * 1e3, 2)),
              final_tcp_to_true_grasp_along_mm=round(float((Tg[:3, 3] - Tt[:3, 3]) @ z) * 1e3, 2),
              near_est_vs_truth_mm=None if 'near' not in cam_stat else round(float(np.linalg.norm(np.array(cam_stat['near']['T_WH'])[:3, 3] - T_true[:3, 3])) * 1e3, 2),
              far_reg_vs_truth_mm=round(float(np.linalg.norm(T_far[:3, 3] - T_true[:3, 3])) * 1e3, 2),
              base_hold_drift_mm=None if not len(hold) else round(float(np.linalg.norm(hold[:, 2:4] - hold[0, 2:4], axis=1).max()) * 1e3, 4),
              base_hold_yaw_drift_deg=None if not len(hold) else round(float(np.degrees(np.abs(hold[:, 4] - hold[0, 4]).max())), 5),
              base_drift_after_stop_mm=None if not len(stop_rows) else round(float(np.linalg.norm(stop_rows[-1, 2:4] - stop_rows[0, 2:4])) * 1e3, 4),
              door_deg_max=round(float(R[:, 25].max()), 5), contact_max=dict(n=round(contact_max['n'], 5), who=contact_max['who']),
              true_clearance=tc,
              contact_monitor_note='淨接觸力 ≤0.05 N 只代表未觸發此接觸監看，不是絕對無接觸證明（多處接觸力可能抵消）',
              door_angle_note='門角用於失敗鎖存＝模擬真值安全監看，不是實機感測')
    timing = dict(sim_s=round(float(R[-1, 0] + DT), 2), wall_s=round(wall_total, 1), rtf=round(float((R[-1, 0] + DT) / wall_total), 4), clock_step_mismatch=int(clock_bad),
                  step_wall_ms=dict(p50=round(float(np.percentile(step_wall, 50)) * 1e3, 1), p95=round(float(np.percentile(step_wall, 95)) * 1e3, 1), max=round(float(np.max(step_wall)) * 1e3, 1)),
                  smoke_sleep_s_per_tick=SMOKE_SLEEP_S if smoke else 0.0, note='非即時計算：控制計算期間模擬暫停；影片若以模擬時間 1× 播放不代表實機執行速度')
    cam_stat['detect_ms'] = dict(n=len(cam_stat['detect_ms']), max=round(max(cam_stat['detect_ms'], default=0.0), 1))
    if smoke:
        if smoke_res['contact_probe'] is None:
            smoke_res['contact_probe'] = dict(latched=False, t_drop=probe_state['t_drop'])
        p = smoke_res['paths']
        checks = dict(clock_no_drift=clock_bad == 0, pipe_delay=pipe_probe['delay_steps'] == DA.D_STEPS,
                      path_new_frame=bool(p.get('new_frame', {}).get('copy_phase_after') == 'TO_PRE'),
                      path_stale=bool(p.get('stale_pre_arrival', {}).get('copy_rejected', {}).get('before_arrival', 0) >= 1 and p['stale_pre_arrival']['copy_phase_after'] == 'LOOK'),
                      path_no_estimate=bool(p.get('no_valid_estimate', {}).get('copy_rejected', {}).get('invalid', 0) >= 1 and p['no_valid_estimate']['copy_phase_after'] == 'LOOK'),
                      main_timeout_stop=bool(rep['stop'] and rep['stop']['why'] == 'near_obs_timeout'), stop_effective=bool(rep['stop_effect'] and rep['stop_effect']['ok']),
                      hold_base_cmd_zero=rep['tracking']['base_cmd_max_hold'] == 0.0, contact_probe_latched=bool(smoke_res['contact_probe']['latched']),
                      finger_model_matches=True, true_clearance_before_probe_ok=bool(tc.get('status') == 'computed' and tc.get('ok')), not_aborted=aborted is None,
                      no_unexpected_before_probe=bool(latch is None or (probe_state['t_drop'] is not None and latch['t'] >= probe_state['t_drop'] and latch['why'] == 'unexpected_contact')))
        ok = all(checks.values())
    else:
        checks = dict(clock_no_drift=clock_bad == 0, pipe_delay=pipe_probe['delay_steps'] == DA.D_STEPS, done=bool(rep['done']), stop_effective=bool(rep['stop_effect'] and rep['stop_effect']['ok']), no_latch=latch is None, not_aborted=aborted is None,
                      true_clearance_ok=bool(tc.get('status') == 'computed' and tc.get('ok')),
                      near_from_live_frame=bool(cam_stat['delivered'] == 1 and rep['target_source'] == 'near'), hold_base_cmd_zero=rep['tracking']['base_cmd_max_hold'] == 0.0)
        ok = all(checks.values())
    res = dict(run=os.path.basename(a.out.rstrip('/')), mode=a.mode, ok=bool(ok), checks=checks, latch=latch, aborted=aborted, readback=readback, controller=rep, eval_truth=ev, timing=timing, camera=cam_stat, far_live=far_live, pipe=pipe_probe,
               smoke=smoke_res if smoke else None, init=dict(far_estimate='登錄檔 source.T_WH_est（既有影格估計）＝初始化條件；不是本趟在線視覺', standoff_and_look='登錄視點（r5 通過、總時間最短）'),
               scope='非接觸；不插入、不閉爪、不開門、無行人；非即時閉迴路（計算時模擬暫停）')
    json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    json.dump(clean(truth), open(os.path.join(a.out, 'truth.json'), 'w'), ensure_ascii=False, indent=1)
    json.dump(clean(dict(frames=frames, K=Kr.tolist(), T0=T0)), open(os.path.join(a.out, 'frames_meta.json'), 'w'), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(a.out, 'log.npz'), rows=R, cols=np.array(['t', 't_sim_rel', 'bx', 'by', 'byaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 's1', 's2', 's3', 's4', 's5', 's6', 'ux', 'uy', 'uw', 'hold_phase', 'stopped', 'w_ctrl', 'w_step', 'contact_n', 'door_deg']))
    print(f"[da] 完成 ok={ok} checks={checks} 停止={rep['stop']} 鎖存={latch} 計時={timing}", flush=True)
    print(f"[da] 評估（真值） {json.dumps(clean(ev), ensure_ascii=False)}", flush=True)
    t_h = time.time()
    while sim_app.is_running() and time.time() - t_h < a.hold_s:
        world.render()
    return 0 if ok else 5


if __name__ == '__main__':
    rc = 99
    try:
        rc = main()
    except Exception:
        import traceback
        print('[da] 例外', traceback.format_exc(), flush=True)
    sim_app.close()
    sys.exit(rc)
