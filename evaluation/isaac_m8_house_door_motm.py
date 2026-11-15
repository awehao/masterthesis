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
ap.add_argument('--arm-friction-model', default='legacy', choices=['legacy', 'urdf_effort', 'no_passive_arm_friction'])   # no_passive_arm_friction（Codex Q2）：僅六個手臂關節的 legacy／逐軸摩擦歸零（研究假設變更，不是摩擦已修好）   # urdf_effort（Codex 20261115_1421xx）：手臂六關節逐軸摩擦 static＝dynamic＝1.0 N·m、viscous＝0；模型假設，不是實機辨識
ap.add_argument('--arm-fric-value', type=float, default=1.0)        # urdf_effort 模型的 static＝dynamic（N·m）；門任務＝1.0；探針用 0／0.2／1.0
ap.add_argument('--probe', action='store_true')                     # 手臂摩擦探針（Codex 核可 O1）：靜置後 joint4 限速限加速度小幅正反向動作，無門任務；不做抓持／綁定／開門
ap.add_argument('--diag', action='store_true')                      # 取證短診斷（Codex 20261115_14xxxx）：不送 P2、開門 0.5 s 後強制走閉鏈一致停止流程、每物理步補記根連桿三個時點等
a = ap.parse_args()

# ---- 啟動模擬器前：只匯入輕量的計算池／規劃程序模組並 fork（子程序自己匯入專案模組）；專案模組在 SimulationApp 之後才匯入
#      （專案模組會放入離線用的 rclpy 替身，若早於 Kit 載入會使 Kit 擴充套件匯入失敗——冒煙 1 實測崩潰 139）
import numpy as np                                               # noqa: E402
os.environ.setdefault('P2_TERM', '1')
os.environ.setdefault('P2_FINAL', '1')
import door_sweep_pool as DSP                                    # noqa: E402
import door_p2_planner_server as PS                              # noqa: E402
import door_first_seg as FSG                                     # noqa: E402
import door_cstop as DCS                                         # noqa: E402
POOL = DSP.SweepPool(8)
PLANNER = PS.PlannerServer()
PLANNER_FS = PS.PlannerServer()                                      # A1：首段滾動規劃用獨立程序（低優先權；交接時若仍在算即終止——可控取消，不停車等、不佔開門拍的核心）
try:
    os.setpriority(os.PRIO_PROCESS, PLANNER_FS.proc.pid, 10)
except Exception:                                                # noqa: BLE001
    pass
CSW = DCS.CStopWorker()                                              # 閉鏈一致備援停法的向前模擬程序（Codex 20261115_005224）

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
XO_D = G.D_STEPS * G.PHYS                                            # 管線延遲（s）
V, H_SEG, L_LAT = 0.05, 2.0, 1.2
L_LEAD, T_LEAD = 0.01, 0.2                                         # 冒煙 2：前導 0.15 m 起步會穿過 x≈0.80 的腕部奇異（關節 5 過零）⇒ 前導縮到 1 cm（σ 全程 ≥0.031）
GOAL = math.radians(85.0)
COND = dict(pos_m=0.005, rot_deg=3.0, rel_lin_mps=0.005, rel_ang_rps=0.02, rel_still_s=0.3, finger_q_m=(0.0005, 0.004), finger_contact_n=0.5, door_pre_deg=0.5, forbidden_n=0.05)
GRASP_LIMIT_S, OPEN_LIMIT_S = 12.0, 40.0
HOLD_DR_MAX = 0.3                                                    # 相對零摩擦基線的保持力矩殘差差值上限（N·m）
HOLD_R_MAX, HOLD_V_MAX = 1.3, 0.01                                   # P2：靜置保持力矩殘差上限（N·m）、低速門檻（rad/s）
# 首段適用窗與滾動參數在 door_first_seg（LEAD 0.10／TOL 0.05／AHEAD 1.0／GAP 0.4 s）
_UNUSED_PLAN = (0.10, 0.05, 1.0)                  # 首段：名目交接在預測 READY 後 0.10 s（第 3、4 趟實況）；適用窗 [0.10, 0.10＋0.05]；過期後重算的預測提前量 1.0 s（規劃在模擬時間約 0.8 s）
DIAG_STOP_T = 0.5                                                   # --diag：開門執行層時間到此即強制本拍「晚到」（走與第四趟相同的閉鏈一致停止流程）
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
    fric_usd = {}
    FRIC_ON = a.arm_friction_model in ('urdf_effort', 'no_passive_arm_friction')
    FRIC_VAL = 0.0 if a.arm_friction_model == 'no_passive_arm_friction' else float(a.arm_fric_value)
    if FRIC_ON:                                                      # world.reset 之前：逐軸 API（該軸優先於舊的無因次 jointFriction）；顯式保留原速度上限與 armature
        for pr in stage.Traverse():
            if pr.GetName() in ARM and pr.IsA(UsdPhysics.RevoluteJoint):
                old = PhysxSchema.PhysxJointAPI(pr)
                mv = old.GetMaxJointVelocityAttr().Get() if old.GetMaxJointVelocityAttr() else None
                am = old.GetArmatureAttr().Get() if old.GetArmatureAttr() and old.GetArmatureAttr().HasAuthoredValue() else 0.0
                jf = old.GetJointFrictionAttr().Get() if old.GetJointFrictionAttr() else None
                if mv is None or not math.isfinite(float(mv)):
                    print('[motm] 手臂關節原速度上限讀不到 ⇒ 拒跑', flush=True)
                    return 31
                pr.ApplyAPI('PhysxJointAxisAPI', 'angular')
                old.CreateJointFrictionAttr().Set(0.0)                # P1（Codex 20261115）：舊的無因次係數＝單位錯置的那一項，明設 0；摩擦由逐軸 1.0 N·m 表示
                for nm_, v_ in (('staticFrictionEffort', FRIC_VAL), ('dynamicFrictionEffort', FRIC_VAL), ('viscousFrictionCoefficient', 0.0), ('maxJointVelocity', float(mv)), ('armature', float(am))):
                    pr.CreateAttribute('physxJointAxis:angular:' + nm_, Sdf.ValueTypeNames.Float).Set(float(v_))
                dr = UsdPhysics.DriveAPI(pr, 'angular')
                fric_usd[pr.GetName()] = dict(legacy_jointFriction_unitless_imported=None if jf is None else float(jf), legacy_jointFriction_unitless_set=0.0, axis_static_Nm=FRIC_VAL, axis_dynamic_Nm=FRIC_VAL, axis_viscous=0.0, maxJointVelocity_deg_s=float(mv), armature=float(am),
                                              drive_maxForce_Nm=None if not dr.GetMaxForceAttr() else float(dr.GetMaxForceAttr().Get()))
        if sorted(fric_usd) != sorted(ARM):
            print('[motm] 手臂關節逐軸摩擦設定不完整 ⇒ 拒跑', sorted(fric_usd), flush=True)
            return 32
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
    try:                                                           # 門扇剛體角阻尼讀回（資產設定有 2.0；「鉸鏈零驅動」≠ 門無阻尼）
        leaf_damp = dict(angular=PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(dauth['leaf_prim'])).GetAngularDampingAttr().Get(),
                         linear=PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(dauth['leaf_prim'])).GetLinearDampingAttr().Get())
    except Exception as e_:                                        # noqa: BLE001
        leaf_damp = dict(error=repr(e_))
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
    # ---- 後端（PhysX tensor view）有效屬性讀回：摩擦三項、速度／力矩上限、armature、增益
    arm_backend = dict(model=a.arm_friction_model)
    try:
        pv = robot._articulation_view._physics_view
        fp = np.asarray(pv.get_dof_friction_properties(), float)[0]
        arm_backend.update(dof=[names[i] for i in idx], friction_static_dynamic_viscous=[[float(x) for x in fp[i]] for i in idx],
                           max_velocity=[float(np.asarray(pv.get_dof_max_velocities(), float)[0][i]) for i in idx], max_force=[float(np.asarray(pv.get_dof_max_forces(), float)[0][i]) for i in idx],
                           armature=[float(np.asarray(pv.get_dof_armatures(), float)[0][i]) for i in idx],
                           kp=[float(np.asarray(pv.get_dof_stiffnesses(), float)[0][i]) for i in idx], kd=[float(np.asarray(pv.get_dof_dampings(), float)[0][i]) for i in idx],
                           legacy_friction_coefficients=[float(np.asarray(pv.get_dof_friction_coefficients(), float)[0][i]) for i in idx] if hasattr(pv, 'get_dof_friction_coefficients') else None,
                           units='摩擦：static／dynamic＝N·m，viscous＝後端單位（tensor API；角度制或弧度制依後端，值為 0）；max_velocity＝後端單位（rad/s）；max_force＝N·m', usd=fric_usd)
    except Exception as e:                                            # noqa: BLE001
        arm_backend['error'] = repr(e)[:300]
    print('[motm] arm_backend', json.dumps(arm_backend, ensure_ascii=False)[:1500], flush=True)
    if FRIC_ON:
        fpv = np.array(arm_backend.get('friction_static_dynamic_viscous', [[np.nan] * 3]), float)
        mvb, mfb = np.array(arm_backend.get('max_velocity', [np.nan]), float), np.array(arm_backend.get('max_force', [np.nan]), float)
        ok_f = ('error' not in arm_backend and fpv.shape == (6, 3) and np.isfinite(fpv).all() and np.allclose(fpv[:, 0], FRIC_VAL, atol=1e-6) and np.allclose(fpv[:, 1], FRIC_VAL, atol=1e-6) and np.allclose(fpv[:, 2], 0.0, atol=1e-9)
                and np.isfinite(mvb).all() and np.allclose(mvb, [math.radians(fric_usd[names[i]]['maxJointVelocity_deg_s']) for i in idx], rtol=1e-3)
                and np.isfinite(mfb).all() and np.allclose(mfb, [fric_usd[names[i]]['drive_maxForce_Nm'] for i in idx], rtol=1e-3)
                and np.allclose(arm_backend['kp'], a.kp, rtol=1e-3) and np.allclose(arm_backend['kd'], a.kd, rtol=1e-3)
                and arm_backend.get('legacy_friction_coefficients') is not None and np.allclose(arm_backend['legacy_friction_coefficients'], 0.0, atol=1e-9))
        if not ok_f:
            print('[motm] 手臂關節後端有效屬性讀回不符（摩擦／速度上限／力矩上限／增益）⇒ 拒跑', flush=True)
            json.dump(dict(refused='arm_backend_readback', arm_backend=arm_backend), open(os.path.join(a.out, 'refused.json'), 'w'), ensure_ascii=False, indent=1)
            return 33
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

    # ---- --diag：每物理步三個時點（覆寫前 A／覆寫後 B／物理步後 C）的根連桿位姿與六維速度＋其他取證訊號
    dlog, dstate, derr = [], {}, []
    v_l6 = [v for v, b in zip(v_o, others + grip) if nm(b) == 'link6'][0]

    def _root():
        p_, q_ = robot.get_world_pose()
        return np.r_[np.asarray(p_, float), np.asarray(q_, float), np.asarray(robot.get_linear_velocity(), float), np.asarray(robot.get_angular_velocity(), float)]

    def _pq(view):
        p_, q_ = view.get_world_poses()
        return np.r_[np.asarray(p_, float)[0], np.asarray(q_, float)[0]]

    def diag_A():
        if a.diag:
            try:
                dstate['A'] = _root()
            except Exception as e:                                     # noqa: BLE001
                derr.append('A:' + repr(e)[:120])

    def diag_B():
        if a.diag:
            try:
                dstate['B'] = _root()
            except Exception as e:                                     # noqa: BLE001
                derr.append('B:' + repr(e)[:120])

    def diag_C(tt, mode_i, twist, pipe_, u_last_):
        if not a.diag:
            return
        try:
            C_ = _root()
            jp_ = np.asarray(robot.get_joint_positions(), float)
            jv_ = np.asarray(robot.get_joint_velocities(), float)
            try:
                me_ = np.asarray(robot.get_measured_joint_efforts(), float)
            except Exception:                                          # noqa: BLE001
                me_ = np.full(len(jp_), np.nan)
            try:
                ae_ = np.asarray(robot.get_applied_joint_efforts(), float)
            except Exception:                                          # noqa: BLE001
                ae_ = np.full(len(jp_), np.nan)
            F_ = np.asarray(robot.get_measured_joint_forces(), float)
            dlog.append(np.r_[tt, mode_i, dstate.get('A', np.full(13, np.nan)), dstate.get('B', np.full(13, np.nan)), C_, np.asarray(twist, float), jp_, jv_, me_, ae_, np.asarray(tgt, float), F_[j6][:6], F_[j6 - 1][:6],
                              _pq(v_l6), _pq(v_grip), _pq(v_leaf), np.concatenate([np.asarray(u_, float) for u_ in pipe_]), np.asarray(u_last_, float), door_q()])
        except Exception as e:                                         # noqa: BLE001
            derr.append('C:' + repr(e)[:160])

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
               gains=dict(kp=a.kp, kd=a.kd), arm_backend=arm_backend, door_readback=dict(hinge=hinge, leaf_rigid_body_damping=leaf_damp),
               finger=dict(command='CLOSE 段由張開斜降到 0（夾緊，同停車版）', geometry='兩指各自實測開度；CLOSE 段掃掠另含預測閉合開度'))
    # ---- 靜置
    quat0 = np.array([math.cos(b_start[2] / 2), 0.0, 0.0, math.sin(b_start[2] / 2)], np.float32)
    jv_hist, ct_hist, ct_err = [], [], []
    for k in range(int(round(a.settle_s / DT))):
        if k < 5:
            robot.set_world_pose(np.array([b_start[0], b_start[1], 0.0], np.float32), quat0)
        apply()
        base_cmd(np.zeros(3), 0.0)
        world.step(render=True)
        jv_hist.append(np.asarray(robot.get_joint_velocities(), float)[idx])
        try:
            fc_ = finger_contacts()
            ch_ = [float(fc_[0][0]), float(fc_[0][1]), float(fc_[1][0]), float(fc_[1][1])]
            for v_, b_ in zip(v_o + [v_base], others + grip + base_b):   # 各連桿原始通道逐一取值（先核有限，再彙整）
                ch_ += [float(x_) for x_ in np.linalg.norm(np.asarray(v_.get_contact_force_matrix(dt=DT), float).reshape(-1, 3), axis=1)]
            ct_hist.append(float(max(ch_)) if (len(ch_) > 4 and np.isfinite(ch_).all()) else float('nan'))
        except Exception as e:                                         # noqa: BLE001
            ct_hist.append(float('nan'))
            ct_err.append(repr(e)[:160])
    qs, jps = q9()
    # P2（Codex 20261115）：任務運動前的工程拒跑檢查——未綁定、無接觸、實測低速持續 0.3 s 後，r＝kp(s−q)＋kd(0−q̇)＋G(q)（速度目標＝0）
    jv_tail = np.abs(np.array(jv_hist[-int(round(0.3 / DT)):]))
    r_hold = a.kp * (s0 - qs[3:]) + a.kd * (0.0 - jv_hist[-1]) + BM.arm_gravity(K, qs, bodies_arm)
    hold_chk = dict(low_speed_0p3s=bool(jv_tail.max() <= HOLD_V_MAX), joint_rate_abs_max_0p3s=round(float(jv_tail.max()), 6), r_Nm=[round(float(x), 3) for x in r_hold], limit_Nm=HOLD_R_MAX, v_limit=HOLD_V_MAX,
                    note='r 含重力模型與驅動近似誤差：超過不證明新舊摩擦疊加，通過也不證明摩擦已校準；1.3＝1.0＋事前工程容差 0.3')
    n_need = int(round(0.3 / DT))
    ct_tail = np.array(ct_hist[-n_need:], float)
    hold_chk.update(contact_read_errors=ct_err[:3], n_contact_read_errors=len(ct_err), n_samples=int(min(len(jv_tail), len(ct_tail))), n_need=n_need, contact_force_max_n=None if not np.isfinite(ct_tail).all() else round(float(ct_tail.max()), 6),
                    no_contact=bool(len(ct_tail) == n_need and np.isfinite(ct_tail).all() and float(ct_tail.max()) <= 1e-6), finite=bool(np.isfinite(jv_tail).all() and np.isfinite(r_hold).all()))
    if a.arm_friction_model == 'no_passive_arm_friction':            # 相對凍結的零摩擦基線 r0（同起始構型；不得用本趟讀數重新歸零）
        B0 = json.load(open(os.path.join(HERE, 'experiments', '20261115_arm_hold_baseline_zero_friction.json')))
        dr = r_hold - np.asarray(B0['r0_Nm'], float)
        hold_chk.update(baseline=dict(file='experiments/20261115_arm_hold_baseline_zero_friction.json', source_run=B0['source_run'], source_sha256=B0['source_result_sha256'], r0_Nm=B0['r0_Nm']),
                        r_minus_r0_Nm=[round(float(x), 3) for x in dr], limit_rel_Nm=HOLD_DR_MAX, criterion='|r−r0|≤0.3 N·m（工程一致性檢查）')
        ok_r = bool(np.isfinite(dr).all() and float(np.abs(dr).max()) <= HOLD_DR_MAX)
    else:
        ok_r = bool(float(np.abs(r_hold).max()) <= HOLD_R_MAX)
    hold_chk['ok'] = bool(hold_chk['low_speed_0p3s'] and hold_chk['finite'] and len(jv_tail) == n_need and hold_chk['no_contact'] and ok_r)
    print('[motm] hold_check', json.dumps(hold_chk, ensure_ascii=False), flush=True)
    if FRIC_ON and not hold_chk['ok'] and not a.probe:
        print('[motm] 靜置保持力矩殘差檢查不過 ⇒ 拒跑', flush=True)
        json.dump(dict(refused='hold_residual', hold_check=hold_chk, arm_backend=arm_backend), open(os.path.join(a.out, 'refused.json'), 'w'), ensure_ascii=False, indent=1)
        return 34
    off_static = qs[3:] - s0                                          # 靜置實測靜態偏置（含未建模）
    door0 = door_q()
    res['settle'] = dict(hold_check=hold_chk, base=qs[:3].tolist(), arm_err_vs_pregrasp=[round(float(x), 5) for x in (qs[3:] - qa)], off_static=off_static.tolist(), door_deg=round(math.degrees(door0), 5))
    if a.probe:
        # ---- 手臂摩擦探針：joint4 設定點以梯形速度（0.1 rad/s、加速度 0.5 rad/s²）走 +0.10 rad → 停 PROBE_HOLD → 回 −0.10 rad → 停；其餘關節設定點不動；底盤零命令
        PV, PA, PD_, PHOLD = 0.1, 0.5, 0.10, 3.0
        def prof(tt):                                                  # 梯形：回 (位移, 速度)
            ta = PV / PA
            tc = (PD_ - PV * ta) / PV
            if tt < ta:
                return 0.5 * PA * tt * tt, PA * tt
            if tt < ta + tc:
                return 0.5 * PA * ta * ta + PV * (tt - ta), PV
            if tt < 2 * ta + tc:
                td = 2 * ta + tc - tt
                return PD_ - 0.5 * PA * td * td, PA * td
            return PD_, 0.0
        Tm = 2 * PV / PA + (PD_ - PV * PV / PA) / PV
        j4 = 3
        s_base = np.array([tgt[i] for i in idx], float)
        plog, perr = [], []
        tt_ = 0.0
        while tt_ < 2 * (Tm + PHOLD) - 1e-9:
            if tt_ < Tm + PHOLD:
                d_, v_ = prof(tt_)
            else:
                d2, v2 = prof(tt_ - Tm - PHOLD)
                d_, v_ = PD_ - d2, -v2
            tgt[idx[j4]] = float(s_base[j4] + d_)
            apply()
            base_cmd(np.zeros(3), 0.0)
            world.step(render=True)
            tt_ += DT
            try:
                jp_ = np.asarray(robot.get_joint_positions(), float)[idx]
                jv_ = np.asarray(robot.get_joint_velocities(), float)[idx]
                me_ = np.asarray(robot.get_measured_joint_efforts(), float)[idx]
                ch_ = []
                for vv_ in v_o + [v_base] + v_f:
                    ch_ += [float(x_) for x_ in np.linalg.norm(np.asarray(vv_.get_contact_force_matrix(dt=DT), float).reshape(-1, 3), axis=1)]
                qq_, _ = q9()
                plog.append(np.r_[tt_, d_, v_, [tgt[i] for i in idx], jp_, jv_, me_, wrist_F6(), max(ch_) if ch_ else np.nan, BM.arm_gravity(K, qq_, bodies_arm), qq_[:3]])
            except Exception as e:                                     # noqa: BLE001
                perr.append(repr(e)[:160])
        P_ = np.array(plog)
        cols_ = ['t', 's4_disp', 's4_vel_ref'] + [f'target_{k_}' for k_ in range(1, 7)] + [f'q_{k_}' for k_ in range(1, 7)] + [f'qd_{k_}' for k_ in range(1, 7)] + [f'meas_effort_{k_}' for k_ in range(1, 7)] + [f'j6_wrench_{k_}' for k_ in range(6)] + ['contact_max_n'] + [f'G_model_{k_}' for k_ in range(1, 7)] + ['bx', 'by', 'byaw']
        np.savez_compressed(os.path.join(a.out, 'probe_steps.npz'), P=P_, cols=np.array(cols_))
        def seg(t0_, t1_):
            S_ = P_[(P_[:, 0] > t0_) & (P_[:, 0] <= t1_)]
            still = bool(np.abs(S_[:, 15:21]).max() < XO.STOP_STILL['arm_rps'])          # 既有停穩判準（手臂 2e-3 rad/s），該段最後 0.3 s 全程
            L_ = S_[-1]
            gi_ = cols_.index('G_model_1')                              # 更正：重力欄依欄名取（原寫 28:34 取到旋量尾段與接觸欄）
            r_ = a.kp * (L_[3:9] - L_[9:15]) - a.kd * L_[15:21] + L_[gi_:gi_ + 6]
            return dict(t=[round(t0_, 2), round(t1_, 2)], settled_per_STOP_STILL=still, qd_abs_max=round(float(np.abs(S_[:, 15:21]).max()), 6), qd4_mean=round(float(S_[:, 18].mean()), 6),
                        target_minus_q_mrad=[round(float(x_) * 1e3, 3) for x_ in (L_[3:9] - L_[9:15])], r_nominal_pd_plus_G_Nm=[round(float(x_), 3) for x_ in r_],
                        meas_effort_Nm=[round(float(x_), 3) for x_ in S_[:, 21:27].mean(axis=0)], contact_max_n=float(np.nanmax(S_[:, 33])))
        mv = P_[(P_[:, 0] > PV / PA) & (P_[:, 0] <= Tm - PV / PA)]
        res['probe'] = dict(fric_value_Nm=float(a.arm_fric_value), model=a.arm_friction_model, motion=dict(joint='joint4', disp_rad=PD_, v_rad_s=PV, a_rad_s2=PA, hold_s=PHOLD, velocity_target='0（位置驅動；未送速度目標）'),
                            after_fwd_last_0p3s=seg(Tm + PHOLD - 0.3, Tm + PHOLD), after_back_last_0p3s=seg(2 * (Tm + PHOLD) - 0.3, 2 * (Tm + PHOLD)),
                            fwd_cruise=dict(lag_target_minus_q4_mrad_mean=round(float((mv[:, 6] - mv[:, 12]).mean()) * 1e3, 3), qd4_mean=round(float(mv[:, 18].mean()), 5), meas_effort4_mean=round(float(mv[:, 24].mean()), 3)),
                            n_rows=int(len(P_)), n_err=len(perr), errors=perr[:3], contact_max_n=float(np.nanmax(P_[:, 33])),
                            note='r＝名目 PD＋重力模型殘差（不是量到的摩擦力矩）；settled 才可稱靜止殘差')
        res['result'] = 'probe_done'
        res['aborted'] = None
        json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
        print('[motm] probe', json.dumps(clean(res['probe']), ensure_ascii=False)[:1800], flush=True)
        return 0
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
    fs = FSG.FirstSeg([PLANNER_FS], G.DT)                               # 首段計畫管理（執行器與離線核對共用；提前滾動更新）
    first = dict(ok=False, res=None, sel=None)
    gc.close_gate = lambda cc: fs.gate_ok()                          # 照本排程 READY 時刻算的首段已可用且可行
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
        # P2 首段：前導第一拍送出（預測 READY 狀態）；過期後依最新狀態重算（Codex 20261115：首段計畫過期修正）
        def make_req(t_now, target):
            T_plan = gc.handle_ref().copy() if gc.lock is not None else gc.T_est.copy()
            b_pred = q[:3] + np.r_[np.array([-V, 0.0]) * max(target - t_now, 0.0), 0.0]      # 通過線等速（抓持控制器的底盤命令；在途管線同為等速）
            qa_pred = ik(b_pred, T_at(T_plan, 0.0), q[3:])
            phi_now = door_q()
            x_pred = np.r_[b_pred, qa_pred, phi_now]
            T_HE_pred = np.linalg.inv(T_plan) @ T_at(T_plan, 0.0)
            WB.set_model(phi0=phi_now, finger_q=G.FINGER_CLOSED)
            hlp = WB.DoorWBOpt(T_plan, np.eye(4), x_pred, GOAL, N=2, dt=0.2, T_HE=T_HE_pred, project_start=False)
            v_pred = np.r_[-V, 0.0, 0.0, hlp.closure_velocity(x_pred, np.r_[-V, 0.0, 0.0], 0.0), 0.0]
            return (dict(T_WH0=T_plan, T_HE=T_HE_pred, x0=x_pred, v0=v_pred, H=H_SEG, goal=GOAL, fq=G.FINGER_CLOSED, phi_model=phi_now, try_final=True),
                    dict(T_plan=T_plan, T_HE_pred=T_HE_pred, x_pred=x_pred, phi_req=phi_now, wall_req=time.perf_counter() - t_wall0))

        if tick and gc.t0 is not None and gc.T_est is not None and gc.stop is None:
            sched_t = gc.t0 + gc.sched['hold_s'] + gc.sched['ins_s'] + gc.sched['close_s'] + gc.sched['confirm_s'] - G.DT
            for e_ in fs.step(t, sched_t, make_req, rolling=not a.smoke):
                if e_['what'] == 'request':
                    log_event('first_segment_request', k=e_['rid'], why=e_['why'], t_ready_pred=e_['target'])
                elif e_['what'] == 'result':
                    pl_ = [p_ for p_ in fs.plans if p_['rid'] == e_['rid']]
                    r__ = pl_[0]['res'] if pl_ else None
                    log_event('first_segment_available', k=e_['rid'], t_ready_pred=e_['target'], feasible=e_['ok'], late=e_['late'], plan_wall_s=e_['plan_wall_s'], failed=None if r__ is None else r__[3], final=None if r__ is None else r__[4], info=None if r__ is None else r__[6])
                    if e_['rid'] == 1 and r__ is not None:
                        first.update(ok=e_['ok'], res=r__, t_avail=t, wall_avail=time.perf_counter() - t_wall0, t_req=pl_[0]['t_req'], t_ready_pred=e_['target'], **{k_: pl_[0]['meta'][k_] for k_ in ('wall_req', 'phi_req')})
                else:
                    log_event('first_segment_desync', **{k_: v_ for k_, v_ in e_.items() if k_ != 'what'})
            if fs.desync is not None and gc.stop is None:              # 規劃通訊失同步：拒交接、走抓持控制器既有停止流程
                gc._stop('planner_desync', detail=fs.desync)
        pl_sel = fs.applicable(t) if tick else None
        plan_fit = pl_sel is not None
        if tick and gc.ready_now and gc.ready_ver is not None and gc.stop is None and not a.smoke and not plan_fit:
            log_event('handover_wait_plan', next_window=fs.next_window(t), ready_ver=gc.ready_ver)
        if tick and gc.ready_now and gc.ready_ver is not None and gc.stop is None and not a.smoke and plan_fit:
            w_ho = time.perf_counter()                                   # 交接計算（綁定條件、建執行層、預熱、試跑）的牆鐘：在本體 50 ms 計時之外，另記
            Th = truth_handle()
            Ttcp = K.fk(q, 'link_tcp')
            Tg_des = T_at(Th, 0.0)
            fc = finger_contacts()
            c = dict(pos_m=float(np.linalg.norm(Ttcp[:3, 3] - Tg_des[:3, 3])), rot_deg=rot_deg(Ttcp, Tg_des), finger_q=[float(jp[f1]), float(jp[f2])], finger_bar_n=[float(fc[0][0]), float(fc[1][0])],
                     door_pre_deg=math.degrees(abs(door_q() - door0)))
            okb, okd = gate.decide(c['pos_m'], c['rot_deg'], c['finger_q'], c['finger_bar_n'], c['door_pre_deg'])
            if not okb:                                                  # 先核實際抓持條件；不過就不做昂貴的預熱與試跑（Codex 20261113_144548 §4）
                log_event('handover_bind_not_ready', bind=okd, bind_values=c, ready_ver=gc.ready_ver, compute_ms=round((time.perf_counter() - w_ho) * 1e3, 1))
            else:
                phi_b = door_q()
                T_WH_b = gc.handle_ref(phi_b)
                T_HE_act = np.linalg.inv(T_WH_b) @ Ttcp
                fq_b = 0.5 * (float(jp[f1]) + float(jp[f2]))
                Xd0 = pl_sel['res'][1]
                first['sel'] = dict(rid=pl_sel['rid'], t_ready_pred=round(pl_sel['target'], 3), t_req=round(pl_sel['t_req'], 3), t_avail=round(pl_sel['t_avail'], 3))
                WB.set_model(phi0=phi_b, finger_q=fq_b)
                hlp2 = WB.DoorWBOpt(T_WH_b, np.eye(4), np.r_[q, phi_b], GOAL, N=2, dt=0.2, T_HE=T_HE_act, project_start=True)
                Xd0p = np.array([hlp2.project(x, iters=30) for x in Xd0])     # 首段參考以本趟實測 T_HE 重投影手臂（底盤、門角不變）；以試跑重核
                ho = dict(s=gc.s.copy(), pipe=[u_.copy() for u_ in gc.pipe], u_last=gc.u_last.copy(), u_prev=gc.u_prev.copy(), bias=np.asarray(gc.cfg.arm_model.bias, float).copy(), moving=True, off_static=gc.off(q))
                exn = XO.DoorOpenExec(ident, Xd0p, 0.05, T_WH_b, phi_b, T_HE_act, GOAL, ho, fq_b, arm_mode='closure', joint_kp=a.kp,
                                      g2=dict(gravity2=True, gravity3=True, ik_fc=True, pool=POOL, prep_fc=True, deadline_ms=50.0, cstop=CSW))
                exn.p2_open = True
                if a.diag:
                    exn.deadline_inject = lambda cc: cc.t >= DIAG_STOP_T - 1e-9   # 取證：0.5 s 後強制本拍晚到 ⇒ 與第四趟相同的閉鏈一致停止流程
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
                    if fs.pend[0] is not None:                         # A1：在途的首段滾動規劃已無用 ⇒ 終止該程序（之後不再送首段請求）
                        PLANNER_FS.proc.kill()
                        log_event('first_segment_cancelled', rid=fs.pend[0]['rid'], target=round(fs.pend[0]['target'], 3))
                        fs.pend[0] = None
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
                    np.savez(os.path.join(a.out, 'first_segment.npz'), Xd0=Xd0, Xd0p=Xd0p, T_plan=pl_sel['meta']['T_plan'], T_HE_pred=pl_sel['meta']['T_HE_pred'], T_WH_b=T_WH_b, T_HE_act=T_HE_act, phi_b=phi_b)
                    log_event('bind_and_handover', base_speed_cmd=round(float(np.linalg.norm(gc.u_last[:2])), 4), joint_enabled=bool(UsdPhysics.Joint(stage.GetPrimAtPath(JOINT)).GetJointEnabledAttr().Get()), warmup_ms=round(wu, 1))
                    break
        o = gc.step(t, q.copy(), door_q(), finger_meas=(float(jp[f1]), float(jp[f2])))   # r12：兩指各自實測開度供碰撞幾何
        if o.get('abort'):
            box['aborted'] = dict(t=round(t, 3), controller=gc.abort)
            log_event('grasp_abort', abort=gc.abort)
            break
        for i, j_ in zip(idx, range(6)):
            tgt[i] = float(o['arm_target'][j_])
        tgt[f1] = float(o['finger'])
        diag_A()
        apply()
        base_cmd(o['base_twist'], float(q[2]))
        diag_B()
        world.step(render=True)
        t += DT
        diag_C(t, 0, o['base_twist'], gc.pipe, gc.u_last)
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
    res['first_segment'] = {k_: first.get(k_) for k_ in ('t_req', 'wall_req', 't_avail', 'wall_avail', 'ok', 't_ready_pred', 'phi_req', 'sel')}
    res['first_segment'].update(n_req=fs.n_req, n_dropped=fs.n_stale_drop, log=fs.log, window=dict(lead_s=fs.lead, tol_s=fs.tol, ahead_s=fs.ahead, gap_s=fs.gap))
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
                xr = ex.ref(ex.sigma)                                  # σ 已以管線延遲起算 ⇒ 與實測同一時刻
                ey, ed = abs(math.degrees(q[2] - xr[2])), abs(math.degrees(phm - xr[9]))
                rec['track_err'] = dict(base_mm=round(float(np.linalg.norm(q[:2] - xr[:2])) * 1e3, 3), yaw_deg=round(ey, 3), door_deg=round(ed, 3))
                if not r_[2] or r_[1] is None:
                    rec['result'] = 'infeasible'
                elif ex.sigma > pend['sigma_s'] + 1e-9 and not pend['rest']:
                    rec['result'] = 'missed_deadline'
                # 偏航／門角追蹤誤差只記錄（rec['track_err']）；接管由接縫核對＋真狀態試跑決定（Codex 20261115_004634 §3）
                elif ex.staged is not None or ex.stop is not None:
                    rec['result'] = 'staged_busy_or_stopped'
                else:
                    okS, info = ex.stage_reference(r_[1], pend['sigma_s'], final=r_[4], ver=rec['k'])
                    rec['result'] = 'staged' if okS else 'stage_rejected'
                    if not okS:
                        rec['why'] = info.get('why')
                log_event('p2_plan_result', k=rec['k'], result=rec['result'], plan_wall_s=rec['plan_wall_s'], final=rec['final'])
            if box['pend'] is None and ex.stop is None and ex.p2_open and not a.diag and t >= box['t_next'] - 1e-9:
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
            diag_A()
            apply()
            base_cmd(out['base_twist'], float(q[2]))
            diag_B()
            world.step(render=True)
            t += DT
            diag_C(t, 1 if ex.stop is None else 2, out['base_twist'], ex.pipe, ex.u_last)
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
        res['open'].update(n_deadline_miss=ex.n_deadline_miss, p2_log=ex.p2_log, reqs=box['reqs'], fb_log=ex.fb_log, base_hold_log_tail=ex.hold_log[-5:], base_hold_n=len(ex.hold_log), base_hold_cmd_max=[max((x[1] for x in ex.hold_log), default=None), max((x[2] for x in ex.hold_log), default=None)], cstop_ticks=len(ex.cs_log), stop_mode=ex.stop_mode, tick_ms_normal=[round(float(m), 1) for i, m in enumerate(ex.tick_ms) if ex.stop is None or i * 0.05 < float(ex.stop['t']) - 1e-9])
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
    if a.diag:
        try:
            nd = len(robot.dof_names)
            lay = [('t', 1), ('mode(0抓持,1開門,2停止)', 1), ('rootA_pos3_quat4_lin3_ang3(覆寫前)', 13), ('rootB(覆寫後)', 13), ('rootC(物理步後)', 13), ('base_twist_cmd(本體座標)', 3), ('joint_pos', nd), ('joint_vel', nd),
                   ('measured_joint_efforts', nd), ('applied_joint_efforts', nd), ('position_targets(tgt)', nd), ('measured_joint_forces[j6]', 6), ('measured_joint_forces[j6-1]', 6),
                   ('link6_pos3_quat4', 7), ('gripper_pos3_quat4', 7), ('leaf_pos3_quat4', 7), ('pipe(8×9)', 72), ('u_last', 9), ('door_q', 1)]
            jinfo = {}
            for pr in stage.Traverse():
                if pr.GetName() in ('joint5', 'joint6') or str(pr.GetPath()) == JOINT:
                    jj = UsdPhysics.Joint(pr)
                    g_ = lambda at: (None if not at or at.Get() is None else [float(x) for x in (list(at.Get()) if not hasattr(at.Get(), 'GetReal') else [at.Get().GetReal(), *at.Get().GetImaginary()])])   # noqa: E731
                    jinfo[str(pr.GetPath())] = dict(type=str(pr.GetTypeName()), body0=[str(x) for x in jj.GetBody0Rel().GetTargets()], body1=[str(x) for x in jj.GetBody1Rel().GetTargets()],
                                                    localPos0=g_(jj.GetLocalPos0Attr()), localRot0_wxyz=g_(jj.GetLocalRot0Attr()), localPos1=g_(jj.GetLocalPos1Attr()), localRot1_wxyz=g_(jj.GetLocalRot1Attr()))
            dmeta = dict(layout=lay, dof_names=list(robot.dof_names), joint_indices={str(k_): int(v_) for k_, v_ in dict(meta.joint_indices).items()}, j6_row=j6,
                         link_names=[str(x) for x in getattr(meta, 'link_names', [])], joint_names=[str(x) for x in getattr(meta, 'joint_names', [])],
                         api_doc=dict(get_measured_joint_forces=str(getattr(robot.get_measured_joint_forces, '__doc__', ''))[:1500], get_measured_joint_efforts=str(getattr(robot.get_measured_joint_efforts, '__doc__', ''))[:1500],
                                      get_applied_joint_efforts=str(getattr(robot.get_applied_joint_efforts, '__doc__', ''))[:1500]),
                         usd_joints=jinfo, errors=derr[:20], n_err=len(derr), diag_stop_t=DIAG_STOP_T, phys_dt=DT)
            json.dump(dmeta, open(os.path.join(a.out, 'diag_meta.json'), 'w'), ensure_ascii=False, indent=1)
            if dlog:
                nmax = max(len(x_) for x_ in dlog)
                np.savez_compressed(os.path.join(a.out, 'diag_steps.npz'), D=np.array([np.r_[x_, np.full(nmax - len(x_), np.nan)] for x_ in dlog]))
        except Exception as e:                                         # noqa: BLE001
            print('[motm] diag 存檔失敗', repr(e)[:300], flush=True)
        res['diag'] = dict(note='取證短診斷：不送 P2；開門執行層 0.5 s 後強制本拍晚到，走閉鏈一致停止流程（與第四趟相同）；不是開門成果', forced_stop_t=DIAG_STOP_T, n_rows=len(dlog), n_err=len(derr))
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
        PLANNER_FS.close()
        CSW.close()
    except Exception:                                              # noqa: BLE001
        pass
    sim_app.close()
    sys.exit(rc)
