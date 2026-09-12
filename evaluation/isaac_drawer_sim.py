"""Isaac：固定底盤的手臂經抓取關係帶動被動抽屜。

與第一階段（isaac_manip_sim.py）的關係：那支停在預抓取、**沒有接觸**，而且維持
凍結不動。本支是新檔，多了抽屜資產、抓取關係、力的量測與階段化的停止處置。

抽屜是**被動**的：drive 五項為 0，本程式不發任何開度命令。開度只能由抓取關係
帶動。（施加外力的方向診斷在 isaac_drawer_probe.py，刻意放在另一個檔案。）

抓取模型兩版，由案例的 grasp_model 決定，**不混稱**：
    fixed_attachment  engage 時在夾爪與抽屜之間建一個固定關節。關節的兩端框架
                      取自**當下**的相對位姿，所以連接瞬間沒有位置跳變。
                      手指維持全開 —— 再去夾住橫桿就不是「理想固定連接」了。
    friction          只有接觸與摩擦（本檔支援，但屬於下一輪）。

力的量測（見案例 force 區塊）：
    get_measured_joint_forces() 回傳「該連桿上游關節」的反作用力，且在**子連桿
    座標系**裡。本程式取 joint6 那一列，用 link6 的世界旋轉轉到世界座標，
    再沿抽屜軸投影得 F_pull。**夾爪慣性項沒有扣除**，所以那是「手腕沿抽屜軸
    傳遞的力」，不是純把手拉力。
    座標慣例用靜止段自我核對：無接觸時 |F| 應該等於 joint6 之後所有連桿的重量。

停止處置分兩種（凍結設定點**不保證**力會下降，位置驅動會持續施力）：
    一般異常  凍結最後一次已套用的設定點
    超力      先解除耦合（拆掉固定關節／張開手指），再凍結，才停止
"""
import argparse, hashlib, json, math, os, sys, threading, time
import numpy as np
import yaml

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(WS, 'evaluation')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

ap = argparse.ArgumentParser()
ap.add_argument('--case', default='drawer_open_a_fixed')
ap.add_argument('--cases', default=os.path.join(
    WS, 'src/my_omnibot_description/config/manipulation_cases.yaml'))
ap.add_argument('--spec', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--poses', default=os.path.join(
    WS, 'src/my_omnibot_description/config/arm_initial_pose.yaml'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--headless', default='true')
ap.add_argument('--sim-limit', type=float, default=45.0)
ap.add_argument('--wall-limit', type=float, default=900.0)
ap.add_argument('--rtf', type=float, default=1.0, help='>0 時以此倍率對齊牆鐘')
ap.add_argument('--kp', type=float, default=1.0e5)
ap.add_argument('--kd', type=float, default=1.0e4)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--cpu-threads', type=int, default=8)
a = ap.parse_args()

import drawer_asset as DA                                           # noqa: E402
from cpu_temp import read as cpu_temp_read                          # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

WB_URDF = os.path.join(WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf')
KIN = WholeBodyKinematics.from_urdf_string(open(WB_URDF).read())
KIDX = [KIN.dof_names.index(f'joint{i}') for i in range(1, 7)]


def fk_tcp(qa, park):
    q = np.zeros(len(KIN.dof_names))
    q[0], q[1], q[2] = park
    q[KIDX] = np.asarray(qa, float)
    return KIN.fk(q, 'link_tcp')

CASES = yaml.safe_load(open(a.cases))
CASE = CASES['cases'][a.case]
SPEC = DA.load(a.spec)
POSES = yaml.safe_load(open(a.poses))
ARM = [f'joint{i}' for i in range(1, 7)]
Q_START = np.array([float(POSES[CASE['pregrasp']['start_config']][j]) for j in ARM])
PARK = (float(CASE['parking']['x']), float(CASE['parking']['y']),
        math.radians(float(CASE['parking']['yaw_deg'])))
POSE = (float(CASE['object']['pose'][0]), float(CASE['object']['pose'][1]))
TOL = CASE['tolerance']
TMO = CASE['timeout_s']
FRC = CASE['force']
AXIS = np.array(FRC['drawer_axis_world'], float)
AXIS = AXIS / np.linalg.norm(AXIS)
GRASP_MODEL = CASE['grasp_model']
TARGET = float(CASE['drawer']['target_opening_m'])
CPU_LIMIT = float(CASE['cpu_limit_c'])
F_OPEN = float(SPEC['grasp_surface']['finger_joint_open'])
TCP_OFF = float(SPEC['grasp_surface']['tcp_offset_along_tool_z'])
BAR = SPEC['drawer']['handle']['bar']['center']

os.makedirs(a.out, exist_ok=True)
SHA = {k: hashlib.sha256(open(v, 'rb').read()).hexdigest()[:16] for k, v in
       (('urdf', a.urdf), ('spec', a.spec), ('cases', a.cases), ('poses', a.poses))}
print(f'[drawer] 案例 {a.case}  抓取模型 {GRASP_MODEL}')
print(f'[drawer] sha {SHA}')
print(f'[drawer] 停放 ({PARK[0]:.4f}, {PARK[1]:.4f}) yaw {CASE["parking"]["yaw_deg"]}°')
print(f'[drawer] 目標開度 {TARGET:.3f} m ± {TOL["opening_m"]:.3f}，'
      f'保持 {TOL["opening_hold_s"]:.1f} s')

from isaacsim import SimulationApp                                  # noqa: E402
_cfg = {'headless': a.headless.lower() == 'true'}
if a.cpu_threads > 0:
    _cfg['limit_cpu_threads'] = a.cpu_threads
sim_app = SimulationApp(_cfg)

from isaacsim.core.api import World                                 # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane      # noqa: E402
from isaacsim.core.prims import SingleArticulation, RigidPrim       # noqa: E402
from isaacsim.core.utils.types import ArticulationAction            # noqa: E402
from pxr import UsdGeom, UsdPhysics, Gf, Sdf, Usd                   # noqa: E402
from isaac_common import import_urdf                                # noqa: E402

import rclpy                                                        # noqa: E402
from rclpy.node import Node                                         # noqa: E402
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)                            # noqa: E402
from rosgraph_msgs.msg import Clock                                 # noqa: E402
from sensor_msgs.msg import JointState                              # noqa: E402
from geometry_msgs.msg import PoseStamped                           # noqa: E402
from std_msgs.msg import Float64MultiArray, String                  # noqa: E402

DRAWER = '/World/drawer_unit/drawer'
ROBOT = '/World/omni_bot'
JOINT_ATTACH = '/World/drawer_unit/grasp_attach'


def q_yaw(t):
    return [math.cos(t / 2), 0.0, 0.0, math.sin(t / 2)]


def yaw_of(q):
    w, x, y, z = [float(v) for v in q]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class DrawerNode(Node):
    def __init__(self):
        super().__init__('isaac_drawer_sim')
        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST)
        rel = QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_ALL,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.tcp_pub = self.create_publisher(PoseStamped, '/manip/tcp_pose', be)
        self.dr_pub = self.create_publisher(Float64MultiArray, '/manip/drawer_state', be)
        self.gr_pub = self.create_publisher(Float64MultiArray, '/manip/grasp_state', be)
        self.ct_pub = self.create_publisher(Float64MultiArray, '/manip/contact', be)
        self.st_pub = self.create_publisher(String, '/manip/status', 10)
        # 一次指派的快照：讀取端一次取走整組，不會讀到半新半舊的 (seq, q)
        self.snap = None
        self.fsnap = None
        self.pending = []           # 尚未套用的事件
        self.phase = 'idle'
        self.arm_cb_n = self.arm_cb_rej = 0
        self.g_cb_n = self.p_cb_n = 0
        self.create_subscription(Float64MultiArray, '/arm/joint_position_cmd',
                                 self._cmd, 10)
        self.create_subscription(Float64MultiArray, '/manip/gripper_cmd',
                                 self._grip, 10)
        self.create_subscription(String, '/manip/phase_cmd', self._phase, rel)

    def _cmd(self, m):
        self.arm_cb_n += 1
        if len(m.data) != 9:
            self.arm_cb_rej += 1
            return
        self.snap = (int(m.data[0]), float(m.data[2]), np.array(m.data[3:9]))

    def _grip(self, m):
        self.g_cb_n += 1
        if len(m.data) != 3:
            return
        self.fsnap = (int(m.data[0]), float(m.data[2]))

    def _phase(self, m):
        self.p_cb_n += 1
        try:
            d = json.loads(m.data)
        except Exception:
            return
        self.phase = d.get('phase', self.phase)
        if d.get('event'):
            self.pending.append(d)

    def say(self, d):
        s = String(); s.data = json.dumps(d, ensure_ascii=False)
        self.st_pub.publish(s)

    def farr(self, pub, vals):
        m = Float64MultiArray(); m.data = [float(v) for v in vals]; pub.publish(m)


def main():
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    if abs(float(world.get_rendering_dt()) - float(a.physics_dt)) > 1e-12:
        print('[drawer] rendering_dt != physics_dt，中止'); return 4
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    authored = DA.build_usd(world.stage, SPEC, POSE)
    import_urdf(a.urdf, ROBOT)

    stage = world.stage
    fingers = [str(p.GetPath()) for p in stage.Traverse()
               if p.GetName() in ('uflite_finger1', 'uflite_finger2')]
    grip_link = [str(p.GetPath()) for p in stage.Traverse()
                 if p.GetName() == 'uflite_gripper_link']
    print(f'[drawer] 手指 prim {fingers}')
    print(f'[drawer] 夾爪 prim {grip_link}')
    if not grip_link:
        print('[drawer] 找不到 uflite_gripper_link，中止'); return 5
    GRIP = grip_link[0]

    # **view 必須在 world.reset() 之前建立**：prepare_contact_sensors 要早於
    # PhysX 場景建好才有效。reset 之後才建，接觸力一律回傳 0（已實測）。
    print('[drawer] 建立抽屜 view ...', flush=True)
    drawer_v = RigidPrim(prim_paths_expr=DRAWER, name='drawer_v',
                         track_contact_forces=True, max_contact_count=128,
                         prepare_contact_sensors=True)
    print('[drawer] 建立手指 view ...', flush=True)
    finger_v = None
    if fingers:
        # 用字元類別而不是 (A|B) 的全路徑交替：兩條路徑只差最後一個字元
        finger_v = RigidPrim(prim_paths_expr=fingers[0][:-1] + '[12]',
                             name='finger_v', track_contact_forces=True,
                             max_contact_count=128, prepare_contact_sensors=True,
                             contact_filter_prim_paths_expr=[DRAWER])
    print('[drawer] world.reset() ...', flush=True)
    world.reset()
    print('[drawer] reset 完成', flush=True)
    robot = SingleArticulation(prim_path=ROBOT, name='omni_bot')
    robot.initialize()
    print(f'[drawer] articulation DOF {robot.num_dof}', flush=True)
    names = list(robot.dof_names)
    idx = {n: i for i, n in enumerate(names)}
    missing = [j for j in ARM if j not in idx]
    if missing:
        print(f'[drawer] URDF 缺關節 {missing}，中止'); return 5

    q = robot.get_joint_positions()
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    for k, j in enumerate(ARM):
        q[idx[j]] = Q_START[k]; kp[idx[j]] = a.kp; kd[idx[j]] = a.kd
    FJ = [j for j in ('finger_joint1', 'finger_joint2') if j in idx]
    for j in FJ:
        q[idx[j]] = F_OPEN; kp[idx[j]] = a.finger_kp; kd[idx[j]] = a.finger_kd
    robot.set_joint_positions(q)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=q))
    robot.set_world_pose(np.array([PARK[0], PARK[1], 0.0]), np.array(q_yaw(PARK[2])))
    print('[drawer] view initialize ...', flush=True)
    drawer_v.initialize()
    if finger_v is not None:
        finger_v.initialize()
    print('[drawer] view 就緒，暖機 30 步', flush=True)
    for _ in range(30):
        world.step(render=False)

    p0, qq0 = robot.get_world_pose()
    BASE0 = (float(p0[0]), float(p0[1]), yaw_of(qq0))
    dp0, _ = drawer_v.get_world_poses()
    DY0 = float(dp0[0][1])
    print(f'[drawer] 初始底盤 ({BASE0[0]:.4f}, {BASE0[1]:.4f}) '
          f'yaw {math.degrees(BASE0[2]):.3f}°，抽屜 y0 {DY0:.5f}')

    prims = {}
    for nm in ('link_tcp', 'link6'):
        pr = next((p for p in stage.Traverse() if p.GetName() == nm), None)
        if pr is None:
            print(f'[drawer] 找不到 {nm}，中止'); return 6
        prims[nm] = pr

    def world_T(pr):
        M = UsdGeom.Xformable(pr).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = M.ExtractTranslation(); r = M.ExtractRotationQuat(); i = r.GetImaginary()
        R = np.array([[M[ri][ci] for ci in range(3)] for ri in range(3)])
        sc = np.linalg.norm(R, axis=1)
        R = R / sc[:, None]
        # USD 是列向量慣例（v' = v·M），要的旋轉矩陣是它的轉置
        return np.array([t[0], t[1], t[2]]), R.T, \
            np.array([r.GetReal(), i[0], i[1], i[2]])

    # joint6 在 measured_joint_forces 裡的列號（該連桿上游關節 = joint_index + 1）
    j6_row, j6_src = None, 'none'
    try:
        md = robot._articulation_view._metadata
        j6_row = int(md.joint_indices['joint6']) + 1
        j6_src = 'metadata.joint_indices'
        jnames = list(md.joint_names)
    except Exception as e:
        jnames = []
        print(f'[drawer] 讀不到 articulation metadata（{e}）')
    if j6_row is None:
        j6_row = idx['joint6'] + 1; j6_src = 'dof_names 順序（後備）'
    print(f'[drawer] joint6 反作用力列號 {j6_row}（來源 {j6_src}）')
    if jnames:
        print(f'[drawer] articulation joint 順序 {jnames}')

    def attach():
        Ma = UsdGeom.Xformable(stage.GetPrimAtPath(GRIP)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        Mb = UsdGeom.Xformable(stage.GetPrimAtPath(DRAWER)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        Mrel = Ma * Mb.GetInverse()      # 夾爪原點框架，用抽屜座標表示
        j = UsdPhysics.FixedJoint.Define(stage, JOINT_ATTACH)
        j.CreateBody0Rel().SetTargets([Sdf.Path(GRIP)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(DRAWER)])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        t = Mrel.ExtractTranslation(); r = Mrel.ExtractRotationQuat()
        i = r.GetImaginary()
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(float(t[0]), float(t[1]), float(t[2])))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(float(r.GetReal()),
                                             float(i[0]), float(i[1]), float(i[2])))
        j.CreateJointEnabledAttr().Set(True)
        return [float(t[0]), float(t[1]), float(t[2])]

    def detach():
        pr = stage.GetPrimAtPath(JOINT_ATTACH)
        if pr.IsValid():
            UsdPhysics.Joint(pr).GetJointEnabledAttr().Set(False)
            return True
        return False

    # --- 力的量測：列號、符號與旋轉慣例 ---
    # 列號規則 joint_index + 1 已用**已知外力**校正過（evaluation/results 的
    # force_known 診斷）：在夾爪施世界 (0,0,−20) N，row 100 的 Δ|F| = 20.109 N
    # （誤差 0.5 %），finger_joint1 那列只變 0.003 N。
    # 符號：回報的是**父對子的反作用力**，與外加負載反號。
    # 偏置：靜止無接觸時該列約 5.2 N，而 joint6 以下的重量只有 4.05 N，
    #       約 1.1 N 的常數偏置未解釋 —— 所以這個量**不能直接當把手拉力**，
    #       只當「手腕傳遞力」的安全監看（30 N 門檻離 5 N 基線還有 25 N）。
    dist_mass_chk = 0.4126          # joint6 以下總質量（含 link6），由 URDF 樹算得
    _, l6R0, _ = world_T(prims['link6'])
    try:
        F0 = np.array(robot.get_measured_joint_forces())
        fw0 = l6R0 @ F0[j6_row][:3]
        force_rows = int(F0.shape[0])
    except Exception as e:
        fw0 = np.zeros(3); force_rows = -1
        print(f'[drawer] 讀不到 measured_joint_forces：{e}')
    exp_w = dist_mass_chk * 9.81
    # 旋轉慣例：stage 讀出的 TCP 姿態要與獨立 FK 算出的一致，否則轉到世界座標
    # 的力就是錯的。全零位時 R 與 R^T 幾乎相同，分辨不出來，所以要在**運動中**
    # 持續比對（見主迴圈的 rot_err）。
    qa0 = np.array([float(robot.get_joint_positions()[idx[j]]) for j in ARM])
    R_fk0 = fk_tcp(qa0, PARK)[:3, :3]
    _, R_st0, _ = world_T(prims['link_tcp'])
    rot_err0 = float(np.degrees(np.arccos(np.clip(
        (np.trace(R_st0.T @ R_fk0) - 1) / 2, -1, 1))))
    chk = {'rows': force_rows, 'row_used': j6_row,
           'f_world_static': [float(v) for v in fw0],
           'norm_N': float(np.linalg.norm(fw0)),
           'distal_weight_N': exp_w,
           'offset_N': float(np.linalg.norm(fw0)) - exp_w,
           'rot_conv_err_deg_at_start': rot_err0}
    print(f'[drawer] 力：列 {j6_row}/{force_rows}，靜止 |F| '
          f'{np.linalg.norm(fw0):.4f} N，joint6 以下重量 {exp_w:.4f} N，'
          f'偏置 {chk["offset_N"]:+.4f} N', flush=True)
    print(f'[drawer] 旋轉慣例：stage 與 FK 的 TCP 姿態差 {rot_err0:.4f}°（起始姿態）',
          flush=True)

    print('[drawer] 啟動 ROS 節點 ...', flush=True)
    rclpy.init()
    node = DrawerNode()
    from rclpy.executors import SingleThreadedExecutor
    ex = SingleThreadedExecutor(); ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    log, events, stop_reason = [], [], 'sim_limit'
    prev_ov, rot_err_max, model_err_max = None, 0.0, 0.0
    f_over_n = 0                       # 連續超過門檻的樣本數
    F_SUSTAIN = int(round(float(FRC.get('abort_sustained_s', 0.0))
                          / a.physics_dt))
    f_peak, f_peak_t, f_peak_ph = 0.0, None, None
    DR_M = float(SPEC['drawer']['physics']['mass_kg'])
    DR_D = float(SPEC['drawer']['physics']['linear_damping'])
    applied_seq = None
    frozen_q = None
    coupled = False
    hold_ok_t = None
    arrived_t = None
    phase_t0 = {'': 0.0}
    last_phase = None
    w0 = time.monotonic()
    nxt = time.monotonic()
    print('[drawer] 進入物理主迴圈', flush=True)
    dist_mass = 0.25 + 2 * 0.0163      # 夾爪殼 + 兩指，用於座標慣例自我核對

    while True:
        world.step(render=False)
        t = float(world.current_time)
        sec = int(t); nsec = int(round((t - sec) * 1e9))
        c = Clock(); c.clock.sec = sec; c.clock.nanosec = min(nsec, 999999999)
        node.clock_pub.publish(c)

        # 底盤固定：每一步歸零速度並回寫位姿
        robot.set_linear_velocity(np.zeros(3)); robot.set_angular_velocity(np.zeros(3))

        ph = node.phase
        if ph != last_phase:
            phase_t0[ph] = t; last_phase = ph

        # --- 事件（在套用設定點之前處理，這樣連接就發生在該時刻的物理步）---
        while node.pending:
            e = node.pending.pop(0)
            ev = e['event']
            dpa, _ = drawer_v.get_world_poses()
            tcp_a, _, _ = world_T(prims['link_tcp'])
            info = {'event': ev, 'sim_t': t, 'seq': e.get('seq'),
                    'drawer_y_before': float(dpa[0][1]),
                    'tcp_before': tcp_a.tolist()}
            if ev == 'engage':
                if GRASP_MODEL == 'fixed_attachment':
                    info['local_pos1'] = attach()
                    coupled = True
                else:
                    coupled = True      # friction：靠手指命令，不建關節
            elif ev == 'release':
                info['detached'] = detach() if GRASP_MODEL == 'fixed_attachment' else True
                coupled = False
            world.step(render=False)    # 讓連接／解除在下一步生效後再量
            dpb, _ = drawer_v.get_world_poses()
            tcp_b, _, _ = world_T(prims['link_tcp'])
            info['drawer_y_after'] = float(dpb[0][1])
            info['drawer_jump_mm'] = (info['drawer_y_after']
                                      - info['drawer_y_before']) * 1000.0
            info['tcp_jump_mm'] = float(np.linalg.norm(tcp_b - tcp_a)) * 1000.0
            events.append(info)
            node.say({'event_applied': ev, 'sim_t': t,
                      'drawer_jump_mm': info['drawer_jump_mm'],
                      'tcp_jump_mm': info['tcp_jump_mm']})
            print(f'[drawer] 事件 {ev} @ sim {t:.3f}  抽屜跳動 '
                  f'{info["drawer_jump_mm"]:+.4f} mm  TCP 跳動 '
                  f'{info["tcp_jump_mm"]:+.4f} mm', flush=True)

        # --- 套用設定點 ---
        snap = node.snap
        if frozen_q is not None:
            tgt = robot.get_joint_positions()
            for k, j in enumerate(ARM):
                tgt[idx[j]] = frozen_q[k]
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=tgt))
        elif snap is not None:
            applied_seq, _, q_cmd = snap
            tgt = robot.get_joint_positions()
            for k, j in enumerate(ARM):
                tgt[idx[j]] = q_cmd[k]
            fs = node.fsnap
            if fs is not None:
                for j in FJ:
                    tgt[idx[j]] = fs[1]
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=tgt))

        # --- 量測 ---
        qm = robot.get_joint_positions()
        qa = np.array([float(qm[idx[j]]) for j in ARM])
        tcp_p, tcp_R, tcp_q = world_T(prims['link_tcp'])
        _, l6_R, _ = world_T(prims['link6'])
        dp, _ = drawer_v.get_world_poses()
        dv = drawer_v.get_velocities()[0]
        opening = DY0 - float(dp[0][1])
        opening_v = -float(dv[1])
        # 夾持關係：TCP 對「當下開度下的預定夾持點」的偏差
        exp_tcp = DA.grasp_tcp_world(SPEC, POSE, opening)
        slip = tcp_p - exp_tcp
        slip_n = float(np.linalg.norm(slip))
        try:
            F = np.array(robot.get_measured_joint_forces())
            f_local = F[j6_row][:3]
        except Exception:
            f_local = np.zeros(3)
        f_world = l6_R @ f_local
        f_norm = float(np.linalg.norm(f_world))
        f_pull = float(AXIS @ f_world)
        fc = np.zeros(3)
        if finger_v is not None:
            try:
                M = np.array(finger_v.get_contact_force_matrix(dt=a.physics_dt))
                fc = M.reshape(-1, 3).sum(axis=0)
            except Exception:
                pass
        fc_n = float(np.linalg.norm(fc))
        # 抽屜側的拉力估計：F = m·a + m·d·v（m、d 都是已讀回核對過的資產參數）。
        # 這一項**不依賴 articulation 的列號對應**，所以拿它當主要的拉力量測，
        # 手腕反作用力當交叉核對。兩者相符是旁證，不等於逐筆對應。
        a_dr = (opening_v - prev_ov) / a.physics_dt if prev_ov is not None else 0.0
        prev_ov = opening_v
        f_drawer = DR_M * a_dr + DR_M * DR_D * opening_v
        # 旋轉慣例的持續比對：運動中 R 與 R^T 會分開，差角應保持在 0 附近
        T_fk = fk_tcp(qa, PARK)
        R_fk = T_fk[:3, :3]
        rot_err = float(np.degrees(np.arccos(np.clip(
            (np.trace(tcp_R.T @ R_fk) - 1) / 2, -1, 1))))
        rot_err_max = max(rot_err_max, rot_err)
        # 同一組關節角下，Isaac 的 TCP 與離線 FK 的 TCP 差多少。
        # 這一項是**模型差異**（omni_bot_manip.urdf vs 展開的 9-DOF 檔），
        # 與「命令還沒追上」的動態落後是兩回事，不能混在 slip 裡一起看。
        model_err = float(np.linalg.norm(tcp_p - T_fk[:3, 3]))
        model_err_max = max(model_err_max, model_err)
        # 命令對應的 TCP（把當下的命令角送進同一個 FK）與實際 TCP 的差 = 追蹤落後
        lag = (float(np.linalg.norm(tcp_p - fk_tcp(snap[2], PARK)[:3, 3]))
               if snap is not None else 0.0)
        bp, bq = robot.get_world_pose()
        drift = math.hypot(float(bp[0]) - BASE0[0], float(bp[1]) - BASE0[1])
        dyaw = abs(yaw_of(bq) - BASE0[2])
        trk = float(np.abs(qa - (snap[2] if snap is not None else qa)).max()) \
            if snap is not None else 0.0
        lm = min(min(qa[k] - LITE6_SAFE.lower[k], LITE6_SAFE.upper[k] - qa[k])
                 for k in range(6))

        node.farr(node.dr_pub, [t, opening, opening_v])
        node.farr(node.gr_pub, [t, slip[0], slip[1], slip[2], slip_n,
                                1.0 if coupled else 0.0])
        node.farr(node.ct_pub, [t, f_world[0], f_world[1], f_world[2], f_norm,
                                f_pull, fc[0], fc[1], fc[2], fc_n, f_drawer])
        js = JointState(); js.header.stamp.sec = sec
        js.header.stamp.nanosec = min(nsec, 999999999)
        js.name = names; js.position = [float(v) for v in qm]
        node.js_pub.publish(js)
        ps = PoseStamped(); ps.header.stamp = js.header.stamp
        ps.header.frame_id = 'world'
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = [float(v) for v in tcp_p]
        ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, \
            ps.pose.orientation.z = [float(v) for v in tcp_q]
        node.tcp_pub.publish(ps)

        log.append([round(t, 4), ph, applied_seq, round(opening, 6),
                    round(opening_v, 5), round(slip_n, 6), round(f_norm, 4),
                    round(f_pull, 4), round(fc_n, 4), round(f_drawer, 5),
                    round(rot_err, 5), round(model_err, 6), round(lag, 6),
                    round(drift, 5),
                    round(math.degrees(dyaw), 4), round(trk, 5), round(lm, 5)]
                   + [round(float(v), 6) for v in qa])

        # --- 到位判定 ---
        if abs(opening - TARGET) <= float(TOL['opening_m']):
            if hold_ok_t is None:
                hold_ok_t = t
            elif arrived_t is None and t - hold_ok_t >= float(TOL['opening_hold_s']):
                arrived_t = t
                print(f'[drawer] 開度到位並保持 {TOL["opening_hold_s"]:.1f} s @ sim {t:.3f}',
                      flush=True)
        else:
            hold_ok_t = None

        # --- 停止條件 ---
        if f_norm > f_peak:
            f_peak, f_peak_t, f_peak_ph = f_norm, t, ph
        if f_norm > float(FRC['abort_threshold_n']):
            f_over_n += 1
        else:
            f_over_n = 0
        stop = None
        # 門檻 30 N 未變；要**連續**超過 abort_sustained_s 才中止，
        # 這樣建立約束的單步暫態不會被當成過載（見案例設定的說明）。
        if f_over_n > F_SUSTAIN:
            stop = 'contact_force'
        elif drift > float(TOL['base_drift_m']):
            stop = 'base_drift'
        elif math.degrees(dyaw) > float(TOL['base_yaw_drift_deg']):
            stop = 'base_drift'
        elif lm < float(TOL['joint_limit_margin_rad']):
            stop = 'joint_limit_margin'
        elif trk > float(TOL['joint_track_err_rad']):
            stop = 'joint_track_err'
        elif coupled and slip_n > float(TOL['grasp_slip_m']):
            stop = 'grasp_lost'
        elif opening > float(SPEC['drawer']['joint']['upper']) - 0.005:
            stop = 'drawer_limit'
        elif ph in TMO and t - phase_t0.get(ph, t) > float(TMO[ph]):
            stop = 'phase_timeout'
        elif t > a.sim_limit:
            stop = 'sim_limit'
        elif time.monotonic() - w0 > a.wall_limit:
            stop = 'wall_timeout'
        elif len(log) % 50 == 0:
            tc, _ = cpu_temp_read()
            if tc is not None and tc >= CPU_LIMIT:
                stop = 'cpu_temp'

        if stop is not None:
            handling = CASE['stop_handling'].get(
                stop, CASE['stop_handling']['default'])
            print(f'[drawer] 停止：{stop}（處置 {handling}）@ sim {t:.3f}', flush=True)
            if handling == 'release_coupling_then_freeze':
                # 凍結設定點**不會**卸力：位置驅動會持續施力。先解除耦合。
                if GRASP_MODEL == 'fixed_attachment':
                    detach()
                coupled = False
                if snap is not None:
                    tgt = robot.get_joint_positions()
                    for j in FJ:
                        tgt[idx[j]] = F_OPEN
                    robot.get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=tgt))
            frozen_q = qa.copy()
            stop_reason = stop
            node.say({'stop': stop, 'handling': handling, 'sim_t': t})
            for _ in range(20):
                world.step(render=False)
            break

        if a.rtf > 0:
            nxt += a.physics_dt / a.rtf
            sl = nxt - time.monotonic()
            if sl > 0:
                time.sleep(sl)
            else:
                nxt = time.monotonic()

    tc, tsrc = cpu_temp_read()
    dp, _ = drawer_v.get_world_poses()
    final_open = DY0 - float(dp[0][1])
    tcp_p, _, _ = world_T(prims['link_tcp'])
    out = {
        'schema': 'drawer_run/1', 'case': a.case, 'grasp_model': GRASP_MODEL,
        'sha256_16': SHA, 'physics_dt': a.physics_dt, 'rtf': a.rtf,
        'park': list(PARK), 'pose': list(POSE), 'drawer_y0': DY0,
        'authored': authored,
        'force_measure': {'row': j6_row, 'row_source': j6_src,
                          'joint_names': jnames,
                          'frame': 'link6 世界旋轉 @ 子連桿座標系的反作用力',
                          'axis_world': AXIS.tolist(),
                          'distal_mass_kg': dist_mass,
                          'distal_weight_N': dist_mass * 9.81,
                          'static_check': chk,
                          'note': ('F_pull 未扣除夾爪慣性項，是「手腕沿抽屜軸傳遞'
                                   '的力」，不是純把手拉力')},
        'f_norm_peak': {'N': f_peak, 'sim_t': f_peak_t, 'phase': f_peak_ph,
                        'threshold_N': float(FRC['abort_threshold_n']),
                        'sustain_samples_required': F_SUSTAIN},
        'events': events, 'stop_reason': stop_reason,
        'sim_time_s': float(world.current_time),
        'wall_s': time.monotonic() - w0,
        'final_opening_m': final_open,
        'opening_err_m': final_open - TARGET,
        'arrived_sim_t': arrived_t,
        'tcp_final_world': tcp_p.tolist(),
        'cb': {'arm': node.arm_cb_n, 'arm_rejected': node.arm_cb_rej,
               'gripper': node.g_cb_n, 'phase': node.p_cb_n},
        'cpu_temp_c': tc, 'cpu_temp_source': tsrc, 'cpu_limit_c': CPU_LIMIT,
        'log_cols': ['t', 'phase', 'applied_seq', 'opening', 'opening_v',
                     'slip', 'f_norm', 'f_pull', 'fc_norm', 'f_drawer',
                     'rot_conv_err_deg', 'model_err', 'cmd_lag', 'base_drift',
                     'base_dyaw_deg', 'track_err', 'limit_margin'] + ARM,
        'rot_conv_err_max_deg': rot_err_max,
        'model_err_max_m': model_err_max,
        'log': log,
    }
    json.dump(out, open(os.path.join(a.out, 'drawer_run.json'), 'w'),
              ensure_ascii=False)
    print(f'[drawer] 結束 {stop_reason}  最終開度 {final_open*1000:.2f} mm '
          f'(誤差 {(final_open-TARGET)*1000:+.2f} mm)  CPU {tc} °C（{tsrc}）')
    print(f'[drawer] -> {os.path.join(a.out, "drawer_run.json")}')
    node.destroy_node(); rclpy.shutdown()
    return 0


rc = 1
try:
    rc = main()
except Exception:
    import traceback
    tb = traceback.format_exc()
    print(tb, flush=True)
    try:
        open(os.path.join(a.out, 'traceback.txt'), 'w').write(tb)
    except Exception:
        pass
    rc = 9
finally:
    sim_app.close()
sys.exit(rc)
