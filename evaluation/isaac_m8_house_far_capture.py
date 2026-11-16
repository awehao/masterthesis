#!/usr/bin/env python3
"""遠距實際取像執行器（Codex 20261116：原導航避障 → 移動進線 → 觀察追蹤 → 持續取像 → 受控停止；不閉爪、不綁定、不開門）。

本程式只做「模擬器＋感測＋受控對象介面」，不含任何導航或任務決策：
  場景＝住宅 m8_house_r1（同 isaac_m8_house_door_approach.py 的建法：牆、家具、門、未知靜物、機器人、腕部相機）。
  ROS 介面（與乾跑世界節點 far_dry/world_node.py 同一組話題，導航鏈與監督節點不用改）：
    出：/clock、/odom_raw（物理位姿，frame odom；＝理想里程，照實標示）、/joint_states、/scan_raw（PhysX 射線 360 道 10 Hz，略過機器人自身形狀）、
        /far/truth（t、q9、設定點 6、在途末端設定點 6、8 步在途命令 8×9）、/far/handle_obs（每格處理過的影像的估計結果）、/far/exec_state
    入：/cmd_vel（GUARD 放行的底盤命令）、/far/arm_cmd（六關節速率命令）、/far/done（監督節點結束）
  命令到受控對象經 8 步在途管線（同門任務控制器的延遲模型）；手臂＝速率命令積分成設定點、位置驅動追設定點。
取像：腕部相機每個新渲染影格都核對「渲染時刻的狀態紀錄」；依 --cam-hz 取用，逐格保存原始深度、彩色、內參、
  相機外參（渲染時刻關節／底盤讀值的 FK）、擷取同刻的里程位姿／關節／最新 map←odom（訂閱 /tf）、偵測與估計的成功或失敗原因。
  估計以「相對擷取同刻底盤」的位姿發布（相機＋FK 給得出的量，不含世界定位）。
保護：非預期接觸（>0.05 N）、門角變化（>0.05°）、手指張開量偏離模型 ⇒ 鎖存：底盤命令歸零、手臂設定點凍結，並在 /far/exec_state 發 latched（監督節點據此全域停止）。
真值（把手、門角）只寫進紀錄供事後評估，不進任何發布的量。"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--reg', default=os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json'))
ap.add_argument('--urdf', default=os.path.join(WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--out', required=True)
ap.add_argument('--start', default='1.5,-2.8,1.5708')
ap.add_argument('--arm', default='-0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018', help='起始手臂構型（預設＝online_1 far_live 影格的構型）')
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--settle-s', type=float, default=1.0)
ap.add_argument('--sim-limit-s', type=float, default=150.0)
ap.add_argument('--cam-hz', type=float, default=5.0, help='取用影格的頻率（偵測＋保存）；相機本身 30 Hz')
ap.add_argument('--scan-hz', type=float, default=10.0)
ap.add_argument('--static-s', type=float, default=0.0, help='>0：靜態取像冒煙——不接受任何運動命令，取像 N 秒後結束')
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': False, 'width': 1280, 'height': 720})
import numpy as np                                               # noqa: E402
import rclpy                                                     # noqa: E402  先匯入真正的 rclpy（部分專案模組會在缺 rclpy 時裝替身）
from geometry_msgs.msg import Twist                              # noqa: E402
from nav_msgs.msg import Odometry                                # noqa: E402
from rclpy.node import Node                                      # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy              # noqa: E402
from rosgraph_msgs.msg import Clock                              # noqa: E402
from sensor_msgs.msg import JointState, LaserScan                # noqa: E402
from std_msgs.msg import Bool, Float64MultiArray, String         # noqa: E402
from tf2_msgs.msg import TFMessage                               # noqa: E402
_RCLPY_REAL = rclpy
_ROS_PREFIX = ('rclpy', 'std_msgs', 'geometry_msgs', 'nav_msgs', 'sensor_msgs', 'tf2_msgs', 'tf2_ros', 'rosgraph_msgs', 'builtin_interfaces', 'rcl_interfaces')
_ROS_SNAP = {k_: v_ for k_, v_ in sys.modules.items() if k_.split('.')[0] in _ROS_PREFIX}
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import FixedCuboid, GroundPlane   # noqa: E402
from isaacsim.core.prims import RigidPrim, SingleArticulation    # noqa: E402
from isaacsim.core.utils.types import ArticulationAction         # noqa: E402
from isaacsim.sensors.camera import Camera                       # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics     # noqa: E402

import door_asset_v3 as DA3                                      # noqa: E402
import door_far_link as FL                                       # noqa: E402
import door_motm_grasp_ctrl as G                                 # noqa: E402
import wrist_v0_capture as W0                                    # noqa: E402
from finger_collision import apply_split                         # noqa: E402
from isaac_common import bind_frictionless, collision_prims, import_urdf, physics_parts   # noqa: E402

# 專案控制模組（ros_stub）會無條件把 sys.modules 裡的 rclpy 與訊息套件換成替身（它們離線用）。這裡在匯入後把真正的模組放回去：
# 專案模組已拿到的替身參照不受影響（它們不經 ROS 通訊），本程式的 ROS 節點用的是真正的 rclpy。
for k_ in [k_ for k_ in sys.modules if k_.split('.')[0] in _ROS_PREFIX and k_ not in _ROS_SNAP]:
    sys.modules.pop(k_, None)
sys.modules.update(_ROS_SNAP)
if sys.modules.get('rclpy') is not _RCLPY_REAL or not hasattr(rclpy, 'init') or getattr(sys.modules.get('rclpy.node'), 'Node', None) is not Node:
    raise SystemExit('rclpy 還原失敗 ⇒ 拒跑')
K = G.K
ROBOT, DOOR = '/World/omni_bot', '/World/door'
ARM = [f'joint{i}' for i in range(1, 7)]
FINGER_OPEN = float(G.XO.FINGER_OPEN)
FINGER_TOL_M = 0.0005
DT, D_STEPS = 0.01, 8
CONTACT_N, DOOR_DEG = 0.05, 0.05
CMD_TIMEOUT_S, ARM_CMD_TIMEOUT_S = 0.5, 0.2                       # 同乾跑世界節點
WALL_LIMIT_S = 1500.0
CAM = dict(res=(640, 480), hfov_deg=69.0, aperture=0.020, clip=(0.05, 10.0), hz=30)
SCAN_N, SCAN_MIN, SCAN_MAX, RANGE_MIN, RANGE_MAX = 360, -3.14159, 3.14159, 0.12, 10.0   # 同乾跑世界節點
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
LIDAR_Z = float(json.load(open(os.path.join(HERE, 'far_dry', 'geom.json')))['lidar_z'])
LO, HI = np.asarray(G.XO.LITE6_SAFE.lower, float), np.asarray(G.XO.LITE6_SAFE.upper, float)


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def T_of(pos, qwxyz):
    T = np.eye(4)
    T[:3, :3] = W0.wxyz_to_R(np.asarray(qwxyz, float))
    T[:3, 3] = np.asarray(pos, float)
    return T


def T3(p):
    c, s = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = p[:2]
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


class Bridge(Node):
    def __init__(self):
        super().__init__('far_capture_sim')
        be = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.p_clock = self.create_publisher(Clock, '/clock', 10)
        self.p_scan = self.create_publisher(LaserScan, '/scan_raw', be)
        self.p_odom = self.create_publisher(Odometry, '/odom_raw', 10)
        self.p_js = self.create_publisher(JointState, '/joint_states', 10)
        self.p_truth = self.create_publisher(Float64MultiArray, '/far/truth', 10)
        self.p_obs = self.create_publisher(Float64MultiArray, '/far/handle_obs', 10)
        self.p_state = self.create_publisher(String, '/far/exec_state', 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
        self.create_subscription(Float64MultiArray, '/far/arm_cmd', self._acmd, 10)
        self.create_subscription(Bool, '/far/done', self._done, 10)
        self.create_subscription(TFMessage, '/tf', self._tf, 50)
        self.t = 0.0
        self.cmd, self.cmd_t, self.acmd, self.acmd_t, self.done = np.zeros(3), -1.0, np.zeros(6), -1.0, False
        self.map_odom = None                                           # (stamp, x, y, yaw)：最新 map←odom（只記錄）
        self.n_cmd, self.n_acmd = 0, 0

    def _cmd(self, m):
        v = np.array([m.linear.x, m.linear.y, m.angular.z], float)
        if np.isfinite(v).all():
            self.cmd, self.cmd_t = v, self.t
            self.n_cmd += 1

    def _acmd(self, m):
        if len(m.data) == 6 and np.isfinite(m.data).all():
            self.acmd, self.acmd_t = np.array(m.data, float), self.t
            self.n_acmd += 1

    def _done(self, m):
        self.done = self.done or bool(m.data)

    def _tf(self, m):
        for tr in m.transforms:
            if tr.header.frame_id == 'map' and tr.child_frame_id == 'odom':
                o = tr.transform.rotation
                self.map_odom = (tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9, float(tr.transform.translation.x), float(tr.transform.translation.y),
                                 math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z)))

    def stamp(self):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec = int(self.t)
        s.nanosec = int(round((self.t - int(self.t)) * 1e9))
        if s.nanosec >= 1000000000:
            s.sec, s.nanosec = s.sec + 1, s.nanosec - 1000000000
        return s


def main():
    os.makedirs(a.out, exist_ok=True)
    fdir = os.path.join(a.out, 'frames')
    os.makedirs(fdir, exist_ok=True)
    reg = json.load(open(a.reg))
    start_b = np.array([float(x) for x in a.start.split(',')], float)
    start_q = np.array([float(x) for x in a.arm.split(',')], float)
    static = a.static_s > 0

    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)
    scn = [pr for pr in stage.Traverse() if pr.IsA(UsdPhysics.Scene)]
    if len(scn) != 1:
        print(f'[fc] 物理場景數 {len(scn)} ≠ 1 ⇒ 拒跑', flush=True)
        return 67
    sapi = PhysxSchema.PhysxSceneAPI(scn[0]) if scn[0].HasAPI(PhysxSchema.PhysxSceneAPI) else PhysxSchema.PhysxSceneAPI.Apply(scn[0])
    sapi.CreateEnableExternalForcesEveryIterationAttr().Set(True)
    env_paths = []
    for o in reg['walls'] + reg['furniture']:
        FixedCuboid(prim_path=f"/World/{o['name']}", name=o['name'], position=np.array(o['c']), scale=np.array(o['s']), color=np.array(o['rgb']))
        env_paths.append(f"/World/{o['name']}")
    dauth = DA3.build_usd(stage, DA3.load(os.path.join(WS, reg['door']['asset'])), root=DOOR)
    ob = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
    for o in ob['static_unknown']:                                    # 未知靜物照場景登錄放；本趟有 LiDAR 與原導航避障
        y2 = math.radians(o['yaw_deg']) / 2
        FixedCuboid(prim_path=f"/World/unknown/{o['name']}", name=o['name'], position=np.array([o['x'], o['y'], o['size'][2] / 2]), scale=np.array(o['size']),
                    orientation=np.array([math.cos(y2), 0.0, 0.0, math.sin(y2)]), color=np.array([0.72, 0.52, 0.28]))
        env_paths.append(f"/World/unknown/{o['name']}")
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(start_b[0]), float(start_b[1]), 0.0))
    xf.AddRotateZOp().Set(float(math.degrees(start_b[2])))
    apply_split(stage, ROBOT)
    sph = collision_prims(stage, ROBOT, lambda p: 'support_' in p)
    if bind_frictionless(stage, sph + collision_prims(stage, '/World/ground')) == 0:
        print('[fc] 零摩擦材質沒綁到 ⇒ 拒跑', flush=True)
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
        print(f'[fc] 剛體不符：{[name_of(b) for b in tool_bodies]}／base {b_base}／arts {arts} ⇒ 拒跑', flush=True)
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
        print(f'[fc] 缺關節 {names} ⇒ 拒跑', flush=True)
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
        print(f'[fc] 把手 prim 數 {len(hprims)} ≠ 1 ⇒ 拒跑', flush=True)
        return 22

    def truth_handle():                                               # 只供事後評估
        M = UsdGeom.XformCache().GetLocalToWorldTransform(hprims[0])
        return np.array([[M[i][j] for j in range(4)] for i in range(4)]).T
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
        set_camera_view(eye=np.array([3.4, -4.2, 3.2]), target=np.array([1.0, -0.8, 0.6]))
    except Exception as e:                                            # noqa: BLE001
        print(f'[fc] 視角設定失敗（不影響物理）：{e!r}', flush=True)
    readback = dict(physics_dt=float(world.get_physics_dt()), ext_forces_every_iter=bool(sapi.GetEnableExternalForcesEveryIterationAttr().Get()), kp=a.kp, kd=a.kd, start_base=start_b.tolist(), start_q=start_q.tolist(),
                    finger_open=FINGER_OPEN, static_s=a.static_s, cam_hz=a.cam_hz, d_steps=D_STEPS)
    print(f'[fc] 讀回 {readback}', flush=True)
    if abs(readback['physics_dt'] - DT) > 1e-12 or not readback['ext_forces_every_iter']:
        print('[fc] 物理設定讀回不符 ⇒ 拒跑', flush=True)
        return 64
    # 射線查詢（LiDAR）：略過機器人自身的所有形狀（乾跑世界節點同樣不含機器人自身；下游 scan_relay 另有自體扇區遮罩）
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()

    def scan(q):
        org = [float(q[0]), float(q[1]), LIDAR_Z]
        out, who_min = [], (float('inf'), None)
        for i in range(SCAN_N):
            th = float(q[2]) + SCAN_MIN + i * SCAN_INC
            best = [float('inf'), None]

            def rep(hit, best=best):
                coll = str(getattr(hit, 'collision', '') or '')
                body = str(getattr(hit, 'rigid_body', '') or getattr(hit, 'rigidBody', '') or '')
                if coll.startswith(ROBOT) or body.startswith(ROBOT):
                    return True
                d = float(getattr(hit, 'distance', 0.0))
                if RANGE_MIN <= d < best[0]:
                    best[0], best[1] = d, coll or body
                return True
            query.raycast_all(org, [math.cos(th), math.sin(th), 0.0], RANGE_MAX, rep)
            out.append(best[0])
            if best[0] < who_min[0]:
                who_min = (best[0], best[1])
        return out, who_min

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
    if abs(float(jp_s[f1]) - FINGER_OPEN) > FINGER_TOL_M or abs(float(jp_s[f2]) - FINGER_OPEN) > FINGER_TOL_M:
        print(f'[fc] 手指張開量 {float(jp_s[f1])}/{float(jp_s[f2])} 與模型 {FINGER_OPEN} 不符 ⇒ 拒跑', flush=True)
        return 65
    door_q0 = door_q()
    r0, who0 = scan(q_s)
    truth = dict(note='只供事後評估；不進任何發布的量', T_handle_start=truth_handle().tolist(), door_q0_rad=door_q0)
    print(f'[fc] 起點讀回 底盤 {np.round(q_s[:3], 4).tolist()} 手臂靜態偏移 {np.round(q_s[3:] - start_q, 4).tolist()} 門角 {math.degrees(door_q0):.4f}° 掃描最近 {who0}', flush=True)
    if not np.isfinite(who0[0]):
        print('[fc] 掃描沒有任何回波 ⇒ 拒跑', flush=True)
        return 68

    rclpy.init()
    node = Bridge()
    obs = FL.HandleObs()
    T0 = float(world.current_time)
    hist, frames, rows, events = {}, [], [], []
    cam_stat = dict(read=0, new=0, used=0, rejects={}, valid=0, fail={}, detect_ms=[])
    last_rf, t_next_cam, seq = [None], 0.0, 0
    s = q_s[3:].copy()                                                # 手臂設定點
    pipe = [np.zeros(9) for _ in range(D_STEPS)]
    latch, contact_max, clock_bad, step_wall = None, dict(n=0.0, who=None), 0, []
    w_all0 = time.perf_counter()
    t_done, scan_min_log = None, float('nan')
    n_steps = int(round((a.static_s if static else a.sim_limit_s) / DT))
    scan_every, pub_every = max(int(round(1.0 / (a.scan_hz * DT))), 1), 2

    def ev(what, **kw):
        events.append(dict(t=round(node.t, 3), what=what, **kw))
        print(f'[fc] t={node.t:.2f} {what} {kw}', flush=True)

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

    def grab():
        fr = cam.get_current_frame()
        cam_stat['read'] += 1
        rfd = fr.get('rendering_frame')
        rf = (rfd.get('referenceTimeNumerator'), rfd.get('referenceTimeDenominator')) if isinstance(rfd, dict) else rfd
        col = fr.get('rgb') if fr.get('rgb') is not None else fr.get('rgba')
        if rf is None or rf == last_rf[0] or rf == (0, 0) or col is None or fr.get('distance_to_image_plane') is None:
            return None
        last_rf[0] = rf
        cam_stat['new'] += 1

        def rej(why):
            cam_stat['rejects'][why] = cam_stat['rejects'].get(why, 0) + 1
        t_r = fr.get('rendering_time')
        if t_r is None:
            return rej('no_rendering_time')
        hp = hist.get(int(round(float(t_r) / DT)))
        if hp is None or abs(hp['t_sim'] - float(t_r)) > 0.25 * DT:
            return rej('no_state_at_render_time')
        dep = np.asarray(fr['distance_to_image_plane'], dtype=np.float32)
        rgba = np.asarray(col)
        if dep.size == 0 or rgba.size == 0:
            return rej('empty_image')
        T_fk = K.fk(hp['q'], W0.OPT)
        return dict(t_obs=float(t_r) - T0, dep=dep[:, :, 0] if dep.ndim == 3 else dep, rgb=rgba[:, :, :3].astype(np.uint8), T_cam=T_fk, q=hp['q'], rf=rf,
                    fk_vs_render_m=float(np.linalg.norm(T_fk[:3, 3] - hp['T_render'][:3, 3])), map_odom=hp['map_odom'], s=hp['s'])

    for k in range(n_steps):
        w_step0 = time.perf_counter()
        t_sim = float(world.current_time)
        node.t = t = t_sim - T0
        q, jp = q9()
        pos_r, q_r = cam.get_world_pose(camera_axes='ros')
        hist[int(round(t_sim / DT))] = dict(t_sim=t_sim, q=q.copy(), T_render=T_of(pos_r, q_r), map_odom=node.map_odom, s=s.copy())
        if len(hist) > 300:
            for kk in sorted(hist)[:len(hist) - 300]:
                hist.pop(kk, None)
        # ---- 保護
        cn, cwho = contacts()
        dq_deg = abs(math.degrees(door_q() - door_q0))
        why = None
        if cn is None:
            why = ('contact_read_fail', dict(detail=cwho))
        else:
            if cn > contact_max['n']:
                contact_max.update(n=cn, who=cwho)
            if cn > CONTACT_N:
                why = ('unexpected_contact', dict(force_n=round(cn, 4), who=cwho))
        if why is None and dq_deg > DOOR_DEG:
            why = ('door_moved', dict(door_deg=round(dq_deg, 4)))
        if why is None and max(abs(float(jp[f1]) - FINGER_OPEN), abs(float(jp[f2]) - FINGER_OPEN)) > FINGER_TOL_M:
            why = ('finger_opening_out_of_model', dict(q1=float(jp[f1]), q2=float(jp[f2])))
        if why is not None and latch is None:
            latch = dict(t=round(t, 3), why=why[0], **why[1])
            ev('latched', **latch)
        # ---- ROS 收命令
        for _ in range(4):
            rclpy.spin_once(node, timeout_sec=0.0)
        u_in = node.cmd if (t - node.cmd_t <= CMD_TIMEOUT_S and node.cmd_t >= 0) else np.zeros(3)
        ua_in = node.acmd if (t - node.acmd_t <= ARM_CMD_TIMEOUT_S and node.acmd_t >= 0) else np.zeros(6)
        if static or latch is not None:
            u_in, ua_in = np.zeros(3), np.zeros(6)
            if latch is not None:
                pipe = [np.zeros(9) for _ in range(D_STEPS)]           # 鎖存：在途命令也清掉（立即停）
        pipe.append(np.r_[u_in, ua_in])
        ue = pipe.pop(0)
        s = np.clip(s + DT * ue[3:], LO, HI)
        for i, j in zip(idx, range(6)):
            tgt[i] = float(s[j])
        tgt[f1] = FINGER_OPEN
        apply(tgt)
        base_cmd(ue[:3], float(q[2]))
        # ---- 發布
        ck = Clock()
        ck.clock = node.stamp()
        node.p_clock.publish(ck)
        if k % pub_every == 0:
            m = Odometry()
            m.header.stamp = node.stamp()
            m.header.frame_id, m.child_frame_id = 'odom', 'base_footprint'
            m.pose.pose.position.x, m.pose.pose.position.y = float(q[0]), float(q[1])
            m.pose.pose.orientation.w, m.pose.pose.orientation.z = math.cos(q[2] / 2), math.sin(q[2] / 2)
            m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z = float(ue[0]), float(ue[1]), float(ue[2])
            node.p_odom.publish(m)
            js = JointState()
            js.header.stamp = node.stamp()
            js.name = ARM + ['finger_joint1', 'finger_joint2']
            js.position = [float(x) for x in q[3:]] + [float(jp[f1]), float(jp[f2])]
            node.p_js.publish(js)
            s_end = s + DT * sum(p_[3:] for p_ in pipe)
            node.p_truth.publish(Float64MultiArray(data=[t, *[float(x) for x in q], *[float(x) for x in s], *[float(x) for x in s_end], *[float(x) for x in np.concatenate(pipe)]]))
            node.p_state.publish(String(data='latched:' + latch['why'] if latch is not None else ('static' if static else 'run')))
        if k % scan_every == 0:
            r, whom = scan(q)
            m = LaserScan()
            m.header.stamp = node.stamp()
            m.header.frame_id = 'lidar_link'
            m.angle_min, m.angle_max, m.angle_increment = SCAN_MIN, SCAN_MAX, SCAN_INC
            m.range_min, m.range_max = RANGE_MIN, RANGE_MAX
            m.ranges = [float(x) if math.isfinite(x) else float('inf') for x in r]
            node.p_scan.publish(m)
            scan_min_log = float(whom[0])
        # ---- 相機：持續取像
        f = grab()
        if f is not None and f['t_obs'] >= t_next_cam - 1e-9:
            t_next_cam = f['t_obs'] + 1.0 / a.cam_hz
            seq += 1
            cam_stat['used'] += 1
            w0 = time.perf_counter()
            T_est, info = obs.intake(dict(dep=f['dep'], K=Kr, T_cam=f['T_cam'], t_obs=f['t_obs'], q=f['q'], t_state=f['t_obs'], pose_source='fk_at_render_time', frame_id='map', fk_vs_render_m=f['fk_vs_render_m']), t)
            ms = (time.perf_counter() - w0) * 1e3
            cam_stat['detect_ms'].append(ms)
            stem = f'f{seq:04d}'
            np.savez_compressed(os.path.join(fdir, f'{stem}_depth_m.npz'), dep=f['dep'])
            try:
                import imageio.v2 as imageio
                imageio.imwrite(os.path.join(fdir, f'{stem}_rgb.png'), f['rgb'])
            except Exception as e:                                    # noqa: BLE001
                cam_stat['rejects']['rgb_save_fail'] = cam_stat['rejects'].get('rgb_save_fail', 0) + 1
                print(f'[fc] 彩色影像保存失敗：{e!r}', flush=True)
            T_bh = None if T_est is None else np.linalg.inv(T3(f['q'][:3])) @ np.asarray(T_est, float)
            if T_est is None:
                key = str(info.get('why') or info.get('cls') or 'unknown')
                cam_stat['fail'][key] = cam_stat['fail'].get(key, 0) + 1
            else:
                cam_stat['valid'] += 1
            Th = truth_handle()
            frames.append(dict(seq=seq, stem=stem, t_obs=f['t_obs'], t_processed=t, age_s=round(t - f['t_obs'], 4), cam_T_fk=f['T_cam'].tolist(), q=[float(v) for v in f['q']], arm_setpoint=[float(v) for v in f['s']],
                               odom_base_at_capture=[float(v) for v in f['q'][:3]], map_odom_latest_at_capture=f['map_odom'], fk_vs_render_m=f['fk_vs_render_m'], detect_ms=round(ms, 1), valid=T_est is not None, info=info,
                               T_base_handle=None if T_bh is None else T_bh.tolist(), rendering_frame=[int(v) if v is not None else None for v in f['rf']] if isinstance(f['rf'], tuple) else f['rf'],
                               eval_only=dict(err_pos_mm=None if T_est is None else round(float(np.linalg.norm(np.asarray(T_est)[:3, 3] - Th[:3, 3])) * 1e3, 2), range_m=round(float(np.linalg.norm(f['T_cam'][:3, 3] - Th[:3, 3])), 3))))
            node.p_obs.publish(Float64MultiArray(data=[float(seq), float(f['t_obs']), 1.0 if T_est is not None else 0.0, *[float(v) for v in f['q'][:3]], *([float(v) for v in T_bh.reshape(-1)] if T_bh is not None else [float('nan')] * 16), float(t)]))
            if seq <= 3 or seq % 25 == 0:
                print(f"[fc] 影格 {seq} t_obs={f['t_obs']:.2f} valid={T_est is not None} {'' if T_est is not None else info.get('why') or info.get('cls')} 距離 {frames[-1]['eval_only']['range_m']} m 偵測 {ms:.0f} ms", flush=True)
        st0 = float(world.current_time)
        world.step(render=True)
        if abs((float(world.current_time) - st0) - DT) > 1e-9:
            clock_bad += 1
        sw = time.perf_counter() - w_step0
        step_wall.append(sw)
        rows.append([t, *q, *s, *ue, *u_in, *ua_in, 0.0 if cn is None else cn, dq_deg, scan_min_log, sw, float(latch is not None)])
        if node.done and t_done is None:
            t_done = t
            ev('supervisor_done')
        if t_done is not None and t - t_done >= 1.0:
            break
        if time.perf_counter() - w_all0 > WALL_LIMIT_S:
            ev('wall_limit')
            break
    wall = time.perf_counter() - w_all0
    cols = ['t'] + [f'q{i}' for i in range(9)] + [f's{i}' for i in range(6)] + ['ux', 'uy', 'uw'] + [f'ua{i}' for i in range(6)] + ['uin_x', 'uin_y', 'uin_w'] + [f'uain{i}' for i in range(6)] + ['contact_n', 'door_deg', 'scan_min', 'step_wall_s', 'latched']
    R = np.array(rows, float)
    np.savez_compressed(os.path.join(a.out, 'log.npz'), L=R, cols=np.array(cols))
    json.dump(clean(dict(K=Kr, cam=CAM, T_mount=T_mount, frames=frames)), open(os.path.join(a.out, 'frames_meta.json'), 'w'), ensure_ascii=False, indent=1)
    truth['T_handle_end'] = truth_handle().tolist()
    truth['door_q_end_rad'] = door_q()
    json.dump(clean(truth), open(os.path.join(a.out, 'truth.json'), 'w'), ensure_ascii=False, indent=1)
    dms = cam_stat.pop('detect_ms')
    res = dict(mode='static' if static else 'run', readback=readback, sim_s=round(float(R[-1, 0]), 2) if len(R) else 0.0, wall_s=round(wall, 1), rtf=round(float(R[-1, 0]) / wall, 3) if len(R) and wall > 0 else None,
               latch=latch, events=events, contact_max=contact_max, door_deg_max=round(float(R[:, cols.index('door_deg')].max()), 4) if len(R) else None, clock_bad=clock_bad,
               camera=dict(cam_stat, detect_ms_p50=round(float(np.median(dms)), 1) if dms else None, detect_ms_max=round(float(np.max(dms)), 1) if dms else None),
               cmds_received=dict(base=node.n_cmd, arm=node.n_acmd), base_travel_m=round(float(np.abs(np.diff(R[:, 1:3], axis=0)).sum()), 3) if len(R) > 1 else 0.0,
               sources=dict(odom='物理位姿直接當里程（理想，無滑移）', scan='PhysX 射線 360 道，略過機器人自身形狀，無雜訊', pose_for_camera='渲染時刻的關節／底盤讀值 FK', delay='8 步在途管線（軟體模型）'),
               step_wall_ms_p50=round(float(np.median(step_wall)) * 1e3, 1) if step_wall else None, supervisor_done=t_done is not None)
    json.dump(clean(res), open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"[fc] 結束 sim {res['sim_s']} s wall {res['wall_s']} s 影格 取用 {cam_stat['used']} 有效 {cam_stat['valid']} 失敗 {cam_stat['fail']} 鎖存 {latch}", flush=True)
    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:                                                 # noqa: BLE001
        pass
    return 0 if latch is None else 70


if __name__ == '__main__':
    rc = 1
    try:
        rc = main()
    finally:
        sim_app.close()
    sys.exit(rc)
