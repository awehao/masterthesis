#!/usr/bin/env python3
"""房間 ＋ 抽屜櫃 ＋ 機器人：**同時**提供導航介面與全身介面的執行端。

為什麼另立一支
--------------
`isaac_bigarena_sim.py` 有導航側（/cmd_vel、/scan_raw），
`isaac_wholebody_sim_e2.py` 有全身側（/wb_vel_cmd、命令鏈、抽屜讀值），
但**沒有一支同時有兩者**。本實驗一趟要先導航再全身，所以需要兩者並存。

E2 是階段 A 的執行端，其結果已封存 —— **不動它**，本檔重用共用模組。

誰在控制
--------
每個物理步**恰有一個控制者**，由 `dual_source_executor.DualSourceExecutor`
決定。導航三維命令直接寫進底盤（不經輪級限制與低速介面界限 —— 它的速度是
那個界限的 6 倍）；全身九維命令走 `CmdChainE2`。兩條路徑都產生套用回報。

控制權在**明確物理步**轉移，轉移當下由執行層把導航最後一筆真正套用的命令
承接給命令鏈。轉移前舊控制者一路持有控制權，中間沒有空窗。

介面
----
    /clock                 (out)  模擬時間
    /scan_raw              (out)  360 條 PhysX raycast，10 Hz，frame lidar_link
    /joint_states          (out)  六軸實測
    /odom                  (out)  底盤位姿真值（本實驗的明示前提）
    /cmd_vel               (in)   導航的本體三維速度
    /wb_vel_cmd            (in)   全身的九維命令
    /coman/applied_cmd     (out)  **本步真正套用**的九維命令（18 欄契約）
    /coman/arm_setpoint    (out)  手臂設定點
    /drawer/handle_pose    (out)  把手的**實測**世界位姿（USD 真值，4x4）
    /drawer/state          (out)  實測開度與把手位姿
    /handover/request      (in)   要求在指定物理步轉移控制權
    /handover/state        (out)  現任控制者、排程、承接內容
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from physics_contact_hold import PhysicsContactHold

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument('--world', default=os.path.join(
    WS, 'src/ammr_bringup/worlds/drawer_room.sdf'))
ap.add_argument('--drawer-asset', default=os.path.join(
    WS, 'src/my_omnibot_description/config/drawer_unit.yaml'))
ap.add_argument('--drawer-pose', default='0.0,1.45')
ap.add_argument('--urdf', default=os.path.join(
    WS, 'evaluation/models/omni_bot_manip.urdf'))
ap.add_argument('--start-pose', default='-2.80,-3.20,1.5708')
# ---- V0 腕部 RGB-D 擷取（預設關閉；開啟時只擺位＋算圖，不跑控制、不建既有 ROS 節點）----
ap.add_argument('--wrist-v0', action='store_true',
                help='V0：腕部相機擷取模式（evaluation/results/vision/V0_spec.yaml）')
ap.add_argument('--wrist-res', default='640x480')
ap.add_argument('--wrist-hfov', type=float, default=69.0)
ap.add_argument('--wrist-frames', type=int, default=20)
ap.add_argument('--wrist-warmup', type=int, default=60)
ap.add_argument('--wrist-max-steps', type=int, default=2000)
ap.add_argument('--wrist-hz', type=float, default=5.0,
                help='腕部相機取樣頻率（模擬時間）；存檔與發布都依此頻率')
ap.add_argument('--wrist-replay', default=None,
                help='重播既有實錄（room_run.json）的一段運動來擷取動態腕部影像（不跑控制）')
ap.add_argument('--wrist-replay-t0', type=float, default=0.0)
ap.add_argument('--wrist-replay-t1', type=float, default=1e9)
ap.add_argument('--wrist-base', default='-0.136412,0.560,1.297349',
                help='擺位底盤 x,y,yaw（預設＝基準停位）')
ap.add_argument('--wrist-q', default='-0.383712,0.301253,0.428917,-0.661155,-1.469862,-1.492557',
                help='擺位手臂六軸（預設＝基準夾持姿態的 TCP 沿 −y 退 0.20 m，離線 IK；'
                     '真值橫桿中心投影約 (375,354)、光軸距離 0.275 m）')
ap.add_argument('--stow-q', default='0,0,0,0,-1.5707963,0',
                help='導航期間手臂維持的收攏姿態（**已核准定版**）')
ap.add_argument('--physics-dt', type=float, default=0.01)
ap.add_argument('--sim-limit', type=float, default=240.0)
ap.add_argument('--scan-rate', type=float, default=10.0)
ap.add_argument('--state-rate', type=float, default=50.0)
ap.add_argument('--kp', type=float, default=600.0)
ap.add_argument('--kd', type=float, default=60.0)
ap.add_argument('--arm-rate-max', type=float, default=1.0)
ap.add_argument('--base-lin-max', type=float, default=0.05,
                help='**低速介面界限**（m/s）。只套用在**全身**那條路徑；'
                     '導航不經命令鏈，故不受它限制')
ap.add_argument('--base-ang-max', type=float, default=0.20)
ap.add_argument('--max-cmd-age-s', type=float, default=0.2)
ap.add_argument('--joint-margin', type=float, default=0.05)
ap.add_argument('--cmd-timeout', type=float, default=1.0,
                help='導航命令逾時（模擬秒）：超過就歸零，**失效處置**')
ap.add_argument('--out', default=os.path.join(WS, 'evaluation/runs/room_sim'))
ap.add_argument('--no-frictionless', action='store_true',
                help='診斷用：不綁零摩擦材質。**不綁會偏航** —— '
                     '底盤被直接設速度，支撐球與地面的摩擦抵抗橫向運動就會'
                     '產生淨力矩（實測 0.25 m/s 驅動 5 s 偏 21.1°）')
ap.add_argument('--v-box-lin', type=float, default=0.035255,
                help='切換當步要核的底盤速度框（逐軸）')
ap.add_argument('--v-box-ang', type=float, default=0.199900)
ap.add_argument('--v-min-lin', type=float, default=0.010,
                help='**滾動交棒下界**：切換當步底盤至少要這麼快。0 = 不要求')
# **PARK_FIXED（固定底盤操作）**：預設關 ⇒ 既有行為不變。開啟時執行端做靜止閘門（轉給全身前）與
# 保持監看（實際轉給全身 → 實際交還導航），並取消滾動交棒下界（MOTM 保持原值）。見 park_fixed.py。
ap.add_argument('--park-fixed', action='store_true')
ap.add_argument('--park-hold', action='store_true',
                help='PARK_HOLD（v2）：閘門／監看同 PARK_FIXED，保持期的套用命令改核停車伺服上界')
ap.add_argument('--park-pose', default='-0.136412,0.560,1.297349',
                help='PARK_FIXED 共同停位 x,y,yaw（入場容差的參考）')
ap.add_argument('--park-stow-tol', type=float, default=0.02,
                help='PARK_FIXED 閘門：手臂各軸距收攏姿態的上限（rad）')
# **底盤命令變化率上限**（逐軸）。既有的兩層都攔不住交棒當步的跳變：
# 低速介面界限只看命令大小，輪級 λ 的 α_max=125 rad/s² 允許 6.25 m/s²。
# 實測交棒當步跳 48.76 mm/s＝4.88 m/s²，λ 沒攔。這裡**套用既有值**
# a_base_lin 0.50 / a_base_ang 2.00（wgmpc_core.py 標為「開發值，
# 無已核准來源」，標籤照搬）。設 0 或負數 = 關閉。
ap.add_argument('--base-accel-lin', type=float, default=0.50,
                help='底盤線速度命令的逐軸變化率上限 [m/s²]；<=0 關閉')
ap.add_argument('--base-accel-ang', type=float, default=2.00,
                help='底盤偏航率命令的變化率上限 [rad/s²]；<=0 關閉')
# **導航直寫路徑的逐軸變化率上限。** 那條路徑不經命令鏈，鏈上三層都沒套到。
# 實測單一物理步逐軸跳 75.000 mm/s＝ax_max(1.5)×dt_cfg(0.05)，而 gmpc 自己
# 量到的實際週期是 0.1800 s ⇒ 那是 7.5 m/s² 的脈衝。這裡套用**導航自己的**
# 既有值（gmpc_node.py 宣告的 ax/ay/az_max），按真實物理步長執行。<=0 關閉。
ap.add_argument('--nav-accel-x', type=float, default=1.5)
ap.add_argument('--nav-accel-y', type=float, default=1.0)
ap.add_argument('--nav-accel-w', type=float, default=2.0)
# ---- 第三人視角錄影（預設關閉，開著會大幅拖慢 RTF）----------------------
# 視角沿用 isaac_drawer_room_snapshot.py 的 `06_route`：同時看得到起點、
# 路線與抽屜櫃。規格要求影片**同時看到機器人與抽屜**，相位與開度由**同一趟**
# 的紀錄標註，不另外配音軌。
ap.add_argument('--cam', default='false')
# ---- 夾爪（沿用 isaac_drawer_sim.py 的規格）----
ap.add_argument('--finger-kp', type=float, default=1.0e4)
ap.add_argument('--finger-kd', type=float, default=1.0e3)
ap.add_argument('--finger-open', type=float, default=0.0089,
                help='手指張開位置（URDF 上限 0.0089 m）')
ap.add_argument('--finger-closed', type=float, default=0.0,
                help='手指閉合目標（URDF 下限 0）')
ap.add_argument('--finger-collision', default='hull', choices=['hull', 'split'],
                help='hull = 匯入時的單一凸包（既有行為）；split = 根部／指片兩個'
                     '凸塊，保留真實的指片開口（finger_collision.py）')
ap.add_argument('--drawer-contact-pairs', default='true',
                help='逐物理步記錄抽屜與機器人各剛體的接觸配對（診斷）')
ap.add_argument('--contact-min-n', type=float, default=0.5,
                help='逐物理步連續接觸計時的每指門檻（N）')
ap.add_argument('--pub-tf', default='false',
                help='發 odom→base_footprint 的 TF（Phase 1 感知鏈需要）')
ap.add_argument('--map-out', default=None,
                help='在光達高度逐格做重疊查詢，輸出 pgm/yaml 靜態地圖後結束')
ap.add_argument('--map-res', type=float, default=0.05)
ap.add_argument('--map-x0', type=float, default=-4.30)
ap.add_argument('--map-x1', type=float, default=4.30)
ap.add_argument('--map-y0', type=float, default=-5.30)
ap.add_argument('--map-y1', type=float, default=2.00)
ap.add_argument('--replay-t0', type=float, default=0.0,
                help='重播只渲染這段模擬時間（起）')
ap.add_argument('--replay-t1', type=float, default=1e9,
                help='重播只渲染這段模擬時間（訖）')
ap.add_argument('--replay', default=None,
                help='重播已封存的 room_run.json，只渲染不跑物理')
ap.add_argument('--cam-hz', type=float, default=10.0)
ap.add_argument('--cam-res', default='1280x720')
ap.add_argument('--cam-eye', default='-1.4,-7.4,4.2')
ap.add_argument('--cam-at', default='-1.2,-1.0,0.3')
ap.add_argument('--cam-foc', type=float, default=20.0)
ap.add_argument('--grip-mu-static', type=float, default=1.0,
                help='夾持面（兩指與把手橫桿）的**靜摩擦**。'
                     '**執行前選定並記錄**，不靠 PhysX 預設')
ap.add_argument('--grip-mu-dynamic', type=float, default=0.8,
                help='夾持面的動摩擦')
ap.add_argument('--allow-friction-mismatch', action='store_true',
                help='摩擦核對未通過仍繼續（**診斷用**；結果必須標明）')
ap.add_argument('--headless', default='true')
a = ap.parse_args()

from isaacsim import SimulationApp                               # noqa: E402
sim_app = SimulationApp({'headless': a.headless.lower() == 'true'})

import numpy as np                                               # noqa: E402
from isaacsim.core.api import World                              # noqa: E402
from isaacsim.core.api.objects import (FixedCuboid,              # noqa: E402
                                       FixedCylinder, GroundPlane)
from isaacsim.core.prims import SingleArticulation               # noqa: E402
from isaacsim.core.prims import RigidPrim as _RigidPrim         # noqa: E402
from isaacsim.core.utils.types import ArticulationAction         # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics                  # noqa: E402

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
import drawer_asset as DA                                        # noqa: E402
from isaac_common import (bind_friction, bind_frictionless,     # noqa: E402
                          collision_prims, import_urdf,
                          physics_parts, walk)
from control_authority import (AUTH_GLIDE, AUTH_NAV,            # noqa: E402
                               AUTH_WHOLEBODY)
from dual_source_executor import DualSourceExecutor              # noqa: E402
from park_fixed import HoldMonitor, ParkCfg, StaticGate            # noqa: E402
from wb_cmd_chain_e2 import CmdChainE2                           # noqa: E402
from wb_wheel_limit import WheelLimitConfig                      # noqa: E402

import rclpy                                                     # noqa: E402
from rclpy.node import Node                                      # noqa: E402
from rclpy.qos import (DurabilityPolicy, QoSProfile)             # noqa: E402
from geometry_msgs.msg import TransformStamped                   # noqa: E402
from tf2_ros import TransformBroadcaster                         # noqa: E402
from geometry_msgs.msg import Twist                              # noqa: E402
from nav_msgs.msg import Odometry                                # noqa: E402
from rosgraph_msgs.msg import Clock                              # noqa: E402
from sensor_msgs.msg import JointState, LaserScan                # noqa: E402
from std_msgs.msg import Float64, Float64MultiArray, String               # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]
SCAN_N, SCAN_MIN, SCAN_MAX = 360, -3.14159, 3.14159
SCAN_INC = (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
RANGE_MIN, RANGE_MAX = 0.12, 10.0
ROBOT = '/World/omni_bot'
DRAWER_SUBTREE = '/World/drawer_unit'
# /coman/applied_cmd 的 18 欄契約（與 E2 相同，下游共用同一份解析）
AP_COLS = ['physics_step_id', 'sim_t', 'bvx_body', 'bvy_body', 'wz',
           'qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6',
           'exec_mode_code', 'cmd_age_s', 'n_recv', 'n_rejected',
           'api_applied', 'src_recv_seq', 'src_recv_sim_t']


def read_world(path):
    raw = re.sub(r'<!--.*?-->', '', open(path, encoding='utf-8').read(),
                 flags=re.S)
    w = ET.fromstring(raw).find('world')
    out = []
    for m in w.findall('model'):
        pose = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
        pose += [0.0] * (6 - len(pose))
        g = m.find('.//collision/geometry')
        kind, dims = None, None
        if g is not None and len(g):
            e = list(g)[0]
            kind = e.tag
            if kind == 'box':
                dims = [float(v) for v in e.findtext('size').split()]
            elif kind == 'cylinder':
                dims = [float(e.findtext('radius')),
                        float(e.findtext('length'))]
        mat = m.find('.//visual/material/diffuse')
        rgb = ([float(v) for v in mat.text.split()][:3] if mat is not None
               else [0.8, 0.8, 0.8])
        out.append(dict(name=m.get('name'), pose=pose, kind=kind, dims=dims,
                        rgb=rgb))
    return out


def yaw_of(q):
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def ray_range(query, origin, direction, max_range, excluded):
    best = [float('inf'), None]

    def report(hit):
        coll = str(getattr(hit, 'collision', '') or '')
        body = str(getattr(hit, 'rigidBody', '') or '')
        d = float(getattr(hit, 'distance', 0.0))
        if any(coll == e or (body and e.startswith(body + '/'))
               for e in excluded):
            return True
        if 0.0 <= d < best[0]:
            best[0], best[1] = d, (coll or body)
        return True

    query.raycast_all(origin, direction, max_range, report)
    return best[0]


class Bridge(Node):
    def __init__(self):
        super().__init__('drawer_room_sim')
        # **關節順序要可被查核。** arm_vel_adapter 會向消費節點查這個參數，
        # 核對它與安全鏈用的順序一致 —— 順序不同的話每個關節都拿到別人的
        # 速度，而下游殘差看起來仍然健康，等到手臂動到不該去的地方才發現。
        self.declare_parameter('joints', list(ARM))
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan_raw', 10)
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.applied_pub = self.create_publisher(
            Float64MultiArray, '/coman/applied_cmd', 10)
        self.sp_pub = self.create_publisher(
            Float64MultiArray, '/coman/arm_setpoint', 10)
        self.drawer_pub = self.create_publisher(
            Float64MultiArray, '/drawer/state', 10)
        self.ho_pub = self.create_publisher(String, '/handover/state', 10)
        # **設定點介面契約**。求解節點的握手閘門把它當 ARM 的必要條件
        #（`_meta_ok is not True` 就不會 ARMED），所以要發。
        # 內容與 isaac_wholebody_sim_e2.py 同一份契約 —— 欄位、關節順序、
        # 取樣時刻與物理步長都要一致，否則閘門會判「介面契約未通過」。
        _lat1 = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sp_meta_pub = self.create_publisher(
            String, '/coman/arm_setpoint_meta', _lat1)
        self._sp_meta_sent = False
        # [sim_t, 開度] ＋ 列優先 4x4 把手世界位姿（USD 真值）
        # **TF（預設關閉）**：Phase 1 的感知鏈要用 TF 把掃描點換到全域座標。
        # 不借 odom_tf_broadcaster —— 它吃 /odom_raw 再重發 /odom，會和本節點
        # 的 /odom 變成兩個發布者。本節點本來就握有真值位姿，自己發最乾淨。
        self.tf_pub = (TransformBroadcaster(self)
                       if a.pub_tf.lower() == 'true' else None)
        self.handle_pub = self.create_publisher(Float64MultiArray,
                                                '/drawer/handle_pose', 10)
        self.sync_pub = self.create_publisher(Float64MultiArray,
                                              '/drawer/sync_state', 10)
        self.create_subscription(Twist, '/cmd_vel', self._counted(self._cmd_vel), 10)
        # **減速交接段**：第三個控制權擁有者，與導航共用直寫路徑。
        self.create_subscription(Twist, '/glide_vel', self._counted(self._glide_vel), 10)
        # 夾爪：目標手指位置（m）。沒收到就維持張開。
        self.grip_cmd = None
        self.create_subscription(Float64, '/gripper/cmd', self._counted(self._grip_cmd), 10)
        # 前五欄相容舊格式；追加 physics_step, continuous_s, threshold_n。
        self.grip_pub = self.create_publisher(Float64MultiArray,
                                              '/gripper/state', 10)
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd',
                                 self._counted(self._wb_cmd), 10)
        self.create_subscription(String, '/handover/request', self._counted(self._ho), 10)
        # **受控停止**：收到就跳出主迴圈並封存。沒有這條，運行器在任務結束後
        # cleanup 會在封存寫出前把模擬器殺掉 —— 逐步紀錄就全沒了。
        self.create_subscription(String, '/room/stop', self._counted(self._stop), 10)
        self.stop_req = None
        self.n_cb = 0
        self.n_drain_max = 0
        self.nav_cmd = None
        self.nav_cmd_t = None
        self.n_nav_cmd = 0
        self.glide_cmd = None
        self.glide_cmd_t = None
        self.n_glide_cmd = 0
        self.ho_req = None
        self.step_id = 0
        self.last_sim_t = 0.0      # 主迴圈每步更新；回呼用它當接收時間

    def _counted(self, f):
        """包一層計數，主迴圈才能判斷佇列是否已清空。"""
        def g(m):
            self.n_cb += 1
            f(m)
        return g

    def stamp(self, t):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec = int(t)
        s.nanosec = int(round((t - int(t)) * 1e9))
        return s

    def _grip_cmd(self, m):
        v = float(m.data)
        if math.isfinite(v):
            self.grip_cmd = v

    def _glide_vel(self, m: Twist):
        self.glide_cmd = (float(m.linear.x), float(m.linear.y),
                          float(m.angular.z))
        self.glide_cmd_t = self.last_sim_t
        self.n_glide_cmd += 1

    def _cmd_vel(self, m: Twist):
        self.nav_cmd = (float(m.linear.x), float(m.linear.y),
                        float(m.angular.z))
        # **接收時間只在這裡更新。** 先前在主迴圈裡每個物理步都刷新，
        # 只要曾收到過命令就永遠算新鮮 —— 導航停止發布後舊命令會一直被沿用，
        # 既定的停止處置永遠不會觸發。
        self.nav_cmd_t = self.last_sim_t
        self.n_nav_cmd += 1

    def _wb_cmd(self, m: Float64MultiArray):
        # 轉給命令鏈；鏈自己有逾時、限位與輪級處理
        if len(m.data) >= 9:
            self.wb_latest = (tuple(float(x) for x in m.data[:9]),)

    def _stop(self, m: String):
        self.stop_req = str(m.data)

    def _stop(self, m: String):
        self.stop_req = str(m.data)

    def _ho(self, m: String):
        try:
            self.ho_req = json.loads(m.data)
        except Exception as e:
            self.get_logger().error(f'交棒請求無法解析：{e}')
            self.ho_req = None

    def publish_clock(self, t):
        c = Clock()
        c.clock = self.stamp(t)
        self.clock_pub.publish(c)


def main() -> int:
    os.makedirs(a.out, exist_ok=True)
    world = World(stage_units_in_meters=1.0, physics_dt=a.physics_dt,
                  rendering_dt=a.physics_dt)
    stage = world.stage
    GroundPlane(prim_path='/World/ground', name='ground', z_position=0.0)

    # ---- 房間 ----
    n_box = n_cyl = 0
    for m in read_world(a.world):
        if m['kind'] in (None, 'plane'):
            continue
        x, y, z = m['pose'][0], m['pose'][1], m['pose'][2]
        path = f"/World/{m['name']}"
        col = np.array(m['rgb'])
        if m['kind'] == 'box':
            FixedCuboid(prim_path=path, name=m['name'],
                        position=np.array([x, y, z]),
                        scale=np.array(m['dims']), color=col)
            n_box += 1
        else:
            r, l = m['dims']
            FixedCylinder(prim_path=path, name=m['name'],
                          position=np.array([x, y, z]), radius=r, height=l,
                          color=col)
            n_cyl += 1
    print(f'[room] 房間：{n_box} 方塊、{n_cyl} 圓柱', flush=True)

    # ---- 抽屜櫃：與階段 A 同一條路徑 ----
    # 載入當下的內容雜湊：事後核對要看「跑的時候」用的是哪份，不是核對時的檔案
    with open(a.drawer_asset, 'rb') as _f:
        drawer_asset_sha256 = hashlib.sha256(_f.read()).hexdigest()
    dspec = DA.load(a.drawer_asset)
    dpose = tuple(float(v) for v in a.drawer_pose.split(','))
    dauth = DA.build_usd(stage, dspec, dpose, root=DRAWER_SUBTREE)
    for k in ('drive_stiffness', 'drive_damping', 'drive_max_force'):
        if abs(float(dauth[k])) > 1e-12:
            print(f'[room] **{k} 讀回 {dauth[k]}，不為零** ⇒ 中止',
                  flush=True)
            return 14
    print(f'[room] 抽屜櫃建於 {dpose}；行程上限 {dauth["limit_upper"]:.4f}',
          flush=True)

    # ---- 機器人 ----
    rx, ry, ryaw = (float(v) for v in a.start_pose.split(','))
    import_urdf(a.urdf, ROBOT, fix_base=False)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(rx, ry, 0.0))
    xf.AddRotateZOp().Set(math.degrees(ryaw))
    finger_col_rep = {'mode': a.finger_collision}
    if a.finger_collision == 'split':
        from finger_collision import apply_split
        finger_col_rep.update(apply_split(stage, ROBOT))
        for _ln, _r in finger_col_rep['fingers'].items():
            print(f'[room] 手指碰撞拆塊 {_ln}：停用 {_r["disabled"]}；'
                  + '；'.join(f'{k} y {v["y_range_mm"]} z {v["z_range_mm"]} mm'
                             for k, v in _r['parts'].items()), flush=True)

    # ---- 零摩擦材質 ----
    # URDF 的 <gazebo><mu1> importer 不讀，產生的 USD 沒有材質，PhysX 退回
    # 預設（約 0.5）。**接觸雙方都要綁**：預設混合規則取平均，只歸零機器人側
    # 仍留一半地面摩擦。不綁的實測後果：0.25 m/s 驅動 5 s 偏航 21.1°。
    fric = None
    if not a.no_frictionless:
        sph = collision_prims(stage, ROBOT, lambda p: 'support_' in p)
        gr = collision_prims(stage, '/World/ground')
        nb = bind_frictionless(stage, sph + gr)
        fric = {'support_spheres': len(sph), 'ground': len(gr), 'bound': nb}
        print(f'[room] 零摩擦材質：支撐球 {len(sph)} + 地面 {len(gr)} '
              f'→ 綁定 {nb}', flush=True)
        if nb == 0:
            print('[room] **一個都沒綁到** ⇒ 底盤會偏航，中止', flush=True)
            return 19
    # ---- 夾持面的摩擦：**明確指定**，不靠 PhysX 預設 ----
    # 兩指與把手橫桿承擔整個夾持。不綁的話它們退回 PhysX 預設（約 0.5）——
    # 那個值不在任何規格裡，也從來沒有被選過。接觸操作的摩擦要像抽屜的
    # 質量與阻尼一樣，執行前選定並記錄。
    grip = [pr for pr in collision_prims(stage, ROBOT)
            if 'finger' in str(pr.GetPath()).lower()]
    grip += [pr for pr in collision_prims(stage, DRAWER_SUBTREE)
             if 'handle_bar' in str(pr.GetPath()).lower()]
    n_grip = bind_friction(stage, grip, a.grip_mu_static, a.grip_mu_dynamic)
    print(f'[room] 夾持面摩擦 μs={a.grip_mu_static} μd={a.grip_mu_dynamic}'
          f' → 綁定 {n_grip}／{len(grip)}', flush=True)
    if n_grip == 0:
        print('[room] **夾持面一個都沒綁到** ⇒ 夾持會靠未指定的預設值，中止',
              flush=True)
        return 21

    # **逐面核實際材質與摩擦值。** 只看綁定數會漏掉未綁的面；而且夾持面
    # （手指、把手橫桿）**不得**被歸零 —— 歸零就夾不住。
    from friction_audit import audit as _fric_audit, summarise as _fric_sum
    fr_ok, fr_rep = _fric_audit(stage, walk, robot_root=ROBOT,
                                drawer_root=DRAWER_SUBTREE)
    print('[room] 摩擦核對：', flush=True)
    for _l in _fric_sum(fr_rep).split('\n'):
        print('   ' + _l, flush=True)
    if not fr_ok and not a.allow_friction_mismatch:
        print('[room] **摩擦核對未通過** ⇒ 中止（要跳過請用 '
              '--allow-friction-mismatch，並在報告標明）', flush=True)
        json.dump(fr_rep, open(os.path.join(a.out, 'friction_audit.json'), 'w'),
                  ensure_ascii=False, indent=1, default=str)
        return 20
    json.dump(fr_rep, open(os.path.join(a.out, 'friction_audit.json'), 'w'),
              ensure_ascii=False, indent=1, default=str)

    _k = UsdLux.DistantLight.Define(stage, '/World/key')
    _k.CreateIntensityAttr(3000.0)
    UsdGeom.Xformable(_k).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 0.0, 30.0))
    UsdLux.DomeLight.Define(stage, '/World/dome').CreateIntensityAttr(600.0)

    bodies, arts = physics_parts(stage, ROBOT)
    if not arts:
        print('[room] **找不到 articulation root** ⇒ 中止', flush=True)
        return 15
    art_root = arts[0]
    print(f'[room] articulation root {art_root}（剛體 {len(bodies)}）',
          flush=True)

    # ---- 夾爪：手指對抽屜的接觸力視圖（**必須在 world.reset() 之前建**）----
    # 做法照搬 isaac_drawer_sim.py（grip_090257_pull20full 用它拉開 20 mm）：
    # 每指一個剛體、只過濾抽屜本體，讀 get_contact_force_matrix。
    # 夾持成立的判準也沿用那一套：每指 ≥ 0.5 N、保持 2 s。
    finger_bodies = sorted(b for b in bodies
                           if str(b).rstrip('/').split('/')[-1]
                           in ('uflite_finger1', 'uflite_finger2'))
    finger_v = None
    if len(finger_bodies) == 2:
        # **用單一字串的字元類別，不用路徑清單。** 清單形式下初始化報
        # 「Provided patterns for sensor and filters did not match any rigid
        # contact entries」，接觸力全程讀成 NaN（motm_cal20_073038）。
        # isaac_drawer_sim.py 也是這樣寫，並註明不要用 (A|B) 的全路徑交替。
        finger_v = _RigidPrim(prim_paths_expr=str(finger_bodies[0])[:-1] + '[12]',
                              name='finger_v', track_contact_forces=True,
                              max_contact_count=128,
                              prepare_contact_sensors=True,
                              contact_filter_prim_paths_expr=[
                                  dauth['drawer_prim']])
        print(f'[room] 手指接觸視圖：{finger_bodies} ↔ {dauth["drawer_prim"]}',
              flush=True)
    else:
        print(f'[room] **找不到兩隻手指剛體**（{finger_bodies}）⇒ 夾爪不可用',
              flush=True)

    # ---- 抽屜↔機器人各剛體的**接觸配對**（診斷用）----
    # 感測端是抽屜本體，每個機器人剛體各為一個過濾對象 ⇒ 每對分開記力向量、
    # 摩擦、接觸點位置與點數。**不把各對加總成一個淨力**：不同部位的力會互相
    # 抵銷，就看不出是誰在推抽屜。另記抽屜未過濾的總接觸力，與各對之和的差
    # 即「非機器人」來源（櫃體等靜態碰撞體），只作殘差，不歸因到任何部位。
    drawer_pair_v = None
    pair_names = [str(b).rstrip('/').split('/')[-1] for b in bodies]
    if a.drawer_contact_pairs.lower() == 'true':
        try:
            drawer_pair_v = _RigidPrim(
                prim_paths_expr=dauth['drawer_prim'], name='drawer_pairs',
                track_contact_forces=True, max_contact_count=512,
                prepare_contact_sensors=True,
                contact_filter_prim_paths_expr=[str(b) for b in bodies])
            print(f'[room] 抽屜接觸配對：{len(bodies)} 個機器人剛體', flush=True)
        except Exception as _e:
            print(f'[room] **抽屜接觸配對視圖建立失敗**：{_e!r}（不記錄，'
                  f'不當成沒有接觸）', flush=True)
            drawer_pair_v = None

    world.reset()
    robot = SingleArticulation(prim_path=art_root, name='omni_bot')
    robot.initialize()
    names = list(robot.dof_names)
    idx = {n: names.index(n) for n in ARM if n in names}
    missing = [n for n in ARM if n not in idx]
    if missing:
        print(f'[room] **URDF 缺關節 {missing}** ⇒ 中止', flush=True)
        return 16

    # 收攏姿態：導航期間由位置驅動維持
    stow = [float(v) for v in a.stow_q.split(',')]
    if len(stow) != len(ARM):
        print(f'[room] **--stow-q 需要 {len(ARM)} 個值** ⇒ 中止', flush=True)
        return 17
    q0 = robot.get_joint_positions()
    for k, j in enumerate(ARM):
        q0[idx[j]] = stow[k]
    kp = np.zeros(robot.num_dof, dtype=np.float32)
    kd = np.zeros(robot.num_dof, dtype=np.float32)
    for j in ARM:
        kp[idx[j]] = a.kp
        kd[idx[j]] = a.kd
    # **手指要有驅動增益**。URDF 匯入時記「Stiffness and damping not
    # available ... actuator will be created without gain parameters」，
    # 先前這裡只給手臂增益，其他關節全是 0 ⇒ 手指是軟的，下目標也不會動。
    # 值沿用 isaac_drawer_sim.py 的 kp 1e4 / kd 1e3。
    FJ = [j for j in ('finger_joint1', 'finger_joint2') if j in names]
    fidx = {j: names.index(j) for j in FJ}
    for j in FJ:
        kp[fidx[j]] = a.finger_kp
        kd[fidx[j]] = a.finger_kd
        q0[fidx[j]] = a.finger_open
    print(f'[room] 手指關節 {FJ}；增益 kp {a.finger_kp} kd {a.finger_kd}；'
          f'初始張開 {a.finger_open}', flush=True)
    if finger_v is not None:
        finger_v.initialize()

    def finger_contact_n():
        """每指對抽屜的接觸力大小（N）。讀不到就回 (nan, nan)，**不當成 0**。"""
        if finger_v is None:
            return float('nan'), float('nan')
        try:
            M = np.asarray(finger_v.get_contact_force_matrix(dt=a.physics_dt))
            # (手指, 過濾對象, 3) ⇒ 每指對抽屜合力的大小
            f = np.linalg.norm(M.reshape(M.shape[0], -1, 3).sum(axis=1), axis=1)
            return float(f[0]), float(f[1])
        except Exception as _e:
            if not _fc_warned[0]:
                _fc_warned[0] = True
                print(f'[room] **手指接觸力讀不到**：{_e!r}（之後回 NaN，'
                      f'不當成 0）', flush=True)
            return float('nan'), float('nan')
    _fc_warned = [False]
    if drawer_pair_v is not None:
        drawer_pair_v.initialize()

    def _np(x):
        return np.asarray(x.cpu().numpy() if hasattr(x, 'cpu') else x,
                          dtype=float)

    def drawer_pairs():
        """每個有接觸的配對：[剛體索引, 點數, 法向力向量(3), 摩擦力向量(3),
        力加權接觸點(3), 最小間距]，全為世界座標、作用在**抽屜**上。
        另回抽屜未過濾總接觸力(3)。讀不到回 (None, None) —— 不寫成空清單。"""
        if drawer_pair_v is None:
            return None, None
        try:
            Fm = _np(drawer_pair_v.get_contact_force_matrix(
                dt=a.physics_dt)).reshape(-1, 3)
            fn, pts, nrm, dist, cnt, start = (
                _np(v) for v in drawer_pair_v.get_contact_force_data(
                    dt=a.physics_dt))
            ff, fpts, fcnt, fstart = (
                _np(v) for v in drawer_pair_v.get_friction_data(
                    dt=a.physics_dt))
            cnt, start = cnt.reshape(-1), start.reshape(-1)
            fcnt, fstart = fcnt.reshape(-1), fstart.reshape(-1)
            out = []
            for j in range(len(cnt)):
                c, s = int(cnt[j]), int(start[j])
                fc_, fs_ = int(fcnt[j]), int(fstart[j])
                if c == 0 and fc_ == 0:
                    continue
                w = np.abs(fn[s:s + c, 0])
                cen = ((w[:, None] * pts[s:s + c]).sum(0) / w.sum()
                       if c and w.sum() > 0 else
                       (pts[s:s + c].mean(0) if c else
                        np.full(3, np.nan)))
                out.append([j, c, *[round(float(v), 4) for v in Fm[j]],
                            *[round(float(v), 4)
                              for v in ff[fs_:fs_ + fc_].sum(0)],
                            *[round(float(v), 5) for v in cen],
                            round(float(dist[s:s + c, 0].min()), 6)
                            if c else None])
            net = _np(drawer_pair_v.get_net_contact_forces(
                dt=a.physics_dt)).reshape(-1)[:3]
            return out, [round(float(v), 4) for v in net]
        except Exception as _e:
            if not _dp_warned[0]:
                _dp_warned[0] = True
                print(f'[room] **抽屜接觸配對讀不到**：{_e!r}（之後記 None）',
                      flush=True)
            return None, None
    _dp_warned = [False]
    robot.set_joint_positions(q0)
    robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=q0))
    print(f'[room] 手臂收攏於 {[round(v, 6) for v in stow]}', flush=True)

    # ---- 抽屜開度零點 ----
    from pxr import Gf as _Gf
    from pxr import UsdGeom as _UG
    xfc = _UG.XformCache()
    dprim = stage.GetPrimAtPath(dauth['drawer_prim'])
    DY0 = float(xfc.GetLocalToWorldTransform(dprim).ExtractTranslation()[1])
    print(f'[room] 抽屜開度零點 DY0 = {DY0:.6f}', flush=True)
    # **把手位姿要由模擬器量出來發布**，不讓下游用硬編常數重算一份。
    # 同一個幾何在兩個地方各寫一次，遲早會漂。接觸前的退讓目標與開啟段的
    # 移動目標都由這一筆算出來（見 evaluation/drawer_align_node.py）。
    hprims = [pr for pr in collision_prims(stage, DRAWER_SUBTREE)
              if 'handle_bar' in str(pr.GetPath()).lower()]
    hprim = hprims[0] if hprims else None
    if hprim is None:
        print('[room] **找不到把手 prim** ⇒ 無法發布把手位姿，中止', flush=True)
        return 22
    print(f'[room] 把手 prim {hprim.GetPath()}', flush=True)

    # ---- 第三人視角相機（錄影用）----
    cam = None
    cam_dir = os.path.join(a.out, 'frames')
    cam_csv = None
    n_frame = 0
    if a.cam.lower() == 'true':
        import imageio.v2 as _imageio
        from isaacsim.sensors.camera import Camera as _Camera
        _W, _H = (int(v) for v in a.cam_res.split('x'))
        os.makedirs(cam_dir, exist_ok=True)
        cam = _Camera(prim_path='/World/rec_cam', resolution=(_W, _H))
        cam.initialize()
        _cg = _UG.Camera(stage.GetPrimAtPath('/World/rec_cam'))
        _cg.GetClippingRangeAttr().Set(_Gf.Vec2f(0.05, 200.0))
        _cg.GetFocalLengthAttr().Set(float(a.cam_foc))
        _eye = [float(v) for v in a.cam_eye.split(',')]
        _at = [float(v) for v in a.cam_at.split(',')]
        _cx = _UG.Xformable(stage.GetPrimAtPath('/World/rec_cam'))
        _cx.ClearXformOpOrder()
        _cx.AddTransformOp().Set(_Gf.Matrix4d().SetLookAt(
            _Gf.Vec3d(*_eye), _Gf.Vec3d(*_at), _Gf.Vec3d(0, 0, 1)).GetInverse())
        cam_csv = open(os.path.join(a.out, 'frames.csv'), 'w')
        cam_csv.write('frame,sim_t,step,owner,kind,opening_m,x,y,yaw\n')
        print(f'[room] **錄影開啟** {_W}x{_H} @ {a.cam_hz} Hz → {cam_dir}',
              flush=True)

    # ---- lidar ----
    lidar_prim = None
    for pr in walk(stage, ROBOT):
        if pr.GetName() == 'lidar_link':
            lidar_prim = pr
            break
    if lidar_prim is None:
        print('[room] **找不到 lidar_link** ⇒ 中止', flush=True)
        return 18
    excluded = []
    for pr in walk(stage, ROBOT):
        if not pr.HasAPI(UsdPhysics.CollisionAPI):
            continue
        p_ = str(pr.GetPath())
        if p_.endswith('/base_link/cylinder') or '/lidar_link/' in p_:
            excluded.append(p_)
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()

    # ---- 產生靜態地圖（只做這件事就結束）------------------------------
    # 給 scan_obstacle_tracker 做靜態背景相減、給 planner 的 costmap 用。
    # 幾何**不手抄尺寸**：直接在這個已經建好的場景上，於光達高度逐格做
    # PhysX 重疊查詢。機器人自己的碰撞體排除在外。
    if a.map_out:
        o0 = xfc.GetLocalToWorldTransform(lidar_prim).ExtractTranslation()
        z_l = float(o0[2])
        res = a.map_res
        x0, x1 = a.map_x0, a.map_x1
        y0, y1 = a.map_y0, a.map_y1
        nx = int(round((x1 - x0) / res))
        ny = int(round((y1 - y0) / res))
        rob = set(str(pr.GetPath()) for pr in walk(stage, ROBOT))
        print(f'[room] 產生地圖 {nx}x{ny} @ {res} m/px，光達高度 z={z_l:.3f}',
              flush=True)
        occ = bytearray(nx * ny)
        hit_box = [False]

        def _rep(h):
            c = str(getattr(h, 'collision', '') or '')
            b = str(getattr(h, 'rigidBody', '') or '')
            if c in rob or b in rob or any(c.startswith(r + '/') for r in rob):
                return True
            hit_box[0] = True
            return False

        for jy in range(ny):
            for ix in range(nx):
                cx = x0 + (ix + 0.5) * res
                cy = y0 + (jy + 0.5) * res
                hit_box[0] = False
                query.overlap_sphere(res * 0.5, [cx, cy, z_l], _rep, False)
                if hit_box[0]:
                    occ[jy * nx + ix] = 1
            if jy % 40 == 0:
                print(f'  [地圖] 列 {jy}/{ny}', flush=True)
        os.makedirs(a.map_out, exist_ok=True)
        pgm = os.path.join(a.map_out, 'drawer_room.pgm')
        with open(pgm, 'wb') as f:
            f.write(f'P5\n{nx} {ny}\n255\n'.encode())
            # PGM 第一列是影像上緣 = 世界 y 最大
            for jy in range(ny - 1, -1, -1):
                f.write(bytes(0 if occ[jy * nx + ix] else 254
                              for ix in range(nx)))
        with open(os.path.join(a.map_out, 'drawer_room.yaml'), 'w') as f:
            f.write(f'image: drawer_room.pgm\nresolution: {res}\n'
                    f'origin: [{x0}, {y0}, 0.0]\nnegate: 0\n'
                    f'occupied_thresh: 0.65\nfree_thresh: 0.196\n')
        n_occ = sum(occ)
        print(f'[room] 地圖完成：{nx}x{ny}，佔據 {n_occ} 格'
              f'（{100.0*n_occ/(nx*ny):.1f}%）→ {pgm}', flush=True)
        sim_app.close()
        return 0

    # ---- 命令鏈 ＋ 雙來源執行層 ----
    def low_speed_bound(vx, vy, wz):
        lin = math.hypot(vx, vy)
        if lin > a.base_lin_max:
            return (False, f'線速度 {lin:.4f} > {a.base_lin_max} m/s')
        if abs(wz) > a.base_ang_max:
            return (False, f'角速度 {abs(wz):.4f} > {a.base_ang_max} rad/s')
        return (True, None)

    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    K = WholeBodyKinematics.from_urdf_file(os.path.join(
        WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
    lim = np.array(K.joint_limits())
    chain = CmdChainE2(max_cmd_age_s=a.max_cmd_age_s,
                       arm_rate_max=a.arm_rate_max, wheel_ok=low_speed_bound,
                       joint_lower=tuple(float(x) for x in lim[0, 3:]),
                       joint_upper=tuple(float(x) for x in lim[1, 3:]),
                       joint_margin=a.joint_margin, mode='solver_drawer',
                       wheel_cfg=WheelLimitConfig(
                           arm_rate_max=a.arm_rate_max,
                           dt_max=max(5.0 * a.physics_dt, 0.05)),
                       base_accel_max_lin=(a.base_accel_lin
                                           if a.base_accel_lin > 0 else None),
                       base_accel_max_ang=(a.base_accel_ang
                                           if a.base_accel_ang > 0 else None))
    # ---- PARK_FIXED：靜止閘門與保持監看（純邏輯在 park_fixed.py）----
    if a.park_fixed and a.park_hold:
        print('[room] **--park-fixed 與 --park-hold 只能擇一** ⇒ 中止', flush=True)
        return 3
    park_on = bool(a.park_fixed or a.park_hold)
    _pk = [float(v) for v in a.park_pose.split(',')]
    park_cfg = ParkCfg(park_x=_pk[0], park_y=_pk[1], park_yaw=_pk[2],
                       hold_s=max(0.5, float(a.max_cmd_age_s)), hold_mode=bool(a.park_hold))
    park_gate = StaticGate(park_cfg) if park_on else None
    park_hold = None
    park_done = False
    park_state = {'violation': None, 'transfer_without_gate': False, 'external': None}
    park_prev_cmd = (float('nan'),) * 3      # 上一物理步實際寫入的底盤命令（閘門用）
    park_last_sample = None                  # 上一物理步 (step, t, x, y, yaw)（保持監看承接用）
    # 規格：PARK_FIXED **明確取消滾動交棒下界**（停住反而不能接手）；MOTM 保持原值。其他預核照做。
    v_min_lin_eff = 0.0 if park_on else a.v_min_lin

    def _park_guard(to, sid, st_):
        """提交切換當步：停車閘門已通過、停車條件仍成立、閘門資料是前一步（新鮮）、未違規。"""
        if park_state['violation'] is not None:
            return False, 'PARK_FIXED 已違規閂鎖'
        if not park_gate.passed:
            return False, '停車靜止閘門尚未通過'
        if not park_gate.ok_now:
            return False, f'停車條件當前不成立：{park_gate.why_now}'
        # **本步**的停車量測（ex.step 之前已以本步位姿更新閘門）；不以前一步的 ok_now 充當當步核對
        if park_gate.last is None or park_gate.last['step'] != int(sid):
            return False, (f'停車閘門沒有本步量測（最後一筆 '
                           f'{None if park_gate.last is None else park_gate.last["step"]}，本步 {sid}）')
        return True, None

    ex = DualSourceExecutor(chain, initial=AUTH_NAV,
                            stow_setpoint=tuple(stow),
                            max_seed_age_s=max(5.0 * a.physics_dt, 0.05),
                            v_box_lin=a.v_box_lin, v_box_ang=a.v_box_ang,
                            v_min_lin=v_min_lin_eff,
                            switch_guard=(_park_guard if park_on else None),
                            nav_accel_max=(None if a.nav_accel_x <= 0 else
                                           (a.nav_accel_x, a.nav_accel_y,
                                            a.nav_accel_w)))

    # ---- 重播渲染模式：只渲染，不跑物理也不開 ROS ----------------------
    # 邊跑邊渲染會和控制搶 CPU：實測 RTF 由 ~1.0 掉到 0.51，控制週期變少，
    # 機器人在**模擬時間**裡也跟著變慢，軌跡就不是原來那條了
    #（nav_video_025746：164 s 模擬時間只走到 d = 0.69 m）。
    # 所以物理與控制照原樣跑，渲染另外一趟重播已封存的逐步位姿。
    # 場景由**同一份程式**建起，不另寫一支重播器。
    if a.wrist_v0:
        from wrist_v0_capture import run_wrist_v0
        _set_drawer = None
        if a.wrist_replay:
            # 抽屜開度：與重播模式同一做法（剛體視圖設世界位姿，開度沿世界 −y）
            _wv = _RigidPrim(prim_paths_expr=dauth['drawer_prim'], name='drawer_wrist_replay')
            _wv.initialize()
            xfc.Clear()
            _w0 = xfc.GetLocalToWorldTransform(dprim).ExtractTranslation()
            _wx0, _wz0 = float(_w0[0]), float(_w0[2])

            def _set_drawer(op):
                _wv.set_world_poses(positions=np.array([[_wx0, DY0 - op, _wz0]], dtype=np.float32))
        return run_wrist_v0(a, world, stage, robot, idx, fidx, ARM, FJ, hprim, ROBOT,
                            walk, dspec, a.urdf, set_drawer=_set_drawer)
    if a.replay:
        if cam is None:
            print('[room] 重播模式要開 --cam', flush=True)
            return 24
        rec = json.load(open(a.replay))
        cols = rec['steps_cols']
        need = ['base_xyth', 'q_arm_meas', 'opening_m']
        miss = [c for c in need if c not in cols]
        if miss:
            print(f'[room] 這份紀錄缺 {miss} ⇒ 無法重播（舊格式）', flush=True)
            return 25
        ib, iq, io = (cols.index(c) for c in need)
        _iqf = cols.index('q_finger') if 'q_finger' in cols else None
        # 抽屜本體：剛體視圖設世界位姿。開度沿世界 −y：y = DY0 − 開度
        _dr_warned = [False]
        _dr_view = None
        try:
            _dr_view = _RigidPrim(prim_paths_expr=dauth['drawer_prim'],
                                  name='drawer_replay')
            _dr_view.initialize()
            xfc.Clear()
            _t0 = xfc.GetLocalToWorldTransform(dprim).ExtractTranslation()
            _dr_x0, _dr_z0 = float(_t0[0]), float(_t0[2])
        except Exception as _e:
            print(f'[room] 重播無法建立抽屜視圖：{_e!r}（抽屜維持不動）',
                  flush=True)
            _dr_view = None
        its, ik, iow = cols.index('sim_t'), cols.index('kind'), 2
        every = max(1, int(round((1.0 / a.cam_hz) / a.physics_dt)))
        steps = [x for x in rec['steps'][::every]
                 if a.replay_t0 <= float(x[1]) <= a.replay_t1]
        ops = [float(x[io]) for x in rec['steps']]
        print(f'[room] **重播** {a.replay}：{len(rec["steps"])} 步 ⇒ '
              f'{len(steps)} 幀（每 {every} 步一幀）；'
              f'開度範圍 {min(ops):.4f}–{max(ops):.4f} m', flush=True)
        q_now = robot.get_joint_positions()
        for n, srow in enumerate(steps, start=1):
            bx, by, byaw = srow[ib]
            robot.set_world_pose(
                np.array([bx, by, 0.0], dtype=np.float32),
                np.array([math.cos(byaw / 2), 0.0, 0.0,
                          math.sin(byaw / 2)], dtype=np.float32))
            qq = np.array(q_now, dtype=np.float32).copy()
            for k, j in enumerate(ARM):
                qq[idx[j]] = float(srow[iq][k])
            # 手指：新格式的紀錄才有；舊紀錄沒有就維持張開
            if _iqf is not None and srow[_iqf] is not None:
                for k, j in enumerate(FJ):
                    qq[fidx[j]] = float(srow[_iqf][k])
            robot.set_joint_positions(qq)
            # **只呼叫 world.render() 不夠**：沒有推進物理時，設進去的位姿
            # 不會傳到渲染，畫面會停在初始位姿（實測 f_00600 的 CSV 已在
            # 櫃子前，畫面卻還在起點）。這裡推一個物理步讓變換生效；速度
            # 每幀歸零，位姿每幀重設，所以那一步不會累積出自己的運動。
            # **抽屜開度**也要擺：開關抽屜的趟次裡它會動。用滑軌關節直接設位置。
            if _dr_view is not None:
                try:
                    _dr_view.set_world_poses(positions=np.array(
                        [[_dr_x0, DY0 - float(srow[io]), _dr_z0]],
                        dtype=np.float32))
                except Exception as _e:
                    if not _dr_warned[0]:
                        _dr_warned[0] = True
                        print(f'[room] 重播設抽屜位姿失敗：{_e!r}', flush=True)
            robot.set_linear_velocity(np.zeros(3, dtype=np.float32))
            robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
            world.step(render=True)
            img = cam.get_rgba()
            if img is None or not img.size:
                continue
            _imageio.imwrite(os.path.join(cam_dir, f'f_{n:05d}.png'),
                             img[:, :, :3].astype(np.uint8))
            _ow = {0: 'nav', 1: 'wholebody', 2: 'glide'}.get(srow[iow], '?')
            cam_csv.write(f'{n},{srow[its]:.4f},{srow[0]},{_ow},'
                          f'{srow[ik]},{float(srow[io]):.6f},'
                          f'{bx:.5f},{by:.5f},{byaw:.5f}\n')
            if n % 50 == 0:
                print(f'  [重播] {n}/{len(steps)} 幀', flush=True)
        cam_csv.close()
        print(f'[room] 重播完成：{len(steps)} 幀 → {cam_dir}', flush=True)
        sim_app.close()
        return 0

    rclpy.init()
    node = Bridge()
    node.wb_latest = None
    if park_on:
        node.park_gate_pub = node.create_publisher(String, '/park/gate', 10)
        node.park_vio_pub = node.create_publisher(String, '/park/violation', 10)
        node.park_ext_vio = None

        def _ext_vio(m):
            try:
                node.park_ext_vio = json.loads(m.data)
            except Exception:
                node.park_ext_vio = {'why': m.data}
        node.create_subscription(String, '/park/violation', _ext_vio, 10)
        print(f'[room] **{"PARK_HOLD" if a.park_hold else "PARK_FIXED"}**：停位 {_pk}；閘門保持 {park_cfg.hold_s:.3f} s；'
              f'滾動交棒下界取消（v_min_lin {a.v_min_lin} → 0）', flush=True)
    print('[room] 進入主迴圈', flush=True)

    se = max(1, int(round((1.0 / a.scan_rate) / a.physics_dt)))
    ste = max(1, int(round((1.0 / a.state_rate) / a.physics_dt)))
    next_scan = next_state = next_cam = next_grip = 0.0
    step_id = 0
    contact_hold = PhysicsContactHold(physics_dt=a.physics_dt,
                                      contact_min_n=a.contact_min_n)
    prev_pose = None
    rec = {'steps': [], 'events': [],
           'steps_cols': ['step', 'sim_t', 'owner(0=nav,1=wb,2=glide)', 'kind',
                          'base_cmd_body', 'meas_vb_body', 'applied_arm',
                          'base_xyth', 'q_arm_meas', 'opening_m',
                          'q_finger', 'finger_contact_n', 'contact_held_s',
                          'drawer_pairs', 'drawer_net_n']
                         + (['park_mode'] if park_on else []),
           'drawer_pairs': {
               'source': 'every_physics_step',
               'row': ['body_index', 'n_points', 'normal_force_xyz_N',
                       'friction_force_xyz_N', 'force_weighted_point_xyz_m',
                       'min_separation_m'],
               'frame': 'world; forces act on the drawer',
               'bodies': pair_names,
               'net': 'drawer unfiltered total; residual vs pair sum = '
                      'non-robot sources, not attributed'},
           'contact_hold': {'source': 'every_physics_step',
                            'physics_dt': a.physics_dt,
                            'contact_min_n': a.contact_min_n},
           'owner_codes': {'0': AUTH_NAV, '1': AUTH_WHOLEBODY,
                           '2': AUTH_GLIDE}}

    while rclpy.ok():
        t = float(world.current_time)
        if t >= a.sim_limit:
            print(f'[room] 達模擬上限 {a.sim_limit} s', flush=True)
            break
        if node.stop_req is not None:
            print(f'[room] **受控停止**：{node.stop_req}', flush=True)
            rec['stop_reason'] = node.stop_req
            break
        # **每個物理步把待處理的回呼全部處理完。** spin_once 一次只處理一筆；
        # 數個持續發布的話題（/cmd_vel、/wb_vel_cmd、/gripper/cmd …）在每步
        # 一筆的處理速率下會在佇列裡積壓 —— 實測全身命令由發布到生效 p50
        # 150–210 ms、最大 390 ms（求解端延遲補償假設 80 ms），這是求解節點
        # 上線後手臂上下擺盪的根因（motm_182255、grip_bar26_161806）。
        # 沒有積壓時只多一次呼叫；全身命令仍只取最新一筆進鏈。
        _n_this = 0
        for _ in range(64):
            _before = node.n_cb
            rclpy.spin_once(node, timeout_sec=0.0)
            if node.n_cb == _before:
                break
            _n_this += 1
        node.n_drain_max = max(node.n_drain_max, _n_this)
        node.publish_clock(t)

        # 全身命令進鏈
        if node.wb_latest is not None:
            chain.receive(list(node.wb_latest[0]), t)
            node.wb_latest = None

        # 交棒請求
        if node.ho_req is not None:
            r = node.ho_req
            node.ho_req = None
            if park_on and park_state['violation'] is not None and r.get('action') != 'cancel':
                # **違規閂鎖後拒絕所有後續正常換手請求**（不只轉給全身）
                rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'handover_refused_after_violation',
                                      'request': r})
                print(f'[room] PARK 違規已閂鎖 ⇒ 拒絕換手請求 {r}', flush=True)
            elif r.get('action') == 'cancel':
                ex.cancel_handover(r.get('why', ''))
            else:
                ex.request_handover(r.get('to', AUTH_WHOLEBODY),
                                    int(r.get('at_step', step_id + 5)), t,
                                    window_steps=int(
                                        r.get('window_steps', 0)))

        # 導航命令逾時 ⇒ 歸零（失效處置）。
        # **接收時間只在回呼裡更新**，不在這裡刷新 —— 否則導航停止發布後
        # 舊命令會一直被算成新鮮，逾時處置永遠不會觸發。
        node.last_sim_t = t
        nav = node.nav_cmd
        nav_timed_out = False
        if nav is not None and node.nav_cmd_t is not None \
                and t - node.nav_cmd_t > a.cmd_timeout:
            nav = (0.0, 0.0, 0.0)
            nav_timed_out = True
            if not getattr(node, '_to_logged', False):
                node._to_logged = True
                print(f'[room] **導航命令逾時** '
                      f'{t - node.nav_cmd_t:.2f}s > {a.cmd_timeout}s ⇒ '
                      f'歸零（失效處置，不宣稱滿足加速度限制）', flush=True)
        elif nav is not None:
            node._to_logged = False
        # 減速段命令逾時 ⇒ 同樣歸零。它與導航共用直寫路徑，所以失效處置
        # 必須一致，否則兩個來源的逾時語意不同，紀錄就對不起來。
        gld = node.glide_cmd
        glide_timed_out = False
        if gld is not None and node.glide_cmd_t is not None \
                and t - node.glide_cmd_t > a.cmd_timeout:
            gld = (0.0, 0.0, 0.0)
            glide_timed_out = True
            if not getattr(node, '_gto_logged', False):
                node._gto_logged = True
                print(f'[room] **減速段命令逾時** '
                      f'{t - node.glide_cmd_t:.2f}s > {a.cmd_timeout}s ⇒ '
                      f'歸零（失效處置）', flush=True)
        elif gld is not None:
            node._gto_logged = False

        q_meas = robot.get_joint_positions()
        q_arm = [float(q_meas[idx[j]]) for j in ARM]

        # **切換當步的實測速度**要在這裡算好交給執行層 —— 它是唯一在這一步
        # 握有真實狀態的地方。任務節點的量測是數十步之前的。
        _p0, _q0 = robot.get_world_pose()
        _yw0 = yaw_of(_q0)
        _mvb_now = None
        if prev_pose is not None:
            _ddx = float(_p0[0]) - prev_pose[0]
            _ddy = float(_p0[1]) - prev_pose[1]
            _cc, _ss = math.cos(_yw0), math.sin(_yw0)
            _mvb_now = [(_ddx * _cc + _ddy * _ss) / a.physics_dt,
                        (-_ddx * _ss + _ddy * _cc) / a.physics_dt,
                        (_yw0 - prev_pose[2]) / a.physics_dt]
        if park_on:
            # 外部違規（減速段逾時、全身／任務節點）⇒ 同一條閂鎖：取消待提交切換、拒絕後續換手
            if node.park_ext_vio is not None and park_state['violation'] is None:
                _v = {'step': step_id, 't': t, 'phase': 'external',
                      'why': f'external：{node.park_ext_vio.get("why")}', 'source': node.park_ext_vio}
                park_state['violation'] = _v
                park_state['external'] = node.park_ext_vio
                ex.cancel_handover(f'PARK_FIXED 違規：{_v["why"]}')
                rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'violation', **_v})
                print(f'[room] **PARK_FIXED 外部違規**（閂鎖中止）@ step {step_id}：{_v["why"]}', flush=True)
            # 閘門以**本步**實測位姿與上一步實際寫入的底盤命令更新（守門在 ex.step 內核本步）
            if park_hold is None and not park_done and ex.owner != AUTH_WHOLEBODY:
                _stowed = max(abs(float(q_arm[k]) - float(stow[k]))
                              for k in range(len(stow))) <= a.park_stow_tol
                park_gate.update(step_id, t, float(_p0[0]), float(_p0[1]), float(_yw0),
                                 park_prev_cmd, _stowed)
        res = ex.step(step_id, t, a.physics_dt, nav_cmd=nav, glide_cmd=gld,
                      q_arm_measured=q_arm, meas_vb=_mvb_now)

        # ---- 真正寫進 API ----
        p_now, qn = robot.get_world_pose()
        yw = yaw_of(qn)
        vx, vy, wz = res.base_cmd
        wx = vx * math.cos(yw) - vy * math.sin(yw)
        wy = vx * math.sin(yw) + vy * math.cos(yw)
        robot.set_linear_velocity(np.array([wx, wy, 0.0], dtype=np.float32))
        robot.set_angular_velocity(np.array([0.0, 0.0, wz], dtype=np.float32))
        _fcmd = (a.finger_open if node.grip_cmd is None
                 else min(max(float(node.grip_cmd), a.finger_closed),
                          a.finger_open))
        if res.arm_setpoint is not None:
            tgt = robot.get_joint_positions()
            for k, j in enumerate(ARM):
                tgt[idx[j]] = res.arm_setpoint[k]
            # **兩隻手指都下目標**（URDF 的 mimic 不保證 PhysX 會強制）
            for j in FJ:
                tgt[fidx[j]] = _fcmd
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=tgt))

        # ---- PARK_FIXED：本步的保持監看（同一物理步的真值位姿與實際寫入的底盤命令）----
        _pmode = None
        if park_on:
            _cmd3 = [float(x) for x in res.base_cmd]
            _x, _y = float(p_now[0]), float(p_now[1])
            if park_hold is None and not park_done and res.owner == AUTH_WHOLEBODY:
                # 實際切換給全身的第一步 ⇒ 開始保持監看；錨點＝閘門錨點（之後不更新）；
                # **承接切換前一物理步的位姿**，切換當步的速度才核得到
                if park_gate.anchor is None:
                    park_state['transfer_without_gate'] = True
                park_hold = HoldMonitor(park_cfg, park_gate.anchor or (_x, _y, float(yw)),
                                        step_id, t, prev=park_last_sample)
                rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'hold_start',
                                      'anchor': park_hold.anchor,
                                      'gate_pass_step': park_gate.pass_step,
                                      'prev_sample': park_last_sample})
            if park_hold is not None and park_hold.end is None:
                if res.owner == AUTH_WHOLEBODY:
                    _v = park_hold.update(step_id, t, _x, _y, float(yw), _cmd3, phase=res.kind)
                elif res.owner == AUTH_NAV:
                    # **交還導航當步**：核最後一個全身控制區間（位姿 k−1→k），不核導航首筆命令；
                    # 交還時手臂須已收攏才算正常結束
                    _v = park_hold.update(step_id, t, _x, _y, float(yw), None, phase='handback')
                    _stowed_hb = max(abs(float(q_arm[k]) - float(stow[k]))
                                     for k in range(len(stow))) <= a.park_stow_tol
                    park_hold.close(step_id, t, stowed=_stowed_hb)
                    park_done = True
                    rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'hold_end',
                                          'verdict': park_hold.verdict(),
                                          'handback_stowed': _stowed_hb})
                else:
                    _v = {'step': step_id, 't': t, 'phase': res.kind,
                          'why': f'保持期內控制者變成 {res.owner}（不是交還導航）'}
                if (_v is not None or park_state['transfer_without_gate']) \
                        and park_state['violation'] is None:
                    _v = _v or {'step': step_id, 't': t, 'phase': res.kind,
                                'why': '未通過停車閘門即切換給全身'}
                    park_state['violation'] = _v
                    # **閂鎖本趟中止**：取消待提交切換、通知所有命令來源停止；不靜默改寫命令
                    ex.cancel_handover(f'PARK_FIXED 違規：{_v["why"]}')
                    node.park_vio_pub.publish(String(data=json.dumps(_v)))
                    rec['events'].append({'sim_t': t, 'step': step_id, 'park': 'violation', **_v})
                    print(f'[room] **PARK_FIXED 違規**（閂鎖中止）@ step {step_id}：{_v["why"]}',
                          flush=True)
            park_prev_cmd = tuple(_cmd3)
            park_last_sample = (step_id, t, _x, _y, float(yw))
            _pmode = ('hold' if (park_hold is not None and park_hold.end is None)
                      else 'post' if park_done else 'gate')
            if step_id % 2 == 0:
                _gs = park_gate.state()
                node.park_gate_pub.publish(String(data=json.dumps({
                    'step': step_id, 'sim_t': t, 'passed': _gs['passed'], 'ok_now': _gs['ok_now'],
                    'why_now': _gs['why_now'], 'pass_step': _gs['pass_step'], 'anchor': _gs['anchor'],
                    'mode': _pmode, 'violation': park_state['violation'] is not None})))

        # ---- 套用回報（18 欄契約）----
        _mode = 0 if res.owner == AUTH_WHOLEBODY else 4
        am = Float64MultiArray()
        am.data = ([float(step_id), float(t)]
                   + [float(x) for x in res.u_applied]
                   + [float(_mode), float('nan'),
                      float(getattr(chain, 'n_recv', 0)),
                      float(getattr(chain, 'n_rejected', 0)),
                      1.0 if res.kind in ('nav_applied', 'glide_applied',
                                          'wholebody_applied')
                      else 0.0, -1.0, float('nan')])
        node.applied_pub.publish(am)
        if res.arm_setpoint is not None:
            if not node._sp_meta_sent:
                node._sp_meta_sent = True
                node.sp_meta_pub.publish(String(data=json.dumps({
                    'cols': ['physics_step_id', 'sim_t', 'ready']
                            + [f'sp_{j}' for j in ARM]
                            + ['exec_mode_code', 'api_applied'],
                    'joint_order': list(ARM),
                    'units': 'rad（prismatic 不在此列；手臂六軸皆 revolute）',
                    'sampling_instant': '**本步寫入後**（apply_action 之後）',
                    'physics_dt_s': float(a.physics_dt),
                    'physics_dt_nominal_s': float(a.physics_dt),
                    'pairing': '與同一輪 /joint_states 與 /odom 共用同一個'
                               ' sim_t（三者的 stamp 都由本步的 t 導出）'
                               ' ⇒ 以**共同時間戳**配對；'
                               'JointState 沒有 physics_step_id 欄位，'
                               '不得只靠本說明視為已核對',
                    'recursion': 'act_{i+1} = act_i + α·(sp_i − act_i) + b'
                                 '　⇒ sp_i **先作用於下一物理段**',
                    'G_not_reported': '合成增益 G 由模型端以 α 與 dt 計算；'
                                      '取樣時刻決定用哪一種合成，'
                                      '故契約只需核對取樣時刻與物理步長',
                    'ready_false_means': '設定點**尚未建立**'
                                         '（執行端在第一筆有效命令時由實測'
                                         '關節位置建立）⇒ 求解器**不得**'
                                         '假設 s = q，須走初始化握手',
                    'integration_rate': '設定點每物理步積分一次'
                                        f'（physics_dt，約 '
                                        f'{1.0/max(a.physics_dt,1e-9):.0f} Hz）'
                                        '；20 Hz 只是上游控制頻率',
                    'does_not_change': '本話題為純新增回報，'
                                       '不改既有命令訊息、不改控制行為',
                }, ensure_ascii=False)))
            sm = Float64MultiArray()
            # 末欄是 `api_applied`，**不是常數**：握手閘門把它當 ARM 的必要
            # 條件（`if not s.api_applied: return self._abnormal(...)`）。
            # 原本硬編 0.0 ⇒ 閘門永遠停在 init，求解節點整趟只送零命令
            #（實跑 nav_align_solve_021536：ever_armed = False、published = 0、
            # 近 600 萬筆初始化命令）。這裡回報**本步設定點是否真的寫進 API**，
            # 與 isaac_wholebody_sim_e2.py 同一個語意。
            sm.data = ([float(step_id), float(t), 1.0]
                       + [float(x) for x in res.arm_setpoint]
                       + [float(_mode), 1.0])
            node.sp_pub.publish(sm)

        # ---- **逐物理步證據**：套用命令、實測速度、誰在控制 ----
        # 這一份由模擬器自己記 —— 它每步都有這些值。靠 ROS 訂閱抓會因為
        # 佇列深度與處理速率而只取到樣本（實測 174／3615 筆），
        # 判斷不了「有沒有非預期零命令或跳變」。
        _mvb = None
        if prev_pose is not None:
            _dx = float(p_now[0]) - prev_pose[0]
            _dy = float(p_now[1]) - prev_pose[1]
            _c, _s = math.cos(yw), math.sin(yw)
            _mvb = [(_dx * _c + _dy * _s) / a.physics_dt,
                    (-_dx * _s + _dy * _c) / a.physics_dt,
                    (yw - prev_pose[2]) / a.physics_dt]
        prev_pose = (float(p_now[0]), float(p_now[1]), yw)
        xfc.Clear()
        _dy_rec = float(xfc.GetLocalToWorldTransform(
            dprim).ExtractTranslation()[1])
        # 與本列 q/opening 同一個物理步。不得在 50 Hz 發布分支內計時。
        _contact = finger_contact_n() if FJ else (float('nan'), float('nan'))
        _contact_held = contact_hold.update(
            step_id, t, *_contact,
            closing=(node.grip_cmd is not None
                     and node.grip_cmd < a.finger_open - 1e-12))
        rec['steps'].append([
            step_id, round(t, 4),
            {AUTH_NAV: 0, AUTH_WHOLEBODY: 1}.get(res.owner, 2),
            res.kind, [round(float(x), 6) for x in res.base_cmd],
            None if _mvb is None else [round(x, 6) for x in _mvb],
            [round(float(x), 6) for x in res.u_applied[3:]],
            [round(float(p_now[0]), 6), round(float(p_now[1]), 6),
             round(float(yw), 6)],
            # **重播渲染要的兩個量**：手臂實測關節角與抽屜開度。
            # 邊跑邊渲染會和控制搶 CPU —— 實測 RTF 由 ~1.0 掉到 0.51，
            # 控制週期變少，機器人在**模擬時間**裡也跟著變慢，軌跡就不是
            # 原來那條了。所以物理照原樣跑，渲染另外一趟重播。
            [round(float(x), 6) for x in q_arm],
            round(float(DY0 - _dy_rec), 6),
            # 手指位置（重播要擺手指）與每指接觸力（判斷夾持與滑脫）。
            # 接觸力只在夾爪被下過命令後才讀，導航期間記 None —— 讀不到
            # 或沒讀**不寫成 0**，淨力為零不等於沒有接觸。
            ([round(float(q_meas[fidx[j]]), 6) for j in FJ] if FJ else None),
            ([round(x, 4) for x in _contact]
             if (FJ and node.grip_cmd is not None) else None),
            float(_contact_held), *drawer_pairs(),
            *([_pmode] if park_on else [])])

        # 發布已完成逐步驗收的同時刻資料；不在 world.step 後重讀並混用時間。
        if FJ and t >= next_grip - 1e-9:
            next_grip = max(next_grip + ste * a.physics_dt, t)
            gm = Float64MultiArray()
            gm.data = [t, float(q_meas[fidx[FJ[0]]]),
                       float(q_meas[fidx[FJ[-1]]]), *_contact,
                       float(step_id), float(_contact_held), a.contact_min_n]
            node.grip_pub.publish(gm)

        world.step(render=False)
        # ---- 錄影：到時間才渲染一次（渲染很貴，不是每個物理步都做）----
        if cam is not None and t >= next_cam - 1e-9:
            next_cam = max(next_cam + 1.0 / a.cam_hz, t)
            world.render()
            xfc.Clear()
            _dy_now = float(xfc.GetLocalToWorldTransform(
                dprim).ExtractTranslation()[1])
            _img = cam.get_rgba()
            if _img is not None and _img.size:
                n_frame += 1
                _imageio.imwrite(
                    os.path.join(cam_dir, f'f_{n_frame:05d}.png'),
                    _img[:, :, :3].astype(np.uint8))
                _ow = {AUTH_NAV: 'nav', AUTH_WHOLEBODY: 'wholebody'}.get(
                    res.owner, 'glide')
                cam_csv.write(f'{n_frame},{t:.4f},{step_id},{_ow},'
                              f'{res.kind},{DY0 - _dy_now:.6f},'
                              f'{float(p_now[0]):.5f},{float(p_now[1]):.5f},'
                              f'{float(yw):.5f}\n')
                cam_csv.flush()
        step_id += 1
        node.step_id = step_id
        t_after = float(world.current_time)

        # ---- 感測與狀態 ----
        if t_after >= next_state - 1e-9:
            next_state = max(next_state + ste * a.physics_dt, t_after)
            qm = robot.get_joint_positions()
            js = JointState()
            js.header.stamp = node.stamp(t_after)
            js.name = list(ARM)
            js.position = [float(qm[idx[j]]) for j in ARM]
            node.js_pub.publish(js)
            p2, q2 = robot.get_world_pose()
            od = Odometry()
            od.header.stamp = node.stamp(t_after)
            od.header.frame_id = 'odom'
            od.child_frame_id = 'base_footprint'
            od.pose.pose.position.x = float(p2[0])
            od.pose.pose.position.y = float(p2[1])
            od.pose.pose.orientation.w = float(q2[0])
            od.pose.pose.orientation.x = float(q2[1])
            od.pose.pose.orientation.y = float(q2[2])
            od.pose.pose.orientation.z = float(q2[3])
            node.odom_pub.publish(od)
            if node.tf_pub is not None:
                tfm = TransformStamped()
                tfm.header.stamp = od.header.stamp
                tfm.header.frame_id = 'odom'
                tfm.child_frame_id = 'base_footprint'
                tfm.transform.translation.x = float(p2[0])
                tfm.transform.translation.y = float(p2[1])
                tfm.transform.rotation.w = float(q2[0])
                tfm.transform.rotation.x = float(q2[1])
                tfm.transform.rotation.y = float(q2[2])
                tfm.transform.rotation.z = float(q2[3])
                node.tf_pub.sendTransform(tfm)
            xfc.Clear()
            dy = float(xfc.GetLocalToWorldTransform(
                dprim).ExtractTranslation()[1])
            dm = Float64MultiArray()
            dm.data = [float(t_after), float(DY0 - dy)]
            node.drawer_pub.publish(dm)
            # **把手的實測世界位姿**（USD 真值，列優先 4x4 ＋ 時刻與開度）
            _hm = xfc.GetLocalToWorldTransform(hprim)
            _hM = np.array([[float(_hm[r][c]) for r in range(4)]
                            for c in range(4)], float)
            hp = Float64MultiArray()
            hp.data = ([float(t_after), float(DY0 - dy)]
                       + [float(x) for x in _hM.reshape(-1)])
            node.handle_pub.publish(hp)
            # **同一物理步的狀態包**：開度、把手、底盤真值、手臂關節角。
            # 分開的話題由訂閱端各自處理，到手時可能差好幾個週期 ——
            # 移動中算夾持漂移會把時間差當成滑脫（motm_163548 實測：
            # 實體相對位移 0.00 mm，卻因 TCP 37 mm/s 判出 > 10 mm）。
            # [sim_t, step, 開度, H(16), x, y, yaw, q_arm(6)]，共 28 欄。
            _yaw2 = math.atan2(2 * (float(q2[0]) * float(q2[3])
                                    + float(q2[1]) * float(q2[2])),
                               1 - 2 * (float(q2[2]) ** 2
                                        + float(q2[3]) ** 2))
            ss = Float64MultiArray()
            ss.data = ([float(t_after), float(step_id), float(DY0 - dy)]
                       + [float(x) for x in _hM.reshape(-1)]
                       + [float(p2[0]), float(p2[1]), _yaw2]
                       + [float(qm[idx[j]]) for j in ARM])
            node.sync_pub.publish(ss)
            hs = String()
            # `pending` 要發出去 —— 任務節點必須能分辨「還在窗口裡等」
            # 與「已被取消」。沒有這個欄位，它只能靠步數猜，就會在執行端
            # 正常等待時誤判為失敗而重試。
            hs.data = json.dumps({'owner': ex.owner, 'step': step_id,
                                  'sim_t': t_after,
                                  'chain_seeded': ex.wb_seeded,
                                  **ex.handover_pending_info(),
                                  'nav_applied': ex.nav_applied_report()},
                                 default=float)
            node.ho_pub.publish(hs)

        if t_after >= next_scan - 1e-9:
            next_scan = max(next_scan + se * a.physics_dt, t_after)
            xfc.Clear()
            o5 = xfc.GetLocalToWorldTransform(
                lidar_prim).ExtractTranslation()
            _, q3 = robot.get_world_pose()
            y5 = yaw_of(q3)
            sc = LaserScan()
            sc.header.stamp = node.stamp(t_after)
            sc.header.frame_id = 'lidar_link'
            sc.angle_min, sc.angle_max = SCAN_MIN, SCAN_MAX
            sc.angle_increment = SCAN_INC
            sc.range_min, sc.range_max = RANGE_MIN, RANGE_MAX
            org = [float(o5[0]), float(o5[1]), float(o5[2])]
            sc.ranges = [ray_range(query, org,
                                   [math.cos(y5 + SCAN_MIN + i * SCAN_INC),
                                    math.sin(y5 + SCAN_MIN + i * SCAN_INC),
                                    0.0], RANGE_MAX, excluded)
                         for i in range(SCAN_N)]
            node.scan_pub.publish(sc)

    # ---- 封存 ----
    rec['summary'] = ex.summary()
    rec['chain'] = chain.summary()
    rec['ros_drain'] = {'max_callbacks_in_one_step': node.n_drain_max,
                        'n_callbacks': node.n_cb}
    rec['config'] = {'finger_collision': finger_col_rep,
                     'drawer_asset': a.drawer_asset,
                     'drawer_asset_sha256': drawer_asset_sha256,
                     'world': a.world, 'drawer_pose': a.drawer_pose,
                     'start_pose': a.start_pose, 'stow_q': stow,
                     'physics_dt': a.physics_dt,
                     'low_speed_bound': {'lin': a.base_lin_max,
                                         'ang': a.base_ang_max},
                     'low_speed_bound_applies_to': '**只有全身那條路徑**；'
                                                   '導航不經命令鏈',
                     'drawer': dauth, 'DY0': DY0, 'frictionless': fric,
                     'nav_cmd_timeout_s': a.cmd_timeout,
                     'switch_speed_check': {'v_box_lin': a.v_box_lin,
                                            'v_box_ang': a.v_box_ang,
                                            'v_min_lin': a.v_min_lin},
                     'nav_rate_cap': ex.summary()['nav_rate_cap'],
                     'base_rate_cap': {
                         'a_lin': a.base_accel_lin,
                         'a_ang': a.base_accel_ang,
                         'source': 'wgmpc_core.py 的 a_base_lin/a_base_ang，'
                                   '原註標為「**開發值**，無已核准來源」；'
                                   '被執行層採用不改變這個標籤',
                         'per_axis': True,
                         'combined_implication_mps': round(
                             (2 ** 0.5) * max(a.base_accel_lin, 0.0)
                             * a.physics_dt, 6),
                         'n_capped': chain.n_base_rate_capped,
                         'max_cut_mps': round(chain.base_rate_cap_max, 6)},
                     'friction_audit': {'ok': fr_ok,
                                        'n_faces': fr_rep['n_faces'],
                                        'violations': fr_rep['violations']}}
    if cam_csv is not None:
        cam_csv.close()
        print(f'[room] 錄影收尾：共 {n_frame} 幀 → {cam_dir}', flush=True)
    out = os.path.join(a.out, 'room_run.json')
    json.dump(rec, open(out, 'w'), ensure_ascii=False, indent=1, default=str)
    if park_on:
        if park_hold is not None and park_hold.end is None:
            pass                          # 未交還導航 ⇒ HoldMonitor 判定 INSUFFICIENT（不是 PASS）
        json.dump({'spec': ('park_hold_v2_spec.md' if a.park_hold else 'park_fixed_vs_motm_spec.md'),
                   'mode': ('PARK_HOLD' if a.park_hold else 'PARK_FIXED'), 'park_pose': _pk,
                   'cfg': park_cfg.__dict__, 'v_min_lin_set': a.v_min_lin,
                   'v_min_lin_effective': v_min_lin_eff,
                   'gate': park_gate.state(), 'gate_last': park_gate.last,
                   'hold': None if park_hold is None else park_hold.report(),
                   'violation': park_state['violation'],
                   'transfer_without_gate': park_state['transfer_without_gate'],
                   'source': '同一物理步的真值底盤位姿（robot.get_world_pose）與實際寫入 API 的底盤命令（res.base_cmd）'},
                  open(os.path.join(a.out, 'park_gate.json'), 'w'), ensure_ascii=False, indent=1,
                  default=str)
    print(f'[room] 封存 {out}', flush=True)
    print(f'[room] 控制者 {ex.owner}；步數 {step_id}；'
          f'導航命令 {node.n_nav_cmd} 筆', flush=True)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sim_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
