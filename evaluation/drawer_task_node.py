#!/usr/bin/env python3
"""抽屜任務編排：ALIGN → 夾持 → 開 → 停 → 關 → 停 → 放 → 退 → 收臂 → 交還導航 → 回起點。

**相位由 `drawer_task_policy.DrawerTaskPolicy` 決定**（已有 95 項測試），本節點
只負責兩件事：把**實測**組成 `DrawerState` 餵給它，以及依它給的相位發出動作。
相位推進一律看實測值 —— 開度看 USD 真值、夾持看每指接觸力、到位看本節點自己
的 FK，**不看命令值，也不靠時間到期**。

各相位的動作
------------
ALIGN        先到退讓 `standoff` 的接觸前位姿；誤差 < `pre_tol` 後，以
             `approach_rate` 把退讓量斜坡收到 0。斜坡期間命令持續，相位機的
             停滯判定不會誤觸。
ENGAGE_WAIT  誤差 < `grip_tol` 連續 `grip_settle_s` 後，在 `close_ramp_s` 內把
             手指由張開斜坡到閉合。每指接觸力 ≥ `contact_min_n` 連續
             `attach_hold_s` 才算**夾持成立**（判準沿用 isaac_drawer_sim.py
             在 grip_090257_pull20full 用過的那一組），成立當下以**實測** TCP
             與把手位姿建 `DrawerTarget`。
OPEN/CLOSE   目標由 `DrawerTarget.step()` 依**實測開度**產生，有界超前。
RELEASE_WAIT 手指張開；每指接觸 < `release_max_n` 且手指張開到 80% 以上，
             連續 `release_hold_s` 才算**解除確認**。
RETREAT      目標回到退讓位姿；退出量沿世界 −y 量。
RESTOW       停求解節點，關節空間收回收攏姿態；**j3 最後動**（收攏時 j3 = 0
             離有效下限只有 0.011 rad，往負方向 0.05 rad 內會自撞）。
HANDBACK     請執行端把控制權交還導航。
NAV_HOME     發一條回起點的計畫。

沒有監看的東西要講清楚
----------------------
`undesignated_contact` 目前**沒有量測來源**，一律回 False。這表示本節點**不會**
因為手臂碰到櫃體而中止 —— 報告裡要寫明，不能說「未發生非預期接觸」。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src',
                                'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (         # noqa: E402
    WholeBodyKinematics)
from drawer_target import DrawerTarget, DrawerTargetConfig      # noqa: E402
from drawer_task_policy import (DrawerState, DrawerTaskConfig,  # noqa: E402
                                DrawerTaskPolicy)
from physics_contact_hold import contact_established             # noqa: E402
import motm_coord as MC                                          # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]
STOW = np.array([0.0, 0.0, 0.0, 0.0, -math.pi / 2, 0.0])


def so3_angle(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


class Task(Node):
    def __init__(self, a, K):
        super().__init__('drawer_task')
        self.a, self.K = a, K
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tgt_pub = self.create_publisher(Float64MultiArray,
                                             '/drawer/tcp_target', 10)
        self.grip_pub = self.create_publisher(Float64, '/gripper/cmd', 10)
        self.stop_pub = self.create_publisher(String, '/wgmpc/stop', 10)
        self.wb_pub = self.create_publisher(Float64MultiArray, '/wb_vel_cmd', 10)
        # PARK_FIXED：執行端違規閂鎖與本節點底盤命令核對（預設不用 ⇒ 既有行為不變）
        self.park_fixed = bool(getattr(a, 'park_fixed', False))
        self.park_violation = None
        self.park_base_bad = None
        if self.park_fixed:
            self.create_subscription(String, '/park/violation', self._park_vio, 1)
        self.ho_pub = self.create_publisher(String, '/handover/request', 10)
        self.plan_pub = self.create_publisher(Path, '/plan', lat)
        self.phase_pub = self.create_publisher(String, '/drawer/phase', 10)
        self.solver_start_pub = self.create_publisher(Bool, '/drawer/solver_start', lat)
        self.coord_pub = self.create_publisher(Float64MultiArray,
                                               '/wgmpc/coord', 10)
        # **深度 1**：狀態流要最新值
        self.create_subscription(Float64MultiArray, '/drawer/handle_pose',
                                 self._h, 1)
        # **同一物理步的狀態包**（模擬器有發就用它，分開的三個話題不再覆寫）
        self.create_subscription(Float64MultiArray, '/drawer/sync_state',
                                 self._sync, 1)
        self.use_sync = False
        self.n_sync = 0
        self.create_subscription(JointState, '/joint_states', self._js, 1)
        self.create_subscription(Odometry, '/odom', self._od, 1)
        self.create_subscription(Float64MultiArray, '/gripper/state',
                                 self._gs, 1)
        self.create_subscription(String, '/handover/state', self._hs, 1)
        self.create_subscription(Float64MultiArray, '/coman/applied_cmd',
                                 self._ap, 1)
        self.create_subscription(String, '/coman/applied_fail', self._af, 1)
        self.create_subscription(Float64MultiArray, '/wgmpc/status',
                                 self._ws, 1)
        self.H = None          # (sim_t, 開度, 4x4)
        self.q = None
        self.pose = None       # (x, y, yaw, t)
        self.gs = None         # (t, f1, f2, n1, n2)
        self.gs_evidence = None  # 完整八欄，連續時間由模擬器逐步計算
        self.hs = None
        self.ap = None         # (t, 9 維)
        self.fail = None
        self.ws = None         # (t, err_p, err_r, in_tol, reached)
        self.n_ws = 0

    def _sync(self, m):
        d = [float(x) for x in m.data]
        if len(d) != 28 or not all(math.isfinite(x) for x in d):
            return
        t = d[0]
        self.H = (t, d[2], np.asarray(d[3:19], float).reshape(4, 4))
        self.pose = (d[19], d[20], d[21], t)
        self.q = np.asarray(d[22:28], float)
        self.use_sync = True
        self.n_sync += 1

    def _h(self, m):
        if self.use_sync:
            return
        d = list(m.data)
        if len(d) == 18 and all(math.isfinite(float(x)) for x in d):
            self.H = (float(d[0]), float(d[1]),
                      np.asarray(d[2:], float).reshape(4, 4))

    def _js(self, m):
        if self.use_sync:
            return
        ix = {n: i for i, n in enumerate(m.name)}
        if all(j in ix for j in ARM):
            self.q = np.array([float(m.position[ix[j]]) for j in ARM])

    def _od(self, m):
        if self.use_sync:
            return
        o = m.pose.pose.orientation
        yaw = math.atan2(2 * (o.w * o.z + o.x * o.y),
                         1 - 2 * (o.y * o.y + o.z * o.z))
        self.pose = (float(m.pose.pose.position.x),
                     float(m.pose.pose.position.y), yaw,
                     float(m.header.stamp.sec) + 1e-9 * m.header.stamp.nanosec)

    def _gs(self, m):
        d = list(m.data)
        self.gs_evidence = tuple(float(x) for x in d) if len(d) == 8 else None
        if len(d) >= 5:
            self.gs = tuple(float(x) for x in d[:5])

    def _hs(self, m):
        try:
            self.hs = json.loads(m.data)
        except Exception:
            pass

    def _ap(self, m):
        d = list(m.data)
        if len(d) >= 11:
            self.ap = (float(d[1]), [float(x) for x in d[2:11]],
                       int(d[0]))

    def _af(self, m):
        self.fail = str(m.data)

    def _ws(self, m):
        d = list(m.data)
        if len(d) >= 5:
            self.ws = tuple(float(x) for x in d[:5])
            self.n_ws += 1

    # ---------------------------------------------------------- 量測
    def sim_t(self):
        ts = [x for x in (self.H and self.H[0], self.pose and self.pose[3],
                          self.gs and self.gs[0]) if x]
        return max(ts) if ts else None

    def tcp(self):
        if self.pose is None or self.q is None:
            return None
        return self.K.fk(np.array([self.pose[0], self.pose[1], self.pose[2]]
                                  + list(self.q), float), self.a.tcp)

    def target_at(self, standoff, H=None):
        """接觸前／抓取位姿：位置取**實測**把手，朝向用設計抓取姿態的 FK。

        `H` 給定時改用那一份把手位姿（凍結用）。手指一開始閉合就**不能再
        跟著實測把手走**：TCP 只要往 −y 落後一點就會把把手往外拉，把手一動
        目標跟著動、再拉 —— 正回授。實測 motm_cal20_074420 在夾持建立的
        24 s 裡把抽屜拉開 25.69 mm。
        """
        p = (self.H[2] if H is None else H)[:3, 3]
        T = np.eye(4)
        T[:3, :3] = self.a.R_grasp
        T[:3, 3] = (float(p[0]),
                    float(p[1]) - (self.a.tcp_offset_m + float(standoff)),
                    float(p[2]))
        return T

    def publish_target(self, T):
        m = Float64MultiArray()
        m.data = [float(x) for x in np.asarray(T, float).reshape(-1)]
        self.tgt_pub.publish(m)

    def grip(self, v):
        m = Float64()
        m.data = float(v)
        self.grip_pub.publish(m)

    def _park_vio(self, m):
        try:
            self.park_violation = json.loads(m.data)
        except Exception:
            self.park_violation = {'why': m.data}

    def wb_cmd(self, u9):
        if self.park_fixed and max(abs(float(x)) for x in list(u9)[:3]) > 0.0:
            # PARK_FIXED 下本節點不應生成非零底盤分量 ⇒ 不發布、記錄並由主迴圈中止（不剪成零後繼續）
            if self.park_base_bad is None:
                self.park_base_bad = {'u_base': [float(x) for x in list(u9)[:3]],
                                      'sim_t': self.sim_t()}
            return
        m = Float64MultiArray()
        m.data = [float(x) for x in u9]
        self.wb_pub.publish(m)

    def build_plan(self, start, goal, goal_yaw):
        """只組路徑，不發、不等待（供不可阻塞的迴圈使用）。"""
        p = Path()
        p.header.frame_id = 'odom'
        p.header.stamp = self.get_clock().now().to_msg()
        s, g = np.array(start, float), np.array(goal, float)
        n = max(2, int(np.linalg.norm(g - s) / 0.05) + 1)
        yp = math.atan2(g[1] - s[1], g[0] - s[0])
        for k in range(n):
            ps = PoseStamped()
            ps.header = p.header
            q = s + (g - s) * (k / (n - 1))
            ps.pose.position.x, ps.pose.position.y = float(q[0]), float(q[1])
            yw = float(goal_yaw) if k == n - 1 else yp
            ps.pose.orientation.z = math.sin(yw / 2)
            ps.pose.orientation.w = math.cos(yw / 2)
            p.poses.append(ps)
        return p

    def publish_plan(self, start, goal, goal_yaw):
        p = Path()
        p.header.frame_id = 'odom'
        p.header.stamp = self.get_clock().now().to_msg()
        s, g = np.array(start, float), np.array(goal, float)
        n = max(2, int(np.linalg.norm(g - s) / 0.05) + 1)
        yp = math.atan2(g[1] - s[1], g[0] - s[0])
        for k in range(n):
            ps = PoseStamped()
            ps.header = p.header
            q = s + (g - s) * (k / (n - 1))
            ps.pose.position.x, ps.pose.position.y = float(q[0]), float(q[1])
            yw = float(goal_yaw) if k == n - 1 else yp
            ps.pose.orientation.z = math.sin(yw / 2)
            ps.pose.orientation.w = math.cos(yw / 2)
            p.poses.append(ps)
        for _ in range(5):
            self.plan_pub.publish(p)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.1)
        return len(p.poses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default=os.path.join(
        HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--tcp', default='link_tcp')
    ap.add_argument('--park', default='-0.136412,0.560,1.297349')
    ap.add_argument('--start', default='-2.80,-3.20,1.5708')
    ap.add_argument('--q-grasp',
                    default='-0.0321,0.9177,1.4853,-0.3580,-1.0325,-1.3813')
    ap.add_argument('--open-m', type=float, default=0.200,
                    help='正式 0.200；校正趟用 0.020')
    ap.add_argument('--band-half-m', type=float, default=0.005)
    ap.add_argument('--grasp-depth-m', type=float, default=0.01226,
                    help='TCP 超過桿心的距離（沿接近方向）。實測自唯一成功的'
                         '摩擦夾持趟 grip_090257_pull20full')
    ap.add_argument('--standoff-m', type=float, default=0.030)
    ap.add_argument('--pre-tol-m', type=float, default=0.005)
    ap.add_argument('--approach-rate', type=float, default=0.010,
                    help='退讓量收斂速率 m/s')
    ap.add_argument('--opening-floor-tol-m', type=float, default=1e-5,
                    help='關閉限位的數值穿透容差：實測開度在 [−tol, 0) 記為 0。'
                         '關閉帶本身不變（2026-10-04 使用者核准 −10 µm）')
    ap.add_argument('--pre-ramp-mps', type=float, default=0.0,
                    help='接觸前目標的**斜坡**（m/s）：求解節點上線前目標 = 手的'
                         '當下位置（初始誤差 0），上線後以此速率移向接觸前位姿。'
                         '0 = 既有行為（一步到位；初始誤差約 20 mm，啟動擺盪 '
                         '±45 mm，motm_172100）')
    ap.add_argument('--pre-settle-s', type=float, default=0.0,
                    help='接觸前位姿要**連續**在 pre_tol 內這麼久才開始收退讓量'
                         '（0 = 既有行為：單一時刻進入即開始）。求解節點上線後'
                         '的頭幾秒手臂會上下擺盪（±30 mm，停車趟也有）；單點'
                         '判定會在擺盪的瞬間放行，上指壓到桿頂（motm_164012）')
    ap.add_argument('--grip-tol-m', type=float, default=0.002)
    ap.add_argument('--grip-settle-s', type=float, default=0.5)
    ap.add_argument('--close-ramp-s', type=float, default=1.0)
    ap.add_argument('--finger-open', type=float, default=0.0089)
    ap.add_argument('--finger-closed', type=float, default=0.0)
    ap.add_argument('--contact-min-n', type=float, default=0.5)
    ap.add_argument('--attach-hold-s', type=float, default=2.0)
    ap.add_argument('--release-max-n', type=float, default=0.1)
    ap.add_argument('--release-hold-s', type=float, default=0.5)
    ap.add_argument('--inject-abort-at-s', type=float, default=-1.0,
                    help='**測試用**：進入 OPEN 後經過這麼多秒（模擬時間）注入中止，'
                         '原因 test_injected_abort；預設 -1 = 關閉，只供功能確認')
    ap.add_argument('--restow-gain', type=float, default=1.0)
    ap.add_argument('--restow-qd-max', type=float, default=0.30)
    ap.add_argument('--restow-mode', default='j3_last',
                    choices=['j3_last', 'sync'],
                    help='j3_last（既有）：其他軸先收、j3 最後 —— 肩先直立、肘還彎著，'
                         '夾爪會被舉高約 44 cm 再放下（motm_170405 TCP 550→989 mm）。'
                         'sync：各軸同比例收回（關節空間直線、同時到達），j3 由正方向'
                         '單調收到 0、往負方向的命令一律不放行；離線核對 TCP 高度 '
                         '546–560 mm、夾爪單調遠離把手')
    ap.add_argument('--j3-last-tol', type=float, default=0.05,
                    help='其他軸都進到這個距離內才動 j3')
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--timeout-s', type=float, default=900.0)
    ap.add_argument('--out', default=None)
    # ---- **移動中操作（MotM）**：預設關閉 ⇒ 既有行為不變 ----
    ap.add_argument('--motm', action='store_true',
                    help='底盤不停：接近／夾持期間照 √ 剖面走向停位、手臂補償；'
                         '開／關由底盤與手臂分擔（手臂名目姿態偏移 '
                         '--motm-arm-share）；關閉後底盤即開始倒退，放開、退開、'
                         '收臂、交還導航全程不停。**只在開與關之間停頓**')
    ap.add_argument('--motm-park-tol', type=float, default=0.20,
                    help='MotM 的 UNFOLD→ALIGN 底盤距停位上限（底盤此時仍在走）')
    ap.add_argument('--motm-v-cap', type=float, default=0.030)
    ap.add_argument('--motm-a-ref', type=float, default=0.002)
    ap.add_argument('--motm-k-yaw', type=float, default=1.0)
    ap.add_argument('--motm-w-cap', type=float, default=0.10)
    ap.add_argument('--motm-w-vref-pre', type=float, default=None,
                    help='接觸前位姿穩定之前（軟啟動期間）的 w_vref。None = 同 '
                         '--motm-w-vref。提高它讓底盤**只照參考速度走、不被拿來'
                         '修 TCP**：底盤的加速度上限、延遲與 0.9 增益不在模型裡，'
                         '拿它定位會衝過頭（motm_181645：底盤 8→14 mm/s、TCP 過衝 '
                         '6 mm，引發 ±25 mm 垂直擺盪）')
    ap.add_argument('--motm-no-soft-start', action='store_true',
                    help='不用軟啟動（接觸前位姿穩定前也用 MotM 權重）。配合 '
                         '--pre-ramp-mps 與求解節點 --offset-init-static 時'
                         '啟動擺盪已由根源處理；軟啟動的貴手臂反而讓底盤停下'
                         '（motm_175859：51.45 s 停 0.88 s）')
    # 接近時不降到零：以蠕行速度越過停位至多 overshoot（手臂補償），
    # 夾持遲遲不成立才在越過上限處停下。兩個節點必須用同一組值。
    ap.add_argument('--motm-v-creep', type=float, default=0.004)
    ap.add_argument('--motm-overshoot', type=float, default=0.030)
    ap.add_argument('--motm-w-vref', type=float, default=0.03)
    ap.add_argument('--motm-w-qn', type=float, default=0.3)
    ap.add_argument('--motm-w-a', type=float, default=0.01)
    ap.add_argument('--motm-w-p', type=float, default=200.0)
    ap.add_argument('--motm-v-pull', type=float, default=0.018)
    ap.add_argument('--motm-a-travel', type=float, default=0.02,
                    help='開／關行程末端的 √ 減速（m/s²）')
    ap.add_argument('--motm-arm-share', type=float, default=0.06,
                    help='開抽屜時手臂收回量（m）；關抽屜時再伸回')
    ap.add_argument('--motm-retreat-rate', type=float, default=None,
                    help='T2：RETREAT 段 TCP 退開量的斜坡速率 m/s（只在釋放確認後的 RETREAT 生效；'
                         '預設 None ＝ 沿用 --approach-rate，與既有行為相同）')
    ap.add_argument('--motm-v-back', type=float, default=0.008,
                    help='關閉後底盤倒退速度（m/s）')
    ap.add_argument('--motm-acc-lin', type=float, default=0.05)
    ap.add_argument('--motm-acc-ang', type=float, default=0.20)
    ap.add_argument('--park-fixed', action='store_true',
                    help='PARK_FIXED：收到 /park/violation 或本節點生成非零底盤命令 ⇒ 走既有中止收尾')
    a = ap.parse_args()
    if a.park_fixed and a.motm:
        print('[task] **--park-fixed 與 --motm 不可同時指定** ⇒ 拒絕啟動', flush=True)
        return 3
    park = [float(v) for v in a.park.split(',')]
    start = [float(v) for v in a.start.split(',')]
    qg = np.array([float(v) for v in a.q_grasp.split(',')])

    K = WholeBodyKinematics.from_urdf_file(a.urdf)
    Tg = K.fk(np.array(park + list(qg), float), a.tcp)
    a.R_grasp = Tg[:3, :3].copy()

    rclpy.init()
    nd = Task(a, K)
    band = (a.open_m - a.band_half_m, a.open_m + a.band_half_m)
    pol = DrawerTaskPolicy(DrawerTaskConfig(
        open_band_m=band,
        **({'park_tol_m': a.motm_park_tol} if a.motm else {})))
    rep = {'scope': ('相位由 DrawerTaskPolicy 決定，推進一律看實測；'
                     'undesignated_contact **沒有量測來源**，一律 False'),
           'open_m': a.open_m, 'open_band_m': list(band),
           'args': {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in vars(a).items() if k != 'out'},
           'events': [], 'trace': []}

    # 等量測到齊；第一筆把手位姿到了才能定 tcp_offset
    t0 = time.monotonic()
    while rclpy.ok() and (nd.H is None or nd.pose is None or nd.q is None):
        rclpy.spin_once(nd, timeout_sec=0.05)
        if time.monotonic() - t0 > 120:
            print('[task] **量測不齊** ⇒ 中止', flush=True)
            return 2
    _fk_off = float(nd.H[2][1, 3]) - float(Tg[1, 3])
    # **抓取深度用實測的摩擦夾持關係，不用設計姿態的 FK。**
    #
    # 設計姿態（附著模型用的）算出來是「把手中心在 TCP **之外** 14.7 mm」，
    # 但指墊網格只延伸到夾爪座標 z = 81.1 mm、TCP 在 83.6 mm ⇒ 把手中心落在
    # 98.3 mm，**在指尖外 17.2 mm**。實跑 motm_cal20_073038 的手指閉合到
    # 1e-6 m、接觸為零 —— 夾到的是桿子前面的空氣。
    #
    # 摩擦夾持唯一成功過的一趟 grip_090257_pull20full（接觸 2.39 N）：
    # 結束開度 16.31 mm ⇒ 把手中心 y = 8.715 − 0.01631 = 8.69869，
    # TCP y = 8.71095 ⇒ **TCP 超過桿心 12.26 mm**（夾爪座標 z = 71.3 mm，
    # 落在指墊 51.6–81.1 mm 中段）。拉動期間滑移 < 0.03 mm，所以這就是
    # 抓取時的關係。
    a.tcp_offset_m = -float(a.grasp_depth_m)
    rep['tcp_offset_m'] = round(a.tcp_offset_m, 6)
    rep['tcp_offset_fk_design_m'] = round(_fk_off, 6)
    rep['tcp_offset_source'] = (
        'grip_090257_pull20full 實測：TCP 超過桿心 12.26 mm；'
        '設計姿態 FK 的 +14.7 mm 會讓桿子落在指尖外 17.2 mm，不採用')
    print(f'[task] 抓取深度：TCP 超過桿心 {-a.tcp_offset_m*1000:.2f} mm'
          f'（設計姿態 FK 給的是 {-_fk_off*1000:+.2f}，不採用）；開到 '
          f'{a.open_m*1000:.0f} mm（帶 {band[0]*1000:.0f}–{band[1]*1000:.0f}）',
          flush=True)

    # 求解節點起來之前就先發接觸前目標 —— 它一上線就有東西可跟，
    # 不必走「保持起始 TCP」的退路
    stage = 'pre'          # ALIGN 內部：pre → approach
    H_frozen = None        # 閉合開始時凍結的把手位姿
    standoff = a.standoff_m
    settle_t0 = close_t0 = release_t0 = pre_t0 = None
    pre_ramp_p = None      # 接觸前目標斜坡的當下位置
    n_floor_clamped = 0    # 開度在 [−tol, 0) 記為 0 的次數
    floor_min = 0.0
    grip_now = a.finger_open
    dtgt = None
    last_T = None
    grasp_y = None
    started = False
    handback_sent = home_sent = stop_sent = False
    inj_open_t0 = None              # 注入中止：進入 OPEN 的時刻
    prev_phase = None
    dt = 1.0 / a.rate
    # ---- MotM 狀態 ----
    pull_ax = np.asarray(DrawerTargetConfig().axis(), float)
    vr = None              # 斜坡後的底盤參考速度（本體）
    home_plan_msg = None   # MotM：提早發的回程計畫（分次發，不阻塞）
    home_plan_n = 0
    retreat_so = 0.0       # MotM：退開量斜坡
    q_att = q_ret = None   # 夾持當下的手臂姿態／收回 X 的名目姿態
    rep['motm'] = ({k[5:]: v for k, v in vars(a).items()
                    if k.startswith('motm_')} if a.motm else None)
    if a.motm:
        rep['motm_trace'] = []

    def motm_coord(phase):
        """本相位的 (底盤參考速度目標, q_nom, w_qn)。"""
        yaw = nd.pose[2]
        if phase in ('NAVIGATE', 'UNFOLD', 'ALIGN', 'ENGAGE_WAIT'):
            v, _ = MC.approach_vref(nd.pose, park, v_cap=a.motm_v_cap,
                                    a_ref=a.motm_a_ref, k_yaw=a.motm_k_yaw,
                                    w_cap=a.motm_w_cap,
                    v_creep=a.motm_v_creep, overshoot=a.motm_overshoot)
            return np.array(v), None, 0.0
        # 名目姿態**隨實測開度**由夾持姿態內插到收回姿態：開度 0 時 = 夾持
        # 姿態、開到位時 = 收回 X。一開始就把名目設成收回姿態，手臂會在
        # 拉開的第一秒被猛拉（motm_163548：j3 0.23 rad/s、TCP 37 mm/s）。
        frac = float(nd.H[1]) / max(a.open_m, 1e-6)
        qn_open = (None if q_att is None else
                   MC.lerp_posture(q_att, q_ret, frac))
        if phase == 'OPEN':
            spd = MC.sqrt_profile(a.open_m - float(nd.H[1]), a.motm_v_pull,
                                  a.motm_a_travel)
            return (np.array(MC.axis_vref(yaw, pull_ax, spd)), qn_open,
                    a.motm_w_qn)
        if phase == 'OPEN_HOLD':
            return np.zeros(3), qn_open, a.motm_w_qn
        if phase == 'CLOSE':
            spd = MC.sqrt_profile(float(nd.H[1]), a.motm_v_pull,
                                  a.motm_a_travel)
            return (np.array(MC.axis_vref(yaw, -pull_ax, spd)), qn_open,
                    a.motm_w_qn)
        if phase in ('CLOSE_HOLD', 'RELEASE_WAIT', 'RETREAT', 'RESTOW',
                     'HANDBACK_WAIT'):
            return (np.array(MC.axis_vref(yaw, pull_ax, a.motm_v_back)),
                    None, 0.0)
        return np.zeros(3), None, 0.0

    def motm_step(phase, st):
        """斜坡更新底盤參考速度並發協同設定。回傳斜坡後的參考速度。"""
        nonlocal vr
        tgt, qn, wq = motm_coord(phase)
        if vr is None or phase in ('NAVIGATE', 'UNFOLD'):
            vr = tgt.copy()        # 與展開節點同一個剖面值，不經斜坡
        else:
            lim = np.array([a.motm_acc_lin, a.motm_acc_lin,
                            a.motm_acc_ang]) * dt
            vr = vr + np.clip(tgt - vr, -lim, lim)
        # **軟啟動**：求解節點剛上線的頭幾秒手臂會大幅擺盪（MotM 權重下
        # 垂直 −63…+48 mm、朝桿 +30 mm，motm_164012／164904；指尖撞桿面
        # 92.7 N）。接觸前位姿穩定之前用**啟動值**（已驗證的 w_a／w_p），
        # 開始收退讓量時才換成 MotM 權重。
        soft = (not a.motm_no_soft_start) and (
            phase in ('NAVIGATE', 'UNFOLD')
            or (phase == 'ALIGN' and stage == 'pre'))
        m = Float64MultiArray()
        _wv = (a.motm_w_vref_pre if (soft and a.motm_w_vref_pre is not None)
               else a.motm_w_vref)
        m.data = MC.coord_msg(w_vref=_wv, vref=vr,
                              w_qn=wq if qn is not None else 0.0,
                              q_nom=qn if qn is not None else None,
                              w_a=None if soft else a.motm_w_a,
                              w_p=None if soft else a.motm_w_p)
        nd.coord_pub.publish(m)
        if int(st * 10) % 2 == 0:
            rep['motm_trace'].append({
                'sim_t': round(st, 3), 'phase': phase,
                'vref': [round(float(x), 5) for x in vr],
                'vref_target': [round(float(x), 5) for x in tgt],
                'w_qn': wq if qn is not None else 0.0, 'soft_start': soft})
        return vr
    t1 = time.monotonic()
    while rclpy.ok() and time.monotonic() - t1 < a.timeout_s:
        # **把待處理的回呼全部處理完**：spin_once 一次只處理一筆，數個
        # 50 Hz 的狀態流在 20 Hz 迴圈裡會各自落後好幾個週期。
        for _ in range(64):
            rclpy.spin_once(nd, timeout_sec=0.0)
        st = nd.sim_t()
        # ---- PARK_FIXED：違規閂鎖 **最先檢查**（在任何目標發布、求解器啟動與提前 continue 之前）----
        # 執行端違規或本節點生成非零底盤命令 ⇒ 走既有中止收尾：停求解、不發目標、夾爪維持、不追加恢復動作。
        if nd.park_fixed and (nd.park_violation is not None or nd.park_base_bad is not None):
            _why = (f'park_fixed_violation：{nd.park_violation.get("why")}'
                    if nd.park_violation is not None else 'park_fixed_task_base_cmd_nonzero')
            rep['abort'] = _why
            rep['park_violation'] = nd.park_violation
            rep['park_base_bad'] = nd.park_base_bad
            rep['park_abort_phase'] = pol.phase
            print(f'[task] **中止：{_why}** @ sim {st} 相位 {pol.phase}', flush=True)
            if not stop_sent:
                m = String()
                m.data = f'drawer_task 中止：{_why}'
                nd.stop_pub.publish(m)
                stop_sent = True
                rep['events'].append({'sim_t': st, 'solver_stop': True, 'reason': f'abort：{_why}'})
                rep['abort_stop'] = {'abort_sim_t': st, 'stop_sent_sim_t': nd.sim_t(), 'reason': _why}
                for _ in range(10):
                    rclpy.spin_once(nd, timeout_sec=0.05)
            break
        if st is None or nd.H is None:
            time.sleep(dt)
            continue
        Tm = nd.tcp()
        W_T_H = nd.H[2]
        d_meas = float(nd.H[1])
        # **限位處的數值穿透**：抽屜頂到關閉限位時 PhysX 以微小穿透產生反力，
        # 開度量到 −2 µm（motm_185942）。關閉帶 (0.000, 0.005) 是凍結判準、不動；
        # 這裡只定義量測：[−tol, 0) 記為 0。比 −tol 更負的照原值（仍在帶外）。
        if -a.opening_floor_tol_m <= d_meas < 0.0:
            d_pol = 0.0
            n_floor_clamped += 1
            floor_min = min(floor_min, d_meas)
        else:
            d_pol = d_meas
        owner = (nd.hs or {}).get('owner')

        # ---- 求解節點上線前 ----
        # 兩件事要分開處理：
        #
        # (1) **UNFOLD → ALIGN 要在展開完成、求解節點還沒動之前判定。**
        #     條件是「底盤 ≤ 1 cm、手臂 ≤ 0.02 rad、把手讀值有效」。先前等
        #     求解節點上線才餵，它一上線就把機器人往退讓位姿移（底盤 25 mm、
        #     手臂為了補下垂也動），條件就再也不成立 —— 實測 motm_cal20 停在
        #     UNFOLD、最後判停滯。展開節點「保持交棒」那段窗口裡條件是真的。
        #     **只在條件成立時才推進**，NAVIGATE→UNFOLD→ALIGN 連續兩步完成，
        #     不在等待中累積停滯計數。
        # (2) **進了 ALIGN 就暫停推進**，直到求解節點上線。ALIGN 屬於「移動
        #     中」相位，命令近零超過 40 週期判停滯；求解節點起來之前命令本來
        #     就是零。相位機的停滯計數只在 step() 時累加，所以暫停就不會誤判。
        if not started:
            T = nd.target_at(a.standoff_m)
            if a.pre_ramp_mps > 0.0 and Tm is not None:
                # 求解節點上線前：目標 = 手的當下位置（姿態用抓取姿態）
                T = T.copy()
                T[:3, 3] = Tm[:3, 3]
                pre_ramp_p = Tm[:3, 3].copy()
            nd.publish_target(T)
            last_T = T
            if pol.phase != 'ALIGN' and nd.pose is not None and nd.q is not None:
                _bpe = math.hypot(nd.pose[0] - park[0], nd.pose[1] - park[1])
                _ape = float(np.max(np.abs(nd.q - qg)))
                if (_bpe <= (a.motm_park_tol if a.motm else 0.01)
                        and _ape <= 0.02 and st - nd.H[0] < 0.2):
                    for _ in range(3):
                        _o = pol.step(DrawerState(
                            sim_t=st, state_age_s=0.0, opening_m=float(nd.H[1]),
                            base_park_err_m=_bpe, arm_posture_err_rad=_ape,
                            handover_zone_dist_m=_bpe, cmd_max_abs=1.0,
                            handle_reading_valid=True))
                        if pol.phase == 'ALIGN':
                            break
                    rep['events'].append({
                        'sim_t': st, 'phase': pol.phase,
                        'base_park_err_m': _bpe, 'arm_posture_err_rad': _ape,
                        'reason': '展開完成、求解節點上線前判定'})
                    print(f'[task] 相位 → **{pol.phase}** @ sim {st:.2f}'
                          f'（底盤 {_bpe*1000:.1f} mm、手臂 {_ape:.4f} rad；'
                          f'展開完成、求解節點上線前判定）', flush=True)
                    prev_phase = pol.phase
            if a.motm and nd.pose is not None:
                motm_step(pol.phase, st)
            if pol.phase == 'ALIGN':
                nd.solver_start_pub.publish(Bool(data=True))
            if pol.phase == 'ALIGN' and nd.n_ws > 0:
                started = True
                rep['events'].append({'sim_t': st, 'solver_up': True})
                print(f'[task] 求解節點已上線 @ sim {st:.2f} ⇒ 恢復推進',
                      flush=True)
            time.sleep(dt)
            continue

        # ---- 組實測狀態 ----
        ep = er = None
        phase = pol.phase
        T_ref = nd.target_at(0.0)          # 抓取位姿（相位機看的是對它的誤差）
        if Tm is not None:
            if phase in ('ALIGN', 'ENGAGE_WAIT'):
                ep = float(np.linalg.norm(Tm[:3, 3] - T_ref[:3, 3]))
                er = so3_angle(T_ref[:3, :3].T @ Tm[:3, :3])
            elif last_T is not None:
                ep = float(np.linalg.norm(Tm[:3, 3] - last_T[:3, 3]))
                er = so3_angle(last_T[:3, :3].T @ Tm[:3, :3])
        n1 = n2 = float('nan')
        if nd.gs is not None:
            _, f1, f2, n1, n2 = nd.gs
        # 使用執行端逐物理步累積的證據；不從 20 Hz 取樣自行推論連續性。
        attach_ready = contact_established(
                    nd.gs_evidence, now=st, close_started=close_t0,
                    contact_min_n=a.contact_min_n, hold_s=a.attach_hold_s)
        if phase == 'ENGAGE_WAIT' and not attach_ready:
            dtgt = None
        if (phase == 'ENGAGE_WAIT' and dtgt is None and Tm is not None
                and attach_ready):
            dtgt = DrawerTarget(Tm, W_T_H, DrawerTargetConfig())
            grasp_y = float(Tm[1, 3])
            if a.motm:
                q_att = nd.q.copy()
                q_ret, _res = MC.shifted_posture(
                    K, nd.pose[:3], q_att, a.motm_arm_share * pull_ax, a.tcp)
                rep['events'].append({
                    'sim_t': st, 'motm_q_att': q_att.tolist(),
                    'motm_q_ret': q_ret.tolist(), 'ik_residual_m': _res,
                    'base_at_attach': list(nd.pose[:3])})
                print(f'[task] MotM 名目姿態：手臂收回 '
                      f'{a.motm_arm_share*1000:.0f} mm（IK 殘差 '
                      f'{_res*1000:.3f} mm）', flush=True)
            rep['events'].append({
                'sim_t': st, 'attached': True, 'n1': n1, 'n2': n2,
                'opening_m': d_meas, 'contact_evidence': list(nd.gs_evidence),
                'source': 'every_physics_step'})
            print(f'[task] **夾持成立** 逐物理步連續 '
                  f'{nd.gs_evidence[6]:.3f} s；接觸 {n1:.2f}／{n2:.2f} N',
                  flush=True)
        # **夾持「維持」與「建立」用不同判準。**
        # 建立：兩指都 ≥ contact_min_n、連續 attach_hold_s（沿用唯一成功過的
        #       摩擦夾持趟 grip_090257_pull20full 那一組）。
        # 維持：**總夾持力** ≥ contact_min_n。推回抽屜時負載換邊，一指的
        #       正向力會掉 —— motm_cal20_074420 在 CLOSE_HOLD 因「兩指都
        #       ≥ 0.5 N」不成立被判滑脫中止，而當下**漂移只有 0.35–0.40 mm**
        #       （限值 10 mm）。真正的滑脫由相位機的漂移判準管，那一條沒有動。
        #
        #       **不用手指位置判「桿子還在中間」。** 原本以為夾住時每指停在
        #       約 5 mm（桿半徑），實測夾住時手指只在 0.03–0.4 mm、接觸力
        #       2.4 N（驅動 kp 1e4 × 0.3 mm ≈ 3 N 吻合）—— 碰撞網格合攏時的
        #       間隙約等於桿徑，手指位置分不出有沒有夾到東西。桿子真的脫出
        #       時接觸力會歸零，所以總夾持力就是判準。
        held = (math.isfinite(n1) and math.isfinite(n2)
                and (n1 + n2) >= a.contact_min_n)
        drift = None
        if dtgt is not None and Tm is not None:
            drift = float(dtgt.drift_vs(Tm, W_T_H))
        cmax = (max(abs(x) for x in nd.ap[1]) if nd.ap else 0.0)
        bpe = (math.hypot(nd.pose[0] - park[0], nd.pose[1] - park[1])
               if nd.pose else None)
        ape = (float(np.max(np.abs(nd.q - qg))) if nd.q is not None else None)
        stow_err = (float(np.max(np.abs(nd.q - STOW)))
                    if nd.q is not None else None)
        home = (math.hypot(nd.pose[0] - start[0], nd.pose[1] - start[1])
                if nd.pose else None)
        ret = (float(grasp_y - Tm[1, 3]) if (grasp_y is not None
                                              and Tm is not None) else 0.0)
        ages = [st - x for x in (nd.H[0], nd.pose[3] if nd.pose else None)
                if x is not None]
        s = DrawerState(
            sim_t=st, state_age_s=max(ages) if ages else 1e9,
            opening_m=d_pol, pos_err_m=ep, rot_err_rad=er,
            grasp_drift_m=drift, base_park_err_m=bpe,
            arm_posture_err_rad=ape, handover_zone_dist_m=bpe,
            retreat_signed_m=ret, cmd_max_abs=cmax,
            stow_err_rad=stow_err, home_dist_m=home,
            nav_in_control=(owner == 'nav'),
            # **RELEASE_WAIT 中 attached 報的是邏輯連接狀態**，一直為真到
            # 「確認解除」那一步。這是相位機測試定下的語意：D2 規定此相位
            # attached=False ⇒ 判滑脫中止；轉換測試用 attached=True 且
            # decouple_confirmed=True ⇒ 進 RETREAT。也就是說「沒有確認解除
            # 就失去連接」才是滑脫；故意張開手指造成的接觸歸零，要經由
            # decouple_confirmed 結束，不是經由 attached=False。
            # motm_cal20_074832 就是在這裡被誤判中止。
            # 這一相位的滑脫仍由漂移判準把關（目標凍結在最後一筆，抽屜不動）。
            attached=(dtgt is not None
                      and (held or phase == 'RELEASE_WAIT')),
            decouple_confirmed=(release_t0 is not None
                                and st - release_t0 >= a.release_hold_s),
            handle_reading_valid=(st - nd.H[0] < 0.2),
            force_monitor_ok=(nd.gs is None or phase not in pol.GRASPED
                              or (math.isfinite(n1) and math.isfinite(n2))),
            cmd_stale_or_fail_latched=(nd.fail is not None),
            undesignated_contact=False)
        out = pol.step(s)
        phase = out['phase']
        if phase != prev_phase:
            print(f'[task] 相位 {prev_phase} → **{phase}** @ sim {st:.2f}'
                  f'  開度 {d_meas*1000:6.1f} mm  {out.get("reason", "")}',
                  flush=True)
            rep['events'].append({'sim_t': st, 'phase': phase,
                                  'opening_m': d_meas,
                                  'reason': out.get('reason', '')})
            m = String()
            m.data = phase
            nd.phase_pub.publish(m)
            prev_phase = phase
        # ---- 測試用注入中止（預設關閉；只供功能確認，原因明記，不冒稱實際滑脫）----
        if a.inject_abort_at_s >= 0.0 and phase == 'OPEN' and not out.get('abort'):
            if inj_open_t0 is None:
                inj_open_t0 = st
            if st - inj_open_t0 >= a.inject_abort_at_s:
                out['abort'] = 'test_injected_abort'
                rep['injected_abort'] = {'sim_t': st, 'phase': phase,
                                         'open_elapsed_s': round(st - inj_open_t0, 3),
                                         'opening_m': d_meas}
        if out.get('abort'):
            rep['abort'] = out['abort']
            print(f'[task] **中止：{out["abort"]}** @ sim {st:.2f}', flush=True)
            # **中止 ⇒ 停止任務推進**：通知求解節點停止（不再求解、不再發布），
            # 本節點也不再發目標／協同設定。夾爪維持目前命令（不自動放開→退開→收臂：
            # 中止可能代表夾持已失效或未知接觸，正常恢復路徑未必安全）。
            if not stop_sent:
                m = String()
                m.data = f'drawer_task 中止：{out["abort"]}'
                nd.stop_pub.publish(m)
                stop_sent = True
                rep['events'].append({'sim_t': st, 'solver_stop': True,
                                      'reason': f'abort：{out["abort"]}'})
                rep['abort_stop'] = {'abort_sim_t': st, 'stop_sent_sim_t': nd.sim_t(),
                                     'reason': out['abort']}
                # 讓停止訊息確實送出再離開（有界：最多約 0.5 s 牆鐘）
                for _ in range(10):
                    rclpy.spin_once(nd, timeout_sec=0.05)
            break
        if out.get('done') or phase == 'DONE':
            print(f'[task] **完成** @ sim {st:.2f}', flush=True)
            break

        vr_now = motm_step(phase, st) if (a.motm and nd.pose is not None) \
            else None

        # ---- 依相位動作 ----
        if phase in ('NAVIGATE', 'UNFOLD'):
            T = nd.target_at(a.standoff_m)
            nd.publish_target(T)
            last_T = T
            nd.grip(a.finger_open)
        elif phase == 'ALIGN':
            if stage == 'pre':
                T = nd.target_at(a.standoff_m)
                _in = (Tm is not None and np.linalg.norm(
                    Tm[:3, 3] - T[:3, 3]) < a.pre_tol_m)
                if a.pre_ramp_mps > 0.0 and pre_ramp_p is not None:
                    # 發出去的目標以斜坡移向接觸前位姿；到位判定仍對**最終**位姿
                    _d = T[:3, 3] - pre_ramp_p
                    _n = float(np.linalg.norm(_d))
                    _st = a.pre_ramp_mps * dt
                    pre_ramp_p = (T[:3, 3].copy() if _n <= _st
                                  else pre_ramp_p + _d * (_st / _n))
                    T = T.copy()
                    T[:3, 3] = pre_ramp_p
                    _in = _in and _n <= _st
                pre_t0 = (pre_t0 if pre_t0 is not None else st) if _in else None
                if _in and st - pre_t0 >= a.pre_settle_s:
                    stage = 'approach'
                    rep['events'].append({'sim_t': st, 'align': 'approach'})
                    print(f'[task] 接觸前位姿到達 ⇒ 開始收退讓量', flush=True)
            else:
                standoff = max(0.0, standoff - a.approach_rate * dt)
                T = nd.target_at(standoff)
            nd.publish_target(T)
            last_T = T
            nd.grip(a.finger_open)
        elif phase == 'ENGAGE_WAIT':
            # 閉合開始後用**凍結**的把手位姿；閉合前（手指張開、無接觸）照常跟隨
            T = nd.target_at(0.0, H=H_frozen)
            nd.publish_target(T)
            last_T = T
            if close_t0 is None:
                ok = ep is not None and ep < a.grip_tol_m
                if ok:
                    settle_t0 = settle_t0 if settle_t0 is not None else st
                    if st - settle_t0 >= a.grip_settle_s:
                        close_t0 = st
                        H_frozen = W_T_H.copy()
                        rep['events'].append({
                            'sim_t': st, 'target_frozen': True,
                            'opening_m': d_meas})
                        rep['events'].append({'sim_t': st, 'grip': 'close',
                                              'err_mm': ep * 1e3})
                        print(f'[task] 到位 {ep*1e3:.2f} mm ⇒ 閉合手指',
                              flush=True)
                else:
                    settle_t0 = None
                grip_now = a.finger_open
            else:
                frac = min(1.0, (st - close_t0) / a.close_ramp_s)
                grip_now = a.finger_open + frac * (a.finger_closed
                                                   - a.finger_open)
            nd.grip(grip_now)
        elif phase in ('OPEN', 'OPEN_HOLD', 'CLOSE', 'CLOSE_HOLD'):
            goal = a.open_m if phase in ('OPEN', 'OPEN_HOLD') else 0.0
            T, _diag = dtgt.step(W_T_H, d_meas, goal, dt)
            nd.publish_target(T)
            last_T = T
            nd.grip(a.finger_closed)
        elif phase == 'RELEASE_WAIT':
            nd.publish_target(last_T)
            nd.grip(a.finger_open)
            opened = (nd.gs is not None
                      and min(nd.gs[1], nd.gs[2]) >= 0.8 * a.finger_open)
            free = (math.isfinite(n1) and math.isfinite(n2)
                    and n1 < a.release_max_n and n2 < a.release_max_n)
            if opened and free:
                release_t0 = release_t0 if release_t0 is not None else st
            else:
                release_t0 = None
        elif phase == 'RETREAT':
            if a.motm:
                # 退開量照進場速率斜坡增加；一次跳到退讓量會讓手臂衝到速率
                # 上限（motm_165219：j3 −1.0 rad/s、底盤 −35 mm/s）
                # T2：退開段**專用**的斜坡速率（只在 RETREAT 生效；未給 ＝ 沿用 approach_rate，行為不變）
                _rr = a.motm_retreat_rate if a.motm_retreat_rate is not None else a.approach_rate
                retreat_so = min(a.standoff_m, retreat_so + _rr * dt)
                T = nd.target_at(retreat_so)
            else:
                T = nd.target_at(a.standoff_m)
            nd.publish_target(T)
            last_T = T
            nd.grip(a.finger_open)
        elif phase == 'RESTOW':
            if not stop_sent:
                m = String()
                m.data = 'drawer_task：退開完成，改由關節空間收臂'
                nd.stop_pub.publish(m)
                stop_sent = True
                rep['events'].append({'sim_t': st, 'solver_stop': True})
            u = np.zeros(9)
            if a.motm:
                if vr_now is not None:
                    u[:3] = vr_now        # 收臂時底盤繼續倒退，不停
                # **回程計畫提早發**：交還導航的當下導航必須已在出命令，
                # 否則切換當步是零命令，底盤當場停住。
                # **不得阻塞**：publish_plan 為了重發會 sleep 0.5 s，迴圈停發
                # 底盤命令 ⇒ 命令鏈逾時送零，底盤停 0.35 s（motm_165219）。
                # 這裡組一次、之後每個週期發一次，共 5 次。
                if home_plan_msg is None and nd.pose is not None:
                    home_plan_msg = nd.build_plan(nd.pose[:2], start[:2],
                                                  start[2])
                    home_sent = True
                    rep['events'].append({'sim_t': st,
                                          'home_plan': len(home_plan_msg.poses),
                                          'early_for_motm': True})
                    print(f'[task] 回起點計畫提早發（收臂中）：'
                          f'{len(home_plan_msg.poses)} 點', flush=True)
                if home_plan_msg is not None and home_plan_n < 5:
                    nd.plan_pub.publish(home_plan_msg)
                    home_plan_n += 1
            if nd.q is not None:
                err = STOW - nd.q
                others = [0, 1, 3, 4, 5]
                if a.restow_mode == 'sync':
                    # **同比例**：整個向量一起縮放，各軸同時到達（不逐軸截斷 ——
                    # 逐軸截斷會讓誤差大的軸落後，路徑就不是直線）
                    qd = a.restow_gain * err
                    mx = float(np.max(np.abs(qd)))
                    if mx > a.restow_qd_max:
                        qd = qd * (a.restow_qd_max / mx)
                    # j3 收攏值離有效下限只有 0.011 rad：**不准往負方向推過收攏值**
                    if nd.q[2] <= STOW[2] and qd[2] < 0.0:
                        qd[2] = 0.0
                else:
                    qd = np.clip(a.restow_gain * err, -a.restow_qd_max,
                                 a.restow_qd_max)
                    # **j3 最後動**：其他軸都進到 j3_last_tol 內才放行
                    if np.max(np.abs(err[others])) > a.j3_last_tol:
                        qd[2] = 0.0
                u[3:] = qd
            nd.wb_cmd(u)
        elif phase == 'HANDBACK_WAIT':
            _u = np.zeros(9)
            if a.motm and vr_now is not None:
                _u[:3] = vr_now               # MotM：交還前底盤繼續走
            nd.wb_cmd(_u)                     # 交還前持續送命令，鏈保持新鮮
            if not handback_sent and nd.ap is not None:
                m = String()
                at = int(nd.ap[2]) + 5
                m.data = json.dumps({'to': 'nav', 'at_step': at,
                                     'window_steps': 400})
                nd.ho_pub.publish(m)
                handback_sent = True
                rep['events'].append({'sim_t': st, 'handback_req': at})
                print(f'[task] 請求交還導航 @ 物理步 {at}', flush=True)
        elif phase == 'NAVIGATE_HOME':
            if not home_sent and nd.pose is not None:
                n = nd.publish_plan(nd.pose[:2], start[:2], start[2])
                home_sent = True
                rep['events'].append({'sim_t': st, 'home_plan': n})
                print(f'[task] 回起點計畫已發：{n} 點', flush=True)

        if int(st * 10) % 5 == 0:
            rep['trace'].append({
                'sim_t': round(st, 3), 'phase': phase,
                'opening_m': round(d_meas, 5),
                'err_mm': None if ep is None else round(ep * 1e3, 3),
                'drift_mm': None if drift is None else round(drift * 1e3, 3),
                'n1': None if not math.isfinite(n1) else round(n1, 3),
                'n2': None if not math.isfinite(n2) else round(n2, 3),
                'f1': None if nd.gs is None else round(nd.gs[1] * 1e3, 3),
                'f2': None if nd.gs is None else round(nd.gs[2] * 1e3, 3),
                'owner': owner})
        time.sleep(dt)

    rep['opening_floor'] = {'tol_m': a.opening_floor_tol_m,
                            'n_cycles_clamped': n_floor_clamped,
                            'min_raw_in_tol_m': floor_min,
                            'note': '只影響送進相位機的開度；關閉帶未改'}
    rep['state_source'] = ('drawer/sync_state（同一物理步）' if nd.use_sync
                           else '分開的三個話題')
    rep['n_sync'] = nd.n_sync
    rep['policy'] = pol.summary()
    rep['final_phase'] = pol.phase
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1,
                  default=float)
    print(f'[task] 結束；最終相位 {pol.phase}'
          f'{"；中止 " + str(rep.get("abort")) if rep.get("abort") else ""}',
          flush=True)
    nd.destroy_node()
    rclpy.try_shutdown()
    return 0 if pol.phase == 'DONE' and not rep.get('abort') else 1


if __name__ == '__main__':
    raise SystemExit(main())
