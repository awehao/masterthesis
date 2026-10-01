"""WG2 的**薄執行介面**：把 WG1 的純數值核心接上既有安全鏈。

設計界線（依指導，**不複製 F18 那套雙 executor 框架**）
-------------------------------------------------------
* **單一 executor 所有權**：只有 `main()` 建立並 spin 一個
  `SingleThreadedExecutor`，節點只加入它一次。
  **不使用** `TransformListener(spin_thread=True)` ——
  那會把同一個節點加進第二個 executor，正是 F18 的成因。
  本節點也**完全不用 TF**：狀態只來自 `/joint_states` 與 `/odom`。
* **不可變狀態快照**：回呼只做一件事 —— 組出一個**新的** `Snap`
  並原子地指派給 `self._snap`。求解迴圈把它讀進區域變數一次，
  整輪不再讀 `self._snap`。**沒有任何跨回呼共用的可變陣列。**
  （F18 的 `_on_pts` 正是在兩次存取之間被換掉。）
* 核心保持純數值：本檔不含任何模型、成本或約束程式碼。

座標（**關鍵，避免雙重轉換**）
------------------------------
    核心輸出 u = (v_x^B, v_y^B, ω, q̇…)   —— **本體**座標
    /wholebody_safety/cmd_in              —— **世界**座標（安全鏈的慣例）
    adapter 再做 world → body 給 /wb_vel_cmd

所以本節點在發布前要做 **body → world**（乘 Rz(θ)）。
**少做這一步，adapter 的轉換就會把命令轉錯一次。**

命令的階段（**正名**）
----------------------
    request            = 本節點發到 /wholebody_safety/cmd_in 的（世界）
    modified           = 安全層發到 /wholebody_safety/cmd_out 的（世界）
    endpoint_requested = adapter 發到 /wb_vel_cmd 的（本體）
                         —— **位於 E2 之前**，之後還可能有輪級修改、
                         逾時處置或閂鎖 ⇒ **不是「實際套用」**
    applied            = 執行端 /coman/applied_cmd（**E2 之後**，
                         真正送進物理 API；附 physics_step_id 與 sim_t）

`u_prev` 的權威來源是 **applied**。
`--u-prev-policy strict`（正式閉迴路）：**等**有效的 applied 回報；
沒有就不發布，不以 endpoint_requested 或零值冒稱真實歷史。
初始零值**只在明確確認初始靜止時**使用（`--assume-initial-rest`）。
`--u-prev-policy diagnostic` 保留替代值，但每筆都標來源，
且該模式**不算這項驗證通過**。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.arm_pregrasp import ARM_JOINTS          # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import (                     # noqa: E402
    NU, WGMPCConfig, body_to_world, solve, task_error)
from ammr_wholebody_mpc.wgmpc_core_sp import (                   # noqa: E402
    ArmSetpointModel, WGMPCConfigSP, make_z, plant_phys_step, solve_sp)
from ammr_wholebody_mpc.wholebody_kinematics import (            # noqa: E402
    WholeBodyKinematics)
from wgmpc_sp_handshake import (ARMED, FAILED, HOLD, INIT,       # noqa: E402
                                SetpointGate, SpSample)


@dataclass(frozen=True)
class Snap:
    """**不可變、同時刻**狀態快照。回呼只建新的，不就地改。

    `sim_t` 是**三方共同的**模擬時間，不是「最舊者」。
    先前用 `min(t_arm, t_base)` 合成會把錯配藏起來：兩個來源各自新鮮、
    但不是同一個物理步，取最舊時間後看起來仍然新鮮。
    現在只有在**共同時間鍵**上三方都有樣本才建快照，
    並保留各來源自己的時間供事後核對。
    """
    q: tuple           # 9 維（底盤 3 ＋ 手臂 6）
    sim_t: float       # **共同**模擬時間
    recv_mono: float   # 收到時的單調牆鐘（取最舊者，僅用於診斷）
    have_js: bool
    have_odom: bool
    t_arm: float = float('nan')     # 各來源自己的時間（應與 sim_t 相同）
    t_base: float = float('nan')
    t_sp: float = float('nan')
    s: tuple = ()                   # 手臂設定點（6），無設定點模式時為空
    sp_step_id: int = -1
    sp_exec_mode: int = -1
    sp_api_applied: bool = False
    paired_sources: tuple = ()


class WGMPCNode(Node):
    def __init__(self, a):
        super().__init__('wgmpc_wg2')
        # **同一時基**：狀態訊息的時間戳是模擬時間，
        # 所以本節點的時鐘也必須是模擬時間，否則
        # `sim_now() - snap.sim_t` 會變成「牆鐘 − 模擬時間」，
        # 每輪都判過期（實測：0 輪發布）。
        from rclpy.parameter import Parameter as _P
        self.set_parameters([_P('use_sim_time', _P.Type.BOOL, True)])
        self.a = a
        self.K = WholeBodyKinematics.from_urdf_file(a.urdf)
        if a.arm_model == 'setpoint':
            # **增廣核心**：手臂設定點納入狀態。α／b 由辨識檔讀入，
            # 不在程式裡硬編碼；列縮放沿用 WG2-ARM-SP-1 的核准值。
            _id = json.load(open(a.arm_ident))
            self.arm_model = ArmSetpointModel(
                alpha=_id['alpha'], bias=_id['bias_rad'],
                phys_dt=_id['phys_dt_measured_s'])
            self.cfg = WGMPCConfigSP(N=a.N, dt=1.0 / a.rate, tcp=a.tcp,
                                     arm_model=self.arm_model,
                                     row_scaling=not a.no_row_scaling)
            # G 由模型自行計算 —— **不從執行端 meta 取**
            _P, _Q, _G, _h, _kp = self.cfg.composed()
            self.composed_G = [float(x) for x in _G]
            self.composed_kp = int(_kp)
        else:
            self.arm_model = None
            self.cfg = WGMPCConfig(N=a.N, dt=1.0 / a.rate, tcp=a.tcp)
            self.composed_G = None
            self.composed_kp = None
        # **同時刻配對**：各來源各自保存最近若干筆，鍵為 round(sim_t, 6)。
        # 物理步 10 ms ⇒ µs 鍵唯一；只有三方（或兩方，理想模型時）在
        # **同一鍵**上都有樣本才合成快照。
        self._buf = {'arm': {}, 'base': {}, 'sp': {}}
        self._buf_keep = 64
        self._need = (('arm', 'base', 'sp') if a.arm_model == 'setpoint'
                      else ('arm', 'base'))
        self._arm = None          # 仍保留最後一筆供診斷
        self._base = None
        self._snap = None
        # **語意**：某個來源先到、該時間鍵還湊不齊的次數。
        # 每個物理步**本來就會有一次**（先到的那一個來源）⇒
        # 這個數字接近步數是正常的，**不是錯誤計數**。
        # 真正的錯配由 `_n_step_mismatch` 與 `snapshot_not_paired` 反映。
        self._n_incomplete = 0
        # 設定點握手閘門（理想模型時不建，避免誤用）
        self._gate = (SetpointGate(max_age_s=a.max_input_age,
                                   phys_dt_s=a.phys_dt,
                                   joint_order=list(ARM_JOINTS))
                      if a.arm_model == 'setpoint' else None)
        self._sp_meta_seen = False
        self._gate_state = None
        self._gate_why = ''
        self._n_init_cmd = 0
        # 步序關聯：step_id → applied 回報的 sim_t（上限筆數，不無限成長）
        self._step_t = {}
        self._step_t_tol = 1e-6
        self._n_step_mismatch = 0
        self._last_step_mismatch = None
        self.handshake_startup = {'n_init_cmd': 0, 'states': []}
        self._t_sim0 = None
        self._stop_why = 'not_started'
        self._t_wall0 = time.monotonic()
        self._sim_slot0 = None
        self._halt = False
        self._n_dup_skip = self._n_missed_slot = self._n_stall = 0
        self._n_warm_discard = 0
        self._last_stall_s = None
        self._last_solved_key = None
        self._last_solved_sim_t = None
        self._cmd_hist = []
        # 命令三階段，各自只存**最後一筆不可變 tuple**
        self._modified = None       # (tuple9_world, mono)
        self._ep_req = None         # (tuple9_body, mono)  —— **E2 之前**
        self._applied = None        # (tuple9_body, mono, step_id, sim_t) E2 之後
        self._applied_meta = None
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(Float64MultiArray,
                                 '/wholebody_safety/cmd_out',
                                 self._on_modified, 10)
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd',
                                 self._on_ep_req, 10)
        # **權威來源**：執行端經 E2 處理、真正送進物理 API 的命令回報
        self.create_subscription(Float64MultiArray, '/coman/applied_cmd',
                                 self._on_applied, 10)
        _lat = QoSProfile(depth=1)
        _lat.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(String, '/coman/applied_cmd_meta',
                                 self._on_applied_meta, _lat)
        if a.arm_model == 'setpoint':
            self.create_subscription(Float64MultiArray,
                                     '/coman/arm_setpoint', self._on_sp, 10)
            self.create_subscription(
                String, '/coman/arm_setpoint_meta', self._on_sp_meta,
                QoSProfile(depth=1,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, '/coman/applied_fail',
                                 self._on_applied_fail, _lat)
        self.pub = self.create_publisher(Float64MultiArray,
                                         '/wholebody_safety/cmd_in', 10)
        self.log = []
        self._hold_t0 = None
        self._reached_held = False
        self._chain_failed = False
        self._chain_fail_info = None
        self._stopped_on_fail = False

    # ---------------------------------------------------------- 回呼
    def _on_js(self, m):
        ix = {n: i for i, n in enumerate(m.name)}
        if not all(j in ix for j in ARM_JOINTS):
            return
        if len(m.position) <= max(ix[j] for j in ARM_JOINTS):
            return
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        v = (tuple(float(m.position[ix[j]]) for j in ARM_JOINTS),
             t, time.monotonic())
        self._arm = v
        self._put('arm', t, v)

    def _on_odom(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2.0 * (o.w * o.z + o.x * o.y),
                         1.0 - 2.0 * (o.y * o.y + o.z * o.z))
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        v = ((float(p.x), float(p.y), float(yaw)), t, time.monotonic())
        self._base = v
        self._put('base', t, v)

    def _publish_u(self, u_body: np.ndarray, theta: float) -> np.ndarray:
        """**body → world** 後發給安全鏈（adapter 之後會轉回 body）。回傳世界系命令。

        握手的全零命令與求解結果走**同一條發布路徑**，
        避免兩條路徑的座標約定日後分歧。零向量旋轉後仍是零。
        """
        u_world = body_to_world(float(theta)) @ np.asarray(u_body, float)
        m = Float64MultiArray()
        m.data = [float(x) for x in u_world]
        self.pub.publish(m)
        # **發布歷史**：延遲補償要知道哪些命令已發出但還沒生效
        self._cmd_hist.append((self.sim_now(),
                               np.asarray(u_body, float).copy()))
        if len(self._cmd_hist) > 128:
            self._cmd_hist.pop(0)
        return u_world

    def _predict_delay(self, q, s, t_snap):
        """把量測狀態推到**命令真正生效的時刻**，回傳 (q, s, n_used)。

        迴路延遲 D 由 `--delay-comp-cycles` 給（以控制週期計）。
        rec7 實錄量到端到端 ≈ **1.40 個週期**（發布延遲 0.60 ＋ cmd_age 0.60
        ＋ 一個物理步），離線重現該趟行為所需的延遲是 1.5 個週期 —— 兩者吻合。

        語意：時刻 τ 作用的命令是 **τ − D** 時已發布的最新一筆。
        要把狀態由 t_snap 推到 t_snap + D，用的是 [t_snap − D, t_snap)
        這段**已發布**的命令 —— 全部已知，**不預測未來輸入**。

        核心的預測模型本身**沒有**這個延遲；本函式只改求解的**起點狀態**，
        不改權重、視界或任何限制。
        """
        D = float(self.a.delay_comp_cycles) * self.cfg.dt
        if D <= 0.0 or not self._cmd_hist:
            return q, s, 0
        dtp = self.cfg.arm_model.phys_dt
        n = int(round(D / dtp))
        qq, ss = np.asarray(q, float).copy(), np.asarray(s, float).copy()
        for j in range(n):
            tau = t_snap + j * dtp - D
            uu = None
            for (tp, up) in self._cmd_hist:
                if tp <= tau + 1e-12:
                    uu = up
                else:
                    break
            if uu is None:
                uu = np.zeros(NU)
            qq, ss = plant_phys_step(qq, ss, uu, self.cfg, 1)
        return qq, ss, n

    @staticmethod
    def _key(t: float) -> int:
        """共同時間鍵：µs 整數。物理步 10 ms ⇒ 鍵唯一、不會把兩步併一起。"""
        return int(round(float(t) * 1e6))

    def _put(self, which: str, t: float, val) -> None:
        """寫入來源緩衝並嘗試在**該時間鍵**上合成快照。"""
        d = self._buf[which]
        d[self._key(t)] = val
        if len(d) > self._buf_keep:
            for k in sorted(d)[:len(d) - self._buf_keep]:
                del d[k]
        self._compose(self._key(t))

    def _compose(self, key: int) -> None:
        """只在**同一時間鍵**上三方（或兩方）齊備時建快照。

        **不**用各話題最新值直接拼接，也**不**用最舊時間掩蓋錯配：
        湊不出同時刻就記一次 `_n_incomplete` 並維持舊快照（每物理步本來就有一次）。
        """
        got = {w: self._buf[w].get(key) for w in self._need}
        if any(v is None for v in got.values()):
            self._n_incomplete += 1
            return
        a, b = got['arm'], got['base']
        sp = got.get('sp')
        # 共同鍵成立 ⇒ 三方時間本就相同；仍把各自的時間記進快照備查
        self._snap = Snap(
            q=b[0] + a[0], sim_t=key * 1e-6,
            recv_mono=min([x[2] for x in got.values()]),
            have_js=True, have_odom=True,
            t_arm=a[1], t_base=b[1],
            t_sp=(sp[1] if sp else float('nan')),
            s=(tuple(sp[0]) if sp else ()),
            sp_step_id=(sp[3] if sp else -1),
            sp_exec_mode=(sp[4] if sp else -1),
            sp_api_applied=(bool(sp[5]) if sp else False),
            paired_sources=tuple(self._need))

    def _on_sp(self, m):
        """`/coman/arm_setpoint`。**只寫緩衝與閘門，不在此下任何裁示。**"""
        try:
            smp = SpSample.from_data(m.data)
        except ValueError:
            return
        self._gate.feed(smp)
        if not (smp.ready and all(math.isfinite(v) for v in smp.sp)):
            return
        # **步序關聯交叉核對**：共同時間戳配對**無法**察覺一個標錯時間的
        # 設定點（把它標成下一步，下一步的 arm/base 就會與它配上）。
        # /coman/applied_cmd 與 /coman/arm_setpoint 由執行端**同一輪**發出，
        # 帶同一個 physics_step_id 與同一個 t。所以拿同一個 step_id
        # 在兩個話題上的 sim_t 互相核對，才是明確的步序關聯。
        prev = self._step_t.get(smp.step_id)
        if prev is not None and abs(prev - smp.sim_t) > self._step_t_tol:
            self._n_step_mismatch += 1
            self._last_step_mismatch = {
                'step_id': smp.step_id, 'sim_t_setpoint': smp.sim_t,
                'sim_t_applied': prev, 'diff_s': round(smp.sim_t - prev, 9)}
            return                       # **不入緩衝** ⇒ 配不出快照 ⇒ 不求解
        self._put('sp', smp.sim_t,
                  (tuple(smp.sp), smp.sim_t, time.monotonic(),
                   smp.step_id, smp.exec_mode, smp.api_applied))

    def _on_sp_meta(self, m):
        try:
            self._gate.feed_meta(json.loads(m.data))
            self._sp_meta_seen = True
        except (ValueError, TypeError):
            self._sp_meta_seen = False

    def _on_modified(self, m):
        if len(m.data) >= NU:
            self._modified = (tuple(float(x) for x in m.data[:NU]),
                              time.monotonic())

    def _on_ep_req(self, m):
        """/wb_vel_cmd —— adapter 輸出，**E2 之前**。不是實際套用。"""
        if len(m.data) >= NU:
            self._ep_req = (tuple(float(x) for x in m.data[:NU]),
                            time.monotonic())

    def _on_applied(self, m):
        """/coman/applied_cmd —— **E2 之後**真正送進物理 API 的命令。

        欄位 [physics_step_id, sim_t, bvx_body, bvy_body, wz, qd1..qd6]。
        底盤為**本體**座標；手臂是**套用設定點對應的命令速率**，
        不是實測關節速度。
        """
        if len(m.data) >= 11:
            d = m.data
            # **執行健康狀態**與套用值分開。零值本身不是錯 ——
            # 正常模式下的零命令合法；exec_mode 才說明停止原因。
            h = {'exec_mode': int(d[11]) if len(d) > 11 else 0,
                 'cmd_age_s': float(d[12]) if len(d) > 12 else float('nan'),
                 'n_recv': int(d[13]) if len(d) > 13 else -1,
                 'n_rejected': int(d[14]) if len(d) > 14 else -1,
                 'api_applied': bool(d[15] > 0.5) if len(d) > 15 else None,
                 'src_recv_seq': int(d[16]) if len(d) > 16 else -1,
                 'src_recv_sim_t': (float(d[17]) if len(d) > 17
                                    else float('nan'))}
            self._applied = (tuple(float(x) for x in d[2:11]),
                             time.monotonic(), int(d[0]), float(d[1]), h)
            # **步序關聯**：同一輪的 applied 與 arm_setpoint 應帶同一個
            # (step_id, sim_t)。記下來供 `_on_sp` 交叉核對。
            self._step_t[int(d[0])] = float(d[1])
            if len(self._step_t) > 256:
                for k in sorted(self._step_t)[:len(self._step_t) - 256]:
                    del self._step_t[k]
            if h['exec_mode'] == 3:
                self._chain_failed = True

    def _on_applied_meta(self, m):
        self._applied_meta = m.data

    def _on_applied_fail(self, m):
        """/coman/applied_fail —— 命令鏈**失效閂鎖**的明確原因。

        收到這個就**停止任務推進**，不能繼續把它當健康的零命令回授。
        """
        self._chain_failed = True
        self._chain_fail_info = m.data

    # ------------------------------------------------- u_prev 的來源
    def _u_prev(self, theta):
        """回傳 (u_prev, 來源, 是否為權威來源)。

        **權威來源只有 `applied`**（執行端 E2 之後的回報）。
        strict 模式下取不到就回 (None, 理由, False) ⇒ 呼叫端**不發布**，
        不以 endpoint_requested 或零值冒稱真實歷史。
        """
        now = time.monotonic()
        if self._applied is not None and now - self._applied[1] <= self.a.hist_age:
            # **正常模式下的零命令仍合法** —— 只有 exec_mode == 3（閂鎖）
            # 才停止任務推進，那由迴圈開頭的 _chain_failed 處理。
            return np.asarray(self._applied[0], float), 'applied', True
        if self.a.u_prev_policy == 'strict':
            if self.a.assume_initial_rest and self._applied is None \
                    and self._ep_req is None and self._modified is None:
                # **只在明確確認初始靜止時**使用零值，且只在還沒有任何
                # 命令流動之前。一旦有命令流過就不再適用。
                return np.zeros(NU), 'zeros_initial_rest', True
            return None, 'no_valid_applied_report', False
        # ---- diagnostic：保留替代值，但**另標來源**，不算驗證通過 ----
        if self._ep_req is not None and now - self._ep_req[1] <= self.a.hist_age:
            return (np.asarray(self._ep_req[0], float),
                    'DIAG:endpoint_requested', False)
        if self._modified is not None and now - self._modified[1] <= self.a.hist_age:
            w = np.asarray(self._modified[0], float)
            return body_to_world(-theta) @ w, 'DIAG:modified_rotated', False
        return np.zeros(NU), "DIAG:zeros", False

    # ---------------------------------------------------------- 迴圈
    def run(self, T_des) -> int:
        t0 = time.monotonic()
        self._t_wall0 = t0            # **牆鐘監看**的原點（上限照舊）
        self._sim_slot0 = None        # 時槽原點（模擬時間），首個可用快照時定
        self._halt = False
        self._n_dup_skip = 0          # 同一時間／步序被跳過的次數
        self._n_missed_slot = 0
        self._n_stall = 0
        self._last_stall_s = None
        self._n_warm_discard = 0
        self._last_solved_key = None  # 已求解過的快照鍵（µs）
        self._last_solved_sim_t = None
        self._cmd_hist = []
        slot = 0
        U_warm = None
        n_pub = n_drop_age = n_no_sol = 0
        n_miss = 0
        _prev_top = None
        self._t_sim0 = None          # 任務的模擬時間起點（首次成功發布時定）
        self._stop_why = 'loop_end'
        while rclpy.ok():
            # **逐輪牆鐘**：核心的 timing_ms['total'] **不含**介面開銷
            # （發布、回呼處理、排程等待）⇒ 要另外量迴圈頂到頂的實際間隔。
            _top = time.monotonic()
            _cycle_wall = (None if _prev_top is None
                           else round((_top - _prev_top) * 1e3, 4))
            _prev_top = _top
            if self._halt:
                # 等待時槽時由 `_reschedule` 設定（牆鐘上限或模擬時鐘停滯）
                break
            if time.monotonic() - t0 > self.a.duration_s:
                self._stop_why = 'wall_duration'
                break
            if self.a.duration_sim_s > 0.0 and self._t_sim0 is not None \
                    and self.sim_now() - self._t_sim0 > self.a.duration_sim_s:
                # **模擬時間預算**：與牆鐘上限並存，先到者為準。
                self._stop_why = 'sim_duration'
                break
            if self._chain_failed:
                # **執行端的命令鏈已失效閂鎖** ⇒ 停止任務推進、記錄原因、
                # 走既定收尾。**不把閂鎖後的零回報當成健康的零命令。**
                self.log.append(dict(slot=slot, sim_t=self.sim_now(),
                                     ok=False, reason='chain_fail_latched',
                                     published=False,
                                     chain_fail_info=self._chain_fail_info,
                                     timing_ms={'total': 0.0},
                                     cycle_wall_ms=_cycle_wall))
                print(f'[wg2] **執行端命令鏈失效閂鎖，停止任務推進**：'
                      f'{self._chain_fail_info}', flush=True)
                self._stopped_on_fail = True
                break
            # ---- 設定點握手閘門（只在設定點模式）----
            # **在讀快照之前裁示**：閂鎖要優先停住，HOLD 不得求解，
            # INIT 只在首次送全零初始化命令。
            if self._gate is not None:
                gs, _gsp, gwhy = self._gate.decide(self.sim_now())
                self._gate_state, self._gate_why = gs, gwhy
                if gs == FAILED:
                    self.log.append(dict(
                        slot=slot, sim_t=self.sim_now(), ok=False,
                        reason='sp_gate_failed', published=False,
                        gate_state=gs, gate_why=gwhy,
                        timing_ms={'total': 0.0}, cycle_wall_ms=_cycle_wall))
                    print(f'[wg2] **設定點閘門回報失效閂鎖**：{gwhy}', flush=True)
                    self._stopped_on_fail = True
                    break
                if gs == INIT:
                    # 首次握手：送全零命令讓執行端建立設定點。
                    # **這是握手步驟，不是求解結果**，所以另記 reason。
                    try:
                        u_init = self._gate.init_command(NU)
                    except RuntimeError as e:
                        u_init = None
                        gwhy = f'{gwhy}｜{e}'
                    if u_init is not None:
                        self._publish_u(np.asarray(u_init, float), 0.0)
                        self._n_init_cmd += 1
                    self.log.append(dict(
                        slot=slot, sim_t=self.sim_now(), ok=False,
                        reason='sp_handshake_init', published=False,
                        handshake_cmd_sent=bool(u_init is not None),
                        gate_state=gs, gate_why=gwhy,
                        timing_ms={'total': 0.0}, cycle_wall_ms=_cycle_wall))
                    slot, _m = self._reschedule(slot)
                    n_miss += _m
                    continue
                if gs == HOLD:
                    # **不求解、不發布、不重送初始化命令、不沿用舊設定點。**
                    self.log.append(dict(
                        slot=slot, sim_t=self.sim_now(), ok=False,
                        reason='sp_gate_hold', published=False,
                        gate_state=gs, gate_why=gwhy,
                        timing_ms={'total': 0.0}, cycle_wall_ms=_cycle_wall))
                    slot, _m = self._reschedule(slot)
                    n_miss += _m
                    continue
            snap = self._snap              # **讀一次**，整輪只用區域變數
            if snap is None:
                self._exec.spin_once(timeout_sec=0.01)
                continue
            _key = self._key(snap.sim_t)
            if self._sim_slot0 is None:
                # 時槽原點 = 第一個可用快照的模擬時間
                self._sim_slot0 = snap.sim_t
            # ---- **重複／過舊快照閘門** ----
            # 同一個時間／步序不得重複求解；較舊的快照也不得覆蓋
            # 已經用過的較新控制依據。rec6 有 423 次相鄰快照時間相同。
            if self._last_solved_key is not None and (
                    _key == self._last_solved_key
                    or snap.sim_t <= self._last_solved_sim_t + 1e-9):
                self._n_dup_skip += 1
                self.log.append(dict(
                    slot=slot, sim_t=snap.sim_t, ok=False,
                    reason=('snapshot_duplicate' if _key == self._last_solved_key
                            else 'snapshot_not_newer'),
                    published=False,
                    last_solved_sim_t=self._last_solved_sim_t,
                    n_dup_skip=self._n_dup_skip,
                    timing_ms={'total': 0.0}, cycle_wall_ms=_cycle_wall))
                slot, _m = self._reschedule(slot)
                n_miss += _m
                continue
            if self._gate is not None and len(snap.s) != 6:
                # 閘門說 ARMED，但**同時刻快照裡沒有設定點** ⇒ 配對未成立。
                # 「各話題都新鮮」不等於「同一個物理步」。
                self.log.append(dict(
                    slot=slot, sim_t=snap.sim_t, ok=False,
                    reason='snapshot_not_paired', published=False,
                    gate_state=self._gate_state,
                    paired_sources=list(snap.paired_sources),
                    n_incomplete=self._n_incomplete,
                    timing_ms={'total': 0.0}, cycle_wall_ms=_cycle_wall))
                slot, _m = self._reschedule(slot)
                n_miss += _m
                continue
            age_in = self.sim_now() - snap.sim_t
            if age_in > self.a.max_input_age:
                # **求解前就過期**也要留紀錄，否則 log 看不到被擋下的輪次
                n_drop_age += 1
                self.log.append(dict(slot=slot, sim_t=snap.sim_t,
                                     age_in=round(age_in, 6), age_out=None,
                                     u_prev_src='n/a', ok=False,
                                     reason='stale_before_solve',
                                     published=False, timing_ms={'total': 0.0},
                                     cycle_wall_ms=_cycle_wall,
                                     dropped=f'求解前輸入已過期 '
                                             f'{age_in * 1e3:.0f} ms'))
                slot, _m = self._reschedule(slot)
                n_miss += _m
                continue
            q0 = np.asarray(snap.q, float)
            u_prev, src, authoritative = self._u_prev(float(q0[2]))
            if u_prev is None:
                # **strict：沒有權威的已套用回報就不求解、不發布。**
                # 不以 endpoint_requested 或零值冒稱真實歷史。
                self.log.append(dict(slot=slot, sim_t=snap.sim_t,
                                     age_in=round(age_in, 6), age_out=None,
                                     u_prev_src=src, u_prev_authoritative=False,
                                     ok=False, reason='no_u_prev',
                                     published=False, timing_ms={'total': 0.0},
                                     cycle_wall_ms=_cycle_wall,
                                     dropped=f'無權威 u_prev（{src}）'))
                slot, _m = self._reschedule(slot)
                n_miss += _m
                continue
            _solve_sim_t0 = self.sim_now()
            _solve_wall0 = time.monotonic()
            _q_sol, _s_sol, _n_comp = q0, np.asarray(snap.s, float), 0
            if self._gate is not None and self.a.delay_comp_cycles > 0.0:
                _q_sol, _s_sol, _n_comp = self._predict_delay(
                    q0, np.asarray(snap.s, float), snap.sim_t)
            if self._gate is not None:
                # 增廣狀態由**同時刻快照**組成：q 與 s 同一個物理步。
                # `make_z` 在 s 含非有限值時拋錯，不以實測關節角代替。
                r = solve_sp(self.K, make_z(_q_sol, _s_sol),
                             u_prev, T_des, self.cfg, U_warm=U_warm)
            else:
                r = solve(self.K, q0, u_prev, T_des, self.cfg, U_warm=U_warm)
            _solve_wall_ms = (time.monotonic() - _solve_wall0) * 1e3
            # **求解返回後重新檢查輸入年齡與任務有效性。**
            # 單一 executor 在求解期間不處理回呼，所以返回時佇列裡可能積了
            # 多筆訊息；**一次 spin_once 不保證已處理到新的 /clock**，
            # 而 /clock 正是 sim_now() 的來源。
            # 所以這裡做**有界**回呼處理，並要求時間依據**可確認已更新**：
            #   若 sim_now() 與 pump 之前相同、且沒有更新的狀態訊息，
            #   就**無法確認**解是否過期 ⇒ **不發布**（fail closed）。
            _t_before = self.sim_now()
            _sn_before = snap.sim_t
            self._pump()
            _t_after = self.sim_now()
            _sn_after = self._snap.sim_t if self._snap else _sn_before
            _clock_moved = _t_after > _t_before + 1e-9
            _state_moved = _sn_after > _sn_before + 1e-9
            # 時間依據取**兩個模擬時鐘來源的較新者**（與 F17 同一紀律）
            _ref = max(_t_after, _sn_after)
            age_out = _ref - snap.sim_t
            rec = dict(slot=slot, sim_t=snap.sim_t, age_in=round(age_in, 6),
                       age_out=round(age_out, 6), u_prev_src=src,
                       ok=bool(r.ok), reason=r.reason,
                       sqp_stop=r.sqp_stop_reason,
                       sqp_converged=bool(r.sqp_converged),
                       n_sqp=r.n_sqp_used, residual=r.max_residual,
                       timing_ms=r.timing_ms, cycle_wall_ms=_cycle_wall,
                       u_prev_authoritative=bool(authoritative),
                       # **同時刻配對的證據**：各來源自己的時間都記下來，
                       # 不只記合成後的 sim_t（那會把錯配藏起來）
                       paired=dict(sources=list(snap.paired_sources),
                                   t_arm=snap.t_arm, t_base=snap.t_base,
                                   t_sp=snap.t_sp,
                                   max_spread_s=round(
                                       max(abs(snap.t_arm - snap.sim_t),
                                           abs(snap.t_base - snap.sim_t),
                                           (abs(snap.t_sp - snap.sim_t)
                                            if snap.s else 0.0)), 9),
                                   sp_step_id=snap.sp_step_id,
                                   sp_exec_mode=snap.sp_exec_mode,
                                   sp_api_applied=snap.sp_api_applied),
                       gate_state=self._gate_state,
                       n_incomplete=self._n_incomplete,
                       # **時間契約的分項紀錄**（快照／求解起點／發布／牆鐘）
                       timing=dict(snap_sim_t=round(snap.sim_t, 6),
                                   solve_start_sim_t=round(_solve_sim_t0, 6),
                                   solve_wall_ms=round(_solve_wall_ms, 4),
                                   slot_target_sim_t=(
                                       None if self._sim_slot0 is None else
                                       round(self._sim_slot0
                                             + slot / self.a.rate, 6))),
                       n_dup_skip=self._n_dup_skip,
                       n_missed_slot=self._n_missed_slot,
                       delay_comp=dict(
                           cycles=float(self.a.delay_comp_cycles),
                           n_phys_steps=int(_n_comp),
                           dq_pos_m=(None if _n_comp == 0 else round(float(
                               np.linalg.norm(np.asarray(_q_sol)[:3]
                                              - q0[:3])), 6)),
                           dq_arm_max_rad=(None if _n_comp == 0 else
                                           round(float(np.abs(
                                               np.asarray(_q_sol)[3:]
                                               - q0[3:]).max()), 6))))
            if not r.ok:
                n_no_sol += 1
                rec['published'] = False
                self.log.append(rec)
                slot, _m = self._reschedule(slot)
                n_miss += _m
                continue
            rec['clock_moved'] = bool(_clock_moved)
            rec['state_moved'] = bool(_state_moved)
            rec['time_ref'] = round(_ref, 6)
            if (not _clock_moved) and (not _state_moved) \
                    and r.timing_ms['total'] > self.a.max_input_age * 1e3:
                # **無法確認時間依據已更新，而求解耗時又超過年齡界限**
                # ⇒ 不能判斷解是否過期 ⇒ **不發布**。
                n_drop_age += 1
                rec['published'] = False
                rec['dropped'] = (f'時間依據未更新（clock/state 皆未前進）'
                                  f'而求解耗時 {r.timing_ms["total"]:.0f} ms '
                                  f'> {self.a.max_input_age*1e3:.0f} ms')
                self.log.append(rec)
                slot, _m = self._reschedule(slot)
                n_miss += _m
                continue
            if age_out > self.a.max_input_age:
                # **過期解不發布，不重新蓋時間**
                n_drop_age += 1
                rec['published'] = False
                rec['dropped'] = f'輸入已過期 {age_out*1e3:.0f} ms'
                self.log.append(rec)
                slot, _m = self._reschedule(slot)
                n_miss += _m
                continue
            if self._gate is not None:
                # **求解後再查一次執行健康**：求解期間（p50 ~25 ms）可能
                # 收到閂鎖或 api_applied = False。只檢查年齡不夠。
                gs2, _s2, why2 = self._gate.decide(_ref)
                rec['gate_state_after'] = gs2
                rec['gate_why_after'] = why2
                if gs2 != ARMED:
                    rec['published'] = False
                    rec['reason'] = f'sp_gate_{gs2}_after_solve'
                    rec['dropped'] = f'求解後執行健康不合格（{gs2}）：{why2}'
                    self.log.append(rec)
                    if gs2 == FAILED:
                        print(f'[wg2] **求解後回報失效閂鎖**：{why2}', flush=True)
                        self._stopped_on_fail = True
                        break
                    slot, _m = self._reschedule(slot)
                    n_miss += _m
                    continue
            if self._t_sim0 is None:
                # 任務的**模擬時間**起點 = 首次成功發布那一輪的快照時間
                self._t_sim0 = snap.sim_t
            self._last_solved_key = _key
            self._last_solved_sim_t = snap.sim_t
            rec['publish_sim_t'] = round(self.sim_now(), 6)
            # 四段紀錄裡的 request 段**用同一個值**，不另算一次轉換
            u_world = self._publish_u(r.u0, float(q0[2]))
            n_pub += 1
            U_warm = r.U
            e = task_error(self.K, q0, T_des, self.cfg.tcp)
            _ep, _er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            # **到達並保持**：用實測狀態算的 FK 誤差，不是核心預測。
            _in_tol = (_ep <= self.a.reach_pos_m and _er <= self.a.reach_rot_rad)
            if _in_tol:
                if self._hold_t0 is None:
                    self._hold_t0 = snap.sim_t
                    print(f'[wg2] 首次進入容差 @ sim {snap.sim_t:.3f}  '
                          f'{_ep*1e3:.2f} mm / {math.degrees(_er):.3f}°', flush=True)
                elif snap.sim_t - self._hold_t0 >= self.a.hold_s:
                    self._reached_held = True
            else:
                if self._hold_t0 is not None:
                    print(f'[wg2] 離開容差 @ sim {snap.sim_t:.3f}  '
                          f'{_ep*1e3:.2f} mm / {math.degrees(_er):.3f}° '
                          f'（保持計時重設）', flush=True)
                self._hold_t0 = None
            # ---- **四段紀錄**：request → modified → endpoint_requested → applied
            # 各自附時間、座標與執行狀態。**無法確定同一筆來源時標 unpaired**，
            # 不拿各話題的「最新值」直接當成同筆的衰減量。
            _now_m = time.monotonic()
            _ap = self._applied
            _stages = {
                'request': {'frame': 'world', 'v': [round(float(x), 8)
                                                    for x in u_world],
                            'body': [round(float(x), 8) for x in r.u0],
                            'yaw': round(float(q0[2]), 9),
                            'src_sim_t': round(snap.sim_t, 6),
                            'at_mono': round(_now_m, 6)},
                'modified': ({'frame': 'world',
                              'v': [round(x, 8) for x in self._modified[0]],
                              'age_mono_s': round(_now_m - self._modified[1], 4)}
                             if self._modified is not None else None),
                'endpoint_requested': ({'frame': 'body',
                                        'v': [round(x, 8) for x in self._ep_req[0]],
                                        'age_mono_s': round(_now_m - self._ep_req[1], 4)}
                                       if self._ep_req is not None else None),
                'applied': ({'frame': 'body', 'v': [round(x, 8) for x in _ap[0]],
                             'physics_step_id': _ap[2], 'sim_t': round(_ap[3], 6),
                             'health': _ap[4]} if _ap is not None else None),
                'pairing': 'unpaired_latest_values_only',
                'pairing_note': '各段取各話題的最新值，**未逐筆配對** ⇒ '
                                '不得相減當成同一筆命令的衰減量',
            }
            rec.update(stages=_stages,
                       in_tol=bool(_in_tol),
                       hold_elapsed=(None if self._hold_t0 is None
                                     else round(snap.sim_t - self._hold_t0, 4)),
                       published=True,
                       request_world=[round(float(x), 8) for x in u_world],
                       request_body=[round(float(x), 8) for x in r.u0],
                       # **兩次座標轉換用的 yaw 與時間**各自記錄：
                       # 本節點用 snap 的 yaw；adapter 用它自己收到的 /odom yaw。
                       # 往返代數正確**只在同一 yaw 下**成立，
                       # 兩者若不同步就會有實際誤差 —— 要能事後對帳。
                       conv_yaw_node=round(float(q0[2]), 9),
                       conv_yaw_src_sim_t=round(snap.sim_t, 6),
                       conv_at_mono=round(time.monotonic(), 6),
                       err_p=_ep, err_r=_er)
            self.log.append(rec)
            if self._reached_held:
                print(f'[wg2] **到達並保持 {self.a.hold_s:.1f} s** '
                      f'@ sim {snap.sim_t:.3f}', flush=True)
                break
            slot, _m = self._reschedule(slot)
            n_miss += _m
            if _m:
                # **跨過多個時槽**：暖啟動序列假設只前進一步，
                # 不能再當成有效的 nominal。最小處理是丟棄重建，
                # **不**臨時改模型 dt 來補救。
                U_warm = None
                self._n_warm_discard += 1
        return dict(stop_why=self._stop_why,
                    delay_comp_cycles=float(self.a.delay_comp_cycles),
                    slot_basis='simulation_clock',
                    nominal_period_sim_s=1.0 / self.a.rate,
                    n_dup_skip=self._n_dup_skip,
                    n_missed_slot=self._n_missed_slot,
                    n_warm_discard=self._n_warm_discard,
                    n_sim_stall=self._n_stall,
                    last_stall_s=self._last_stall_s,
                    task_sim_t0=self._t_sim0,
                    task_sim_span_s=(None if self._t_sim0 is None
                                     else round(self.sim_now()
                                                - self._t_sim0, 3)),
                    n_step_mismatch=self._n_step_mismatch,
                    last_step_mismatch=self._last_step_mismatch,
                    n_compose_incomplete=self._n_incomplete,
                    n_compose_incomplete_note='先到的來源還湊不齊的次數；'
                                              '每物理步本來就有一次，**非錯誤**',
                    handshake_startup=self.handshake_startup,
                    n_init_cmd_in_run=self._n_init_cmd,
                    gate_report=(self._gate.report() if self._gate else None),
                    composed_G=self.composed_G, composed_kp=self.composed_kp,
                    published=n_pub, dropped_stale=n_drop_age,
                    no_solution=n_no_sol, deadline_miss=int(n_miss),
                    rate_hz=self.a.rate,
                    reached_held=bool(self._reached_held),
                    stopped_on_chain_fail=bool(self._stopped_on_fail),
                    chain_fail_info=self._chain_fail_info)

    def _reschedule(self, slot):
        """等到**下一個模擬時間時槽**。回傳 (slot, missed)。

        時槽由**同一個模擬時鐘**驅動，名目間隔 `period = 1/rate`
        （0.05 s **模擬時間**，與核心的 dt 同一個量）。

        先前用 `time.monotonic()` 排程：RTF < 1 時，以模擬時間計的控制率
        會高於名目值（rec6 實測 34.7 Hz 對名目 20 Hz，間隔 p50 0.030 s），
        而核心的 dt 仍是 0.050 —— **時間契約不一致**。

        等待期間仍服務回呼與失效訊息。模擬時間暫停時就一直等
        （因此不會持續求解，保持計時也不會累積，它是以模擬時間算的）；
        **牆鐘監看與上限照舊**，由本函式在等待中檢查並設 `self._halt`。

        錯過時槽就**跳到下一個未來時槽**，不密集補發。
        """
        period = 1.0 / self.a.rate
        slot += 1
        if self._sim_slot0 is None:
            return slot, False
        target = self._sim_slot0 + slot * period
        stall_t0 = time.monotonic()
        sim_seen = self.sim_now()
        while True:
            now_sim = self.sim_now()
            if now_sim > sim_seen + 1e-9:
                sim_seen = now_sim
                stall_t0 = time.monotonic()
            if now_sim >= target - 1e-9:
                break
            # **牆鐘監看照舊**：等待中也要能被上限中止，否則模擬時鐘
            # 停住時會永遠卡在這裡。
            if time.monotonic() - self._t_wall0 > self.a.duration_s:
                self._halt, self._stop_why = True, 'wall_duration'
                return slot, False
            _st = time.monotonic() - stall_t0
            if _st > self.a.sim_stall_wall_s:
                self._n_stall += 1
                self._last_stall_s = round(_st, 3)
                self._halt, self._stop_why = True, 'sim_clock_stalled'
                return slot, False
            self._exec.spin_once(timeout_sec=0.002)
        # 已經越過幾個時槽？跳到第一個未來時槽，**不追趕**
        now_sim = self.sim_now()
        k = int((now_sim - self._sim_slot0) // period)
        if k > slot:
            self._n_missed_slot += (k - slot)
            slot = k
            return slot, True
        return slot, False

    def _pump(self):
        """**有界**回呼處理。保持單一 executor，**不加第二個 spin 執行緒**。

        至多 `pump_budget` 次回呼、至多一次 `pump_wait_s` 的阻塞。
        一次 spin_once 不保證所有必要主題都更新 ⇒ 呼叫端仍要查時間依據。
        """
        self._exec.spin_once(timeout_sec=self.a.pump_wait_s)
        for _ in range(max(0, int(self.a.pump_budget) - 1)):
            self._exec.spin_once(timeout_sec=0.0)

    def sim_now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default=os.path.join(
        _HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--tcp', default='link_tcp')
    ap.add_argument('--N', type=int, default=5)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--target', nargs=3, type=float, default=None,
                    help='**絕對**世界座標 TCP 目標')
    ap.add_argument('--target-offset', nargs=3, type=float, default=None,
                    help='相對**實測起始 TCP** 的偏移（事前定版用；'
                         '避免猜生成位姿）。與 --target 二擇一。')
    ap.add_argument('--reach-pos-m', type=float, default=0.005)
    ap.add_argument('--reach-rot-rad', type=float, default=0.02)
    ap.add_argument('--hold-s', type=float, default=2.0,
                    help='首次到達後要連續維持在容差內多久才算完成（模擬時間）')
    ap.add_argument('--target-rot-deg', type=float, default=0.0)
    ap.add_argument('--duration-s', type=float, default=30.0,
                    help='**牆鐘**時間上限')
    ap.add_argument('--duration-sim-s', type=float, default=0.0,
                    help='**模擬時間**預算上限（0 = 不啟用）。'
                         '錄影會拖慢 sim:wall，只靠牆鐘上限會讓任務拿到的'
                         '模擬時間比無錄影趟次少 ⇒ 要與 free4 對齊時用這個。')
    ap.add_argument('--delay-comp-cycles', type=float, default=0.0,
                    help='**迴路延遲補償**（以控制週期計，0 = 關閉）。'
                         '求解前用已驗證的受控對象模型與**已發布**的在途命令'
                         '把狀態推到命令生效的時刻。'
                         'rec7 實測端到端延遲 ≈ 1.40 個週期。'
                         '**不改權重、視界或任何限制。**')
    ap.add_argument('--sim-stall-wall-s', type=float, default=10.0,
                    help='模擬時鐘停滯多久（**牆鐘**）就中止等待。'
                         '等待期間不求解、不累積保持時間。')
    ap.add_argument('--max-input-age', type=float, default=0.2)
    ap.add_argument('--hist-age', type=float, default=0.5)
    ap.add_argument('--u-prev-policy', default='strict',
                    choices=['strict', 'diagnostic'],
                    help="strict = **只接受** /coman/applied_cmd 的權威回報，"
                         "取不到就不發布；diagnostic = 保留替代值但標來源，"
                         "**不算驗證通過**")
    ap.add_argument('--pump-budget', type=int, default=32)
    ap.add_argument('--pump-wait-s', type=float, default=0.002)
    ap.add_argument('--assume-initial-rest', action='store_true',
                    help='明確確認初始靜止時，允許第一輪以零值作 u_prev')
    ap.add_argument('--arm-model', default='ideal',
                    choices=['ideal', 'setpoint'],
                    help='ideal = 原核心（理想速度積分，free4 基準，預設不變）；'
                         'setpoint = 增廣核心（手臂設定點納入狀態）')
    ap.add_argument('--arm-ident',
                    default=os.path.join(_HERE, 'results',
                                         'wgmpc_arm_sp_ident_free4.json'),
                    help='setpoint 模式用的辨識檔（α、b、physics_dt）')
    ap.add_argument('--no-row-scaling', action='store_true',
                    help='關閉等價正值列縮放（對照用）')
    ap.add_argument('--handshake-timeout-s', type=float, default=20.0,
                    help='啟動握手與狀態齊備的等待上限')
    ap.add_argument('--phys-dt', type=float, default=0.01,
                    help='介面契約核對用的物理步長；與執行端 meta 比對')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    rclpy.init()
    nd = WGMPCNode(a)

    def _bail(code, why):
        """**拒絕啟動也要留紀錄**：只印在終端的話，事後無從對帳。"""
        print(f'**{why}**', flush=True)
        if a.out:
            json.dump({'args': vars(a), 'started': False,
                       'exit_code': code, 'refuse_reason': why,
                       'handshake_startup': nd.handshake_startup,
                       'n_step_mismatch': nd._n_step_mismatch,
                       'last_step_mismatch': nd._last_step_mismatch,
                       'n_compose_incomplete': nd._n_incomplete,
                       'gate_report': (nd._gate.report() if nd._gate
                                       else None),
                       'log': nd.log},
                      open(a.out, 'w'), ensure_ascii=False)
        ex.shutdown(); nd.destroy_node(); rclpy.try_shutdown()
        return code
    # **單一 executor 所有權**：只有這裡建立，節點只加入一次
    ex = SingleThreadedExecutor()
    ex.add_node(nd)
    nd._exec = ex
    # ---- 等狀態齊備；設定點模式還要先完成**初始化握手** ----
    # **死鎖防治**：設定點模式的同時刻快照需要 /coman/arm_setpoint，
    # 而真實執行端**只在收到第一筆有效命令時**才建立設定點
    # （wb_cmd_chain_e2 的 setpoint_init）。若在這裡只等快照，
    # 就會「等設定點 ← 等命令 ← 等 run()」互相卡住
    # （實測：替身用計時器自行建立設定點，把這個缺陷遮掉了）。
    # 所以握手要在**啟動路徑**就做：按控制週期送全零命令直到就緒。
    t0 = time.monotonic()
    _hs_period = 1.0 / a.rate
    _hs_next = time.monotonic()
    while time.monotonic() - t0 < a.handshake_timeout_s:
        ex.spin_once(timeout_sec=0.02)
        if nd._gate is None:
            if nd._snap is not None:
                break
            continue
        gs, _sp, why = nd._gate.decide(nd.sim_now())
        if not nd.handshake_startup['states'] or \
                nd.handshake_startup['states'][-1][0] != gs:
            nd.handshake_startup['states'].append(
                (gs, round(nd.sim_now(), 4), why[:90]))
        if gs == FAILED:
            return _bail(4, f'啟動時執行端已失效閂鎖：{why}')
        if gs == ARMED and nd._snap is not None and len(nd._snap.s) == 6:
            break
        if gs == INIT and time.monotonic() >= _hs_next:
            # **握手命令**：全零，只為讓執行端走到 setpoint_init。
            # 用快照的 yaw（沒有快照時用 0；零向量旋轉後仍是零）。
            _yaw = float(nd._snap.q[2]) if nd._snap is not None else 0.0
            try:
                nd._publish_u(np.asarray(nd._gate.init_command(NU), float),
                              _yaw)
                nd.handshake_startup['n_init_cmd'] += 1
            except RuntimeError:
                pass
            _hs_next = time.monotonic() + _hs_period
    if nd._snap is None:
        _extra = ''
        if nd._gate is not None:
            _extra = (f'；步序不符 {nd._n_step_mismatch} 次'
                      f'（{nd._last_step_mismatch}）'
                      f'、未湊齊 {nd._n_incomplete} 次'
                      f'、握手送出 {nd.handshake_startup["n_init_cmd"]} 筆')
        return _bail(3, '狀態未齊備（缺 /joint_states、/odom 或同時刻設定點）'
                        + _extra)
    if nd._gate is not None and len(nd._snap.s) != 6:
        return _bail(5, f'初始化握手未完成：送出 '
                        f'{nd.handshake_startup["n_init_cmd"]} 筆全零命令後'
                        f'仍未取得同時刻設定點。'
                        f'狀態歷程 {nd.handshake_startup["states"]}')
    if nd._gate is not None:
        print(f'[wg2] 初始化握手完成：送出 '
              f'{nd.handshake_startup["n_init_cmd"]} 筆全零命令，'
              f'設定點 step {nd._snap.sp_step_id}', flush=True)
    q0 = np.asarray(nd._snap.q, float)
    T = nd.K.fk(q0, a.tcp).copy()
    if (a.target is None) == (a.target_offset is None):
        return _bail(2, '--target 與 --target-offset 必須且只能給一個')
    T_des = np.eye(4)
    T_des[:3, 3] = (np.asarray(a.target, float) if a.target is not None
                    else T[:3, 3] + np.asarray(a.target_offset, float))
    T_des[:3, :3] = T[:3, :3]        # 首版只給位置目標＋**保持起始姿態**
    print(f'[wg2] N={a.N} dt={1.0/a.rate:.3f}  '
          f'起始 q {np.round(q0,5).tolist()}', flush=True)
    print(f'[wg2] 起始 TCP {np.round(T[:3,3],5).tolist()}  '
          f'目標 TCP {np.round(T_des[:3,3],5).tolist()}  '
          f'偏移 {a.target_offset if a.target_offset else "（絕對）"}', flush=True)
    print(f'[wg2] 到達 ≤{a.reach_pos_m*1e3:.1f} mm / '
          f'{math.degrees(a.reach_rot_rad):.2f}°、保持 {a.hold_s:.1f} s（模擬時間）、'
          f'u_prev 政策 {a.u_prev_policy}'
          f'{"＋初始靜止" if a.assume_initial_rest else ""}', flush=True)
    stats = None          # finally 會讀它；run() 丟例外時不可變成 NameError
    try:
        stats = nd.run(T_des)
    finally:
        if a.out:
            json.dump({'args': vars(a), 'started': True,
                       'stats': stats,
                       'handshake_startup': nd.handshake_startup,
                       'n_step_mismatch': nd._n_step_mismatch,
                       'last_step_mismatch': nd._last_step_mismatch,
                       'n_compose_incomplete': nd._n_incomplete,
                       'gate_report': (nd._gate.report() if nd._gate
                                       else None),
                       'composed_G': nd.composed_G,
                       'composed_kp': nd.composed_kp,
                       'target_tcp': [float(x) for x in T_des[:3, 3]],
                       'start_tcp': [float(x) for x in T[:3, 3]],
                       'start_q': [float(x) for x in q0],
                       'reached_held': bool(nd._reached_held),
                       'stopped_on_chain_fail': bool(nd._stopped_on_fail),
                       'chain_fail_info': nd._chain_fail_info,
                       'applied_meta': nd._applied_meta,
                       'log': nd.log},
                      open(a.out, 'w'), ensure_ascii=False)
            print(f'[wg2] {len(nd.log)} 輪寫入 {a.out}', flush=True)
        ex.shutdown()
        nd.destroy_node()
        rclpy.try_shutdown()
    print(f'[wg2] {stats}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
