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
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.arm_pregrasp import ARM_JOINTS          # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import (                     # noqa: E402
    NU, WGMPCConfig, body_to_world, solve, task_error)
from ammr_wholebody_mpc.wgmpc_core_sp import (                   # noqa: E402
    ArmSetpointModel, WGMPCConfigSP, make_z, plant_phys_step,
    shape_near_target, solve_sp)
from ammr_wholebody_mpc import wgmpc_margin_guard as MG           # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import (            # noqa: E402
    WholeBodyKinematics)
import wgmpc_cmd_envelope as ENV                                 # noqa: E402
from wgmpc_cycle_record import solver_io_record                  # noqa: E402
import wg4b_b1_core as B1CORE                                    # noqa: E402
from park_fixed import HoldServo, hold_output_ok, hold_uprev_ok, park_hold_cmd  # noqa: E402
from offset_moving import MovingCfg, MovingOffsetObserver        # noqa: E402
from wgmpc_sp_handshake import (ARMED, FAILED, HOLD, INIT,       # noqa: E402
                                SetpointGate, SpSample)


def validate_target_matrix(d):
    """檢查 16 元素列優先 4x4 是不是合格的剛體變換。

    回傳 `(T, None)` 或 `(None, 理由)`。**不合格要說出理由**，不靜默忽略 ——
    目標位姿是執行期由別的節點算出來送進來的，靜默丟棄會讓「求解器還在用
    舊目標」看起來像「目標沒變」。
    """
    if len(d) != 16:
        return None, f'長度 {len(d)} 不是 16'
    try:
        vals = [float(x) for x in d]
    except (TypeError, ValueError) as e:
        return None, f'無法轉成浮點數：{e}'
    if not all(math.isfinite(x) for x in vals):
        return None, '含非有限值'
    T = np.asarray(vals, float).reshape(4, 4)
    R = T[:3, :3]
    if not np.allclose(T[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        return None, f'最後一列不是 (0,0,0,1)：{T[3].tolist()}'
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
        return None, '旋轉塊不是正交'
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > 1e-6:
        return None, f'旋轉塊行列式 {det:.6f} 不是 +1'
    return T, None


COORD_LEN = 14


def parse_coord(d):
    """整機協同設定（執行期，由任務節點依相位送）。回傳 (dict, None) 或 (None, 理由)。

    版面 v1（14 元素）：
        [1, w_vref, vx, vy, wz, w_qn, q1..q6, w_a, w_p]
    v_ref 為**本體座標**底盤速度；w_a／w_p 給 NaN = 沿用啟動值。
    權重 0 時對應的參考值可為 NaN（不使用）。**不合格就說出理由**。
    """
    if len(d) != COORD_LEN:
        return None, f'長度 {len(d)} 不是 {COORD_LEN}'
    try:
        v = [float(x) for x in d]
    except (TypeError, ValueError) as e:
        return None, f'無法轉成浮點數：{e}'
    if v[0] != 1.0:
        return None, f'版面版本 {v[0]} 不是 1'
    w_vref, vref, w_qn, qn, w_a, w_p = (v[1], v[2:5], v[5], v[6:12],
                                         v[12], v[13])
    for name, w in (('w_vref', w_vref), ('w_qn', w_qn)):
        if not (math.isfinite(w) and w >= 0.0):
            return None, f'{name} 必須為有限非負：{w}'
    if w_vref > 0.0 and not all(math.isfinite(x) for x in vref):
        return None, 'w_vref > 0 但 v_ref 含非有限值'
    if w_qn > 0.0 and not all(math.isfinite(x) for x in qn):
        return None, 'w_qn > 0 但 q_nom 含非有限值'
    for name, w in (('w_a', w_a), ('w_p', w_p)):
        if not math.isnan(w) and not (math.isfinite(w) and w > 0.0):
            return None, f'{name} 必須為正或 NaN：{w}'
    return dict(w_vref=w_vref,
                base_vref=tuple(vref) if w_vref > 0.0 else None,
                w_qn=w_qn, arm_q_nom=tuple(qn) if w_qn > 0.0 else None,
                w_a=None if math.isnan(w_a) else w_a,
                w_p=None if math.isnan(w_p) else w_p), None


def is_ros_shutdown_exc(e, ros_ok):
    """例外是否屬於「ROS 外部關閉」。只認 ExternalShutdownException，或 context 已關閉
    （ros_ok 為 False）時 ROS 層丟出的 RCLError；其他例外不論 ros_ok 都不算。"""
    if isinstance(e, ExternalShutdownException):
        return True
    return (not ros_ok) and type(e).__name__ == 'RCLError'


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
            # `w_s` 是**命令變化率**的權重（S = w_s/vmax² · I）。原預設 1e-3 相對
            # 誤差項（w_p=50）小了四個數量級：0.8 m 誤差給 50×0.8²=32／步，
            # 滿框跳變只給 1e-3×2²=4e-3 ⇒ 命令怎麼甩都幾乎不花成本。
            # 以 CLI 顯式傳入；**預設仍是 1e-3，不傳就與既有趟次完全相同。**
            self.cfg = WGMPCConfigSP(N=a.N, dt=1.0 / a.rate, tcp=a.tcp,
                                     arm_model=self.arm_model,
                                     w_s=a.w_s, w_a=a.w_a,
                                     w_s_base=a.w_s_base, w_s_arm=a.w_s_arm,
                                     row_scaling=not a.no_row_scaling)
            # G 由模型自行計算 —— **不從執行端 meta 取**
            _P, _Q, _G, _h, _kp = self.cfg.composed()
            self.composed_G = [float(x) for x in _G]
            self.composed_kp = int(_kp)
        else:
            self.arm_model = None
            self.cfg = WGMPCConfig(N=a.N, dt=1.0 / a.rate, tcp=a.tcp,
                                   w_s=a.w_s, w_a=a.w_a,
                                   w_s_base=a.w_s_base, w_s_arm=a.w_s_arm)
            self.composed_G = None
            self.composed_kp = None
        # **WG4-B B1**（experiment_spec_WG4B.yaml）：只換求解式，見 wg4b_b1_core.py。
        # 預設 wgmpc ⇒ 既有路徑一位元不變。b1 需要設定點快照（s_meas），故要求增廣模式的閘門。
        # **PARK_FIXED（固定底盤）**：整個時域 u_base = 0（核心 base_fixed 等式）。預設關 ⇒ 既有行為不變。
        self._base_fixed = bool(getattr(a, 'base_fixed', False))
        if self._base_fixed:
            if a.arm_model != 'setpoint':
                raise ValueError('--base-fixed 需要 --arm-model setpoint（只在增廣核心實作）')
            self.cfg.base_fixed = True
        # **PARK_HOLD（v2）**：整個時域 u_base = u_hold（停車伺服律，錨點來自執行端 /park/gate）
        self._park_hold = bool(getattr(a, 'park_hold', False))
        self._park_anchor = None
        self._park_servo = HoldServo()
        if self._park_hold:
            if a.arm_model != 'setpoint':
                raise ValueError('--park-hold 需要 --arm-model setpoint')
            if self._base_fixed:
                raise ValueError('--park-hold 與 --base-fixed 不可同時指定')
        self._b1 = (getattr(a, 'solver_kind', 'wgmpc') == 'b1')
        self._b1p = (B1CORE.B1Params(kp=float(a.b1_kp)) if self._b1 else None)
        self._n_b1_mu = {}
        if self._b1 and a.arm_model != 'setpoint':
            raise ValueError('--solver-kind b1 需要 --arm-model setpoint（s_meas 取自同時刻快照）')
        if a.margin_guard:
            # 有效界取**求解器自己的** joint_margin，兩端用同一個數。
            self._mg_bounds = MG.effective_bounds(self.cfg)
            self._mguard = MG.BreachPolicy(
                max_consecutive_qp_fail=a.mg_max_qp_fail,
                max_consecutive_breach=a.mg_max_breach)
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
        self._applied_by_seq = {}
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
        if a.cmd_env and a.use_applied_for_predict:
            # **實際套用值**：逐筆帶 source_seq，可與自己的發布對上。
            self.create_subscription(Float64MultiArray,
                                     ENV.TOPIC[ENV.ST_ENDPOINT],
                                     self._on_applied_env, 10)
        if a.arm_model == 'setpoint':
            self.create_subscription(Float64MultiArray,
                                     '/coman/arm_setpoint', self._on_sp, 10)
            self.create_subscription(
                String, '/coman/arm_setpoint_meta', self._on_sp_meta,
                QoSProfile(depth=1,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # ---- 執行期目標位姿（預設關閉）--------------------------------
        self._T_rx = None            # 最近一筆**已通過檢查**的目標
        self._T_rx_t = None
        self._n_target_rx = 0
        self._n_tol_blocked_fallback = 0
        self._reach_announced = False
        self._T_last = None
        self._n_target_changes = 0
        self._stop_req = None
        self._stop_rx = None           # 收到停止的時刻（模擬時間、牆鐘、原因）
        self._c_pub = self._c_drop_age = self._c_no_sol = 0
        self._c_solve_calls = self._c_miss = 0
        self._c_pub_after_stop = 0     # 收到停止後仍發布的次數（應為 0）
        self._c_solve_logged = 0       # 已寫入紀錄的求解列數（獨立計數）
        self._shutdown_exc = None
        self._drawer_released = False
        self._drawer_start = False
        if getattr(a, 'wait_for_drawer_handover', False):
            _lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.drawer_ready_pub = self.create_publisher(Bool, '/wgmpc/ready', _lat)
            self.create_subscription(String, '/wholebody/state',
                                     self._on_drawer_handover, _lat)
            self.create_subscription(Bool, '/drawer/solver_start',
                                     self._on_drawer_start, _lat)
        self.status_pub = (self.create_publisher(Float64MultiArray,
                                                 '/wgmpc/status', 10)
                           if a.continuous else None)
        if a.continuous:
            self.create_subscription(String, '/wgmpc/stop', self._on_stop, 1)
        self._of_d = np.zeros(6)       # 無偏移追蹤：手臂穩態偏移估計
        self._of_init = None
        self._of_s_prev = None
        self._of_n_upd = 0
        # **v2.1 運動中偏移觀測**（--offset-moving，預設關 ⇒ 既有行為不變）
        self._of_moving = None
        self._of_n_upd_moving = 0
        if getattr(a, 'offset_moving', False):
            if a.arm_model != 'setpoint' or not a.offset_free:
                raise ValueError('--offset-moving 需要 --arm-model setpoint 與 --offset-free')
            self._of_moving = MovingOffsetObserver(
                MovingCfg(), self.cfg.arm_model.alpha, self.cfg.arm_model.phys_dt)
        self._n_target_rejected = 0
        self._target_reject = {}
        self._n_by_target_src = {}
        self._max_target_age_s = 0.0
        if getattr(self.a, 'target_topic', ''):
            self.create_subscription(Float64MultiArray,
                                     str(self.a.target_topic),
                                     self._on_target, 1)
        # ---- 整機協同（預設關閉 ⇒ 行為與既有趟次一位元相同）----
        self._coord = None
        self._coord_t = None
        self._n_coord_rx = 0
        self._coord_reject = {}
        self._coord_launch = dict(w_vref=0.0, base_vref=None, w_qn=0.0,
                                  arm_q_nom=None, w_a=self.cfg.w_a,
                                  w_p=self.cfg.w_p)
        self._n_by_coord_src = {}
        if getattr(self.a, 'coord_topic', ''):
            self.create_subscription(Float64MultiArray,
                                     str(self.a.coord_topic),
                                     self._on_coord, 1)
        if getattr(self.a, 'park_hold', False):
            self.create_subscription(String, '/park/gate', self._on_park_gate, 1)
        self.create_subscription(String, '/coman/applied_fail',
                                 self._on_applied_fail, _lat)
        # **輸出話題可指定，預設 '/wholebody_safety/cmd_in' = 既有行為。**
        # 抽屜實驗的管線裡沒有安全層（展開節點本來就直接發 /wb_vel_cmd），
        # 而 WG2 自由空間那條是以 `freespace_confirmed:=true` 放行的 ——
        # 櫃體就在旁邊，那在抽屜房裡是**假前提**，不能照搬。
        # 所以抽屜趟次把輸出直接接到 /wb_vel_cmd，並在報告裡明寫
        # 「安全層不在此管線內」；該趟的保護是命令鏈的三層
        #（低速介面界限 → 輪級 λ → 底盤變化率上限）。
        # 這**不是**放寬現有保護：安全層本來就不在抽屜管線裡。
        self.pub = self.create_publisher(Float64MultiArray,
                                         str(a.cmd_topic), 10)
        # **命令追蹤封裝**（WG2 明確啟用的額外路徑；既有九維話題不變）。
        # 身分與九維值在同一份訊息，不另發旁路資料。
        self.env_pub = (self.create_publisher(
            Float64MultiArray, ENV.TOPIC[ENV.ST_SOLVER], 10)
            if a.cmd_env else None)
        self.env_meta = (self.create_publisher(
            String, ENV.TOPIC[ENV.ST_SOLVER] + '_meta',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            if a.cmd_env else None)
        # **兩種時間分開**（rec9 逐筆量到 D_state ≈ 1.60、D_cmd ≈ 1.00）。
        # 給負值時沿用舊的單一 --delay-comp-cycles，保持回溯相容。
        _d0 = float(a.delay_comp_cycles)
        self.d_state = (float(a.delay_comp_state_cycles)
                        if a.delay_comp_state_cycles >= 0.0 else _d0)
        self.d_cmd = (float(a.delay_comp_cmd_cycles)
                      if a.delay_comp_cmd_cycles >= 0.0 else _d0)
        self._n_shaped = 0            # 輸出被近目標整形縮放的輪數
        self._shape_scales = []
        self._n_pred_applied = 0      # 預推時用到**實際套用值**的次數
        self._n_pred_requested = 0    # 只能用請求值（尚未回報）的次數
        self._last_pred_sel = []      # 最近一輪預推逐步選到的 (source_seq, 來源)
        self._last_pub_t = None       # 最近一次發布的模擬時刻（未捨入）
        # **關節餘裕的命令時間線守衛**。只在明確給 --margin-guard 時啟用；
        # 預設關閉 ⇒ 既有趟次行為完全不變。
        self._mguard = None
        self._mg_bounds = None
        self._mg_info = None
        self._run_id_n = ENV.run_id_num(a.run_id or 'wgmpc_wg2')
        self._source_seq = 0
        self._env_meta_sent = False
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

    def _on_drawer_handover(self, msg):
        try:
            self._drawer_released = json.loads(msg.data).get('phase') == 'RELEASED'
        except (ValueError, AttributeError):
            self._drawer_released = False

    def _on_drawer_start(self, msg):
        self._drawer_start = bool(msg.data)

    def _publish_u(self, u_body: np.ndarray, theta: float) -> np.ndarray:
        """**body → world** 後發給安全鏈（adapter 之後會轉回 body）。回傳世界系命令。

        握手的全零命令與求解結果走**同一條發布路徑**，
        避免兩條路徑的座標約定日後分歧。零向量旋轉後仍是零。
        """
        u_world = body_to_world(float(theta)) @ np.asarray(u_body, float)
        if self._stop_req is not None:
            self._c_pub_after_stop += 1    # 獨立計數；正常應為 0（驗收項）
        m = Float64MultiArray()
        m.data = [float(x) for x in u_world]
        self.pub.publish(m)
        # **一次讀時鐘，兩處共用**：封裝 stamp 與發布歷史先前各讀一次
        # `sim_now()`，離線重播就分不清 _cmd_hist 用的是哪一個。
        # 模擬時鐘 10 ms 才跳一次，兩次讀值幾乎總是相同；併成一次是去掉
        # 這個歧義，不改發布內容。
        _tpub = self.sim_now()
        self._last_pub_t = float(_tpub)
        # **封裝**：身分與值同訊息。source_seq 逐筆遞增，全鏈據此關聯。
        if self.env_pub is not None:
            self._source_seq += 1
            _t = _tpub
            _em = Float64MultiArray()
            _em.data = ENV.encode(self._run_id_n, self._source_seq,
                                  ENV.ST_SOLVER, self._source_seq,
                                  self._source_seq, True, _t, u_world)
            self.env_pub.publish(_em)
            if not self._env_meta_sent:
                self._env_meta_sent = True
                self.env_meta.publish(String(data=json.dumps(
                    dict(ENV.describe(), run_id=self.a.run_id,
                         run_id_num=self._run_id_n, stage='solver'),
                    ensure_ascii=False)))
        # **發布歷史**：延遲補償要知道哪些命令已發出但還沒生效。
        # 記下 source_seq，才能與執行端回報的實際套用值逐筆對上。
        self._cmd_hist.append((_tpub,
                               np.asarray(u_body, float).copy(),
                               int(self._source_seq)))
        if len(self._cmd_hist) > 128:
            self._cmd_hist.pop(0)
        return u_world

    def _predict_delay(self, q, s, t_snap):
        """把量測狀態推到**命令真正生效的時刻**，回傳 (q, s, n_used)。

        **兩種時間必須分開**（先前用同一個 D 做兩件事是錯的）：

            量測 ──D_pub──> 發布 ──D_cmd──> 生效
            └──────── D_state = D_pub + D_cmd ────────┘

        * `d_state`：狀態要往前推多久。
        * `d_cmd`：由發布歷史倒查命令時的偏移 —— 時刻 τ 作用的命令是
          **τ − d_cmd** 時已發布的那一筆。

        rec9 **逐筆量到**：D_pub p50 0.60、D_cmd p50 1.00 ⇒ D_state ≈ 1.60。
        先前節點用單一 1.4 同時當兩者：對 D_state 接近，對 D_cmd 偏高 0.4，
        查表會取到錯的命令。離線以量到的值重跑：只有**拆分**能到達並保持。

        **命令來源**：rec9 量到安全層會修改 **38.1%** 的命令，所以
        「發布的命令就是手臂會收到的」不成立。啟用
        `--use-applied-for-predict` 時，若該筆已有執行端回報的實際套用值
        就用它；尚未回報的（真正還在途中）才退回請求值。

        核心的預測模型本身**沒有**這個延遲；本函式只改求解的**起點狀態**，
        不改權重、視界或任何限制。
        """
        D_state = self.d_state * self.cfg.dt
        D_cmd = self.d_cmd * self.cfg.dt
        if D_state <= 0.0 or not self._cmd_hist:
            # **提早返回也要清空**：否則 `_last_pred_sel` 會留著上一輪的值，
            # 紀錄裡看起來像本輪的選取 —— 陳舊值冒充當前值。
            self._last_pred_sel = []
            return q, s, 0
        dtp = self.cfg.arm_model.phys_dt
        n = int(round(D_state / dtp))
        qq, ss = np.asarray(q, float).copy(), np.asarray(s, float).copy()
        use_applied = bool(self.a.use_applied_for_predict)
        # **逐步選取紀錄**：離線重播要能逐筆核對「這一步用了哪個 source_seq、
        # 用的是實際套用值還是請求值」。只寫欄位，不改選取規則。
        sel = []
        for j in range(n):
            tau = t_snap + j * dtp - D_cmd
            uu, seq = None, None
            for (tp, up, sq) in self._cmd_hist:
                if tp <= tau + 1e-12:
                    uu, seq = up, sq
                else:
                    break
            if uu is None:
                uu = np.zeros(NU)
                sel.append((-1, 'zeros'))
            elif use_applied and seq in self._applied_by_seq:
                # **實際套用值**優先（安全層／E2 可能已改過它）
                uu = self._applied_by_seq[seq][0]
                self._n_pred_applied += 1
                sel.append((int(seq), 'applied'))
            else:
                self._n_pred_requested += 1
                sel.append((int(seq) if seq is not None else -1, 'requested'))
            qq, ss = plant_phys_step(qq, ss, uu, self.cfg, 1)
        self._last_pred_sel = sel
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

    def _on_applied_env(self, m):
        """`/coman/applied_env`：執行端回報的**實際套用值**（body 座標）。

        以 `source_seq` 建索引，供延遲補償改用實際值而非自己發布的請求。
        rec9 逐筆量到安全層會修改 **38.1%** 的命令，所以用請求值預推會錯。
        `derived = 0`（停止／無命令）的那些**不可歸屬**，不入索引。
        """
        try:
            e = ENV.decode(m.data)
        except ValueError:
            return
        if not e['derived'] or e['source_seq'] < 0:
            return
        self._applied_by_seq[e['source_seq']] = (
            np.asarray(e['u'], float), float(e['stamp_sim_t']))
        if len(self._applied_by_seq) > 256:
            for k in sorted(self._applied_by_seq)[:len(self._applied_by_seq) - 256]:
                self._applied_by_seq.pop(k, None)

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
        # **exec_mode == 4（no_command）的回報不算套用回報。**
        # 執行端在設定點還沒建立時每一步也會發一筆，欄位是零值佔位、
        # api_applied = False。把它當 'applied' 會把「什麼都沒套用」
        # 標成「量到的套用值」，strict 政策的標籤就失去意義了
        # （而且 --assume-initial-rest 會永遠觸發不到）。
        _no_cmd = (self._applied is not None
                   and int(self._applied[4].get('exec_mode', 0)) == 4)
        if (self._applied is not None and not _no_cmd
                and now - self._applied[1] <= self.a.hist_age):
            # **正常模式下的零命令仍合法** —— 只有 exec_mode == 3（閂鎖）
            # 才停止任務推進，那由迴圈開頭的 _chain_failed 處理。
            return np.asarray(self._applied[0], float), 'applied', True
        if self.a.u_prev_policy == 'strict':
            if (self.a.assume_initial_rest
                    and (self._applied is None or _no_cmd)
                    and self._ep_req is None and self._modified is None):
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
    def _on_stop(self, m):
        if self._stop_req is None:
            self._stop_rx = {'sim_t': self.sim_now(), 'wall_mono': time.monotonic(),
                             'reason': str(m.data) or '（空）'}
        self._stop_req = str(m.data) or '（空）'

    def _on_target(self, m: Float64MultiArray):
        """目標位姿回呼。**不合格就記下理由**，不靜默忽略。"""
        T, why = validate_target_matrix(list(m.data))
        if why is not None:
            self._n_target_rejected += 1
            self._target_reject[why] = self._target_reject.get(why, 0) + 1
            self.get_logger().warn(f'目標位姿不合格（{why}）⇒ 不採用',
                                   throttle_duration_sec=2.0)
            return
        self._T_rx = T
        self._T_rx_t = self.sim_now()
        self._n_target_rx += 1

    def _on_coord(self, m: Float64MultiArray):
        c, why = parse_coord(list(m.data))
        if why is not None:
            self._coord_reject[why] = self._coord_reject.get(why, 0) + 1
            self.get_logger().warn(f'協同設定不合格（{why}）⇒ 不採用',
                                   throttle_duration_sec=2.0)
            return
        self._coord = c
        self._coord_t = self.sim_now()
        self._n_coord_rx += 1

    def _on_park_gate(self, m):
        """錨點只取一次（閘門通過當步的實測位姿），之後不更新。"""
        if self._park_anchor is not None:
            return
        try:
            g = json.loads(m.data)
        except Exception:
            return
        if g.get('passed') and g.get('anchor') is not None:
            self._park_anchor = tuple(float(x) for x in g['anchor'])

    def b1_report(self) -> dict:
        """WG4-B B1 的每趟一次紀錄（參數、限制、整形、OSQP 設定）；非 b1 時回空 dict。"""
        if not self._b1:
            return {}
        a, c = self.a, self.cfg
        return {'solver_kind': 'b1',
                'b1': {'params': self._b1p.record(),
                       'mu_reason_counts': dict(self._n_b1_mu),
                       'limits': {'vmax': c.vmax().tolist(), 'amax': c.amax().tolist(),
                                  'joint_margin': c.joint_margin,
                                  'wheel_radius': c.wheel_radius,
                                  'wheel_w_max': c.wheel_w_max,
                                  'wheel_a_max': c.wheel_a_max, 'dt': c.dt},
                       'shaping': {'gamma': a.near_target_gamma,
                                   'tol_p': a.reach_pos_m, 'tol_r': a.reach_rot_rad,
                                   'deadband_m': a.near_target_deadband_m,
                                   'deadband_rad': a.near_target_deadband_rad},
                       'note': 'WG4-B B1：改編自凍結原型的單步 QP；無增廣動態預測模型、'
                               '無延遲補償與偏移估計；每輪冷啟動'}}

    def _apply_coord(self, sim_t):
        """本輪的協同設定寫進 cfg。**過期或沒有 ⇒ 退回啟動值（協同關閉）**，
        不沿用舊的底盤參考速度 —— 那會讓底盤在任務節點停掉之後繼續走。"""
        c = self._coord
        age = (None if (c is None or sim_t is None or self._coord_t is None)
               else float(sim_t) - float(self._coord_t))
        if c is None:
            src, use = 'off', self._coord_launch
        elif age is not None and age > float(self.a.coord_max_age_s):
            src, use = 'stale_off', self._coord_launch
        else:
            src = 'topic'
            use = dict(c)
            use['w_a'] = c['w_a'] if c['w_a'] is not None else \
                self._coord_launch['w_a']
            use['w_p'] = c['w_p'] if c['w_p'] is not None else \
                self._coord_launch['w_p']
        for k, v in use.items():
            setattr(self.cfg, k, v)
        self._n_by_coord_src[src] = self._n_by_coord_src.get(src, 0) + 1
        return dict(src=src, age=None if age is None else round(age, 4),
                    w_vref=use['w_vref'], base_vref=use['base_vref'],
                    w_qn=use['w_qn'], arm_q_nom=use['arm_q_nom'],
                    w_a=use['w_a'], w_p=use['w_p'])

    def _resolve_target(self, T_launch, sim_t):
        """決定**本輪**要用的目標，求解與到達判定共用同一份。

        兩處不能各自取 —— 那會變成「用一個目標求解、用另一個目標判到達」。
        """
        if not getattr(self.a, 'target_topic', ''):
            src = 'launch_arg'
            T, age = T_launch, None
        elif self._T_rx is None:
            src = 'launch_arg_fallback'
            T, age = T_launch, None
        else:
            age = (None if (sim_t is None or self._T_rx_t is None)
                   else float(sim_t) - float(self._T_rx_t))
            if age is not None and age > float(self.a.target_max_age_s):
                src, T = 'topic_stale_hold', self._T_rx
            else:
                src, T = 'topic', self._T_rx
            if age is not None and age > self._max_target_age_s:
                self._max_target_age_s = age
        self._n_by_target_src[src] = self._n_by_target_src.get(src, 0) + 1
        return T, src, age

    def run(self, T_des):
        """主迴圈外包一層：ROS 外部關閉（含 SIGTERM 觸發的 shutdown）時，
        以 stop_why='external_shutdown' 收尾，照常由計數器產生統計。
        其他例外不在此吞掉（由呼叫端保留真正原因）。"""
        try:
            out = self._run_loop(T_des)
        except Exception as e:                  # noqa: BLE001
            # 只有 ROS 外部關閉才在這裡收尾：ExternalShutdownException，或 context 已關閉
            # 時 ROS 呼叫丟出的 RCLError。其他錯誤（ValueError 等）一律往外拋，
            # 由 main 保留真正原因與部分統計。
            if not is_ros_shutdown_exc(e, rclpy.ok()):
                raise
            self._stop_why = 'external_shutdown'
            self._shutdown_exc = f'{type(e).__name__}: {e}'[:300]
            print('[wg2] 外部關閉（ROS shutdown／SIGTERM）⇒ 以累積計數寫出統計'
                  f'（{type(e).__name__}）', flush=True)
            return self._stats()
        # ROS 正常關閉、不拋例外而離開迴圈（while rclpy.ok() 為 False）也要標示
        if not rclpy.ok() and self._stop_why == 'loop_end':
            self._stop_why = 'external_shutdown'
            self._shutdown_exc = '（無例外：rclpy.ok() 為 False，迴圈結束）'
            out = self._stats()
        return out

    def _log_solve(self, rec):
        """求解列寫入紀錄；另計數（與求解呼叫數相減 = 被中斷而未寫入的求解輪）。"""
        self.log.append(rec)
        self._c_solve_logged += 1

    def _run_loop(self, T_des):
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
        self._applied_by_seq = {}
        slot = 0
        U_warm = None
        # **計數器是節點屬性**：外部關閉或例外時，統計仍由實際計數器產生
        self._c_pub = self._c_drop_age = self._c_no_sol = 0
        self._c_solve_calls = 0            # 求解器呼叫次數（獨立計數，供與求解列對帳）
        self._c_solve_logged = 0
        self._c_miss = 0
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
            if self._stop_req is not None:
                # **收到停止後不得再開始求解或發布**（迴圈頂端檢查）
                self._stop_why = f'stop_topic：{self._stop_req}'
                print(f'[wg2] 收到停止請求：{self._stop_req}（迴圈頂端，本輪不求解）',
                      flush=True)
                break
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
            # ---- 在途命令的關節餘裕守衛 ----
            # **即使此刻立刻全停**，已發布尚未生效完畢的那幾筆仍會把某軸
            # 推過有效餘裕線 ⇒ 本輪再怎麼求解都來不及補救，立即記錄並收尾。
            # 只用**當下已知**的在途命令（剛求出、尚未發布的那一筆不算）。
            if self._mguard is not None and self._snap is not None \
                    and len(self._snap.s) == 6 and len(self._cmd_hist) >= 1:
                _nph = int(round(self.cfg.dt / self.cfg.arm_model.phys_dt))
                _left = max(1, int(round((1.0 - (self.d_cmd % 1.0)) * _nph)))
                _ua = self._cmd_hist[-2][1] if len(self._cmd_hist) >= 2 else None
                _un = self._cmd_hist[-1][1]
                _sch = MG.inflight_schedule(_ua, _left, _un, _nph)
                _f = MG.inflight_unavoidable_breach(
                    np.asarray(self._snap.q, float),
                    np.asarray(self._snap.s, float), _sch, self.cfg,
                    n_tail=2 * _nph, bounds=self._mg_bounds)
                if _f['unavoidable']:
                    self._mg_info = dict(
                        why='inflight_unavoidable_breach', **{
                            k: _f[k] for k in ('worst_margin', 'worst_joint',
                                               'worst_step', 'worst_kind',
                                               'first_breach_step',
                                               'n_inflight_steps')})
                    self.log.append(dict(
                        slot=slot, sim_t=self._snap.sim_t, ok=False,
                        reason='inflight_unavoidable_breach', published=False,
                        margin_guard=self._mg_info,
                        timing_ms={'total': 0.0}, cycle_wall_ms=_cycle_wall))
                    print(f'[wg2] **在途命令必然穿過關節餘裕線，停止任務推進**：'
                          f'j{_f["worst_joint"]} {_f["worst_kind"]} '
                          f'餘裕 {_f["worst_margin"]:+.6f} rad', flush=True)
                    self._stop_why = 'inflight_unavoidable_breach'
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
                    self._c_miss += _m
                    continue
                if gs == HOLD:
                    # **不求解、不發布、不重送初始化命令、不沿用舊設定點。**
                    self.log.append(dict(
                        slot=slot, sim_t=self.sim_now(), ok=False,
                        reason='sp_gate_hold', published=False,
                        gate_state=gs, gate_why=gwhy,
                        timing_ms={'total': 0.0}, cycle_wall_ms=_cycle_wall))
                    slot, _m = self._reschedule(slot)
                    self._c_miss += _m
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
                self._c_miss += _m
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
                self._c_miss += _m
                continue
            age_in = self.sim_now() - snap.sim_t
            if age_in > self.a.max_input_age:
                # **求解前就過期**也要留紀錄，否則 log 看不到被擋下的輪次
                self._c_drop_age += 1
                self.log.append(dict(slot=slot, sim_t=snap.sim_t,
                                     age_in=round(age_in, 6), age_out=None,
                                     u_prev_src='n/a', ok=False,
                                     reason='stale_before_solve',
                                     published=False, timing_ms={'total': 0.0},
                                     cycle_wall_ms=_cycle_wall,
                                     dropped=f'求解前輸入已過期 '
                                             f'{age_in * 1e3:.0f} ms'))
                slot, _m = self._reschedule(slot)
                self._c_miss += _m
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
                self._c_miss += _m
                continue
            _park_rec = None
            if self._park_hold:
                _ub = [float(x) for x in np.asarray(u_prev, float)[:3]]
                _why = None
                if self._park_anchor is None:
                    _why = '沒有停車錨點（/park/gate 未通過或未收到）'
                else:
                    _okp, _why = hold_uprev_ok(_ub, self._park_servo, float(self.a.park_uprev_tol))
                if _why is not None:
                    self.log.append(dict(slot=slot, sim_t=snap.sim_t, age_in=round(age_in, 6),
                                         age_out=None, u_prev_src=src, ok=False,
                                         reason='park_mode_refused', published=False,
                                         timing_ms={'total': 0.0}, cycle_wall_ms=_cycle_wall,
                                         park_hold=True, u_prev_base=_ub, dropped=f'PARK_HOLD：{_why}'))
                    self._stop_why = f'park_mode_refused：{_why}'
                    print(f'[wg2] **PARK_HOLD 模式閘門拒絕**：{_why}', flush=True)
                    break
                # 停車伺服律：同時刻快照的實測底盤位姿 → u_hold（本輪整個時域的底盤等式值）
                _pose = [float(q0[0]), float(q0[1]), float(q0[2])]
                _uh = park_hold_cmd(self._park_anchor, _pose, self._park_servo)
                self.cfg.base_hold = tuple(float(x) for x in _uh)
                _park_rec = {'u_hold': [float(x) for x in _uh], 'pose': _pose,
                             'anchor': list(self._park_anchor),
                             'servo': dict(self._park_servo.__dict__)}
            if self._base_fixed and float(np.max(np.abs(
                    np.asarray(u_prev, float)[:3]))) > float(self.a.park_uprev_tol):
                # **模式閘門**：固定底盤下承接的底盤套用值超出容許 ⇒ 拒絕並停止。
                # 不是宣稱 QP 必然無解（小的非零值本可在加速度框內降到零）；是接線／模式不符。
                _ub = [float(x) for x in np.asarray(u_prev, float)[:3]]
                self.log.append(dict(slot=slot, sim_t=snap.sim_t,
                                     age_in=round(age_in, 6), age_out=None,
                                     u_prev_src=src, ok=False,
                                     reason='park_mode_refused', published=False,
                                     timing_ms={'total': 0.0}, cycle_wall_ms=_cycle_wall,
                                     base_fixed=True, u_prev_base=_ub,
                                     dropped=f'固定底盤：u_prev 底盤 {_ub} 超出 {self.a.park_uprev_tol}'))
                self._stop_why = f'park_mode_refused：u_prev 底盤 {_ub}'
                print(f'[wg2] **固定底盤模式閘門拒絕**：u_prev 底盤 {_ub}', flush=True)
                break
            _solve_sim_t0 = self.sim_now()
            # **本輪的目標只解析一次**，求解與到達判定共用同一份
            T_cyc, _tgt_src, _tgt_age = self._resolve_target(
                T_des, _solve_sim_t0)
            _solve_wall0 = time.monotonic()
            # **無偏移追蹤（offset-free）**：線上估計手臂的恆定穩態偏移，
            # 餵進模型的偏差項。預設關閉，既有趟次行為一位元未改。
            #
            # 為什麼需要：手臂模型 act⁺ = act + α(s − act) + b 預測 act → s，
            # 而 b 是在自由空間（free4）、另一個姿態下辨識的（≈ 0）。抽屜的
            # 抓取姿態下 j2 承重，實測**穩態**下垂 +0.0243 rad、j3 −0.0099 rad，
            # 全程恆定。求解器把設定點擺到「若到位就是 3.24 mm」的位置，手臂卻
            # 停在 12.13 mm；模型預測誤差會自己消失，於是不再推 —— 包括底盤。
            # 離線重現：有下垂、模型不知道 ⇒ 平台 10.88 mm；加估計 ⇒ 0.02 mm。
            #
            # 估計式：d̂ ← d̂ + (dt/τ)·((q_arm − s) − d̂)，**只在設定點近乎
            # 靜止時更新** —— 移動中 q 落後 s 是模型本來就有的一階遲滯，
            # 不能當成偏移。再令 b = α ⊙ d̂，模型穩態就變成 act = s + d̂。
            _om_rec = None
            if self.a.offset_free and not self._b1:
                _s_now = np.asarray(snap.s, float)
                _e = np.asarray(q0[3:], float) - _s_now
                # **靜態初值**（--offset-init-static，預設關）：第一輪時手臂
                # 剛由展開節點交出、設定點靜止，q − s 就是下垂量本身。
                # 由 0 起算的話，求解器一動、門檻就關上，整段啟動暫態模型都以為
                # 沒有下垂（j2 實測 0.027 rad）—— motm_172100 啟動擺盪 ±45 mm。
                if (self.a.offset_init_static and self._of_n_upd == 0
                        and self._of_s_prev is None):
                    self._of_d = np.clip(_e, -self.a.offset_max_rad,
                                         self.a.offset_max_rad)
                    self._of_init = [float(x) for x in self._of_d]
                _gate_ok = (self._of_s_prev is not None and float(np.max(
                    np.abs(_s_now - self._of_s_prev))) < self.a.offset_gate_rad)
                _om = (None if self._of_moving is None else
                       self._of_moving.observe(snap.sim_t, _s_now, np.asarray(q0[3:], float)))
                _om_used = 'none'
                if _gate_ok:
                    self._of_d += (self.cfg.dt / self.a.offset_tau_s) * (
                        _e - self._of_d)
                    self._of_n_upd += 1
                    _om_used = 'static'
                elif _om is not None and _om['ok']:
                    # v2.1：持續等速 ⇒ 模型一致的觀測 d_obs = (q − s) + (dt_p/α)·ṡ（同一濾波、同一限幅）
                    self._of_d += (self.cfg.dt / self.a.offset_tau_s) * (
                        np.asarray(_om['d_obs'], float) - self._of_d)
                    self._of_n_upd += 1
                    self._of_n_upd_moving += 1
                    _om_used = 'moving'
                self._of_d = np.clip(self._of_d, -self.a.offset_max_rad,
                                     self.a.offset_max_rad)
                self._of_s_prev = _s_now.copy()
                self.cfg.arm_model.bias = (
                    np.asarray(self.cfg.arm_model.alpha, float) * self._of_d)
                if _om is not None:
                    _om_rec = dict(_om, used=_om_used,
                                   d_hat_after=[float(x) for x in self._of_d])
            _q_sol, _s_sol, _n_comp = q0, np.asarray(snap.s, float), 0
            # 本輪若根本不做延遲補償，選取紀錄必須是空的，不是上一輪的殘留
            self._last_pred_sel = []
            # B1 不做延遲補償（規格：以量測狀態直接求解）
            if self._gate is not None and self.d_state > 0.0 and not self._b1:
                _q_sol, _s_sol, _n_comp = self._predict_delay(
                    q0, np.asarray(snap.s, float), snap.sim_t)
            _coord_rec = self._apply_coord(_solve_sim_t0)
            # **求解前複製**實際送進求解器的暖啟動、偏移與偏差（逐輪重播用）。
            # 只記錄，不影響求解。
            _U_warm_in = (None if U_warm is None
                          else np.array(U_warm, dtype=float, copy=True))
            _d_hat_in = (np.array(self._of_d, dtype=float, copy=True)
                         if (self.a.offset_free and not self._b1) else None)
            _bias_in = (np.array(self.cfg.arm_model.bias, dtype=float,
                                 copy=True)
                        if getattr(self.cfg, 'arm_model', None) is not None
                        else None)
            self._c_solve_calls += 1
            if self._b1:
                # 單步 QP：量測 q 與同時刻設定點 s；s 缺失 ⇒ solve_b1 拒絕（不以實測角代替）
                r = B1CORE.solve_b1(self.K, q0, snap.s, u_prev, T_cyc,
                                    self.cfg, self._b1p)
                # μ 開關原因細分：協同關閉／過期時 cfg 已退回啟動值（無 q_nom）
                if r.b1 and _coord_rec['src'] != 'topic':
                    r.b1['mu_reason'] = {'off': 'coord_off',
                                         'stale_off': 'coord_stale'}.get(
                                             _coord_rec['src'], r.b1['mu_reason'])
                if r.ok:
                    _k = r.b1['mu_reason']
                    self._n_b1_mu[_k] = self._n_b1_mu.get(_k, 0) + 1
            elif self._gate is not None:
                # 增廣狀態由**同時刻快照**組成：q 與 s 同一個物理步。
                # `make_z` 在 s 含非有限值時拋錯，不以實測關節角代替。
                r = solve_sp(self.K, make_z(_q_sol, _s_sol),
                             u_prev, T_cyc, self.cfg, U_warm=U_warm)
            else:
                r = solve(self.K, q0, u_prev, T_cyc, self.cfg, U_warm=U_warm)
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
                       sqp_converged=(None if self._b1    # B1：SQP 欄位不適用
                                      else bool(r.sqp_converged)),
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
                       coord=_coord_rec,
                       n_incomplete=self._n_incomplete,
                       # B1：求解器自己的原因另存（求解後閘門會覆寫 reason）
                       **({'solver_kind': 'b1', 'b1': r.b1,
                           'solver_reason': r.reason} if self._b1 else {}),
                       **({'base_fixed': True} if self._base_fixed else {}),
                       **({'park_hold': _park_rec} if self._park_hold else {}),
                       **({'offset_moving': _om_rec} if self._of_moving is not None else {}),
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
                           d_state_cycles=float(self.d_state),
                           d_cmd_cycles=float(self.d_cmd),
                           use_applied=bool(self.a.use_applied_for_predict),
                           n_phys_steps=int(_n_comp),
                           dq_pos_m=(None if _n_comp == 0 else round(float(
                               np.linalg.norm(np.asarray(_q_sol)[:3]
                                              - q0[:3])), 6)),
                           dq_arm_max_rad=(None if _n_comp == 0 else
                                           round(float(np.abs(
                                               np.asarray(_q_sol)[3:]
                                               - q0[3:]).max()), 6))))
            # ---- **求解輸入的完整紀錄**（供離線逐筆重播核對）----
            # 先前只記 dq_pos_m／dq_arm_max_rad 兩個純量摘要、且都捨入到
            # 1e-6，無法驗證求解器實際收到的 15 維預測狀態；離線重播因此
            # 停在 p50 2.3e-4 / max 1.8e-2 rad/s 的殘差上。
            # **全部不捨入**；時間也保留原始精度（log 另外那些欄位仍捨入
            # 到 1e-6，維持既有工具可用）。
            rec['solve_in'] = dict(
                q_pred=[float(v) for v in np.asarray(_q_sol, float)],
                s_pred=[float(v) for v in np.asarray(_s_sol, float)],
                q_meas=[float(v) for v in np.asarray(q0, float)],
                s_meas=[float(v) for v in np.asarray(snap.s, float)],
                u_prev=[float(v) for v in np.asarray(u_prev, float)],
                u_prev_src=src,
                snap_sim_t_exact=float(snap.sim_t),
                solve_start_sim_t_exact=float(_solve_sim_t0),
                pred_sel=[[int(a_), str(b_)] for a_, b_ in self._last_pred_sel],
                n_pred_applied_cycle=sum(
                    1 for _, b_ in self._last_pred_sel if b_ == 'applied'),
                n_pred_requested_cycle=sum(
                    1 for _, b_ in self._last_pred_sel if b_ == 'requested'))
            # 逐輪求解輸入／輸出（失敗與未發布輪也留；未捨入）
            rec['solve_in'].update(solver_io_record(
                T_cyc=T_cyc, target_src=_tgt_src, target_age_s=_tgt_age,
                # B1：單步、冷啟動、無增廣模型 ⇒ 暖啟動與模型偏差不是求解輸入，步數 1
                U_warm=(None if self._b1 else _U_warm_in),
                U_sol=(r.U if r.ok else None),
                offset_d_hat=_d_hat_in,
                arm_bias=(None if self._b1 else _bias_in),
                solver_N=(1 if self._b1 else self.cfg.N)))
            if not r.ok:
                self._c_no_sol += 1
                rec['published'] = False
                self._log_solve(rec)
                slot, _m = self._reschedule(slot)
                self._c_miss += _m
                continue
            rec['clock_moved'] = bool(_clock_moved)
            rec['state_moved'] = bool(_state_moved)
            rec['time_ref'] = round(_ref, 6)
            if (not _clock_moved) and (not _state_moved) \
                    and r.timing_ms['total'] > self.a.max_input_age * 1e3:
                # **無法確認時間依據已更新，而求解耗時又超過年齡界限**
                # ⇒ 不能判斷解是否過期 ⇒ **不發布**。
                self._c_drop_age += 1
                rec['published'] = False
                rec['dropped'] = (f'時間依據未更新（clock/state 皆未前進）'
                                  f'而求解耗時 {r.timing_ms["total"]:.0f} ms '
                                  f'> {self.a.max_input_age*1e3:.0f} ms')
                self._log_solve(rec)
                slot, _m = self._reschedule(slot)
                self._c_miss += _m
                continue
            if age_out > self.a.max_input_age:
                # **過期解不發布，不重新蓋時間**
                self._c_drop_age += 1
                rec['published'] = False
                rec['dropped'] = f'輸入已過期 {age_out*1e3:.0f} ms'
                self._log_solve(rec)
                slot, _m = self._reschedule(slot)
                self._c_miss += _m
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
                    self._log_solve(rec)
                    if gs2 == FAILED:
                        print(f'[wg2] **求解後回報失效閂鎖**：{why2}', flush=True)
                        self._stopped_on_fail = True
                        break
                    slot, _m = self._reschedule(slot)
                    self._c_miss += _m
                    continue
            # **到達判定與整形共用同一組誤差**：都用實測狀態 q0 的 FK，
            # 不是核心預測。先算，供整形使用。
            e = task_error(self.K, q0, T_cyc, self.cfg.tcp)
            _ep, _er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            # ---- **近目標輸出整形**（求解之後，發布之前）----
            # 與離線模擬共用 `shape_near_target`，不各寫一份。
            _u_out, _shape_sc, _shape_rs = r.u0, 1.0, 'off'
            if self._park_hold:
                # PARK_HOLD：近目標整形會縮放完整九維（死區還會歸零），會改掉 u_hold ⇒ 停用（策略差異，照實記錄）
                _shape_rs = 'disabled_park_hold'
            elif (self.a.near_target_gamma > 0.0
                    or self.a.near_target_deadband_m > 0.0):
                _u_out, _shape_sc, _shape_rs = shape_near_target(
                    r.u0, self.K.jacobian(q0, self.cfg.tcp), float(q0[2]),
                    _ep, _er, u_prev, self.cfg,
                    gamma=self.a.near_target_gamma,
                    deadband_m=self.a.near_target_deadband_m,
                    deadband_rad=self.a.near_target_deadband_rad,
                    tol_p=self.a.reach_pos_m, tol_r=self.a.reach_rot_rad)
                if _shape_sc < 1.0:
                    self._n_shaped += 1
                self._shape_scales.append(float(_shape_sc))
            rec['shape'] = {'scale': round(float(_shape_sc), 6),
                            'reason': _shape_rs}
            if self._t_sim0 is None:
                # 任務的**模擬時間**起點 = 首次成功發布那一輪的快照時間
                self._t_sim0 = snap.sim_t
            self._last_solved_key = _key
            self._last_solved_sim_t = snap.sim_t
            if self._stop_req is not None:
                # **發布前**再查一次：求解期間收到停止 ⇒ 本輪不發布
                rec['published'] = False
                rec['dropped'] = f'收到停止請求，不發布：{self._stop_req}'
                self._log_solve(rec)
                self._stop_why = f'stop_topic：{self._stop_req}'
                print(f'[wg2] 收到停止請求：{self._stop_req}（發布前，本輪不發布）',
                      flush=True)
                break
            _hok, _hwhy = ((True, None) if not (self._park_hold and _park_rec is not None) else
                           hold_output_ok(np.asarray(_u_out, float)[:3], _park_rec['u_hold'],
                                          float(self.a.park_out_tol)))
            if not _hok:
                rec['published'] = False
                rec['reason'] = 'park_mode_output'
                rec['dropped'] = f'PARK_HOLD：{_hwhy}'
                self._log_solve(rec)
                self._stop_why = 'park_mode_output'
                print(f'[wg2] **PARK_HOLD 輸出核對失敗**：{rec["dropped"]}', flush=True)
                break
            if self._base_fixed and float(np.max(np.abs(
                    np.asarray(_u_out, float)[:3]))) > float(self.a.park_out_tol):
                # **輸出核對**：整形後的發布命令底盤分量必須為零；否則不發布、停止（不剪掉後繼續）
                rec['published'] = False
                rec['reason'] = 'park_mode_output'
                rec['dropped'] = (f'固定底盤：發布命令底盤 '
                                  f'{[float(x) for x in np.asarray(_u_out)[:3]]} 非零')
                self._log_solve(rec)
                self._stop_why = 'park_mode_output'
                print(f'[wg2] **固定底盤輸出核對失敗**：{rec["dropped"]}', flush=True)
                break
            rec['publish_sim_t'] = round(self.sim_now(), 6)
            # 四段紀錄裡的 request 段**用同一個值**，不另算一次轉換
            u_world = self._publish_u(_u_out, float(q0[2]))
            # `_publish_u` 內部只讀一次時鐘，這裡記下**它用的那一個值**
            # （未捨入）。既有的 publish_sim_t 仍保留，不動既有工具。
            rec['publish_sim_t_exact'] = self._last_pub_t
            self._c_pub += 1
            U_warm = r.U
            # **到達並保持**：用實測狀態算的 FK 誤差，不是核心預測。
            # （e／_ep／_er 已於整形前算出，此處沿用同一組值）
            _in_tol = (_ep <= self.a.reach_pos_m and _er <= self.a.reach_rot_rad)
            # **目標換了就重新計算保持。** 連續模式下目標會一路移動；不重設
            # 的話，第二個目標會繼承第一個目標的「已到達」。門檻取到達容差
            # 的一半：比它小的移動屬於追蹤中的連續變化，不算換目標。
            if self.a.continuous and self._T_last is not None:
                _dT = float(np.linalg.norm(T_cyc[:3, 3] - self._T_last[:3, 3]))
                if _dT > 0.5 * self.a.reach_pos_m:
                    if self._reached_held or self._hold_t0 is not None:
                        self._n_target_changes += 1
                    self._reached_held = False
                    self._reach_announced = False
                    self._hold_t0 = None
            self._T_last = np.array(T_cyc, float)
            # **開著目標話題卻還沒收到任何一筆時，不得宣告到達。**
            # 退路目標是「保持實測起始 TCP」，誤差天生就是 0 ⇒ 會立刻「到達
            # 並保持」。實跑 nav_full_030409 正是如此：41 輪全部 launch_arg_
            # fallback、n_rx = 0，卻印出「到達並保持 2.0 s」@ 0.00 mm。
            # 那不是 ALIGN 的到達，是對自己起點的到達。
            if _tgt_src == 'launch_arg_fallback':
                _in_tol = False
                self._n_tol_blocked_fallback += 1
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
                            'body': [round(float(x), 8) for x in _u_out],
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
                       request_body=[round(float(x), 8) for x in _u_out],
                       request_body_presolve=[round(float(x), 8)
                                              for x in r.u0],
                       # **兩次座標轉換用的 yaw 與時間**各自記錄：
                       # 本節點用 snap 的 yaw；adapter 用它自己收到的 /odom yaw。
                       # 往返代數正確**只在同一 yaw 下**成立，
                       # 兩者若不同步就會有實際誤差 —— 要能事後對帳。
                       conv_yaw_node=round(float(q0[2]), 9),
                       conv_yaw_src_sim_t=round(snap.sim_t, 6),
                       conv_at_mono=round(time.monotonic(), 6),
                       err_p=_ep, err_r=_er)
            self._log_solve(rec)
            # ---- 狀態給任務編排節點（每輪）----
            if self.status_pub is not None:
                _sm = Float64MultiArray()
                _sm.data = [float(snap.sim_t), float(_ep), float(_er),
                            1.0 if _in_tol else 0.0,
                            1.0 if self._reached_held else 0.0]
                self.status_pub.publish(_sm)
            if self._reached_held:
                if not self._reach_announced:
                    print(f'[wg2] **到達並保持 {self.a.hold_s:.1f} s** '
                          f'@ sim {snap.sim_t:.3f}', flush=True)
                    self._reach_announced = True
                # **連續模式不結束**：整個抽屜任務的目標會一路移動
                #（接觸前 → 抓取 → 隨開度移動 → 退開），由任務節點決定
                # 相位，求解節點只負責跟。單一目標模式照舊在此結束。
                if not self.a.continuous:
                    break
            if self._stop_req is not None:
                self._stop_why = f'stop_topic：{self._stop_req}'
                print(f'[wg2] 收到停止請求：{self._stop_req}', flush=True)
                break
            slot, _m = self._reschedule(slot)
            self._c_miss += _m
            if _m:
                # **跨過多個時槽**：暖啟動序列假設只前進一步，
                # 不能再當成有效的 nominal。最小處理是丟棄重建，
                # **不**臨時改模型 dt 來補救。
                U_warm = None
                self._n_warm_discard += 1
        return self._stats()

    def _stats(self):
        """由節點計數器組統計（正常結束、外部關閉、例外收尾共用）。"""
        return dict(stop_why=self._stop_why,
                    base_fixed=bool(self._base_fixed),
                    park_hold=bool(self._park_hold),
                    offset_moving=bool(self._of_moving is not None),
                    offset_n_upd_moving=int(self._of_n_upd_moving),
                    stop_rx=self._stop_rx,
                    last_publish_sim_t=self._last_pub_t,
                    n_published_after_stop_rx=int(self._c_pub_after_stop),
                    n_solve_rows_logged=int(self._c_solve_logged),
                    n_solve_calls_unlogged=int(self._c_solve_calls - self._c_solve_logged),
                    shutdown_exc=self._shutdown_exc,
                    delay_comp_state_cycles=float(self.d_state),
                    delay_comp_cmd_cycles=float(self.d_cmd),
                    use_applied_for_predict=bool(self.a.use_applied_for_predict),
                    near_target_gamma=float(self.a.near_target_gamma),
                    n_shaped=int(self._n_shaped),
                    shape_scale_p50=(float(np.percentile(self._shape_scales, 50))
                                     if self._shape_scales else None),
                    shape_scale_min=(float(min(self._shape_scales))
                                     if self._shape_scales else None),
                    n_pred_applied=int(self._n_pred_applied),
                    n_pred_requested=int(self._n_pred_requested),
                    cmd_env=bool(self.a.cmd_env),
                    n_source_seq=int(self._source_seq),
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
                    published=self._c_pub, dropped_stale=self._c_drop_age,
                    no_solution=self._c_no_sol, deadline_miss=int(self._c_miss),
                    n_solve_calls=self._c_solve_calls,
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


def validate_target_rot(values):
    """列優先 9 個值 → 3x3 旋轉矩陣。**不合法就拋出**，不默默當成單位矩陣。

    為什麼要核：任務誤差的姿態項是從 R_des 算出來的。傳進一個不是旋轉的矩陣
    （例如含反射或非正交）時，誤差仍算得出數字，但那個數字不對應任何姿態，
    而且不會有任何訊息。
    """
    import numpy as _np
    v = _np.asarray(values, dtype=float)
    if v.size != 9:
        raise ValueError(f'目標姿態需要 9 個值（列優先），收到 {v.size} 個')
    if not _np.isfinite(v).all():
        raise ValueError('目標姿態含非有限值')
    R = v.reshape(3, 3)
    e = float(_np.abs(R.T @ R - _np.eye(3)).max())
    if e > 1e-9:
        raise ValueError(f'目標姿態不是正交矩陣：R^T R − I 最大 {e:.3e}')
    d = float(_np.linalg.det(R))
    if abs(d - 1.0) > 1e-9:
        raise ValueError(f'目標姿態的行列式 {d:.12f} 不是 +1（反射不是旋轉）')
    return R


def wait_for_drawer_handover(nd, ex, timeout_s):
    """Initialize early without sending even a zero handshake command."""
    nd.drawer_ready_pub.publish(Bool(data=True))
    print('[wg2] 已初始化，待命等展開端 RELEASED 與任務 ALIGN', flush=True)
    t0 = time.monotonic()
    while rclpy.ok():
        if nd._stop_req is not None:
            return '待命中收到停止要求'
        if nd._drawer_released and nd._drawer_start:
            return None
        if time.monotonic() - t0 >= timeout_s:
            return '等待抽屜控制交棒逾時；未發任何命令'
        ex.spin_once(timeout_sec=0.02)
    return '等待抽屜控制交棒時 ROS 結束'


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
    # **六維目標的姿態**：列優先的 3x3 旋轉矩陣（9 個值）。
    # 不給 ⇒ 沿用起始姿態（既有行為完全不變）。
    # 用矩陣而非 rpy：階段 A 的 R_DES 本來就是以矩陣定義的，
    # 轉成角度再轉回來會引入不必要的誤差與順序約定。
    ap.add_argument('--target-rot', nargs=9, type=float, default=None,
                    help='目標姿態，列優先 3x3 旋轉矩陣；不給 = 保持起始姿態')
    # **執行期目標話題。預設 '' = 關閉，行為與既有趟次一位元相同。**
    # 抽屜實驗需要這個：接觸前的退讓目標要由**實測把手位姿**算出，而開啟／
    # 關閉段的目標會隨實測開度移動（見 evaluation/drawer_target.py）。
    # 啟動參數給的是固定目標，做不到這件事。
    ap.add_argument('--continuous', action='store_true',
                    help='到達後不結束，持續跟隨目標話題；每輪發 /wgmpc/status，'
                         '收到 /wgmpc/stop 才結束（整個抽屜任務用）')
    ap.add_argument('--offset-free', action='store_true',
                    help='線上估計手臂恆定穩態偏移並餵進模型偏差項（預設關閉）')
    ap.add_argument('--offset-tau-s', type=float, default=0.5,
                    help='偏移估計的時間常數（模擬時間）')
    ap.add_argument('--offset-moving', action='store_true',
                    help='v2.1：設定點持續等速時，以 d_obs = (q − s) + (dt_p/α)·ṡ 更新偏移估計（預設關）')
    ap.add_argument('--offset-gate-rad', type=float, default=0.002,
                    help='設定點每輪變化小於此值才更新估計（避開一階遲滯）')
    ap.add_argument('--offset-init-static', action='store_true',
                    help='偏移估計第一輪以 q − s 初始化（交接時手臂靜止，'
                         '那就是下垂量）；預設關 = 由 0 起算（既有行為）')
    ap.add_argument('--offset-max-rad', type=float, default=0.05,
                    help='估計值上限；防止估計跑掉，不是控制限制')
    ap.add_argument('--cmd-topic', default='/wholebody_safety/cmd_in',
                    help='九維命令的輸出話題。預設 = 既有行為（經安全層）。'
                         '抽屜實驗用 /wb_vel_cmd —— 那條管線沒有安全層，'
                         '要在報告裡明寫')
    ap.add_argument('--target-topic', default='',
                    help='16 元素列優先 4x4 的目標位姿話題；'
                         "'' = 關閉，只用啟動參數的固定目標")
    ap.add_argument('--coord-topic', default='',
                    help="整機協同設定話題（parse_coord 版面 v1）；'' = 關閉")
    ap.add_argument('--coord-max-age-s', type=float, default=0.5,
                    help='協同設定最大年齡；過期即退回啟動值（協同關閉）')
    ap.add_argument('--target-max-age-s', type=float, default=0.5,
                    help='目標位姿的最大年齡；超過就**沿用上一筆已接受的**'
                         '並記錄，不以過期值當新鮮值')
    # **這個參數從未被實作**。先前它宣告了卻沒有任何讀取處，傳了會靜默無效 ——
    # 那比沒有這個選項更糟。保留名稱以免既有命令列直接報錯，但非零即中止。
    ap.add_argument('--target-rot-deg', type=float, default=0.0,
                    help='**未實作**：非零即中止，請改用 --target-rot')
    ap.add_argument('--duration-s', type=float, default=30.0,
                    help='**牆鐘**時間上限')
    ap.add_argument('--duration-sim-s', type=float, default=0.0,
                    help='**模擬時間**預算上限（0 = 不啟用）。'
                         '錄影會拖慢 sim:wall，只靠牆鐘上限會讓任務拿到的'
                         '模擬時間比無錄影趟次少 ⇒ 要與 free4 對齊時用這個。')
    ap.add_argument('--near-target-gamma', type=float, default=0.0,
                    help='**近目標輸出整形**：把單週期命令的 TCP 位移限制在'
                         'γ × 當下殘差以內（0 = 關閉）。'
                         'rec10 量到該比值在 <2 mm 時達 8.2 倍 ⇒ 必然過衝。'
                         '**求解之後的輸出整形，不改權重／視界／約束集合。**')
    ap.add_argument('--near-target-deadband-m', type=float, default=0.0)
    ap.add_argument('--near-target-deadband-rad', type=float, default=0.0)
    ap.add_argument('--cmd-env', action='store_true',
                    help='啟用**命令追蹤封裝**（身分與九維值同訊息）。'
                         '既有九維路徑與限制不變。')
    ap.add_argument('--run-id', default='',
                    help='趟次識別，進封裝的 run_id')
    ap.add_argument('--delay-comp-state-cycles', type=float, default=-1.0,
                    help='**D_state**：量測 → 命令生效（狀態要往前推多久）。'
                         '< 0 時沿用 --delay-comp-cycles。'
                         'rec9 逐筆量到 ≈ 1.60 個週期'
                         '（D_pub 0.60 ＋ D_cmd 1.00）。')
    ap.add_argument('--delay-comp-cmd-cycles', type=float, default=-1.0,
                    help='**D_cmd**：發布 → 生效（由發布歷史倒查命令時的偏移）。'
                         '< 0 時沿用 --delay-comp-cycles。'
                         'rec9 逐筆量到 ≈ 1.00 個週期。'
                         '**與 D_state 不是同一個量，不可混用。**')
    ap.add_argument('--use-applied-for-predict', action='store_true',
                    help='預推時優先採用**執行端回報的實際套用值**'
                         '（需 --cmd-env）。rec9 量到安全層會修改 38.1% 的命令，'
                         '所以「發布的命令就是手臂會收到的」不成立。')
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
    ap.add_argument('--base-fixed', action='store_true',
                    help='PARK_FIXED：整個時域 u_base = 0（核心等式）；預設關 = 既有行為')
    ap.add_argument('--park-hold', action='store_true',
                    help='PARK_HOLD（v2）：整個時域 u_base = 停車伺服 u_hold；停用近目標整形；錨點取自 /park/gate')
    ap.add_argument('--park-uprev-tol', type=float, default=1e-6,
                    help='固定底盤模式閘門：承接 u_prev 底盤各軸上限（超出 ⇒ 拒絕並停止）')
    ap.add_argument('--park-out-tol', type=float, default=1e-6,
                    help='固定底盤輸出核對：發布命令底盤各軸上限')
    ap.add_argument('--solver-kind', default='wgmpc', choices=['wgmpc', 'b1'],
                    help='wgmpc = 既有 W-GMPC（預設，不變）；b1 = WG4-B 單步 QP 對照（wg4b_b1_core.py）')
    ap.add_argument('--b1-kp', type=float, default=1.0,
                    help='B1 任務增益 kp_p = kp_r（事前登錄候選 1.0 → 2.0 → 0.5）')
    ap.add_argument('--arm-model', default='ideal',
                    choices=['ideal', 'setpoint'],
                    help='ideal = 原核心（理想速度積分，free4 基準，預設不變）；'
                         'setpoint = 增廣核心（手臂設定點納入狀態）')
    ap.add_argument('--arm-ident',
                    default=os.path.join(_HERE, 'results',
                                         'wgmpc_arm_sp_ident_free4.json'),
                    help='setpoint 模式用的辨識檔（α、b、physics_dt）')
    ap.add_argument('--w-s', type=float, default=1e-3,
                    help='命令變化率權重（S = w_s/vmax² · I）。'
                         '預設 1e-3 = 既有值；離線掃描指出 0.1 可同時抑制'
                         '移動段的命令甩動與保持窗的極限環')
    ap.add_argument('--w-a', type=float, default=1e-3,
                    help='手臂命令大小權重（R 的手臂部分，以 vmax² 正規化）。'
                         '預設 1e-3 = 既有值；0.05 可把 SQP 的 no_progress 壓到 0')
    ap.add_argument('--margin-guard', action='store_true',
                    help='啟用關節餘裕的**命令時間線**守衛：在途命令若必然'
                         '穿過「硬限位±joint_margin」就立即記錄並走既定收尾。'
                         '預設關閉 ⇒ 既有趟次行為不變。'
                         '**提前量有物理上限**（在途占 1.6 週期，實測只早 1 輪）；'
                         '真正防止設定點穿線的是執行端的拒寫。')
    ap.add_argument('--mg-max-qp-fail', type=int, default=10,
                    help='連續 qp_failed 的有界收尾門檻（診斷用，'
                         '**不是**防止首次穿線的保護）')
    ap.add_argument('--mg-max-breach', type=int, default=5,
                    help='連續前瞻到穿線的有界收尾門檻（同上）')
    ap.add_argument('--w-s-arm', type=float, default=None,
                    help='只給**手臂**的變化率權重；None = 沿用 --w-s。'
                         'S 以 vmax² 正規化，底盤 vmax 0.0353 ⇒ 既有 1e-3 '
                         '對底盤等效 0.805、對手臂只有 1e-3（差 800 倍），'
                         '甩動只發生在手臂 ⇒ 分開設才對症')
    ap.add_argument('--w-s-base', type=float, default=None,
                    help='只給**底盤**的變化率權重；None = 沿用 --w-s')
    ap.add_argument('--no-row-scaling', action='store_true',
                    help='關閉等價正值列縮放（對照用）')
    ap.add_argument('--handshake-timeout-s', type=float, default=20.0,
                    help='啟動握手與狀態齊備的等待上限')
    ap.add_argument('--phys-dt', type=float, default=0.01,
                    help='介面契約核對用的物理步長；與執行端 meta 比對')
    ap.add_argument('--out', default='')
    ap.add_argument('--wait-for-drawer-handover', action='store_true',
                    help='提早初始化並待命；展開端交出且任務進入 ALIGN 才啟動握手')
    ap.add_argument('--drawer-start-timeout-s', type=float, default=240.0)
    a = ap.parse_args()
    rclpy.init()
    nd = WGMPCNode(a)

    def _bail(code, why):
        """**拒絕啟動也要留紀錄**：只印在終端的話，事後無從對帳。"""
        print(f'**{why}**', flush=True)
        if a.out:
            json.dump({'args': vars(a), 'started': False,
                       'exit_code': code, 'refuse_reason': why,
                       **nd.b1_report(),
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
    if a.wait_for_drawer_handover:
        # 在初始化握手之前等待，連全零握手命令都不得覆寫展開控制。
        why = wait_for_drawer_handover(nd, ex, a.drawer_start_timeout_s)
        if why is not None:
            return _bail(6, why)
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
    if abs(float(a.target_rot_deg)) > 0.0:
        return _bail(2, '--target-rot-deg 從未被實作（傳了會靜默無效）；'
                        '請改用 --target-rot 給完整旋轉矩陣')
    if a.target_rot is None:
        T_des[:3, :3] = T[:3, :3]    # 不給姿態 ⇒ **保持起始姿態**（既有行為）
        _rot_src = 'start_orientation'
    else:
        try:
            T_des[:3, :3] = validate_target_rot(a.target_rot)
        except ValueError as e:
            return _bail(2, f'--target-rot {e}')
        _rot_src = 'target_rot'
    print(f'[wg2] 目標姿態來源：{_rot_src}', flush=True)
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
    _exc = None
    try:
        stats = nd.run(T_des)
    except BaseException as e:          # 保留真正原因後照樣往外拋
        _exc = e
        raise
    finally:
        if stats is None:
            # 例外收尾：仍由**實際計數器**組統計，並標部分統計與真正原因
            try:
                stats = nd._stats()
            except Exception as e2:     # noqa: BLE001
                stats = {'stats_error': f'{type(e2).__name__}: {e2}'}
            stats['stats_partial'] = True
            stats['stop_why'] = (f'exception：{type(_exc).__name__}：{_exc}'
                                 if _exc is not None else stats.get('stop_why'))
        if a.out:
            json.dump({'args': vars(a), 'started': True,
                       'stats': stats,
                       # **執行期目標的來源要逐輪可查。**
                       # 「用了啟動參數的固定目標」與「用了話題給的目標」
                       # 是兩件不同的事，混在一起就說不清那一趟到達的是誰。
                       'coord': {
                           'topic': getattr(a, 'coord_topic', ''),
                           'max_age_s': getattr(a, 'coord_max_age_s', None),
                           'n_rx': nd._n_coord_rx,
                           'reject_reasons': dict(nd._coord_reject),
                           'n_cycles_by_source': dict(nd._n_by_coord_src),
                           'launch_values': {k: v for k, v in
                                             nd._coord_launch.items()}},
                       'runtime_target': {
                           'topic': getattr(a, 'target_topic', ''),
                           'max_age_s': getattr(a, 'target_max_age_s', None),
                           'n_rx': nd._n_target_rx,
                           'n_rejected': nd._n_target_rejected,
                           'reject_reasons': dict(nd._target_reject),
                           'n_cycles_by_source': dict(nd._n_by_target_src),
                           'max_observed_age_s': round(
                               nd._max_target_age_s, 6),
                           'n_tol_blocked_fallback':
                               nd._n_tol_blocked_fallback,
                           'tol_blocked_note':
                               '開著目標話題卻還沒收到任何一筆時，到達判定'
                               '一律不成立 —— 退路目標是「保持起始 TCP」，'
                               '誤差天生為 0，會產生假的「到達並保持」',
                           'note': ('launch_arg = 關閉；launch_arg_fallback = '
                                    '開著但一筆都還沒收到；topic = 用了新鮮的'
                                    '話題目標；topic_stale_hold = 超過年齡'
                                    '上限，沿用上一筆已接受的')},
                       'offset_free': {
                           'enabled': bool(getattr(a, 'offset_free', False)),
                           'd_hat_rad': [round(float(x), 6) for x in nd._of_d],
                           'init_static': getattr(nd, '_of_init', None),
                           'n_updates': int(nd._of_n_upd),
                           'n_updates_moving': int(nd._of_n_upd_moving),
                           'moving_enabled': bool(nd._of_moving is not None),
                           'tau_s': getattr(a, 'offset_tau_s', None),
                           'gate_rad': getattr(a, 'offset_gate_rad', None),
                           'note': '手臂恆定穩態偏移的線上估計；b = α ⊙ d̂'},
                       **nd.b1_report(),
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
