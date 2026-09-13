"""完整 9 維命令的輪級限制（執行版本 E2）。

**座標系（重要）**：輪速映射 `r·ω = W·ξ` 的 ξ 必須是**底盤本體座標**速度。
Isaac 執行端收到的 `/wb_vel_cmd` 已由 `arm_vel_adapter` 從安全層的
`report_frame` 轉成本體座標（執行端再轉回世界座標才呼叫速度 API），
所以本模組的輸入就是**本體命令**。`u_prev` 必須保存為
**上一物理步真正套用的本體命令**，不是 report frame 的值、也不是世界速度。
yaw ≈ 0 時兩者碰巧一致，不可據此混稱。

策略見 `evaluation/results/specs/wb_wheel_limit_policy_v2.md`。
本檔是**純函式模組**，不含 ROS、不含模擬器，可完全離線測試。

核心：λ 由**底盤**分量的輪速／輪加速度約束決定，但**施加於完整 9 維增量**——

    u_out = u_prev + λ · (u_req − u_prev)

只縮底盤會改掉底盤與手臂的相對比例，上游求解出的耦合方向就沒了。
同一個 λ 讓 u_out 落在 9 維空間中 u_prev→u_req 的線段上。

共用 λ 保留的是 **`u_req − u_prev` 這個增量的方向**，
**不是**一般情況下請求命令本身的底盤／手臂比例 ——
只有 `u_prev = 0` 時兩者才恰好一致。

**不得由此宣稱上游全身安全性不變**：u_out 不是求解器的輸出，
上游的距離約束是對 u_req 算的。λ < 1 時 `modified` 為 True，
呼叫端必須記錄。

**逾時是完整 9 維插值的例外**，見 `limit9_timeout`。

E1 保持不變；本檔是另立的版本，不改動 E1 的檔案。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POLICY = 'evaluation/results/specs/wb_wheel_limit_policy_v2.md'
VERSION = 'wb_wheel_limit/1'

NORMAL = 'normal'
TIMEOUT = 'timeout'                     # 逾時減速：**不是**完整 9 維插值
STOP_UNVERIFIED = 'stop_unverified'     # 不在輪級加速度保證之內


def wheel_matrix(L: float) -> np.ndarray:
    """r·ω = W·ξ_body。**ξ 必須是底盤本體座標速度**，不是 report frame、
    也不是世界速度。列序與 gmpc 的耦合約束一致。"""
    return np.array([[0.0, 1.0, L],
                     [-1.0, 0.0, L],
                     [0.0, -1.0, L],
                     [1.0, 0.0, L]])


@dataclass
class WheelLimitConfig:
    wheel_radius: float = 0.05          # m
    wheel_base_L: float = 0.245         # m，取自 URDF base_link_rim_*_joint
    wheel_w_max: float = 5.55           # rad/s
    wheel_a_max: float = 125.0          # rad/s²
    arm_rate_max: float = 1.0           # rad/s（與 E1 的整筆拒收門檻一致）
    dt_max: float = 0.05                # s；超過視為中斷，不當成加速度預算
    tol: float = 1e-9

    @property
    def w_lim(self) -> float:
        return self.wheel_radius * self.wheel_w_max

    def a_lim(self, dt: float) -> float:
        return self.wheel_radius * self.wheel_a_max * dt


@dataclass
class LimitResult:
    u_out: np.ndarray | None
    lam: float | None
    reason: str
    mode: str
    modified: bool = False
    wheel_speed_max: float = float('nan')
    wheel_accel_max: float = float('nan')
    binding: list = field(default_factory=list)
    # 逾時模式為 False：手臂被獨立歸零，本步不保留耦合方向
    coupling_preserved: bool = True

    def as_row(self, u_req, u_prev, dt):
        """policy_v2 §5 的記錄欄位。"""
        f = lambda v: [round(float(x), 6) for x in np.asarray(v, float)]  # noqa
        return {'u_req': f(u_req), 'u_prev': f(u_prev),
                'u_out': None if self.u_out is None else f(self.u_out),
                'lam': None if self.lam is None else round(float(self.lam), 6),
                'modified': bool(self.modified), 'reason': self.reason,
                'dt': round(float(dt), 6), 'mode': self.mode,
                'wheel_speed_max': round(float(self.wheel_speed_max), 6),
                'wheel_accel_max': round(float(self.wheel_accel_max), 6),
                'binding': list(self.binding),
                'coupling_preserved': bool(self.coupling_preserved)}


def limit9(u_req, u_prev, dt: float, cfg: WheelLimitConfig) -> LimitResult:
    """對完整 9 維命令施加輪級限制。

    回傳的 `u_out` 為 None 時表示**不可在正常模式下套用**；
    呼叫端應改送 `stop_command` 並記錄 `mode = stop_unverified`。
    **不在此偷偷把基準換成零** —— 那會讓加速度保證換了對象。
    """
    u_req = np.asarray(u_req, float)
    u_prev = np.asarray(u_prev, float)
    if u_req.shape != (9,) or u_prev.shape != (9,):
        raise ValueError('命令長度必須為 9')
    if not (np.isfinite(u_req).all() and np.isfinite(u_prev).all()):
        raise ValueError('命令必須全為有限值')

    W = wheel_matrix(cfg.wheel_base_L)
    b = W @ u_prev[:3]

    if not (0.0 < dt <= cfg.dt_max):
        return LimitResult(None, None,
                           'dt_invalid' if dt <= 0.0 else 'dt_gap',
                           STOP_UNVERIFIED)
    if np.max(np.abs(b)) > cfg.w_lim + 1e-6:
        # 前提不成立：前一筆命令本身就超速，交集可能為空
        return LimitResult(None, None, 'prev_infeasible', STOP_UNVERIFIED,
                           wheel_speed_max=float(np.max(np.abs(b))))

    d = W @ (u_req[:3] - u_prev[:3])
    a_lim = cfg.a_lim(dt)
    lam = 1.0
    binding = []
    for k, (bi, di) in enumerate(zip(b, d)):
        if abs(di) <= cfg.tol:
            continue
        hi = (cfg.w_lim - bi) / di if di > 0 else (-cfg.w_lim - bi) / di
        cand_s = max(hi, 0.0)
        cand_a = a_lim / abs(di)
        if cand_s < lam - 1e-12:
            binding = [('wheel_speed', k)]
            lam = cand_s
        elif abs(cand_s - lam) <= 1e-12 and cand_s < 1.0:
            binding.append(('wheel_speed', k))
        if cand_a < lam - 1e-12:
            binding = [('wheel_accel', k)]
            lam = cand_a
        elif abs(cand_a - lam) <= 1e-12 and cand_a < 1.0:
            binding.append(('wheel_accel', k))
    lam = float(min(max(lam, 0.0), 1.0))

    # **同一個 λ 施加於完整 9 維增量**
    u_out = u_prev + lam * (u_req - u_prev)

    kinds = {k for k, _ in binding}
    if lam >= 1.0 - 1e-12:
        reason = 'ok'
    elif kinds == {'wheel_speed'}:
        reason = 'wheel_speed'
    elif kinds == {'wheel_accel'}:
        reason = 'wheel_accel'
    elif kinds:
        reason = 'both'
    else:
        reason = 'ok'

    ws = float(np.max(np.abs(W @ u_out[:3])))
    wa = float(np.max(np.abs(W @ (u_out[:3] - u_prev[:3])))) / dt
    return LimitResult(u_out, lam, reason, NORMAL,
                       modified=lam < 1.0 - 1e-12,
                       wheel_speed_max=ws, wheel_accel_max=wa,
                       binding=[f'{k}[{i}]' for k, i in binding])


def limit9_timeout(u_prev, dt: float, cfg: WheelLimitConfig) -> LimitResult:
    """逾時減速：**底盤經輪級限制、手臂速率立即歸零**。

    **這是完整 9 維插值的例外，而且是必要的例外。**
    若沿用 `u_out = u_prev + λ(0 − u_prev) = (1 − λ)·u_prev`，
    則 λ < 1 且前一筆手臂速度非零時，輸出的手臂速度**仍然非零**，
    設定點會**繼續積分** —— 與「逾時凍結設定點」（E1 語意）直接矛盾。

        例：u_prev 底盤 0.25 m/s、joint2 0.05 rad/s，單步預算 0.0625 m/s
            → λ = 0.25，v_out = 0.1875，dq2_out = 0.0375（**設定點還在動**）

    因此本函式：

        u_out[:3] = (1 − λ) · u_prev[:3]     底盤照輪級限制減速
        u_out[3:] = 0                        手臂**立即**停止積分（保留 E1 語意）

    代價要說清楚：**本步不保留耦合方向**，`coupling_preserved` 為 False。
    呼叫端必須記錄**最後真正套用的整筆命令**。
    """
    u_prev = np.asarray(u_prev, float)
    if u_prev.shape != (9,):
        raise ValueError('命令長度必須為 9')
    target = np.zeros(9)
    r = limit9(target, u_prev, dt, cfg)
    if r.u_out is None:
        r.mode = STOP_UNVERIFIED
        return r
    u_out = r.u_out.copy()
    u_out[3:] = 0.0                      # **例外**：手臂立即歸零，不隨 λ 縮放
    ws = float(np.max(np.abs(wheel_matrix(cfg.wheel_base_L) @ u_out[:3])))
    wa = float(np.max(np.abs(wheel_matrix(cfg.wheel_base_L)
                             @ (u_out[:3] - u_prev[:3])))) / dt
    return LimitResult(u_out, r.lam, 'timeout_decel', TIMEOUT,
                       modified=True, wheel_speed_max=ws, wheel_accel_max=wa,
                       binding=r.binding, coupling_preserved=False)


def stop_command(u_prev) -> np.ndarray:
    """`stop_unverified` 模式下直接送出的命令：全零。

    **這一步不在輪級加速度保證之內**，呼叫端必須以
    `mode = stop_unverified` 記錄。
    """
    return np.zeros(9)


def arm_within(u, cfg: WheelLimitConfig) -> bool:
    """凸組合的手臂分量是否仍在 arm_rate_max 內。

    若 u_prev 與 u_req 的手臂分量都合規，本式必為真（policy §2）；
    留作執行期的斷言與離線測試項。
    """
    return bool(np.max(np.abs(np.asarray(u, float)[3:]))
                <= cfg.arm_rate_max + 1e-9)
