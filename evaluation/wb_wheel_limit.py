"""完整 9 維命令的輪級限制（執行版本 E2）。

策略見 `evaluation/results/specs/wb_wheel_limit_policy_v1.md`。
本檔是**純函式模組**，不含 ROS、不含模擬器，可完全離線測試。

核心：λ 由**底盤**分量的輪速／輪加速度約束決定，但**施加於完整 9 維增量**——

    u_out = u_prev + λ · (u_req − u_prev)

只縮底盤會改掉底盤與手臂的相對比例，上游求解出的耦合方向就沒了。
同一個 λ 讓 u_out 落在 9 維空間中 u_prev→u_req 的線段上。

**不得由此宣稱上游全身安全性不變**：u_out 不是求解器的輸出，
上游的距離約束是對 u_req 算的。λ < 1 時 `modified` 為 True，
呼叫端必須記錄。

E1 保持不變；本檔是另立的版本，不改動 E1 的檔案。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

POLICY = 'evaluation/results/specs/wb_wheel_limit_policy_v1.md'
VERSION = 'wb_wheel_limit/1'

NORMAL = 'normal'
STOP_UNVERIFIED = 'stop_unverified'     # 不在輪級加速度保證之內


def wheel_matrix(L: float) -> np.ndarray:
    """r·ω = W·u_base。列序與 gmpc 的耦合約束一致。"""
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

    def as_row(self, u_req, u_prev, dt):
        """policy_v1 §5 的記錄欄位。"""
        f = lambda v: [round(float(x), 6) for x in np.asarray(v, float)]  # noqa
        return {'u_req': f(u_req), 'u_prev': f(u_prev),
                'u_out': None if self.u_out is None else f(self.u_out),
                'lam': None if self.lam is None else round(float(self.lam), 6),
                'modified': bool(self.modified), 'reason': self.reason,
                'dt': round(float(dt), 6), 'mode': self.mode,
                'wheel_speed_max': round(float(self.wheel_speed_max), 6),
                'wheel_accel_max': round(float(self.wheel_accel_max), 6),
                'binding': list(self.binding)}


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


def stop_target(u_prev) -> np.ndarray:
    """逾時／失效時的目標：底盤三分量 0、手臂速率 0。

    **目標為零不等於輸出為零** —— 仍要經過 limit9 才知道這一步能走多少。
    手臂速率為 0 使設定點停止積分，與 E1 的凍結行為一致。
    """
    return np.zeros(9)


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
