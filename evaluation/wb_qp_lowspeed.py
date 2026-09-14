"""**另立配置**：把執行界限放進既有 QP 的約束集（低速自由空間）。

不新寫控制器。求解目標函式、權重、DLS 線性化點全部沿用
`wholebody_pregrasp.WholeBody.solve` 的 `--solver qp` 路徑；
本檔只提供**約束集的組裝方式**，並把 E2 的執行界限放進去。

**不冒稱 B 基線。** B 凍結於 `baseline_B_frozen_20260909.md`
（`--solver qp`、OSQP eps 1e-6、μ=0.03、下游濾波器保留）；
本檔是**不同的速度框設定**，用於低速自由空間測試。

------------------------------------------------------------------
沒有障礙物列，**不等於**沒有約束
------------------------------------------------------------------

`wholebody_pregrasp._constraints` 在 `pts` 為空時拋出
「約束集合為空」。那個保護**不移除**：本檔另外提供一條
**明確確認空場景**才能走的路徑，仍然組裝

* `_joint_limit_rows` —— 關節位置限位
* `_box_rows` —— 速度框與**既有加速度框**

只有障礙物列可以為零。**失敗不退回無約束答案**：
OSQP 未收斂即拋出，由呼叫端停止。

------------------------------------------------------------------
空場景的確認：**不看列，看設定**
------------------------------------------------------------------

距離節點在「場景確實沒有障礙物」與「該連桿的 TF 取不到」兩種情況下
產生**完全相同**的列（`STATUS_NODATA`、位置全零）。
因此**不得**以「列全是 NODATA」推論空場景。
本檔要求呼叫端提供兩項獨立事實：

1. `obstacles_configured == 0` —— 距離節點的 `obstacles` 參數為空
2. `tf_ok_links == n_links` —— 每個連桿的 TF 都取得到

兩項皆成立才允許障礙物列為零；否則拋出，由呼叫端停止。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ammr_wholebody_mpc.wholebody_safety_filter import (
    SafetyConfig, _box_rows, _joint_limit_rows)

VERSION = 'wb_qp_lowspeed/1'
NOT_B_BASELINE = ('本配置不是 B 基線。B 凍結於 baseline_B_frozen_20260909.md；'
                  '本檔只是把 E2 執行界限放進約束集的低速自由空間配置。')

# E2 執行端的界限（不得為了讓求解通過而放寬）
E2_LIN_NORM_MAX = 0.05        # hypot(vx, vy) 上界
E2_ANG_MAX = 0.2
E2_ARM_RATE_MAX = 1.0

# 逐軸線速度框是滿足**模長**界限的保守作法：
#   |vx|, |vy| <= b  =>  hypot <= b*sqrt(2)，故 b = 0.05/sqrt(2)
LIN_AXIS_MAX = E2_LIN_NORM_MAX / np.sqrt(2.0)

# **求解器容差餘裕**：OSQP 以 eps_abs = eps_rel = 1e-6 收斂，解會落在
# 約束邊界上並可能超出該容差量。E2 的檢查是**嚴格大於**即拒收
# （`abs(r) > arm_rate_max`），所以求解側的框必須**嚴格內縮**，
# 否則每次都會在邊界被拒。
#
# 實測：不留餘裕時 QP 解出 |dq5| 恰好落在 1.0 邊界，E2 整筆失效。
#
# 取 1e-4（OSQP 容差的 100 倍，佔界限的 0.01 %）。
# **這是吸收求解器容差，不是放寬任何執行界限** ——
# 執行端的 0.05／0.2／1.0 一個都沒動，E2 仍保留最終檢查。
SOLVE_MARGIN = 1.0e-4


@dataclass(frozen=True)
class SceneFacts:
    """空場景的確認事實。**由呼叫端查核後提供，不是從列推論的。**"""
    obstacles_configured: int
    tf_ok_links: int
    n_links: int
    rows_total: int
    rows_status_ok: int

    @property
    def empty_scene_confirmed(self) -> bool:
        return (self.obstacles_configured == 0
                and self.n_links > 0
                and self.tf_ok_links == self.n_links)

    def why_not(self) -> str:
        if self.obstacles_configured != 0:
            return (f'距離節點設定了 {self.obstacles_configured} 個障礙物，'
                    f'不是空場景')
        if self.n_links <= 0:
            return '連桿名單為空'
        if self.tf_ok_links != self.n_links:
            return (f'只有 {self.tf_ok_links}/{self.n_links} 個連桿取得到 TF '
                    f'—— 缺 TF 與空場景在列的編碼上相同，**不得當成自由空間**')
        return ''


def lowspeed_cfg(base_cfg: SafetyConfig | None = None) -> SafetyConfig:
    """把 E2 的執行界限寫進安全設定的速度框。

    **下游安全層必須用同一份設定**，否則求解時滿足的界限會在下游被放寬回去。
    加速度框與 jerk 設定**沿用既有值**，不在此更動。
    """
    cfg = base_cfg or SafetyConfig()
    vm = np.array(cfg.vmax, dtype=float).copy()
    vm[0] = vm[1] = LIN_AXIS_MAX - SOLVE_MARGIN
    vm[2] = E2_ANG_MAX - SOLVE_MARGIN
    vm[3:] = np.minimum(vm[3:], E2_ARM_RATE_MAX - SOLVE_MARGIN)
    cfg.vmax = vm
    return cfg


def constraints_lowspeed(K, q, v_lin, pts, cfg, facts: SceneFacts,
                         v_prev=None, dt=None):
    """組裝低速配置的約束集。回傳 (A, b, n_barrier_rows, info)。

    `pts` 允許為空，**但只有在 facts 確認空場景時**；
    關節限位、速度框與加速度框**一律組裝**。
    """
    if K is None:
        raise RuntimeError('QP: 沒有運動學模型')
    if not pts and not facts.empty_scene_confirmed:
        raise RuntimeError(f'QP: 沒有可用的障礙物列，且空場景未獲確認 —— '
                           f'{facts.why_not()}')
    from ammr_wholebody_mpc.wholebody_safety_filter import _rows_from_points
    n = len(K.dof_names)
    if pts:
        Ab, bb, cap, _ = _rows_from_points(K, q, pts, cfg, v_lin)
    else:
        Ab, bb, cap = [], [], float('inf')
    Aj, bj = _joint_limit_rows(K, q, cfg)
    Ax, bx = _box_rows(cfg, n, cap, v_prev, dt if dt is not None else cfg.dt)
    A = np.array(Ab + Aj + Ax)
    b = np.array(bb + bj + bx)
    info = {'n_barrier': len(Ab), 'n_joint_limit': len(Aj), 'n_box': len(Ax),
            'n_total': len(A), 'empty_scene_confirmed': facts.empty_scene_confirmed,
            'obstacles_configured': facts.obstacles_configured,
            'tf_ok_links': f'{facts.tf_ok_links}/{facts.n_links}',
            'rows_total': facts.rows_total,
            'rows_status_ok': facts.rows_status_ok}
    if len(A) == 0:
        raise RuntimeError('QP: 約束集合為空（連關節限位與速度框都沒有組出來）')
    return A, b, len(Ab), info


def check_e2_bounds(v9) -> tuple[bool, str]:
    """對照 E2 執行端的界限。**求解後仍要檢查**，不以「已放進約束」代替。"""
    v = np.asarray(v9, float)
    lin = float(np.hypot(v[0], v[1]))
    if lin > E2_LIN_NORM_MAX + 1e-9:
        return False, f'線速度 {lin:.4f} > {E2_LIN_NORM_MAX}'
    if abs(v[2]) > E2_ANG_MAX + 1e-9:
        return False, f'角速度 {abs(v[2]):.4f} > {E2_ANG_MAX}'
    m = float(np.max(np.abs(v[3:])))
    if m > E2_ARM_RATE_MAX + 1e-9:
        return False, f'手臂速率 {m:.4f} > {E2_ARM_RATE_MAX}'
    return True, ''
