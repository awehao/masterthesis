"""W-GMPC 核心的**增廣狀態版本**：把手臂位置設定點納入預測狀態。

與 `wgmpc_core` 的關係
----------------------
`wgmpc_core` **原檔不動**，仍是 free4 的比較基準。本檔只改一件事：
手臂的執行模型。其餘（N、dt、任務誤差定義、權重、速度／加速度框、
輪級限制、SQP 政策、QP 設定、停止保護）全部沿用，且直接 import
`wgmpc_core` 的實作，不複製。

為什麼要改
----------
`wgmpc_core.step` 假設 `q_{k+1} = q_k + dt·B(θ)u`，即**命令速度在 dt 內
完全達成**。實機不是這樣：執行端把手臂命令速率積分成**位置設定點**
（`wb_cmd_chain_e2.py::_apply`，每物理步一次），再由 Isaac 的
articulation 位置控制器去追。於是多了一個核心狀態向量裡沒有的隱藏狀態。

free4 的離線辨識（`wgmpc_arm_setpoint_ident.py`，前半估、後半驗證）：

* 每物理步 α ≈ 0.09501（六軸一致），等效一階時間常數 ≈ 0.100 s
* 50 ms 預測最差關節 RMSE：理想速度積分 27.9 mrad、設定點追蹤 0.44 mrad

增廣狀態
--------
    z = [q (9) ; s (6)] ∈ R^15      s = 手臂位置設定點
    u = (v_x^B, v_y^B, ω, q̇1..q̇6) ∈ R^9        **控制維度不變**

`s` 是**致動器內部狀態，不是新增六個致動自由度** ——
它只能由 `u` 的手臂分量驅動，沒有自己的輸入。

控制步長的正確組合
------------------
辨識出的 α 是**每物理步**（10 ms）的係數。控制步長是 50 ms，
**不能**把 α 直接當成 50 ms 的係數。把一步關係式

    x⁺ = x + α⊙(s − x) + b ,      s⁺ = s + u_a·dt_p

連乘 kp = dt_c/dt_p 次，得到**閉式且嚴格線性**的控制步映射

    x_next = P⊙x + Q⊙s + G⊙u_a + h
    s_next = s + u_a·dt_c

    β = 1 − α ,  P = β^kp ,  Q = 1 − β^kp
    G = α·dt_p·Σ_{j=0}^{kp−1} j·β^{kp−1−j} ,  h = b·(1 − β^kp)/α

以 α = 0.09501、kp = 5 計：P = 0.6070、Q = 0.3930、G = 0.008640。
注意 **G ≠ dt_c = 0.05**（相差 5.8 倍）；位移主要來自 Q⊙(s − x) 這一項，
這正是設定點必須進狀態的原因。

因為手臂區塊對 (x, s, u) **嚴格線性**，它在仿射模型裡**沒有線性化誤差**；
只有底盤區塊仍是在 nominal 處的線性化（與原核心相同）。

限位
----
執行端的硬失效（`_apply` 的 `_fail`）是檢查**設定點**是否超出關節限位。
因此本檔同時約束兩者：

* `setpoint_position` —— 預測設定點（對應執行端真正會失效的量）
* `measured_position` —— 預測實測關節角

兩者都保留 `cfg.joint_margin`；執行端本身不留餘量，故這是保守方向。

數值處理：等價正值列縮放
------------------------
`measured_position` 的複合增益 G ≈ 0.00864 遠小於其他區塊的係數，
欄正規化後整體 |係數| 跨度達 115.8×，OSQP 會在限位逼近時打到迭代上限。
本版對每列同乘有限正數 d_i：

    l_i ≤ A_i U ≤ h_i   ⟺   d_i l_i ≤ d_i A_i U ≤ d_i h_i

**精確算術下不改可行集合**，不等同刪除限制或放寬門檻。成本與限制內容不變。
候選解的閘門與最終殘差核對一律使用**未縮放的 A、lo、hi 與原單位**，
`r_tol` 不動，`accepted_status` 仍只接受 `solved`（超迭代解不接受）。
可用 `cfg.row_scaling = False` 關閉以做對照。

**兩組關節限位都保留。** 先前以「偏置累積 < 餘量」論證 `measured_position`
被 `setpoint_position` 蘊含是**錯的**：含非零偏置時
x⁺ = (1−α)x + αs + b 不再只是 x 與 s 的凸組合，穩態偏移是 b/α，
不是 b。joint2 的反例（α = 0.09504、b = 2.715e-4，故 b/α = 2.856e-3）：
取 x = s = U − 0.5 mrad、u = 0，五個物理步後 x = U + 0.623 mrad ——
**設定點界成立、實測界不成立**。

界線
----
α 與 b 由**單趟**（free4，n = 1）辨識，是**候選模型參數，未定版**。
預測改善不等於閉迴路會到達並保持 —— 那要另行安排同目標的物理對照。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .arm_limits import LITE6_SAFE
from .wgmpc_core import (NE, NQ, NU, WGMPCConfig, WGMPCResult, _diff_operator,
                         _solve_qp, affine_model, body_to_world, residuals,
                         task_error, task_error_and_jacobian)

NS = 6                    # 手臂設定點維度
NZ = NQ + NS              # 增廣狀態維度 = 15
ARM = slice(3, 9)         # z 與 u 裡的手臂區塊
SP = slice(NQ, NZ)        # z 裡的設定點區塊
BASE = slice(0, 3)


# ------------------------------------------------------- 手臂執行模型（候選）
@dataclass
class ArmSetpointModel:
    """每物理步的一階追蹤：x⁺ = x + α⊙(s − x) + b。

    `source` 與 `status` 一律隨模型走，**不讓候選參數在下游被當成定版**。
    """
    alpha: np.ndarray = field(
        default_factory=lambda: np.full(NS, 0.09501))
    bias: np.ndarray = field(default_factory=lambda: np.zeros(NS))
    phys_dt: float = 0.01
    source: str = ('free4 離線辨識（wgmpc_arm_setpoint_ident.py，'
                   '前半估／後半驗證）')
    status: str = '候選模型參數，未定版'

    def __post_init__(self):
        self.alpha = np.asarray(self.alpha, float).reshape(NS)
        self.bias = np.asarray(self.bias, float).reshape(NS)
        if not np.all((self.alpha > 0.0) & (self.alpha < 1.0)):
            raise ValueError(f'α 必須逐軸落在 (0, 1)：{self.alpha}')
        if not (self.phys_dt > 0.0):
            raise ValueError(f'phys_dt 必須為正：{self.phys_dt}')

    def compose(self, dt_ctrl: float):
        """把每物理步的係數組合成**控制步**的係數，回傳 (P, Q, G, h, kp)。

        `dt_ctrl` 必須是 `phys_dt` 的整數倍 —— 否則一步映射不是整數次
        連乘，**拒絕靜默近似**。
        """
        r = dt_ctrl / self.phys_dt
        kp = int(round(r))
        if kp < 1 or abs(r - kp) > 1e-9:
            raise ValueError(
                f'dt_ctrl={dt_ctrl} 不是 phys_dt={self.phys_dt} 的整數倍'
                f'（比值 {r}）⇒ 無法以整數次連乘組合，拒絕近似')
        b = 1.0 - self.alpha
        bk = b ** kp
        S = np.sum([j * b ** (kp - 1 - j) for j in range(kp)], axis=0)
        P = bk
        Q = 1.0 - bk
        G = self.alpha * self.phys_dt * S
        h = self.bias * (1.0 - bk) / self.alpha
        return P, Q, G, h, kp


@dataclass
class WGMPCResultSP(WGMPCResult):
    """沿用 WGMPCResult 的全部欄位，另記設定點預測與手臂模型出處。"""
    row_scale_range: tuple = (1.0, 1.0)   # 本輪列縮放係數的 (min, max)
    Z_pred: np.ndarray | None = None      # (N+1, 15) 增廣狀態 rollout
    S_pred: np.ndarray | None = None      # (N+1, 6) 設定點預測
    arm_model_status: str = ''


@dataclass
class WGMPCConfigSP(WGMPCConfig):
    """沿用 WGMPCConfig 的**全部**已核准數值，只加上手臂執行模型。"""
    arm_model: ArmSetpointModel = field(default_factory=ArmSetpointModel)
    # **等價正值列縮放**（Howard 2026-10-01 核准的唯一數值改善）。
    # 對第 i 列同乘有限正數 d_i：l_i ≤ A_i U ≤ h_i ⟺ d_i l_i ≤ d_i A_i U ≤ d_i h_i。
    # 精確算術下**不改可行集合**，不等同刪除限制或放寬門檻。
    # 殘差一律以**未縮放的原始限制、原單位**檢查（見 solve_sp）。
    row_scaling: bool = True

    def composed(self):
        return self.arm_model.compose(self.dt)


# -------------------------------------------------------------- 增廣動力學
def split(z: np.ndarray):
    z = np.asarray(z, float)
    if z.shape[-1] != NZ:
        raise ValueError(f'增廣狀態必須是 {NZ} 維，收到 {z.shape[-1]}')
    return z[..., :NQ], z[..., NQ:]


def make_z(q, s) -> np.ndarray:
    """由 q（9）與**執行端的真實設定點** s（6）組增廣狀態。

    s **不得**預設為實測關節角：那是 `setpoint_init` 當下才成立的巧合，
    之後兩者會差一個與命令速率成比例的量（free4 實測相關 ~0.8）。
    """
    q = np.asarray(q, float).reshape(NQ)
    s = np.asarray(s, float).reshape(NS)
    if not np.isfinite(s).all():
        raise ValueError('手臂設定點含非有限值 ⇒ 拒絕求解'
                         '（不以實測關節角代替）')
    return np.concatenate([q, s])


def step_sp(z: np.ndarray, u: np.ndarray, cfg: WGMPCConfigSP) -> np.ndarray:
    """f(z, u)。底盤沿用原核心的顯式 Euler；手臂用**閉式連乘**，無近似。"""
    q, s = split(z)
    u = np.asarray(u, float)
    P, Q, G, h, _ = cfg.composed()
    out = np.empty(NZ)
    # 底盤：與 wgmpc_core.step 完全相同（只取前三維）
    out[BASE] = q[BASE] + cfg.dt * (body_to_world(float(q[2]))
                                    @ u)[BASE]
    out[ARM] = P * q[ARM] + Q * s + G * u[ARM] + h
    out[SP] = s + cfg.dt * u[ARM]
    return out


def shape_near_target(u0, J, theta, e_p, e_r, u_prev, cfg,
                      gamma=0.0, deadband_m=0.0, deadband_rad=0.0,
                      tol_p=0.0, tol_r=0.0):
    """**近目標輸出整形**：讓單週期命令的位移不超過當下殘差。

    為什麼需要（rec10 實測）：每週期命令造成的 TCP 位移對當下誤差的比值
    在遠離目標時是 0.1–0.4×，**但 5–20 mm 時 1.6×、2–5 mm 時 4.3×、
    <2 mm 時 8.2×** —— 命令要求的位移遠大於要修正的誤差，必然過衝，
    下一週期反向修正，形成極限環。近目標的命令翻號率
    wz 18.1%、手臂 j2 31.6%（遠離目標皆 0%）。

    規則：預測本週期的 TCP 位移 d_p = |J_p·B(θ)·u|·dt 與姿態變化
    d_r = |J_ω·B(θ)·u|·dt；若超過 γ·**max(殘差, 容差)** 就把整個 u
    等比例縮小到剛好不超過。**γ = 1 表示「一個週期最多把誤差走完」。**

    **為什麼用 max(殘差, 容差) 而不是殘差本身**：姿態目標就是起始姿態，
    所以 e_r 在起點恰好是 **0** —— 直接用 γ·e_r 會把整個命令縮到零、
    機器人完全停住（實測踩過）。進了容差以內本來就不需要比容差更精確，
    所以下限取容差。`tol_p`／`tol_r` 給 0 時退回用殘差本身。

    這條規則**自己會在遠處失效**：遠離目標時 e 大、比值本來就 < 1，
    不會縮放（實測 0.1–0.4×）。所以不需要另設「近目標區」的切換門檻。

    `deadband_*`：殘差進入內圈時直接歸零（預設 0 = 不啟用）。
    縮放已經讓 u 隨 e 平滑趨零，硬歸零只在需要完全停止輸出時才用。

    **加速度框**：縮放後仍以 u_prev 為基準夾進 ±a_max·dt，
    否則由大命令驟降到小命令會違反加速度保證。

    回傳 (u_out, scale, reason)。**不改權重、視界或任何約束集合** ——
    這是求解之後的輸出整形。
    """
    u = np.asarray(u0, float).copy()
    if gamma <= 0.0 and deadband_m <= 0.0 and deadband_rad <= 0.0:
        return u, 1.0, 'off'
    B = body_to_world(float(theta))
    qd = B @ u
    dt = cfg.dt
    d_p = float(np.linalg.norm(np.asarray(J)[:3, :] @ qd)) * dt
    d_r = float(np.linalg.norm(np.asarray(J)[3:, :] @ qd)) * dt
    reason = 'none'
    scale = 1.0
    if gamma > 0.0:
        lim = 1.0
        b_p = max(float(e_p), float(tol_p))
        b_r = max(float(e_r), float(tol_r))
        if d_p > 1e-12 and gamma * b_p < d_p:
            lim = min(lim, gamma * b_p / d_p)
        if d_r > 1e-12 and gamma * b_r < d_r:
            lim = min(lim, gamma * b_r / d_r)
        if lim < 1.0:
            scale = lim
            u = u * lim
            reason = 'scaled'
    if (deadband_m > 0.0 and e_p <= deadband_m
            and deadband_rad > 0.0 and e_r <= deadband_rad):
        u = np.zeros_like(u)
        scale = 0.0
        reason = 'deadband'
    # **加速度框**：相對 u_prev 夾住，維持原有的加速度保證
    up = np.asarray(u_prev, float)
    adt = cfg.amax() * dt
    u = up + np.clip(u - up, -adt, adt)
    return u, scale, reason


def plant_phys_step(q: np.ndarray, s: np.ndarray, u: np.ndarray,
                    cfg: WGMPCConfigSP, n_phys: int = 1):
    """受控對象的**逐物理步**推進，回傳 (q, s)。

    與 `step_sp` 的差別：`step_sp` 是一個**控制步**的閉式映射（供 MPC 預測），
    這裡是逐物理步，供**延遲補償**把量測狀態推到命令真正生效的時刻。
    兩者用同一組 α、b、dt_p，所以不會分歧。

    延遲補償的語意：時刻 τ 作用的命令是 τ − D 時發出的那一筆；
    要把狀態由 t 推到 t + D，就用 [t − D, t) 這段**已發布**的命令
    —— 全部已知，不需預測未來輸入。
    """
    al, b = cfg.arm_model.alpha, cfg.arm_model.bias
    dtp = cfg.arm_model.phys_dt
    q = np.asarray(q, float).copy()
    s = np.asarray(s, float).copy()
    u = np.asarray(u, float)
    for _ in range(int(n_phys)):
        q[:3] = q[:3] + dtp * (body_to_world(float(q[2])) @ u)[:3]
        x = q[ARM]
        q[ARM] = x + al * (s - x) + b
        s = s + u[ARM] * dtp
    return q, s


def rollout_sp(z0: np.ndarray, U: np.ndarray,
               cfg: WGMPCConfigSP) -> np.ndarray:
    U = np.atleast_2d(np.asarray(U, float))
    Z = np.zeros((len(U) + 1, NZ))
    Z[0] = np.asarray(z0, float)
    for k, u in enumerate(U):
        Z[k + 1] = step_sp(Z[k], u, cfg)
    return Z


def affine_model_sp(z_nom: np.ndarray, u_nom: np.ndarray,
                    cfg: WGMPCConfigSP):
    """回傳 (A, B, c)，A ∈ R^{15×15}、B ∈ R^{15×9}、c ∈ R^15。

    手臂與設定點區塊**嚴格線性** ⇒ 那些列沒有線性化誤差；
    只有底盤三列是在 (z_nom, u_nom) 處的線性化，與原核心一致。
    """
    q, s = split(z_nom)
    P, Q, G, h, _ = cfg.composed()
    Ab, Bb, cb = affine_model(q, u_nom, cfg.dt)      # 9×9 / 9×9 / 9
    A = np.zeros((NZ, NZ))
    B = np.zeros((NZ, NU))
    c = np.zeros(NZ)
    # 底盤三列：沿用原核心（含 ∂/∂θ 的那一項）
    A[BASE, :NQ] = Ab[BASE, :]
    B[BASE, :] = Bb[BASE, :]
    c[BASE] = cb[BASE]
    # 手臂實測六列
    for i in range(NS):
        A[3 + i, 3 + i] = P[i]
        A[3 + i, NQ + i] = Q[i]
        B[3 + i, 3 + i] = G[i]
        c[3 + i] = h[i]
    # 設定點六列
    for i in range(NS):
        A[NQ + i, NQ + i] = 1.0
        B[NQ + i, 3 + i] = cfg.dt
    return A, B, c


def build_prediction_sp(A_list, B_list, c_list):
    """z_stack = Φ z_0 + Γ ζ + γ，與 `wgmpc_core.build_prediction` 同形，
    只是狀態維度為 NZ。原函式把 NQ 寫死，故不能重用。"""
    N = len(A_list)
    Phi = np.zeros((N * NZ, NZ))
    Gam = np.zeros((N * NZ, N * NU))
    gam = np.zeros(N * NZ)
    for k in range(N):
        r = slice(k * NZ, (k + 1) * NZ)
        if k == 0:
            Phi[r] = A_list[0]
            Gam[r, 0:NU] = B_list[0]
            gam[r] = c_list[0]
        else:
            rp = slice((k - 1) * NZ, k * NZ)
            Ak = A_list[k]
            Phi[r] = Ak @ Phi[rp]
            Gam[r, 0:k * NU] = Ak @ Gam[rp, 0:k * NU]
            Gam[r, k * NU:(k + 1) * NU] = B_list[k]
            gam[r] = Ak @ gam[rp] + c_list[k]
    return Phi, Gam, gam


# ------------------------------------------------------------ 任務誤差與成本
def task_error_sp(K, z, T_des, tcp, pi_band=1e-5):
    """任務誤差只取決於 q（設定點不在 TCP 位姿裡）。"""
    q, _ = split(z)
    return task_error(K, q, T_des, tcp, pi_band)


def task_error_and_jacobian_sp(K, z, T_des, tcp, pi_band=1e-5):
    """H_aug = [H | 0_{6×6}] —— 設定點不直接進任務誤差。"""
    q, _ = split(z)
    e, H = task_error_and_jacobian(K, q, T_des, tcp, pi_band)
    Haug = np.zeros((NE, NZ))
    Haug[:, :NQ] = H
    return e, Haug


def nonlinear_cost_sp(K, z0, U, u_prev, T_des, cfg: WGMPCConfigSP) -> float:
    """與 `wgmpc_core.nonlinear_cost` **同一個成本函式**，只換 rollout。"""
    Z = rollout_sp(z0, U, cfg)
    Qw, Rw, Sw = cfg.Q(), cfg.R(), cfg.S()
    J = 0.0
    for k in range(1, len(Z)):
        e = task_error_sp(K, Z[k], T_des, cfg.tcp, cfg.pi_band)
        W = Qw * cfg.Qf_scale if k == len(Z) - 1 else Qw
        J += float(e @ W @ e)
    up = np.asarray(u_prev, float)
    for k, u in enumerate(U):
        J += float(u @ Rw @ u)
        d = u - (up if k == 0 else U[k - 1])
        J += float(d @ Sw @ d)
    return J


# ---------------------------------------------------------------- 約束組裝
def build_constraints_sp(z0, U_nom, u_prev, cfg: WGMPCConfigSP, delta,
                         Phi, Gam, gam):
    """回傳 (A, lo, hi, blocks)。

    與 `wgmpc_core.build_constraints` 的差別只有關節限位那一段：
    原本是 `q_arm0 + Σ dt·u`（理想模型的手臂位置），
    現在拆成**設定點**與**實測關節角**兩組，係數由 (Φ, Γ, γ) 取出。

    其餘（速度框、加速度框、輪級速度與加速度、信賴區域）**逐字沿用**
    原本的形式，因為它們只作用在 u 上。
    """
    N, dt = cfg.N, cfg.dt
    n = N * NU
    vmax, amax = cfg.vmax(), cfg.amax()
    rows, lo, hi, blocks = [], [], [], {}
    z0 = np.asarray(z0, float)

    def add(name, Arows, l, u):
        i0 = len(rows)
        rows.extend(Arows)
        lo.extend(l)
        hi.extend(u)
        blocks[name] = (i0, len(rows))

    # 1 速度框
    I = np.eye(n)
    add('velocity', list(I), list(np.tile(-vmax, N)), list(np.tile(vmax, N)))
    # 2 加速度框
    D = _diff_operator(N)
    f = np.zeros(n)
    f[0:NU] = -np.asarray(u_prev, float)
    bnd = np.tile(amax * dt, N)
    add('acceleration', list(D), list(-bnd - f), list(bnd - f))
    # 3 關節限位：**設定點**與**實測**各一組（都嚴格線性）
    lo_j, hi_j = LITE6_SAFE.lower, LITE6_SAFE.upper
    m = cfg.joint_margin
    const = Phi @ z0 + gam               # 預測狀態堆疊的常數部分
    for name, base_idx in (('setpoint_position', NQ), ('measured_position', 3)):
        Aj, lj, hj = [], [], []
        for k in range(1, N + 1):
            for i in range(NS):
                r = (k - 1) * NZ + base_idx + i
                Aj.append(Gam[r].copy())
                lj.append(lo_j[i] + m - const[r])
                hj.append(hi_j[i] - m - const[r])
        add(name, Aj, lj, hj)
    # 4 輪級速度與加速度
    W = cfg.wheel_matrix()
    wlim = cfg.wheel_radius * cfg.wheel_w_max
    alim = cfg.wheel_radius * cfg.wheel_a_max * dt
    Aw, lw, hw = [], [], []
    for k in range(N):
        for row in W:
            r = np.zeros(n)
            r[k * NU:k * NU + 3] = row
            Aw.append(r); lw.append(-wlim); hw.append(wlim)
    add('wheel_speed', Aw, lw, hw)
    Aa, la, ha = [], [], []
    for k in range(N):
        for row in W:
            r = np.zeros(n)
            r[k * NU:k * NU + 3] = row
            off = 0.0
            if k > 0:
                r[(k - 1) * NU:(k - 1) * NU + 3] = -row
            else:
                off = float(row @ np.asarray(u_prev, float)[BASE])
            Aa.append(r); la.append(-alim + off); ha.append(alim + off)
    add('wheel_accel', Aa, la, ha)
    # 5 信賴區域
    tr = np.tile(delta * vmax, N)
    z_nom = np.asarray(U_nom, float).reshape(-1)
    add('trust_region', list(I), list(z_nom - tr), list(z_nom + tr))
    return (np.asarray(rows, float), np.asarray(lo, float),
            np.asarray(hi, float), blocks)


def row_scale(A: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """逐列等化：d_i = 1 / ‖A_i‖_∞。回傳**有限正值**向量。

    輸入是**已做欄正規化**（×Σ）的矩陣，所以 d 只處理列間的係數跨度。
    全零列（理論上不應出現）給 d_i = 1，不讓 1/0 進去。
    """
    m = np.abs(np.asarray(A, float)).max(axis=1)
    d = np.where(m > floor, 1.0 / np.maximum(m, floor), 1.0)
    if not np.isfinite(d).all() or np.any(d <= 0.0):
        raise ValueError('列縮放係數必須為有限正值')
    return d


# ---------------------------------------------------------------- QP 與 SQP
def solve_sp(K, z0, u_prev, T_des, cfg: WGMPCConfigSP,
             U_warm=None) -> WGMPCResultSP:
    """一個控制週期的多步求解（增廣狀態）。

    **SQP 政策逐字沿用** `wgmpc_core.solve`：暖啟動末步以最大允許減速朝零收、
    nominal 可行性預檢、逐候選閘門（有限值＋硬約束殘差）、可行性恢復步、
    五種停止狀態、任一次 QP 失敗即 no_valid_solution、
    以及對**實際回傳序列**的最終殘差核對。此處只換動力學與關節限位。
    """
    t_all = time.monotonic()
    N = cfg.N
    z0 = np.asarray(z0, float)
    if z0.shape[-1] != NZ:
        raise ValueError(f'solve_sp 需要 {NZ} 維增廣狀態（見 make_z）')
    u_prev = np.asarray(u_prev, float)
    res = WGMPCResultSP(ok=False)
    res.arm_model_status = cfg.arm_model.status
    tm = {'H': 0.0, 'ABc': 0.0, 'qp_build': 0.0, 'qp_prep': 0.0,
          'qp_setup': 0.0, 'qp_iter': 0.0, 'rollout': 0.0,
          'cost': 0.0, 'post': 0.0}

    if U_warm is None:
        U_nom = np.zeros((N, NU))
    else:
        _W = np.asarray(U_warm, float)
        if len(_W) != N:
            U_nom = np.zeros((N, NU))
        else:
            _adt = cfg.amax() * cfg.dt
            _last = _W[-1] + np.clip(-_W[-1], -_adt, _adt)
            U_nom = np.vstack([_W[1:], _last[None, :]])
    U_nom = np.clip(U_nom, -cfg.vmax(), cfg.vmax())

    Qw, Rw, Sw = cfg.Q(), cfg.R(), cfg.S()
    Qbar = np.zeros((N * NE, N * NE))
    for k in range(N):
        Qbar[k * NE:(k + 1) * NE, k * NE:(k + 1) * NE] = (
            Qw * cfg.Qf_scale if k == N - 1 else Qw)
    Rbar = np.kron(np.eye(N), Rw)
    Sbar = np.kron(np.eye(N), Sw)
    D = _diff_operator(N)
    fvec = np.zeros(N * NU)
    fvec[0:NU] = -u_prev

    def _lin(U):
        """nominal rollout ＋ 逐步線性化 ＋ 凝縮。回傳 (Z, Phi, Gam, gam)。"""
        Z = rollout_sp(z0, U, cfg)
        A_l, B_l, c_l = [], [], []
        for k in range(N):
            a, b, c = affine_model_sp(Z[k], U[k], cfg)
            A_l.append(a); B_l.append(b); c_l.append(c)
        return (Z,) + build_prediction_sp(A_l, B_l, c_l)

    t0 = time.monotonic()
    J_cur = nonlinear_cost_sp(K, z0, U_nom, u_prev, T_des, cfg)
    tm['cost'] += (time.monotonic() - t0) * 1e3
    res.J_nl.append(J_cur)

    # nominal 可行性預檢（排除信賴區域）
    _Z0, _P0, _G0, _g0 = _lin(U_nom)
    _Am, _lo, _hi, _bl = build_constraints_sp(z0, U_nom, u_prev, cfg,
                                              cfg.delta_max, _P0, _G0, _g0)
    _b2 = {k: v for k, v in _bl.items() if k != 'trust_region'}
    _a = min(v[0] for v in _b2.values())
    _b = max(v[1] for v in _b2.values())
    _sub = {k: (v[0] - _a, v[1] - _a) for k, v in _b2.items()}
    _mx, _ = residuals(_Am[_a:_b], _lo[_a:_b], _hi[_a:_b],
                       U_nom.reshape(-1), _sub)
    nominal_feasible = bool(_mx <= cfg.r_tol)
    res.nominal_infeasible = not nominal_feasible
    res.nominal_residual = float(_mx)

    delta = cfg.delta_0
    U_best = None
    stop = 'iter_limit'
    conv = False

    for it in range(cfg.n_sqp):
        res.n_sqp_used = it + 1
        res.delta.append(delta)
        t0 = time.monotonic()
        Zn, Phi, Gam, gam = _lin(U_nom)
        tm['rollout'] += (time.monotonic() - t0) * 1e3
        t0 = time.monotonic()
        _eh = [task_error_and_jacobian_sp(K, Zn[k], T_des, cfg.tcp,
                                          cfg.pi_band)
               for k in range(1, N + 1)]
        Hs = [x[1] for x in _eh]
        e_nom = np.concatenate([x[0] for x in _eh])
        tm['H'] += (time.monotonic() - t0) * 1e3

        t0 = time.monotonic()
        Hblk = np.zeros((N * NE, N * NZ))
        for k in range(N):
            Hblk[k * NE:(k + 1) * NE, k * NZ:(k + 1) * NZ] = Hs[k]
        z_nom_stack = Zn[1:].reshape(-1)
        G = Hblk @ Gam
        d = e_nom + Hblk @ (Phi @ z0 + gam - z_nom_stack)
        P = 2.0 * (G.T @ Qbar @ G + Rbar + D.T @ Sbar @ D)
        qv = 2.0 * (G.T @ Qbar @ d + D.T @ Sbar @ fvec)
        _d_eff = cfg.delta_max * 1e3 if not nominal_feasible else delta
        Am, lo, hi, blocks = build_constraints_sp(z0, U_nom, u_prev, cfg,
                                                 _d_eff, Phi, Gam, gam)
        tm['qp_build'] += (time.monotonic() - t0) * 1e3

        sig = np.tile(cfg.vmax(), N)
        # **欄正規化**（原核心既有）＋**列等化**（本版新增，等價變換）。
        # 送進 OSQP 的是縮放後的問題；下面的閘門與最終核對一律回到
        # **未縮放的 Am/lo/hi、原單位**，所以縮放不可能放過違反約束的解。
        _Ac = Am * sig[None, :]
        if cfg.row_scaling:
            _d = row_scale(_Ac)
            res.row_scale_range = (float(_d.min()), float(_d.max()))
        else:
            _d = np.ones(_Ac.shape[0])
        zh, st, nit, _tq = _solve_qp(P * np.outer(sig, sig), qv * sig,
                                     _Ac * _d[:, None], lo * _d, hi * _d, cfg)
        tm['qp_prep'] += _tq['prep']
        tm['qp_setup'] += _tq['setup']
        tm['qp_iter'] += _tq['iter']
        z = None if zh is None else zh * sig
        res.qp_status.append(st)
        res.qp_iters.append(nit)
        if z is None:
            stop = 'qp_failed'
            break

        U_cand = z.reshape(N, NU)
        # 逐候選閘門：**先過閘才進成本比較**
        _finite = bool(np.isfinite(U_cand).all())
        _mxc, _byc = (residuals(_Am[_a:_b], _lo[_a:_b], _hi[_a:_b],
                                U_cand.reshape(-1), _sub)
                      if _finite else (float('inf'), {}))
        if not (_finite and _mxc <= cfg.r_tol):
            res.n_rejected += 1
            res.n_gate_rejected += 1
            res.last_gate_reject = ('non_finite' if not _finite else
                                    'residual:' + ','.join(
                                        k for k, v in _byc.items()
                                        if v > cfg.r_tol))
            delta *= cfg.gamma_dn
            if delta < cfg.delta_min_conv:
                stop = 'trust_region_exhausted'
                break
            stop = 'no_progress'
            continue

        step_inf = float(np.max(np.abs(U_cand - U_nom) / cfg.vmax()))
        t0 = time.monotonic()
        J_cand = nonlinear_cost_sp(K, z0, U_cand, u_prev, T_des, cfg)
        tm['cost'] += (time.monotonic() - t0) * 1e3
        dJ = J_cand - J_cur
        res.J_nl.append(J_cand)

        _accept = dJ <= cfg.tol_accept
        _was_recovery = False
        if not nominal_feasible:
            _accept = True
            _was_recovery = True
            nominal_feasible = True
            res.first_feasible_unconditional = True
            res.n_recovery_steps += 1
        if _accept:
            res.n_accepted += 1
            U_best = U_cand.copy()
            rel = abs(dJ) / max(1.0, abs(J_cur))
            U_nom, J_cur = U_cand, J_cand
            delta = min(cfg.delta_max, delta * cfg.gamma_up)
            if _was_recovery:
                stop = 'feasibility_recovered'
            elif delta >= cfg.delta_min_conv:
                if step_inf <= cfg.tol_step:
                    stop, conv = 'converged_step', True
                    break
                if rel <= cfg.tol_cost:
                    stop, conv = 'converged_cost', True
                    break
        else:
            res.n_rejected += 1
            delta *= cfg.gamma_dn
            if delta < cfg.delta_min_conv:
                stop = 'trust_region_exhausted'
                break
            stop = 'no_progress'

    res.sqp_stop_reason = stop
    res.sqp_converged = bool(conv)

    def _finish():
        res.timing_ms = {k: round(v, 4) for k, v in tm.items()}
        res.timing_ms['total'] = round((time.monotonic() - t_all) * 1e3, 4)
        return res

    if stop == 'qp_failed' or U_best is None:
        res.reason = ('qp_failed' if stop == 'qp_failed'
                      else 'no_accepted_candidate')
        return _finish()

    # 最終檢查：針對**實際要返回的序列**及其非線性 rollout
    t0 = time.monotonic()
    Zf, Phf, Gmf, gmf = _lin(U_best)
    tm['rollout'] += (time.monotonic() - t0) * 1e3
    Am, lo, hi, blocks = build_constraints_sp(z0, U_best, u_prev, cfg,
                                              cfg.delta_max, Phf, Gmf, gmf)
    mx, by = residuals(Am, lo, hi, U_best.reshape(-1), blocks)
    res.max_residual, res.residual_by_block = mx, by
    res.violated_blocks = [n for n, v in by.items() if v > cfg.r_tol]
    if res.violated_blocks:
        res.reason = 'residual_check_failed:' + ','.join(res.violated_blocks)
        return _finish()

    _t_post = time.monotonic()
    res.ok = True
    res.u0 = U_best[0].copy()
    res.U = U_best
    # Q_pred 仍是 **9 維 q 的預測**（與原核心同形，下游不必改讀法）
    res.Q_pred = Zf[:, :NQ].copy()
    res.Z_pred = Zf
    res.S_pred = Zf[:, NQ:].copy()
    res.E_pred = np.array([task_error_sp(K, Zf[k], T_des, cfg.tcp,
                                         cfg.pi_band)
                           for k in range(len(Zf))])
    res.J_nl.append(J_cur)
    # 線性化誤差指標（**只記錄，不設門檻**）。手臂與設定點區塊嚴格線性，
    # 故這裡剩下的差距應只來自底盤三維 —— 分開報，便於核對這個預期。
    z_aff = (Phf @ z0 + Gmf @ U_best.reshape(-1) + gmf).reshape(N, NZ)
    res.lin_error = {
        'max_abs_state': float(np.abs(z_aff - Zf[1:]).max()),
        'max_abs_base_xy': float(np.abs(z_aff[:, :2] - Zf[1:, :2]).max()),
        'max_abs_theta': float(np.abs(z_aff[:, 2] - Zf[1:, 2]).max()),
        'max_abs_arm': float(np.abs(z_aff[:, 3:NQ] - Zf[1:, 3:NQ]).max()),
        'max_abs_setpoint': float(np.abs(z_aff[:, NQ:] - Zf[1:, NQ:]).max()),
        'note': '仿射預測 vs 非線性 rollout 的差距。手臂與設定點區塊'
                '**嚴格線性** ⇒ 該兩項應為數值誤差量級；底盤三維仍是線性化。',
    }
    res.reason = ''
    tm['post'] += (time.monotonic() - _t_post) * 1e3
    return _finish()
