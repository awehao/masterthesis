"""W-GMPC WG1：全身多步控制的**純數值核心**（自由空間原型）。

**不訂閱、不發布、無背景執行緒、無跨執行緒共享可變狀態。**
輸入固定狀態快照，輸出資料結構。不 import rclpy 或任何 ROS 訊息。

規格：evaluation/results/specs/wgmpc_wg0_problem_spec.yaml（WG0 定版）
開發配置：evaluation/results/specs/wgmpc_wg1_dev_config.yaml（WG1-DEV-1）

狀態與控制
----------
    q = (x, y, θ, q1..q6) ∈ R^9        θ 為底盤 yaw
    u = (v_x^B, v_y^B, ω, q̇1..q̇6)     底盤三維為**本體座標**

    q̇ = B(θ) u,  B(θ) = blkdiag(Rz(θ)_{2×2}, 1, I_6)

**關鍵**：展開 URDF 的 base_x／base_y 是**世界軸**平移關節且在 base_theta
之前，所以 `jacobian()` 的**前兩欄**是世界平移。直接用 J 乘本體 u 是 frame 錯誤。

任務誤差（**位置差 ＋ SO(3) log 姿態誤差**，不是完整 SE(3) log）
----------------------------------------------------------------
    e(q) = [ p_des − p(q) ;  log(R(q)^T R_des)^∨ ]
           位置在**世界**、姿態在**本體**；Q 為區塊對角，不混座標。

    H = ∂e/∂q = [ −J_p ;  −J_l(e_r)^{-1} R^T J_ω ]

預測與線性化
------------
    非線性： q_{k+1} = f(q_k, u_k) = q_k + dt·B(θ_k) u_k
    仿射：   q_{k+1} ≈ A_k q_k + B_k u_k + c_k
    誤差：   e_k ≈ e(q^nom_k) + H_k (q_k − q^nom_k)

**不另外積分一套誤差遞推。**
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .arm_limits import LITE6_SAFE

NQ = 9            # 狀態維度
NU = 9            # 控制維度
NE = 6            # 任務誤差維度
ARM = slice(3, 9)  # 控制／狀態中的手臂區塊
BASE = slice(0, 3)


# --------------------------------------------------------------- SO(3) 工具
def skew(w) -> np.ndarray:
    w = np.asarray(w, float)
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def so3_log(R: np.ndarray, pi_band: float = 1e-5) -> np.ndarray:
    """SO(3) 主值對數，含近 π 的穩定分支。

    **分支處置**（WG0 branch_policy_v1）：
      θ 很小       → 一階式（避免 0/0）
      π − θ > band → 標準 θ/(2 sinθ) 式
      π − θ ≤ band → 由 (R+I)/2 的主特徵向量定軸，符號由 skew 部分挑

    **正好 π 時 ±軸表示同一旋轉**，本函式只能回其中一個主值；
    跨 π 時主值會換向。因此近 π 與跨 π **不要求中央差分一致**。
    """
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = math.acos(c)
    s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * s
    if math.pi - th > pi_band:
        return s * (th / (2.0 * math.sin(th)))
    A = (R + np.eye(3)) * 0.5
    w, v = np.linalg.eigh(A)
    ax = v[:, int(np.argmax(w))]
    n = np.linalg.norm(ax)
    ax = ax / n if n > 0 else np.array([1.0, 0.0, 0.0])
    if float(s @ ax) < 0.0:
        ax = -ax
    return ax * th


def so3_Jl_inv(phi: np.ndarray) -> np.ndarray:
    """SO(3) **左** Jacobian 的逆。

    用 **a = 1/θ² − cot(θ/2)/(2θ)** 的穩定形式。
    **不可**用 (1+cosθ)/(2θ sinθ) —— 那在 θ→π 是兩個趨零量相除，
    會抵銷掉有效位數。穩定形式下 ‖J_l^{-1}‖₂ 平滑趨近 π/2。
    """
    phi = np.asarray(phi, float)
    th = float(np.linalg.norm(phi))
    P = skew(phi)
    if th < 1e-8:
        return np.eye(3) - 0.5 * P + (P @ P) / 12.0
    a = 1.0 / (th * th) - math.cos(th * 0.5) / (2.0 * th * math.sin(th * 0.5))
    return np.eye(3) - 0.5 * P + a * (P @ P)


# ------------------------------------------------------------ 模型與導數
def body_to_world(theta: float) -> np.ndarray:
    """B(θ) = blkdiag(Rz(θ)_{2×2}, 1, I_6)，把**本體**控制映到 q̇。"""
    B = np.eye(NQ)
    c, s = math.cos(theta), math.sin(theta)
    B[0, 0], B[0, 1] = c, -s
    B[1, 0], B[1, 1] = s, c
    return B


def step(q: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """f(q, u) —— 顯式 Euler。單步局部誤差 O(dt²)。"""
    return np.asarray(q, float) + dt * (body_to_world(float(q[2]))
                                        @ np.asarray(u, float))


def rollout(q0: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray:
    """**非線性** rollout，回傳 (N+1, 9)。"""
    U = np.atleast_2d(np.asarray(U, float))
    Q = np.zeros((len(U) + 1, NQ))
    Q[0] = np.asarray(q0, float)
    for k, u in enumerate(U):
        Q[k + 1] = step(Q[k], u, dt)
    return Q


def affine_model(q_nom: np.ndarray, u_nom: np.ndarray, dt: float):
    """回傳 (A, B, c)，於 (q_nom, u_nom) 展開。

    A 的第 3 欄（θ）就是「**朝向改變會影響未來平移方向**」那一項，
    不得忽略 —— WG0 實測 |A[0:2, 2]| 達 0.0235。
    """
    th = float(q_nom[2])
    ct, st = math.cos(th), math.sin(th)
    A = np.eye(NQ)
    A[0, 2] = dt * (-st * u_nom[0] - ct * u_nom[1])
    A[1, 2] = dt * (ct * u_nom[0] - st * u_nom[1])
    B = dt * body_to_world(th)
    c = step(q_nom, u_nom, dt) - A @ q_nom - B @ u_nom
    return A, B, c


def task_error(K, q: np.ndarray, T_des: np.ndarray, tcp: str,
               pi_band: float = 1e-5) -> np.ndarray:
    """e = [p_des − p(q) ; log(R(q)^T R_des)^∨]；位置世界、姿態本體。"""
    T = K.fk(np.asarray(q, float), tcp)
    e_p = T_des[:3, 3] - T[:3, 3]
    e_r = so3_log(T[:3, :3].T @ T_des[:3, :3], pi_band)
    return np.concatenate([e_p, e_r])


def task_error_jacobian(K, q: np.ndarray, T_des: np.ndarray, tcp: str,
                        pi_band: float = 1e-5) -> np.ndarray:
    """H = ∂e/∂q ∈ R^{6×9}。

    位置區塊 −J_p（導數精確）；
    姿態區塊 −J_l(e_r)^{-1} R^T J_ω —— `J_l^{-1}` 就是 **log 微分修正**，
    少了它在 30° 就有約 20 % 偏差（WG0 實測）。
    """
    q = np.asarray(q, float)
    T = K.fk(q, tcp)
    R = T[:3, :3]
    J = K.jacobian(q, tcp)
    e_r = so3_log(R.T @ T_des[:3, :3], pi_band)
    H = np.zeros((NE, NQ))
    H[:3, :] = -J[:3, :]
    H[3:, :] = -so3_Jl_inv(e_r) @ R.T @ J[3:, :]
    return H


def task_error_and_jacobian(K, q: np.ndarray, T_des: np.ndarray, tcp: str,
                            pi_band: float = 1e-5):
    """同時回傳 (e, H)，**fk 只算一次**。

    **等價改寫**：數值定義與 `task_error` / `task_error_jacobian` 完全相同，
    只是避免重複的 FK。原本每個預測步做
    `task_error_jacobian`（fk ＋ jacobian）再加 `task_error`（又一次 fk），
    實測 fk 0.1192 ms、jacobian 0.3748 ms ⇒ 合併省約 19 %。
    """
    q = np.asarray(q, float)
    T = K.fk(q, tcp)
    R = T[:3, :3]
    J = K.jacobian(q, tcp)
    e_r = so3_log(R.T @ T_des[:3, :3], pi_band)
    e = np.concatenate([T_des[:3, 3] - T[:3, 3], e_r])
    H = np.zeros((NE, NQ))
    H[:3, :] = -J[:3, :]
    H[3:, :] = -so3_Jl_inv(e_r) @ R.T @ J[3:, :]
    return e, H


# ------------------------------------------------------------------ 配置
@dataclass
class WGMPCConfig:
    """WG1-DEV-1 的開發值。**開發參數可調；驗收門檻不在此檔。**"""
    N: int = 10
    dt: float = 0.05
    # 成本（無量綱權重；參考尺度見下）
    w_p: float = 50.0
    w_r: float = 25.0
    w_bt: float = 1.0e-3
    w_br: float = 1.0e-3
    w_a: float = 1.0e-3
    w_s: float = 1.0e-3
    # 變化率權重可分底盤／手臂；None = 沿用 w_s（預設，行為不變）
    w_s_base: float | None = None
    w_s_arm: float | None = None
    # **整機協同（移動中操作）**——兩項皆預設關閉（權重 0），行為與既有完全相同。
    # 僅增廣核心（wgmpc_core_sp）實作；本核心見到非零值會拒絕求解。
    #   底盤參考速度：w_vref·Σ_k Σ_i ((u_k,i − v_ref,i)/vmax_i)²，i ∈ 底盤三軸，
    #     v_ref 為**本體座標**。讓底盤照外部給的剖面持續移動，TCP 由手臂補償。
    #   手臂名目姿態：w_qn·Σ_k ‖q_arm,k − q_nom‖²（rad²）。手臂吸收短期的
    #     差異，長期位移由底盤承擔 —— 少了它，手臂只要比底盤便宜就會一路
    #     伸到關節餘量。
    w_vref: float = 0.0
    base_vref: tuple | None = None
    w_qn: float = 0.0
    arm_q_nom: tuple | None = None
    Qf_scale: float = 5.0
    # 速度框（逐軸）——  L1
    v_base_lin: float = 0.035255
    v_base_ang: float = 0.199900
    v_arm: float = 0.999900
    # 加速度框
    a_base_lin: float = 0.50      # **開發值**，無已核准來源
    a_base_ang: float = 2.00      # **開發值**
    a_arm: float = 19.984         # LITE6_SAFE
    joint_margin: float = 0.05
    # 輪級
    wheel_radius: float = 0.05
    wheel_base_L: float = 0.245
    wheel_w_max: float = 5.55
    wheel_a_max: float = 125.0
    # SQP
    n_sqp: int = 10
    delta_0: float = 0.2
    delta_max: float = 1.0
    delta_min_conv: float = 0.01
    gamma_up: float = 2.0
    gamma_dn: float = 0.5
    tol_accept: float = 1e-9
    tol_step: float = 1e-4
    tol_cost: float = 1e-4
    # QP
    # **必須比 r_tol 緊**：OSQP 的 solved 只保證**縮放後**殘差 <= eps，
    # 與未縮放的約束列不是同一個量。eps 與 r_tol 同階時，
    # 合法的 solved 解可能通不過自己的殘差核對（實測發生過）。
    eps_abs: float = 1e-7
    eps_rel: float = 1e-7
    max_iter: int = 20000
    polish: bool = True
    r_tol: float = 1e-6
    accepted_status: tuple = ('solved',)
    # H_k
    pi_band: float = 1e-5
    tcp: str = 'link_tcp'

    def vmax(self) -> np.ndarray:
        return np.concatenate([[self.v_base_lin, self.v_base_lin,
                                self.v_base_ang],
                               np.full(6, self.v_arm)])

    def amax(self) -> np.ndarray:
        return np.concatenate([[self.a_base_lin, self.a_base_lin,
                                self.a_base_ang], np.full(6, self.a_arm)])

    def Q(self) -> np.ndarray:
        """Q = blkdiag(w_p/L_p² I_3, w_r/L_r² I_3)，L_p = 1 m、L_r = 1 rad。"""
        return np.diag(np.concatenate([np.full(3, self.w_p),
                                       np.full(3, self.w_r)]))

    def R(self) -> np.ndarray:
        """各分量以**自己的限制**正規化 ⇒ 權重 1 = 飽和代價相同。"""
        v = self.vmax()
        w = np.concatenate([[self.w_bt, self.w_bt, self.w_br],
                            np.full(6, self.w_a)])
        return np.diag(w / (v * v))

    def S(self) -> np.ndarray:
        """命令**變化率**權重。底盤與手臂可分開設。

        `w_s_base` / `w_s_arm` 為 None 時沿用 `w_s`（與原行為完全相同）。
        甩動發生在手臂；底盤的命令本來就平順，用同一個高權重壓它只會
        拖慢末段沉降。
        """
        v = self.vmax()
        wb = self.w_s if self.w_s_base is None else self.w_s_base
        wa = self.w_s if self.w_s_arm is None else self.w_s_arm
        w = np.concatenate([np.full(3, wb), np.full(6, wa)])
        return np.diag(w / (v * v))

    def wheel_matrix(self) -> np.ndarray:
        L = self.wheel_base_L
        return np.array([[0.0, 1.0, L], [-1.0, 0.0, L],
                         [0.0, -1.0, L], [1.0, 0.0, L]])


# ------------------------------------------------------------------ 結果
@dataclass
class WGMPCResult:
    ok: bool
    reason: str = ''
    u0: np.ndarray | None = None
    U: np.ndarray | None = None               # (N, 9) 最終控制序列
    Q_pred: np.ndarray | None = None          # (N+1, 9) **最終非線性 rollout**
    E_pred: np.ndarray | None = None          # (N+1, 6) 對應的真實任務誤差
    sqp_converged: bool = False
    sqp_stop_reason: str = ''
    n_sqp_used: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    qp_status: list = field(default_factory=list)
    qp_iters: list = field(default_factory=list)
    J_nl: list = field(default_factory=list)
    delta: list = field(default_factory=list)
    max_residual: float = float('nan')
    residual_by_block: dict = field(default_factory=dict)
    violated_blocks: list = field(default_factory=list)
    timing_ms: dict = field(default_factory=dict)
    lin_error: dict = field(default_factory=dict)   # t3b：線性化誤差指標
    nominal_infeasible: bool = False        # 暖啟動 nominal 是否不可行
    nominal_residual: float = float('nan')
    first_feasible_unconditional: bool = False  # 是否因無基準而無條件接受首個候選
    n_recovery_steps: int = 0               # 可行性恢復步次數（不計為收斂）
    n_gate_rejected: int = 0                # 未通過逐候選閘門的次數
    last_gate_reject: str = ''


# ---------------------------------------------------- 凝縮預測（含仿射項）
def build_prediction(A_list, B_list, c_list):
    """q_stack = Φ q_0 + Γ z + γ，支援**逐步** A_k, B_k, c_k。

    `gmpc._build_prediction` 寫死 Γ 的對角塊為 dt·I 且狀態與輸入同維、
    **沒有仿射項**，因此不能重用（WG0 C6）。這裡自己組。
    """
    N = len(A_list)
    Phi = np.zeros((N * NQ, NQ))
    Gam = np.zeros((N * NQ, N * NU))
    gam = np.zeros(N * NQ)
    for k in range(N):
        r = slice(k * NQ, (k + 1) * NQ)
        if k == 0:
            Phi[r] = A_list[0]
            Gam[r, 0:NU] = B_list[0]
            gam[r] = c_list[0]
        else:
            rp = slice((k - 1) * NQ, k * NQ)
            Ak = A_list[k]
            Phi[r] = Ak @ Phi[rp]
            Gam[r, 0:k * NU] = Ak @ Gam[rp, 0:k * NU]
            Gam[r, k * NU:(k + 1) * NU] = B_list[k]
            gam[r] = Ak @ gam[rp] + c_list[k]
    return Phi, Gam, gam


def _diff_operator(N):
    """Δ = D z + f，其中 f 帶 −u_prev（k = 0 的那一塊）。"""
    D = np.zeros((N * NU, N * NU))
    for k in range(N):
        r = slice(k * NU, (k + 1) * NU)
        D[r, r] = np.eye(NU)
        if k > 0:
            D[r, (k - 1) * NU:k * NU] = -np.eye(NU)
    return D


def nonlinear_cost(K, q0, U, u_prev, T_des, cfg: WGMPCConfig) -> float:
    """**真實**成本 J_nl：用非線性 rollout 與真實 e(q_k)，不用線性化近似。

    SQP 的步長接受一律用這個值（WG0 trust_region.acceptance_rule）。
    """
    Q = rollout(q0, U, cfg.dt)
    Qw, Rw, Sw = cfg.Q(), cfg.R(), cfg.S()
    J = 0.0
    for k in range(1, len(Q)):
        e = task_error(K, Q[k], T_des, cfg.tcp, cfg.pi_band)
        W = Qw * cfg.Qf_scale if k == len(Q) - 1 else Qw
        J += float(e @ W @ e)
    up = np.asarray(u_prev, float)
    for k, u in enumerate(U):
        J += float(u @ Rw @ u)
        d = u - (up if k == 0 else U[k - 1])
        J += float(d @ Sw @ d)
    return J


# ---------------------------------------------------------------- 約束組裝
def build_constraints(q0, U_nom, u_prev, cfg: WGMPCConfig, delta: float):
    """回傳 (A, lo, hi, blocks)。blocks 記錄每段的列範圍，供殘差歸類。

    **輪級限制在 QP 之內**（WG0 constraints.wheel_level.in_qp_not_after）。
    """
    N, dt = cfg.N, cfg.dt
    n = N * NU
    vmax, amax = cfg.vmax(), cfg.amax()
    rows, lo, hi, blocks = [], [], [], {}

    def add(name, Arows, l, u):
        i0 = len(rows)
        rows.extend(Arows)
        lo.extend(l)
        hi.extend(u)
        blocks[name] = (i0, len(rows))

    # 1 速度框
    I = np.eye(n)
    add('velocity', list(I), list(np.tile(-vmax, N)), list(np.tile(vmax, N)))
    # 2 加速度框：D z + f ∈ ±amax·dt
    D = _diff_operator(N)
    f = np.zeros(n)
    f[0:NU] = -np.asarray(u_prev, float)
    bnd = np.tile(amax * dt, N)
    add('acceleration', list(D), list(-bnd - f), list(bnd - f))
    # 3 關節位置（**嚴格線性**，手臂積分無非線性）
    lo_j, hi_j = LITE6_SAFE.lower, LITE6_SAFE.upper
    m = cfg.joint_margin
    Aj, lj, hj = [], [], []
    q_arm0 = np.asarray(q0, float)[ARM]
    for k in range(1, N + 1):
        for i in range(6):
            r = np.zeros(n)
            for j in range(k):
                r[j * NU + 3 + i] = dt
            Aj.append(r)
            lj.append(lo_j[i] + m - q_arm0[i])
            hj.append(hi_j[i] - m - q_arm0[i])
    add('joint_position', Aj, lj, hj)
    # 4 輪級速度與加速度
    W = cfg.wheel_matrix()
    wlim = cfg.wheel_radius * cfg.wheel_w_max
    alim = cfg.wheel_radius * cfg.wheel_a_max * dt
    Aw, lw, hw = [], [], []
    for k in range(N):
        for row in W:
            r = np.zeros(n)
            r[k * NU:k * NU + 3] = row
            Aw.append(r)
            lw.append(-wlim)
            hw.append(wlim)
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
            Aa.append(r)
            la.append(-alim - (-off))
            ha.append(alim - (-off))
    add('wheel_accel', Aa, la, ha)
    # 5 信賴區域：|u_k − u^nom_k| ≤ delta · vmax（**相對量**）
    tr = np.tile(delta * vmax, N)
    z_nom = np.asarray(U_nom, float).reshape(-1)
    add('trust_region', list(I), list(z_nom - tr), list(z_nom + tr))
    return np.asarray(rows, float), np.asarray(lo, float), \
        np.asarray(hi, float), blocks


def residuals(A, lo, hi, z, blocks):
    """逐列殘差（**自己核**，不只信 OSQP 的 status）。"""
    v = A @ z
    r = np.maximum(lo - v, 0.0) + np.maximum(v - hi, 0.0)
    by = {n: float(r[a:b].max()) if b > a else 0.0 for n, (a, b) in blocks.items()}
    return float(r.max()) if len(r) else 0.0, by


# ---------------------------------------------------------------- QP 與 SQP
def _solve_qp(P, qv, A, lo, hi, cfg: WGMPCConfig):
    """單次 QP。回傳 (z, status, iters, t)；status 非 solved 時 z 為 None。

    `t` 把 QP 成本**拆成三段**，因為它們的優化手段完全不同：
      prep  —— 稀疏矩陣轉換（csc_matrix）
      setup —— OSQP 物件建立與 setup()（含其內部的縮放與因式分解）
      iter  —— solve() 本身，才是迭代求解成本
    先前把三者合記為 `qp_solve`，會把「轉換與建立」誤判成「迭代太慢」。
    """
    import contextlib
    import io
    import osqp
    from scipy import sparse
    t = {}
    t0 = time.monotonic()
    Pm = sparse.csc_matrix((P + P.T) * 0.5)
    Am = sparse.csc_matrix(A)
    t['prep'] = (time.monotonic() - t0) * 1e3
    t0 = time.monotonic()
    m = osqp.OSQP()
    with contextlib.redirect_stdout(io.StringIO()):
        m.setup(P=Pm, q=qv, A=Am, l=lo, u=hi, verbose=False,
                eps_abs=cfg.eps_abs, eps_rel=cfg.eps_rel,
                max_iter=cfg.max_iter, polish=cfg.polish)
    t['setup'] = (time.monotonic() - t0) * 1e3
    t0 = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        r = m.solve()
    t['iter'] = (time.monotonic() - t0) * 1e3
    st = str(r.info.status)
    it = int(r.info.iter)
    if st not in cfg.accepted_status:
        return None, st, it, t
    return np.asarray(r.x, float), st, it, t


def solve(K, q0, u_prev, T_des, cfg: WGMPCConfig, U_warm=None) -> WGMPCResult:
    """一個控制週期的多步求解。

    回傳 **完整序列 ＋ 最終非線性 rollout 的預測狀態**；
    只有 `u0` 是要執行的那一步。

    SQP 的五種結果分開回報，**固定跑完 n_sqp 不代表收斂**。
    """
    if cfg.w_vref != 0.0 or cfg.w_qn != 0.0:
        raise ValueError('整機協同項（w_vref／w_qn）只在增廣核心 '
                         'wgmpc_core_sp 實作；本核心不靜默忽略')
    t_all = time.monotonic()
    N, dt = cfg.N, cfg.dt
    q0 = np.asarray(q0, float)
    u_prev = np.asarray(u_prev, float)
    res = WGMPCResult(ok=False)
    tm = {'H': 0.0, 'ABc': 0.0, 'qp_build': 0.0,
          'qp_prep': 0.0, 'qp_setup': 0.0, 'qp_iter': 0.0,
          'rollout': 0.0, 'cost': 0.0, 'post': 0.0}

    # 暖啟動：上一輪最終序列左移一格、末步複製；無則為零
    if U_warm is None:
        U_nom = np.zeros((N, NU))
    else:
        # 末步**以最大允許減速朝零收**，不複製也不直接補零：
        #   複製末步 ⇒ 多走一步，實測把關節推過餘量（joint_position 殘差 4.51e-3）
        #   直接補零 ⇒ 由飽和值跳到 0，實測違反加速度框（殘差 9.99e-2）
        # 這裡用 u_last + clip(0 − u_last, ±a_max·dt)，**依建構滿足加速度框**。
        _W = np.asarray(U_warm, float)
        if len(_W) != N:
            U_nom = np.zeros((N, NU))
        else:
            _tail = _W[-1] if N == 1 else _W[-1]
            _adt = cfg.amax() * cfg.dt
            _last = _tail + np.clip(-_tail, -_adt, _adt)
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

    t0 = time.monotonic()
    J_cur = nonlinear_cost(K, q0, U_nom, u_prev, T_des, cfg)
    tm['cost'] += (time.monotonic() - t0) * 1e3
    res.J_nl.append(J_cur)

    # **nominal 可行性**（排除信賴區域；它以 nominal 為中心，恆含 nominal）。
    # 若 nominal 不可行，ΔJ 比較**沒有有效基準** —— QP 的可行集不含 nominal，
    # 回傳解的線性化目標可以比 nominal 高，於是每個候選都被拒、
    # 信賴區域繞著一個不可行的中心縮小，最後必然 primal infeasible。
    # 實測：第 25 輪暖啟動 nominal 違反 joint_position 4.51e-3 rad。
    _Am, _lo, _hi, _bl = build_constraints(q0, U_nom, u_prev, cfg, cfg.delta_max)
    _b2 = {k: v for k, v in _bl.items() if k != 'trust_region'}
    _a = min(v[0] for v in _b2.values())
    _b = max(v[1] for v in _b2.values())
    _mx, _ = residuals(_Am[_a:_b], _lo[_a:_b], _hi[_a:_b], U_nom.reshape(-1),
                       {k: (v[0] - _a, v[1] - _a) for k, v in _b2.items()})
    nominal_feasible = bool(_mx <= cfg.r_tol)
    res.nominal_infeasible = not nominal_feasible
    res.nominal_residual = float(_mx)

    delta = cfg.delta_0
    U_best = None               # **只有被接受的候選才能成為回傳對象**
    J_best = J_cur
    stop = 'iter_limit'
    conv = False

    for it in range(cfg.n_sqp):
        res.n_sqp_used = it + 1
        res.delta.append(delta)
        # --- nominal rollout 與逐步線性化 ---
        t0 = time.monotonic()
        Qn = rollout(q0, U_nom, dt)
        tm['rollout'] += (time.monotonic() - t0) * 1e3
        t0 = time.monotonic()
        A_l, B_l, c_l = [], [], []
        for k in range(N):
            a, b, c = affine_model(Qn[k], U_nom[k], dt)
            A_l.append(a); B_l.append(b); c_l.append(c)
        tm['ABc'] += (time.monotonic() - t0) * 1e3
        t0 = time.monotonic()
        _eh = [task_error_and_jacobian(K, Qn[k], T_des, cfg.tcp, cfg.pi_band)
               for k in range(1, N + 1)]
        Hs = [x[1] for x in _eh]
        e_nom = np.concatenate([x[0] for x in _eh])
        tm['H'] += (time.monotonic() - t0) * 1e3

        # --- 凝縮與成本 ---
        t0 = time.monotonic()
        Phi, Gam, gam = build_prediction(A_l, B_l, c_l)
        Hblk = np.zeros((N * NE, N * NQ))
        for k in range(N):
            Hblk[k * NE:(k + 1) * NE, k * NQ:(k + 1) * NQ] = Hs[k]
        q_nom_stack = Qn[1:].reshape(-1)
        G = Hblk @ Gam
        d = e_nom + Hblk @ (Phi @ q0 + gam - q_nom_stack)
        P = 2.0 * (G.T @ Qbar @ G + Rbar + D.T @ Sbar @ D)
        qv = 2.0 * (G.T @ Qbar @ d + D.T @ Sbar @ fvec)
        # **nominal 不可行時不設信賴區域**：繞著一個不可行點的區域會把問題
        # 切空（實測：不含信賴區域可行、含 Δ=0.2 不可行 ⇒ primal infeasible）。
        # 信賴區域的用途是限制步長在線性化的有效範圍內，
        # 以不可行點為中心沒有意義。找到可行點後即恢復。
        _d_eff = cfg.delta_max * 1e3 if not nominal_feasible else delta
        Am, lo, hi, blocks = build_constraints(q0, U_nom, u_prev, cfg, _d_eff)
        tm["qp_build"] += (time.monotonic() - t0) * 1e3

        # **正規化決策變數**：z = Σ ẑ，Σ = diag(tile(vmax, N))。
        # 這是純變數變換（同一個最佳化問題），目的是改善條件數 ——
        # 未正規化時 R 的底盤項與手臂項相差約 800 倍
        # （0.805 vs 1.0e-3），OSQP 的迭代數因此暴增。
        # 正規化後速度框成為 |ẑ| ≤ 1、R 的對角等於無量綱權重。
        sig = np.tile(cfg.vmax(), N)
        Ps = P * np.outer(sig, sig)
        qs = qv * sig
        Ams = Am * sig[None, :]
        zh, st, nit, _tq = _solve_qp(Ps, qs, Ams, lo, hi, cfg)
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
        # --- **逐候選閘門**：有限值 ＋ 硬約束殘差，**先過閘才進成本比較** ---
        # 不能只靠最後的輸出檢查：未過閘的候選若先進了 nominal，
        # 後續的線性化與 ΔJ 比較就建立在一個不合格的點上。
        # `_Am[_a:_b]` 是**與候選無關**的硬約束（速度／加速度／關節積分／輪級）；
        # 信賴區域區塊以 nominal 為中心，不納入本閘門。
        _finite = bool(np.isfinite(U_cand).all())
        _mxc, _byc = (residuals(_Am[_a:_b], _lo[_a:_b], _hi[_a:_b],
                                U_cand.reshape(-1),
                                {k: (v[0] - _a, v[1] - _a)
                                 for k, v in _b2.items()})
                      if _finite else (float('inf'), {}))
        _gate_ok = _finite and _mxc <= cfg.r_tol
        if not _gate_ok:
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
        # --- **在覆寫 nominal 之前**算更新大小與成本差 ---
        step_inf = float(np.max(np.abs(U_cand - U_nom) / cfg.vmax()))
        t0 = time.monotonic()
        J_cand = nonlinear_cost(K, q0, U_cand, u_prev, T_des, cfg)
        tm['cost'] += (time.monotonic() - t0) * 1e3
        dJ = J_cand - J_cur
        res.J_nl.append(J_cand)

        # **nominal 不可行 ⇒ 沒有有效的比較基準**：
        # 無條件接受第一個通過殘差核對的候選，之後才開始 ΔJ 比較。
        # 不是放寬下降要求 —— 是因為「比一個不可行點更差」沒有意義。
        _accept = dJ <= cfg.tol_accept
        _was_recovery = False
        if not nominal_feasible:
            # **可行性恢復步**（W3，政策修訂 —— 見 wgmpc_wg1_round1_result）：
            # nominal 不可行時沒有有效的 ΔJ 基準，無條件接受**已過閘**的候選。
            # 候選已經通過上面的硬約束殘差核對，所以它是可行點。
            _accept = True
            _was_recovery = True
            nominal_feasible = True           # 之後有了有效基準
            res.first_feasible_unconditional = True
            res.n_recovery_steps += 1
        if _accept:
            # **接受**（含零更新：ΔJ ≤ tol_accept 的等號情形）
            res.n_accepted += 1
            U_best, J_best = U_cand.copy(), J_cand
            rel = abs(dJ) / max(1.0, abs(J_cur))
            U_nom, J_cur = U_cand, J_cand
            delta = min(cfg.delta_max, delta * cfg.gamma_up)
            # 收斂判定**附帶 Δ ≥ delta_min_conv** ——
            # 信賴區域造成的小步長不單獨作為收斂證據
            # **可行性恢復步本身不判為 SQP 收斂** ——
            # 它是為了離開不可行點，不是因為已近最佳。
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

    # --- 輸出政策（固定，不留給實作者臨場決定）---
    # **任一次 QP 失敗即回 no_valid_solution**，即使先前已接受過候選。
    # WG0 output_policy_fixed.on_qp_failed 如此規定；
    # 先前的實作在 qp_failed 後只要 U_best 非 None 仍會走到 ok=True。
    if stop == 'qp_failed' or U_best is None:
        res.reason = ('qp_failed' if stop == 'qp_failed'
                      else 'no_accepted_candidate')
        res.timing_ms = {k: round(v, 4) for k, v in tm.items()}
        res.timing_ms['total'] = round((time.monotonic() - t_all) * 1e3, 4)
        return res          # **不得把暖啟動序列冒稱為新求解成功**

    # --- 最終接受檢查：針對**實際要返回的序列**及其非線性 rollout ---
    t0 = time.monotonic()
    Qf_ = rollout(q0, U_best, dt)
    tm['rollout'] += (time.monotonic() - t0) * 1e3
    Am, lo, hi, blocks = build_constraints(q0, U_best, u_prev, cfg,
                                           cfg.delta_max)
    mx, by = residuals(Am, lo, hi, U_best.reshape(-1), blocks)
    res.max_residual, res.residual_by_block = mx, by
    res.violated_blocks = [n for n, v in by.items() if v > cfg.r_tol]
    if res.violated_blocks:
        res.reason = 'residual_check_failed:' + ','.join(res.violated_blocks)
        res.timing_ms = {k: round(v, 4) for k, v in tm.items()}
        res.timing_ms['total'] = round((time.monotonic() - t_all) * 1e3, 4)
        return res

    _t_post = time.monotonic()
    res.ok = True
    res.u0 = U_best[0].copy()
    res.U = U_best
    res.Q_pred = Qf_
    res.E_pred = np.array([task_error(K, Qf_[k], T_des, cfg.tcp, cfg.pi_band)
                           for k in range(len(Qf_))])
    res.J_nl.append(J_best)
    # t3b：線性化誤差指標（**只記錄，不設門檻**）
    A_l, B_l, c_l = [], [], []
    for k in range(N):
        a, b, c = affine_model(Qf_[k], U_best[k], dt)
        A_l.append(a); B_l.append(b); c_l.append(c)
    Phi, Gam, gam = build_prediction(A_l, B_l, c_l)
    q_aff = (Phi @ q0 + Gam @ U_best.reshape(-1) + gam).reshape(N, NQ)
    res.lin_error = {
        'max_abs_state': float(np.abs(q_aff - Qf_[1:]).max()),
        'max_abs_base_xy': float(np.abs(q_aff[:, :2] - Qf_[1:, :2]).max()),
        'max_abs_theta': float(np.abs(q_aff[:, 2] - Qf_[1:, 2]).max()),
        'note': '仿射預測 vs 非線性 rollout 的差距 —— **線性化誤差指標，'
                '不要求為零**',
    }
    # **total 量到 return 之前**：E_pred 與線性化診斷也計入
    tm['post'] += (time.monotonic() - _t_post) * 1e3
    res.timing_ms = {k: round(v, 4) for k, v in tm.items()}
    res.timing_ms['total'] = round((time.monotonic() - t_all) * 1e3, 4)
    return res
