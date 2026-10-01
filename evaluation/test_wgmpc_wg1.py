"""WG1 第一輪驗證：t1 模型／導數、t2 限制與失敗、t3 預測與到達、t4 時域成本。

**不接 ROS、不開 Isaac、不用 GPU、單執行緒。**
門檻取自 wgmpc_wg1_dev_config.yaml 的 acceptance_criteria（**不可自行放寬**）。
"""
from __future__ import annotations
import math
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                   # noqa
from ammr_wholebody_mpc.wgmpc_core import (                            # noqa
    ARM, NE, NQ, NU, WGMPCConfig, affine_model, body_to_world,
    build_constraints, nonlinear_cost, residuals, rollout, skew, so3_Jl_inv,
    so3_log, solve, step, task_error, task_error_jacobian)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
TCP = 'link_tcp'
# ---- acceptance_criteria（WG1-DEV-1）----
TOL_AB_FD = 1e-8
TOL_AFFINE = 1e-10
TOL_HP_FD = 1e-8
TOL_HR_REL = {'le179_9': 1e-4, 'at179_99': 2e-3}
EULER_RATIO = (3.6, 4.4)
TOL_R = 1e-6
REACH_P, REACH_R, REACH_MAX = 0.005, 0.02, 400

_bad = 0


def ck(name, cond, extra=''):
    global _bad
    print(f'  {name:56s} {"ok" if cond else "**錯**"}{extra}')
    _bad += not cond


def expm_so3(w):
    th = float(np.linalg.norm(w))
    if th < 1e-14:
        return np.eye(3) + skew(w)
    Kk = skew(np.asarray(w, float) / th)
    return np.eye(3) + math.sin(th) * Kk + (1 - math.cos(th)) * Kk @ Kk


def target_from(K, q, rot_deg=0.0, dp=(0.10, -0.05, 0.07), axis=(0.3, -0.5, 0.81)):
    T = K.fk(q, TCP).copy()
    ax = np.asarray(axis, float)
    ax /= np.linalg.norm(ax)
    Td = np.eye(4)
    Td[:3, 3] = T[:3, 3] + np.asarray(dp, float)
    Td[:3, :3] = T[:3, :3] @ expm_so3(ax * math.radians(rot_deg))
    return Td


# ========================================================= t1
def t1(K):
    print('t1 模型／導數')
    rng = np.random.default_rng(11)
    dt = 0.05
    for th in (0.0, math.pi / 2, -1.234):
        q = np.zeros(NQ); q[2] = th
        got = step(q, np.array([1.0] + [0.0] * 8), dt)[:2] / dt
        ck(f'B(θ={th:+.3f})：本體 +x → 世界 (cos,sin)',
           np.allclose(got, [math.cos(th), math.sin(th)]))
    mx_A = mx_B = mx_aff = 0.0
    for _ in range(5):
        q = np.concatenate([rng.normal(0, 1, 3), rng.uniform(-1, 1, 6)])
        u = rng.uniform(-0.5, 0.5, NU)
        A, B, c = affine_model(q, u, dt)
        h = 1e-7
        Afd = np.zeros((NQ, NQ)); Bfd = np.zeros((NQ, NU))
        for i in range(NQ):
            e = np.zeros(NQ); e[i] = h
            Afd[:, i] = (step(q + e, u, dt) - step(q - e, u, dt)) / (2 * h)
            eu = np.zeros(NU); eu[i] = h
            Bfd[:, i] = (step(q, u + eu, dt) - step(q, u - eu, dt)) / (2 * h)
        mx_A = max(mx_A, np.abs(A - Afd).max())
        mx_B = max(mx_B, np.abs(B - Bfd).max())
        mx_aff = max(mx_aff, np.abs(step(q, u, dt) - (A @ q + B @ u + c)).max())
    ck('A_k vs 有限差分', mx_A < TOL_AB_FD, f'  max|Δ| {mx_A:.2e}')
    ck('B_k vs 有限差分', mx_B < TOL_AB_FD, f'  max|Δ| {mx_B:.2e}')
    ck('仿射自洽 f = Aq+Bu+c', mx_aff < TOL_AFFINE, f'  max|Δ| {mx_aff:.2e}')
    # H 位置區塊
    q = np.concatenate([np.array([0.1, 0.2, -0.4]), rng.uniform(-0.6, 0.6, 6)])
    Td = target_from(K, q, 20.0)
    H = task_error_jacobian(K, q, Td, TCP)
    h = 1e-7
    Hfd = np.zeros((NE, NQ))
    for i in range(NQ):
        d = np.zeros(NQ); d[i] = h
        Hfd[:, i] = (task_error(K, q + d, Td, TCP)
                     - task_error(K, q - d, Td, TCP)) / (2 * h)
    ck('H 位置區塊 −J_p vs 有限差分',
       np.abs(H[:3] - Hfd[:3]).max() < TOL_HP_FD,
       f'  max|Δ| {np.abs(H[:3]-Hfd[:3]).max():.2e}')
    # H 姿態區塊，含大誤差
    print('    H 姿態區塊（步長按角度；≥180° 依分支處置不做差分）')
    for deg in (0.5, 30, 90, 150, 170, 179.0, 179.9, 179.99):
        q = np.concatenate([np.array([0.2, -0.3, 0.6]), rng.uniform(-0.6, 0.6, 6)])
        Td = target_from(K, q, deg)
        hh = 1e-5 if deg > 179.0 else 1e-7
        Hfd = np.zeros((NE, NQ))
        for i in range(NQ):
            d = np.zeros(NQ); d[i] = hh
            Hfd[:, i] = (task_error(K, q + d, Td, TCP)
                         - task_error(K, q - d, Td, TCP)) / (2 * hh)
        Ha = task_error_jacobian(K, q, Td, TCP)
        rel = np.abs(Ha[3:] - Hfd[3:]).max() / max(1.0, np.abs(Hfd[3:]).max())
        tol = TOL_HR_REL['at179_99'] if deg >= 179.99 else TOL_HR_REL['le179_9']
        ck(f'  {deg:7.2f}°（h={hh:.0e}）', rel < tol, f'  相對 {rel:.2e}')
        if deg == 30:
            Hno = np.zeros((3, NQ))
            Hno[:] = -(K.fk(q, TCP)[:3, :3].T @ K.jacobian(q, TCP)[3:, :])
            relno = np.abs(Hno - Hfd[3:]).max() / max(1.0, np.abs(Hfd[3:]).max())
            ck('  **反例**：30° 省略 J_l⁻¹ 必須明顯偏', relno > 0.05,
               f'  相對 {relno:.1%}')
    # 反例：省略 B(θ)
    q = np.array([0.0, 0.0, 1.0, 0.1, 0.2, -0.1, 0.0, 0.3, 0.0])
    u = np.array([0.03, 0.02, 0.0] + [0.0] * 6)
    ck('**反例**：省略 B(θ) 的積分必須與正確者不同',
       not np.allclose(q + 0.05 * u, step(q, u, 0.05)),
       f'  差 {np.abs((q+0.05*u)-step(q,u,0.05)).max():.4f}')
    # 反例：A_k 的 θ 欄
    A, _, _ = affine_model(q, u, 0.05)
    ck('**反例**：A_k 的 θ 欄非零（置零會偏）', np.abs(A[:2, 2]).max() > 1e-6,
       f'  |A[0:2,2]| {np.abs(A[:2,2]).max():.4f}')
    # J_l^{-1} 穩定性
    ax = np.array([0.3, -0.5, 0.81]); ax /= np.linalg.norm(ax)
    nrm = [np.linalg.norm(so3_Jl_inv(ax * math.radians(d)), 2)
           for d in (90, 150, 179, 179.99)]
    ck('J_l⁻¹ 穩定（‖·‖₂ 平滑趨近 π/2）',
       all(np.isfinite(nrm)) and abs(nrm[-1] - math.pi / 2) < 1e-3,
       f'  {[round(x,4) for x in nrm]}')
    # 跨 π 的分支（**只核分支處置，不核差分**）
    e_a = so3_log(expm_so3(ax * math.radians(179.99)))
    e_b = so3_log(expm_so3(ax * math.radians(180.01)))
    ck('跨 π 主值換向（已知分支行為，不要求差分一致）',
       float(e_a @ e_b) < 0, f'  內積 {float(e_a@e_b):+.4f}')
    # Euler 局部階數
    q0 = np.array([0.0, 0.0, 0.3, 0.1, 0.2, -0.1, 0.0, 0.3, 0.0])
    u = np.array([0.03, 0.02, 0.18] + [0.0] * 6)

    def exact(q0, u, T):
        x, y, th = q0[:3]; vx, vy, w = u[:3]
        th2 = th + w * T
        X = x + ((math.sin(th2) - math.sin(th)) * vx
                 + (math.cos(th2) - math.cos(th)) * vy) / w
        Y = y + (-(math.cos(th2) - math.cos(th)) * vx
                 + (math.sin(th2) - math.sin(th)) * vy) / w
        return np.concatenate([[X, Y, th2], q0[3:] + T * u[3:]])
    errs = [np.linalg.norm(step(q0, u, d) - exact(q0, u, d))
            for d in (0.08, 0.04, 0.02, 0.01)]
    rat = [errs[i] / errs[i + 1] for i in range(3)]
    ck('Euler 單步局部誤差為 O(dt²)',
       all(EULER_RATIO[0] <= r <= EULER_RATIO[1] for r in rat),
       f'  比值 {[round(r,2) for r in rat]}')


# ========================================================= t2
def t2(K):
    print('\nt2 限制與失敗處置')
    cfg = WGMPCConfig()
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 0.0, 0.5, 0.0])
    Td = target_from(K, q0, 15.0)
    r = solve(K, q0, np.zeros(NU), Td, cfg)
    ck('正常案例有解', r.ok, f'  {r.reason}')
    if r.ok:
        v = cfg.vmax()
        ck('逐步 |u_k| 在速度框內',
           bool((np.abs(r.U) <= v + 1e-9).all()),
           f'  max 比值 {float((np.abs(r.U)/v).max()):.4f}')
        du = np.diff(np.vstack([np.zeros((1, NU)), r.U]), axis=0)
        ck('逐步 |Δu_k| 在加速度框內',
           bool((np.abs(du) <= cfg.amax() * cfg.dt + 1e-9).all()))
        qa = r.Q_pred[1:, ARM]
        lo = np.asarray(LITE6_SAFE.lower) + cfg.joint_margin
        hi = np.asarray(LITE6_SAFE.upper) - cfg.joint_margin
        ck('逐步關節位置在界內', bool((qa >= lo - 1e-9).all()
                                 and (qa <= hi + 1e-9).all()))
        W = cfg.wheel_matrix()
        ws = np.abs(r.U[:, :3] @ W.T)
        ck('逐步輪速在界內',
           bool((ws <= cfg.wheel_radius * cfg.wheel_w_max + 1e-9).all()),
           f'  max {ws.max():.4f} / {cfg.wheel_radius*cfg.wheel_w_max:.4f}')
        ck('最終殘差 ≤ r_tol', r.max_residual <= TOL_R,
           f'  {r.max_residual:.2e}')
        ck('回傳完整序列與預測狀態（可還原）',
           r.U.shape == (cfg.N, NU) and r.Q_pred.shape == (cfg.N + 1, NQ))

    # 不可行 (a)：速度框與加速度框無交集
    c2 = WGMPCConfig(a_base_lin=0.5, dt=0.05)
    up = np.zeros(NU); up[0] = 0.20
    gap = abs(up[0]) - c2.a_base_lin * c2.dt
    ra = solve(K, q0, up, Td, c2)
    ck(f'不可行(a) 速度×加速度衝突（{gap:.4f} > {c2.v_base_lin:.6f}）',
       (not ra.ok) and ra.u0 is None,
       f'  reason={ra.reason}')
    ck('  (a) 失敗時沒有命令輸出', ra.u0 is None and ra.U is None)

    # 不可行 (b)：關節位置超界量 > dt·V_a
    c3 = WGMPCConfig()
    need = 1.5 * c3.dt * c3.v_arm
    q_bad = q0.copy()
    q_bad[3] = LITE6_SAFE.upper[0] - c3.joint_margin + need
    rb = solve(K, q_bad, np.zeros(NU), target_from(K, q_bad, 10.0), c3)
    ck(f'不可行(b) 關節超界 {need:.6f} > dt·V_a {c3.dt*c3.v_arm:.6f}',
       (not rb.ok) and rb.u0 is None, f'  reason={rb.reason}')
    ck('  (b) 失敗時沒有命令輸出', rb.u0 is None and rb.U is None)

    # 狀態閘：只接受 solved
    ck('狀態閘只接受 solved', cfg.accepted_status == ('solved',),
       f'  {cfg.accepted_status}')
    c4 = WGMPCConfig(accepted_status=('solved', 'solved inaccurate'))
    ck('  （對照）放寬狀態閘是**可區分的配置**，不是預設',
       c4.accepted_status != cfg.accepted_status)

    # 殘差核對能擋下「回報 solved 但違反約束」—— 直接測 residuals
    Am, lo_, hi_, blocks = build_constraints(q0, np.zeros((cfg.N, NU)),
                                             np.zeros(NU), cfg, cfg.delta_max)
    z_bad = np.tile(cfg.vmax() * 3.0, cfg.N)
    mx, by = residuals(Am, lo_, hi_, z_bad, blocks)
    ck('殘差核對抓出違反約束的解（與狀態閘獨立）', mx > TOL_R,
       f'  max 殘差 {mx:.4f}、違反區塊 '
       f'{[n for n,v in by.items() if v>TOL_R]}')


# ========================================================= t3
def t3(K):
    print('\nt3 預測一致性與到達')
    cfg = WGMPCConfig()
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 0.0, 0.5, 0.0])
    Td = target_from(K, q0, 20.0)
    r = solve(K, q0, np.zeros(NU), Td, cfg)
    ck('有解', r.ok, f'  {r.reason}')
    if r.ok:
        # t3a：同初值同序列的獨立重播，**要求逐值相等**
        Q2 = rollout(q0, r.U, cfg.dt)
        ck('t3a 回傳 rollout 可獨立重播（要求相等）',
           np.abs(Q2 - r.Q_pred).max() < 1e-12,
           f'  max|Δ| {np.abs(Q2-r.Q_pred).max():.2e}')
        # t3b：線性化誤差，**只記錄**
        print(f'    t3b 線性化誤差（**只記錄不設門檻**）：'
              f'狀態 {r.lin_error["max_abs_state"]:.3e}、'
              f'底盤 xy {r.lin_error["max_abs_base_xy"]:.3e}、'
              f'θ {r.lin_error["max_abs_theta"]:.3e}')
    # 理想閉迴路（**u_prev 每輪更新**）
    print('    理想閉迴路（每輪執行首步、u_prev 每輪更新）')
    cases = [('小位移 ＋ 20°', 0.0, 20.0, (0.10, -0.05, 0.07)),
             ('大轉身 90°', 0.0, 90.0, (0.25, 0.15, 0.05)),
             ('底盤起始 yaw 1.0 ＋ 120°', 1.0, 120.0, (0.30, -0.20, 0.10))]
    for name, yaw0, rot, dp in cases:
        q = np.array([0.0, 0.0, yaw0, 0.0, 0.3, 0.6, 0.0, 0.5, 0.0])
        Td = target_from(K, q, rot, dp)
        up = np.zeros(NU)       # 第一輪：初始靜止、上一命令為零
        Uw = None
        nprev_updates = 0
        ep = er = float('nan')
        fail = None
        for it in range(REACH_MAX):
            rr = solve(K, q, up, Td, cfg, U_warm=Uw)
            if not rr.ok:
                fail = f'第 {it} 輪 {rr.reason}'
                break
            q = step(q, rr.u0, cfg.dt)
            up = rr.u0.copy()          # **每輪更新 u_prev**
            nprev_updates += 1
            Uw = rr.U
            e = task_error(K, q, Td, TCP)
            ep, er = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
            if ep <= REACH_P and er <= REACH_R:
                break
        ok = fail is None and ep <= REACH_P and er <= REACH_R
        ck(f'  {name}', ok,
           f'  {it+1} 輪、位置 {ep*1e3:.2f} mm、姿態 {math.degrees(er):.3f}°'
           + (f'、{fail}' if fail else ''))
        ck(f'    u_prev 每輪都更新', nprev_updates == it + (0 if fail else 1),
           f'  {nprev_updates} 次 / {it + (0 if fail else 1)} 輪')


# ========================================================= t4
def t4(K):
    print('\nt4 時域成本（**量測，不設門檻**）')
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.6, 0.0, 0.5, 0.0])
    Td = target_from(K, q0, 20.0)
    print(f'  {"N":>3} {"total":>9} {"H":>8} {"ABc":>7} {"qp_build":>9} '
          f'{"qp_solve":>9} {"rollout":>8} {"cost":>8} {"iters":>12} {"狀態":>18}')
    for N in (1, 5, 10, 20):
        cfg = WGMPCConfig(N=N)
        ts = []
        for _ in range(3):
            r = solve(K, q0, np.zeros(NU), Td, cfg)
            ts.append(r)
        r = ts[-1]
        t = r.timing_ms
        print(f'  {N:3d} {t["total"]:9.2f} {t["H"]:8.2f} {t["ABc"]:7.2f} '
              f'{t["qp_build"]:9.2f} {t["qp_solve"]:9.2f} {t["rollout"]:8.2f} '
              f'{t["cost"]:8.2f} {str(r.qp_iters):>12} '
              f'{r.sqp_stop_reason:>18}')
        ck(f'  N={N} 有解且回傳完整序列',
           r.ok and r.U is not None and r.U.shape == (N, NU))
    print('  **不設耗時門檻** —— 本輪是量測；量完才決定可行的實作方向。')


def main() -> int:
    K = WholeBodyKinematics.from_urdf_file(URDF)
    print('WG1 純數值核心驗證（不接 ROS、不開 Isaac、單執行緒）\n')
    t1(K); t2(K); t3(K); t4(K)
    print('\nWG1 第一輪：' + ('全部通過' if _bad == 0 else f'**{_bad} 項失敗**'))
    print('WG1 通過只代表**求解器數值行為成立**；'
          '**不得**稱「W-GMPC 自由空間核心已驗證」（那要 WG2）。')
    return 1 if _bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
