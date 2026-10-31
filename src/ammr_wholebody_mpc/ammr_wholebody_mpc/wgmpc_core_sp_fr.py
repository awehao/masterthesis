"""W-GMPC 增廣核心的**時域內未來參考**版本（2026-10-31；瓶子移動夾取最小設計 §4、Codex 20261031 有條件核可）。
wgmpc_core_sp.solve_sp 的版本化複本：只把「同一個目標放滿整個時域」改成可逐步給定，其餘（動力學、約束、SQP 政策、閘門、
最終殘差核對）一律從 wgmpc_core_sp 匯入、逐字沿用。凍結核心不改。
  T_des：4×4（單一目標，與 solve_sp 逐位元相同）或 (N+1, 4, 4)（第 k 個用於預測狀態 Z[k]，k＝0..N；k＝0 只進 E_pred 紀錄）。
  cfg.base_vref：(3,)（每步同值，與 solve_sp 相同）或 (N, 3)（第 k 個對應 u_k，k＝0..N−1，本體座標）。
形狀不符或含非有限值 ⇒ ValueError（不以末值補齊、不以零代替）。
"""
import math
import time

import numpy as np

from .wgmpc_core_sp import (ARM, BASE, NE, NQ, NS, NU, NZ, WGMPCConfigSP, WGMPCResultSP,   # noqa: F401
                            _base_hold_vec, _diff_operator, _solve_qp, affine_model_sp, build_constraints_sp,
                            build_prediction_sp, residuals, rollout_sp, row_scale, task_error_and_jacobian_sp,
                            task_error_sp)
from .wgmpc_core import WGMPCConfig


def target_seq(T_des, N):
    """單一 4×4 ⇒ 重複 N+1 次（同一物件內容）；(N+1, 4, 4) ⇒ 原樣（核形狀與有限值）。"""
    T = np.asarray(T_des, float)
    if T.shape == (4, 4):
        T = np.broadcast_to(T, (N + 1, 4, 4))
    elif T.shape != (N + 1, 4, 4):
        raise ValueError(f'T_des 形狀須為 (4,4) 或 ({N + 1},4,4)，收到 {T.shape}')
    if not np.isfinite(T).all():
        raise ValueError('T_des 含非有限值')
    return T


def coord_params_fr(cfg: WGMPCConfig, N):
    """同 wgmpc_core_sp.coord_params，但 v_ref 回傳 (N, 3)：單一 (3,) ⇒ 每步同值。"""
    wv, wq = float(cfg.w_vref), float(cfg.w_qn)
    if not (math.isfinite(wv) and wv >= 0.0 and math.isfinite(wq) and wq >= 0.0):
        raise ValueError(f'協同權重必須為有限非負：w_vref={wv} w_qn={wq}')
    vr = qn = None
    if wv > 0.0:
        if cfg.base_vref is None:
            raise ValueError('w_vref > 0 但沒有 base_vref')
        v = np.asarray(cfg.base_vref, float)
        if v.shape == (3,):
            vr = np.tile(v, (N, 1))
        elif v.shape == (N, 3):
            vr = v
        else:
            raise ValueError(f'base_vref 形狀須為 (3,) 或 ({N},3)，收到 {v.shape}')
        if not np.isfinite(vr).all():
            raise ValueError(f'base_vref 含非有限值：{vr}')
    if wq > 0.0:
        if cfg.arm_q_nom is None:
            raise ValueError('w_qn > 0 但沒有 arm_q_nom')
        qn = np.asarray(cfg.arm_q_nom, float).reshape(NS)
        if not np.isfinite(qn).all():
            raise ValueError(f'arm_q_nom 含非有限值：{qn}')
    return wv, vr, wq, qn


def coord_cost_fr(Z, U, cfg: WGMPCConfig) -> float:
    U2 = np.atleast_2d(U)
    wv, vr, wq, qn = coord_params_fr(cfg, len(U2))
    J = 0.0
    if wv > 0.0:
        vb = cfg.vmax()[BASE]
        for k, u in enumerate(U2):
            r = (np.asarray(u, float)[BASE] - vr[k]) / vb
            J += wv * float(r @ r)
    if wq > 0.0:
        for z in np.asarray(Z)[1:]:
            r = np.asarray(z, float)[ARM] - qn
            J += wq * float(r @ r)
    return J


def coord_qp_terms_fr(cfg: WGMPCConfig, N, Phi, Gam, gam, z0):
    wv, vr, wq, qn = coord_params_fr(cfg, N)
    n = N * NU
    P = np.zeros((n, n))
    qv = np.zeros(n)
    if wv > 0.0:
        w = np.zeros(NU)
        w[BASE] = wv / cfg.vmax()[BASE] ** 2
        r = np.zeros((N, NU))
        r[:, BASE] = vr
        Wb = np.kron(np.eye(N), np.diag(w))
        P += 2.0 * Wb
        qv += -2.0 * Wb @ r.reshape(-1)
    if wq > 0.0:
        rows = np.concatenate([np.arange(k * NZ + 3, k * NZ + 3 + NS) for k in range(N)])
        Ga = Gam[rows]
        off = (Phi @ np.asarray(z0, float) + gam)[rows] - np.tile(qn, N)
        P += 2.0 * wq * (Ga.T @ Ga)
        qv += 2.0 * wq * (Ga.T @ off)
    return P, qv


def nonlinear_cost_fr(K, z0, U, u_prev, T_des, cfg: WGMPCConfigSP) -> float:
    """同 nonlinear_cost_sp；第 k 個預測狀態對 T_des[k]（T_des 已為 (N+1,4,4)）。"""
    Z = rollout_sp(z0, U, cfg)
    Qw, Rw, Sw = cfg.Q(), cfg.R(), cfg.S()
    J = 0.0
    for k in range(1, len(Z)):
        e = task_error_sp(K, Z[k], T_des[k], cfg.tcp, cfg.pi_band)
        W = Qw * cfg.Qf_scale if k == len(Z) - 1 else Qw
        J += float(e @ W @ e)
    up = np.asarray(u_prev, float)
    for k, u in enumerate(U):
        J += float(u @ Rw @ u)
        d = u - (up if k == 0 else U[k - 1])
        J += float(d @ Sw @ d)
    return J + coord_cost_fr(Z, U, cfg)


def solve_sp_fr(K, z0, u_prev, T_des, cfg: WGMPCConfigSP,
                U_warm=None) -> WGMPCResultSP:
    """一個控制週期的多步求解（增廣狀態）。

    **SQP 政策逐字沿用** `wgmpc_core.solve`：暖啟動末步以最大允許減速朝零收、
    nominal 可行性預檢、逐候選閘門（有限值＋硬約束殘差）、可行性恢復步、
    五種停止狀態、任一次 QP 失敗即 no_valid_solution、
    以及對**實際回傳序列**的最終殘差核對。此處只換動力學與關節限位。
    """
    t_all = time.monotonic()
    N = cfg.N
    T_des = target_seq(T_des, N)                       # FR：(N+1, 4, 4)；單一目標 ⇒ 每步同一矩陣
    coord_params_fr(cfg, N)                             # FR：提前檢查 base_vref 形狀
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
    U_nom = np.clip(U_nom, -cfg.vbox(), cfg.vbox())   # base_fixed 時名目底盤亦為 0
    if cfg.base_hold is not None:
        U_nom[:, :3] = _base_hold_vec(cfg)                # PARK_HOLD：名目底盤 = u_hold

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
    J_cur = nonlinear_cost_fr(K, z0, U_nom, u_prev, T_des, cfg)
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
        _eh = [task_error_and_jacobian_sp(K, Zn[k], T_des[k], cfg.tcp,
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
        if cfg.w_vref > 0.0 or cfg.w_qn > 0.0:
            _Pc, _qc = coord_qp_terms_fr(cfg, N, Phi, Gam, gam, z0)
            P = P + _Pc
            qv = qv + _qc
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
        J_cand = nonlinear_cost_fr(K, z0, U_cand, u_prev, T_des, cfg)
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
    res.E_pred = np.array([task_error_sp(K, Zf[k], T_des[k], cfg.tcp,
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
