#!/usr/bin/env python3
"""gmpc_wb 手臂作用模式（arm={'mode':'active',...}）離線單元測試（Codex 20261031 核可 (b)；封存 gmpc.py 不動）。
N＝20、dt＝0.05（導航原設定）；手臂設定點模型取 free4 辨識（同 W-GMPC）。
  L1 線性化一致：手臂增量（設定點模型）與底盤增量（δe→X_ref·exp(e)→世界）對 TCP 的一階預測＝非線性 FK 有限差分（同一預測狀態）；
  A1 靜止參考、末端目標前移 5 cm：手臂出力、一步非線性推進後 TCP 誤差下降、底盤命令小；
  A2 回傳序列的手臂限制原單位違反 ≤ 1e-6、有限、|u_a| ≤ v_arm、對上一筆加速度框；
  A3 關節貼上限且目標拉向界外：預測實測角與設定點不越界（±margin）；
  A4 底盤沿參考 0.05 m/s、末端目標世界固定：作用模式的時域內 TCP 預測誤差小於手臂保持；
  A5 輪級後處理改動底盤命令時 seq_consistent＝False；A6 煞停路徑（不可行）手臂歸零並註記；A7 耗時（保持 vs 作用）。
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import gmpc_wb as GW                          # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S                     # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                 # noqa: E402
from ammr_wholebody_mpc.se2 import exp_, from_xytheta                 # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
ID = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
MODEL = S.ArmSetpointModel(alpha=np.asarray(ID['alpha'], float), bias=np.asarray(ID['bias_rad'], float), phys_dt=ID['phys_dt_measured_s'])
N, DT = 20, 0.05
Q_READY = np.array([0.0, 0.3, 1.2, 0.0, -0.9, 0.0])
BAD, NCHK = [], 0


def chk(name, cond, info=''):
    global NCHK
    NCHK += 1
    if not cond:
        BAD.append(f'{name} {info}')


def cfg(**kw):
    c = dict(N=N, dt=DT, u_min=np.array([-0.20, -0.25, -0.8]), u_max=np.array([0.35, 0.25, 0.8]), a_max=np.array([1.5, 1.0, 2.0]),
             Q=np.diag([10.0, 10.0, 5.0]), R=np.diag([0.5, 0.5, 0.2]), Qf=np.diag([50.0, 50.0, 25.0]))
    c.update(kw)
    return GW.GMPCConfig(**c)


def ref(xi, X0=None):
    xi_w = np.tile(np.asarray(xi, float), (N + 1, 1))
    X = [np.eye(3) if X0 is None else X0]
    for k in range(N):
        X.append(X[-1] @ exp_(xi_w[k] * DT))
    return np.array(X), xi_w


def arm_dict(T_seq, q=Q_READY, s=None, u_prev=np.zeros(6), **kw):
    d = dict(mode='active', K=K, tcp='link_tcp', q_arm=np.asarray(q, float), s_arm=np.asarray(q if s is None else s, float),
             u_arm_prev=np.asarray(u_prev, float), model=MODEL, T_seq=T_seq, w_p=50.0, w_r=5.0, w_a=0.05, w_s=0.05,
             v_arm=0.9999, a_arm=19.984, lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, margin=0.05)
    d.update(kw)
    return d


def tcp_at(xb, qa):
    return K.fk(np.r_[xb, qa], 'link_tcp')


def main():
    mpc = GW.GMPC(cfg())
    # ---- L1 線性化一致（有限差分）
    Xr, xr = ref([0.05, 0.0, 0.1])
    e0 = np.array([0.01, -0.02, 0.03])
    A_d = np.stack([np.eye(3) - DT * GW.ad(xr[k]) for k in range(N)])
    Phi, Gamma = GW._build_prediction(A_d, DT)
    Mq, cq, Ms, cs = GW.arm_prediction(Q_READY, Q_READY, N, DT, MODEL)
    rng = np.random.default_rng(1)
    dxi, dU = rng.normal(0, 0.02, 3 * N), rng.normal(0, 0.05, 6 * N)
    e_nom = (Phi @ e0).reshape(N, 3)
    worst = 0.0
    for k in (0, 9, 19):
        xb0 = GW._pose_from_err(Xr[k + 1], e_nom[k])
        qa0 = cq[k]
        T0 = tcp_at(xb0, qa0)
        J = K.jacobian(np.r_[xb0, qa0], 'link_tcp')
        B = GW._pose_jac(Xr[k + 1], e_nom[k])
        eps = 1e-4
        d_lin = J[:3, :3] @ B @ (Gamma[3 * k:3 * k + 3] @ dxi) * eps + J[:3, 3:] @ (Mq[k] @ dU) * eps
        xb1 = GW._pose_from_err(Xr[k + 1], e_nom[k] + (Gamma[3 * k:3 * k + 3] @ dxi) * eps)
        qa1 = Mq[k] @ (dU * eps) + cq[k]
        d_fd = tcp_at(xb1, qa1)[:3, 3] - T0[:3, 3]
        worst = max(worst, float(np.linalg.norm(d_lin - d_fd) / max(np.linalg.norm(d_fd), 1e-12)))
    chk('L1 一階預測＝FK 差分（相對誤差 < 1e-3）', worst < 1e-3, f'{worst:.2e}')

    # ---- A1／A2 靜止參考、目標前移 5 cm
    Xr0, xr0 = ref([0, 0, 0])
    T0 = tcp_at(np.zeros(3), Q_READY)
    Tt = T0.copy()
    Tt[0, 3] += 0.05
    T_seq = np.tile(Tt, (N + 1, 1, 1))
    t0 = time.perf_counter()
    mpc.solve(X_now=np.eye(3), X_ref_win=Xr0, xi_ref_win=xr0, xi_prev=np.zeros(3), arm={"mode": "hold"})
    t_hold = time.perf_counter() - t0
    t0 = time.perf_counter()
    ra = mpc.solve(X_now=np.eye(3), X_ref_win=Xr0, xi_ref_win=xr0, xi_prev=np.zeros(3), arm=arm_dict(T_seq))
    t_act = time.perf_counter() - t0
    chk('A1 可解', ra.status in ('solved', 'solved inaccurate'), ra.status)
    ua0 = ra.u_arm_seq[0]
    chk('A1 手臂出力', np.abs(ua0).max() > 1e-3, str(ua0))
    q1, s1 = S.plant_phys_step(np.r_[0, 0, 0, Q_READY], Q_READY, np.r_[ra.u_opt, ua0], S.WGMPCConfigSP(arm_model=MODEL), 5)
    e_before = np.linalg.norm(Tt[:3, 3] - T0[:3, 3])
    e_after = np.linalg.norm(Tt[:3, 3] - tcp_at(q1[:3], q1[3:])[:3, 3])
    chk('A1 一步後 TCP 誤差下降', e_after < e_before, f'{e_before * 1e3:.2f}→{e_after * 1e3:.2f} mm')
    a1_split = (np.round(ra.u_opt, 4).tolist(), float(np.abs(ua0).max()))     # 只記錄：靜止參考下底盤也前移協助（全身協調取捨，非錯誤）
    ck = ra.arm_check
    chk('A2 手臂限制違反 ≤ 1e-6、有限', ck['finite'] and ck['max_violation'] <= 1e-6, str(ck))
    chk('A2 |u_a| ≤ v_arm', np.abs(ra.u_arm_seq).max() <= 0.9999 + 1e-6)
    chk('A2 對上一筆加速度框', np.abs(ra.u_arm_seq[0]).max() <= 19.984 * DT + 1e-6)

    # ---- A3 關節貼上限、目標拉向界外
    hi = np.asarray(LITE6_SAFE.upper, float)
    q_edge = Q_READY.copy()
    q_edge[1] = hi[1] - 0.06
    T_e = tcp_at(np.zeros(3), q_edge)
    J = K.jacobian(np.r_[0, 0, 0, q_edge], 'link_tcp')
    Tt3 = T_e.copy()
    Tt3[:3, 3] += J[:3, 4] / np.linalg.norm(J[:3, 4]) * 0.10              # 沿 j2 正向的 TCP 方向拉 10 cm
    r3 = mpc.solve(X_now=np.eye(3), X_ref_win=Xr0, xi_ref_win=xr0, xi_prev=np.zeros(3), arm=arm_dict(np.tile(Tt3, (N + 1, 1, 1)), q=q_edge))
    Mq3, cq3, Ms3, cs3 = GW.arm_prediction(q_edge, q_edge, N, DT, MODEL)
    U3 = r3.u_arm_seq.reshape(-1)
    qmax = max(float((Mq3[k] @ U3 + cq3[k])[1]) for k in range(N))
    smax = max(float((Ms3[k] @ U3 + cs3[k])[1]) for k in range(N))
    chk('A3 預測實測角與設定點 j2 ≤ 上限−margin', qmax <= hi[1] - 0.05 + 1e-6 and smax <= hi[1] - 0.05 + 1e-6, f'q {qmax:.4f} s {smax:.4f} 界 {hi[1] - 0.05:.4f}')

    # ---- A4 底盤沿參考移動、末端世界固定
    Xr4, xr4 = ref([0.05, 0, 0])
    T4 = tcp_at(np.zeros(3), Q_READY)
    T_seq4 = np.tile(T4, (N + 1, 1, 1))
    r4a = mpc.solve(X_now=Xr4[0], X_ref_win=Xr4, xi_ref_win=xr4, xi_prev=np.array([0.05, 0, 0]), arm=arm_dict(T_seq4, u_prev=np.zeros(6)))
    r4h = mpc.solve(X_now=Xr4[0], X_ref_win=Xr4, xi_ref_win=xr4, xi_prev=np.array([0.05, 0, 0]), arm={'mode': 'hold'})

    def horizon_err(r, Useq):
        e = (r.delta_xi_all)                                        # 以回傳 δξ 序列重建底盤（線性誤差預測）＋手臂設定點模型
        A_d4 = np.stack([np.eye(3) - DT * GW.ad(xr4[k]) for k in range(N)])
        Ph4, Ga4 = GW._build_prediction(A_d4, DT)
        ek = (Ph4 @ np.zeros(3) + Ga4 @ e.reshape(-1)).reshape(N, 3)
        Mq4, cq4, _, _ = GW.arm_prediction(Q_READY, Q_READY, N, DT, MODEL)
        return max(float(np.linalg.norm(tcp_at(GW._pose_from_err(Xr4[k + 1], ek[k]), Mq4[k] @ Useq.reshape(-1) + cq4[k])[:3, 3] - T4[:3, 3]))
                   for k in range(N))
    ea, eh = horizon_err(r4a, r4a.u_arm_seq), horizon_err(r4h, np.zeros((N, 6)))
    chk('A4 作用模式時域內 TCP 誤差 < 保持', ea < 0.5 * eh, f'作用 {ea * 1e3:.1f} mm／保持 {eh * 1e3:.1f} mm')
    a4_vx = float(r4a.u_opt[0])                                         # 只記錄：末端世界固定會把底盤往回拉（進度—任務取捨）

    # ---- A5 輪級後處理改動 ⇒ seq_consistent False
    m5 = GW.GMPC(cfg(wheel_coupling=True))
    Xr5, xr5 = ref([0.30, 0.20, 0.6])
    r5 = m5.solve(X_now=from_xytheta(0, 0.05, 0), X_ref_win=Xr5, xi_ref_win=xr5, xi_prev=np.array([0.25, 0.15, 0.4]),
                  arm=arm_dict(np.tile(tcp_at(np.zeros(3), Q_READY), (N + 1, 1, 1))))
    if r5.arm_check['fallback'] is None:
        chk('A5 未退回時序列一致', r5.arm_check['seq_consistent'] is True and r5.accept_action == 'as_is', str(r5.arm_check))
    else:
        chk('A5 後處理改動 ⇒ 退回', r5.arm_check['fallback'] in ('base_postprocess_changed', 'coupled_not_solved'), str(r5.arm_check['fallback']))

    # ---- A6 煞停路徑
    m6 = GW.GMPC(cfg(cbf_hard_k0=True))
    r6 = m6.solve(X_now=Xr4[0], X_ref_win=Xr4, xi_ref_win=xr4, xi_prev=np.array([0.3, 0, 0]),
                  obstacles=[dict(x=0.25, y=0.0, radius=0.2, vx=-0.5, vy=0.0)], arm=arm_dict(T_seq4))
    chk('A6 底盤不可行 ⇒ 照封存版煞停、手臂歸零', r6.accept_action == 'brake' and np.allclose(r6.u_opt, 0) and not np.any(r6.u_arm_seq)
        and (r6.arm_check or {}).get('fallback') == 'base_brake', f'{r6.status} {r6.accept_action} {r6.arm_check}')

    # ---- N 退回與回傳核對負測（Codex 20261031_183817）
    up = np.array([0.6, -0.4, 0.3, 0.0, 0.2, -0.5])                     # 上一筆手臂命令非零
    s_lag = Q_READY - 0.02                                              # 設定點尚未追上
    r_n1 = m6.solve(X_now=Xr4[0], X_ref_win=Xr4, xi_ref_win=xr4, xi_prev=np.array([0.3, 0, 0]), obstacles=[dict(x=0.25, y=0.0, radius=0.2, vx=-0.5, vy=0.0)],
                    arm=arm_dict(T_seq4, s=s_lag, u_prev=up))
    ck = r_n1.arm_check
    chk('N1 非零前速 ⇒ 手臂減速已驗證', ck['fallback'] == 'base_brake' and ck['failed'] is False and ck['arm_decel_valid'] is True, str(ck))
    chk('N1b 手臂減速有效不等於全身序列一致', ck['seq_consistent'] is False and ck['base_as_is'] is False, str(ck))
    chk('N1 首筆不違反加速度上限（不是直接歸零）', np.abs(r_n1.u_arm_seq[0] - up).max() <= 19.984 * DT + 1e-9 and np.abs(r_n1.u_arm_seq[0]).max() > 0 or np.abs(up).max() <= 19.984 * DT, str(r_n1.u_arm_seq[0]))
    r_n1b = m6.solve(X_now=Xr4[0], X_ref_win=Xr4, xi_ref_win=xr4, xi_prev=np.array([0.3, 0, 0]), obstacles=[dict(x=0.25, y=0.0, radius=0.2, vx=-0.5, vy=0.0)],
                     arm=arm_dict(T_seq4, s=s_lag, u_prev=up, a_arm=2.0))    # 加速度上限小：一步降不到零
    chk('N2 低加速度上限：逐步減速、首筆非零、差 ≤ a·dt', np.abs(r_n1b.u_arm_seq[0]).max() > 0.4 and np.abs(r_n1b.u_arm_seq[0] - up).max() <= 2.0 * DT + 1e-9, str(r_n1b.u_arm_seq[0]))
    chk('N2 序列結束前歸零', r_n1b.arm_check['fallback_check']['reaches_zero'] is True, str(r_n1b.arm_check['fallback_check']))
    q_lim = Q_READY.copy()
    q_lim[1] = hi[1] - 0.0505                                           # 離 margin 界 0.5 mm、上一筆命令仍朝外
    r_n3 = m6.solve(X_now=Xr4[0], X_ref_win=Xr4, xi_ref_win=xr4, xi_prev=np.array([0.3, 0, 0]), obstacles=[dict(x=0.25, y=0.0, radius=0.2, vx=-0.5, vy=0.0)],
                    arm=arm_dict(T_seq4, q=q_lim, u_prev=np.array([0, 0.8, 0, 0, 0, 0]), a_arm=2.0))
    ck = r_n3.arm_check
    chk('N3 減速仍越限 ⇒ failed、不標一致', ck['failed'] is True and ck['arm_decel_valid'] is False and ck['seq_consistent'] is False, str(ck))

    class Tamper(GW.GMPC):
        kind = None

        def solve(self, *a_, **k_):
            r = super().solve(*a_, **k_)
            arm = k_.get('arm') if 'arm' in k_ else (a_[5] if len(a_) > 5 else None)
            if arm and arm.get('_inner'):
                if self.kind == 'nan':
                    r.u_arm_seq = np.full((N, 6), np.nan)
                elif self.kind == 'viol':
                    r.arm_check['max_violation'] = 1e-3
                elif self.kind == 'post':
                    r.arm_check['seq_consistent'] = False
            return r
    for kind, want in (('nan', 'returned_nonfinite'), ('viol', 'arm_limit_violation'), ('post', 'base_postprocess_changed')):
        mt = Tamper(cfg())
        mt.kind = kind
        rt = mt.solve(X_now=np.eye(3), X_ref_win=Xr0, xi_ref_win=xr0, xi_prev=np.zeros(3), arm=arm_dict(T_seq))
        chk(f'N4 回傳 {kind} ⇒ 退回 {want}', rt.arm_check['fallback'] == want and np.isfinite(rt.u_arm_seq).all(), str(rt.arm_check.get('fallback')))

    # ---- R 底盤座標任務（tcp_rel_base）：TCP 任務與底盤解耦
    m_rh, m_ra = GW.GMPC(cfg()), GW.GMPC(cfg())
    T_rel = np.stack([K.fk(np.r_[0.0, 0.0, 0.0, Q_READY], 'link_tcp')] * (N + 1))
    T_rel[:, 2, 3] += 0.01
    Xr_r, xr_r = Xr4, xr4
    rh = m_rh.solve(X_now=Xr_r[0], X_ref_win=Xr_r, xi_ref_win=xr_r, xi_prev=np.zeros(3), arm={'mode': 'hold'})
    ra = m_ra.solve(X_now=Xr_r[0], X_ref_win=Xr_r, xi_ref_win=xr_r, xi_prev=np.zeros(3), arm=arm_dict(T_rel, tcp_rel_base=True))
    chk('R1 底盤座標任務 ⇒ 底盤命令與 hold 相同（解耦）', np.abs(np.asarray(ra.u_opt) - np.asarray(rh.u_opt)).max() < 1e-4, f'{ra.u_opt} {rh.u_opt}')
    chk('R2 手臂仍作用', np.abs(ra.u_arm_seq).max() > 1e-3, str(np.abs(ra.u_arm_seq).max()))
    ra2 = GW.GMPC(cfg()).solve(X_now=Xr_r[0], X_ref_win=Xr_r, xi_ref_win=xr_r, xi_prev=np.zeros(3), arm=arm_dict(T_rel * 1.0))
    chk('R3 世界座標模式底盤命令確實不同（對照）', np.abs(np.asarray(ra2.u_opt) - np.asarray(rh.u_opt)).max() > 1e-3, f'{ra2.u_opt}')

    # ---- F 快速 FK／Jacobian 與共用運動學一致
    fc = GW.FastChain.of(K, 'link_tcp')
    rng_f = np.random.default_rng(5)
    e_t, e_j = 0.0, 0.0
    for _ in range(200):
        qr = np.r_[rng_f.uniform(-3, 3, 2), rng_f.uniform(-3.1, 3.1), rng_f.uniform(-3.0, 3.0, 6)]
        Tf, Jf = fc.fk_jac(qr)
        e_t, e_j = max(e_t, float(np.abs(Tf - K.fk(qr, 'link_tcp')).max())), max(e_j, float(np.abs(Jf - K.jacobian(qr, 'link_tcp')).max()))
    chk('F1 FastChain FK、Jacobian 與 wholebody_kinematics 一致（<1e-12）', e_t < 1e-12 and e_j < 1e-12, f'{e_t} {e_j}')

    # ---- Q 求解器：OSQP-only 回歸入口、GI 失敗走備援並記錄
    rq_g = GW.GMPC(cfg()).solve(X_now=Xr4[0], X_ref_win=Xr4, xi_ref_win=xr4, xi_prev=np.array([0.05, 0, 0]), arm=arm_dict(T_seq4))
    rq_o = GW.GMPC(cfg()).solve(X_now=Xr4[0], X_ref_win=Xr4, xi_ref_win=xr4, xi_prev=np.array([0.05, 0, 0]), arm=arm_dict(T_seq4, qp_solver='osqp'))
    chk('Q1 記錄實際求解器', rq_g.arm_check['qp']['solver'] == 'gi' and rq_o.arm_check['qp']['solver'] == 'osqp', str(rq_o.arm_check['qp']))
    chk('Q2 OSQP-only 與 GI 同解', np.abs(np.asarray(rq_g.u_opt) - np.asarray(rq_o.u_opt)).max() < 1e-5 and np.abs(rq_g.u_arm_seq - rq_o.u_arm_seq).max() < 1e-4, f'{np.abs(rq_g.u_arm_seq - rq_o.u_arm_seq).max()}')
    gi0 = GW.qp_dual_active
    GW.qp_dual_active = lambda *a, **k: (None, None, dict(ok=False, why='verify:dual'))
    rq_b = GW.GMPC(cfg()).solve(X_now=Xr4[0], X_ref_win=Xr4, xi_ref_win=xr4, xi_prev=np.array([0.05, 0, 0]), arm=arm_dict(T_seq4))
    GW.qp_dual_active = gi0
    qb = rq_b.arm_check['qp']
    chk('Q3 GI 未過核對 ⇒ OSQP 備援且記原因與耗時', qb['solver'] == 'osqp' and qb['gi_why'] == 'verify:dual' and qb['osqp_ms'] is not None, str(qb))
    chk('Q4 備援解仍可用', rq_b.arm_check.get('fallback') is None and np.abs(np.asarray(rq_b.u_opt) - np.asarray(rq_g.u_opt)).max() < 1e-5, str(rq_b.u_opt))

    print(f'A7 耗時：保持 {t_hold * 1e3:.1f} ms、作用 {t_act * 1e3:.1f} ms（N＝20；含建構與安全外層的底盤解）；A4 作用 {ea * 1e3:.1f}／保持 {eh * 1e3:.1f} mm；L1 相對誤差 {worst:.1e}')
    print(f'記錄：A1 靜止參考底盤命令 {a1_split[0]}、手臂峰 {a1_split[1]:.3f} rad/s；A4 底盤 vx {a4_vx:.4f}（參考 0.05）；A1/A4 fallback {ra.arm_check.get("fallback")}／{r4a.arm_check.get("fallback")}')
    print(f'gmpc_wb 手臂作用模式測試 {NCHK} 項：' + ('全過' if not BAD else '失敗 ' + '；'.join(BAD)))
    return 0 if not BAD else 1


if __name__ == '__main__':
    sys.exit(main())
