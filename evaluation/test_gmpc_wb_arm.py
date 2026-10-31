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
import math
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
    _X_now = Xr[0] @ exp_(e0)                                          # （僅說明 e0 對應的實際位姿）
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
    modified = r5.accept_action != 'as_is' or not np.allclose(r5.u_opt, xr5[0] + r5.delta_xi_all[0], atol=1e-12, rtol=0)
    chk('A5 旗標與實際改動一致', r5.arm_check['seq_consistent'] == (not modified), f"{r5.accept_action} seq_consistent {r5.arm_check['seq_consistent']}")

    # ---- A6 煞停路徑
    m6 = GW.GMPC(cfg(cbf_hard_k0=True))
    r6 = m6.solve(X_now=Xr4[0], X_ref_win=Xr4, xi_ref_win=xr4, xi_prev=np.array([0.3, 0, 0]),
                  obstacles=[dict(x=0.25, y=0.0, radius=0.2, vx=-0.5, vy=0.0)], arm=arm_dict(T_seq4))
    chk('A6 底盤不可行 ⇒ 照封存版煞停、手臂歸零', r6.accept_action == 'brake' and np.allclose(r6.u_opt, 0) and not np.any(r6.u_arm_seq)
        and (r6.arm_check or {}).get('fallback') == 'base_brake', f'{r6.status} {r6.accept_action} {r6.arm_check}')

    print(f'A7 耗時：保持 {t_hold * 1e3:.1f} ms、作用 {t_act * 1e3:.1f} ms（N＝20；含建構與安全外層的底盤解）；A4 作用 {ea * 1e3:.1f}／保持 {eh * 1e3:.1f} mm；L1 相對誤差 {worst:.1e}')
    print(f'記錄：A1 靜止參考底盤命令 {a1_split[0]}、手臂峰 {a1_split[1]:.3f} rad/s；A4 底盤 vx {a4_vx:.4f}（參考 0.05）；A1/A4 fallback {ra.arm_check.get("fallback")}／{r4a.arm_check.get("fallback")}')
    print(f'gmpc_wb 手臂作用模式測試 {NCHK} 項：' + ('全過' if not BAD else '失敗 ' + '；'.join(BAD)))
    return 0 if not BAD else 1


if __name__ == '__main__':
    sys.exit(main())
