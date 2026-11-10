#!/usr/bin/env python3
"""TPA v3：手臂軌跡相位改以底盤實測進度參數化＋切換一致性閘門（Codex 20261111_032519 第 2 項；版本化子類，基線＝bottle_w1_tpa_cj_ctrl）。
問題（bottle_motm_tpa_A_1）：v2 依時間走完手臂軌跡，18.7 s 在 t_pre 切世界座標 PREGRASP 時底盤被行人拖慢、仍在 x≈−0.77（應在 x_pre −0.32），
  切換後 TCP 固定在世界、底盤爬行進不了側向管，目標逾時。切換兩側（相對底盤參考 vs 世界固定 T_pre）位置／速度沒有一致性保證。
做法：
  1 名目進度剖面：承接時以同一底盤剖面（v_nav'、v_c'、A_SLEW）模擬 σ_nom(τ)，τ∈[0, T]（到 x_pre 為止）。
  2 相位更新（tau_step，每控制拍）：τ_σ＝σ_nom⁻¹(σ_實測)（實測進度 sig_robot；到 x_pre 後 τ_σ＝T，手臂依時間收尾、底盤由 S1 規則在 x_pre 減速等）；
    τ ← min(τ＋Δt, max(τ, τ_σ))：只進不退、速率 ≤ 計畫（手臂不比計畫快）；底盤慢／停 ⇒ 手臂跟著慢／停。速率估計 rate（一階濾波）供時域預覽。
    手臂參考＝q(τ)；預覽 τ(t＋kΔ)＝τ＋rate·(t＋kΔ−t)。關節速度＝q'(τ)·rate，起點 rate≈1（名目剖面起點速度＝承接時底盤參考速度）。
  3 切換閘門（switch_check）：τ 到 T 時，以命令生效時刻預測構型 qp（_stop_rest：在途命令走完）在同一世界座標比較
    相對側 Tb(qp)⊗FK(q(T)) 與世界側 T_pre：位置差 ≤ SW_P、姿態差 ≤ SW_R；相對側 TCP 世界速度（底盤 u_last＋手臂 q̇(T)·rate）≤ SW_V（世界側為 0）。
    通過 ⇒ 切 blending、st.t_pre＝t（step_phase 進 PREGRASP）。不通過 ⇒ 有界等待：手臂停在 q(T)（rate＝0）、底盤參考減速，最多 SW_WAIT 秒內重核；
    逾時仍不一致 ⇒ 受控停止 tpa_switch_inconsistent（不直接切換再靠追蹤拉回）。
  st.t_pre 在承接時設為無限大（不再依時間進 PREGRASP）。
"""
import math

import numpy as np

import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_w1_tpa_cj_ctrl as CJ
import bottle_w1_tpa_ctrl as TPA

SW_P, SW_R, SW_V, SW_WAIT = 0.010, math.radians(2.0), 0.03, 1.0
RATE_LP = 0.5


def progress_profile(sig, x_pre, x_slow, v_nav, v_c, v0, T, dt=0.02):
    """同 bottle_w1_tpa_ctrl.base_time 的剖面；回 (τ 陣列, σ 陣列)，σ 嚴格遞增直到 x_pre（或 τ＝T）。"""
    t, x, v, a = 0.0, float(sig), max(float(v0), 0.0), S1.A_SLEW
    ts, xs = [0.0], [x]
    while x < x_pre and t < T + 1e-9 and len(ts) < 20000:
        vt = v_nav if x < x_slow else v_c
        v = min(vt, v + a * dt) if v < vt else max(vt, v - a * dt)
        x += max(v, 1e-6) * dt
        t += dt
        ts.append(t)
        xs.append(min(x, x_pre))
    return np.array(ts), np.array(xs)


def tau_step(tau, rate, dt, sig_meas, prof, T):
    """回 (τ_new, rate_new)。prof＝(τ_k, σ_k)。"""
    tk, sk = prof
    if dt <= 0.0:
        return tau, rate
    tau_sig = T if sig_meas >= sk[-1] - 1e-9 else float(np.interp(sig_meas, sk, tk))
    tau_new = min(tau + dt, max(tau, tau_sig), T)
    r = (tau_new - tau) / dt
    return tau_new, (1.0 - RATE_LP) * rate + RATE_LP * r


def switch_check(K, qp, q_end, qd_end, rate, u_base_body, T_pre):
    """回 dict(ok, dp, dr, dv)。qp＝命令生效時刻預測全身構型（底盤取其位姿）；手臂取 q(T)。"""
    from ammr_wholebody_mpc.wgmpc_core import so3_log
    qf = np.r_[np.asarray(qp, float)[:3], np.asarray(q_end, float)]
    T_rel_w = K.fk(qf, 'link_tcp')
    dp = float(np.linalg.norm(T_rel_w[:3, 3] - T_pre[:3, 3]))
    dr = float(np.linalg.norm(so3_log(T_rel_w[:3, :3].T @ T_pre[:3, :3])))
    c_, s_ = math.cos(qf[2]), math.sin(qf[2])
    ub = np.asarray(u_base_body, float)
    vb = np.r_[c_ * ub[0] - s_ * ub[1], s_ * ub[0] + c_ * ub[1], ub[2]]
    J = K.jacobian(qf, 'link_tcp')
    dv = float(np.linalg.norm((J @ np.r_[vb, np.asarray(qd_end, float) * rate])[:3]))
    return dict(ok=bool(dp <= SW_P and dr <= SW_R and dv <= SW_V), dp=round(dp, 5), dr=round(dr, 5), dv=round(dv, 4))


class W1PlaceController(CJ.W1PlaceController):
    group = CJ.W1PlaceController.group + '_TPAS'

    def _tpa_start(self, q):
        super()._tpa_start(q)
        P = self.tpa
        if P is None:
            return
        sig_e = float(getattr(self, 'sig_robot', q[0])) + float(self.v_b) * (P['t_e'] - float(self.t))
        prof = progress_profile(sig_e, float(self.x_pre), float(self.x_slow), P['v_nav_eff'], P['v_c_eff'], float(self.v_b), P['T'])
        P.update(prof=prof, tau=0.0, rate=1.0, t_last=P['t_e'], wait_t0=None, sw=None, n_wait=0, tau_log=[])
        self.st.t_pre = float('inf')                                     # 不再依時間進 PREGRASP；由切換閘門決定
        self._ev('tpas_profile', n=len(prof[0]), sig0=round(float(prof[1][0]), 4), sig_end=round(float(prof[1][-1]), 4), tau_end=round(float(prof[0][-1]), 3))

    def _tau_at(self, tau_t):
        P = self.tpa
        return P['tau'] + P['rate'] * (tau_t - float(self.t))

    def _rel_target(self, tau_t):
        P = self.tpa
        if P is None or P['switched']:
            return super()._rel_target(tau_t)
        tau = max(self._tau_at(tau_t), 0.0)
        if tau <= P['T'] or P['wait_t0'] is not None:
            return self.K.fk(np.r_[0.0, 0.0, 0.0, TPA.traj_eval(P['P'], min(tau, P['T']))], 'link_tcp')
        be = P['base_end'] + np.array([float(P['v_c_eff']) * (tau - P['T']), 0.0, 0.0])   # 預覽超過 T：世界固定 T_pre 換到沿車道前進的底盤座標
        c_, s_ = math.cos(be[2]), math.sin(be[2])
        Tb = np.eye(4)
        Tb[:3, :3] = [[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]
        Tb[:3, 3] = [be[0], be[1], 0.0]
        return np.linalg.inv(Tb) @ P['T_pre']

    def _advance_inner(self, T_tcp, close_done, q):
        P = self.tpa
        if P is not None and not P['switched'] and self.st.phase == 'CRUISE' and self.judge.stop is None and float(self.t) >= P['t_e'] - 1e-9:
            t = float(self.t)
            if P['wait_t0'] is None:
                P['tau'], P['rate'] = tau_step(P['tau'], P['rate'], t - P['t_last'], float(self.sig_robot), P['prof'], P['T'])
            P['t_last'] = t
            if len(P['tau_log']) < 2000:
                P['tau_log'].append((round(t, 3), round(float(self.sig_robot), 4), round(P['tau'], 4), round(P['rate'], 4)))
            if P['tau'] >= P['T'] - 1e-9:
                qp, _, _ = self._stop_rest(np.asarray(q, float))
                rate = 0.0 if P['wait_t0'] is not None else P['rate']
                sw = switch_check(self.K, qp, P['q1'], P['qd1'], rate, np.asarray(self.u_last, float)[:3], P['T_pre'])
                P['sw'] = dict(sw, t=round(t, 3), x=round(float(q[0]), 4), y=round(float(q[1]), 4), rate=round(rate, 4))
                if sw['ok']:
                    P['switched'] = True
                    self.blending = True
                    self.st.t_pre = t
                    self._ev('tpa_switch', **P['sw'], waited=None if P['wait_t0'] is None else round(t - P['wait_t0'], 3))
                elif P['wait_t0'] is None:
                    P['wait_t0'], P['rate'] = t, 0.0
                    self._ev('tpa_switch_wait', **P['sw'])
                elif t - P['wait_t0'] > SW_WAIT:
                    self.judge._stop(self.t, 'tpa_switch_inconsistent', **P['sw'])
                else:
                    P['n_wait'] += 1
        super()._advance_inner(T_tcp, close_done, q)

    def _v_target(self, T_tcp, x):
        P = self.tpa
        if P is not None and not P['switched'] and P.get('wait_t0') is not None and self.st.phase == 'CRUISE':
            return max(self.v_b - B.A_REF * 2.0 * B.DT, 0.0)            # 切換等待：底盤參考依斜率減速
        return super()._v_target(T_tcp, x)

    def report(self):
        r = super().report()
        P = self.tpa
        if P is not None:
            r['tpas'] = dict(sw=P.get('sw'), n_wait=P.get('n_wait'), wait_t0=P.get('wait_t0'), tau_log=P.get('tau_log', [])[::2],
                             gate=dict(sw_p=SW_P, sw_r=SW_R, sw_v=SW_V, sw_wait=SW_WAIT, rate_lp=RATE_LP),
                             note='手臂相位隨底盤實測進度；τ 只進不退、速率 ≤ 計畫；τ 到 T 以命令生效預測構型核切換兩側一致才切，否則有界等待後拒絕')
            r['tpa'] = {k: v for k, v in r['tpa'].items() if k not in ('prof', 'tau_log')}
        return r
