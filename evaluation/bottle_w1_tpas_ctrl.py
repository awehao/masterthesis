#!/usr/bin/env python3
"""TPA v3：手臂軌跡相位改以底盤實測進度參數化＋切換一致性閘門（Codex 20261111_032519 第 2 項；版本化子類，基線＝bottle_w1_tpa_cj_ctrl）。
問題（bottle_motm_tpa_A_1）：v2 依時間走完手臂軌跡，18.7 s 在 t_pre 切世界座標 PREGRASP 時底盤被行人拖慢、仍在 x≈−0.77（應在 x_pre −0.32），
  切換後 TCP 固定在世界、底盤爬行進不了側向管，目標逾時。切換兩側（相對底盤參考 vs 世界固定 T_pre）位置／速度沒有一致性保證。
做法：
  1 名目進度剖面：承接時以同一底盤剖面（v_nav'、v_c'、A_SLEW）模擬 σ_nom(τ)，τ∈[0, T]（到 x_pre 為止）。
  2 相位更新（tau_step，每控制拍）：τ_σ＝σ_nom⁻¹(σ_實測)（實測進度 sig_robot；到 x_pre 後 τ_σ＝T，手臂依時間收尾、底盤由 S1 規則在 x_pre 減速等）；
    τ ← min(τ＋Δt, max(τ, τ_σ))：只進不退、速率 ≤ 計畫（手臂不比計畫快）；底盤慢／停 ⇒ 手臂跟著慢／停。速率估計 rate（一階濾波）供時域預覽。
    手臂參考＝q(τ)；預覽 τ(t＋kΔ)＝τ＋rate·(t＋kΔ−t)。關節速度＝q'(τ)·rate，起點 rate≈1（名目剖面起點速度＝承接時底盤參考速度）。
  3 切換閘門（switch_check）：τ 到 T 時，在同一世界座標、同一命令生效時刻 t_eff 比較兩側：
    相對側＝Tb(qp)⊗rel_ref(τ_eff)（qp＝_stop_rest 的在途命令走完構型之底盤位姿；rel_ref＝控制器實際送出的同一參考函式，τ_eff＝τ＋rate·延遲）；
    世界側＝T_pre（靜止）。位置差 ≤ SW_P、姿態差 ≤ SW_R；相對側 TCP 世界速度（底盤照 u_last、相位照 rate 前推差分）≤ SW_V。
    處置：全部 ≤ (SW_P, SW_R, SW_V) ⇒ 直接切（blending、st.t_pre＝t，step_phase 進 PREGRASP）；
          位置 ≤ SW_P_MAX、姿態 ≤ SW_R_MAX、速度 ≤ SW_V ⇒ 有界重接：切換後 PREGRASP 的世界目標由相對側位姿 T_from 以 min-jerk 在
            t_rj＝max(1.875·dp/V_RJ, 1.875·dr/W_RJ, 0.3) s（> 1.5 s 拒絕）內接到 T_pre（_T_seq；含手腕近距修正後的目標）——位置連續、速度有界；
          其餘 ⇒ 受控停止 tpa_switch_inconsistent（不直接切換再靠追蹤拉回）。
  4 計畫位姿偏差補償（Codex 042529 第 1 項）：每拍以命令生效預測底盤位姿 qp 投影到路徑得進度點，計畫底盤位姿＝(路徑點, 計畫偏航 th_ref)；
    偏差 D（只取橫向 dy、偏航 dθ）從相對參考扣除：送出參考＝D⁻¹⊗rel_ref(τ) ⇒ TCP 世界路徑維持規劃值（手臂吸收底盤橫偏，手指不因橫偏偏向瓶）；
    |dy| > DEV_Y_MAX 或 |dθ| > DEV_TH_MAX ⇒ 接近段即受控停止 tpa_plan_deviation（不等到切換）。切換閘門用同一 D。
  5 rate 不濾波＝本拍實際相位增量／Δt；重接時長依 min-jerk 峰值（1.875×平均）使峰值 ≤ V_RJ／W_RJ，超過 T_RJ_MAX 即拒絕（不截短）。
  st.t_pre 在承接時設為無限大（不再依時間進 PREGRASP）。
"""
import math

import numpy as np

import bottle_ref as BR
import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_w1_tpa_cj_ctrl as CJ
import bottle_w1_tpa_ctrl as TPA

SW_P, SW_R, SW_V = 0.010, math.radians(2.0), 0.03            # 直接切換
SW_P_MAX, SW_R_MAX = 0.040, math.radians(5.0)                 # 有界重接上限；超過 ⇒ 受控拒絕
V_RJ, W_RJ, T_RJ_MIN, T_RJ_MAX = 0.03, math.radians(10.0), 0.3, 1.5   # 重接：TCP 世界目標以 ≤ 3 cm/s、≤ 10°/s 由相對側位姿 min-jerk 內插到 T_pre
RATE_LP = 1.0                                                   # 不濾波：rate＝本拍實際相位增量／Δt（Codex 042529：預覽與實際送出參考一致）
MJ_PEAK = 1.875                                                 # min-jerk 峰值速度／平均速度
DEV_Y_MAX, DEV_TH_MAX = 0.08, math.radians(8.0)                 # 底盤相對計畫底盤位姿的橫向／偏航偏差上限（超過 ⇒ 接近段即受控拒絕）


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
    """回 (τ_new, rate_new)。prof＝(τ_k, σ_k)。
    τ_σ＝σ_nom⁻¹(σ_實測)；σ 超過剖面終點時以剖面終端速度延伸到 T 之後（τ_σ＝T＋(σ−σ_end)/v_end），使最後一拍的 rate 仍是實際進度速率。
    未截斷增量＝min(Δt, max(0, τ_σ−τ))（只進不退、速率 ≤ 計畫）；rate＝未截斷增量／Δt（RATE_LP＝1 ⇒ 不濾波）；送出的 τ 截在 T。"""
    tk, sk = prof
    if dt <= 0.0:
        return tau, rate
    if sig_meas < sk[-1] - 1e-9:
        tau_sig = float(np.interp(sig_meas, sk, tk))
    else:
        j = -2 if len(sk) >= 3 else -1                                    # 剖面最後一步在 x_pre 截斷 ⇒ 取最後一個完整區間的斜率當終端速度
        v_end = (sk[j] - sk[j - 1]) / max(tk[j] - tk[j - 1], 1e-9) if len(sk) > 1 else 0.0
        tau_sig = T + (sig_meas - sk[-1]) / v_end if v_end > 1e-4 else T + dt
    inc = min(dt, max(0.0, tau_sig - tau))
    r = inc / dt
    return min(tau + inc, T), (1.0 - RATE_LP) * rate + RATE_LP * r


def _tb(b):
    c_, s_ = math.cos(b[2]), math.sin(b[2])
    T = np.eye(4)
    T[:3, :3] = [[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]
    T[:3, 3] = [b[0], b[1], 0.0]
    return T


def rel_ref(K, P, tau):
    """相位 τ 的相對底盤參考（控制器 _rel_target 與切換閘門共用）：τ ≤ T ⇒ FK(q(τ))；τ > T ⇒ 世界固定 T_pre 換到「沿車道以 v_c' 前進」的底盤座標。"""
    if tau <= P['T']:
        return K.fk(np.r_[0.0, 0.0, 0.0, TPA.traj_eval(P['P'], max(tau, 0.0))], 'link_tcp')
    be = np.asarray(P['base_end'], float) + np.array([float(P['v_c_eff']) * (tau - P['T']), 0.0, 0.0])
    return np.linalg.inv(_tb(be)) @ P['T_pre']


def plan_dev(qp_base, lane_xy, th_ref):
    """底盤（命令生效預測位姿）相對「計畫底盤位姿」（路徑上同一進度點、計畫偏航）的偏差 D（SE2，只取橫向與偏航；縱向由相位 τ 處理）。回 (D 4×4, dy, dθ)。"""
    qb = np.asarray(qp_base, float)[:3]
    Tl = _tb(np.r_[np.asarray(lane_xy, float)[:2], float(th_ref)])
    Dm = np.linalg.inv(Tl) @ _tb(qb)
    dy, dth = float(Dm[1, 3]), float(math.atan2(Dm[1, 0], Dm[0, 0]))
    return _tb(np.r_[0.0, dy, dth]), dy, dth


def switch_check(K, qp_base, u_base_body, P, tau_eff, rate, T_pre, dt=0.02, D=None):
    """同一世界座標、同一命令生效時刻比較切換兩側：相對側 Tb(qp)⊗rel_ref(τ_eff) vs 世界側 T_pre（靜止）。
    速度：相對側 TCP 世界速度以「底盤照 u_last、相位照 rate」前推 dt 的差分。回 dict(ok, dp, dr, dv)。"""
    from ammr_wholebody_mpc.wgmpc_core import so3_log
    qb = np.asarray(qp_base, float)[:3]
    Di = np.eye(4) if D is None else np.linalg.inv(D)                      # 偏差補償：相對參考扣掉底盤相對計畫位姿的橫向／偏航偏差
    Tw0 = _tb(qb) @ Di @ rel_ref(K, P, tau_eff)
    c_, s_ = math.cos(qb[2]), math.sin(qb[2])
    ub = np.asarray(u_base_body, float)
    qb1 = qb + dt * np.r_[c_ * ub[0] - s_ * ub[1], s_ * ub[0] + c_ * ub[1], ub[2]]
    Tw1 = _tb(qb1) @ Di @ rel_ref(K, P, tau_eff + rate * dt)
    dp = float(np.linalg.norm(Tw0[:3, 3] - T_pre[:3, 3]))
    dr = float(np.linalg.norm(so3_log(Tw0[:3, :3].T @ T_pre[:3, :3])))
    dv = float(np.linalg.norm(Tw1[:3, 3] - Tw0[:3, 3]) / dt)
    if dp <= SW_P and dr <= SW_R and dv <= SW_V:
        kind = 'direct'
    elif dp <= SW_P_MAX and dr <= SW_R_MAX and dv <= SW_V:
        kind = 'rejoin'
    else:
        kind = 'reject'
    t_rj = max(MJ_PEAK * dp / V_RJ, MJ_PEAK * dr / W_RJ, T_RJ_MIN) if kind == 'rejoin' else 0.0   # 依 min-jerk 峰值速度定時長（峰值 ≤ V_RJ、W_RJ）
    if kind == 'rejoin' and t_rj > T_RJ_MAX:
        kind = 'reject'                                                     # 超過時長上限不截短，拒絕
    return dict(ok=kind != 'reject', kind=kind, dp=round(dp, 5), dr=round(dr, 5), dv=round(dv, 4), t_rj=round(t_rj if kind == 'rejoin' else 0.0, 3), T_from=Tw0)


class W1PlaceController(CJ.W1PlaceController):
    group = CJ.W1PlaceController.group + '_TPAS'

    def _tpa_start(self, q):
        super()._tpa_start(q)
        P = self.tpa
        if P is None:
            return
        sig_e = float(getattr(self, 'sig_robot', q[0])) + float(self.v_b) * (P['t_e'] - float(self.t))
        prof = progress_profile(sig_e, float(self.x_pre), float(self.x_slow), P['v_nav_eff'], P['v_c_eff'], float(self.v_b), P['T'])
        P.update(prof=prof, tau=0.0, rate=1.0, t_last=P['t_e'], sw=None, rj=None, tau_log=[])
        self.st.t_pre = float('inf')                                     # 不再依時間進 PREGRASP；由切換閘門決定
        self._ev('tpas_profile', n=len(prof[0]), sig0=round(float(prof[1][0]), 4), sig_end=round(float(prof[1][-1]), 4), tau_end=round(float(prof[0][-1]), 3))

    def _tau_at(self, tau_t):
        P = self.tpa
        return P['tau'] + P['rate'] * (tau_t - float(self.t))

    def _rel_target(self, tau_t):
        P = self.tpa
        if P is None or P['switched']:
            return super()._rel_target(tau_t)
        Di = np.eye(4) if P.get('D') is None else np.linalg.inv(P['D'])
        return Di @ rel_ref(self.K, P, max(self._tau_at(tau_t), 0.0))

    def _advance_inner(self, T_tcp, close_done, q):
        P = self.tpa
        if P is not None and not P['switched'] and self.st.phase == 'CRUISE' and self.judge.stop is None and float(self.t) >= P['t_e'] - 1e-9:
            t = float(self.t)
            P['tau'], P['rate'] = tau_step(P['tau'], P['rate'], t - P['t_last'], float(self.sig_robot), P['prof'], P['T'])
            P['t_last'] = t
            qp, _, _ = self._stop_rest(np.asarray(q, float))
            sp = self.path.project(qp[:2], float(self.sig_robot), back=0.1, ahead=0.3)
            P['D'], dy, dth = plan_dev(qp[:3], self.path.xy(sp)[0], float(self.th_ref))
            P['dev_max'] = max(P.get('dev_max', 0.0), abs(dy))
            if abs(dy) > DEV_Y_MAX or abs(dth) > DEV_TH_MAX:
                self.judge._stop(self.t, 'tpa_plan_deviation', dy=round(dy, 4), dth=round(dth, 4))
            if len(P['tau_log']) < 2000:
                P['tau_log'].append((round(t, 3), round(float(self.sig_robot), 4), round(P['tau'], 4), round(P['rate'], 4)))
            if self.judge.stop is None and P['tau'] >= P['T'] - 1e-9:
                lat = C0.D_STEPS * C0.PHYS
                sw = switch_check(self.K, qp[:3], np.asarray(self.u_last, float)[:3], P, P['tau'] + P['rate'] * lat, P['rate'], P['T_pre'], D=P['D'])
                T_from = sw.pop('T_from')
                P['sw'] = dict(sw, t=round(t, 3), x=round(float(q[0]), 4), y=round(float(q[1]), 4), rate=round(P['rate'], 4))
                if sw['ok']:
                    P['switched'] = True
                    self.blending = True
                    self.st.t_pre = t
                    if sw['kind'] == 'rejoin':
                        P['rj'] = dict(t0=t + lat, T=sw['t_rj'], T_from=T_from)
                    self._ev('tpa_switch', **P['sw'])
                else:
                    self.judge._stop(self.t, 'tpa_switch_inconsistent', **P['sw'])
        super()._advance_inner(T_tcp, close_done, q)

    def _T_seq(self, q, t_eff):
        T = super()._T_seq(q, t_eff)
        P = self.tpa
        rj = None if P is None else P.get('rj')
        if rj is not None and self.st.phase == 'PREGRASP' and not self._rel_task():
            out = []
            for k in range(len(T)):
                s_ = (t_eff + k * B.DT - rj['t0']) / rj['T']
                out.append(T[k] if s_ >= 1.0 else BR.interp_pose(rj['T_from'], T[k], BR.min_jerk(max(s_, 0.0))))   # 有界重接：相對側位姿 → 世界目標（含手腕近距修正）
            T = np.stack(out)
        return T

    def report(self):
        r = super().report()
        P = self.tpa
        if P is not None:
            rj = P.get('rj')
            r['tpas'] = dict(sw=P.get('sw'), rejoin=None if rj is None else dict(t0=rj['t0'], T=rj['T']), tau_log=P.get('tau_log', [])[::2],
                             gate=dict(sw_p=SW_P, sw_r=SW_R, sw_v=SW_V, sw_p_max=SW_P_MAX, sw_r_max=SW_R_MAX, v_rj=V_RJ, w_rj=W_RJ, t_rj=[T_RJ_MIN, T_RJ_MAX], rate_lp=RATE_LP, mj_peak=MJ_PEAK, dev_y_max=DEV_Y_MAX, dev_th_max=DEV_TH_MAX), dev_max=P.get('dev_max'),
                             note='手臂相位隨底盤實測進度；τ 只進不退、速率 ≤ 計畫；τ 到 T 以同一參考、同一命令生效時刻核切換兩側：小差直接切、中差有界重接、大差受控拒絕')
            r['tpa'] = {k: v for k, v in r['tpa'].items() if k not in ('prof', 'tau_log', 'rj', 'D')}
        return r
