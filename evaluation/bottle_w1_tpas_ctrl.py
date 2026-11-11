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
  6 obs_nom_1 修正（2026-11-11）：相位進度 σ 改由本類每拍投影實測底盤位置求得（_sig_meas，只進不退），不再用過車道入口即凍結的 sig_robot；
    父類 TPA 的時間切換停用（P['time_switch']＝False），切換只經 switch_check；τ 遲遲不到 T 由既有期限受控停止（不新增期限）。
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
R_MAX, RDOT_UP = 1.2, 1.0                                       # 有界相位追趕（Codex 20261111_142637 (A)）：τ 落後 τ_σ 時 rate 可超過 1，上限 1.2（唯一候選，不掃）；
                                                                #   rate > 1 的部分每秒最多增加 1.0（≥ 0.2 s 才到 1.2）；rate ≤ 1 的行為不變（同前：跟底盤進度、可即時降）


def progress_profile(sig, x_pre, x_slow, v_nav, v_c, v0, T, dt=0.02, scale_to_T=True):
    """同 bottle_w1_tpa_ctrl.base_time 的剖面；回 (τ 陣列, σ 陣列)，σ 嚴格遞增直到 x_pre（或 τ＝T）。
    scale_to_T（終點修正）：末步按剩餘距離取部分時間，且剖面在 T 前到 x_pre 時 τ 軸等比縮放使 x_pre 處 τ＝T（obs_nom_2：離散剖面 τ_end 10.38、
    T 10.40，且末步 0.77 mm 記滿 0.02 s ⇒ σ 到 x_pre 前後 τ_σ 不連續，τ 追不及、S1 已減速 ⇒ 切換速度差被拒）。False＝舊行為（對照用）。"""
    t, x, v, a = 0.0, float(sig), max(float(v0), 0.0), S1.A_SLEW
    ts, xs = [0.0], [x]
    while x < x_pre and t < T + 1e-9 and len(ts) < 20000:
        vt = v_nav if x < x_slow else v_c
        v = min(vt, v + a * dt) if v < vt else max(vt, v - a * dt)
        if scale_to_T and x + max(v, 1e-6) * dt > x_pre:               # 末步按剩餘距離取部分時間（原碼末步截在 x_pre 仍記滿 Δt ⇒ 終點斜率失真）
            t += (x_pre - x) / max(v, 1e-6)
            x = x_pre
            ts.append(t)
            xs.append(x)
            break
        x += max(v, 1e-6) * dt
        t += dt
        ts.append(t)
        xs.append(min(x, x_pre))
    ts, xs = np.array(ts), np.array(xs)
    if scale_to_T and xs[-1] >= x_pre - 1e-9 and ts[-1] > 1e-9 and abs(ts[-1] - T) > 1e-9:
        ts = ts * (float(T) / ts[-1])
    return ts, xs


V_MARGIN = 0.98                                                 # 追趕上限的關節速度裕度（右側整合離線：右看起點＋末段快 7% 時 1.2 倍追趕使 q̇ 達 1.062 v_arm）


def rate_cap(Ptraj, tau, T, v_arm, look=0.3):
    """追趕上限（Codex 142637：每拍仍須 q̇＝q_τ·r ≤ v_arm）：min(R_MAX, max(1, V_MARGIN·v_arm／max|q′(τ…τ＋look)|))。
    計畫本身在 rate 1 已滿足 v_arm（plan_joint），故下限 1 不放寬既有界；控制器與離線核對共用。"""
    t1 = min(float(tau) + look, float(T))
    qd = max(float(np.max(np.abs(TPA.traj_eval(Ptraj, s, 1)))) for s in np.linspace(float(tau), max(t1, float(tau)), 4))
    if qd <= 1e-9:
        return float(R_MAX)
    return float(min(R_MAX, max(1.0, V_MARGIN * float(v_arm) / qd)))


def tau_step(tau, rate, dt, sig_meas, prof, T, r_max=R_MAX, rdot_up=RDOT_UP):
    """回 (τ_new, rate_new)。prof＝(τ_k, σ_k)。
    τ_σ＝σ_nom⁻¹(σ_實測)；σ 超過剖面終點時以剖面終端速度延伸到 T 之後（τ_σ＝T＋(σ−σ_end)/v_end），使最後一拍的 rate 仍是實際進度速率。
    未截斷增量＝min(r_cap·Δt, max(0, τ_σ−τ))（只進不退）；r_cap＝min(r_max, max(1, rate＋rdot_up·Δt))：τ 落後時可有界追趕（obs_nom_2：末段底盤快 7%、
    τ 速率上限 1 ⇒ 落後 0.19 s，底盤到 x_pre 減速等手臂、手臂照原末速收尾 ⇒ 切換速度差 0.061 被拒）；rate＝未截斷增量／Δt（RATE_LP＝1 ⇒ 不濾波）；送出的 τ 截在 T。
    r_max＝1、rdot_up＝0 即舊行為（速率 ≤ 計畫）。預覽（_tau_at）與切換閘門都用同一 rate。"""
    tk, sk = prof
    if dt <= 0.0:
        return tau, rate
    if sig_meas < sk[-1] - 1e-9:
        tau_sig = float(np.interp(sig_meas, sk, tk))
    else:
        j = -2 if len(sk) >= 3 else -1                                    # 剖面最後一步在 x_pre 截斷 ⇒ 取最後一個完整區間的斜率當終端速度
        v_end = (sk[j] - sk[j - 1]) / max(tk[j] - tk[j - 1], 1e-9) if len(sk) > 1 else 0.0
        tau_sig = T + (sig_meas - sk[-1]) / v_end if v_end > 1e-4 else T + dt
    r_cap = min(float(r_max), max(1.0, float(rate) + float(rdot_up) * dt))
    inc = min(r_cap * dt, max(0.0, tau_sig - tau))
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
        sig0 = self._sig_of(np.asarray(q, float)[:2], float(getattr(self, 'sig_robot', q[0])))   # 同一進度定義（車道段解析 σ＝x，其前取樣投影）
        sig_e = sig0 + float(self.v_b) * (P['t_e'] - float(self.t))
        prof = self._tpas_prof(sig_e, P)
        P.update(prof=prof, tau=0.0, rate=1.0, t_last=P['t_e'], sw=None, rj=None, tau_log=[], time_switch=False, sig_raw=sig0, sig_phase=sig0, sig_path=self.path,
                 regress_max=0.0, regress_ev=False)
        # time_switch＝False：父類 TPA 依時間 t_e＋T 切換停用（obs_nom_1：τ 未到 T 卻被時間切換，st.t_pre＝∞ ⇒ 永不進 PREGRASP）；切換只經 switch_check
        self.st.t_pre = float('inf')                                     # 不再依時間進 PREGRASP；由切換閘門決定
        self._ev('tpas_profile', n=len(prof[0]), sig0=round(float(prof[1][0]), 4), sig_end=round(float(prof[1][-1]), 4), tau_end=round(float(prof[0][-1]), 3))

    def _tpas_prof(self, sig_e, P):
        """名目進度表 (τ, σ)（hook；預設＝既有梯形剖面；A0 子類回傳與 TPA 時長同一份剖面的取樣）。"""
        return progress_profile(sig_e, float(self.x_pre), float(self.x_slow), P['v_nav_eff'], P['v_c_eff'], float(self.v_b), P['T'])

    def _plan_yaw(self, sig):
        """進度點 σ 的計畫底盤偏航（本版＝th_ref；航向沿路徑的子類覆寫為該進度點的路徑航向）。"""
        return float(self.th_ref)

    def _sig_of(self, xy, hint):
        """進度定義（初始化、相位、偏差補償、紀錄共用）：直線車道段（x ≥ x_lane，路徑點＝(σ, y_lane)）解析 σ＝x；其前以 hint 附近取樣投影（窗 back 0.1／ahead 0.3）。"""
        x = float(xy[0])
        if x >= float(self.path.x_lane):
            return x
        s0 = float(self.path.project(np.asarray(xy[:2], float), float(hint), back=0.1, ahead=0.3))
        lo, hi = s0 - 0.01, s0 + 0.01                                      # 次格點細化（取樣投影 1 cm 量化使 rate 在 0.57／1.0 間跳動、ṙ 達 ±14/s）
        p_ = np.asarray(xy[:2], float)
        for _ in range(14):                                                # 三分搜尋（局部單峰）：剩餘區間 ≈ 0.02·(2/3)^14 ≈ 0.1 mm
            m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
            d1, d2 = (float(np.linalg.norm(self.path.xy(np.array([m]))[0] - p_)) for m in (m1, m2))
            if d1 <= d2:
                hi = m2
            else:
                lo = m1
        return 0.5 * (lo + hi)

    def _sig_meas(self, q):
        """相位用的底盤實測進度（obs_nom_1 修正；Codex 20261111_140507 第 1 項）：每拍由實測底盤位置以 _sig_of 求 σ_raw（不單調，照實記錄）；
        手臂相位用 σ_phase＝max(上一拍 σ_phase, σ_raw)（只進不退只用於相位）。不沿用 sig_robot（只在 s_ref < x_lane 更新，過車道入口即凍結）。
        換路徑版本：σ 依新舊路徑 sig0 差平移（＝tlane 的 dsig0；接段前的舊路徑保留）。實際倒退 σ_phase − σ_raw 記最大值與事件，安全檢查照舊（偏差補償用 σ_raw 鄰域）。"""
        P = self.tpa
        if P.get('sig_path') is not self.path:
            old = P.get('sig_path')
            ds = 0.0 if old is None else float(self.path.sig0) - float(old.sig0)
            P['sig_raw'], P['sig_phase'], P['sig_path'] = float(P['sig_raw']) + ds, float(P['sig_phase']) + ds, self.path
            self._ev('tpas_path_version', ds=round(ds, 4))
        raw = self._sig_of(np.asarray(q, float)[:2], P['sig_raw'])
        P['sig_raw'] = raw
        P['sig_phase'] = max(float(P['sig_phase']), raw)
        reg = float(P['sig_phase']) - raw
        if reg > P['regress_max']:
            P['regress_max'] = reg
        if reg > 0.03 and not P['regress_ev']:
            P['regress_ev'] = True
            self._ev('tpas_base_regress', sig_raw=round(raw, 4), sig_phase=round(float(P['sig_phase']), 4))   # 實測倒退：記錄（相位不退），不新增停止規則
        return float(P['sig_phase'])

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
            sm = self._sig_meas(q)
            rc_ = rate_cap(P['P'], P['tau'], P['T'], B.W_ARM['v_arm'])
            P['rcap_min'] = min(P.get('rcap_min', R_MAX), rc_)
            P['tau'], P['rate'] = tau_step(P['tau'], P['rate'], t - P['t_last'], sm, P['prof'], P['T'], r_max=rc_)
            P['t_last'] = t
            qp, _, _ = self._stop_rest(np.asarray(q, float))
            sp = self._sig_of(qp[:2], P['sig_raw'])                       # 同一進度定義（不再以凍結的 sig_robot 當提示）
            P['D'], dy, dth = plan_dev(qp[:3], self.path.xy(sp)[0], self._plan_yaw(sp))
            P['dev_max'] = max(P.get('dev_max', 0.0), abs(dy))
            if abs(dy) > DEV_Y_MAX or abs(dth) > DEV_TH_MAX:
                self.judge._stop(self.t, 'tpa_plan_deviation', dy=round(dy, 4), dth=round(dth, 4))
            if len(P['tau_log']) < 2000:
                P['tau_log'].append((round(t, 3), round(sm, 4), round(P['tau'], 4), round(P['rate'], 4), round(float(P['sig_raw']), 4)))   # (t, σ_phase, τ, rate, σ_raw)
            if self.judge.stop is None and P['tau'] >= P['T'] - 1e-9:
                lat = C0.D_STEPS * C0.PHYS
                sw = switch_check(self.K, qp[:3], np.asarray(self.u_last, float)[:3], P, P['tau'] + P['rate'] * lat, P['rate'], P['T_pre'], D=P['D'])
                T_from = sw.pop('T_from')
                P['sw'] = dict(sw, t_sw=round(t, 3), x=round(float(q[0]), 4), y=round(float(q[1]), 4), rate=round(P['rate'], 4))   # 時刻欄名 t_sw：_ev／S0Judge._stop 以 t 為固定參數，原名 t 使接受與拒絕路徑都 TypeError（tpas_progress_counterexample 抓到）
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
            r['tpa'] = {k: v for k, v in r['tpa'].items() if k not in ('prof', 'tau_log', 'rj', 'D', 'sig_path')}
            r['tpas'].update(progress=dict(sig_raw_end=round(float(P.get('sig_raw', float('nan'))), 4), sig_phase_end=round(float(P.get('sig_phase', float('nan'))), 4),
                                           regress_max=round(float(P.get('regress_max', 0.0)), 4), time_switch=P.get('time_switch'), r_max=R_MAX, rdot_up=RDOT_UP, rcap_min=P.get('rcap_min'), v_margin=V_MARGIN,
                                           note='σ：車道段解析 σ＝x、其前取樣投影；相位用 σ_phase（只進不退），σ_raw 照實記錄；tau_log 欄＝(t, σ_phase, τ, rate, σ_raw)'))
        return r
