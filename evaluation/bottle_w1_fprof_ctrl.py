#!/usr/bin/env python3
"""A0 接線（版本化子類，基線＝bottle_w1_side_ctrl；Howard 11/11 流暢度 A 項；Codex 20261111_192902 核可離線接線，**未經 Isaac**）。
導航接近段（CRUISE、未漸變、x < x_pre）的底盤時間參數化改成**一份**輪級可行剖面（bottle_feasible_profile），下列使用點全部讀它：
  1 參考速度 _v_target：v(σ)（取代 v_nav／v_c 梯形與航向層另算的 v_cap）。
  2 MPC 預覽窗 _ref_window：未來每一步的進度 σ_k、速度 v_k、偏航與偏航率都由同一剖面積分取得，不以當下速度固定外推。
  3 航向層轉彎降速：偏航率 W_MAX、偏航角加速度 ALPHA_MAX 已在剖面內，_v_target 不再另取 hd.v_cap。
  4 站位候選的底盤時間成本 _standoff_t_base。
  5 TPA 時長（_tpa_timing）與 TPAS 進度表（_tpas_prof）：同一份最終剖面；手臂需要更久 ⇒ 共同重新定時（降巡航目標速度使底盤到位時間＝手臂時長），
    再由 plan_tpa 既有流程重核終點速度與關節限制；不縮放 TPAS 時間軸。
沒有改：v_nav／v_c／x_slow 數值、加減速 A_SLEW、抓取低速、階段時長、切換閘門、品質與安全判準、GMPC／CBF／Shield、運送與放置段。
剖面失敗（起速無法承接等）⇒ 受控停止 fprof_infeasible，不默默退回舊剖面。輪級參數取 wb_wheel_limit.WheelLimitConfig 預設並記入報告（須與實際配置核對）。"""
import math

import numpy as np
from ammr_wholebody_mpc.se2 import from_xytheta

import bottle_feasible_profile as FP
import bottle_heading as HD
import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_w1_side_ctrl as SC
import wb_wheel_limit as WL

N, DT = B.N, B.DT
DS = 0.01
RETIME_TOL = 0.01                                                # 共同定時的到位時間容許殘差（s；二分法 30 次的數值殘差量級，非放寬）


def build(path, hd, sig0, v0, x_pre, x_slow, v_nav, v_c, sig_end=None):
    """由路徑＋航向參考建 A0 剖面。σ 用路徑參數（＝弧長；以 arc_err 回報偏差）。回 dict(ok, why, sig, v, t, T_pre, pr, P, arc_err)。"""
    s_end = float(x_pre if sig_end is None else sig_end)
    if s_end <= sig0 + 2 * DS:
        return dict(ok=False, why='too_short')
    g = np.arange(float(sig0), s_end + 1e-9, DS)
    if s_end - g[-1] > 1e-6:
        g = np.r_[g, s_end]
    P = FP.path_arrays(np.asarray(path.xy(g), float), hd.yaw(g) if hd is not None else np.zeros(len(g)))
    if len(P['sig']) != len(g):
        return dict(ok=False, why='path_degenerate')
    arc_err = float(np.max(np.abs((P['sig'] + g[0]) - g)))
    vt = np.where(g < x_slow, float(v_nav), float(v_c))
    pr = FP.profile(P, vt, float(v0), S1.A_SLEW, S1.A_SLEW, w_max=HD.W_MAX, alpha_max=HD.ALPHA_MAX)
    if not pr['ok']:
        return dict(ok=False, why=pr['why'], detail={k: v for k, v in pr['detail'].items() if k != 'ok'})
    bad, vf, over = FP.hard_check(pr)
    if bad:                                                                     # 接受閘門：既有硬限制（無容差）；超限 ⇒ 不可行（由呼叫端重新定時或受控停止）
        return dict(ok=False, why='hard_limit:' + ','.join(bad), detail={k: round(float(v), 5) for k, v in vf.items()})
    return dict(ok=True, why=None, sig=g, v=pr['v'], t=pr['t'], T=float(np.interp(float(x_pre), g, pr['t'])), pr=pr, P=P, arc_err=arc_err, v_nav=float(v_nav), v_c=float(v_c),
                x_pre=float(x_pre), x_slow=float(x_slow), entry_over_cap=pr['entry_over_cap'], over_margin=over, verify={k: round(float(v), 5) for k, v in vf.items()})


def v_at(F, sig):
    return float(np.interp(float(sig), F['sig'], F['v']))


def v_next(F, sig, dt):
    """下一拍的參考速度：以**時間軸**查剖面——t(σ) 到 t(σ)＋dt 的平均速度。不能用 v(σ＋v·dt)：靜止起步時 v(σ0)＝0 ⇒ 永遠查到 0、參考卡住
    （fprof_wiring_check r1 首輪抓到）。σ 超過剖面終點 ⇒ 終端速度。"""
    if sig >= F['sig'][-1]:
        return float(F['v'][-1])
    t0 = float(np.interp(float(sig), F['sig'], F['t']))
    t1 = t0 + float(dt)
    over = max(t1 - float(F['t'][-1]), 0.0)
    s1 = float(np.interp(t1, F['t'], F['sig'])) + float(F['v'][-1]) * over
    return (s1 - float(sig)) / float(dt)                                   # 這一拍的平均速度：S1 以 s_ref += v·dt 前進 ⇒ 參考進度逐拍落在剖面上（取拍末瞬時速度會累積超前）


def window(F, path, hd, sig0, v_b, n=N, dt=DT, v_after=None):
    """預覽窗：由 t0＝t(σ_0) 起以時間軸取 σ_k＝σ(t0＋k·dt)、v_k＝v(t0＋k·dt)；v_0 取當下參考速度 v_b。剖面終點之後以 v_after（預設終端速度）等速延伸。
    回 (pose (n+1,3), xi (n+1,3)＝本體 vx, vy, ω, σ (n+1,), v (n+1,))。"""
    v_end = float(F['v'][-1] if v_after is None else v_after)
    s0 = float(sig0)
    if s0 >= F['sig'][-1]:
        tq_over = dt * np.arange(n + 1)
        sg, vv = s0 + v_end * tq_over, np.full(n + 1, v_end)
    else:
        t0 = float(np.interp(s0, F['sig'], F['t']))
        tq = t0 + dt * np.arange(n + 1)
        over = np.maximum(tq - F['t'][-1], 0.0)
        sg = np.interp(tq, F['t'], F['sig']) + v_end * over
        vv = np.where(over > 0, v_end, np.interp(tq, F['t'], F['v']))
    vv = vv.astype(float)
    vv[0] = float(v_b)
    P_ = np.asarray(path.xy(sg), float)
    Tn = np.asarray(path.tangent(sg), float)
    th = hd.yaw(sg) if hd is not None else np.zeros(n + 1)
    w = vv * (hd.dyaw(sg) if hd is not None else 0.0)
    c_, s_ = np.cos(th), np.sin(th)
    vw = vv[:, None] * Tn
    vb = np.column_stack([c_ * vw[:, 0] + s_ * vw[:, 1], -s_ * vw[:, 0] + c_ * vw[:, 1]])
    return np.column_stack([P_, th]), np.column_stack([vb, w]), sg, vv


class Timing:
    """TPA 用的底盤到位時間模型（plan_tpa 的 timing hook）：t_base()＝名目剖面到 x_pre 的時間；retime(T)＝手臂需要 T 時共同重新定時。
    retime 沿用 base_profile 的語義：先降 x_slow 前的巡航目標速度（下限 v_c），仍太早再把巡航與接近一起降（下限 v_floor）；最終剖面存 self.F。"""

    def __init__(self, path, hd, sig, v0, x_pre, x_slow, v_nav, v_c, v_floor=0.02):
        self.a = (path, hd, float(sig), float(v0), float(x_pre), float(x_slow))
        self.v_nav, self.v_c, self.v_floor = float(v_nav), float(v_c), float(v_floor)
        self.F = self._b(self.v_nav, self.v_c)
        self.F0, self.log = self.F, []

    def _b(self, vn, vc):
        path, hd, sig, v0, x_pre, x_slow = self.a
        return build(path, hd, sig, v0, x_pre, x_slow, vn, vc)

    def ok(self):
        return bool(self.F0.get('ok'))

    def t_base(self):
        return float(self.F0['T'])

    def retime(self, T):
        T = float(T)
        F = self.F0
        if F['T'] >= T - 1e-6:
            self.F = F
            self.log.append(dict(T=T, v_nav=self.v_nav, v_c=self.v_c, t=F['T'], kind='nominal'))
            return self.v_nav, self.v_c, float(F['T'])

        def bis(fn, lo, hi):
            Fb = None
            for _ in range(30):
                m = 0.5 * (lo + hi)
                Fm = fn(m)
                if not Fm.get('ok'):
                    lo = m                                                 # 太慢反而承接不了起速 ⇒ 往快的一側找
                    continue
                if Fm['T'] > T:
                    lo = m
                else:
                    hi, Fb = m, Fm
            return hi, Fb
        Fc = self._b(self.v_c, self.v_c)
        if Fc.get('ok') and Fc['T'] >= T:
            vn, Fb = bis(lambda m: self._b(m, self.v_c), self.v_c, self.v_nav)
            vc = self.v_c
        else:
            Ff = self._b(self.v_floor, self.v_floor)
            if Ff.get('ok') and Ff['T'] < T:                                   # 連最低速度剖面都比手臂早到 ⇒ 無法共同定時：明確拒絕（不縮放時間軸、不退回舊剖面）
                self.log.append(dict(T=T, kind='retime_cannot_meet_T', t_slowest=round(float(Ff['T']), 4), v_floor=self.v_floor))
                self.F = dict(ok=False, why='retime_cannot_meet_T')
                return self.v_nav, self.v_c, float(F['T'])
            else:
                vn, Fb = bis(lambda m: self._b(m, m), self.v_floor, self.v_c)
                vc = vn
        if Fb is None:
            self.log.append(dict(T=T, kind='retime_failed'))
            self.F = dict(ok=False, why='retime_failed')
            return self.v_nav, self.v_c, float(F['T'])
        if abs(float(Fb['T']) - T) > RETIME_TOL:                                # 二分法沒有收斂到 T（例：可行區間不含 T）⇒ 拒絕
            self.log.append(dict(T=T, kind='retime_not_converged', t=round(float(Fb['T']), 4)))
            self.F = dict(ok=False, why='retime_not_converged')
            return self.v_nav, self.v_c, float(F['T'])
        self.F = Fb
        self.log.append(dict(T=T, v_nav=round(vn, 4), v_c=round(vc, 4), t=round(Fb['T'], 4), kind='retimed'))
        return float(vn), float(vc), float(Fb['T'])


def tpas_table(F, T, dt=0.02):
    """TPAS 進度表 (τ, σ)：最終剖面的等時間取樣（含終點）。剖面到位時間與 T 的差須 ≤ RETIME_TOL（二分法殘差，毫秒級）；
    此時只把終點時間對齊到 T（比例記錄）。差更大＝沒有共同定時成功 ⇒ 回 (None, None, 比例)，由呼叫端受控停止——不以縮放掩蓋。"""
    m = F['sig'] <= F['x_pre'] + 1e-9
    tt, sg = F['t'][m], F['sig'][m]
    scale = float(T) / float(tt[-1])
    if abs(float(tt[-1]) - float(T)) > RETIME_TOL:
        return None, None, scale
    tk = np.arange(0.0, tt[-1] + 1e-9, dt)
    if tt[-1] - tk[-1] > 1e-9:
        tk = np.r_[tk, tt[-1]]
    sk = np.interp(tk, tt, sg)
    return tk * scale, sk, scale


class W1PlaceController(SC.W1PlaceController):
    group = SC.W1PlaceController.group + '_FPROF'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fp, self._fp_key, self.fp_log, self._fp_timing = None, None, [], None
        c = WL.WheelLimitConfig()
        self.fp_cfg = dict(wheel_radius=c.wheel_radius, wheel_base_L=c.wheel_base_L, wheel_w_max=c.wheel_w_max, cap=c.wheel_radius * c.wheel_w_max, margin=FP.MARGIN,
                           w_max=HD.W_MAX, alpha_max=HD.ALPHA_MAX, a=S1.A_SLEW)

    # ---- 剖面管理
    def _fp_active(self, x=None):
        return self.st.phase == 'CRUISE' and not self.blending and (x is None or x < float(self.x_pre))

    def _fp_ensure(self):
        """導航剖面：路徑或航向參考換版時由當下參考狀態（s_ref、v_b）重建；TPA 承接後改用 TPA 定時後的同一份剖面（_tpa_timing）。"""
        self._ensure_nav_hd()
        key = (id(self.path), id(self.hd), float(self.x_pre), float(self.x_slow))
        if self.fp is not None and self._fp_key == key:
            return self.fp
        if self.tpa is not None and self._fp_timing is not None and self._fp_key is not None and self._fp_key[:2] == key[:2]:
            return self.fp
        F = build(self.path, self.hd, float(self.s_ref), float(self.v_b), float(self.x_pre), float(self.x_slow), float(self.v_nav), float(self.v_c))
        self.fp_log.append(dict(t=round(float(self.t), 3), what='nav', ok=bool(F.get('ok')), why=F.get('why'), sig0=round(float(self.s_ref), 4), v0=round(float(self.v_b), 4),
                                T=round(F['T'], 3) if F.get('ok') else None, arc_err=round(F['arc_err'], 5) if F.get('ok') else None, entry_over_cap=F.get('entry_over_cap'),
                                over_margin=F.get('over_margin'), verify=F.get('verify'), detail=F.get('detail')))
        if not F.get('ok'):
            if self.judge.stop is None:
                self.judge._stop(self.t, 'fprof_infeasible', reason=str(F.get('why')), where='nav')
            return None
        self.fp, self._fp_key = F, key
        return F

    # ---- 使用點 1、3：參考速度
    def _v_target(self, T_tcp, x):
        if not self._fp_active(x):
            return super()._v_target(T_tcp, x)
        if self.tpa is None and x >= self.x_blend and not self._unfolded():     # 既有規則：手臂還沒展完，底盤等手臂
            return super()._v_target(T_tcp, x)
        F = self._fp_ensure()
        if F is None:
            return max(self.v_b - B.A_REF * 2.0 * DT, 0.0)
        return v_next(F, float(self.s_ref), 2.0 * DT)                         # 下一拍（時間軸）的剖面速度：減速點前就開始降；靜止起步不卡住

    # ---- 使用點 2：MPC 預覽窗
    def _ref_window(self, xs_eff):
        if not self._fp_active() or (self.tpa is None and float(self.s_ref) >= self.x_blend and not self._unfolded()):
            return super()._ref_window(xs_eff)
        F = self._fp_ensure()
        if F is None:
            return super()._ref_window(xs_eff)
        pose, xi, _, _ = window(F, self.path, self.hd, float(xs_eff), float(self.v_b), v_after=float(self.tpa['v_c_eff']) if self.tpa is not None else None)
        X_ref = np.array([from_xytheta(float(a), float(b), float(c)) for a, b, c in pose])
        return X_ref, xi

    # ---- 使用點 4：站位時間成本
    def _standoff_t_base(self, path_new, sig, x_pre_c):
        th0 = float(self.hd.yaw(float(self.s_ref))) if self.hd is not None else float(self._yaw_meas)
        hd = HD.HeadingRef(path_new, float(sig), th0)
        F = build(path_new, hd, float(sig), float(self.v_b), float(x_pre_c), float(x_pre_c) - 0.6, float(self.v_nav), float(self.v_c))
        if not F.get('ok'):
            return float('inf')                                                 # 該候選以同源剖面不可行 ⇒ 成本無限大（不改用梯形估）
        return float(F['T'])

    # ---- 使用點 5：TPA 時長與 TPAS 進度表
    def _tpa_timing(self, sig_e):
        self._ensure_nav_hd()
        tm = Timing(self.path, self.hd, float(sig_e), float(self.v_b), float(self.x_pre), float(self.x_slow), float(self.v_nav), float(self.v_c))
        if not tm.ok():
            self.fp_log.append(dict(t=round(float(self.t), 3), what='tpa', ok=False, why=tm.F0.get('why')))
            if self.judge.stop is None:
                self.judge._stop(self.t, 'fprof_infeasible', reason=str(tm.F0.get('why')), where='tpa')
            return None                                                         # 已停止；plan_tpa 仍會跑完既有路徑，但結果不會被執行
        self._fp_timing = tm
        return tm

    def _tpa_start(self, q):
        self._fp_timing = None
        super()._tpa_start(q)
        tm, P = self._fp_timing, self.tpa
        if P is None or tm is None or self.judge.stop is not None:
            return
        F = tm.F
        if not F.get('ok'):
            self.judge._stop(self.t, 'fprof_infeasible', reason=str(F.get('why')), where='tpa_retime')
            return
        self.fp, self._fp_key = F, (id(self.path), id(self.hd), float(self.x_pre), float(self.x_slow))
        self.fp_log.append(dict(t=round(float(self.t), 3), what='tpa', ok=True, T_arm=round(float(P['T']), 3), T_base=round(tm.t_base(), 3), T_final=round(float(F['T']), 3),
                                dT=round(float(F['T']) - float(P['T']), 4), retime=tm.log[-1] if tm.log else None, entry_over_cap=F.get('entry_over_cap')))

    def _tpas_prof(self, sig_e, P):
        tm = self._fp_timing
        if tm is None or not tm.F.get('ok'):
            if tm is not None and self.judge.stop is None:                      # 有 A0 定時但最終剖面不可用 ⇒ 受控停止
                self.judge._stop(self.t, 'fprof_infeasible', reason=str(tm.F.get('why')), where='tpas')
            return super()._tpas_prof(sig_e, P)
        tk, sk, scale = tpas_table(tm.F, P['T'])
        if tk is None:
            if self.judge.stop is None:
                self.judge._stop(self.t, 'fprof_infeasible', reason='tpas_table_T_mismatch', where='tpas', scale=round(float(scale), 4))
            return super()._tpas_prof(sig_e, P)                                 # 已停止；回傳只為讓父類流程不崩，結果不會被執行
        P['fprof'] = dict(tau_scale=round(scale, 6), sig0_plan=round(float(tm.a[2]), 4), sig0_tpas=round(float(sig_e), 4))
        return tk, sk + (float(sig_e) - float(tm.a[2]))                       # 同一進度定義下的起點對齊（_sig_of 與 sig_robot 的差；記錄）

    def report(self):
        r = super().report()
        r['fprof'] = dict(cfg=self.fp_cfg, log=self.fp_log, tpa=(self.tpa or {}).get('fprof') if self.tpa else None,
                          note='導航接近段參考速度、MPC 預覽、站位時間成本、TPA 時長、TPAS 進度表共用同一份輪級可行剖面；margin 為固定工程餘裕，非已辨識的追蹤誤差界')
        return r
