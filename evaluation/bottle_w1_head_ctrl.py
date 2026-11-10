#!/usr/bin/env python3
"""車頭沿規劃路徑切線（Howard 11/11「車頭朝路徑方向」；Codex 20261111_042529 (a)、045253 第 3 項；版本化子類，基線＝bottle_w1_tpas_ctrl）。
導航（CRUISE，非 CARRY）與運送（CARRY_NAV／PLACE_APPROACH）的底盤參考：
  位姿＝(路徑 xy(σ_k), θ(σ_k))，θ 由 bottle_heading.HeadingRef（切線 atan2、unwrap、0.5 m 平滑、承接混合 0.6 m）；
  本體速度前饋 ξ＝(R(θ)ᵀ·v·t_path, v·dθ/dσ)（bottle_heading.ref_window；承接混合期間航向≠切線時有本體橫向分量——Codex 050821）。
  換路徑（起跑前導引 → 量測後接近路徑 → 持瓶運送路徑）時以「舊參考在當下進度的偏航」（無舊參考則用實測偏航）為 θ0 連續承接，不瞬間轉頭。
  轉彎能力：參考速度 ≤ HeadingRef.v_cap（ω ≤ W_MAX、ω̇ 近似 ≤ ALPHA_MAX，前瞻減速）；運送段併入既有 vcap（驗收與執行同一函式）。
全身一致：
  TPAS 偏差補償的計畫偏航＝該進度點的路徑航向（_plan_yaw），不再用固定 th_ref；停止模型、相機外參照實際底盤姿態（不需改）。
  TPA 終點 base_end 的偏航沿用 0（車道直線段切線＝0）；承接時核 |θ(x_pre)| 並記錄，> HEAD_END_TOL ⇒ 受控停止（不默默套用）。
  運送段：航向參考起點＝實測位置在持瓶路徑上的投影、θ0＝實測偏航；直線段到停車切線＝0（＝place['theta']，記錄 yaw_lane 供核），停車後參考偏航＝place['theta']（同基線）。
未改：GMPC、CBF、Shield、抓取／放置判準；抓取窗與側向管仍只在車道直線段（偏航≈0）驗證過——本版不把導航鎖回固定朝向，抓取段照實適用範圍。
"""
import math

import numpy as np
from ammr_wholebody_mpc.se2 import from_xytheta

import bottle_heading as HD
import bottle_s0b_ctrl as B
import bottle_w1_tpas_ctrl as TS

N, DT = B.N, B.DT
HEAD_END_TOL = math.radians(2.0)


class W1PlaceController(TS.W1PlaceController):
    group = TS.W1PlaceController.group + '_HEAD'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.hd, self._hd_path, self.hd_c, self._hdc_path = None, None, None, None
        self.hd_log = []
        self._yaw_meas, self._xy_meas = 0.0, None

    # ---- 航向參考管理
    def _ensure_nav_hd(self):
        if self.path is self._hd_path:
            return
        prev, sig = self.hd, float(self.s_ref)
        if prev is None:
            th0, src = float(self._yaw_meas), 'yaw_meas'
        else:
            ds = float(self.tlane.get('dsig0', 0.0)) if getattr(self, 'tlane', None) else 0.0
            th0, src = float(prev.yaw(sig - ds)), 'prev_ref'                 # 舊參考在同一實體進度點的偏航（新路徑 σ 已平移 Δσ0）
        self.hd, self._hd_path = HD.HeadingRef(self.path, sig, th0), self.path
        rec = dict(t=round(float(self.t), 3), what='nav_path', sig=round(sig, 4), theta0=round(th0, 4), src=src, yaw_meas=round(float(self._yaw_meas), 4))
        self.hd_log.append(rec)
        self._ev('heading_ref', kind='nav', sig=rec['sig'], theta0=rec['theta0'], src=src)

    def _advance_inner(self, T_tcp, close_done, q):
        q = np.asarray(q, float)
        if np.isfinite(q[:3]).all():
            self._yaw_meas, self._xy_meas = float(q[2]), q[:2].copy()
        super()._advance_inner(T_tcp, close_done, q)

    # ---- 導航段參考窗
    def _ref_window(self, xs_eff):
        if self.st.phase == 'CARRY':
            return super()._ref_window(xs_eff)
        self._ensure_nav_hd()
        xs = xs_eff + self.v_b * DT * np.arange(N + 1)
        pose, xi_ref = HD.ref_window(self.path, self.hd, xs, self.v_b)
        X_ref = np.array([from_xytheta(float(a), float(b), float(c)) for a, b, c in pose])
        return X_ref, xi_ref

    def _v_target(self, T_tcp, x):
        v = super()._v_target(T_tcp, x)
        if self.st.phase == 'CRUISE' and self.hd is not None and not self.blending:
            v = min(v, self.hd.v_cap(float(self.s_ref)))
        return v

    def _plan_yaw(self, sig):
        if self.hd is None:
            return super()._plan_yaw(sig)
        return float(self.hd.yaw(sig))

    def _tpa_start(self, q):
        self._ensure_nav_hd()
        th_end = float(self.hd.yaw(float(self.x_pre)))
        self.hd_log.append(dict(t=round(float(self.t), 3), what='tpa_end_yaw', x_pre=round(float(self.x_pre), 4), yaw=round(th_end, 5)))
        if abs(math.atan2(math.sin(th_end), math.cos(th_end))) > HEAD_END_TOL:
            self.judge._stop(self.t, 'heading_end_not_lane', yaw=round(th_end, 4))
            return
        super()._tpa_start(q)

    # ---- 運送段
    def _carry_vcap(self, cp, R):
        base = super()._carry_vcap(cp, R)
        xy = getattr(self, '_xy_meas', None)
        sig0 = float(cp.project(xy, cp.sig0, back=0.0, ahead=0.3)) if xy is not None else float(cp.sig0)
        th0 = float(self._yaw_meas)
        hc = HD.HeadingRef(cp, sig0, th0)
        self.hd_c, self._hdc_path = hc, cp
        rec = dict(t=round(float(self.t), 3), what='carry_path', sig=round(sig0, 4), theta0=round(th0, 4), yaw_lane=round(float(hc.yaw(float(cp.x_lane))), 4))
        self.hd_log.append(rec)
        self._ev('heading_ref', kind='carry', sig=rec['sig'], theta0=rec['theta0'], yaw_lane=rec['yaw_lane'])
        return lambda s_, b_=base, h_=hc: min(b_(s_), h_.v_cap(s_))       # noqa: E731  驗收（ref_profile_check）與執行同一函式

    def _ref_window_place(self, xs_eff):
        if self.pph not in ('CARRY_NAV', 'PLACE_APPROACH') or self.hd_c is None or self.cpath is not self._hdc_path:
            return super()._ref_window_place(xs_eff)
        xs = xs_eff + self.v_b * DT * np.arange(N + 1)
        pose, xi = HD.ref_window(self.cpath, self.hd_c, xs, self.v_b)
        X = np.array([from_xytheta(float(a), float(b), float(c)) for a, b, c in pose])
        return X, xi

    def report(self):
        r = super().report()
        r['heading'] = dict(params=dict(L_BLEND=HD.L_BLEND, SMOOTH_M=HD.SMOOTH_M, W_MAX=HD.W_MAX, ALPHA_MAX=HD.ALPHA_MAX, A_DEC=HD.A_DEC, head_end_tol=HEAD_END_TOL),
                            log=self.hd_log, note='導航與運送參考偏航沿路徑切線；換路徑以舊參考偏航連續承接；轉彎依 ω／ω̇ 上限降速；TPAS 計畫偏航取進度點路徑航向')
        return r

