#!/usr/bin/env python3
"""運送進度與手臂剩餘動作協調（Codex 20261102_175049／184223；版本化子類，基線＝bottle_w1_gaze_ctrl：接近、視向、抓取、離桌完全相同）。
問題：online 的轉回最晚處置規則（_rp_late）用固定 v_carry 與最低時長，要求「距直線段入口 ≥ v_carry×(收近＋升高)＋在途＋煞停＝1.503 m 時已開始收近」，
  且在輪詢計畫結果之前檢查；短運送路徑（R-B2 持瓶路徑 1.048 m，觸發時剩餘 0.862 m）一進 CARRY_NAV 就判晚、受控停止。這是排程准入規則不接受短路徑，不是已證明放不下。
做法：只改持瓶段（CARRY_NAV／PLACE_APPROACH）底盤進度。手臂時長、第一桌收近閘、外伸閘、停止核對、路徑、放置判準、期限都不改。
1 持守點 σ_hold（每拍計算）：
  a 剩餘手臂包絡（底盤座標，夾爪＋持瓶點，姿態＝側持姿態，同 compactcarry 參考）：
    實測構型下的點；尚未完成的收近掃掠（側持→緊湊，NS 點；收近未開始＝整段，進行中＝剩餘段）；
    轉回計畫已承接時，再加升高（緊湊 xy、z 由緊湊到 z_back）與外伸（z_back、緊湊 xy→側持 xy）兩段掃掠（各 NS 點，完成前整段保守納入）。
    轉回計畫未承接時收近不能開始（收近閘前置），底盤也被 d 限在直線段入口前，升高與外伸不會發生，故不納入。
  b 底盤位姿＝沿運送路徑（含直線段）從實測進度 csig 起每 0.02 m 到停車位，保持實測側向偏移與偏航（同 _poses 的路徑前方）。
  c 判準同外伸閘（rect_margin）：對第二桌（放置先驗），低於桌面頂＋Z_PAD 的點須在投影外擴 EDGE_PAD 之外。第一個不過的位姿往前 0.02 m＝幾何持守點。
    第一桌由既有收近閘負責（收近只在第一桌淨空閘通過後開始）；這裡不納入第一桌，否則離桌後仍在桌旁時整段收近掃掠會把持守點壓在原地。
  d 收近未開始時另限 σ_hold ≤ x_lane − 0.02：收近只允許在 CARRY_NAV 開始（既有規則），不能讓底盤先進直線段。
  e σ_hold 不超過停車位。
2 車速：持守點在停車位之前時，speed_target 的剩餘距離改用 (σ_hold − GOV_BUF) − s_ref，沿用同一 a_dec（0.15 m/s²）平方根減速與同一斜率限制；
  持守點＝停車位時照原式（停車位減速不變）。GOV_BUF 0.04 m：涵蓋幾何持守點的 0.02 m 網格跳動、每 2 控制拍才更新車速與接近停止時斜率限制的拖尾
  （第一版只停在持守點本身時，延遲案參考煞停點最多超出 4.0 cm）。這是車速目標的緩衝，不是一般停止保證。
3 保底（以實測／在途狀態與既有停止模型 _stop_rest：命令生效時刻構型、現在停止點、延後一拍停止點）：
  a 超越監看（持守啟用期間全程，收近前後都監看）：現在停與延後一拍停的底盤停止位姿，配合各自的手臂停止構型與剩餘包絡，對第二桌判準不過
    ⇒ carry_hold_overrun 受控停止（持守點突然後退、追蹤落後等，參考煞停點看不到的情形）。
  b 計畫晚到（覆寫 _rp_late，不刪）：收近未開始時，延後一拍停止點沿路徑的進度 > σ_hold ⇒ return_plan_late；進 PLACE_APPROACH 仍未開始收近 ⇒ 同原規則判晚。
  c 非有限：持守啟用期間任何輸入（實測構型、進度、包絡、停止模型）非有限 ⇒ carry_gov_invalid 受控停止，不放行到停車位。
4 等待上限：停在持守點（v_b＝0 且受持守點限制）累計超過 HOLD_MAX ⇒ carry_hold_timeout 受控停止。
5 停車求解器開 gmpc_wb.zero_box_clip_tol＝(1e-9 m/s, 1e-9 rad/s)（Codex 184223 另項；只對三軸上下限皆為零的停車求解器，數值容差內一致另記錄）。
R-B1 原路徑上持守點始終在煞停點之前、速度不受限（假世界核對）。
"""
import dataclasses
import math

import numpy as np

import bottle_place_ctrl as PCm
import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_ref as BR
import bottle_w1_compactcarry2_ctrl as C2m
import bottle_w1_gaze_ctrl as GZ

C3, C2 = GZ.C3, GZ.C2
GOV_DS, GOV_BACK, GOV_BUF = 0.02, 0.02, 0.04
HOLD_MAX = 10.0
PARK_CLIP_TOL = (1e-9, 1e-9)


def rect_margins(Pw, rect):
    """Pw：(M, K, 3) 世界點（M 個底盤位姿 × K 點）。回每個位姿的 rect_margin（低點到投影外擴 EDGE_PAD 的最小水平距離；無低點＝+inf）。"""
    (cx, cy), (hx, hy), top = rect
    low = Pw[..., 2] < top + C2m.Z_PAD
    dx, dy = np.abs(Pw[..., 0] - cx) - hx, np.abs(Pw[..., 1] - cy) - hy
    d = np.where((dx <= 0) & (dy <= 0), np.maximum(dx, dy), np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))) - C2m.EDGE_PAD
    return np.where(low, d, np.inf).min(axis=1)


def place_world(E, poses):
    """底盤座標點 E (K,3) × 位姿 poses (M,3) ⇒ (M, K, 3) 世界點。"""
    c, s_ = np.cos(poses[:, 2:3]), np.sin(poses[:, 2:3])
    X = poses[:, 0:1] + c * E[:, 0] - s_ * E[:, 1]
    Y = poses[:, 1:2] + s_ * E[:, 0] + c * E[:, 1]
    return np.stack([X, Y, np.broadcast_to(E[:, 2], X.shape)], axis=-1)


class W1PlaceController(GZ.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_ONLINE5_CENTERCONV_GAZE_CARRYGOV'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gov_hold, self.gov_info, self.gov_stop = None, None, None
        self.gov_log, self.gov_ev = [], []
        self.gov_hold_s, self.gov_bind = 0.0, False
        self.gov_min = dict(hold_minus_brake=math.inf, t=None, stop_margin=math.inf, t_stop=None)
        self.mpc_park = B.GW.GMPC(dataclasses.replace(self.mpc_park.cfg, zero_box_clip_tol=PARK_CLIP_TOL))
        self.park_tol_log = []
        _s = self.mpc_park.solve

        def solve_logged(*a_, **k_):
            r_ = _s(*a_, **k_)
            st_ = (r_.arm_check or {}).get('seq_tol')
            if st_ is not None and len(self.park_tol_log) < 200:
                self.park_tol_log.append(dict(t=round(float(self.t), 3), pph=self.pph, **st_))
            return r_
        self.mpc_park.solve = solve_logged

    # ---- 包絡
    def _gov_env(self, q_arms):
        """剩餘手臂包絡（底盤座標）：q_arms 各構型的實際點＋剩餘的收近／升高／外伸掃掠。"""
        Pt = self._pts_tcp()
        P = []
        for qa in q_arms:
            Tq = self.K.fk(np.r_[0.0, 0.0, 0.0, np.asarray(qa, float)], 'link_tcp')
            P.append((Tq[:3, :3] @ Pt.T).T + Tq[:3, 3])
        f = self.cc
        stage = None if f is None else f['stage']
        T_lift = np.asarray(self.st.T_carry_rel if f is None else f.get('T_side_lift', f['T_side']), float)
        R, pc = T_lift[:3, :3], np.asarray(self._tcp_c(), float)
        Pr = (R @ Pt.T).T
        if stage in (None, 'to_compact'):
            pa = T_lift[:3, 3]
            s0 = 0.0 if f is None else BR.min_jerk(min(max((float(self.t) - f['t0']) / self._ccd('T_TO'), 0.0), 1.0))
            P += [Pr + (pa + (pc - pa) * s) for s in np.linspace(s0, 1.0, C2m.NS)]
        rp = getattr(self, 'rp', None)
        if rp is not None and rp.get('state') == 'accepted' and stage != 'side':
            zb = float(rp['res']['z_back'])
            if stage in (None, 'to_compact', 'compact', 'raise'):
                P += [Pr + np.array([pc[0], pc[1], pc[2] + (zb - pc[2]) * s]) for s in np.linspace(0.0, 1.0, C2m.NS)]
            pa_xy = T_lift[:2, 3]
            P += [Pr + np.array([pc[0] + (pa_xy[0] - pc[0]) * s, pc[1] + (pa_xy[1] - pc[1]) * s, zb]) for s in np.linspace(0.0, 1.0, C2m.NS)]
        return np.vstack(P)

    # ---- 持守點與保底
    def _gov_compute(self, q):
        q = np.asarray(q, float)
        park = float(self.park[0])
        sig = np.arange(float(self.csig), park + 1e-9, GOV_DS)
        if len(sig) == 0:
            sig = np.array([park])
        off = q[:2] - self.cpath.xy(np.array([self.csig]))[0]
        xy = self.cpath.xy(sig) + off
        E = self._gov_env([q[3:]])
        if not (np.isfinite(E).all() and np.isfinite(xy).all()):
            return False
        m = rect_margins(place_world(E, np.column_stack([xy, np.full(len(xy), q[2])])), self._rects()[1])
        bad = np.flatnonzero(m < 0.0)
        geom = math.inf if not len(bad) else float(sig[bad[0]]) - GOV_BACK
        hold, why = park, 'park'
        if geom < hold:
            hold, why = geom, 'table2'
        if self.cc is None and float(self.cpath.x_lane) - GOV_BACK < hold:
            hold, why = float(self.cpath.x_lane) - GOV_BACK, 'lane_before_compact'
        self.gov_hold = hold
        self.gov_info = dict(why=why, geom=geom, m_min=float(m.min()) if len(m) else None, n_pose=len(sig), n_pts=len(E))
        return True

    def _gov_stop_check(self, q):
        """既有停止模型：命令生效時刻構型 qp、現在停止點 q_rest、延後一拍停止點 q_next。回 dict 或 None（非有限）。"""
        qp, q_rest, q_next = self._stop_rest(np.asarray(q, float))
        Q = np.vstack([qp, q_rest, q_next])
        if not np.isfinite(Q).all():
            return None
        E = self._gov_env([q[3:], q_rest[3:], q_next[3:]])
        if not np.isfinite(E).all():
            return None
        m = rect_margins(place_world(E, Q[1:, :3]), self._rects()[1])
        sig_next = float(self.cpath.project(np.asarray(q_next[:2], float), float(self.csig)))
        if not math.isfinite(sig_next):
            return None
        return dict(m_rest=float(m[0]), m_next=float(m[1]), sig_next=sig_next)

    # ---- 每拍
    def _advance_inner(self, T_tcp, close_done, q):
        active = self.judge.stop is None and self.pph in ('CARRY_NAV', 'PLACE_APPROACH') and self.cpath is not None and self.csig is not None \
            and self.park is not None and self.judge.T_rel is not None
        if active:
            q = np.asarray(q, float)
            ok = np.isfinite(q).all() and math.isfinite(float(self.csig)) and math.isfinite(float(self.s_ref)) and self._gov_compute(q)
            sc = self._gov_stop_check(q) if ok else None
            if not ok or sc is None:
                self.gov_stop = None
                self._pstop('carry_gov_invalid', stage=None if self.cc is None else self.cc['stage'])
                return
            self.gov_stop = sc
            ms = min(sc['m_rest'], sc['m_next'])
            if ms < self.gov_min['stop_margin']:
                self.gov_min.update(stop_margin=ms, t_stop=float(self.t))
            if ms < 0.0:
                self._pstop('carry_hold_overrun', m_rest=round(sc['m_rest'], 4), m_next=round(sc['m_next'], 4), hold=round(self.gov_hold, 4), csig=round(float(self.csig), 4),
                            stage=None if self.cc is None else self.cc['stage'])
                return
            self._gov_track()
        elif self.pph not in ('CARRY_NAV', 'PLACE_APPROACH'):
            self.gov_hold, self.gov_stop = None, None
        if self.judge.stop is None and self.gov_hold is not None and self.gov_hold_s > HOLD_MAX:
            self._pstop('carry_hold_timeout', hold=round(self.gov_hold, 4), waited=round(self.gov_hold_s, 3), hold_why=self.gov_info.get('why'))
            return
        super()._advance_inner(T_tcp, close_done, q)

    def _gov_track(self):
        h, gi, sc = self.gov_hold, self.gov_info, self.gov_stop
        R = self.preg
        v = float(self.v_b)
        brake = float(self.s_ref) + v * v / (2.0 * R['a_dec'])
        binding = h < float(self.park[0]) - 1e-9 and math.sqrt(2.0 * R['a_dec'] * max(h - GOV_BUF - float(self.s_ref), 0.0)) < float(self.v_carry) - 1e-9
        if gi['why'] != 'park' and h - brake < self.gov_min['hold_minus_brake']:
            self.gov_min.update(hold_minus_brake=h - brake, t=float(self.t), cc=None if self.cc is None else self.cc['stage'])
        held = binding and v == 0.0 and h - GOV_BUF - float(self.s_ref) <= 0.03
        if held:
            self.gov_hold_s += B.DT
        if binding != self.gov_bind:
            self.gov_bind = binding
            self.gov_ev.append((round(float(self.t), 3), 'bind' if binding else 'release', gi['why'], round(h, 4), round(float(self.csig), 4), round(v, 4)))
            self._ev('carry_gov_' + ('bind' if binding else 'release'), why=gi['why'], hold=round(h, 4), csig=round(float(self.csig), 4), v_b=round(v, 4))
        if self.n_step % 2 == 0 and len(self.gov_log) < 3000:
            mm = gi['m_min']
            self.gov_log.append((round(float(self.t), 3), self.pph, None if self.cc is None else self.cc['stage'], round(float(self.csig), 4), round(float(self.s_ref), 4), round(v, 4),
                                 round(h, 4), gi['why'], None if not math.isfinite(gi['geom']) else round(gi['geom'], 4), None if mm is None or not math.isfinite(mm) else round(mm, 4),
                                 None if not math.isfinite(sc['m_next']) else round(sc['m_next'], 4), round(sc['sig_next'], 4)))

    def _place_speed(self):
        if self.pph not in ('CARRY_NAV', 'PLACE_APPROACH') or self.gov_hold is None:
            return super()._place_speed()
        R = self.preg
        park = float(self.park[0])
        rem = (park if self.gov_hold >= park - 1e-9 else float(self.gov_hold) - GOV_BUF) - self.s_ref
        return PCm.speed_target(self.s_ref, self.v_b, rem, self.vcap, self.v_carry, R['a_dec'], S1.A_SLEW * 2.0 * B.DT)

    def _rp_late(self):
        """計畫晚到：收近未開始時，延後一拍停止點沿路徑進度越過持守點；進 PLACE_APPROACH 仍未收近同原規則。來不及停的監看另在 _advance_inner（全程）。"""
        if self.cc is not None:
            return False
        if self.pph == 'PLACE_APPROACH':
            return True
        if self.pph != 'CARRY_NAV' or self.gov_hold is None or self.gov_stop is None:
            return False
        return self.gov_stop['sig_next'] > self.gov_hold

    def report(self):
        r = super().report()
        gm = self.gov_min
        r['carry_gov'] = dict(ds=GOV_DS, back=GOV_BACK, buf=GOV_BUF, hold_max=HOLD_MAX, a_dec=self.preg['a_dec'], events=self.gov_ev, hold_s=round(self.gov_hold_s, 3),
                              min_hold_minus_brake=None if not math.isfinite(gm['hold_minus_brake']) else dict(v=round(gm['hold_minus_brake'], 4), t=gm['t'], cc=gm.get('cc')),
                              min_stop_margin=None if not math.isfinite(gm['stop_margin']) else dict(v=round(gm['stop_margin'], 4), t=gm['t_stop']),
                              park_clip_tol=list(PARK_CLIP_TOL), park_tol_log=self.park_tol_log,
                              log_cols=['t', 'pph', 'cc_stage', 'csig', 's_ref', 'v_b', 'hold', 'why', 'geom', 'm_min', 'm_stop_next', 'sig_next'], log=self.gov_log,
                              note='持守點＝剩餘手臂包絡對第二桌第一次不過之前 0.02 m；收近未開始另限直線段入口前；受限時車速停在持守點前 GOV_BUF；'
                                   '保底＝停止模型（現在停／延後一拍停）對第二桌判準（全程）、計畫晚到＝延後一拍停止點越過持守點、非有限受控停止')
        return r
