#!/usr/bin/env python3
"""運送進度與手臂剩餘動作協調（Codex 20261102_175049；版本化子類，基線＝bottle_w1_gaze_ctrl：接近、視向、抓取、離桌完全相同）。
問題：online 的轉回最晚處置規則（_rp_late）用固定 v_carry 與最低時長，要求「距直線段入口 ≥ v_carry×(收近＋升高)＋在途＋煞停＝1.503 m 時已開始收近」，
  且在輪詢計畫結果之前檢查；短運送路徑（R-B2 持瓶路徑 1.048 m，觸發時剩餘 0.862 m）一進 CARRY_NAV 就判晚、受控停止。這是排程准入規則不接受短路徑，不是已證明放不下。
做法：只改持瓶段（CARRY_NAV／PLACE_APPROACH）底盤進度。手臂時長、第一桌收近閘、外伸閘、停止核對、路徑、放置判準、期限都不改。
1 持守點 σ_hold（每拍計算）：
  a 剩餘手臂包絡（底盤座標）＝實測構型下的持瓶／夾爪點，加上尚未完成的收近掃掠（側持→緊湊，NS 點；收近未開始＝整段，進行中＝剩餘段，含緊湊端點）。
    緊湊後的升高在緊湊 xy 原地向上，水平投影與緊湊相同；外伸在 z_back 進行，z_back 由轉回計畫依第二桌面＋Z_PAD＋H_CLEAR 求得，高於判準高度。兩者都不另加。
  b 底盤位姿＝沿運送路徑（含直線段）從實測進度 csig 起每 0.02 m 到停車位，保持實測側向偏移與偏航（同 _poses 的路徑前方）。
  c 判準同外伸閘（rect_margin）：對第二桌（放置先驗），低於桌面頂＋Z_PAD 的點須在投影外擴 EDGE_PAD 之外。第一個不過的位姿往前 0.02 m＝幾何持守點。
    第一桌由既有收近閘負責（收近只在第一桌淨空閘通過後開始）；這裡不納入第一桌，否則離桌後仍在桌旁時整段收近掃掠會把持守點壓在原地。
  d 收近未開始時另限 σ_hold ≤ x_lane − 0.02：收近只允許在 CARRY_NAV 開始（既有規則），不能讓底盤先進直線段。
  e σ_hold 不超過停車位。
2 車速：持守點在停車位之前時，speed_target 的剩餘距離改用 (σ_hold − GOV_BUF) − s_ref，沿用同一 a_dec（0.15 m/s²）平方根減速與同一斜率限制；
  持守點＝停車位時照原式（停車位減速不變）。計畫未到或手臂未完成時，底盤在進入危險範圍前有界減速、停在持守點前 GOV_BUF 等待；持守點前移後依既有斜率起步。
  GOV_BUF 0.04 m：涵蓋幾何持守點的 0.02 m 網格跳動（收近進行中實測在 ±0.02 m 間跳）、每 2 控制拍才更新車速（0.25 m/s 下 0.0125 m）與接近停止時斜率限制的拖尾
  （第一版只停在持守點本身時，延遲案煞停點最多超出 4.0 cm）。
3 晚判（覆寫 _rp_late，不刪）：收近未開始時，參考煞停點 s_ref＋v²／(2 a_dec) > σ_hold ⇒ return_plan_late 受控停止
  （持守點突然後退或速度異常時的保底）；進 PLACE_APPROACH 仍未開始收近 ⇒ 同原規則判晚。煞停點與車速曲線同式（在途延遲已由 s_ref 的在途補償處理）。
  核對記受持守點限制期間的 min(σ_hold − 煞停點)。
4 等待上限：停在持守點（v_b＝0 且受持守點限制）累計超過 HOLD_MAX ⇒ carry_hold_timeout 受控停止。
R-B1 原路徑上持守點始終在煞停點之前、速度不受限（假世界核對）。
"""
import math

import numpy as np

import bottle_place_ctrl as PCm
import bottle_s0_ctrl as C0
import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_ref as BR
import bottle_w1_compactcarry2_ctrl as C2m
import bottle_w1_gaze_ctrl as GZ

C3, C2 = GZ.C3, GZ.C2
GOV_DS, GOV_BACK, GOV_BUF = 0.02, 0.02, 0.04
HOLD_MAX = 10.0


def rect_margins(Pw, rect):
    """Pw：(M, K, 3) 世界點（M 個底盤位姿 × K 點）。回每個位姿的 rect_margin（低點到投影外擴 EDGE_PAD 的最小水平距離；無低點＝+inf）。"""
    (cx, cy), (hx, hy), top = rect
    low = Pw[..., 2] < top + C2m.Z_PAD
    dx, dy = np.abs(Pw[..., 0] - cx) - hx, np.abs(Pw[..., 1] - cy) - hy
    d = np.where((dx <= 0) & (dy <= 0), np.maximum(dx, dy), np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))) - C2m.EDGE_PAD
    return np.where(low, d, np.inf).min(axis=1)


class W1PlaceController(GZ.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3_SEGTARGET_VHIGH_RELWIN_SLIPWARN_ONLINE5_CENTERCONV_GAZE_CARRYGOV'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gov_hold, self.gov_info = None, None
        self.gov_log, self.gov_ev = [], []
        self.gov_hold_s, self.gov_bind = 0.0, False
        self.gov_min = dict(hold_minus_brake=math.inf, t=None)

    # ---- 包絡與持守點
    def _gov_env(self, q):
        Pt = self._pts_tcp()
        Tq = self.K.fk(np.r_[0.0, 0.0, 0.0, np.asarray(q, float)[3:]], 'link_tcp')
        P = [(Tq[:3, :3] @ Pt.T).T + Tq[:3, 3]]
        f = self.cc
        if f is None or f['stage'] == 'to_compact':
            T_side = np.asarray(self.st.T_carry_rel if f is None else f['T_side'], float)
            s0 = 0.0 if f is None else BR.min_jerk(min(max((float(self.t) - f['t0']) / self._ccd('T_TO'), 0.0), 1.0))
            R, pa, pc = T_side[:3, :3], T_side[:3, 3], C2m.TCP_C
            Pr = (R @ Pt.T).T
            P += [Pr + (pa + (pc - pa) * s) for s in np.linspace(s0, 1.0, C2m.NS)]
        return np.vstack(P)

    def _gov_compute(self, q):
        q = np.asarray(q, float)
        park = float(self.park[0])
        sig = np.arange(float(self.csig), park + 1e-9, GOV_DS)
        if len(sig) == 0:
            sig = np.array([park])
        off = q[:2] - self.cpath.xy(np.array([self.csig]))[0]
        xy = self.cpath.xy(sig) + off
        E = self._gov_env(q)
        c, s_ = math.cos(q[2]), math.sin(q[2])
        X = xy[:, 0:1] + c * E[:, 0] - s_ * E[:, 1]
        Y = xy[:, 1:2] + s_ * E[:, 0] + c * E[:, 1]
        Pw = np.stack([X, Y, np.broadcast_to(E[:, 2], X.shape)], axis=-1)
        m = rect_margins(Pw, self._rects()[1])
        bad = np.flatnonzero(m < 0.0)
        geom = math.inf if not len(bad) else float(sig[bad[0]]) - GOV_BACK
        hold, why = park, 'park'
        if geom < hold:
            hold, why = geom, 'table2'
        if self.cc is None and float(self.cpath.x_lane) - GOV_BACK < hold:
            hold, why = float(self.cpath.x_lane) - GOV_BACK, 'lane_before_compact'
        self.gov_hold = hold
        self.gov_info = dict(why=why, geom=geom, m_min=float(m.min()) if len(m) else None, n_pose=len(sig), n_pts=len(E))

    # ---- 每拍
    def _advance_inner(self, T_tcp, close_done, q):
        if self.judge.stop is None and self.pph in ('CARRY_NAV', 'PLACE_APPROACH') and self.cpath is not None and self.csig is not None and self.park is not None \
                and self.judge.T_rel is not None and np.isfinite(np.asarray(q, float)).all():
            self._gov_compute(q)
            self._gov_track()
        elif self.pph not in ('CARRY_NAV', 'PLACE_APPROACH'):
            self.gov_hold = None
        if self.judge.stop is None and self.gov_hold is not None and self.gov_hold_s > HOLD_MAX:
            self._pstop('carry_hold_timeout', hold=round(self.gov_hold, 4), waited=round(self.gov_hold_s, 3), hold_why=self.gov_info.get('why'))
            return
        super()._advance_inner(T_tcp, close_done, q)

    def _gov_track(self):
        h, gi = self.gov_hold, self.gov_info
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
            self.gov_log.append((round(float(self.t), 3), self.pph, None if self.cc is None else self.cc['stage'], round(float(self.csig), 4), round(float(self.s_ref), 4), round(v, 4),
                                 round(h, 4), gi['why'], None if not math.isfinite(gi['geom']) else round(gi['geom'], 4), None if gi['m_min'] is None or not math.isfinite(gi['m_min']) else round(gi['m_min'], 4)))

    def _place_speed(self):
        if self.pph not in ('CARRY_NAV', 'PLACE_APPROACH') or self.gov_hold is None:
            return super()._place_speed()
        R = self.preg
        park = float(self.park[0])
        rem = (park if self.gov_hold >= park - 1e-9 else float(self.gov_hold) - GOV_BUF) - self.s_ref
        return PCm.speed_target(self.s_ref, self.v_b, rem, self.vcap, self.v_carry, R['a_dec'], S1.A_SLEW * 2.0 * B.DT)

    def _rp_late(self):
        if self.cc is not None:
            return False
        if self.pph == 'PLACE_APPROACH':
            return True
        if self.pph != 'CARRY_NAV' or self.cpath is None or self.csig is None or self.gov_hold is None:
            return False
        v = float(self.v_b)
        brake = float(self.s_ref) + v * v / (2.0 * self.preg['a_dec'])
        return brake > self.gov_hold

    def report(self):
        r = super().report()
        r['carry_gov'] = dict(ds=GOV_DS, back=GOV_BACK, buf=GOV_BUF, hold_max=HOLD_MAX, a_dec=self.preg['a_dec'], events=self.gov_ev, hold_s=round(self.gov_hold_s, 3),
                              min_hold_minus_brake=None if not math.isfinite(self.gov_min['hold_minus_brake']) else dict(v=round(self.gov_min['hold_minus_brake'], 4), t=self.gov_min['t'], cc=self.gov_min.get('cc')),
                              log_cols=['t', 'pph', 'cc_stage', 'csig', 's_ref', 'v_b', 'hold', 'why', 'geom', 'm_min'], log=self.gov_log,
                              note='持守點＝剩餘手臂包絡（實測點＋未完成收近掃掠）對第二桌第一次不過之前 0.02 m；收近未開始另限直線段入口前；受限時車速停在持守點前 GOV_BUF；min_hold_minus_brake 只統計受持守點限制期間')
        return r
