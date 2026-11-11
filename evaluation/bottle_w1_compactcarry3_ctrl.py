#!/usr/bin/env python3
"""緊湊持物 v3：轉回排程單一候選（Codex 20261101_204716 第 4 項；bottle_w1_compactcarry21_ctrl 的版本化子類，v2／v2.1 原樣保留）。
v2 的轉回要等進 PLACE_APPROACH 才開始（升高 2.0 s ＋ 外伸 3.8 s），底盤到停車位後約等 2.9 s。v3 只把「升高」提前，外伸時機不提前：
1 提前升高（保持近身）：緊湊穩態且仍在 CARRY_NAV，沿運送路徑實測進度 csig 距直線段入口（＝PLACE_APPROACH 觸發點 x_lane）≤ v_carry×T_UP 時，
  在緊湊 xy 原地只升高到側持高度（2.0 s、起訖零速）。升高後包絡半徑不變（瓶身＋夾爪仍近身），瓶底由約 0.50 m 升到約 0.65 m。
  若進 PLACE_APPROACH 時尚未開始升高，就在那時開始（同 v2）；收近未完成時先走完。
2 外伸（3.8 s、起訖零速）仍在 PLACE_APPROACH 才開始，且須升高完成並通過外伸淨空閘：外伸整段路徑（21 點）的瓶身＋夾爪點 × 底盤位姿集合（同 v2.1：實測、在途、預測時域、
  路徑前方 V_CMD_MAX×剩餘時間）＋兩條停止軌跡（同 v2.1：底盤停止位姿 × 手臂停止構型交叉），對第一桌與第二桌桌面投影套同一判準（低於桌面頂＋Z_PAD 的點須在投影外擴 EDGE_PAD 之外）。
  不過就等待（不縮短動作、不放寬保護；底盤到停車位也不累計停車靜止，由既有 PLACE_APPROACH 期限兜底）；外伸中每拍重核，不過或非有限 ⇒ 受控停止 place_reach_clear_lost／place_reach_clear_invalid。
時長仍依 bottle_w1_compact_timing 的路徑關節速度／加速度要求（升高 2.0 s、外伸 3.8 s），不縮短。第一桌閘與停止軌跡核對同 v2.1（取樣幾何檢查，不是連續安全保證）。
"""
import math

import numpy as np

import bottle_ref as BR
import bottle_w1_compactcarry21_ctrl as C21
import bottle_w1_compactcarry2_ctrl as C2

T_UP, T_OUT = C2.T_UP, C2.T_OUT


def rect_margin(P, rect):
    """P（世界座標點）中低於 rect 桌面頂＋Z_PAD 者，到 rect 投影的最小水平距離減 EDGE_PAD；沒有低點 ⇒ +inf。"""
    (cx, cy), (hx, hy), top = rect
    lw = P[P[:, 2] < top + C2.Z_PAD]
    if not len(lw):
        return math.inf
    dx, dy = np.abs(lw[:, 0] - cx) - hx, np.abs(lw[:, 1] - cy) - hy
    d = np.where((dx <= 0) & (dy <= 0), np.maximum(dx, dy), np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0)))
    return float(d.min()) - C2.EDGE_PAD


class W1PlaceController(C21.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT3'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.reach_log, self.reach_trig, self.reach_mon_min = [], None, math.inf

    def _ccd3(self, name):
        """升高／外伸時長：實例參數 cc_t_up／cc_t_out 優先；未設＝呼叫當下的本模組常數 T_UP／T_OUT（舊版本與暫換常數的子類行為不變）。"""
        v = getattr(self, 'cc_' + name.lower(), None)
        return float(v) if v is not None else globals()[name]

    def _rects(self):
        P = self.place
        top1, c1, h1 = self._t1()
        return [(tuple(c1), tuple(h1), top1), (tuple(P['c']), (P['size'][0] / 2, P['size'][1] / 2), float(P['top']))]

    def _T_raised(self):
        T = self.cc['T_side'].copy()
        pc_ = self._tcp_c()
        T[:3, 3] = [pc_[0], pc_[1], T[2, 3]]
        return T

    def _cc_T(self, t):
        f = self.cc
        if f.get('t_raise') is None:
            return super()._cc_T(t)
        pa, pc = f['T_side'][:3, 3], self._tcp_c()
        T = f['T_side'].copy()
        tau = t - f['t_raise']
        s = BR.min_jerk(min(max(tau / self._ccd3('T_UP'), 0.0), 1.0))
        T[:3, 3] = [pc[0], pc[1], pc[2] + (pa[2] - pc[2]) * s]
        if f.get('t_out') is not None:
            u = BR.min_jerk(min(max((t - f['t_out']) / self._ccd3('T_OUT'), 0.0), 1.0))
            T[:3, 3] = [pc[0] + (pa[0] - pc[0]) * u, pc[1] + (pa[1] - pc[1]) * u, T[2, 3]]
        return T

    def _reach_gate(self, q, t_rem):
        """外伸淨空閘：外伸路徑 × 位姿集合＋兩條停止軌跡，對兩桌。回 (餘裕, 位姿數)；非有限 ⇒ (None, n)。"""
        q = np.asarray(q, float)
        if self.judge.T_rel is None or not np.isfinite(q).all():
            return None, 0
        Pt = self._pts_tcp()
        Ta, Tb = self._T_raised(), self.cc['T_side']
        R = Ta[:3, :3] @ Pt.T
        sweep = np.vstack([R.T + (Ta[:3, 3] + (Tb[:3, 3] - Ta[:3, 3]) * s) for s in np.linspace(0.0, 1.0, C2.NS)])
        poses = self._poses(q, t_rem)
        Q = np.vstack([C21.stop_traj(q, self.s, self.pipe, self.u_last, self.cfg, e) for e in (0, 1)])
        if not (np.isfinite(sweep).all() and np.isfinite(poses).all() and np.isfinite(Q).all()):
            return None, len(poses)

        def world(Pb, ps):                                   # 底盤座標點 × 位姿集合（交叉，不依時間對齊）
            c, s_ = np.cos(ps[:, 2:3]), np.sin(ps[:, 2:3])
            X = ps[:, 0:1] + c * Pb[:, 0] - s_ * Pb[:, 1]
            Y = ps[:, 1:2] + s_ * Pb[:, 0] + c * Pb[:, 1]
            return np.column_stack([X.ravel(), Y.ravel(), np.tile(Pb[:, 2], len(ps))])
        PB = []
        for qk in Q:                                         # 停止軌跡：手臂停止構型（底盤座標）× 底盤停止位姿，同 v2.1
            Tb_ = self.K.fk(np.r_[0.0, 0.0, 0.0, qk[3:]], 'link_tcp')
            PB.append((Tb_[:3, :3] @ Pt.T).T + Tb_[:3, 3])
        Tw = self.K.fk(q, 'link_tcp')
        Pw = np.vstack([world(sweep, poses), world(np.vstack(PB), Q[:, :3]), (Tw[:3, :3] @ Pt.T).T + Tw[:3, 3]])
        m = math.inf
        for rect in self._rects():
            m = min(m, rect_margin(Pw, rect))
        return m, len(poses)

    def _advance_inner(self, T_tcp, close_done, q):
        """不走 v2 的轉回分支：相位推進交給連續展臂基線（CA）以上，收近閘／監看同 v2（閘本身是 v2.1 的），之後換成 v3 的升高／外伸排程。"""
        C2.CA.W1PlaceController._advance_inner(self, T_tcp, close_done, q)
        if self.judge.stop is not None:
            return
        t, f = float(self.t), self.cc
        if f is None:
            if self.pph == 'CARRY_NAV' and self.judge.t_grasp is not None and self.judge.t_lift_ok is not None:
                m, n = self._gate(q, self._ccd('T_TO'))
                if self.n_step % 5 == 0 or (m is not None and m >= 0.0):
                    self.gate_log.append((round(t, 3), 'pre', None if m is None else round(m, 4), n))
                if m is not None and m >= 0.0:
                    self.cc = dict(stage='to_compact', t0=t, t_ret=None, t_raise=None, t_out=None, T_side=np.array(self.st.T_carry_rel, float))
                    self.gate_trig = dict(t=t, margin=m, n_poses=n)
                    self._ev('compact_start', gate_margin=round(m, 4), n_poses=n)
                    self.cc_ev.append(('to_compact', t))
            return
        if f['stage'] == 'to_compact':
            t_rem = f['t0'] + self._ccd('T_TO') - t
            m, n = self._gate(q, t_rem)
            if self.n_step % 5 == 0 or m is None or m < 0.0:
                self.gate_log.append((round(t, 3), 'mon', None if m is None else round(m, 4), n))
            if m is None:
                self._pstop('compact_clear_invalid')
                return
            self.gate_mon_min = min(self.gate_mon_min, m)
            if m < 0.0:
                self._pstop('compact_table_clear_lost', margin=round(m, 4))
                return
            if t_rem <= 1e-9:
                f['stage'] = 'compact'
                self._ev('compact_reached')
                self.cc_ev.append(('compact', t))
        if f['stage'] == 'compact':
            trig = self._cc_raise_trigger()
            if trig is not None:
                self._start_raise(*trig)
        if f['stage'] == 'raise' and t - f['t_raise'] >= self._ccd3('T_UP') - 1e-9:
            f['stage'] = 'raised'
            self._ev('compact_raised')
            self.cc_ev.append(('raised', t))
        if f['stage'] == 'raised' and self._cc_out_phase_ok():
            m, n = self._reach_gate(q, self._ccd3('T_OUT'))
            self.reach_log.append((round(t, 3), 'pre', None if m is None else round(m, 4), n))
            if m is not None and m >= 0.0:
                f.update(stage='out', t_out=t)
                self.reach_trig = dict(t=t, margin=m, n_poses=n)
                self._ev('compact_out_start', reach_margin=round(m, 4))
                self.cc_ev.append(('out', t))
        elif f['stage'] == 'out':
            t_rem = f['t_out'] + self._ccd3('T_OUT') - t
            m, n = self._reach_gate(q, max(t_rem, 0.0))
            if self.n_step % 5 == 0 or m is None or m < 0.0:
                self.reach_log.append((round(t, 3), 'mon', None if m is None else round(m, 4), n))
            if m is None:
                self._pstop('place_reach_clear_invalid')
                return
            self.reach_mon_min = min(self.reach_mon_min, m)
            if m < 0.0:
                self._pstop('place_reach_clear_lost', margin=round(m, 4))
                return
            if t_rem <= 1e-9:
                f['stage'] = 'side'
                self._ev('compact_returned')
                self.cc_ev.append(('side', t))
        if f['stage'] in ('to_compact', 'compact', 'raise', 'raised', 'out') and self.pph == 'PLACE_APPROACH':
            self.n_still_p = 0                                   # 外伸完成前不累計停車靜止
        if self.n_step % 5 == 0:
            self.cc_log.append((round(t, 3), f['stage']))

    def _cc_raise_trigger(self):
        """緊湊穩態時何時開始升高：回 (原因, rem) 或 None。預設＝v3 原規則（PLACE_APPROACH 即升；CARRY_NAV 距直線段入口 ≤ v_carry×T_UP 提前升）；版本化子類可覆寫。"""
        lead = getattr(self, 'v_carry', 0.25) * self._ccd3('T_UP')
        rem = None if self.cpath is None or self.csig is None else float(self.cpath.x_lane) - float(self.csig)
        if self.pph == 'PLACE_APPROACH':
            return ('place_approach', rem)
        if self.pph == 'CARRY_NAV' and rem is not None and np.isfinite(rem) and rem <= lead:
            return ('early', rem)
        return None

    def _cc_out_phase_ok(self):
        """升高完成後哪些放置相位允許開始外伸（仍須外伸淨空閘通過）。預設＝v3 原規則：只在 PLACE_APPROACH。"""
        return self.pph == 'PLACE_APPROACH'

    def _start_raise(self, why, rem=None):
        self.cc.update(stage='raise', t_raise=float(self.t))
        self._ev('compact_raise_start', why=why, rem=None if rem is None or not np.isfinite(rem) else round(rem, 3))
        self.cc_ev.append(('raise', float(self.t)))

    def report(self):
        r = super().report()
        cc = r['compact_carry']
        cc.update(version='v3', raise_lead_m=getattr(self, 'v_carry', 0.25) * T_UP,
                  reach_gate=dict(trigger=self.reach_trig, monitor_min=None if not math.isfinite(self.reach_mon_min) else self.reach_mon_min, log=self.reach_log),
                  schedule='升高提前到距直線段入口 v_carry×T_UP 起（近身原地升高）；外伸仍在 PLACE_APPROACH、升高完成且外伸淨空閘通過後才開始')
        return r
