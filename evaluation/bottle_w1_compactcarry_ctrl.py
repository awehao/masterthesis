#!/usr/bin/env python3
"""低、近身、直立的運送姿態（版本化子類，2026-11-01；Howard 修正 reviews/20261101_howard_compact_carry_and_motion.md）。基線＝連續展臂（bottle_w1_continuous_approach_ctrl）；
舊前持（bottle_w1_frontcarry_ctrl，只轉 q1、保留高度與半徑）只作既有證據，不是本版。
緊湊姿態（bottle_w1_compact_pose.py v2 封存）：TCP 底盤座標 (0.0, 0.2, 0.6)、姿態同側持（ψ 0，瓶身鉛直）；選法＝瓶身＋夾爪包絡最大水平半徑最小（1 cm 等級）→ 瓶底最低，
  且從側持沿直線過渡路徑 IK 延續可達（每步 IK 收斂、關節餘裕 ≥ 0.05、自碰 ≥ 8.39 mm、持瓶↔連桿 ≥ 5.5 mm、車體頂淨空、不跳分支）。
  名目參考：包絡半徑 0.444 → 0.212 m、瓶底 0.649 → 0.498 m；過渡最大關節變化 1.18 rad ⇒ T_tr＝1.875·Δq_max／v_arm＝2.22 s。
收近時機：CARRY_NAV 中，抓持證據（judge.t_grasp）、原離桌判定（judge.t_lift_ok）都成立，且模型瓶心（TCP FK∘judge.T_rel）已離開第一桌桌面投影外擴 0.05 m（清出桌緣）後才開始降低。
收近路徑：TCP 位置 min-jerk 直線（側持 → 緊湊），姿態不變（瓶身鉛直）。
桌邊：進 PLACE_APPROACH 時轉回原側持放置姿態；路徑先升高（xy 不動）再外伸（高度保持），避免低持的瓶外伸時掃過第二桌緣；轉回完成前不累計停車靜止。
  停車位、下降、退離沿用原算法（放置構型＝原側持姿態）。
不變：抓持、滑移、禁止接觸、淨空、CBF、Shield、第一拍列、資料期限、放置判準。緊湊姿態不等於二維掃描看得見所有高位障礙；包絡在車體圓內也不等於已受 Shield 保護（Shield 半徑 0.39 m 只在掃描面）。
"""
import numpy as np

import bottle_ref as BR
import bottle_s0b_ctrl as B
import bottle_w1_continuous_approach_ctrl as CA

TCP_C = np.array([0.0, 0.2, 0.6])
T_TR = 2.22
TABLE1 = ((0.0, 0.61), (0.80, 0.50))                    # 第一桌桌面中心與尺寸（bottle_s1_geom.table_boxes 的 table_top）
EDGE_PAD = 0.05


class W1PlaceController(CA.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cc, self.cc_ev, self.cc_log = None, [], []

    def _bottle_xy(self, q):
        Tb = self.K.fk(np.asarray(q, float), 'link_tcp') @ self.judge.T_rel
        return Tb[:2, 3]

    def _cleared_table1(self, q):
        if self.judge.T_rel is None:
            return False
        p = self._bottle_xy(q)
        (cx, cy), (sx, sy) = TABLE1
        return abs(p[0] - cx) > sx / 2 + EDGE_PAD or abs(p[1] - cy) > sy / 2 + EDGE_PAD

    def _cc_s(self, t):
        f = self.cc
        return BR.min_jerk(min(max((t - f['t0']) / f['T_tr'], 0.0), 1.0))

    def _cc_T(self, t):
        f, s = self.cc, self._cc_s(t)
        pa, pb = f['T_a'][:3, 3], f['T_b'][:3, 3]
        T = f['T_a'].copy()
        if f['path'] == 'straight':
            T[:3, 3] = pa + (pb - pa) * s
        else:                                                    # raise_then_out：前半段只升高，後半段只外伸
            if s <= 0.5:
                T[:3, 3] = [pa[0], pa[1], pa[2] + (pb[2] - pa[2]) * (s / 0.5)]
            else:
                u = (s - 0.5) / 0.5
                T[:3, 3] = [pa[0] + (pb[0] - pa[0]) * u, pa[1] + (pb[1] - pa[1]) * u, pb[2]]
        return T

    def _cc_done(self, t):
        return self.cc is not None and t >= self.cc['t0'] + self.cc['T_tr'] - 1e-9

    def _advance_inner(self, T_tcp, close_done, q):
        super()._advance_inner(T_tcp, close_done, q)
        if self.judge.stop is not None:
            return
        if (self.cc is None and self.pph == 'CARRY_NAV' and self.judge.t_grasp is not None and self.judge.t_lift_ok is not None
                and self._cleared_table1(q)):
            T_side = np.array(self.st.T_carry_rel, float)
            T_c = T_side.copy()
            T_c[:3, 3] = TCP_C
            self.cc = dict(stage='to_compact', t0=float(self.t), T_a=T_side, T_b=T_c, T_side=T_side, T_tr=T_TR, path='straight')
            self._ev('compact_start', bottle_xy=[round(float(v), 3) for v in self._bottle_xy(q)])
            self.cc_ev.append(('to_compact', float(self.t)))
        elif self.cc is not None and self.cc['stage'] in ('to_compact', 'compact') and self.pph == 'PLACE_APPROACH':
            T_now = self._cc_T(self.t)
            self.cc.update(stage='to_side', t0=float(self.t), T_a=T_now, T_b=self.cc['T_side'], T_tr=T_TR, path='raise_then_out')
            self._ev('compact_return')
            self.cc_ev.append(('to_side', float(self.t)))
        elif self.cc is not None and self.cc['stage'] == 'to_compact' and self._cc_done(self.t):
            self.cc['stage'] = 'compact'
            self._ev('compact_reached')
            self.cc_ev.append(('compact', float(self.t)))
        elif self.cc is not None and self.cc['stage'] == 'to_side' and self._cc_done(self.t):
            self.cc['stage'] = 'side'
            self._ev('compact_returned')
            self.cc_ev.append(('side', float(self.t)))
        if self.cc is not None and self.cc['stage'] == 'to_side' and self.pph == 'PLACE_APPROACH':
            self.n_still_p = 0                                   # 轉回放置姿態完成前不累計停車靜止
        if self.cc is not None and self.n_step % 5 == 0:
            self.cc_log.append((round(float(self.t), 3), self.cc['stage'], round(float(self._cc_s(self.t)), 4)))

    def _T_seq_place(self, qp, t_eff):
        T, rel = super()._T_seq_place(qp, t_eff)
        if self.cc is None or self.pph not in ('CARRY_NAV', 'PLACE_APPROACH', 'PARKED'):
            return T, rel
        return np.stack([self._cc_T(t_eff + k * B.DT) for k in range(B.N + 1)]), rel

    def report(self):
        r = super().report()
        r['compact_carry'] = dict(events=self.cc_ev, tcp_compact=TCP_C.tolist(), T_tr=T_TR, log=self.cc_log,
                                  scope='抓持＋離桌判定＋清出第一桌桌緣後收近降低到緊湊姿態；PLACE_APPROACH 先升高再外伸回原側持放置姿態；停車與放置沿用原算法')
        return r
