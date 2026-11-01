#!/usr/bin/env python3
"""抓牢離桌後轉到車頭前方持物（版本化子類，2026-11-01；Howard 補充要求 reviews/20261101_howard_front_carry.md、Codex 20261101_191812）。
基線＝連續展臂候選（bottle_w1_continuous_approach_ctrl，SHROW 子類），原 SHROW／側持／連續展臂版本都不改。
啟動條件：CARRY_NAV 中、抓持證據成立（judge.t_grasp）且原離桌判定成立（judge.t_lift_ok：離桌 ≥ 40 mm 連續 1 s）。不以 CLOSE 結束或進 CARRY 代替。
前持姿態（唯一候選，有機制依據、不掃參數）：把側持的底盤座標 TCP 位姿繞手臂第一軸（link_base z 軸，鉛直）旋轉，直到 TCP 落在通過第一軸、沿底盤 +x 的直線上。
  繞第一軸旋轉＝只改 q1（FK 數值核對 1e-16），手臂其他關節、瓶身直立、與 TCP 的距離與高度都不變，可達性與關節餘裕承接已驗證的側持姿態；
  bottle_w1_cont_nominal1 的側持：TCP 方位 89.25°、半徑 0.417 m ⇒ 轉 −89.25°，q1 1.556 → −0.002（限位 ±2.94）。
過渡：同一全身 GMPC 追底盤座標任務（tcp_rel_base），目標沿旋轉角 φ(t)＝φ_a＋(φ_b−φ_a)·min_jerk((t−t0)/T_tr) 逐時域步給出（承接目前目標，不跳）；
  T_tr＝1.875·|Δφ|／v_arm（min-jerk 峰值角速率＝既有手臂設定點速率上限 W_ARM['v_arm']）。
桌邊：進 PLACE_APPROACH 時由目前 φ 平順轉回 0（原側持放置姿態），轉回完成前不讓停車判定累計靜止（n_still_p 歸零）；
  停車位、下降與退離沿用原算法——放置時的構型就是原側持姿態，與停車位計算相容，不沿用前持構型去對桌。
不變：抓持、滑移、禁止接觸、淨空、Shield、CBF、第一拍列、資料期限、放置判準。前持會增加車頭前方包絡（瓶中心在底盤前方約 0.53 m），不預先宣稱更安全或更穩。
"""
import math

import numpy as np

import bottle_ref as BR
import bottle_s0b_ctrl as B
import bottle_w1_continuous_approach_ctrl as CA

V_ARM = float(B.W_ARM['v_arm'])


def rz(a):
    R = np.eye(4)
    R[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return R


class W1PlaceController(CA.W1PlaceController):
    group = 'G_SHROW_CONT_FRONT'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._Lb = self.K.fk(np.r_[0.0, 0.0, 0.0, self.q_arm_ready], 'link_base')     # 底盤→第一軸座標（與手臂關節無關）
        self._Lbi = np.linalg.inv(self._Lb)
        self.fc, self.fc_log, self.fc_ev = None, [], []

    def _fc_front_angle(self, T_side):
        p = self._Lbi @ T_side[:, 3]
        return -math.atan2(float(p[1]), float(p[0]))

    def _fc_phi(self, t):
        f = self.fc
        s = BR.min_jerk(min(max((t - f['t0']) / f['T_tr'], 0.0), 1.0)) if f['T_tr'] > 0 else 1.0
        return f['phi_a'] + (f['phi_b'] - f['phi_a']) * s

    def _fc_T(self, phi):
        return self._Lb @ rz(phi) @ self._Lbi @ self.fc['T_side']

    def _fc_done(self, t):
        return self.fc is not None and t >= self.fc['t0'] + self.fc['T_tr'] - 1e-9

    def _advance_inner(self, T_tcp, close_done, q):
        super()._advance_inner(T_tcp, close_done, q)
        if self.judge.stop is not None:
            return
        if self.fc is None and self.pph == 'CARRY_NAV' and self.judge.t_grasp is not None and self.judge.t_lift_ok is not None:
            T_side = np.array(self.st.T_carry_rel, float)
            phi = self._fc_front_angle(T_side)
            self.fc = dict(stage='to_front', t0=float(self.t), phi_a=0.0, phi_b=phi, T_tr=1.875 * abs(phi) / V_ARM, T_side=T_side, phi_front=phi)
            p = self._Lbi @ T_side[:, 3]
            self._ev('front_carry_start', phi_deg=round(math.degrees(phi), 2), T_tr=round(self.fc['T_tr'], 3), radius=round(float(math.hypot(p[0], p[1])), 4))
            self.fc_ev.append(('to_front', float(self.t)))
        elif self.fc is not None and self.fc['stage'] in ('to_front', 'front') and self.pph == 'PLACE_APPROACH':
            phi_now = self._fc_phi(self.t)
            self.fc.update(stage='to_side', t0=float(self.t), phi_a=phi_now, phi_b=0.0, T_tr=1.875 * abs(phi_now) / V_ARM)
            self._ev('front_carry_return', phi_deg=round(math.degrees(phi_now), 2), T_tr=round(self.fc['T_tr'], 3))
            self.fc_ev.append(('to_side', float(self.t)))
        elif self.fc is not None and self.fc['stage'] == 'to_front' and self._fc_done(self.t):
            self.fc['stage'] = 'front'
            self._ev('front_carry_reached')
            self.fc_ev.append(('front', float(self.t)))
        elif self.fc is not None and self.fc['stage'] == 'to_side' and self._fc_done(self.t):
            self.fc['stage'] = 'side'
            self._ev('front_carry_returned')
            self.fc_ev.append(('side', float(self.t)))
        if self.fc is not None and self.fc['stage'] == 'to_side' and self.pph == 'PLACE_APPROACH':
            self.n_still_p = 0                                   # 轉回側持完成前不累計停車靜止
        if self.fc is not None and self.n_step % 5 == 0:
            self.fc_log.append((round(float(self.t), 3), self.fc['stage'], round(math.degrees(self._fc_phi(self.t)), 3)))

    def _T_seq_place(self, qp, t_eff):
        T, rel = super()._T_seq_place(qp, t_eff)
        if self.fc is None or self.pph not in ('CARRY_NAV', 'PLACE_APPROACH', 'PARKED'):
            return T, rel
        return np.stack([self._fc_T(self._fc_phi(t_eff + k * B.DT)) for k in range(B.N + 1)]), rel

    def report(self):
        r = super().report()
        r['front_carry'] = dict(events=self.fc_ev, phi_front_deg=None if self.fc is None else round(math.degrees(self.fc['phi_front']), 3), v_arm=V_ARM,
                                log=self.fc_log, link_base_in_base=self._Lb[:3, 3].tolist(),
                                scope='抓持＋原離桌判定成立後繞第一軸轉到車頭前方；PLACE_APPROACH 轉回原側持放置姿態；停車與放置沿用原算法')
        return r
