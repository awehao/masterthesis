#!/usr/bin/env python3
"""任期入口＋底盤箱界微量投影（door_m4r_7；Codex 20261030_000022 修正版 A）。door_k3m_applied_src_t 不改，以子類別擴充。
命令自求解端車體→世界→轉接器以最新偏航轉回車體，旋轉捨入使貼箱界的底盤分量超出權威形狀界（車體每軸 0.035255、|ω| 0.1999、tol 1e-9）。
進鏈前（已轉成實際車體座標）：底盤三軸任一超界量 0 < e ≤ SNAP_MAX（1e-6）⇒ 投影回箱界；e > SNAP_MAX ⇒ **不修正**（照原值進鏈，由權威照原規則判）；
非有限不處理（交給既有入口與權威）。只改進鏈副本，不改權威持有的原封裝；保存原值、修正值、超界量、source_seq、mode_seq、tenure。
"""
from __future__ import annotations

import math

from door_k3m_applied_src_t import EnvIntakeT

V_LIN, V_ANG, SNAP_MAX = 0.035255, 0.1999, 1e-6


class EnvIntakeTB(EnvIntakeT):
    def __init__(self, *a, v_lin=V_LIN, v_ang=V_ANG, snap_max=SNAP_MAX, **kw):
        super().__init__(*a, **kw)
        self.lim = (float(v_lin), float(v_lin), float(v_ang))
        self.snap_max = float(snap_max)
        self.n_snapped = self.n_over = 0
        self.snap_log = []
        self.max_excess_snapped = 0.0

    def receive(self, env, chain, t, filt=lambda u: u):
        if isinstance(env, dict) and isinstance(env.get('u'), (list, tuple)) and len(env['u']) >= 3:
            u = [float(x) for x in env['u']]
            if all(math.isfinite(x) for x in u[:3]):
                ex = [abs(u[i]) - self.lim[i] for i in range(3)]
                big = max(ex)
                if 0.0 < big <= self.snap_max:
                    u2 = list(u)
                    for i in range(3):
                        if ex[i] > 0.0:
                            u2[i] = math.copysign(self.lim[i], u[i])
                    self.n_snapped += 1
                    self.max_excess_snapped = max(self.max_excess_snapped, big)
                    if len(self.snap_log) < 200:
                        self.snap_log.append({'t': float(t), 'source_seq': env.get('source_seq'), 'mode_seq': env.get('mode_seq'),
                                              'tenure': env.get('tenure'), 'u_in': u[:3], 'u_out': u2[:3], 'excess': big})
                    env = dict(env, u=u2 + list(env['u'][len(u2):]))
                elif big > self.snap_max:
                    self.n_over += 1
        return super().receive(env, chain, t, filt=filt)

    def report(self):
        r = super().report()
        r.update(box_snap={'limits': list(self.lim), 'snap_max': self.snap_max, 'n_snapped': self.n_snapped,
                           'n_over_not_snapped': self.n_over, 'max_excess_snapped': self.max_excess_snapped, 'log': self.snap_log[:50]})
        return r
