#!/usr/bin/env python3
"""理想抓持綁定的判定純函式（執行器與測試共用；不依賴 Isaac）。
  rel_motion      夾爪殼相對門扇的運動：用**同一相對變換**相鄰兩步的平移／轉動差分（不是兩個剛體質心線速度相減——共同剛轉時那個差不為零）。
  classify_points 接觸點分類：到橫桿軸線段的距離 ≤ 半徑＋tol 算橫桿，其餘（門板、立柱）算非橫桿。
  BindGate        逐步餵入；禁止接觸一出現即鎖存（之後永不放行）；全部條件連續成立才放行。"""
import math

import numpy as np


def rel_motion(Trel_prev, Trel_now, dt):
    dp = float(np.linalg.norm(Trel_now[:3, 3] - Trel_prev[:3, 3])) / dt
    c = float(np.clip((np.trace(Trel_prev[:3, :3].T @ Trel_now[:3, :3]) - 1) / 2, -1, 1))
    return dp, math.acos(c) / dt


def classify_points(points, forces, bar_c, bar_u, bar_hl, bar_r, tol=0.003):
    """回 (橫桿合力大小, 非橫桿合力大小, 非橫桿點數)。points (n,3)、forces (n,) 或 (n,3)。"""
    P = np.asarray(points, float).reshape(-1, 3)
    F = np.asarray(forces, float)
    F = np.abs(F).reshape(len(P), -1).sum(axis=1) if len(P) else np.zeros(0)
    if not len(P):
        return 0.0, 0.0, 0
    s = np.clip((P - bar_c) @ bar_u, -bar_hl, bar_hl)
    d = np.linalg.norm(P - (bar_c + s[:, None] * bar_u), axis=1)
    on = d <= bar_r + tol
    return float(F[on].sum()), float(F[~on].sum()), int(((~on) & (F > 0)).sum())


class BindGate:
    def __init__(self, cond, dt):
        self.c, self.dt = cond, dt
        self.latched, self.n_still = None, 0

    def forbid(self, t, why, **kw):
        if self.latched is None:
            self.latched = dict(t=round(float(t), 3), why=why, **kw)

    def step(self, t, rel_lin, rel_ang, other_contact_n, finger_nonbar_n):
        if other_contact_n > self.c['forbidden_n']:
            self.forbid(t, 'nonfinger_contact', force_n=round(float(other_contact_n), 4))
        if finger_nonbar_n > self.c['forbidden_n']:
            self.forbid(t, 'finger_nonbar_contact', force_n=round(float(finger_nonbar_n), 4))
        still = rel_lin <= self.c['rel_lin_mps'] and rel_ang <= self.c['rel_ang_rps']
        self.n_still = self.n_still + 1 if still else 0

    def decide(self, pos_m, rot_deg, finger_q, finger_bar_n, door_pre_deg):
        c = self.c
        ok = dict(no_forbidden_contact=self.latched is None, pos=pos_m <= c['pos_m'], rot=rot_deg <= c['rot_deg'], rel_still=self.n_still * self.dt >= c['rel_still_s'] - 1e-9,
                  finger_q=all(c['finger_q_m'][0] <= q <= c['finger_q_m'][1] for q in finger_q), finger_bar_contact=all(f > c['finger_contact_n'] for f in finger_bar_n), door_pre=door_pre_deg <= c['door_pre_deg'])
        return bool(all(ok.values())), ok
