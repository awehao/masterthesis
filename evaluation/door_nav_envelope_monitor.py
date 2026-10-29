#!/usr/bin/env python3
"""導航段整機包絡線上核對（Codex 20261029_145654：交棒及導航期間都核**實際構型**，不因參考寫 STOW 就認定已收好）。
幾何同 door_nav_envelope_v1（URDF 碰撞幾何全部頂點、圓柱外接）；每連桿預取凸包頂點（水平半徑最大值必在凸包頂點）⇒ 單次約 0.8 ms。
ρ(q)＝max 手臂＋夾爪＋手指幾何的水平半徑（底盤中心＝base_footprint 原點；與底盤位置、偏航無關）。
EnvelopeMonitor：
  admit(q_arm, q_finger)        交權前准入：ρ ≤ r_env − admit_slack；
  step(q_arm, q_finger, t)      導航期間逐週期：ρ > r_env ⇒ 鎖存違反（first 記錄 t、ρ）；不自行解除。
q 須為量測值；非有限 ⇒ 視為違反（不略過）。
"""
from __future__ import annotations

import math

import numpy as np

R_ENV = 0.39
ADMIT_SLACK = 0.005


def hull_backend():
    from scipy.spatial import ConvexHull
    import door_nav_envelope_v1 as E
    import door_offline_check_v1 as C
    CL = E.full_clouds(open(C.URDF).read())
    arm = [n for n in CL if n.startswith('link') or n.startswith('uflite')]
    Hv = {n: CL[n][ConvexHull(CL[n]).vertices] for n in arm}

    def rho(q_arm, q_finger):
        qf, best = [0.0, 0.0, 0.0] + [float(v) for v in q_arm], (0.0, None)
        for n in arm:
            T = np.asarray(C.K.fk(qf, n))
            P = Hv[n] + (C.FINGER_AXIS[n] * q_finger if n in C.FINGER_AXIS else 0.0)
            W = (T[:3, :3] @ P.T).T + T[:3, 3]
            r = float(np.hypot(W[:, 0], W[:, 1]).max())
            if r > best[0]:
                best = (r, n)
        return best
    return rho


class EnvelopeMonitor:
    def __init__(self, rho_fn=None, r_env=R_ENV, admit_slack=ADMIT_SLACK):
        self.rho_fn = rho_fn if rho_fn is not None else hull_backend()
        self.r_env, self.admit_slack = float(r_env), float(admit_slack)
        self.violated = False
        self.first = None
        self.n = 0
        self.rho_max = 0.0

    def _rho(self, q_arm, q_finger):
        if not (all(math.isfinite(float(v)) for v in q_arm) and math.isfinite(float(q_finger))):
            return math.inf, 'nonfinite'
        return self.rho_fn(q_arm, q_finger)

    def admit(self, q_arm, q_finger):
        r, n = self._rho(q_arm, q_finger)
        return r <= self.r_env - self.admit_slack, {'rho_m': r, 'link': n, 'limit_m': self.r_env - self.admit_slack}

    def step(self, q_arm, q_finger, t):
        r, n = self._rho(q_arm, q_finger)
        self.n += 1
        self.rho_max = max(self.rho_max, r)
        if r > self.r_env and not self.violated:
            self.violated = True
            self.first = {'t': float(t), 'rho_m': r, 'link': n}
        return not self.violated, r

    def report(self):
        return {'version': 'door_nav_envelope_monitor/1', 'r_env_m': self.r_env, 'admit_slack_m': self.admit_slack,
                'n': self.n, 'rho_max_m': self.rho_max, 'violated': self.violated, 'first': self.first}
