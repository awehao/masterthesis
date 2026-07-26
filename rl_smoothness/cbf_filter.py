"""CBF-QP safety filter for a 2D omni (single-integrator) robot.

Given a nominal velocity command u_nom, return the closest SAFE command:

    u_safe = argmin_u ||u - u_nom||^2  +  rho * sum_i s_i^2
             s.t.  2 (p - o_i)·u + s_i >= 2 (p - o_i)·v_obs_i - alpha * h_i   (discrete CBF)
                   s_i >= 0                                                   (soft slack -> always feasible)
                   -v_max <= u_x, u_y <= v_max                               (actuator box)

with barrier   h_i = ||p - o_i||^2 - r_eff_i^2 ,  r_eff_i = r_obs_i + margin .

This is the SAME horizon-CBF idea used in gmpc.py, reduced to the current step
for a holonomic base -> a small QP solved with OSQP. It is the "CBF 保命" layer:
whatever (RL) command comes in, the output is projected onto the safe set.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import osqp


class CBFFilter:
    def __init__(self, v_max=0.6, alpha=3.0, margin=0.38, slack_rho=1.0e4):
        self.v_max = float(v_max)
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.rho = float(slack_rho)

    def filter(self, p, u_nom, obstacles):
        """p: (2,) robot xy.  u_nom: (2,) desired vx,vy.
        obstacles: list of dict {x,y,vx,vy,r}.  Returns (u_safe (2,), info)."""
        p = np.asarray(p, float)
        u_nom = np.asarray(u_nom, float)
        m = len(obstacles)
        n = 2 + m                                   # [vx, vy, s_0..s_{m-1}]

        # --- objective: ||u - u_nom||^2 + rho * ||s||^2 ---
        P = sp.diags(np.concatenate([[2.0, 2.0], np.full(m, 2.0 * self.rho)])).tocsc()
        q = np.concatenate([-2.0 * u_nom, np.zeros(m)])

        rows, hmin = [], np.inf
        l, u = [], []
        # CBF rows
        for i, o in enumerate(obstacles):
            d = p - np.array([o['x'], o['y']])      # p - o_i
            r_eff = float(o['r']) + self.margin
            h = float(d @ d - r_eff * r_eff)
            hmin = min(hmin, h)
            vobs = np.array([o.get('vx', 0.0), o.get('vy', 0.0)])
            a = np.zeros(n)
            a[0:2] = 2.0 * d                        # 2(p-o)·u
            a[2 + i] = 1.0                           # + s_i
            rows.append(a)
            l.append(2.0 * d @ vobs - self.alpha * h)   # >= b_i
            u.append(np.inf)
        # slack >= 0
        for i in range(m):
            a = np.zeros(n); a[2 + i] = 1.0
            rows.append(a); l.append(0.0); u.append(np.inf)
        # velocity box
        for k in range(2):
            a = np.zeros(n); a[k] = 1.0
            rows.append(a); l.append(-self.v_max); u.append(self.v_max)

        A = sp.csc_matrix(np.array(rows)) if rows else sp.csc_matrix((0, n))
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=np.array(l), u=np.array(u),
                   verbose=False, polish=False, eps_abs=1e-4, eps_rel=1e-4)
        res = prob.solve()
        if res.info.status_val in (1, 2) and res.x is not None:   # solved / solved inaccurate
            u_safe = np.clip(res.x[:2], -self.v_max, self.v_max)
            slack = float(np.sum(np.maximum(res.x[2:], 0.0)))
        else:                                        # fallback: clip nominal
            u_safe = np.clip(u_nom, -self.v_max, self.v_max)
            slack = float('nan')
        intervention = float(np.linalg.norm(u_safe - np.clip(u_nom, -self.v_max, self.v_max)))
        return u_safe, dict(h_min=hmin, slack=slack, intervention=intervention)
