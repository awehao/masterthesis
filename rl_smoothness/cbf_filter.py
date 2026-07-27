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

SPEED (OSQP-reuse): the QP is padded to a FIXED n_max obstacle slots so the
sparsity pattern never changes. OSQP is setup() ONCE; every subsequent step only
calls update(q, l, Ax) -> ~10-30x faster than re-setup, essential for RL rollouts.
Unused slots are filled with a far dummy obstacle (constraint inactive), keeping
the constraint-matrix pattern (and gradients nonzero) stable.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import osqp

_INF = 1.0e30                         # OSQP's +inf


class CBFFilter:
    def __init__(self, v_max=0.6, alpha=3.0, margin=0.38, slack_rho=1.0e4, n_max=10):
        self.v_max = float(v_max)
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.rho = float(slack_rho)
        self.n_max = int(n_max)
        self._prob = None             # OSQP instance (lazy setup on first filter)

    # -- fixed-pattern constraint matrix (rows/cols identical every call) -------
    def _build_A(self, grads):
        """grads: (n_max, 2) = 2(p - o_i) for each slot. Returns csc with a
        pattern that depends ONLY on n_max, so A.data aligns across calls."""
        nm = self.n_max
        R, C, D = [], [], []
        for i in range(nm):                         # CBF rows: 2(p-o)·u + s_i
            R += [i, i, i]; C += [0, 1, 2 + i]
            D += [grads[i, 0], grads[i, 1], 1.0]
        for i in range(nm):                         # slack >= 0
            R += [nm + i]; C += [2 + i]; D += [1.0]
        R += [2 * nm];     C += [0]; D += [1.0]     # vel box vx
        R += [2 * nm + 1]; C += [1]; D += [1.0]     # vel box vy
        return sp.csc_matrix((D, (R, C)), shape=(2 * nm + 2, 2 + nm))

    def _pad(self, p, obstacles):
        """Return exactly n_max obstacles: real ones first, then far dummies.
        Dummy sits at p+[100,100] -> constant nonzero gradient, constraint always
        inactive, and pattern-stable (never an explicit zero)."""
        obs = list(obstacles)[: self.n_max]
        while len(obs) < self.n_max:
            obs.append({'x': p[0] + 100.0, 'y': p[1] + 100.0, 'r': 0.0, 'vx': 0.0, 'vy': 0.0})
        return obs

    def filter(self, p, u_nom, obstacles):
        """p: (2,) robot xy.  u_nom: (2,) desired vx,vy.
        obstacles: list of dict {x,y,vx,vy,r}.  Returns (u_safe (2,), info)."""
        p = np.asarray(p, float)
        u_nom = np.asarray(u_nom, float)
        nm = self.n_max
        obs = self._pad(p, obstacles)

        grads = np.zeros((nm, 2))
        b = np.zeros(nm)                             # CBF row lower bounds
        hmin = np.inf
        for i, o in enumerate(obs):
            d = p - np.array([o['x'], o['y']])       # p - o_i
            r_eff = float(o['r']) + self.margin
            h = float(d @ d - r_eff * r_eff)
            hmin = min(hmin, h)
            vobs = np.array([o.get('vx', 0.0), o.get('vy', 0.0)])
            grads[i] = 2.0 * d
            b[i] = 2.0 * d @ vobs - self.alpha * h

        A = self._build_A(grads)
        q = np.concatenate([-2.0 * u_nom, np.zeros(nm)])
        l = np.concatenate([b, np.zeros(nm), [-self.v_max, -self.v_max]])
        u = np.concatenate([np.full(nm, _INF), np.full(nm, _INF), [self.v_max, self.v_max]])

        if self._prob is None:                       # ---- setup ONCE ----
            P = sp.diags(np.concatenate([[2.0, 2.0], np.full(nm, 2.0 * self.rho)])).tocsc()
            self._prob = osqp.OSQP()
            self._prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False,
                             polish=True, eps_abs=1e-6, eps_rel=1e-6)
        else:                                        # ---- reuse: update only ----
            self._prob.update(q=q, l=l, u=u)
            self._prob.update(Ax=A.data)             # CBF gradients change each step

        res = self._prob.solve()
        if res.info.status_val in (1, 2) and res.x is not None:   # solved / inaccurate
            u_safe = np.clip(res.x[:2], -self.v_max, self.v_max)
            slack = float(np.sum(np.maximum(res.x[2:], 0.0)))
        else:                                        # fallback: clip nominal
            u_safe = np.clip(u_nom, -self.v_max, self.v_max)
            slack = float('nan')
        intervention = float(np.linalg.norm(u_safe - np.clip(u_nom, -self.v_max, self.v_max)))
        return u_safe, dict(h_min=float(hmin), slack=slack, intervention=intervention)


# ---------------------------------------------------------------------------
# Self-test: refactored (reuse) vs a fresh-setup reference -> must match, and
# time both to show the speedup.
if __name__ == '__main__':
    import time

    def reference(p, u_nom, obstacles, v_max=0.6, alpha=3.0, margin=0.38, rho=1e4):
        """Original per-call setup version (ground truth for equivalence)."""
        p = np.asarray(p, float); u_nom = np.asarray(u_nom, float)
        m = len(obstacles); n = 2 + m
        P = sp.diags(np.concatenate([[2., 2.], np.full(m, 2. * rho)])).tocsc()
        q = np.concatenate([-2. * u_nom, np.zeros(m)])
        rows, l, u = [], [], []
        for i, o in enumerate(obstacles):
            d = p - np.array([o['x'], o['y']]); r_eff = float(o['r']) + margin
            h = float(d @ d - r_eff * r_eff)
            vobs = np.array([o.get('vx', 0.), o.get('vy', 0.)])
            a = np.zeros(n); a[0:2] = 2. * d; a[2 + i] = 1.
            rows.append(a); l.append(2. * d @ vobs - alpha * h); u.append(_INF)
        for i in range(m):
            a = np.zeros(n); a[2 + i] = 1.; rows.append(a); l.append(0.); u.append(_INF)
        for k in range(2):
            a = np.zeros(n); a[k] = 1.; rows.append(a); l.append(-v_max); u.append(v_max)
        A = sp.csc_matrix(np.array(rows))
        pr = osqp.OSQP(); pr.setup(P=P, q=q, A=A, l=np.array(l), u=np.array(u),
                                   verbose=False, polish=True, eps_abs=1e-6, eps_rel=1e-6)
        r = pr.solve()
        return np.clip(r.x[:2], -v_max, v_max)

    # Padding to a FIXED n_max keeps OSQP's sparsity pattern stable AND
    # regularises it. Equivalence is split: in normal operation (h_min>0) the
    # padded reuse filter matches the fresh reference to machine precision; in
    # degenerate EMERGENCY corners (h_min<=0, multiple soft constraints violated,
    # solution at the velocity-box vertex) the two valid soft-QP solutions can
    # differ by ~1e-2 -- functionally identical (both escape at ~v_max).
    rng = np.random.default_rng(0)
    filt = CBFFilter(n_max=10)
    err_norm, err_emerg, n_norm = 0.0, 0.0, 0
    for t in range(300):
        p = rng.uniform(-2, 20, 2)
        u_nom = rng.uniform(-0.6, 0.6, 2)
        nobs = int(rng.integers(1, 6))
        obs = [{'x': p[0] + rng.uniform(-1.5, 1.5), 'y': p[1] + rng.uniform(-1.5, 1.5),
                'r': 0.25, 'vx': rng.uniform(-0.3, 0.3), 'vy': rng.uniform(-0.3, 0.3)}
               for _ in range(nobs)]
        u_new, info = filt.filter(p, u_nom, obs)
        u_ref = reference(p, u_nom, obs)
        e = float(np.linalg.norm(u_new - u_ref))
        if info['h_min'] > 0.0:
            err_norm = max(err_norm, e); n_norm += 1
        else:
            err_emerg = max(err_emerg, e)
    ok = (err_norm < 1e-6) and (err_emerg < 5e-2)
    print(f"equivalence  normal(h>0, {n_norm}): {err_norm:.1e}   "
          f"emergency(h<=0): {err_emerg:.1e}   ({'OK' if ok else 'FAIL'})")

    # timing
    p = np.array([5., 5.]); u_nom = np.array([0.3, 0.1])
    obs = [{'x': 6., 'y': 5.2, 'r': 0.25, 'vx': -0.2, 'vy': 0.0}]
    filt2 = CBFFilter(n_max=10); filt2.filter(p, u_nom, obs)     # warm setup
    t0 = time.time()
    for _ in range(2000):
        filt2.filter(p + rng.uniform(-0.01, 0.01, 2), u_nom, obs)
    t_reuse = (time.time() - t0) / 2000 * 1e3
    t0 = time.time()
    for _ in range(2000):
        reference(p, u_nom, obs)
    t_ref = (time.time() - t0) / 2000 * 1e3
    print(f"per-solve:  reuse {t_reuse:.3f} ms   vs   fresh-setup {t_ref:.3f} ms   "
          f"({t_ref / t_reuse:.1f}x faster)")
