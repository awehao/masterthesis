"""Faithful closed-loop 2D env: REAL GMPC + in-loop SmacPlanner(A*) + real map/
obstacles. A fast replica of the gz benchmark for training an RL smoothing
residual (CBF stays as the safety layer).

Fidelity to the real system:
  * controller  = the actual gmpc.py (imported), params from gmpc_params.yaml
  * global path = A* (= SmacPlanner2D core) on costmap = walls + DISCOVERED
                  static pillars, re-planned every replan_period; dynamic
                  obstacles are NOT in the costmap (mimics /scan_filtered)
  * dynamic obs = dyn_obs_0/1/2/5 ping-pong (randomised phase), fed to the CBF
                  with perception NOISE (mimics the tracker/KF jitter -> wiggle)
  * static CBF  = nearest map wall point (v=0, margin 0.33), like _publish_static
  * margins     = dynamic 0.38 / static 0.33 (as in the real config)

action (RL residual) is a body-twist delta added to the GMPC output; the result
is re-projected through a CBF safety filter. action=None -> pure GMPC baseline.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, '/home/howardchen/masterthesis/src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.gmpc import GMPC, GMPCConfig            # noqa: E402
from ammr_wholebody_mpc import se2                              # noqa: E402
import gmpc_fast                                                # noqa: E402,F401
# ^ training-only patch: rebuilds the GMPC constraint matrix directly in CSC
#   instead of scipy lil per-block assignment (bit-identical, ~500x faster).

from scenario import Scenario, DYN, PILLARS, START, GOAL, ROBOT_R   # noqa: E402


def _mk_cfg():
    """GMPCConfig matching src/ammr_wholebody_mpc/config/gmpc_params.yaml."""
    return GMPCConfig(
        N=20, dt=0.05,
        u_min=np.array([-0.20, -0.25, -0.80]), u_max=np.array([0.35, 0.25, 0.80]),
        a_max=np.array([0.8, 0.6, 1.2]),
        Q=np.diag([15., 15., 7.]), R=np.diag([2., 2., 1.]), Qf=5 * np.diag([15., 15., 7.]),
        S=np.diag([15., 15., 8.]),
        cbf_alpha=3.0, cbf_safe_margin=0.38, cbf_slack_weight=5e2, cbf_eps0_scale=30.0,
        cbf_danger_thresh=0.4, cbf_Q_min_scale=0.20, cbf_slack_max_scale=20.0)


class RealAvoidEnv:
    V_NOM = 0.22
    STATIC_MARGIN = 0.33
    STATIC_RANGE = 1.2
    PILLAR_SENSE = 4.0                 # pillar becomes "known" to planner within this
    CBF_DYN_RANGE = 3.0
    REPLAN_EVERY = int(3.0 / 0.05)     # replan_period 3.0 s

    def _plan(self):
        raw = self.sc.astar(self.pose[:2], np.array(GOAL),
                            self.sc.occ_with_pillars(self.known_pillars))
        return self.sc.smooth_path(raw)                  # = SmacPlanner2D + SimpleSmoother

    def __init__(self, obs_noise_pos=0.05, obs_noise_vel=0.20, max_steps=3000, seed=0,
                 lag_beta=0.0):
        self.sc = Scenario()
        self.cfg = _mk_cfg()
        self.gmpc = GMPC(self.cfg)
        self.dt = self.cfg.dt
        self.npos, self.nvel = obs_noise_pos, obs_noise_vel
        # first-order lag: lumps gz's velocity_smoother + mass/inertia + motor lag
        # (a natural low-pass the pure-kinematic 2D sim lacks). u_app = b*u_app + (1-b)*u.
        # Calibrate b so the 2D wiggle matches the gz benchmark (~85 deg/m).
        self.lag_beta = float(lag_beta)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.act_dim = 3
        self.reset(seed)

    # ------------------------------------------------------------------
    def reset(self, seed=None, start_frac=0.0):
        """start_frac > 0 spawns the robot that fraction along the planned route
        (heading along the path tangent) instead of always at START. Training
        needs this: a full run is ~3000 steps, so fixed-start episodes give very
        few episodes and almost no state diversity. Evaluation keeps frac=0."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pose = np.array([START[0], START[1], np.deg2rad(45)])   # x,y,theta
        self.xi_prev = np.zeros(3)
        self.u_app = np.zeros(3)
        self.t = 0
        self.phase = self.rng.uniform(0, 20, size=len(DYN))          # random dyn phase
        self.known_pillars = []
        self._discover()
        self.path = self._plan()
        if start_frac > 0.0:
            idx = int(np.clip(start_frac, 0.0, 0.92) * (len(self.path) - 1))
            p = self.path[idx]
            nxt = self.path[min(idx + 3, len(self.path) - 1)]
            th = float(np.arctan2(nxt[1] - p[1], nxt[0] - p[0]))
            self.pose = np.array([p[0], p[1], th])
            self._discover()                     # pillars visible from the new spot
            self.path = self._plan()             # replan from here
        self.log = dict(u=[], clr=[], interv=[], xy=[])
        return self._obs()

    def _discover(self):
        for p in PILLARS:
            if p not in self.known_pillars and \
               np.hypot(p[0] - self.pose[0], p[1] - self.pose[1]) < self.PILLAR_SENSE:
                self.known_pillars.append(p)

    # ------------------------------------------------------------------
    def _dyn_true(self):
        out = []
        for o, ph in zip(DYN, self.phase):
            pos, vel = self.sc.dyn_at(o, self.t * self.dt + ph)
            out.append((pos, vel, o['r']))
        return out

    def _obstacles_for_cbf(self):
        """dynamic (noisy, margin 0.38) + nearest wall (v=0, 0.33) + near pillars."""
        obs = []
        p = self.pose[:2]
        for pos, vel, r in self._dyn_true():
            if np.hypot(pos[0] - p[0], pos[1] - p[1]) > self.CBF_DYN_RANGE + 1.5:
                continue
            obs.append(dict(x=pos[0] + self.rng.normal(0, self.npos),
                            y=pos[1] + self.rng.normal(0, self.npos),
                            vx=vel[0] + self.rng.normal(0, self.nvel),
                            vy=vel[1] + self.rng.normal(0, self.nvel), radius=r))  # margin->0.38 default
        w, d = self.sc.nearest_wall(p)
        if d < self.STATIC_RANGE:
            obs.append(dict(x=w[0], y=w[1], vx=0., vy=0., radius=0.05, margin=self.STATIC_MARGIN))
        for (x, y, r) in PILLARS:
            if np.hypot(x - p[0], y - p[1]) < self.STATIC_RANGE + r:
                obs.append(dict(x=x, y=y, vx=0., vy=0., radius=r, margin=self.STATIC_MARGIN))
        return obs

    def _ref_window(self):
        path = self.path
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        i0 = int(np.argmin(np.linalg.norm(path - self.pose[:2], axis=1)))
        ds = self.V_NOM * self.dt
        tgt = s[i0] + ds * np.arange(self.cfg.N + 1)
        xs = np.interp(tgt, s, path[:, 0]); ys = np.interp(tgt, s, path[:, 1])
        th = np.arctan2(np.gradient(ys), np.gradient(xs))
        X = np.array([se2.from_xytheta(xs[k], ys[k], th[k]) for k in range(self.cfg.N)])
        xi = np.zeros((self.cfg.N, 3))
        for k in range(self.cfg.N):
            dth = (th[k + 1] - th[k] + np.pi) % (2 * np.pi) - np.pi
            xi[k] = [self.V_NOM, 0.0, dth / self.dt]
        return X, xi

    def _obs(self):
        """state for the RL policy (compact)."""
        gv = (np.array(GOAL) - self.pose[:2]) / 20.0
        parts = [gv, [np.cos(self.pose[2]), np.sin(self.pose[2])], self.xi_prev / 0.6]
        p = self.pose[:2]
        dyn = sorted(self._dyn_true(), key=lambda o: np.hypot(o[0][0] - p[0], o[0][1] - p[1]))[:3]
        for pos, vel, r in dyn:
            parts.append((pos - p) / 20.0); parts.append(vel / 0.6)
        while len(parts) < 3 + 3 * 2:
            parts.append(np.zeros(2))
        return np.concatenate([np.ravel(x) for x in parts]).astype(np.float32)

    # ------------------------------------------------------------------
    def step(self, action=None):
        X_ref, xi_ref = self._ref_window()
        X_now = se2.from_xytheta(*self.pose)
        obstacles = self._obstacles_for_cbf()
        res = self.gmpc.solve(X_now, X_ref, xi_ref, self.xi_prev, obstacles)
        u = res.u_opt.copy()
        interv = 0.0
        if action is not None:                       # RL residual + CBF re-filter
            u_res = u + np.clip(np.asarray(action, float), -0.2, 0.2)
            u = self._cbf_refilter(u_res, obstacles)
            interv = float(np.linalg.norm(u - u_res))

        # first-order lag (gz velocity_smoother + physics), then integrate SE(2)
        self.u_app = self.lag_beta * self.u_app + (1 - self.lag_beta) * u
        u = self.u_app
        self.pose = se2.to_xytheta(X_now @ se2.exp_(u * self.dt))
        self.xi_prev = u.copy(); self.t += 1

        # world updates
        self._discover()
        if self.t % self.REPLAN_EVERY == 0:
            self.path = self._plan()

        # metrics + reward
        p = self.pose[:2]
        clr_dyn = min([np.hypot(o[0][0] - p[0], o[0][1] - p[1]) - o[2] - ROBOT_R
                       for o in self._dyn_true()])
        _, wd = self.sc.nearest_wall(p)
        clr = min(clr_dyn, wd - ROBOT_R)
        self.log['u'].append(u.copy()); self.log['clr'].append(clr)
        self.log['interv'].append(interv); self.log['xy'].append(p.copy())

        accel = np.linalg.norm((u - (self.log['u'][-2] if len(self.log['u']) > 1 else u)) / self.dt)
        prog = float((np.array(GOAL) - p) @ (se2.from_xytheta(*self.pose)[:2, :2] @ u[:2])) \
            / (np.linalg.norm(np.array(GOAL) - p) + 1e-6)
        reward = 1.0 * prog - 0.15 * accel - 0.30 * interv
        collided = clr < 0.0
        reached = np.linalg.norm(np.array(GOAL) - p) < 0.30
        if collided: reward -= 20.0
        if reached: reward += 20.0
        done = collided or reached or self.t >= self.max_steps
        return self._obs(), reward, done, dict(clr=clr, collided=collided, reached=reached,
                                               min_h=res.min_h, jerk=accel)

    def _cbf_refilter(self, u, obstacles):
        """Project (vx,vy) onto the safe set (single-step CBF), keep omega."""
        import scipy.sparse as sp, osqp
        p = self.pose[:2]; th = self.pose[2]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        rows, l = [], []
        for o in obstacles:
            d = p - np.array([o['x'], o['y']])
            r_eff = o['radius'] + o.get('margin', self.cfg.cbf_safe_margin)
            h = float(d @ d - r_eff ** 2)
            a = 2.0 * d @ R                                  # coeff on (vx,vy)
            rows.append(a); l.append(2.0 * d @ np.array([o.get('vx', 0), o.get('vy', 0)])
                                     - self.cfg.cbf_alpha * h)
        P = sp.diags([2., 2.]).tocsc(); q = -2.0 * u[:2]
        A = sp.csc_matrix(np.array(rows)) if rows else sp.csc_matrix((0, 2))
        pr = osqp.OSQP()
        pr.setup(P=P, q=q, A=A, l=np.array(l), u=np.full(len(l), np.inf),
                 verbose=False, polish=False)
        r = pr.solve()
        vxy = r.x if r.x is not None and r.info.status_val in (1, 2) else u[:2]
        vxy = np.clip(vxy, self.cfg.u_min[:2], self.cfg.u_max[:2])
        return np.array([vxy[0], vxy[1], u[2]])

    # ------------------------------------------------------------------
    def metrics(self):
        u = np.array(self.log['u']); xy = np.array(self.log['xy'])
        if len(u) < 3:
            return {}
        accel = np.diff(u, axis=0) / self.dt
        dxy = np.diff(xy, axis=0)
        keep = np.linalg.norm(dxy, axis=1) > 1e-4          # ignore near-stationary steps
        head = np.arctan2(dxy[keep, 1], dxy[keep, 0])       # heading of the xy PATH
        dh = np.abs((np.diff(head) + np.pi) % (2 * np.pi) - np.pi)
        dist = np.sum(np.linalg.norm(dxy, axis=1)) + 1e-6
        return dict(jerk_p95=float(np.percentile(np.linalg.norm(accel, axis=1), 95)),
                    deg_per_m=float(np.degrees(np.sum(dh)) / dist),
                    min_clr=float(np.min(self.log['clr'])), path_m=float(dist))


if __name__ == '__main__':
    import statistics as st
    rows = []
    for e in range(8):
        env = RealAvoidEnv(seed=e)
        done = False
        while not done:
            _, _, done, info = env.step(None)                # baseline: pure GMPC
        m = env.metrics(); m.update(reached=info['reached'], collided=info['collided'])
        rows.append(m)
        print(f"ep{e}: reached={info['reached']} coll={info['collided']} "
              f"deg/m={m.get('deg_per_m',0):.1f} jerk={m.get('jerk_p95',0):.2f} "
              f"min_clr={m.get('min_clr',0):+.2f} path={m.get('path_m',0):.1f}")
    ok = [r for r in rows if 'deg_per_m' in r]
    print("\n=== BASELINE (真 GMPC + 真場景, 8 ep) ===")
    print(f"success {sum(r['reached'] for r in rows)}/8  collision {sum(r['collided'] for r in rows)}/8")
    print(f"heading  {st.mean(r['deg_per_m'] for r in ok):.1f} deg/m   <- 對照 gz")
    print(f"jerk p95 {st.mean(r['jerk_p95'] for r in ok):.2f}")
