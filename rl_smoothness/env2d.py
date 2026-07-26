"""Lightweight 2D dynamic-avoidance environment for the RL-smoothness study.

Purpose: a FAST (pure-numpy, no gz) sandbox to train an RL residual that smooths
the trajectory, while a CBF-QP filter guarantees safety. Trains here (cheap on
CPU), evaluate later in the full gz stack.

Pipeline per step (this is the "RL 管平滑 + CBF 保命" architecture):
    u_nom  = go_to_goal(p, goal)            # nominal controller (stand-in for GMPC)
    u_res  = u_nom + a_rl                    # RL residual (a_rl = action; 0 = baseline)
    u_safe = CBF_filter(p, u_res, obstacles) # safety projection (never removed)
    p     += u_safe * dt

Metrics: jerk (|Δu/dt|), heading-change (deg/m), min clearance, collision, success.
"""
from __future__ import annotations
import numpy as np
from cbf_filter import CBFFilter


class Avoid2DEnv:
    K_OBS = 3                    # nearest obstacles exposed to the policy

    def __init__(self, n_obs=4, dt=0.05, v_max=0.6, robot_r=0.30, obs_r=0.25,
                 world=10.0, max_steps=600, seed=0,
                 obs_noise_pos=0.0, obs_noise_vel=0.0):
        self.dt = dt; self.v_max = v_max
        self.robot_r = robot_r; self.obs_r = obs_r
        self.world = world; self.max_steps = max_steps
        self.n_obs = n_obs
        # perception noise on the obstacle ESTIMATE the CBF sees (mimics KF jitter,
        # the main wiggle source in the real gz system). True obstacles move cleanly.
        self.obs_noise_pos = float(obs_noise_pos)
        self.obs_noise_vel = float(obs_noise_vel)
        self.cbf = CBFFilter(v_max=v_max, alpha=3.0, margin=robot_r + 0.08)
        self.rng = np.random.default_rng(seed)
        self.act_dim = 2
        self.obs_dim = 2 + 2 + self.K_OBS * 4
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.p = np.array([1.0, 1.0])
        self.goal = np.array([self.world - 1.0, self.world - 1.0])
        self.prev_u = np.zeros(2)
        self.t = 0
        # obstacles: constant-velocity sweepers crossing the diagonal path
        self.obs = []
        for _ in range(self.n_obs):
            side = self.rng.uniform(2.0, self.world - 2.0)
            horiz = self.rng.random() < 0.5
            speed = self.rng.uniform(0.15, 0.30)
            if horiz:
                x, y = self.rng.uniform(0, self.world), side
                v = np.array([speed * self.rng.choice([-1, 1]), 0.0])
            else:
                x, y = side, self.rng.uniform(0, self.world)
                v = np.array([0.0, speed * self.rng.choice([-1, 1])])
            self.obs.append(dict(x=x, y=y, vx=v[0], vy=v[1], r=self.obs_r))
        self._log = dict(u=[], clr=[], interv=[])
        return self._get_obs()

    # ------------------------------------------------------------------
    def _nearest(self):
        d = [np.hypot(o['x'] - self.p[0], o['y'] - self.p[1]) for o in self.obs]
        idx = np.argsort(d)[:self.K_OBS]
        return [self.obs[i] for i in idx]

    def _get_obs(self):
        goal_vec = (self.goal - self.p) / self.world
        parts = [goal_vec, self.prev_u / self.v_max]
        for o in self._nearest():
            parts.append(np.array([o['x'] - self.p[0], o['y'] - self.p[1]]) / self.world)
            parts.append(np.array([o['vx'], o['vy']]) / self.v_max)
        while len(parts) < 2 + self.K_OBS * 2:                 # pad if < K_OBS
            parts.append(np.zeros(2))
        return np.concatenate(parts).astype(np.float32)

    def go_to_goal(self):
        d = self.goal - self.p
        v = 2.0 * d                                            # P controller
        n = np.linalg.norm(v)
        if n > self.v_max:
            v = v / n * self.v_max
        return v

    # ------------------------------------------------------------------
    def step(self, action):
        a_rl = np.clip(np.asarray(action, float), -self.v_max, self.v_max) \
            if action is not None else np.zeros(2)
        u_nom = self.go_to_goal() + a_rl
        # CBF sees a NOISY estimate of the obstacles (perception jitter) -> this is
        # what makes the constraint jitter and the trajectory wiggle.
        obs_est = self.obs
        if self.obs_noise_pos > 0 or self.obs_noise_vel > 0:
            obs_est = [dict(x=o['x'] + self.rng.normal(0, self.obs_noise_pos),
                            y=o['y'] + self.rng.normal(0, self.obs_noise_pos),
                            vx=o['vx'] + self.rng.normal(0, self.obs_noise_vel),
                            vy=o['vy'] + self.rng.normal(0, self.obs_noise_vel),
                            r=o['r']) for o in self.obs]
        u_safe, info = self.cbf.filter(self.p, u_nom, obs_est)

        # integrate robot + obstacles
        self.p = self.p + u_safe * self.dt
        for o in self.obs:
            o['x'] += o['vx'] * self.dt; o['y'] += o['vy'] * self.dt
            for k, lim in (('x', self.world), ('y', self.world)):   # bounce at walls
                if o[k] < 0 or o[k] > lim:
                    o['v' + k] *= -1; o[k] = np.clip(o[k], 0, lim)

        # metrics
        accel = (u_safe - self.prev_u) / self.dt
        jerk = float(np.linalg.norm(accel))
        clr = min(np.hypot(o['x'] - self.p[0], o['y'] - self.p[1]) - o['r'] - self.robot_r
                  for o in self.obs)
        self._log['u'].append(u_safe.copy()); self._log['clr'].append(clr)
        self._log['interv'].append(info['intervention'])

        # reward: progress - jerk - CBF intervention (- collision + success handled by done)
        prog = float((self.goal - self.p) @ u_safe) / (np.linalg.norm(self.goal - self.p) + 1e-6)
        reward = 1.0 * prog - 0.15 * jerk - 0.30 * info['intervention']

        self.prev_u = u_safe.copy(); self.t += 1
        collided = clr < 0.0
        reached = np.linalg.norm(self.goal - self.p) < 0.30
        done = collided or reached or self.t >= self.max_steps
        if collided: reward -= 20.0
        if reached:  reward += 20.0
        info.update(jerk=jerk, clr=clr, collided=collided, reached=reached)
        return self._get_obs(), reward, done, info

    # ------------------------------------------------------------------
    def episode_metrics(self):
        u = np.array(self._log['u'])
        clr = np.array(self._log['clr'])
        if len(u) < 3:
            return {}
        accel = np.diff(u, axis=0) / self.dt
        jerk_p95 = float(np.percentile(np.linalg.norm(accel, axis=1), 95))
        jerk_med = float(np.median(np.linalg.norm(accel, axis=1)))
        # heading change per metre
        head = np.arctan2(u[:, 1], u[:, 0])
        dhead = np.abs((np.diff(head) + np.pi) % (2 * np.pi) - np.pi)
        dist = np.sum(np.linalg.norm(u[1:] * self.dt, axis=1)) + 1e-6
        deg_per_m = float(np.degrees(np.sum(dhead)) / dist)
        return dict(jerk_p95=jerk_p95, jerk_med=jerk_med, deg_per_m=deg_per_m,
                    min_clr=float(np.min(clr)), interv_mean=float(np.mean(self._log['interv'])))


# --------------------------------------------------------------------------
def _run_baseline(n_ep=20, seed0=0):
    """Run the go-to-goal + CBF baseline (no RL) and report smoothness/safety."""
    import statistics as st
    rows = []
    for e in range(n_ep):
        env = Avoid2DEnv(seed=seed0 + e)
        env.reset(seed=seed0 + e)
        done = False
        while not done:
            _, _, done, info = env.step(None)          # action=None -> baseline
        m = env.episode_metrics(); m['reached'] = info['reached']; m['collided'] = info['collided']
        rows.append(m)
    succ = sum(r['reached'] for r in rows)
    coll = sum(r['collided'] for r in rows)
    agg = lambda k: st.mean(r[k] for r in rows if k in r)
    print(f"=== BASELINE (go-to-goal + CBF, no RL), {n_ep} episodes ===")
    print(f"success   : {succ}/{n_ep}")
    print(f"collision : {coll}/{n_ep}")
    print(f"jerk p95  : {agg('jerk_p95'):.3f}   (m/s^2)   <- RL 要壓低這個")
    print(f"jerk med  : {agg('jerk_med'):.3f}")
    print(f"heading   : {agg('deg_per_m'):.1f} deg/m       <- 也是平滑度指標")
    print(f"min clr   : {agg('min_clr'):+.3f} m            <- CBF 保命,應 >= 0")
    print(f"CBF interv: {agg('interv_mean'):.3f}")


if __name__ == '__main__':
    _run_baseline()
