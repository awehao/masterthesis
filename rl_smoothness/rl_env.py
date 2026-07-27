"""Gymnasium wrapper around RealAvoidEnv for the residual-RL smoothness study.

Control flow per step (the "RL 管平滑, CBF 保命" architecture):

    u_gmpc  = GMPC(x, ref, obstacles)            # the validated 98% baseline
    u_nom   = u_gmpc + Δu_RL                     # RL residual, clipped to ±ACT_LIM
    u_safe  = CBF_filter(u_nom)                  # hard safety projection (unchanged)

The policy can only nudge the baseline (small ACT_LIM), so it cannot destroy the
navigation competence it starts from; safety is never delegated to the network.

REWARD -- what we actually want is "same success, less wiggle":
    + w_prog * progress          along the goal direction (keep it moving)
    - w_dsmooth * ||Δu||         control-increment penalty (jerk)
    - w_yaw * |ω|                yaw-rate penalty (the direct wiggle source)
    - w_interv * ||u_safe-u_nom||^2   CBF INTERVENTION penalty  <-- key term
    - collision / + reached      hard terminal signals

The intervention term is the important one: it teaches the agent to PREDICT the
CBF boundary and steer smoothly around obstacles *before* the shield has to fire.
An agent that lets the CBF do the avoiding keeps paying this penalty; one that
plans a gentle detour pays nothing. That is exactly what removes the wiggle.

OBSERVATION: goal vector, heading, last command, the 3 nearest dynamic obstacles
(relative position + velocity), plus LOOKAHEAD reference waypoints in the body
frame -- the policy needs to see the upcoming path bend to smooth it in advance.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, '/home/howardchen/masterthesis/rl_smoothness')

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAVE_GYM = True
except ImportError:                                   # allow import before SB3 install
    _HAVE_GYM = False

    class _Shim:                                      # minimal stand-in for testing
        class Env: pass
    gym = _Shim()                                     # type: ignore

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class spaces:                                     # type: ignore
        Box = _Box

from env_real import RealAvoidEnv, GOAL, ROBOT_R      # noqa: E402
from ammr_wholebody_mpc import se2                    # noqa: E402

ACT_LIM = 0.10          # residual bound [m/s, m/s, rad/s] -- "smooth nudge only"
N_LOOKAHEAD = 4         # reference waypoints fed to the policy
LOOKAHEAD_STRIDE = 5    # every 5th ref step (~0.25 s apart at dt=0.05)


class ResidualSmoothEnv(gym.Env):
    """Gymnasium env: action = residual Δu added to the GMPC command."""

    metadata = {'render_modes': []}

    def __init__(self, seed=0, lag_beta=0.5, max_steps=3000,
                 w_prog=1.0, w_dsmooth=0.15, w_yaw=0.05, w_interv=0.30,
                 r_collision=-20.0, r_reached=20.0, randomize_seed=True):
        self.core = RealAvoidEnv(max_steps=max_steps, seed=seed, lag_beta=lag_beta)
        self.w_prog, self.w_dsmooth = w_prog, w_dsmooth
        self.w_yaw, self.w_interv = w_yaw, w_interv
        self.r_collision, self.r_reached = r_collision, r_reached
        self.randomize_seed = randomize_seed
        self._rng = np.random.default_rng(seed)
        self._u_prev = np.zeros(3)

        obs_dim = self._observe().size
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (3,), np.float32)   # scaled to ACT_LIM

    # -- observation ------------------------------------------------------
    def _observe(self):
        base = self.core._obs()                        # goal, heading, u_prev, 3 dyn obs
        # upcoming reference waypoints in the BODY frame: lets the policy see the
        # bend ahead and start turning early instead of reacting late (= wiggle).
        X_ref, _ = self.core._ref_window()
        p = self.core.pose[:2]
        th = self.core.pose[2]
        Rt = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        look = []
        for i in range(N_LOOKAHEAD):
            k = min((i + 1) * LOOKAHEAD_STRIDE, len(X_ref) - 1)
            look.append(Rt @ (X_ref[k][:2, 2] - p) / 2.0)
        return np.concatenate([base, np.ravel(look)]).astype(np.float32)

    # -- gym API ----------------------------------------------------------
    def reset(self, seed=None, options=None):
        if seed is None and self.randomize_seed:
            seed = int(self._rng.integers(0, 1_000_000))
        self.core.reset(seed)
        self._u_prev = np.zeros(3)
        return self._observe(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0) * ACT_LIM
        _, _, done_core, info = self.core.step(a)

        u = self.core.xi_prev                           # command actually applied
        du = u - self._u_prev
        self._u_prev = u.copy()

        p = self.core.pose[:2]
        gvec = np.array(GOAL) - p
        gdist = np.linalg.norm(gvec) + 1e-6
        th = self.core.pose[2]
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        progress = float(gvec @ (R @ u[:2])) / gdist    # m/s toward the goal

        interv = float(info.get('interv', self.core.log['interv'][-1]
                                if self.core.log['interv'] else 0.0))

        reward = (self.w_prog * progress
                  - self.w_dsmooth * float(np.linalg.norm(du)) / self.core.dt * 0.1
                  - self.w_yaw * abs(float(u[2]))
                  - self.w_interv * interv ** 2)       # squared: predict the shield
        if info['collided']:
            reward += self.r_collision
        if info['reached']:
            reward += self.r_reached

        terminated = bool(info['collided'] or info['reached'])
        truncated = bool(done_core and not terminated)
        info = dict(info, interv=interv, progress=progress)
        return self._observe(), float(reward), terminated, truncated, info

    def metrics(self):
        return self.core.metrics()


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    print(f"gymnasium available: {_HAVE_GYM}")
    env = ResidualSmoothEnv(seed=0, lag_beta=0.5)
    obs, _ = env.reset(seed=0)
    print(f"obs_dim={obs.size}  act_dim=3  ACT_LIM=±{ACT_LIM}")

    # 1) zero-residual must reproduce the GMPC baseline behaviour
    t0 = time.time(); n = 0; term = trunc = False
    total_r = 0.0
    while not (term or trunc) and n < 6000:
        obs, r, term, trunc, info = env.step(np.zeros(3))
        total_r += r; n += 1
    el = time.time() - t0
    m = env.metrics()
    print(f"zero-residual episode: {n} steps in {el:.1f}s ({1000*el/n:.2f} ms/step)")
    print(f"  reached={info['reached']} collided={info['collided']} "
          f"deg/m={m.get('deg_per_m', 0):.1f} jerk={m.get('jerk_p95', 0):.2f} "
          f"return={total_r:.1f}")

    # 2) random residual: should still be SAFE (CBF shield) though less smooth
    env2 = ResidualSmoothEnv(seed=1, lag_beta=0.5)
    env2.reset(seed=1)
    rng = np.random.default_rng(0)
    term = trunc = False; n = 0; rr = 0.0
    while not (term or trunc) and n < 6000:
        _, r, term, trunc, info = env2.step(rng.uniform(-1, 1, 3))
        rr += r; n += 1
    m2 = env2.metrics()
    print(f"random-residual episode: reached={info['reached']} collided={info['collided']} "
          f"deg/m={m2.get('deg_per_m', 0):.1f} return={rr:.1f}")
    print("  (CBF shield should keep collided=False even with random actions)")
