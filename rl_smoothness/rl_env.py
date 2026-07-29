"""Gymnasium wrapper around RealAvoidEnv for the residual-RL smoothness study.

Control flow per step (the "RL 管平滑, CBF 保命" architecture):

    u_gmpc  = GMPC(x, ref, obstacles)            # the validated 98% baseline
    u_nom   = u_gmpc + Δu_RL                     # RL residual, clipped to ±ACT_LIM
    u_safe  = CBF_filter(u_nom)                  # hard safety projection (unchanged)

The policy can only nudge the baseline (small ACT_LIM), so it cannot destroy the
navigation competence it starts from; safety is never delegated to the network.

REWARD:
    + c_prog * (d_goal_prev - d_goal_now)   potential-based progress
    - w_res  * ||du_RL||^2                  keep the residual small
    - w_yaw  * |omega|                      yaw-rate penalty (wiggle)
    - w_interv * ||u_safe - u_nom||^2       CBF INTERVENTION penalty  <-- key
    - collision / + reached                 terminal signals

Progress MUST be potential-based. A first attempt rewarded the velocity
projected on the goal direction, which pays out every single step with no upper
bound: over a 400-step fragment it accumulated to about +80 while every
smoothness term stayed below 5, so the policy learned to thrash sideways as long
as it kept inching forward. Measured result: heading change went from 135 deg/m
to 932 deg/m, i.e. +589%. The telescoping form here sums to the distance
actually covered (~4 m per fragment), which cannot outrun the shaping terms.

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

    # lag_beta MUST stay 0 for smoothness work: the first-order lag was added to
    # bring the 2D wiggle number closer to gz, but measured on the fixed sandbox
    # it is a cliff -- beta 0 saturates the acceleration limit 25.3% of the time
    # (the real system does 17.8%), while beta 0.2 already drops that to 0.1%.
    # Training with any lag therefore optimises jerk in a world that has none.
    def __init__(self, seed=0, lag_beta=0.0, max_steps=5000,
                 c_prog=5.0, w_res=2.0, w_yaw=0.05, w_interv=0.30,
                 r_collision=-20.0, r_reached=20.0, randomize_seed=True,
                 train_mode=True, frag_steps=400):
        """train_mode: short fragments from a random point along the route.

        A full run is ~3000 steps and can even hit the cap, so fixed-start
        full-length episodes would give only ~170 episodes per 500k steps (far
        too few for SAC) and would show the policy the same opening every time.
        Fragments give ~10x more episodes AND cover the whole route. Evaluation
        uses train_mode=False -> full run from START, directly comparable to the
        gz benchmark.
        """
        self.core = RealAvoidEnv(max_steps=max_steps, seed=seed, lag_beta=lag_beta)
        self.c_prog, self.w_res = c_prog, w_res
        self.w_yaw, self.w_interv = w_yaw, w_interv
        self.r_collision, self.r_reached = r_collision, r_reached
        self.randomize_seed = randomize_seed
        self.train_mode, self.frag_steps = train_mode, frag_steps
        self._rng = np.random.default_rng(seed)
        self._u_prev = np.zeros(3)
        self._ep_steps = 0

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
        frac = float(self._rng.uniform(0.0, 0.9)) if self.train_mode else 0.0
        self.core.reset(seed, start_frac=frac)
        self._u_prev = np.zeros(3)
        self._ep_steps = 0
        self._d_prev = float(np.linalg.norm(np.array(GOAL) - self.core.pose[:2]))
        return self._observe(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0) * ACT_LIM
        _, _, done_core, info = self.core.step(a)

        u = self.core.xi_prev                           # command actually applied
        self._u_prev = u.copy()

        d_now = float(np.linalg.norm(np.array(GOAL) - self.core.pose[:2]))
        progress = self._d_prev - d_now                 # metres closed this step
        self._d_prev = d_now

        interv = float(info.get('interv', self.core.log['interv'][-1]
                                if self.core.log['interv'] else 0.0))

        reward = (self.c_prog * progress
                  - self.w_res * float(a @ a)           # keep the residual small
                  - self.w_yaw * abs(float(u[2]))
                  - self.w_interv * interv ** 2)       # squared: predict the shield
        if info['collided']:
            reward += self.r_collision
        if info['reached']:
            reward += self.r_reached

        self._ep_steps += 1
        terminated = bool(info['collided'] or info['reached'])
        truncated = bool((done_core and not terminated)
                         or (self.train_mode and self._ep_steps >= self.frag_steps))
        info = dict(info, interv=interv, progress=progress,
                    yaw_rate=abs(float(u[2])), residual=float(a @ a))
        return self._observe(), float(reward), terminated, truncated, info

    def metrics(self):
        return self.core.metrics()


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time
    print(f"gymnasium available: {_HAVE_GYM}")
    env = ResidualSmoothEnv(seed=0, lag_beta=0.5, train_mode=False)
    obs, _ = env.reset(seed=0)
    print(f"obs_dim={obs.size}  act_dim=3  ACT_LIM=±{ACT_LIM}")

    # 1) zero-residual, EVAL mode: must reproduce the GMPC baseline behaviour
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

    # 2) random residual, EVAL mode: should still be SAFE (CBF shield)
    env2 = ResidualSmoothEnv(seed=1, lag_beta=0.5, train_mode=False)
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

    # 3) TRAIN mode: short fragments from varied points along the route
    print(f"\ntrain_mode fragments (frag_steps={400}):")
    env3 = ResidualSmoothEnv(seed=7, lag_beta=0.5, train_mode=True, frag_steps=400)
    lens, starts = [], []
    t0 = time.time()
    for ep in range(6):
        env3.reset()
        starts.append(env3.core.pose[:2].copy())
        term = trunc = False; n = 0
        while not (term or trunc):
            _, _, term, trunc, info = env3.step(np.zeros(3))
            n += 1
        lens.append(n)
        print(f"  ep{ep}: start=({starts[-1][0]:5.1f},{starts[-1][1]:5.1f}) "
              f"steps={n:4d} coll={info['collided']} reached={info['reached']}")
    el = time.time() - t0
    print(f"  mean {np.mean(lens):.0f} steps/ep, {el/6:.1f}s/ep -> "
          f"{60*6/el:.0f} ep/min (1 proc); starts spread "
          f"{np.ptp([s[0] for s in starts]):.1f}x{np.ptp([s[1] for s in starts]):.1f} m")
