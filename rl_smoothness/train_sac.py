"""Train the SAC residual-smoothing policy on ResidualSmoothEnv.

    u = GMPC(x) + Δu_RL   ->   CBF shield   ->   u_safe

The policy only learns the residual Δu (bounded to ±ACT_LIM), so it starts from
the validated GMPC baseline instead of learning to navigate from scratch; the
CBF shield keeps the safety guarantee regardless of what the network outputs.

Run (laptop smoke test, few steps, CPU is fine):
    rl_smoothness/.venv/bin/python rl_smoothness/train_sac.py --steps 2000 --n-envs 2

Run (desktop, real training):
    rl_smoothness/.venv/bin/python rl_smoothness/train_sac.py --steps 500000 --n-envs 8

Evaluation compares the trained policy against the zero-residual baseline on the
SAME seeds -- the headline is "less wiggle at equal success/safety".
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def make_env(seed, lag_beta, rank=0):
    def _init():
        from rl_env import ResidualSmoothEnv
        return ResidualSmoothEnv(seed=seed + rank, lag_beta=lag_beta)
    return _init


def evaluate(model, n_ep, lag_beta, seed0=10_000):
    """Run n_ep episodes with the policy and with a zero residual, same seeds."""
    from rl_env import ResidualSmoothEnv
    out = {}
    for tag in ('baseline', 'policy'):
        env = ResidualSmoothEnv(seed=seed0, lag_beta=lag_beta, randomize_seed=False,
                                train_mode=False)      # full run from START
        dpm, jerk, succ, coll, interv = [], [], 0, 0, []
        for e in range(n_ep):
            obs, _ = env.reset(seed=seed0 + e)
            term = trunc = False
            ivs = []
            while not (term or trunc):
                if tag == 'policy':
                    act, _ = model.predict(obs, deterministic=True)
                else:
                    act = np.zeros(3)
                obs, _, term, trunc, info = env.step(act)
                ivs.append(info.get('interv', 0.0))
            m = env.metrics()
            succ += int(info['reached']); coll += int(info['collided'])
            if 'deg_per_m' in m:
                dpm.append(m['deg_per_m']); jerk.append(m.get('jerk_p95', 0.0))
            interv.append(float(np.mean(ivs)) if ivs else 0.0)
        out[tag] = dict(success=succ, collided=coll, n=n_ep,
                        deg_per_m=float(np.mean(dpm)) if dpm else float('nan'),
                        jerk=float(np.mean(jerk)) if jerk else float('nan'),
                        interv=float(np.mean(interv)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200_000)
    ap.add_argument('--n-envs', type=int, default=4)
    ap.add_argument('--lag-beta', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-ep', type=int, default=5)
    ap.add_argument('--out', default='rl_smoothness/runs/sac_residual')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
    import torch

    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    VecCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = VecCls([make_env(args.seed, args.lag_beta, i) for i in range(args.n_envs)])
    venv = VecMonitor(venv)

    model = SAC('MlpPolicy', venv, verbose=1, device=args.device, seed=args.seed,
                learning_rate=3e-4, buffer_size=300_000, batch_size=256,
                tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,
                learning_starts=5_000, policy_kwargs=dict(net_arch=[256, 256]),
                tensorboard_log=os.path.join(args.out, 'tb'))

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    print(f"training done in {(time.time()-t0)/60:.1f} min")
    model.save(os.path.join(args.out, 'model'))
    venv.close()

    print("\n=== evaluation (same seeds, policy vs zero-residual baseline) ===")
    res = evaluate(model, args.eval_ep, args.lag_beta)
    b, p = res['baseline'], res['policy']
    print(f"{'':10s} {'success':>9} {'coll':>6} {'deg/m':>8} {'jerk':>7} {'interv':>8}")
    for tag, r in (('baseline', b), ('policy', p)):
        print(f"{tag:10s} {r['success']:>4}/{r['n']:<4} {r['collided']:>6} "
              f"{r['deg_per_m']:>8.1f} {r['jerk']:>7.2f} {r['interv']:>8.3f}")
    if np.isfinite(b['deg_per_m']) and b['deg_per_m'] > 0:
        print(f"\nwiggle change: {100*(p['deg_per_m']-b['deg_per_m'])/b['deg_per_m']:+.1f}%  "
              f"(negative = smoother, the goal)")


if __name__ == '__main__':
    main()
