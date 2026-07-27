"""Is the jerk/wiggle caused by the DYNAMIC obstacles, or by the static scene?

Howard's question: the benchmark's 4 dynamic obstacles ping-pong back and forth
over a short segment (1.6-3 m, period 10-34 s), so while the robot passes through
their region they sweep across it 1-2 times -> the CBF fires repeatedly. A real
pedestrian crosses ONCE: one brief correction, no repeated bursts. So is the
"jerk_vx p95 = a_max (saturated)" result an artefact of an adversarial scenario?

Configurations (same map, same path, same perception noise, same seeds):
  A  ping-pong   : the benchmark as-is
  B  frozen      : same obstacles, speed = 0 (present but not moving)
  C  removed     : obstacles teleported far away (static scene only)

Metrics: heading change (deg/m), jerk p95, and the fraction of steps where the
CBF is actually pressing (min_h below a danger threshold).

Run:  python3 rl_smoothness/exp_dynamic_attribution.py
"""
from __future__ import annotations
import sys, copy
import numpy as np

sys.path.insert(0, '/home/howardchen/masterthesis/rl_smoothness')
import scenario                                    # noqa: E402
from scenario import DYN as DYN_ORIG               # noqa: E402

N_SEEDS = 3
LAG_BETA = 0.5
DANGER_H = 0.4          # same threshold the controller uses for gain scheduling


def set_dyn(mode):
    """Rewrite the module-level DYN list that env_real reads."""
    dyn = copy.deepcopy(DYN_ORIG)
    if mode == 'frozen':
        for o in dyn:
            o['end'] = o['start']                  # zero-length sweep -> stationary
    elif mode == 'removed':
        for o in dyn:                              # park them far outside the room
            o['start'] = (o['start'][0] + 500.0, o['start'][1] + 500.0)
            o['end'] = o['start']
    scenario.DYN[:] = dyn
    import env_real
    env_real.DYN = scenario.DYN
    return dyn


def run(mode):
    set_dyn(mode)
    import importlib, env_real
    importlib.reload(env_real)                     # pick up the patched DYN
    dpm, jerk, danger_frac, reach, coll = [], [], [], 0, 0
    for s in range(N_SEEDS):
        env = env_real.RealAvoidEnv(seed=s, lag_beta=LAG_BETA, max_steps=5000)
        done, n, dang = False, 0, 0
        while not done and n < 6000:
            _, _, done, info = env.step(None)
            n += 1
            if info.get('min_h', 1e9) < DANGER_H:
                dang += 1
        m = env.metrics()
        reach += int(info['reached']); coll += int(info['collided'])
        if 'deg_per_m' in m:
            dpm.append(m['deg_per_m']); jerk.append(m.get('jerk_p95', 0.0))
        danger_frac.append(dang / max(n, 1))
    return dict(deg_per_m=float(np.mean(dpm)) if dpm else float('nan'),
                jerk=float(np.mean(jerk)) if jerk else float('nan'),
                danger=float(np.mean(danger_frac)) * 100,
                reach=reach, coll=coll)


if __name__ == '__main__':
    print(f"baseline GMPC+CBF, no RL, {N_SEEDS} seeds, lag_beta={LAG_BETA}\n")
    print(f"{'config':<22} {'deg/m':>8} {'jerk p95':>9} {'CBF壓迫%':>9} "
          f"{'到達':>6} {'碰撞':>6}")
    res = {}
    for mode, label in (('pingpong', 'A 來回掃(現況)'),
                        ('frozen',   'B 凍結不動'),
                        ('removed',  'C 完全移走')):
        r = run(mode); res[mode] = r
        print(f"{label:<20} {r['deg_per_m']:>8.1f} {r['jerk']:>9.2f} "
              f"{r['danger']:>8.1f}% {r['reach']:>4}/{N_SEEDS} {r['coll']:>4}/{N_SEEDS}")

    a, c = res['pingpong'], res['removed']
    if np.isfinite(a['deg_per_m']) and np.isfinite(c['deg_per_m']):
        share = 100 * (a['deg_per_m'] - c['deg_per_m']) / a['deg_per_m']
        print(f"\n動態障礙貢獻了 {share:.0f}% 的扭曲;"
              f"剩下 {100-share:.0f}% 來自靜態幾何 + 感知噪")
        print(f"jerk: 來回 {a['jerk']:.2f} -> 無動態 {c['jerk']:.2f} "
              f"(a_max=0.80,打滿代表避障爆發 >5% 時間)")
