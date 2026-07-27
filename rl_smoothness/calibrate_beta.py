"""Calibrate the first-order low-pass beta so the 2D env_real wiggle (deg/m)
matches the gz benchmark (~85 deg/m). The pure-kinematic 2D sim lacks gz's
velocity_smoother + physical inertia, so it reads too wiggly; a low-pass
u_app = beta*u_app + (1-beta)*u emulates that. Sweep beta, report mean deg/m.

Run:  python3 rl_smoothness/calibrate_beta.py   (writes result to console)
"""
import sys, statistics as st
sys.path.insert(0, 'rl_smoothness')
from env_real import RealAvoidEnv

GZ_TARGET = 85.0
BETAS = [0.0, 0.4, 0.6, 0.75, 0.85]
N_EP = 3
STEP_CAP = 6000

print(f"{'beta':>5} {'reach':>6} {'coll':>5} {'deg/m':>7} {'jerk':>6}  ({N_EP} ep/point)")
res = {}
for beta in BETAS:
    dpm, jk, rc, cc = [], [], 0, 0
    for e in range(N_EP):
        env = RealAvoidEnv(seed=e, lag_beta=beta)
        done, n = False, 0
        while not done and n < STEP_CAP:
            _, _, done, info = env.step(None)
            n += 1
        m = env.metrics()
        if 'deg_per_m' in m:
            dpm.append(m['deg_per_m']); jk.append(m.get('jerk_p95', 0))
        rc += int(info['reached']); cc += int(info['collided'])
    md = st.mean(dpm) if dpm else float('nan')
    res[beta] = md
    print(f"{beta:>5.2f} {rc:>4}/{N_EP} {cc:>4}/{N_EP} {md:>7.1f} "
          f"{st.mean(jk) if jk else 0:>6.2f}")

print(f"\n=== beta vs deg/m  (target gz = {GZ_TARGET}) ===")
best = min(res, key=lambda b: abs(res[b] - GZ_TARGET))
for b, d in res.items():
    print(f"  beta={b:.2f} -> {d:6.1f} deg/m" + ("   <== closest to gz" if b == best else ""))
print(f"\nPICK beta ~ {best:.2f}  (2D wiggle {res[best]:.1f} deg/m vs gz {GZ_TARGET})")
