"""Three-arm attribution on seed27: classification repair vs independent layer.

  U0  baseline
  U1  min_track_speed 0.10 -> 0.05        (repair inside the perception chain)
  U2  U1 plus the raw-scan safety shield  (layer that needs no classification)

Splitting it this way is the point. A single "with everything on" arm could not
say whether the improvement came from admitting the slow mover to the CBF or
from the shield catching what the CBF never heard about, and the two have very
different implications for what the thesis can claim.

Ordered by what the evidence has to show, not by what is easiest to report:

  1  the track still exists as often as before   (no regression in tracking)
  2  its publish rate rises                      (the causal claim)
  3  rejections stop coming from the speed gate  (the mechanism, by reason code)
  4  its estimated speed still matches truth     (nothing was faked to pass)
  5  the number of tracks called movers does not balloon (statics not admitted)
  6  only then: clearance, contacts, arrival

A drop in contacts without (2) and (3) would mean the ten replays got lucky on
obstacle phase, which at this noise floor they easily can.
"""
import glob, math, os, sys
import statistics as st

sys.path.insert(0, 'evaluation')
os.environ.setdefault('BIGARENA', '1')
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
import analyze as A

ARMS = [('S0', 'U0 baseline (gate 0.10, no shield)'),
        ('S1', 'U1 gate 0.05, no shield'),
        ('U2', 'U2 gate 0.05 + raw-scan shield')]
TRK = ['tid', 'x', 'y', 'vx', 'vy', 'age', 'misses', 'n_frag',
       'coast_s', 'innov', 'confirmed', 'reject']
REASON = {0: 'published', 1: 'too young', 2: 'instant speed', 3: 'net displacement'}
A.load_static_clearance()
GEOM = A.__dict__.get('DYN_GEOM', {})
R = A.ROBOT_RADIUS_M
NEAR = 1.2


def one_run(bag):
    T = {'/odom': Odometry, '/gmpc/tracks_debug': Float32MultiArray}
    for i in range(10):
        T[f'/model/dyn_obs_{i}/pose'] = PoseStamped
    s = {k: [] for k in T}
    rd = rosbag2_py.SequentialReader()
    try:
        rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    except Exception:
        return None
    have = {t.name for t in rd.get_all_topics_and_types()}
    rd.set_filter(rosbag2_py.StorageFilter(topics=[k for k in T if k in have]))
    while rd.has_next():
        tp, buf, t = rd.read_next()
        s[tp].append((t * 1e-9, deserialize_message(buf, T[tp])))

    ot = np.array([t for t, _ in s['/odom']])
    if len(ot) < 10:
        return None
    oxy = np.array([[m.pose.pose.position.x, m.pose.pose.position.y]
                    for _, m in s['/odom']])
    best = None
    for mi in range(10):
        ser = s.get(f'/model/dyn_obs_{mi}/pose') or []
        if not ser:
            continue
        ts = np.array([t for t, _ in ser])
        p = np.array([[m.pose.position.x, m.pose.position.y] for _, m in ser])
        jj = np.clip(np.searchsorted(ts, ot) - 1, 0, len(ts) - 1)
        g = GEOM.get(f'dyn_obs_{mi}')
        if g and g[0] == 'box':
            dx = np.maximum(np.abs(p[jj, 0] - oxy[:, 0]) - g[1] / 2, 0)
            dy = np.maximum(np.abs(p[jj, 1] - oxy[:, 1]) - g[2] / 2, 0)
            surf = np.hypot(dx, dy)
        else:
            surf = np.hypot(p[jj, 0] - oxy[:, 0],
                            p[jj, 1] - oxy[:, 1]) - (g[1] if g else 0.25)
        c = surf - R
        if best is None or c.min() < best[0].min():
            best = (c, p, jj, mi, ts)
    if best is None:
        return None
    clr, pxy, j, mover, mts = best
    k = int(np.argmin(clr))
    t0 = ot[k]
    tv = np.hypot(np.gradient(pxy[:, 0], mts), np.gradient(pxy[:, 1], mts))

    exist = pub = 0
    n_cyc = 0
    reasons = {0: 0, 1: 0, 2: 0, 3: 0}
    est, n_mover = [], []
    for t, m in s.get('/gmpc/tracks_debug', []):
        if not (-2.0 <= t - t0 <= 0.0):
            continue
        i = max(0, min(int(np.searchsorted(ot, t) - 1), len(ot) - 1))
        mx, my = pxy[j[i]]
        d = np.array(m.data, dtype=float)
        n_cyc += 1
        if len(d) < len(TRK):
            continue
        d = d.reshape(-1, len(TRK))
        n_mover.append(int((d[:, 10] > 0.5).sum()))
        dist = np.hypot(d[:, 1] - mx, d[:, 2] - my)
        near = np.where(dist <= NEAR)[0]
        if not len(near):
            continue
        exist += 1
        b = near[int(np.argmin(dist[near]))]
        est.append(math.hypot(d[b, 3], d[b, 4]))
        if d[b, 10] > 0.5:
            pub += 1
        reasons[int(d[b, 11])] = reasons.get(int(d[b, 11]), 0) + 1
    if not n_cyc:
        return None
    return dict(depth=float(clr[k]), mover=mover,
                true_v=float(np.median(tv[j[max(0, k - 40):k]])),
                est_v=float(np.median(est)) if est else float('nan'),
                exist=exist / n_cyc, pub=pub / n_cyc, reasons=reasons,
                n_mover=st.mean(n_mover) if n_mover else 0.0,
                goal_d=float(np.hypot(oxy[-1, 0] - 0.79, oxy[-1, 1] - 10.37)))


print("# Instantaneous-speed gate ablation (seed27, ten replays per arm)\n")
print("`dyn_obs_5` is configured at 0.10 m/s and `min_track_speed` was 0.10, so "
      "noise decided each cycle whether it was a mover. Everything else is "
      "stock: no Mahalanobis gate, no fragment merge, no coasting.\n")
print("| arm | runs | reached | track exists | **published** | est. speed | "
      "true speed | movers/cycle | contacts | worst |")
print("|---|---|---|---|---|---|---|---|---|---|")
store = {}
for key, label in ARMS:
    runs = [one_run(b) for b in sorted(glob.glob(f'evaluation/bags/archive_{key}/rep*'))]
    runs = [r for r in runs if r]
    store[key] = runs
    if not runs:
        print(f"| {label} | - | | (not run) | | | | | | |")
        continue
    d = [r['depth'] for r in runs]
    print(f"| {label} | {len(runs)} | "
          f"{sum(1 for r in runs if r['goal_d'] < 0.35)}/{len(runs)} | "
          f"{st.mean([r['exist'] for r in runs])*100:.0f}% | "
          f"**{st.mean([r['pub'] for r in runs])*100:.0f}%** | "
          f"{st.median([r['est_v'] for r in runs if not math.isnan(r['est_v'])]):.3f} | "
          f"{st.median([r['true_v'] for r in runs]):.3f} | "
          f"{st.mean([r['n_mover'] for r in runs]):.1f} | "
          f"**{sum(1 for x in d if x < 0)}/{len(runs)}** | {min(d):+.3f} |")

print("\n## Why the mover was withheld, by reason code\n")
for key, label in ARMS:
    runs = store.get(key) or []
    if not runs:
        continue
    tot = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in runs:
        for k2, v in r['reasons'].items():
            tot[k2] = tot.get(k2, 0) + v
    n = sum(tot.values()) or 1
    parts = ", ".join(f"{REASON[k2]} {v/n:.0%}" for k2, v in sorted(tot.items()) if v)
    print(f"- **{label}**: {parts}")

print("""
**Reading it**

- `published` is the causal metric. If it does not rise, nothing else matters.
- `instant speed` should vanish from the reason breakdown; if the rejections
  simply move to `net displacement`, the gate was not the binding constraint
  and this arm proves nothing.
- `movers/cycle` guards the other side: admitting statics would show up as more
  tracks called movers, and would cost corridor width rather than buy safety.
- Contacts come last, and at ten runs they cannot carry the claim on their own.
""")


# ---------------------------------------------------------------------------
# Shield behaviour, where it ran. A last-resort layer has to be silent on
# ordinary driving; if it trims most cycles it is a speed limit, not a shield.
SF = ['cycle', 'active', 'n_pts', 'd_min', 'dv', 'vx_in', 'vy_in', 'wz_in',
      'vx_out', 'vy_out', 'wz_out', 'scan_age', 'stale',
      'viol_before', 'viol_after', 'iters', 'fallback', 'unresolved']
rows = []
for bag in sorted(glob.glob('evaluation/bags/archive_U2/rep*')):
    rd = rosbag2_py.SequentialReader()
    try:
        rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
                rosbag2_py.ConverterOptions('', ''))
    except Exception:
        continue
    if '/shield/diag' not in {t.name for t in rd.get_all_topics_and_types()}:
        continue
    rd.set_filter(rosbag2_py.StorageFilter(topics=['/shield/diag']))
    while rd.has_next():
        _, buf, _ = rd.read_next()
        d = list(deserialize_message(buf, Float32MultiArray).data)
        if len(d) >= len(SF):
            rows.append(d)
if rows:
    a = np.array(rows); v = {k: a[:, i] for i, k in enumerate(SF)}
    act = v['active'] > 0.5
    print("\n## Shield behaviour (U2)\n")
    print(f"- cycles {len(a)}, intervened {act.sum()} ({act.mean():.1%})")
    if act.any():
        print(f"- |dv| when active: median {np.median(v['dv'][act]):.3f}, "
              f"p95 {np.percentile(v['dv'][act], 95):.3f} m/s")
        print(f"- clearance when active: median {np.median(v['d_min'][act]):.3f}, "
              f"min {v['d_min'][act].min():.3f} m")
    print(f"- **max violation after: {v['viol_after'].max():+.4f}** "
          f"(> 1e-3 in {(v['viol_after'] > 1e-3).sum()} cycles)")
    print(f"- fallback engaged {(v['fallback'] > 0.5).sum()}, "
          f"unresolved {(v['unresolved'] > 0.5).sum()}")
    print(f"- iterations median {np.median(v['iters']):.0f}, max {v['iters'].max():.0f}")
    print(f"- scan age p99 {np.percentile(v['scan_age'], 99):.3f} s, "
          f"stale cycles {(v['stale'] > 0.5).sum()}")
    si = np.hypot(v['vx_in'], v['vy_in']); so = np.hypot(v['vx_out'], v['vy_out'])
    print(f"- speed cost: median {np.median(si):.3f} -> {np.median(so):.3f} m/s "
          f"({1 - np.median(so)/max(np.median(si), 1e-6):.1%})")
