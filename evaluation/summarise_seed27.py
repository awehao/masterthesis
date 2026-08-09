"""seed27 replay: did the fragmentation recovery close the perception gap?

Judged per control cycle, not per run: ten contact counts cannot resolve
anything at this noise floor (report S6.6), while coverage and dropout are
sampled thousands of times inside one replay and speak to the mechanism -- a
1.5 s window with nothing published while a 1.6 x 0.4 m box drove into a
stopped robot.

Coverage is measured from /gmpc/tracks_debug, which carries a track id per
entry. Asking "was anything published within 1.2 m of the mover" cannot tell a
real detection from a neighbour drifting into range, and cannot see an
association change at all. With ids, a track that keeps its identity across the
encounter is distinguishable from one that is torn down and rebuilt -- the
failure being fixed.
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

ARMS = [('T0', 'as-is'),
        ('T1', '+ predicted association, Mahalanobis gate'),
        ('T2', '+ fragment merge (geometry only)'),
        ('T3', '+ coast publish, points translated with the centre')]
MOVER = 5
TRK = ['tid', 'x', 'y', 'vx', 'vy', 'age', 'misses', 'n_frag',
       'coast_s', 'innov', 'confirmed']
A.load_static_clearance()
GEOM = A.__dict__.get('DYN_GEOM', {})
R = A.ROBOT_RADIUS_M
NEAR = 1.2          # m, a track this close to the mover's true centre is it


def one_run(bag):
    T = {'/odom': Odometry, '/gmpc/tracks_debug': Float32MultiArray,
         f'/model/dyn_obs_{MOVER}/pose': PoseStamped}
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
    ser = s.get(f'/model/dyn_obs_{MOVER}/pose') or []
    if len(ot) < 10 or not ser:
        return None
    oxy = np.array([[m.pose.pose.position.x, m.pose.pose.position.y]
                    for _, m in s['/odom']])
    ts = np.array([t for t, _ in ser])
    pxy = np.array([[m.pose.position.x, m.pose.position.y] for _, m in ser])
    j = np.clip(np.searchsorted(ts, ot) - 1, 0, len(ts) - 1)
    g = GEOM.get(f'dyn_obs_{MOVER}')
    if g and g[0] == 'box':
        dx = np.maximum(np.abs(pxy[j, 0] - oxy[:, 0]) - g[1] / 2, 0)
        dy = np.maximum(np.abs(pxy[j, 1] - oxy[:, 1]) - g[2] / 2, 0)
        surf = np.hypot(dx, dy)
    else:
        surf = np.hypot(pxy[j, 0] - oxy[:, 0], pxy[j, 1] - oxy[:, 1]) - 0.25
    clr = surf - R
    k = int(np.argmin(clr))
    t0 = ot[k]

    seen, ids, frag, innov, vels, coast = [], [], [], [], [], []
    tw = []
    for t, m in s.get('/gmpc/tracks_debug', []):
        if not (-2.0 <= t - t0 <= 0.0):
            continue
        i = max(0, min(int(np.searchsorted(ot, t) - 1), len(ot) - 1))
        mx, my = pxy[j[i]]
        d = np.array(m.data, dtype=float)
        hit = None
        if len(d) >= len(TRK):
            d = d.reshape(-1, len(TRK))
            dist = np.hypot(d[:, 1] - mx, d[:, 2] - my)
            cand = np.where((dist <= NEAR) & (d[:, 10] > 0.5))[0]
            if len(cand):
                hit = int(np.argmin(np.where(dist <= NEAR, dist, 9e9)))
        tw.append(t)
        seen.append(hit is not None)
        if hit is not None:
            ids.append(int(d[hit, 0]))
            frag.append(float(d[hit, 7]))
            innov.append(float(d[hit, 9]))
            vels.append(math.hypot(d[hit, 3], d[hit, 4]))
            coast.append(float(d[hit, 8]))

    cov = float(np.mean(seen)) if seen else float('nan')
    worst = run = 0.0
    n_run = 0
    for i, ok in enumerate(seen):
        if ok:
            n_run = 0
        else:
            n_run += 1
            if n_run > 1:
                worst = max(worst, tw[i] - tw[i - n_run + 1])
    switches = sum(1 for a, b in zip(ids, ids[1:]) if a != b)
    return dict(depth=float(clr[k]), cov=cov, dropout=worst,
                switches=switches,
                frag=st.mean(frag) if frag else 0.0,
                innov_p95=float(np.percentile(innov, 95)) if innov else float('nan'),
                vel_sd=float(np.std(vels)) if len(vels) > 1 else float('nan'),
                coast_max=max(coast) if coast else 0.0)


print("# seed27 replay\n")
print("Route 27 only, ten replays per arm, fresh obstacle phase each time. The "
      "goal (0.79, 10.37) lies on the path of dyn_obs_5, a 1.6 x 0.4 m box; on "
      "final approach the centre distance falls inside the box's own 0.80 m "
      "half-length, so the visible arc spans every masked sector and the "
      "cluster breaks up.\n")
print("| arm | runs | contacts | worst | median | coverage | longest dropout | "
      "frag/cycle | ID switches | innov p95 | speed SD |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for key, label in ARMS:
    runs = [one_run(b) for b in sorted(glob.glob(f'evaluation/bags/archive_{key}/rep*'))]
    runs = [r for r in runs if r]
    if not runs:
        print(f"| {label} | - | (not run) | | | | | | | | |")
        continue
    d = [r['depth'] for r in runs]
    cov = [r['cov'] for r in runs if not math.isnan(r['cov'])]
    iv = [r['innov_p95'] for r in runs if not math.isnan(r['innov_p95'])]
    vs = [r['vel_sd'] for r in runs if not math.isnan(r['vel_sd'])]
    print(f"| {label} | {len(runs)} | **{sum(1 for x in d if x < 0)}/{len(runs)}** | "
          f"{min(d):+.3f} | {st.median(d):+.3f} | "
          f"{(st.mean(cov)*100 if cov else float('nan')):.0f}% | "
          f"{max(r['dropout'] for r in runs):.2f} s | "
          f"{st.mean([r['frag'] for r in runs]):.1f} | "
          f"{sum(r['switches'] for r in runs)} | "
          f"{(st.mean(iv) if iv else float('nan')):.3f} | "
          f"{(st.mean(vs) if vs else float('nan')):.3f} |")

print("""
**Acceptance**

- T1 must cut the dropout without raising ID switches or the innovation p95 --
  a gate that fixes seed27 by letting tracks grab neighbouring clusters would
  show up in both.
- T2 must raise fragments per cycle (more of the body covered) while leaving
  speed SD alone; fragments feed geometry only, never the motion state, so a
  rise in speed SD would mean that separation leaked.
- T3 must drive the longest dropout below what the controller can react to,
  with the coasted geometry translated to the predicted centre rather than held
  at the last observation.
- The contact must disappear without the run turning into a stall: check that
  arrival still happens.
""")
