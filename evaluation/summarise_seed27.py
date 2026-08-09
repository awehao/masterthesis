"""seed27 replay: did the fragmentation recovery close the perception gap?

The acceptance criteria are per-cycle, not per-run: ten contact counts cannot
resolve anything (report S6.6), but track coverage and dropout length are
sampled thousands of times inside each replay and speak directly to the
mechanism -- a 1.5 s window with nothing published while a 1.6 m box drove into
a stopped robot.

  coverage        fraction of the 2 s before closest approach with the mover
                  present in /gmpc/obstacles
  longest dropout the worst uninterrupted gap in that window
  fragments       clusters published per cycle near the robot (the symptom)
  ID switches     new track ids created, a regression guard: a wider gate that
                  fixes seed27 by merging neighbours would show up here
"""
import glob, math, os, sys, csv
import statistics as st

sys.path.insert(0, 'evaluation')
os.environ.setdefault('BIGARENA', '1')
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32MultiArray
import analyze as A

ARMS = [('T0', 'as-is'),
        ('T1', '+ predicted association, Mahalanobis gate'),
        ('T2', '+ fragment merge'),
        ('T3', '+ coast publish (1.0 s, radius +0.15 m/s)')]
MOVER = 5                       # dyn_obs_5, the 1.6 x 0.4 m box
A.load_static_clearance()       # populates DYN_GEOM
GEOM = A.__dict__.get('DYN_GEOM', {})
R = A.ROBOT_RADIUS_M


def one_run(bag):
    T = {'/odom': Odometry, '/gmpc/obstacles': Float32MultiArray,
         '/cmd_vel': Twist, f'/model/dyn_obs_{MOVER}/pose': PoseStamped}
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
    if len(ot) < 10 or not s[f'/model/dyn_obs_{MOVER}/pose']:
        return None
    oxy = np.array([[m.pose.pose.position.x, m.pose.pose.position.y]
                    for _, m in s['/odom']])
    ser = s[f'/model/dyn_obs_{MOVER}/pose']
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

    # was the mover in the published set, in the 2 s before closest approach?
    win = [(t, m) for t, m in s['/gmpc/obstacles'] if -2.0 <= t - t0 <= 0.0]
    seen, frags = [], []
    for t, m in win:
        i = max(0, min(int(np.searchsorted(ot, t) - 1), len(ot) - 1))
        mx, my = pxy[j[i]]
        d = np.array(m.data, dtype=float)
        n_near = 0
        if len(d):
            d = d.reshape(-1, 5)
            n_near = int(np.sum(np.hypot(d[:, 0] - mx, d[:, 1] - my) <= 1.2))
        seen.append(n_near > 0)
        frags.append(n_near)
    cov = float(np.mean(seen)) if seen else float('nan')
    # longest run of consecutive False, in seconds
    worst = run = 0
    tw = [t for t, _ in win]
    for i, ok in enumerate(seen):
        if ok:
            run = 0
        else:
            run += 1
            if run > 1 and i < len(tw):
                worst = max(worst, tw[i] - tw[i - run + 1])
    return dict(depth=float(clr[k]), cov=cov, dropout=float(worst),
                frag=float(np.mean([f for f in frags if f])) if any(frags) else 0.0)


print("# seed27 replay\n")
print("Route 27 only, ten replays per arm, fresh obstacle phase each time. "
      "Goal (0.79, 10.37) lies on the path of dyn_obs_5, a 1.6 x 0.4 m box; on "
      "final approach the centre distance falls inside the box's own half-"
      "length, so the visible arc spans every masked sector.\n")
print("| arm | runs | contacts | worst depth | median depth | "
      "track coverage | longest dropout | fragments/cycle |")
print("|---|---|---|---|---|---|---|---|")
for key, label in ARMS:
    root = f'evaluation/bags/archive_{key}'
    runs = [one_run(b) for b in sorted(glob.glob(f'{root}/rep*'))]
    runs = [r for r in runs if r]
    if not runs:
        print(f"| {label} | - | (not run) | | | | | |")
        continue
    d = [r['depth'] for r in runs]
    neg = sum(1 for x in d if x < 0)
    cov = [r['cov'] for r in runs if not math.isnan(r['cov'])]
    print(f"| {label} | {len(runs)} | **{neg}/{len(runs)}** | {min(d):+.3f} | "
          f"{st.median(d):+.3f} | {st.mean(cov)*100:.0f}% | "
          f"{max(r['dropout'] for r in runs):.2f} s | "
          f"{st.mean([r['frag'] for r in runs]):.1f} |")

print("\n**Acceptance**: coverage near 100% in the 2 s before impact, longest "
      "dropout under 0.2 s, and no deep penetration in ten replays. A drop in "
      "contacts without a rise in coverage would mean the arm got lucky on "
      "phase, not that the gap closed.\n")
