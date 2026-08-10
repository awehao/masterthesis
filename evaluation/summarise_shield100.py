"""Contact rate for the shield over 100 routes, with the bound that goes with it.

Zero events do not mean zero rate. The one-sided 95% upper bound for 0 in N is
1 - 0.05^(1/N), which is close to 3/N; quoting the observation without the bound
would overstate what 100 trials can support.

Contacts are split static/dynamic from ground truth. The shield acts on any
return regardless of what the tracker made of it, so both classes are in scope,
but they have different causes and a fall confined to one of them means
something different from a fall in both.
"""
import csv, glob, math, os, sys
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

sc = A.load_static_clearance()
GEOM = A.__dict__.get('DYN_GEOM', {})
R = A.ROBOT_RADIUS_M
SF = ['cycle', 'active', 'n_pts', 'd_min', 'dv', 'vx_in', 'vy_in', 'wz_in',
      'vx_out', 'vy_out', 'wz_out', 'scan_age', 'stale',
      'viol_before', 'viol_after', 'iters', 'fallback', 'unresolved']


def split(archive):
    out = {}
    for bag in sorted(glob.glob(f'{archive}/gmpc_cbf__scan_seed*')):
        if not os.path.exists(f'{bag}/metadata.yaml'):
            continue
        seed = int(bag.split('seed')[-1])
        T = {'/odom': Odometry}
        for i in range(10):
            T[f'/model/dyn_obs_{i}/pose'] = PoseStamped
        s = {k: [] for k in T}
        rd = rosbag2_py.SequentialReader()
        try:
            rd.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'),
                    rosbag2_py.ConverterOptions('', ''))
        except Exception:
            continue
        have = {t.name for t in rd.get_all_topics_and_types()}
        rd.set_filter(rosbag2_py.StorageFilter(topics=[k for k in T if k in have]))
        while rd.has_next():
            tp, buf, t = rd.read_next()
            s[tp].append((t * 1e-9, deserialize_message(buf, T[tp])))
        ot = np.array([t for t, _ in s['/odom']])
        if len(ot) < 10:
            continue
        oxy = np.array([[m.pose.pose.position.x, m.pose.pose.position.y]
                        for _, m in s['/odom']])
        cs = np.array([sc(x, y) - R for x, y in oxy])
        cd = np.full(len(ot), 9.9)
        for i in range(10):
            ser = s.get(f'/model/dyn_obs_{i}/pose') or []
            if not ser:
                continue
            ts = np.array([t for t, _ in ser])
            p = np.array([[m.pose.position.x, m.pose.position.y] for _, m in ser])
            j = np.clip(np.searchsorted(ts, ot) - 1, 0, len(ts) - 1)
            g = GEOM.get(f'dyn_obs_{i}')
            if g and g[0] == 'box':
                dx = np.maximum(np.abs(p[j, 0] - oxy[:, 0]) - g[1] / 2, 0)
                dy = np.maximum(np.abs(p[j, 1] - oxy[:, 1]) - g[2] / 2, 0)
                surf = np.hypot(dx, dy)
            else:
                surf = np.hypot(p[j, 0] - oxy[:, 0],
                                p[j, 1] - oxy[:, 1]) - (g[1] if g else 0.25)
            cd = np.minimum(cd, surf - R)
        out[seed] = (float(cs.min()), float(cd.min()))
    return out


rows = {}
f = 'evaluation/results/shield100/batch.csv'
if os.path.exists(f):
    for x in csv.DictReader(open(f)):
        if x.get('min_clearance_m'):
            rows[int(x['run'].split('seed')[-1])] = x

print("# Shield over 100 routes\n")
if not rows:
    print("(not run)")
    sys.exit(0)

clr = [float(v['min_clearance_m']) for v in rows.values()]
n = len(rows)
neg = sum(1 for c in clr if c < 0)
arr = sum(1 for v in rows.values() if v['success'] == 'True')
t = [float(v['arrival_time_s']) for v in rows.values()
     if v['success'] == 'True' and v['arrival_time_s']]
pl = [float(v['path_length_m']) for v in rows.values()]
jk = [float(v['jerk_vx']) for v in rows.values() if v.get('jerk_vx')]

print(f"| n | arrived | contacts | rate | median clr | worst | arrival | path | jerk_vx |")
print("|---|---|---|---|---|---|---|---|---|")
print(f"| {n} | {arr}/{n} | **{neg}** | {neg/n:.1%} | {st.median(clr):+.3f} | "
      f"{min(clr):+.3f} | {(st.median(t) if t else float('nan')):.0f} s | "
      f"{st.median(pl):.1f} m | {(st.median(jk) if jk else float('nan')):.3f} |")

if neg == 0:
    ub = 1 - 0.05 ** (1.0 / n)
    print(f"\n**0 contacts in {n} trials — 95% upper bound on the contact rate "
          f"is {ub:.1%}.** Zero observed is not zero rate; that bound is what "
          f"the sample supports.")
else:
    lo = max(0.0, neg/n - 1.96*math.sqrt(neg/n*(1-neg/n)/n))
    hi = min(1.0, neg/n + 1.96*math.sqrt(neg/n*(1-neg/n)/n))
    print(f"\n**{neg} contacts in {n} trials = {neg/n:.1%} "
          f"(95% CI {lo:.1%}–{hi:.1%}).**")

sp = split('evaluation/bags/archive_shield100')
if sp:
    ns = sum(1 for a, b in sp.values() if min(a, b) < 0 and a < b)
    nd = sum(1 for a, b in sp.values() if min(a, b) < 0 and b <= a)
    print(f"\n- static contacts {ns}, dynamic {nd} (of {len(sp)} bags scored)")
    bad = sorted(((k, a, b) for k, (a, b) in sp.items() if min(a, b) < 0),
                 key=lambda r: min(r[1], r[2]))
    for k, a, b in bad[:10]:
        print(f"  - seed{k}: static {a:+.3f}, dynamic {b:+.3f} "
              f"({'STATIC' if a < b else 'DYNAMIC'})")

rows2 = []
for bag in sorted(glob.glob('evaluation/bags/archive_shield100/gmpc_cbf__scan_seed*')):
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
            rows2.append(d)
if rows2:
    a = np.array(rows2)
    v = {k: a[:, i] for i, k in enumerate(SF)}
    act = v['active'] > 0.5
    print("\n## Shield behaviour\n")
    print(f"- cycles {len(a)}, intervened {act.sum()} ({act.mean():.2%})")
    if act.any():
        print(f"- |dv| when active: median {np.median(v['dv'][act]):.3f} m/s")
        print(f"- clearance when active: median {np.median(v['d_min'][act]):.3f}, "
              f"min {v['d_min'][act].min():.3f} m")
    print(f"- fallback engaged {(v['fallback'] > 0.5).sum()} "
          f"({(v['fallback'] > 0.5).mean():.2%}), "
          f"max violation after {v['viol_after'].max():+.4f}")
    si = np.hypot(v['vx_in'], v['vy_in']); so = np.hypot(v['vx_out'], v['vy_out'])
    print(f"- speed cost {np.median(si):.3f} -> {np.median(so):.3f} m/s "
          f"({1 - np.median(so)/max(np.median(si), 1e-6):.1%})")
    print(f"- stale cycles {(v['stale'] > 0.5).sum()}, "
          f"scan age p99 {np.percentile(v['scan_age'], 99):.3f} s")
    print("\n`fallback` is the full barrier being infeasible, at which point the "
          "layer solves the weaker non-approach condition instead. That "
          "degradation is by design, but the diagnostic does not yet record "
          "whether the non-approach condition itself held -- a `non_approach_ok` "
          "field is still to be added.")
