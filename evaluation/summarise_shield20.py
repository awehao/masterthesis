"""Shield on the full scenario, paired against poseF on the same routes.

poseF is the control: same route file, same POSE_SOURCE=odom, same soft slack,
same min_track_speed. The only difference is the shield, so anything here is
attributable to it.

Contacts are split static/dynamic from ground truth, because the two have
different causes and the shield addresses only one of them directly -- a fall in
the total that came entirely from static grazes would say something quite
different from one that removed dynamic contacts.
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


def split_contacts(archive, limit=None):
    """Per trial: (static min, dynamic min) from gz ground truth."""
    out = {}
    for bag in sorted(glob.glob(f'{archive}/gmpc_cbf__scan_seed*')):
        seed = bag.split('seed')[-1]
        if limit and int(seed) > limit:
            continue
        if not os.path.exists(f'{bag}/metadata.yaml'):
            continue
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


def csv_rows(path, limit=None):
    if not os.path.exists(path):
        return {}
    r = {}
    for x in csv.DictReader(open(path)):
        if not x.get('min_clearance_m'):
            continue
        seed = x['run'].split('seed')[-1]
        if limit and int(seed) > limit:
            continue
        r[seed] = x
    return r


N = 20
ctl_csv = csv_rows('evaluation/results/poseF/batch.csv', N)
shd_csv = csv_rows('evaluation/results/shield20/batch.csv', N)
ctl = split_contacts('evaluation/bags/archive_poseF', N)
shd = split_contacts('evaluation/bags/archive_shield20', N)

print("# Raw-scan shield on the full scenario\n")
print(f"{N} routes. Control is poseF: same routes, same pose source, same soft "
      "slack, same min_track_speed. The shield is the only difference.\n")
print("| arm | n | arrived | contacts | static | dynamic | median clr | worst | "
      "arrival | path |")
print("|---|---|---|---|---|---|---|---|---|---|")
for label, rows, spl in (('control (no shield)', ctl_csv, ctl),
                         ('**shield**', shd_csv, shd)):
    if not rows:
        print(f"| {label} | - | (not run) | | | | | | | |")
        continue
    clr = [float(v['min_clearance_m']) for v in rows.values()]
    arr = sum(1 for v in rows.values() if v['success'] == 'True')
    ns = sum(1 for k, (a, b) in spl.items() if min(a, b) < 0 and a < b)
    nd = sum(1 for k, (a, b) in spl.items() if min(a, b) < 0 and b <= a)
    t = [float(v['arrival_time_s']) for v in rows.values()
         if v['success'] == 'True' and v['arrival_time_s']]
    pl = [float(v['path_length_m']) for v in rows.values()]
    print(f"| {label} | {len(rows)} | {arr}/{len(rows)} | "
          f"**{sum(1 for c in clr if c < 0)}** | {ns} | {nd} | "
          f"{st.median(clr):+.3f} | {min(clr):+.3f} | "
          f"{(st.median(t) if t else float('nan')):.0f} s | "
          f"{st.median(pl):.1f} m |")

common = sorted(set(ctl_csv) & set(shd_csv), key=int)
if common:
    a = {k: float(ctl_csv[k]['min_clearance_m']) for k in common}
    b = {k: float(shd_csv[k]['min_clearance_m']) for k in common}
    only_ctl = [k for k in common if a[k] < 0 <= b[k]]
    only_shd = [k for k in common if b[k] < 0 <= a[k]]
    print(f"\n## Paired on the same {len(common)} routes\n")
    print(f"- contacts: control {sum(1 for k in common if a[k]<0)}, "
          f"shield {sum(1 for k in common if b[k]<0)}")
    print(f"- fixed by the shield: {len(only_ctl)}   "
          f"introduced by it: {len(only_shd)}")
    try:
        from scipy.stats import binomtest
        if only_ctl or only_shd:
            print(f"- McNemar p = "
                  f"{binomtest(len(only_shd), len(only_ctl)+len(only_shd)).pvalue:.4f}")
    except Exception:
        pass

rows = []
for bag in sorted(glob.glob('evaluation/bags/archive_shield20/gmpc_cbf__scan_seed*')):
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
    arr = np.array(rows)
    v = {k: arr[:, i] for i, k in enumerate(SF)}
    act = v['active'] > 0.5
    print("\n## Shield behaviour\n")
    print(f"- cycles {len(arr)}, intervened {act.sum()} ({act.mean():.1%})")
    if act.any():
        print(f"- |dv| when active: median {np.median(v['dv'][act]):.3f} m/s")
        print(f"- clearance when active: median "
              f"{np.median(v['d_min'][act]):.3f}, min {v['d_min'][act].min():.3f} m")
    print(f"- **max violation after {v['viol_after'].max():+.4f}**, "
          f"unresolved {(v['unresolved'] > 0.5).sum()} cycles, "
          f"fallback {(v['fallback'] > 0.5).sum()}")
    si = np.hypot(v['vx_in'], v['vy_in']); so = np.hypot(v['vx_out'], v['vy_out'])
    print(f"- speed cost: {np.median(si):.3f} -> {np.median(so):.3f} m/s "
          f"({1 - np.median(so)/max(np.median(si), 1e-6):.1%})")
    print(f"- stale cycles {(v['stale'] > 0.5).sum()}, "
          f"scan age p99 {np.percentile(v['scan_age'], 99):.3f} s")

print("""
**Scope**: 20 routes cannot settle a contact RATE -- the trial-to-trial spread
is 0.043 m median and 0.479 m worst on identical configurations, so only a
large effect is resolvable here. It answers whether the shield transfers off
seed27, not what the residual rate is.
""")
