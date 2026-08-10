"""GMPC+CBF+shield against nav2 MPPI and RPP, 100 routes each.

The comparison the report has been missing since the hardware analysis voided
the old one. All three now run the same motion box, so a difference is
attributable to the controller rather than to how much authority each was given.

Arrival rate is reported next to every safety number on purpose: a controller
that stops short has few contacts because it covered little ground, and the
earlier baselines did exactly that -- MPPI's unfinished runs averaged a third of
the path length of its finished ones.
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
import analyze as A

ARMS = [('gmpc100', 'GMPC + CBF + shield'),
        ('mppi100', 'nav2 MPPI'),
        ('rpp100', 'nav2 RPP  ($v_y=0$)')]
sc = A.load_static_clearance()
GEOM = A.__dict__.get('DYN_GEOM', {})
R = A.ROBOT_RADIUS_M


def rows(path):
    if not os.path.exists(path):
        return {}
    return {int(x['run'].split('seed')[-1]): x
            for x in csv.DictReader(open(path)) if x.get('min_clearance_m')}


def split(archive):
    """static/dynamic minimum clearance per trial, from gz truth."""
    out = {}
    for bag in sorted(glob.glob(f'{archive}/*seed*')):
        if not os.path.exists(f'{bag}/metadata.yaml'):
            continue
        try:
            seed = int(bag.split('seed')[-1])
        except ValueError:
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


def ub(n):
    """one-sided 95% upper bound on a rate given zero events in n trials"""
    return 1 - 0.05 ** (1.0 / n)


print("# 三方對照：統一硬體上限，100 條路線\n")
print("所有方法使用相同的運動上限（0.2775 m/s / 軸、6.25 m/s²、1.1327 rad/s）與"
      "相同路線。RPP 的 $v_y$ 維持 0：純追蹤法不產生側移指令，屬演算法特性。\n")
print("| 方法 | n | 到達 | 碰撞 | 碰撞率 | 靜態 | 動態 | 間距中位 | 最差 | "
      "到達時間 | 路徑 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
store = {}
for key, label in ARMS:
    r = rows(f'evaluation/results/{key}/batch.csv')
    store[key] = r
    if not r:
        print(f"| {label} | - | | (未跑) | | | | | | | |")
        continue
    sp = split(f'evaluation/bags/archive_{key}')
    clr = [float(v['min_clearance_m']) for v in r.values()]
    n = len(r)
    neg = sum(1 for c in clr if c < 0)
    arr = sum(1 for v in r.values() if v['success'] == 'True')
    ns = sum(1 for a, b in sp.values() if min(a, b) < 0 and a < b)
    nd = sum(1 for a, b in sp.values() if min(a, b) < 0 and b <= a)
    t = [float(v['arrival_time_s']) for v in r.values()
         if v['success'] == 'True' and v['arrival_time_s']]
    pl = [float(v['path_length_m']) for v in r.values()]
    rate = f"{neg/n:.1%}" if neg else f"0 (上界 {ub(n):.1%})"
    print(f"| {label} | {n} | {arr}/{n} | **{neg}** | {rate} | {ns} | {nd} | "
          f"{st.median(clr):+.3f} | {min(clr):+.3f} | "
          f"{(st.median(t) if t else float('nan')):.0f} s | {st.median(pl):.1f} m |")

# do the baselines look safe only because they stopped short?
print("\n## 未到達的趟數在哪裡停下\n")
for key, label in ARMS:
    r = store.get(key) or {}
    fail = [v for v in r.values() if v['success'] != 'True']
    ok = [v for v in r.values() if v['success'] == 'True']
    if not r:
        continue
    if not fail:
        print(f"- **{label}**: 全部到達")
        continue
    pf = st.median([float(v['path_length_m']) for v in fail])
    po = st.median([float(v['path_length_m']) for v in ok]) if ok else float('nan')
    print(f"- **{label}**: 未到達 {len(fail)}/{len(r)}，"
          f"其路徑中位 {pf:.1f} m vs 到達趟的 {po:.1f} m "
          f"（{pf/max(po,1e-6):.0%}）")

# paired, on the routes every arm completed
common = sorted(set.intersection(*[set(store[k]) for k, _ in ARMS if store.get(k)])) \
    if all(store.get(k) for k, _ in ARMS) else []
if common:
    print(f"\n## 配對（三組都跑完的 {len(common)} 條）\n")
    print("| 方法 | 碰撞 | 到達 | 間距中位 |")
    print("|---|---|---|---|")
    for key, label in ARMS:
        r = store[key]
        c = [float(r[k]['min_clearance_m']) for k in common]
        print(f"| {label} | {sum(1 for x in c if x < 0)} | "
              f"{sum(1 for k in common if r[k]['success']=='True')}/{len(common)} | "
              f"{st.median(c):+.3f} |")

print("""
**讀法**：碰撞數必須和到達率一起看。基線若有大量未到達，其低碰撞有一部分來自
沒有走完全程；「未到達的趟數在哪裡停下」那節就是為了讓這件事無法被忽略。
零事件的 95% 上界約為 3/N，100 趟為 3%。
""")
