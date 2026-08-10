"""Did correcting the shield fallback cost distance?

Paired on the same twenty routes: shield20 ran them with the old relaxation
(np.minimum, which froze the robot when returns surrounded it), shieldfix20 with
the corrected one (np.maximum). One replay of seed70 went 24.7 m -> 47.2 m after
the fix, so the question is whether that near-doubling is typical.

Path length is the headline here. Contacts are reported but at n=20 they cannot
carry a conclusion on their own (report S6.6).
"""
import csv, glob, math, os, sys
import statistics as st

sys.path.insert(0, 'evaluation')
os.environ.setdefault('BIGARENA', '1')
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from std_msgs.msg import Float32MultiArray

SF = ['cycle', 'active', 'n_pts', 'd_min', 'dv', 'vx_in', 'vy_in', 'wz_in',
      'vx_out', 'vy_out', 'wz_out', 'scan_age', 'stale',
      'viol_before', 'viol_after', 'iters', 'fallback', 'unresolved']


def rows(path):
    if not os.path.exists(path):
        return {}
    return {int(x['run'].split('seed')[-1]): x
            for x in csv.DictReader(open(path)) if x.get('min_clearance_m')}


def shield_stats(archive):
    acc = []
    for bag in sorted(glob.glob(f'{archive}/gmpc_cbf__scan_seed*')):
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
                acc.append(d)
    if not acc:
        return None
    a = np.array(acc)
    v = {k: a[:, i] for i, k in enumerate(SF)}
    zero = ((np.abs(v['vx_out']) < 1e-6) & (np.abs(v['vy_out']) < 1e-6)
            & (np.abs(v['wz_out']) < 1e-6)
            & (np.hypot(v['vx_in'], v['vy_in']) > 0.02))
    return dict(n=len(a), act=float((v['active'] > 0.5).mean()),
                fb=int((v['fallback'] > 0.5).sum()),
                fbf=float((v['fallback'] > 0.5).mean()),
                zero=int(zero.sum()),
                viol=float(v['viol_after'].max()))


old = rows('evaluation/results/shield20/batch.csv')
new = rows('evaluation/results/shieldfix20/batch.csv')
common = sorted(set(old) & set(new))

print("# Shield fallback fix: does it cost distance?\n")
if not common:
    print("(not run)"); sys.exit(0)
print(f"Paired on {len(common)} routes. Old relaxation used np.minimum, which "
      "made distant returns forbid any approach; the fix uses np.maximum so "
      "only the violated rows relax.\n")

print("| | 舊 fallback | 修正後 |")
print("|---|---|---|")
for lbl, key, fmt in (('路徑中位 [m]', 'path_length_m', '{:.1f}'),
                      ('到達時間中位 [s]', 'arrival_time_s', '{:.0f}'),
                      ('間距中位 [m]', 'min_clearance_m', '{:+.3f}')):
    a = [float(old[k][key]) for k in common if old[k].get(key)]
    b = [float(new[k][key]) for k in common if new[k].get(key)]
    print(f"| {lbl} | {fmt.format(st.median(a))} | {fmt.format(st.median(b))} |")
na = sum(1 for k in common if float(old[k]['min_clearance_m']) < 0)
nb = sum(1 for k in common if float(new[k]['min_clearance_m']) < 0)
ar_a = sum(1 for k in common if old[k]['success'] == 'True')
ar_b = sum(1 for k in common if new[k]['success'] == 'True')
print(f"| 到達 | {ar_a}/{len(common)} | {ar_b}/{len(common)} |")
print(f"| 碰撞 | {na} | {nb} |")

d = [float(new[k]['path_length_m']) - float(old[k]['path_length_m'])
     for k in common]
worse = sum(1 for x in d if x > 0.5)
print(f"\n## 路徑逐條變化\n")
print(f"- 中位變化 **{st.median(d):+.2f} m**，範圍 {min(d):+.1f} … {max(d):+.1f} m")
print(f"- 變長超過 0.5 m 的路線：**{worse}/{len(common)}**")
big = sorted(((k, float(old[k]['path_length_m']), float(new[k]['path_length_m']))
              for k in common), key=lambda r: r[1] - r[2])[:5]
for k, a, b in big:
    print(f"  - seed{k}: {a:.1f} → {b:.1f} m  ({b - a:+.1f})")
try:
    from scipy.stats import wilcoxon
    if any(abs(x) > 1e-9 for x in d):
        print(f"- Wilcoxon p = {wilcoxon(d).pvalue:.4f}")
except Exception:
    pass

print("\n## Shield 行為\n")
for lbl, arc in (('舊 fallback', 'evaluation/bags/archive_shield20'),
                 ('修正後', 'evaluation/bags/archive_shieldfix20')):
    s = shield_stats(arc)
    if not s:
        print(f"- **{lbl}**: (無 diag)")
        continue
    print(f"- **{lbl}**: 週期 {s['n']:,}，介入 {s['act']:.2%}，"
          f"fallback {s['fb']} ({s['fbf']:.2%})，"
          f"**輸出零但輸入非零 {s['zero']}**，殘差 after 最大 {s['viol']:+.4f}")

print("""
**判準**：路徑中位變化若超過 +2 m，或變長的路線超過半數，這個修正就要改法 —
安全不能用普遍繞遠買。fallback 與「輸出零」的次數應該降到接近零，那是修正
本來要達成的目標。
""")
