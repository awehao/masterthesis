"""CBF x shield, 2x2, on the routes all four arms completed.

The point of the design is the two differences, not the four numbers:

    B - A   what predicting obstacles buys
    C - A   what a reactive layer buys with no prediction at all
    D - B   what the shield adds where perception or the barrier misses
    D - C   what the barrier adds once the shield is already there

D - C is the one that decides whether the shield makes the CBF redundant. If C
alone matched D, the barrier would be carrying nothing the reactive layer does
not already cover; if C is safe but slow or gets stuck, the barrier is buying
early avoidance rather than safety, which is a different and more honest claim.
"""
import csv, glob, os, sys
import statistics as st

sys.path.insert(0, 'evaluation')
os.environ.setdefault('BIGARENA', '1')
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import analyze as A

ARMS = [('ablA', 'A  GMPC', 'gmpc_cbf__nocbf_seed'),
        ('ablB', 'B  + CBF', 'gmpc_cbf__scan_seed'),
        ('ablC', 'C  + shield', 'gmpc_cbf__nocbf_seed'),
        ('gmpc100', 'D  + CBF + shield', 'gmpc_cbf__scan_seed')]
sc = A.load_static_clearance()
GEOM = A.__dict__.get('DYN_GEOM', {})
R = A.ROBOT_RADIUS_M


def rows(key):
    p = f'evaluation/results/{key}/batch.csv'
    if not os.path.exists(p):
        return {}
    return {int(x['run'].split('seed')[-1]): x
            for x in csv.DictReader(open(p)) if x.get('min_clearance_m')}


def split(archive, prefix):
    """static/dynamic minimum, truncated at arrival like analyze.py."""
    out = {}
    for bag in sorted(glob.glob(f'{archive}/{prefix}*')):
        if not os.path.exists(f'{bag}/metadata.yaml'):
            continue
        try:
            seed = int(bag.split('seed')[-1])
        except ValueError:
            continue
        T = {'/odom': Odometry, '/goal_pose': PoseStamped}
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
        g = s.get('/goal_pose') or []
        if g:
            gx = g[-1][1].pose.position.x
            gy = g[-1][1].pose.position.y
            near = np.hypot(oxy[:, 0] - gx, oxy[:, 1] - gy) < 0.30
            if near.any():
                oxy = oxy[:int(np.argmax(near)) + 1]
        cs = np.array([sc(x, y) - R for x, y in oxy])
        cd = np.full(len(oxy), 9.9)
        for i in range(10):
            ser = s.get(f'/model/dyn_obs_{i}/pose') or []
            if not ser:
                continue
            ts = np.array([t for t, _ in ser])
            p = np.array([[m.pose.position.x, m.pose.position.y] for _, m in ser])
            j = np.clip(np.searchsorted(ts, ot[:len(oxy)]) - 1, 0, len(ts) - 1)
            gg = GEOM.get(f'dyn_obs_{i}')
            if gg and gg[0] == 'box':
                dx = np.maximum(np.abs(p[j, 0] - oxy[:, 0]) - gg[1] / 2, 0)
                dy = np.maximum(np.abs(p[j, 1] - oxy[:, 1]) - gg[2] / 2, 0)
                surf = np.hypot(dx, dy)
            else:
                surf = np.hypot(p[j, 0] - oxy[:, 0],
                                p[j, 1] - oxy[:, 1]) - (gg[1] if gg else 0.25)
            cd = np.minimum(cd, surf - R)
        out[seed] = (float(cs.min()), float(cd.min()))
    return out


data = {k: rows(k) for k, _, _ in ARMS}
have = [k for k, _, _ in ARMS if data[k]]
common = sorted(set.intersection(*[set(data[k]) for k in have])) if have else []

print("# CBF × Shield 消融（2×2）\n")
if not common:
    print("(尚未跑完)"); sys.exit(0)
print(f"配對 {len(common)} 條路線，其餘設定完全相同。\n")
print("| 組 | GMPC | CBF | Shield | n | 到達 | 碰撞 | 靜態 | 動態 | "
      "間距中位 | 最差 | 到達時間 | 路徑 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
flags = {'ablA': ('✓', '✗', '✗'), 'ablB': ('✓', '✓', '✗'),
         'ablC': ('✓', '✗', '✓'), 'gmpc100': ('✓', '✓', '✓')}
S = {}
for key, label, pref in ARMS:
    r = data.get(key) or {}
    if not r:
        print(f"| {label} | | | | - | | (未跑) | | | | | | |")
        continue
    sp = split(f'evaluation/bags/archive_{key}', pref)
    c = [float(r[k]['min_clearance_m']) for k in common if k in r]
    c = [v for v in c if np.isfinite(v)]
    arr = sum(1 for k in common if k in r and r[k]['success'] == 'True')
    ns = sum(1 for k, (a, b) in sp.items() if k in common and a < 0 and a <= b)
    nd = sum(1 for k, (a, b) in sp.items() if k in common and b < 0 and b < a)
    t = [float(r[k]['arrival_time_s']) for k in common
         if k in r and r[k]['success'] == 'True' and r[k]['arrival_time_s']]
    pl = [float(r[k]['path_length_m']) for k in common if k in r]
    g, cb, sh = flags[key]
    neg = sum(1 for v in c if v < 0)
    S[key] = dict(neg=neg, arr=arr, med=st.median(c), worst=min(c),
                  t=st.median(t) if t else float('nan'), n=len(c))
    print(f"| {label} | {g} | {cb} | {sh} | {len(c)} | {arr}/{len(common)} | "
          f"**{neg}** | {ns} | {nd} | {st.median(c):+.3f} | {min(c):+.3f} | "
          f"{(st.median(t) if t else float('nan')):.0f} s | {st.median(pl):.1f} m |")

print("\n## 兩兩比較\n")


def diff(a, b, name, what):
    if a not in S or b not in S:
        print(f"- **{name}** — 資料不足")
        return
    x, y = S[a], S[b]
    print(f"- **{name}**（{what}）：碰撞 {x['neg']} → {y['neg']}，"
          f"到達 {x['arr']} → {y['arr']}，"
          f"間距中位 {x['med']:+.3f} → {y['med']:+.3f}，"
          f"最差 {x['worst']:+.3f} → {y['worst']:+.3f}，"
          f"到達時間 {x['t']:.0f} → {y['t']:.0f} s")


diff('ablA', 'ablB', 'B − A', '預測式避障的貢獻')
diff('ablA', 'ablC', 'C − A', '純反應式保護的貢獻')
diff('ablB', 'gmpc100', 'D − B', '感知或 CBF 漏失時，shield 的補位')
diff('ablC', 'gmpc100', 'D − C', 'CBF 對提前繞行與效率的貢獻')

print("""
**這組要回答的**：shield 是否讓 CBF 變得多餘。若 C 單獨就與 D 相當，屏障就沒有
剩下需要辯護的功能；若 C 安全但繞行差、耗時長或卡住，則屏障買到的是**效率與
提前性**而非安全 —— 那是不同、也更誠實的宣稱。
""")
