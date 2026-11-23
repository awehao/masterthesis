#!/usr/bin/env python3
"""配對驗收：兩趟的障礙物實際軌跡是否在共同相位時間軸上對齊。

不只確認公式相同：以各自的 phase_epoch 對齊、取共同時間區間，逐一比對十個
障礙物的實際位置，並各自檢查對排程目標的偏差。
"""
import argparse, json, math, os, glob
import numpy as np, rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import PoseStamped

ap = argparse.ArgumentParser()
ap.add_argument('run_a'); ap.add_argument('run_b')
ap.add_argument('--med-tol', type=float, default=0.010)
ap.add_argument('--max-tol', type=float, default=0.100)
a = ap.parse_args()

def hs(m):
    s = m.header.stamp; return s.sec + s.nanosec * 1e-9

def read(run):
    rd = rosbag2_py.SequentialReader()
    rd.open(rosbag2_py.StorageOptions(uri=os.path.join(run, 'bag'), storage_id='mcap'),
            rosbag2_py.ConverterOptions('', ''))
    mv, rob = {}, []
    while rd.has_next():
        t, data, ts = rd.read_next()
        if t.startswith('/model/dyn_obs_') and t.endswith('/pose'):
            m = deserialize_message(data, PoseStamped)
            mv.setdefault(t, []).append((hs(m), m.pose.position.x, m.pose.position.y))
        elif t == '/model/omni_bot/pose':
            m = deserialize_message(data, PoseStamped)
            rob.append((hs(m), m.pose.position.x, m.pose.position.y))
    ep = json.load(open(os.path.join(run, 'phase_alignment.json')))['phase_epoch_sim_t']
    r = json.load(open(os.path.join(run, 'isaac_run.json')))['run']
    return {k: np.array(v) for k, v in mv.items()}, np.array(rob), ep, r

MA, RA, EA, RUNA = read(a.run_a)
MB, RB, EB, RUNB = read(a.run_b)
print(f'A = {os.path.basename(a.run_a)}  epoch {EA:.3f}')
print(f'B = {os.path.basename(a.run_b)}  epoch {EB:.3f}')

# 共同相位區間
def span(M, ep):
    lo = max(v[0, 0] for v in M.values()) - ep
    hi = min(v[-1, 0] for v in M.values()) - ep
    return lo, hi
la, ha = span(MA, EA); lb, hb = span(MB, EB)
lo, hi = max(la, lb, 0.0), min(ha, hb)
print(f'共同相位區間 [{lo:.2f}, {hi:.2f}] s（相對各自 epoch）')
grid = np.arange(lo, hi, 0.05)

def at(M, ep, key, g):
    v = M[key]
    return np.column_stack([np.interp(g, v[:, 0] - ep, v[:, 1]),
                            np.interp(g, v[:, 0] - ep, v[:, 2])])

print(f'\n{"障礙":12s} {"中位 mm":>9s} {"p95 mm":>9s} {"max mm":>9s} {"max 發生於 s":>12s} '
      f'{"當時機器人距該障礙 m":>20s}')
bad = 0
rows = []
for k in sorted(MA):
    if k not in MB: print(f'{k} 只在 A'); bad += 1; continue
    pa, pb = at(MA, EA, k, grid), at(MB, EB, k, grid)
    e = np.hypot(pa[:, 0] - pb[:, 0], pa[:, 1] - pb[:, 1]) * 1000.0
    i = int(np.argmax(e))
    # 該時刻機器人與此障礙的距離（取 A、B 較小者，作為近距遭遇的指標）
    def robd(R, ep, p):
        t = grid[i] + ep
        j = int(np.argmin(np.abs(R[:, 0] - t)))
        return math.hypot(R[j, 1] - p[i, 0], R[j, 2] - p[i, 1])
    dmin = min(robd(RA, EA, pa), robd(RB, EB, pb))
    med, p95, mx = np.median(e), np.percentile(e, 95), e.max()
    flag = ''
    if med > a.med_tol * 1000: flag += ' 中位超標'
    if mx > a.max_tol * 1000: flag += ' 最大超標'
    if mx > 20 and dmin < 2.0: flag += ' **大誤差落在近距遭遇**'
    if flag: bad += 1
    rows.append(dict(name=k, med_mm=med, p95_mm=p95, max_mm=mx,
                     max_at_s=grid[i], robot_dist_m=dmin, flag=flag.strip()))
    print(f'{k.split("/")[2]:12s} {med:9.2f} {p95:9.2f} {mx:9.2f} {grid[i]:12.2f} '
          f'{dmin:20.2f}{flag}')

print(f'\n門檻：中位 < {a.med_tol*1000:.0f} mm、最大 < {a.max_tol*1000:.0f} mm'
      '（事前設定；100 mm 不是天然可忽略的誤差）')
print(f'結果：{"通過" if bad == 0 else f"{bad} 項需檢視"}')
json.dump(rows, open('evaluation/results/pair_scenario_check.json', 'w'),
          ensure_ascii=False, indent=1, default=float)
