#!/usr/bin/env python3
"""離線重建朝向參考的前視點，回答：末段的前視點是否已落在路徑末端、弦長多少。

方法上先自我驗證：用重建值算出的弦角必須能重現 bag 裡記錄的 `raw`
（/gmpc/heading.x）。重現不準就不採信由它導出的前視點與弦長。

控制器用的是 TF map->base_footprint 的位姿，這裡用 /odometry/filtered
（EKF 輸出，同一條鏈）作為位置來源，yaw 直接取 /gmpc/heading.z（控制器當下
看到的值），所以只有位置是重建的。
"""
import argparse, json, math, os
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry, Path

ap = argparse.ArgumentParser()
ap.add_argument('run_dir')
ap.add_argument('--lookahead', type=float, default=1.2)
a = ap.parse_args()

def hs(m):
    s = m.header.stamp; return s.sec + s.nanosec * 1e-9
def wrap(x): return (x + math.pi) % (2 * math.pi) - math.pi

rd = rosbag2_py.SequentialReader()
rd.open(rosbag2_py.StorageOptions(uri=os.path.join(a.run_dir, 'bag'),
                                  storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''))
head, plans, ekf = [], [], []
while rd.has_next():
    t, data, ts = rd.read_next()
    if t == '/gmpc/heading':
        m = deserialize_message(data, Vector3Stamped)
        head.append((hs(m), m.vector.x, m.vector.y, m.vector.z))
    elif t == '/plan':
        m = deserialize_message(data, Path)
        if len(m.poses):
            plans.append((hs(m), np.array([[p.pose.position.x, p.pose.position.y]
                                           for p in m.poses])))
    elif t == '/odometry/filtered':
        m = deserialize_message(data, Odometry)
        ekf.append((hs(m), m.pose.pose.position.x, m.pose.pose.position.y))
head = np.array(head); ekf = np.array(ekf)
run = json.load(open(os.path.join(a.run_dir, 'isaac_run.json')))['run']
t0 = run['motion_start_sim_t']; te = run['at_trigger']['sim_t']
L = run['log']; lt = np.array([x['t'] for x in L])

print(f'== {os.path.basename(a.run_dir)} 前視點重建（lookahead {a.lookahead} m）==')
print(f'  /plan {len(plans)} 則，/odometry/filtered {len(ekf)} 則，'
      f'/gmpc/heading {len(head)} 筆')
if not plans or len(ekf) < 5:
    print('  !! 資料不足'); raise SystemExit(1)

pt = np.array([p[0] for p in plans])
et = ekf[:, 0]
rows = []
for th, raw, ref, yaw in head:
    if th < t0 or th > te: continue
    ip = int(np.searchsorted(pt, th)) - 1          # 當下生效的計畫
    if ip < 0: continue
    path = plans[ip][1]
    ie = int(np.argmin(np.abs(et - th)))
    if abs(et[ie] - th) > 0.2: continue
    rp = ekf[ie, 1:3]
    d = np.linalg.norm(path - rp, axis=1)
    i0 = int(np.argmin(d))
    seg = np.linalg.norm(np.diff(path[i0:], axis=0), axis=1)
    if seg.size == 0: continue
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    j = min(int(np.searchsorted(cum, a.lookahead)), len(cum) - 1)
    tgt = path[i0 + j]
    chord = tgt - rp
    ang = math.atan2(chord[1], chord[0])
    rows.append((th, math.degrees(wrap(ang - raw)), float(np.linalg.norm(chord)),
                 float(cum[-1]), 1 if (i0 + j) == len(path) - 1 else 0,
                 float(cum[j])))
rows = np.array(rows)
err = np.abs(rows[:, 1])
print(f'\n  -- 自我驗證：重建弦角 vs 記錄的 raw（{len(rows)} 筆）--')
print(f'     |差| p50 {np.percentile(err,50):.2f}°  p95 {np.percentile(err,95):.2f}°  '
      f'max {np.max(err):.2f}°')
ok = np.percentile(err, 95) < 5.0
print(f'     {"重建可採信" if ok else "!! 重建與記錄不符，以下數值不採信"}')

print(f'\n  -- 前視點是否已落在路徑末端 --')
print(f'     全區間 {len(rows)} 筆中，前視點 = 路徑末點的有 '
      f'{int(rows[:,4].sum())} 筆 = {100*rows[:,4].mean():.1f} %')
print(f'\n{"t":>7s} {"弦長 m":>8s} {"剩餘路徑 m":>10s} {"前視弧長 m":>10s} {"末點?":>5s}')
last = -9
for r in rows:
    if r[0] - last < 0.5: continue
    last = r[0]
    x = L[int(np.argmin(np.abs(lt - r[0])))]
    print(f'{r[0]:7.2f} {r[2]:8.3f} {r[3]:10.3f} {r[5]:10.3f} {int(r[4]):5d}'
          f'   距目標(真值) {x["dist_goal"]:.3f}')
