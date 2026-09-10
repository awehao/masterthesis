#!/usr/bin/env python3
"""左側趟的離線對齊分析：把朝向鏈的每一段放在同一條模擬時間軸上。

要回答的問題只有一個：累計/淨轉角 = 2.00 的反向轉動，是**朝向參考本身折返**，
還是**控制器對一個單調參考的追蹤過衝**。這兩者的修法完全不同，所以先分清。

同軸比對：
  raw   /gmpc/heading.x  限速前的 機器人->前視點 弦角
  ref   /gmpc/heading.y  限速後、實際進 QP 的參考
  yaw   /gmpc/heading.z  控制器當下看到的實際 yaw
  真值 yaw（isaac_run.json 的 log，與上面獨立）
  wz    /cmd_vel 的角速度命令
  /plan 到達時刻
"""
import argparse, json, math, os
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Path

ap = argparse.ArgumentParser()
ap.add_argument('run_dir')
ap.add_argument('--csv', default='')
a = ap.parse_args()

def wrap(x): return (x + math.pi) % (2 * math.pi) - math.pi
def hs(m):
    s = m.header.stamp; return s.sec + s.nanosec * 1e-9

rd = rosbag2_py.SequentialReader()
rd.open(rosbag2_py.StorageOptions(uri=os.path.join(a.run_dir, 'bag'),
                                  storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''))
head, plans, cmd = [], [], []
while rd.has_next():
    t, data, ts = rd.read_next()
    if t == '/gmpc/heading':
        m = deserialize_message(data, Vector3Stamped)
        head.append((hs(m), m.vector.x, m.vector.y, m.vector.z))
    elif t == '/plan':
        m = deserialize_message(data, Path)
        plans.append((hs(m), len(m.poses)))
    elif t == '/cmd_vel':
        m = deserialize_message(data, Twist)
        cmd.append((ts * 1e-9, m.angular.z))
head = np.array(head); plans = np.array(plans)

run = json.load(open(os.path.join(a.run_dir, 'isaac_run.json')))['run']
L = run['log']
t0 = run['motion_start_sim_t']
tend = run['at_trigger']['sim_t']
tru = np.array([(x['t'], x['yaw'], x['cmd'][2]) for x in L
                if t0 - 0.5 <= x['t'] <= tend + 1e-9])

h = head[(head[:, 0] >= t0 - 0.5) & (head[:, 0] <= tend)]
print(f'== {os.path.basename(a.run_dir)} 朝向鏈對齊 ==')
print(f'  任務區間（運動開始 -> 停止觸發）{t0:.2f} -> {tend:.2f} s，'
      f'/gmpc/heading {len(h)} 筆，真值 {len(tru)} 筆，/plan {len(plans)} 則')

# --- 1. 參考本身是否折返 -------------------------------------------------
def turns(seq):
    """單調段落與反向量：回傳 (累計正轉, 累計反轉, 反向段數)."""
    d = np.array([wrap(b - x) for x, b in zip(seq[:-1], seq[1:])])
    pos = float(d[d > 0].sum()); neg = float(-d[d < 0].sum())
    sg = np.sign(np.where(np.abs(d) > math.radians(0.05), d, 0.0))
    sg = sg[sg != 0]
    flips = int(np.sum(sg[1:] != sg[:-1])) if len(sg) > 1 else 0
    return pos, neg, flips

for nm, col in (('原始朝向 raw', 1), ('限速後 ref', 2), ('實際 yaw（控制器看到）', 3)):
    v = h[:, col]
    v = v[np.isfinite(v)]
    if len(v) < 3: continue
    pos, neg, fl = turns(v)
    print(f'  {nm:24s} 正轉 {math.degrees(pos):7.2f}°  反轉 {math.degrees(neg):7.2f}°  '
          f'淨 {math.degrees(pos-neg):7.2f}°  方向反轉段 {fl}')
pos, neg, fl = turns(tru[:, 1])
print(f'  {"真值 yaw":24s} 正轉 {math.degrees(pos):7.2f}°  反轉 {math.degrees(neg):7.2f}°  '
      f'淨 {math.degrees(pos-neg):7.2f}°  方向反轉段 {fl}')
print('  （門檻：單步變化 > 0.05° 才計入方向判定，避免近零雜訊）')

# --- 2. 反轉發生在哪一段時間 ---------------------------------------------
print('\n  -- 真值 yaw 的反向轉動區段（連續反向且累積 > 1°）--')
d = np.array([wrap(b - x) for x, b in zip(tru[:-1, 1], tru[1:, 1])])
i = 0
segs = []
while i < len(d):
    if d[i] < 0:
        j = i
        while j < len(d) and d[j] <= 0: j += 1
        amt = -float(d[i:j].sum())
        if amt > math.radians(1.0):
            segs.append((tru[i, 0], tru[j, 0], math.degrees(amt)))
        i = j
    else:
        i += 1
for s0, s1, amt in segs:
    k = (h[:, 0] >= s0) & (h[:, 0] <= s1)
    if k.sum() >= 2:
        raw0, raw1 = h[k][0, 1], h[k][-1, 1]
        ref0, ref1 = h[k][0, 2], h[k][-1, 2]
        dr = math.degrees(wrap(raw1 - raw0)); df = math.degrees(wrap(ref1 - ref0))
        print(f'    t={s0:7.2f}->{s1:7.2f} s  真值反轉 {amt:6.2f}°   '
              f'同期 raw 變化 {dr:+7.2f}°   ref 變化 {df:+7.2f}°')
    else:
        print(f'    t={s0:7.2f}->{s1:7.2f} s  真值反轉 {amt:6.2f}°   （無朝向樣本）')

# --- 3. 追蹤誤差與過衝 ----------------------------------------------------
fin = np.isfinite(h[:, 1]) & np.isfinite(h[:, 2])
e = np.array([wrap(r - y) for r, y in zip(h[fin][:, 2], h[fin][:, 3])])
print(f'\n  -- 追蹤 --')
print(f'    ref - yaw：p50 {math.degrees(np.percentile(np.abs(e),50)):6.2f}°  '
      f'p95 {math.degrees(np.percentile(np.abs(e),95)):6.2f}°  '
      f'max {math.degrees(np.max(np.abs(e))):6.2f}°')
# 過衝 = 誤差變號（yaw 追過 ref）
sg = np.sign(np.where(np.abs(e) > math.radians(0.5), e, 0.0)); sg = sg[sg != 0]
print(f'    ref-yaw 誤差變號（>0.5° 才計）= '
      f'{int(np.sum(sg[1:] != sg[:-1])) if len(sg)>1 else 0} 次'
      '  （變號代表 yaw 越過參考，即過衝）')

# --- 4. 重規劃與 raw 的關係 ----------------------------------------------
pl = plans[(plans[:, 0] >= t0) & (plans[:, 0] <= tend)]
print(f'\n  -- 重規劃 {len(pl)} 次（任務區間內）--')
for tp, n in pl:
    k = int(np.searchsorted(h[:, 0], tp))
    if 1 <= k < len(h) and np.isfinite(h[k, 1]) and np.isfinite(h[k-1, 1]):
        print(f'    t={tp:7.2f}  路徑點 {int(n):4d}  '
              f'raw 跳 {math.degrees(wrap(h[k,1]-h[k-1,1])):+6.2f}°  '
              f'ref 跳 {math.degrees(wrap(h[k,2]-h[k-1,2])):+6.2f}°')

if a.csv:
    import csv
    with open(a.csv, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['t', 'raw_deg', 'ref_deg', 'yaw_ctrl_deg'])
        for r in h:
            w.writerow([f'{r[0]:.3f}', f'{math.degrees(r[1]):.3f}',
                        f'{math.degrees(r[2]):.3f}', f'{math.degrees(r[3]):.3f}'])
    print(f'\n  -> {a.csv}')
