#!/usr/bin/env python3
"""診斷原始朝向參考的最大單步跳動來源。不改控制器、不重跑。

前視點座標與路徑索引在執行時未被記錄，只能用保存的 /plan、/odometry/filtered
與各筆 /gmpc/heading 的時戳**重建**。重建欄位一律標示「(重建)」，並先以
「重建弦角能否重現記錄下來的 raw」自我驗證；重現不準就不採信。

分類目標：路徑切換、索引跳躍、前視弦過短、或路徑本身急轉。
"""
import argparse, json, math, os
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry, Path

ap = argparse.ArgumentParser()
ap.add_argument('run_dir')
ap.add_argument('--lookahead', type=float, default=1.2)
ap.add_argument('--window', type=float, default=2.0)
a = ap.parse_args()

def hs(m):
    s = m.header.stamp; return s.sec + s.nanosec * 1e-9
def wrap(x): return (x + math.pi) % (2 * math.pi) - math.pi
def deg(x): return math.degrees(x)

rd = rosbag2_py.SequentialReader()
rd.open(rosbag2_py.StorageOptions(uri=os.path.join(a.run_dir, 'bag'),
                                  storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''))
head, plans, ekf, cmd = [], [], [], []
for_ = 0
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
    elif t == '/cmd_vel':
        m = deserialize_message(data, Twist)
        cmd.append((ts * 1e-9, m.angular.z))
head = np.array(head); ekf = np.array(ekf)
run = json.load(open(os.path.join(a.run_dir, 'isaac_run.json')))['run']
t0 = run['motion_start_sim_t']; te = run['at_trigger']['sim_t']
L = run['log']; lt = np.array([x['t'] for x in L])
pt = np.array([p[0] for p in plans]); et = ekf[:, 0]

H = head[(head[:, 0] >= t0) & (head[:, 0] <= te)]
print(f'== {os.path.basename(a.run_dir)} 原始朝向跳動診斷 ==')
print(f'  任務區間 {t0:.2f}-{te:.2f} s，/gmpc/heading {len(H)} 筆，'
      f'/plan {len(plans)} 則')

# ---- 0. 跳動的計算方式：包裹 vs 未包裹 -----------------------------------
raw = H[:, 1]
d_wrap = np.array([wrap(b - x) for x, b in zip(raw[:-1], raw[1:])])
d_naive = np.diff(raw)
i = int(np.argmax(np.abs(d_wrap)))
print(f'\n  -- 0. 跳動計算方式 --')
print(f'     以包裹到 ±pi 的角度差計算：max |Δraw| = {deg(np.max(np.abs(d_wrap))):.2f}°'
      f'（t = {H[i,0]:.2f} -> {H[i+1,0]:.2f}）')
print(f'     若不包裹直接相減：      max |Δraw| = {deg(np.max(np.abs(d_naive))):.2f}°')
n_fake = int(np.sum(np.abs(d_naive) > math.pi))
print(f'     |未包裹差| > 180° 的樣本 {n_fake} 筆'
      f' -> {"存在表示方式造成的假跳動，已由包裹排除" if n_fake else "無表示方式假跳動"}')
print(f'     兩者最大值不同 = {abs(deg(np.max(np.abs(d_wrap))) - deg(np.max(np.abs(d_naive)))) > 0.01}')

tj0, tj1 = H[i, 0], H[i + 1, 0]
print(f'     最大跳動：{deg(raw[i]):.2f}° -> {deg(raw[i+1]):.2f}°，'
      f'Δ = {deg(d_wrap[i]):+.2f}°，Δt = {tj1-tj0:.3f} s')

# ---- 1. 重建前視點 --------------------------------------------------------
def rebuild(th):
    ip = int(np.searchsorted(pt, th)) - 1
    if ip < 0: return None
    path = plans[ip][1]
    ie = int(np.argmin(np.abs(et - th)))
    if abs(et[ie] - th) > 0.2: return None
    rp = ekf[ie, 1:3]
    d = np.linalg.norm(path - rp, axis=1)
    i0 = int(np.argmin(d))
    seg = np.linalg.norm(np.diff(path[i0:], axis=0), axis=1)
    if seg.size == 0: return None
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    j = min(int(np.searchsorted(cum, a.lookahead)), len(cum) - 1)
    tgt = path[i0 + j]; chord = tgt - rp
    return dict(plan_idx=ip, plan_t=plans[ip][0], n=len(path), rp=rp,
                i0=i0, j=j, tgt_idx=i0 + j, tgt=tgt,
                chord_len=float(np.linalg.norm(chord)),
                ang=math.atan2(chord[1], chord[0]),
                remain=float(cum[-1]), arc=float(cum[j]),
                at_end=int((i0 + j) == len(path) - 1))

rec = [(h[0], h[1], h[2], h[3], rebuild(h[0])) for h in H]
val = [(t, r_, f_, y_, b) for t, r_, f_, y_, b in rec if b]
err = np.array([abs(wrap(b['ang'] - r_)) for t, r_, f_, y_, b in val])
print(f'\n  -- 1. 重建自我驗證（{len(val)} 筆）--')
print(f'     |重建弦角 − 記錄 raw| p50 {deg(np.percentile(err,50)):.2f}°  '
      f'p95 {deg(np.percentile(err,95)):.2f}°  max {deg(np.max(err)):.2f}°')
ok95 = deg(np.percentile(err, 95)) < 5.0
print(f'     {"重建可採信（以下前視點/索引欄位皆為重建值）" if ok95 else "!! 重建不可採信"}')

# ---- 2. 跳動前後 ±window 的對齊表 ----------------------------------------
print(f'\n  -- 2. 跳動時刻 ±{a.window:.1f} s 對齊（前視點、索引、剩餘長度皆為重建）--')
print(f'{"t":>7s} {"raw°":>8s} {"ref°":>8s} {"yaw°":>8s} {"wz_cmd":>7s} '
      f'{"位置(x,y)":>16s} {"前視點(x,y)":>16s} {"弦長":>6s} '
      f'{"i0":>4s} {"j":>3s} {"tgt":>4s} {"n":>4s} {"剩餘":>6s} {"末":>2s} {"plan#":>5s} {"plan_t":>7s}')
cmd_a = np.array(cmd) if cmd else np.zeros((0, 2))
for t, r_, f_, y_, b in rec:
    if not (tj0 - a.window <= t <= tj1 + a.window): continue
    if b is None:
        print(f'{t:7.2f} {deg(r_):8.2f} {deg(f_):8.2f} {deg(y_):8.2f}   （重建失敗）')
        continue
    w = float('nan')
    if len(cmd_a):
        k = int(np.argmin(np.abs(cmd_a[:, 0] - t)))
        w = cmd_a[k, 1]
    mark = ' <<' if abs(t - tj1) < 1e-6 else ''
    print(f'{t:7.2f} {deg(r_):8.2f} {deg(f_):8.2f} {deg(y_):8.2f} {w:7.3f} '
          f'({b["rp"][0]:6.3f},{b["rp"][1]:6.3f}) ({b["tgt"][0]:6.3f},{b["tgt"][1]:6.3f}) '
          f'{b["chord_len"]:6.3f} {b["i0"]:4d} {b["j"]:3d} {b["tgt_idx"]:4d} {b["n"]:4d} '
          f'{b["remain"]:6.3f} {b["at_end"]:2d} {b["plan_idx"]:5d} {b["plan_t"]:7.2f}{mark}')

# ---- 3. 分類 --------------------------------------------------------------
ba = next((b for t, r_, f_, y_, b in rec if abs(t - tj0) < 1e-6), None)
bb = next((b for t, r_, f_, y_, b in rec if abs(t - tj1) < 1e-6), None)
print(f'\n  -- 3. 分類 --')
if ba and bb:
    print(f'     路徑版本 plan#{ba["plan_idx"]}(t={ba["plan_t"]:.2f}, n={ba["n"]}) '
          f'-> plan#{bb["plan_idx"]}(t={bb["plan_t"]:.2f}, n={bb["n"]})'
          f'  {"**切換**" if ba["plan_idx"] != bb["plan_idx"] else "未切換"}')
    print(f'     最近索引 i0 {ba["i0"]} -> {bb["i0"]}  （Δ {bb["i0"]-ba["i0"]:+d}）')
    print(f'     前視索引 tgt {ba["tgt_idx"]} -> {bb["tgt_idx"]}  （Δ {bb["tgt_idx"]-ba["tgt_idx"]:+d}）')
    print(f'     前視點 ({ba["tgt"][0]:.3f},{ba["tgt"][1]:.3f}) -> '
          f'({bb["tgt"][0]:.3f},{bb["tgt"][1]:.3f})  移動 '
          f'{np.linalg.norm(bb["tgt"]-ba["tgt"]):.3f} m')
    print(f'     機器人 ({ba["rp"][0]:.3f},{ba["rp"][1]:.3f}) -> '
          f'({bb["rp"][0]:.3f},{bb["rp"][1]:.3f})  移動 '
          f'{np.linalg.norm(bb["rp"]-ba["rp"]):.3f} m')
    print(f'     弦長 {ba["chord_len"]:.3f} -> {bb["chord_len"]:.3f} m'
          f'  （< 0.01 m 會被 1 cm 保護擋下）')
    print(f'     剩餘路徑 {ba["remain"]:.3f} -> {bb["remain"]:.3f} m；'
          f'前視點在末端 {ba["at_end"]} -> {bb["at_end"]}')

# ---- 4. 跳動之後是否造成實際轉向 -----------------------------------------
print(f'\n  -- 4. 跳動之後的實際轉向（限速後）--')
seg = [(t, r_, f_, y_) for t, r_, f_, y_, b in rec if tj1 - 0.2 <= t <= tj1 + 3.0]
if len(seg) > 2:
    ys = [s[3] for s in seg]
    cum_turn = sum(abs(wrap(b - x)) for x, b in zip(ys[:-1], ys[1:]))
    net = wrap(ys[-1] - ys[0])
    print(f'     跳動後 3 s 內實際 yaw {deg(ys[0]):.2f}° -> {deg(ys[-1]):.2f}°，'
          f'淨 {deg(net):+.2f}°，累計絕對 {deg(cum_turn):.2f}°')
    fs = [s[2] for s in seg]
    print(f'     同期限速後參考 {deg(fs[0]):.2f}° -> {deg(fs[-1]):.2f}°，'
          f'淨 {deg(wrap(fs[-1]-fs[0])):+.2f}°')
    rs = [s[1] for s in seg]
    print(f'     同期原始參考   {deg(rs[0]):.2f}° -> {deg(rs[-1]):.2f}°，'
          f'淨 {deg(wrap(rs[-1]-rs[0])):+.2f}°')
    # 原始是否很快回到跳動前的值（單筆突刺）
    back = [abs(wrap(r_ - raw[i])) for t, r_, f_, y_ in seg]
    print(f'     原始參考回到跳動前值的最小差 = {deg(min(back)):.2f}°'
          f'（{"是單筆突刺，很快回復" if min(back) < math.radians(10) else "未回復，是持續性變化"}）')
