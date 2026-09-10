#!/usr/bin/env python3
"""scheduled 情境的三段檢查：定位、相位零點、開始移動。

  preposition  障礙物是否已在相位 0 的位置且靜止（任務開始前就放好並確認穩定）
  epoch        /dynamic_obstacles/phase_epoch 是否已被驅動採用，回傳其值
  moving       /case_start 之後障礙物是否確實開始移動
"""
import argparse, json, math, os, sys, time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64

ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['preposition', 'epoch', 'moving'])
ap.add_argument('--traj', default='', help='相位資產（preposition 需要）')
ap.add_argument('--window', type=float, default=3.0)
ap.add_argument('--tol', type=float, default=0.05, help='定位容差 m')
ap.add_argument('--max-speed', type=float, default=0.02)
ap.add_argument('--min-moving', type=int, default=8)
ap.add_argument('--out', default='')
a = ap.parse_args()

import yaml

def phase0(d):
    sx, sy = [float(v) for v in d['start']]
    ex, ey = [float(v) for v in d['end']]
    L = math.hypot(ex - sx, ey - sy)
    if L < 1e-9:
        return sx, sy
    ux, uy = (ex - sx) / L, (ey - sy) / L
    p0 = float(d.get('phase0_m', 0.0)); dr = float(d.get('direction', 1.0))
    s0 = p0 if dr >= 0 else (2.0 * L - p0)
    s = s0 % (2.0 * L)
    dd = s if s <= L else 2.0 * L - s
    return sx + ux * dd, sy + uy * dd

rclpy.init()
n = Node('case_start_check')
n.set_parameters([Parameter('use_sim_time', value=True)])
be = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST)
hist = {}
def mk(t):
    def cb(m, tt=t):
        s = m.header.stamp
        hist.setdefault(tt, []).append((s.sec + s.nanosec * 1e-9,
                                        m.pose.position.x, m.pose.position.y))
    n.create_subscription(PoseStamped, t, cb, be)

names = []
if a.traj:
    names = [d['name'] for d in (yaml.safe_load(open(a.traj))
                                 .get('dynamic_obstacles') or [])]
else:
    names = [f'dyn_obs_{i}' for i in range(10)]
for nm in names:
    mk(f'/model/{nm}/pose')

epoch = {'v': None}
n.create_subscription(Float64, '/dynamic_obstacles/phase_epoch',
                      lambda m: epoch.__setitem__('v', float(m.data)), 10)

t0 = time.monotonic()
while time.monotonic() - t0 < a.window:
    rclpy.spin_once(n, timeout_sec=0.05)

def speeds():
    out = {}
    for t, h in hist.items():
        v = 0.0
        for (ta, xa, ya), (tb, xb, yb) in zip(h, h[1:]):
            dt = tb - ta
            if dt > 1e-6:
                v = max(v, math.hypot(xb - xa, yb - ya) / dt)
        out[t] = (v, len(h), (h[-1][1], h[-1][2]) if h else None)
    return out

rc = 0
res = {'stage': a.stage}
if a.stage == 'preposition':
    cfg = {d['name']: phase0(d) for d in
           (yaml.safe_load(open(a.traj)).get('dynamic_obstacles') or [])}
    sp = speeds()
    print(f'--- 相位 0 定位檢查（容差 {a.tol} m，靜止門檻 {a.max_speed} m/s）---')
    rows = []
    for nm, (px, py) in sorted(cfg.items()):
        t = f'/model/{nm}/pose'
        if t not in sp or sp[t][1] < 2:
            print(f'  {nm:10s} 無足夠樣本'); rc = 1; continue
        v, cnt, pos = sp[t]
        err = math.hypot(pos[0] - px, pos[1] - py)
        ok = err <= a.tol and v <= a.max_speed
        rc = rc or (0 if ok else 1)
        rows.append(dict(name=nm, err_m=err, speed=v, pos=list(pos),
                         phase0=[px, py], ok=ok))
        print(f'  {nm:10s} 實際 ({pos[0]:7.3f},{pos[1]:7.3f})  '
              f'相位0 ({px:7.3f},{py:7.3f})  偏差 {err*1000:7.2f} mm  '
              f'速度 {v:.4f} m/s  {"" if ok else "<< 未過"}')
    res['rows'] = rows
elif a.stage == 'epoch':
    print(f'--- 相位零點 ---')
    if epoch['v'] is None:
        print('  !! 未收到 /dynamic_obstacles/phase_epoch：驅動未採用 /case_start')
        rc = 1
    else:
        print(f'  phase_epoch = {epoch["v"]:.3f} s（模擬時間）')
        res['phase_epoch'] = epoch['v']
else:
    sp = speeds()
    mov = sum(1 for t in sp if sp[t][0] > a.max_speed)
    print(f'--- /case_start 之後的移動確認 ---')
    for t in sorted(sp):
        print(f'  {t:28s} 最大瞬時速度 {sp[t][0]:.4f} m/s')
    print(f'  移動中 {mov}/{len(sp)}（要求 >= {a.min_moving}）')
    res['moving'] = mov
    if mov < a.min_moving:
        print('  !! 移動體數量不足，情境未如預期啟動'); rc = 1

n.destroy_node(); rclpy.shutdown()
if a.out:
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)
print(f'--- 結果：{"通過" if rc == 0 else "未通過"} ---')
sys.exit(rc)
