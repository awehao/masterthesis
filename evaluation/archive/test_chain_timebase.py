#!/usr/bin/env python3
"""以序號配對，分別用 guard 自身時間基準與接收時間基準計算輪加速度。

回答：那幾筆超出是**控制輸出真的超限**，還是**量測時間基準不同**。
"""
import argparse, json, math, os, time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='evaluation/results/chain_timebase.json')
ap.add_argument('--rate', type=float, default=20.0)
a = ap.parse_args()

R, L, W_MAX, A_MAX = 0.05, 0.245, 5.55, 125.0
W = np.array([[0.0, 1.0, L], [-1.0, 0.0, L], [0.0, -1.0, L], [1.0, 0.0, L]])
om = lambda u: (W @ np.asarray(u, float)) / R      # noqa: E731


def schedule():
    seq = []
    h = lambda u, n: seq.extend([tuple(u)] * n)     # noqa: E731
    h((0., 0., 0.), 10); h((0.20, 0., 0.), 30); h((0., 0., 1.00), 30)
    h((0., 0., -1.00), 30); h((0., 0., 0.), 20)
    h((0.15, -0.15, 0.), 30); h((-0.15, 0.15, 0.40), 30); h((0., 0., 0.), 20)
    return seq


rclpy.init()
n = Node('chain_timebase')
pub = n.create_publisher(Twist, '/t_cmd_in', 10)
recv, stat = [], []
n.create_subscription(Twist, '/t_cmd_out',
                      lambda m: recv.append((time.monotonic(), m.linear.x,
                                             m.linear.y, m.angular.z)), 10)
n.create_subscription(String, '/wheel_guard/status',
                      lambda m: stat.append(json.loads(m.data)), 50)

t0 = time.monotonic()
while time.monotonic() - t0 < 10 and pub.get_subscription_count() < 1:
    rclpy.spin_once(n, timeout_sec=0.1)
if pub.get_subscription_count() < 1:
    print('  !! 上游未訂閱'); raise SystemExit(2)

period, nxt = 1.0 / a.rate, time.monotonic()
for u in schedule():
    m = Twist(); m.linear.x, m.linear.y, m.angular.z = map(float, u)
    pub.publish(m); nxt += period
    while time.monotonic() < nxt:
        rclpy.spin_once(n, timeout_sec=0.002)
te = time.monotonic()
while time.monotonic() - te < 1.0:
    rclpy.spin_once(n, timeout_sec=0.01)
n.destroy_node(); rclpy.shutdown()

print(f'  /t_cmd_out {len(recv)} 則   /wheel_guard/status {len(stat)} 則')
S = [s for s in stat if s.get('schema') == 'wheel_guard/1']
S.sort(key=lambda s: s['seq'])
print(f'  schema 相符 {len(S)} 則；seq 連續？ '
      f'{all(S[i+1]["seq"] - S[i]["seq"] == 1 for i in range(len(S)-1))}')

# ---- guard 自身時間基準：用它記錄的 dt 與自己前後筆的 out 配對 ----
rows = []
for i in range(1, len(S)):
    p, c = S[i - 1], S[i]
    if c.get('dt') is None:
        continue
    dw = float(np.max(np.abs(om(c['out']) - om(p['out']))))
    rows.append(dict(seq=c['seq'], dt_guard=c['dt'], dwmax=dw,
                     acc_guard=dw / c['dt'], stamp=c['stamp'],
                     action=c['action'], lam=c['lam'],
                     accel_guaranteed=c['accel_guaranteed']))
G = np.array([[r['acc_guard'], r['dt_guard'], r['dwmax']] for r in rows])
overg = int((G[:, 0] > A_MAX + 1e-6).sum())
print(f'\n  === guard 自身時間基準 ===')
print(f'    輪加速 max {G[:,0].max():.4f} / 上限 {A_MAX}  超出 {overg}/{len(G)}')
print(f'    最大超出 {max(G[:,0].max()-A_MAX, 0.0):.6f}')
print(f'    dt 中位 {np.median(G[:,1]):.6f}  範圍 {G[:,1].min():.6f}–{G[:,1].max():.6f}')

# ---- 接收時間基準 ----
O = np.array(recv)
w = np.array([om(r[1:]) for r in O])
dtr = np.diff(O[:, 0]); dwr = np.max(np.abs(np.diff(w, axis=0)), axis=1)
accr = dwr / np.maximum(dtr, 1e-9)
overr = int((accr > A_MAX + 1e-6).sum())
print(f'\n  === 接收時間基準 ===')
print(f'    輪加速 max {accr.max():.4f} / 上限 {A_MAX}  超出 {overr}/{len(accr)}')
print(f'    最大超出 {max(accr.max()-A_MAX, 0.0):.6f}')
print(f'    Δt 中位 {np.median(dtr):.6f}  範圍 {dtr.min():.6f}–{dtr.max():.6f}')

bad = [r for r in rows if r['acc_guard'] > A_MAX + 1e-6]
print(f'\n  === guard 基準下的超出逐筆（{len(bad)} 筆）===')
for r in bad[:10]:
    print(f'    seq {r["seq"]}  dt={r["dt_guard"]:.6f}  Δω={r["dwmax"]:.5f}  '
          f'acc={r["acc_guard"]:.4f}  action={r["action"]}  λ={r["lam"]}')
ib = int(np.argmax(accr))
print(f'\n  === 接收基準下最大那筆 ===')
print(f'    Δt={dtr[ib]:.6f}  Δω={dwr[ib]:.5f}  acc={accr[ib]:.4f}')

res = dict(guard_basis=dict(acc_max=float(G[:, 0].max()), over=overg, n=len(G),
                            excess=float(max(G[:, 0].max()-A_MAX, 0.0))),
           recv_basis=dict(acc_max=float(accr.max()), over=overr, n=len(accr),
                           excess=float(max(accr.max()-A_MAX, 0.0))),
           bad_guard=bad[:20], limit=A_MAX)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(dict(result=res, rows=rows, recv=O.tolist()), open(a.out, 'w'), indent=1)
print(f'\n  -> {a.out}')
print(f'  判定：guard 基準 {"PASS" if overg == 0 else "FAIL"}  '
      f'／ 接收基準 {"PASS" if overr == 0 else "FAIL"}')
