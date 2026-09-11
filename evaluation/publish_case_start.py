#!/usr/bin/env python3
"""發布 /case_start：先確認端點匹配，再**單次**發布，發布器持續存活。

為何不是「多等幾秒然後發」：訂閱者數 >= 1 只說明「有人訂閱」，不說明那是
本趟的 dynamic_obstacle_driver，也不說明型別與 QoS 相容。本檔逐項核對：
節點身分、topic 型別、QoS 相容性、以及本行程自己的發布器是否已看到匹配。

為何不重發：dynamic_obstacle_driver 收到每一則 /case_start 都會**重設相位零點**。
重送的第一則可能其實已被採用，只是確認訊息還沒被讀到；此時重送會把零點改掉。
「採用到的 epoch 比發布前的 clock 新」不能取代「同一次啟動只採用一次」。
若日後確實需要重送，正確方向是**帶 case ID 的冪等協定**（同一 ID 重送只回覆
原 epoch，不重設相位），不是時間門檻。

流程：匹配確認 -> 單次發布 -> 發布器保持存活並重讀 phase_epoch 直到有效或逾時。
"""
import argparse, json, math, os, sys, time

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Empty, Float64

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--driver-node', default='dynamic_obstacle_driver')
ap.add_argument('--match-timeout', type=float, default=30.0)
ap.add_argument('--epoch-timeout', type=float, default=30.0)
a = ap.parse_args()

rclpy.init()
n = Node('case_start_publisher')
n.set_parameters([Parameter('use_sim_time', value=True)])
pub = n.create_publisher(Empty, '/case_start', 10)
epoch = {'v': None}
n.create_subscription(Float64, '/dynamic_obstacles/phase_epoch',
                      lambda m: epoch.__setitem__('v', float(m.data)), 10)

rec = dict(schema='case_start/1', matched=False, published=False,
           publish_sim_t=None, phase_epoch=None, endpoints=None, reason=None)


def endpoints():
    subs = n.get_subscriptions_info_by_topic('/case_start')
    out = []
    for e in subs:
        try:
            qos = e.qos_profile
            q = dict(reliability=str(qos.reliability), durability=str(qos.durability),
                     history=str(qos.history), depth=int(qos.depth))
        except Exception:
            q = None
        out.append(dict(node=f'{e.node_namespace}/{e.node_name}'.replace('//', '/'),
                        topic_type=e.topic_type, qos=q))
    return out


print(f'--- 等待端點匹配（driver = {a.driver_node}，domain '
      f'{os.environ.get("ROS_DOMAIN_ID", "0")}）---')
t0 = time.monotonic()
eps = []
while time.monotonic() - t0 < a.match_timeout:
    rclpy.spin_once(n, timeout_sec=0.1)
    eps = endpoints()
    tgt = [e for e in eps if e['node'].endswith(a.driver_node)]
    if not tgt:
        continue
    # 型別必須一致；發布器自己看到的匹配數也要 >= 1
    bad = [e for e in tgt if e['topic_type'] not in ('std_msgs/msg/Empty',)]
    if bad:
        rec['reason'] = f'型別不符: {[e["topic_type"] for e in bad]}'
        break
    if pub.get_subscription_count() >= 1:
        rec['matched'] = True
        break
rec['endpoints'] = eps
for e in eps:
    print(f'    {e["node"]}  type={e["topic_type"]}  qos={e["qos"]}')
print(f'    發布器看到的匹配訂閱數 = {pub.get_subscription_count()}')

if not rec['matched']:
    rec['reason'] = rec['reason'] or '逾時：未找到相容的 driver 訂閱端'
    print(f'  !! {rec["reason"]}')
else:
    sim = n.get_clock().now().nanoseconds * 1e-9
    pub.publish(Empty())
    rec['published'] = True
    rec['publish_sim_t'] = sim
    print(f'--- 已單次發布 /case_start（模擬時間 {sim:.3f} s）；'
          f'發布器保持存活，只重讀不重發 ---')
    t1 = time.monotonic()
    while time.monotonic() - t1 < a.epoch_timeout:
        rclpy.spin_once(n, timeout_sec=0.1)
        v = epoch['v']
        if v is not None and math.isfinite(v):
            rec['phase_epoch'] = v
            print(f'    driver 已採用：phase_epoch = {v:.3f} s')
            break
    if rec['phase_epoch'] is None:
        rec['reason'] = '已發布但逾時未取得有限的 phase_epoch'
        print(f'  !! {rec["reason"]}')

n.destroy_node(); rclpy.shutdown()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(rec, open(a.out, 'w'), ensure_ascii=False, indent=1)
ok = rec['phase_epoch'] is not None
print(f'--- 結果：{"通過" if ok else "未通過"} ---')
sys.exit(0 if ok else 1)
