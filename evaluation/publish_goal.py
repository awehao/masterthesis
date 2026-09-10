#!/usr/bin/env python3
"""發布 /goal_pose，並留下可核對的發布事件。

`ros2 topic pub` 送出的 header.stamp 是零，所以「發布時間」在事後無從查證，
只能退而用「模擬器收到目標的時間」，而後者已被證實會晚於 /plan 與運動開始
（後方趟：收到 98.11 s，第一個 /plan 97.48 s）。這支程式以模擬時間戳記每一則
目標，並把發布事件寫成 JSON，讓任務計時可以從發布本身起算。
"""
import argparse, json, math, os, time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped

ap = argparse.ArgumentParser()
ap.add_argument('--x', type=float, required=True)
ap.add_argument('--y', type=float, required=True)
ap.add_argument('--frame', default='map')
ap.add_argument('--count', type=int, default=5)
ap.add_argument('--period', type=float, default=1.0, help='牆鐘秒')
ap.add_argument('--subs', type=int, default=2, help='等到這麼多訂閱者才發第一則')
ap.add_argument('--wait', type=float, default=30.0, help='等訂閱者的牆鐘上限')
ap.add_argument('--out', required=True)
a = ap.parse_args()

rclpy.init()
n = Node('goal_publisher')
n.set_parameters([Parameter('use_sim_time', value=True)])
pub = n.create_publisher(PoseStamped, '/goal_pose', 10)

def sim_now():
    t = n.get_clock().now()
    return t.nanoseconds * 1e-9

# 等 /clock 真的在走，否則時間戳會是 0
t0 = time.monotonic()
while time.monotonic() - t0 < a.wait:
    rclpy.spin_once(n, timeout_sec=0.1)
    if sim_now() > 0.0:
        break
if sim_now() <= 0.0:
    print('  !! /clock 未前進，拒絕發布（時間戳會是 0，發布事件將無法核對）')
    n.destroy_node(); rclpy.shutdown(); raise SystemExit(2)

t0 = time.monotonic()
while time.monotonic() - t0 < a.wait:
    rclpy.spin_once(n, timeout_sec=0.1)
    if pub.get_subscription_count() >= a.subs:
        break
subs = pub.get_subscription_count()

events = []
for i in range(a.count):
    m = PoseStamped()
    st = sim_now()
    m.header.stamp = n.get_clock().now().to_msg()
    m.header.frame_id = a.frame
    m.pose.position.x = a.x
    m.pose.position.y = a.y
    m.pose.orientation.w = 1.0
    pub.publish(m)
    events.append(dict(index=i, sim_t=st, wall=time.time(),
                       subscribers=pub.get_subscription_count()))
    print(f'  發布 {i+1}/{a.count}  模擬時間 {st:.3f} s  訂閱者 '
          f'{events[-1]["subscribers"]}')
    t1 = time.monotonic()
    while time.monotonic() - t1 < a.period:
        rclpy.spin_once(n, timeout_sec=0.05)

rec = dict(goal=[a.x, a.y], frame=a.frame, subscribers_at_first=subs,
           publish_events=events,
           first_publish_sim_t=events[0]['sim_t'] if events else None)
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(rec, open(a.out, 'w'), ensure_ascii=False, indent=1)
print(f'  發布事件 -> {a.out}')
n.destroy_node(); rclpy.shutdown()
