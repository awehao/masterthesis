"""第三方計數器：在獨立行程裡數指定 topic 的到達則數與起訖時間。

用途是把「發布端有沒有發」和「某個接收端有沒有收到」分開。如果這個行程收得到
/cmd_vel 而 Isaac 收不到，那問題就不在 guard 的發布側；如果連這裡都收不到，
指向的是發布側或共同的通訊設定。

    python3 evaluation/count_topics.py --seconds 25 --out out.json \
        --topic /cmd_vel=geometry_msgs/msg/Twist \
        --topic /wheel_guard/status=std_msgs/msg/String
"""
import argparse, importlib, json, sys, time
import rclpy
from rclpy.node import Node

ap = argparse.ArgumentParser()
ap.add_argument('--topic', action='append', required=True,
                help='TOPIC=pkg/msg/Type')
ap.add_argument('--seconds', type=float, default=25.0)
ap.add_argument('--out', required=True)
a = ap.parse_args()


def load(t):
    pkg, _, name = t.split('/')
    return getattr(importlib.import_module(f'{pkg}.msg'), name)


rclpy.init()
n = Node('count_topics')
stats = {}
for spec in a.topic:
    topic, tname = spec.split('=', 1)
    cls = load(tname)
    stats[topic] = dict(type=tname, count=0, first_wall=None, last_wall=None)

    def cb(msg, _t=topic):
        # 計在入口，不看內容
        s = stats[_t]
        s['count'] += 1
        now = time.time()
        if s['first_wall'] is None:
            s['first_wall'] = now
        s['last_wall'] = now

    n.create_subscription(cls, topic, cb, 10)

t0 = time.monotonic()
while time.monotonic() - t0 < a.seconds:
    rclpy.spin_once(n, timeout_sec=0.1)

for t, s in stats.items():
    s['matched_publishers'] = len(n.get_publishers_info_by_topic(t))
    span = ((s['last_wall'] - s['first_wall'])
            if s['first_wall'] and s['last_wall'] else None)
    s['span_s'] = round(span, 3) if span else None
    s['hz'] = round(s['count'] / span, 2) if span and span > 0 else None
    print(f"  {t:<26} 收到 {s['count']:5d} 則  匹配發布者 {s['matched_publishers']}"
          + (f"  跨度 {s['span_s']:.1f} s  約 {s['hz']} Hz" if s['hz'] else ''))

n.destroy_node(); rclpy.shutdown()
json.dump(dict(schema='topic_count/1', seconds=a.seconds, stats=stats),
          open(a.out, 'w'), ensure_ascii=False, indent=1)
print(f'  -> {a.out}')
