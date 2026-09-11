#!/usr/bin/env python3
"""關鍵 topic 的端點身分檢查（不只計數）。

發目標前確認：
  * /cmd_vel 的發布端**只有 wheel_limit_guard**；
  * 每個關鍵 topic 的發布／訂閱節點名稱逐一列出並存檔。

先前一趟 ON 因同 domain 殘留的導航鏈而作廢，當時只有訂閱者「計數」可看，
無法辨識來源。這支檢查記錄的是節點身分。
"""
import argparse, json, os, sys, time
import rclpy
from rclpy.node import Node

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--sole-publisher', action='append', default=[],
                help='TOPIC=NODE，要求該 topic 的發布端只有這一個節點')
ap.add_argument('--topics', default='/cmd_vel,/cmd_vel_nav,/cmd_vel_smoothed,'
                                    '/goal_pose,/plan,/clock,/scan,/odom')
ap.add_argument('--wait', type=float, default=20.0)
a = ap.parse_args()

rclpy.init()
n = Node('endpoint_check')
time.sleep(2.0)          # 讓 discovery 完成
topics = [t for t in a.topics.split(',') if t]
info = {}
t0 = time.monotonic()
while time.monotonic() - t0 < a.wait:
    info = {}
    for t in topics:
        pubs = [(e.node_name, e.node_namespace)
                for e in n.get_publishers_info_by_topic(t)]
        subs = [(e.node_name, e.node_namespace)
                for e in n.get_subscriptions_info_by_topic(t)]
        info[t] = dict(publishers=[f'{ns}/{nm}'.replace('//', '/') for nm, ns in pubs],
                       subscribers=[f'{ns}/{nm}'.replace('//', '/') for nm, ns in subs])
    if all(info[t]['publishers'] for t in ('/cmd_vel', '/clock')):
        break
    time.sleep(1.0)

rc = 0
print('--- 關鍵 topic 端點身分 ---')
for t in topics:
    print(f'  {t}')
    print(f'      發布: {info[t]["publishers"] or "（無）"}')
    print(f'      訂閱: {info[t]["subscribers"] or "（無）"}')

print('--- 唯一發布者要求 ---')
for spec in a.sole_publisher:
    topic, want = spec.split('=', 1)
    got = info.get(topic, {}).get('publishers', [])
    ok = len(got) == 1 and got[0].endswith(want)
    print(f'  {topic} 必須只由 {want} 發布 -> 實際 {got}  {"OK" if ok else "**不符**"}')
    if not ok:
        rc = 1

n.destroy_node(); rclpy.shutdown()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(dict(schema='endpoint_check/1', topics=info,
               sole_publisher=a.sole_publisher, rc=rc),
          open(a.out, 'w'), ensure_ascii=False, indent=1)
print(f'--- 結果：{"通過" if rc == 0 else "未通過"} ---')
sys.exit(rc)
