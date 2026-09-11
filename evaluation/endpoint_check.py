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
# 以下為選用；不給就完全維持原行為（導航跑批用的就是原行為）。
ap.add_argument('--resolve-wait', type=float, default=0.0,
                help='>0 時，額外等待端點「名稱解析完成」的秒數上限')
ap.add_argument('--require-endpoint', action='append', default=[],
                help='TOPIC=pub|sub，要求該 topic 至少有一個該方向的端點')
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

UNKNOWN = '_NODE_NAME_UNKNOWN_'


def scan():
    d = {}
    for t in topics:
        pubs = [(e.node_name, e.node_namespace)
                for e in n.get_publishers_info_by_topic(t)]
        subs = [(e.node_name, e.node_namespace)
                for e in n.get_subscriptions_info_by_topic(t)]
        d[t] = dict(publishers=[f'{ns}/{nm}'.replace('//', '/') for nm, ns in pubs],
                    subscribers=[f'{ns}/{nm}'.replace('//', '/') for nm, ns in subs])
    return d


def unresolved(d):
    out = []
    for t, v in d.items():
        for k in ('publishers', 'subscribers'):
            for nm in v[k]:
                if UNKNOWN in nm:
                    out.append((t, k, nm))
    return out


# 名稱解析等待（有上限）。未解析的名稱**不當作通過** —— 先前一趟就是因為
# guard 的名稱還沒解析，唯一發布者檢查誤判為不符而中止；反過來也發生過
# Isaac 未解析。等不到就據實列為「端點名稱尚未解析」，不靠固定等待後放行。
resolve_note = None
if a.resolve_wait > 0:
    t1 = time.monotonic()
    while time.monotonic() - t1 < a.resolve_wait:
        info = scan()
        if not unresolved(info):
            break
        time.sleep(0.5)
    left = unresolved(info)
    resolve_note = dict(waited_s=round(time.monotonic() - t1, 2),
                        unresolved=[list(x) for x in left])
    print(f'--- 名稱解析等待 {resolve_note["waited_s"]:.1f} s（上限 '
          f'{a.resolve_wait:.0f} s）---')
    if left:
        print(f'  !! 仍有 {len(left)} 個端點名稱未解析：')
        for t, k, nm in left:
            print(f'     {t} {k} {nm}')
    else:
        print('  所有列出的端點名稱都已解析')

rc = 0
print('--- 關鍵 topic 端點身分 ---')
for t in topics:
    print(f'  {t}')
    print(f'      發布: {info[t]["publishers"] or "（無）"}')
    print(f'      訂閱: {info[t]["subscribers"] or "（無）"}')

print('--- 端點存在要求 ---' if a.require_endpoint else '', end='')
if a.require_endpoint:
    print()
for spec in a.require_endpoint:
    topic, side = spec.split('=', 1)
    key = 'publishers' if side.startswith('pub') else 'subscribers'
    got = info.get(topic, {}).get(key, [])
    if not got:
        print(f'  {topic} 需要至少一個{"發布" if key=="publishers" else "訂閱"}端 '
              f'-> **端點缺失**')
        rc = 1
    elif all(UNKNOWN in g for g in got):
        print(f'  {topic} 的{"發布" if key=="publishers" else "訂閱"}端存在但 '
              f'**名稱尚未解析** -> {got}')
        rc = 1
    else:
        print(f'  {topic} {"發布" if key=="publishers" else "訂閱"}端 {got}  OK')

print('--- 唯一發布者要求 ---')
for spec in a.sole_publisher:
    topic, want = spec.split('=', 1)
    got = info.get(topic, {}).get('publishers', [])
    # 三種失敗分開講，不混為一談
    if not got:
        why = '**端點缺失**'
    elif any(UNKNOWN in g for g in got):
        why = '**名稱尚未解析**（不當作通過）'
    elif len(got) != 1:
        why = '**非預期端點：發布者不只一個**'
    elif not got[0].endswith(want):
        why = '**非預期端點**'
    else:
        why = None
    ok = why is None
    print(f'  {topic} 必須只由 {want} 發布 -> 實際 {got}  '
          f'{"OK" if ok else why}')
    if not ok:
        rc = 1

n.destroy_node(); rclpy.shutdown()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(dict(schema='endpoint_check/2', topics=info,
               sole_publisher=a.sole_publisher,
               require_endpoint=a.require_endpoint,
               name_resolution=resolve_note, rc=rc),
          open(a.out, 'w'), ensure_ascii=False, indent=1)
print(f'--- 結果：{"通過" if rc == 0 else "未通過"} ---')
sys.exit(rc)
