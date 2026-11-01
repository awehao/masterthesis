#!/usr/bin/env python3
"""候選轉接 SegTargetLink 的 ROS 接線核對（Codex 20261102_002139；獨立網域、無 Isaac）。
發布端把候選節點實際輸出的 schema 3 列（results/vision/seg3d_candidates_replay_ad045.json 的 out）照原標籤重發到 /bottle/target_candidates，
另插入四種壞訊息：無標籤、schema 2 標籤、欄位數不符、t_eval 不遞增。接收端是 bridge 型的 rclpy 節點＋SegTargetLink，sink 只收集呼叫。
核對：轉送數＝合格列數、四種壞訊息各拒收 1、欄位對應（snap_cx 等與發布值相同）、時間換算（snap_t_last＝ROS − T0）、年齡重算（＝轉送延遲＋節點評估時年齡）。
    source /opt/ros/jazzy/setup.bash; ROS_DOMAIN_ID=91 .venv-dl0/bin/python evaluation/test_bottle_segtarget_link.py
"""
import json
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_segtarget_link as SL                     # noqa: E402

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(name)
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def main():
    out = json.load(open(os.path.join(HERE, 'results', 'vision', 'seg3d_candidates_replay_ad045.json')))['out']
    fields = list(out[0].keys())
    label = 'bottle_target_candidates schema=3 frame=odom(world, z up, m) stamps=ros_sim_clock fields=' + ','.join(fields)
    rclpy.init()
    pub_n, sub_n = Node('link_test_pub'), Node('link_test_sub')
    got = []
    link = SL.SegTargetLink(sub_n, None, Float64MultiArray, sink=lambda t, c: got.append((t, c)))
    T0 = 3.30
    link.set_T0(T0)
    pub = pub_n.create_publisher(Float64MultiArray, '/bottle/target_candidates', 50)
    w0 = time.perf_counter()
    while pub_n.count_subscribers('/bottle/target_candidates') < 1 and time.perf_counter() - w0 < 10:
        rclpy.spin_once(pub_n, timeout_sec=0.1)

    def send(lb, data):
        m = Float64MultiArray()
        if lb is not None:
            m.layout.dim = [MultiArrayDimension(label=lb, size=len(data), stride=len(data))]
        m.data = [float(v) for v in data]
        pub.publish(m)
    sent = []
    DELAY = 0.03
    for i, o in enumerate(out[:60]):
        send(label, [o[f] for f in fields])
        sent.append(o)
        if i == 10:
            send(None, [o[f] for f in fields])
            send(label.replace('schema=3', 'schema=2'), [o[f] for f in fields])
            send(label, [o[f] for f in fields][:-1])
            send(label, [o[f] for f in fields])                              # 同一 t_eval 再送一次 ⇒ 不遞增
        for _ in range(3):
            rclpy.spin_once(sub_n, timeout_sec=0.01)
        link.tick(o['t_eval'] - T0 + DELAY)
    for _ in range(50):
        rclpy.spin_once(sub_n, timeout_sec=0.02)
    link.tick(out[59]['t_eval'] - T0 + 1.0)
    rep = link.report()
    print(f"   轉送 {rep['n_forwarded']}／{len(sent)}；拒收 {rep['rejected']}")
    chk('轉送數＝合格列數、四種壞訊息各拒收 1', rep['n_forwarded'] == len(sent) and rep['rejected'] == dict(no_label=1, schema_or_frame=1, field_count=1, t_eval_not_increasing=1), rep)
    ok_map = all(abs(c['snap_cx'] - o['k_snap_cx']) < 1e-12 or (np.isnan(c['snap_cx']) and np.isnan(o['k_snap_cx'])) for (_, c), o in zip(got, sent))
    ok_t = all((np.isnan(o['k_snap_t_last']) and np.isnan(c['snap_t_last'])) or abs(c['snap_t_last'] - (o['k_snap_t_last'] - T0)) < 1e-9 for (_, c), o in zip(got, sent))
    ages = [(c['age_s'], DELAY + o['k_age_s']) for (_, c), o in zip(got, sent) if np.isfinite(o['k_age_s'])]
    ok_age = all(abs(a - b) < 1e-6 for a, b in ages)
    chk('欄位對應（snap_cx 等＝發布值）、時間換算（snap_t_last＝ROS − T0）、年齡重算（＝轉送延遲＋節點評估年齡）', ok_map and ok_t and ok_age and len(ages) > 0, (ok_map, ok_t, ok_age, ages[:3]))
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    rclpy.shutdown()
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
