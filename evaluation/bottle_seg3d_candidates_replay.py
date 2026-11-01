#!/usr/bin/env python3
"""候選節點的既有訊息重播核對（Codex 20261102_001039；獨立 ROS 網域、無 Isaac）：把某趟紀錄的 /w1/bottle_seg3d 原始輸出列（含 layout 標籤）照收到時刻、1 倍速重發，
同時發 /clock；收 /bottle/target_candidates，核對：欄位數＝layout 標籤、標籤含 frame=odom、t_eval 單調；每筆輸出的 ready 狀態與離線 CandidateTracker 在同一時刻的評估一致；
ik／local 狀態最後收斂（ready 版本最終出現 ok／fail，不永遠 pending）；evaluate 耗時。
    source /opt/ros/jazzy/setup.bash; ROS_DOMAIN_ID=91 .venv-dl0/bin/python evaluation/bottle_seg3d_candidates_replay.py --run seg3d_ad_z045_p1b --out OUT.json
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_seg3d_candidates as CT                    # noqa: E402


def stats_(x):
    x = np.array([v for v in x if np.isfinite(v)], float)
    return None if not len(x) else dict(median=round(float(np.median(x)), 4), min=round(float(x.min()), 4))


def stamp(t):
    ns = int(round(float(t) * 1e9))
    return Time(sec=ns // 1000000000, nanosec=ns % 1000000000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    Z = np.load(os.path.join(HERE, 'runs', a.run, 'seg3d_w1.npz'))
    label = str(Z['label'])
    fs = label.split('fields=')[1].split(',')
    rows = sorted(([float(r[0])] + [float(v) for v in r[1:]] for r in Z['rows']), key=lambda r: r[0])
    rclpy.init()
    pub_n, rec_n = Node('cand_replay_pub'), Node('cand_replay_rec')
    pc = pub_n.create_publisher(Clock, '/clock', 10)
    ps = pub_n.create_publisher(Float64MultiArray, '/w1/bottle_seg3d', 50)
    got, lab = [], {}

    def cb(m):
        if m.layout.dim:
            lab['l'] = m.layout.dim[0].label
        got.append(list(m.data))
    rec_n.create_subscription(Float64MultiArray, '/bottle/target_candidates', cb, 500)
    ex = SingleThreadedExecutor()
    ex.add_node(rec_n)
    threading.Thread(target=ex.spin, daemon=True).start()
    w0 = time.perf_counter()
    while pub_n.count_subscribers('/w1/bottle_seg3d') < 1:
        rclpy.spin_once(pub_n, timeout_sec=0.1)
        if time.perf_counter() - w0 > 60:
            raise SystemExit('候選節點未就緒')
    time.sleep(1.0)
    t0s, w0 = rows[0][0] - 0.5, time.perf_counter()
    sim = lambda: t0s + (time.perf_counter() - w0)              # noqa: E731
    k = 0
    while k < len(rows) or sim() < rows[-1][0] + 3.0:
        t = sim()
        pc.publish(Clock(clock=stamp(t)))
        while k < len(rows) and rows[k][0] <= t:
            m = Float64MultiArray()
            m.layout.dim = [MultiArrayDimension(label=label, size=len(fs), stride=len(fs))]
            m.data = rows[k][1:]
            ps.publish(m)
            k += 1
        time.sleep(0.005)
    time.sleep(2.0)
    ex.shutdown()
    L = lab.get('l', '')
    cf = L.split('fields=')[1].split(',') if 'fields=' in L else []
    out = [dict(zip(cf, g)) for g in got if len(g) == len(cf)]
    # 離線同時刻評估
    tr = CT.CandidateTracker()
    by_src = {}
    for r in rows:
        d = dict(zip(fs, r[1:]))
        d['recv_age_s'] = r[0] - d['stamp']
        tr.add(d)
        by_src[round(d['stamp'], 4)] = tr.evaluate(r[0])
    match, n_cmp = 0, 0
    for o in out:
        e = by_src.get(round(o['src_stamp'], 4))
        if e is None:
            continue
        n_cmp += 1
        match += int(bool(o['c_ready']) == (e['complete']['status'] == 'ready') and bool(o['k_ready']) == (e['known_type']['status'] == 'ready'))
    ready = [o for o in out if o['c_ready'] > 0.5 or o['k_ready'] > 0.5]
    final_states = [(o['c_ik'], o['c_local'], o['k_ik'], o['k_local']) for o in ready[-3:]]
    res = dict(run=a.run, n_sent=len(rows), n_out=len(out), label_ok=('frame=odom' in L and len(cf) > 0 and all(len(g) == len(cf) for g in got)), label=L[:80],
               monotonic=all(b['t_eval'] >= a_['t_eval'] for a_, b in zip(out, out[1:])), ready_match=f'{match}/{n_cmp}',
               ever_ok=dict(c_ik=any(o['c_ik'] == 2 for o in out), c_local=any(o['c_local'] == 2 for o in out), k_ik=any(o['k_ik'] == 2 for o in out), k_local=any(o['k_local'] == 2 for o in out)),
               pending_frac_ready=round(float(np.mean([o['k_ik'] == 1 for o in ready])), 2) if ready else None, last_ready_states=final_states,
               last_checked=dict(k_last_ik_ok=round(float(np.mean([o['k_last_ik'] == 2 for o in ready])), 2) if ready else None,
                                 k_last_local_ok=round(float(np.mean([o['k_last_local'] == 2 for o in ready])), 2) if ready else None,
                                 k_last_local_min=stats_(o['k_last_local_min'] for o in ready)),
               eval_ms=dict(median=round(float(np.median([o['eval_ms'] for o in out])), 3), max=round(float(np.max([o['eval_ms'] for o in out])), 3)) if out else None,
               n_reach_run=out[-1]['n_reach_run'] if out else None, n_reach_replaced=out[-1]['n_reach_replaced'] if out else None,
               delta_mm=dict(median=round(float(np.nanmedian([o['k_delta_mm'] for o in ready])), 3), max=round(float(np.nanmax([o['k_delta_mm'] for o in ready])), 3)) if ready else None)
    json.dump(dict(summary=res, out=out), open(a.out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
