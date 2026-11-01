#!/usr/bin/env python3
"""bottle_seg3d_node 有界最新影格佇列的同程序注入測試（Codex 20261101_214354 (b)；離線、獨立 ROS 網域）。
重播端用 rclpy 發 2.4 MB 影像每幀約 0.35 s，無法製造連發壓力；這裡改在同一程序內直接呼叫節點的訊息入口 _in()：
先用某趟真實影格預建 M 組訊息（rgb／depth／camera_info／cam_pose），再以 DT（牆鐘）間隔、遞增戳連續注入 N 組，工作執行緒照常做真實推論與三維定位。
核對：
  1 帳目：發布數＋最新一格槽丟棄數（n_drop_latest）＋配對丟棄數（n_drop_pair）＝注入數（結束時槽與配對表都已清空）；
  2 不回頭：發布戳嚴格遞增、n_stale＝0；
  3 最後注入的影格一定被處理並發布（槽只丟較舊的）；
  4 亂序注入：另注入一組戳比已發布者舊的完整影格 ⇒ 不得發布；
  5 晚配齊的舊格（Codex 20261101_220915 第 2 項）：A 處理中、新格 C 已配齊入槽、舊格 B 才配齊 ⇒ B 不入槽（n_old_complete 增加）、C 必須發布。
    source /opt/ros/jazzy/setup.bash; ROS_DOMAIN_ID=91 .venv-dl0/bin/python evaluation/bottle_seg3d_slot_test.py --run bottle_w1_compact3_nominal1 --out OUT.json
"""
import argparse
import array
import copy
import json
import math
import os
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64MultiArray

import bottle_seg3d_node as SN
import bottle_seg3d_replay_check as RC

HERE = os.path.dirname(os.path.abspath(__file__))


def stamp(t):
    s = int(math.floor(t))
    return Time(sec=s, nanosec=int(round((t - s) * 1e9)) % 1000000000)


def build(F, i, opt):
    H_, W_ = F['depth'][i].shape
    rgb = Image(height=H_, width=W_, encoding='rgb8', step=3 * W_)
    rgb.header.frame_id, rgb.data = opt, array.array('B', F['rgb'][i].tobytes())
    dep = Image(height=H_, width=W_, encoding='32FC1', step=4 * W_)
    dep.header.frame_id, dep.data = opt, array.array('B', F['depth'][i].astype(np.float32).tobytes())
    ci = CameraInfo(height=H_, width=W_)
    ci.header.frame_id, ci.k = opt, [float(v) for v in F['K'].ravel()]
    ps = PoseStamped()
    ps.header.frame_id = 'odom'
    T = F['T_wc'][i]
    w, x, y, z = RC.R_to_wxyz(T[:3, :3])
    ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = w, x, y, z
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in T[:3, 3])
    return dict(rgb=rgb, depth=dep, info=ci, pose=ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--ns', default='w1')
    ap.add_argument('--weights', default=os.path.join(HERE, 'dl0_ckpt', 'bottle_seg_v1_continue3', 'best.pth'))
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--m', type=int, default=8)
    ap.add_argument('--dt', type=float, default=0.01)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    F = np.load(os.path.join(HERE, 'runs', a.run, f'{a.ns}_frames.npz'))
    idx = np.linspace(0, len(F['stamp']) - 1, a.m).astype(int)
    sets = [build(F, int(i), RC.OPT[a.ns]) for i in idx]
    rclpy.init()
    node = SN.Seg3dNode(a.weights, a.ns)
    rec = Node('slot_rec')
    out = []
    rec.create_subscription(Float64MultiArray, f'/{a.ns}/bottle_seg3d', lambda m: out.append(list(m.data)), 500)
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    ex.add_node(rec)
    threading.Thread(target=ex.spin, daemon=True).start()
    time.sleep(1.0)
    injected, latest_at = [], []
    t0 = 100.0
    for k in range(a.n):
        st_ = round(t0 + 0.05 * k, 6)
        e = sets[k % a.m]
        for key in ('pose', 'info', 'depth', 'rgb'):
            msg = copy.copy(e[key])
            msg.header = copy.copy(e[key].header)
            msg.header.stamp = stamp(st_)
            node._in(key, msg)
        injected.append(st_)
        latest_at.append((time.perf_counter(), st_))
        time.sleep(a.dt)
    while node.slot is not None or node.ev.is_set():
        time.sleep(0.05)
    time.sleep(1.5)
    n_pub_before_old = len(out)
    old_st = round(t0 + 0.05 * (a.n // 2) + 0.001, 6)                   # 亂序：戳比已發布的最新者舊
    e = sets[0]
    for key in ('pose', 'info', 'depth', 'rgb'):
        msg = copy.copy(e[key])
        msg.header = copy.copy(e[key].header)
        msg.header.stamp = stamp(old_st)
        node._in(key, msg)
    time.sleep(1.5)
    # 5 晚配齊的舊格：處理中 A（戳 t1）→ 新格 C（t1＋0.2）配齊入槽 → 舊格 B（t1＋0.1）才配齊 ⇒ B 不得入槽、C 必須發布
    n_before5 = len(out)
    t1 = round(t0 + 0.05 * a.n + 10.0, 6)
    def inject(st_, e):
        for key in ('pose', 'info', 'depth', 'rgb'):
            msg = copy.copy(e[key])
            msg.header = copy.copy(e[key].header)
            msg.header.stamp = stamp(st_)
            node._in(key, msg)
    inject(t1, sets[1])
    w_ = time.perf_counter()
    while node.inflight is None and time.perf_counter() - w_ < 2.0:   # 等 A 開始處理
        time.sleep(0.001)
    inflight_seen = node.inflight
    e_b = sets[2]
    for key in ('pose', 'info', 'depth'):                             # B 先到三則（未配齊）
        msg = copy.copy(e_b[key]); msg.header = copy.copy(e_b[key].header); msg.header.stamp = stamp(t1 + 0.1); node._in(key, msg)
    inject(round(t1 + 0.2, 6), sets[3])                               # C 配齊入槽
    msg = copy.copy(e_b['rgb']); msg.header = copy.copy(e_b['rgb'].header); msg.header.stamp = stamp(t1 + 0.1); node._in('rgb', msg)   # B 才配齊
    old5 = node.n['old_complete']
    time.sleep(1.5)
    out5 = [d[0] for d in out[n_before5:]]
    case5 = dict(inflight_seen=inflight_seen, published=out5, B_published=any(abs(x - (t1 + 0.1)) < 1e-6 for x in out5), C_published=any(abs(x - (t1 + 0.2)) < 1e-6 for x in out5),
                 n_old_complete=old5)
    rows = [dict(zip(SN.FIELDS, d)) for d in out]
    st_out = [r['stamp'] for r in rows]
    mono = all(b > a_ for a_, b in zip(st_out, st_out[1:]))
    last = rows[-1] if rows else {}
    n_main = n_pub_before_old
    acc = n_main + int(node.n['drop_latest']) + int(node.n['drop_pair'])
    res = dict(run=a.run, n_injected=a.n, inject_dt_s=a.dt, n_published=n_main, n_drop_latest=node.n['drop_latest'], n_drop_pair=node.n['drop_pair'],
               n_stale=node.n['stale'], accounting_ok=acc == a.n, monotonic=mono, old_injected_published=any(abs(s_ - old_st) < 1e-6 for s_ in st_out),
               last_injected_published=any(abs(s_ - injected[-1]) < 1e-6 for s_ in st_out),
               proc_ms=dict(median=round(float(np.median([r['proc_ms'] for r in rows])), 1), max=round(float(np.max([r['proc_ms'] for r in rows])), 1)) if rows else None,
               queue_ms_max=round(float(np.max([r['queue_ms'] for r in rows])), 2) if rows else None,
               late_old_frame=case5, late_old_ok=bool(case5['C_published'] and not case5['B_published'] and case5['n_old_complete'] >= 1))
    json.dump(dict(summary=res, rows=rows), open(a.out, 'w'), indent=1, default=float)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    node.alive = False
    ex.shutdown()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
