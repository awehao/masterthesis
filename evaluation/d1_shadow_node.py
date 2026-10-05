#!/usr/bin/env python3
"""D1 S4 線上 shadow 節點：以**凍結的** d1_handle_detect.detect() 處理腕部深度，只發布觀測，不接控制。

計畫 evaluation/results/vision/D1_S4_online_plan.md；Codex reviews/20261005_144036_reply.md。

資料路徑（全部有界）：
  ROS 接收（每話題 KEEP_LAST 深度 2）→ PairBuffer（同擷取戳四段：depth／camera_info／pose／capture_meta；
  容量 3 組、逾時 1.0 s 牆鐘）→ check_contract（32FC1 m、optical frame、odom 位姿）→ LatestSlot（待處理最多一張，
  新影格覆蓋舊待處理影格）→ 工作執行緒 detect() → 發布 /d1/handle_obs（JSON；擷取戳不改寫）。
不訂閱也不發布任何控制話題。輸出年齡：模擬年齡＝輸出當下 /clock − 擷取戳；牆鐘年齡＝輸出牆鐘 − 來源擷取匹配牆鐘
（capture_meta.source_wall_t，同一台機器的 time.time()）。
紀錄 <out>/d1_shadow.jsonl：每個事件（received_part、paired、evicted、contract_reject、overwritten、processed、published、
shutdown_unprocessed）附影格序號 n 與擷取戳；處理順序、隨機種子（0）另記，供離線重現。
正常結束與 TERM 都有界收尾（工作執行緒最多等 1.5 s），寫 <out>/d1_shadow_summary.json。

    python3 -u evaluation/d1_shadow_node.py --out runs/<RUN>/d1_shadow
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d1_handle_detect as D1                                     # noqa: E402  凍結版，不改
from d1_shadow_core import (LatestSlot, PairBuffer, check_contract,  # noqa: E402
                            stamp_key)

SEED = 0


def _js(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--pair-cap', type=int, default=3)
    ap.add_argument('--pair-timeout-s', type=float, default=1.0)
    ap.add_argument('--qos-depth', type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    log = open(os.path.join(a.out, 'd1_shadow.jsonl'), 'w')
    log_lock = threading.Lock()
    C = {k: 0 for k in ('received_depth', 'received_info', 'received_pose', 'received_meta', 'paired',
                        'evicted', 'contract_reject', 'overwritten', 'processed', 'published',
                        'detect_reject', 'L2', 'L1', 'L0')}

    def ev(kind, **kw):
        rec = {'ev': kind, 'wall_t': time.time(), **kw}
        with log_lock:
            log.write(json.dumps(rec, ensure_ascii=False, default=_js) + '\n')
            log.flush()

    rclpy.init()
    nd = Node('d1_shadow')
    qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=a.qos_depth,
                     reliability=QoSReliabilityPolicy.RELIABLE)
    pub = nd.create_publisher(String, '/d1/handle_obs', 10)
    pb = PairBuffer(cap=a.pair_cap, timeout_s=a.pair_timeout_s)
    pb_lock = threading.Lock()
    slot = LatestSlot()
    sim_now = {'t': None}
    stop = threading.Event()
    rng = np.random.default_rng(SEED)
    order = []

    def key_of(h):
        return stamp_key(h.stamp.sec, h.stamp.nanosec)

    def on_part(part, key, msg):
        C['received_' + part] += 1
        now = time.time()
        with pb_lock:
            done, evicted = pb.put(key, part, msg, now)
        for k, why, parts in evicted:
            C['evicted'] += 1
            ev('evicted', stamp=list(k), why=why, parts=parts)
        if done is None:
            return
        C['paired'] += 1
        meta = done['meta']
        dep, K, T, why = check_contract(done['depth'], done['info'], done['pose'])
        if why is not None:
            C['contract_reject'] += 1
            ev('contract_reject', n=meta.get('n'), stamp=list(key), why=why)
            return
        item = {'n': meta.get('n'), 'stamp': list(key), 't_cap': key[0] + key[1] * 1e-9,
                'src_wall': meta.get('source_wall_t'), 'recv_wall': now, 'dep': dep, 'K': K, 'T': T}
        ev('paired', n=item['n'], stamp=item['stamp'], recv_minus_src_wall_s=(
            None if item['src_wall'] is None else now - item['src_wall']))
        old = slot.put(item)
        if old is not None:
            C['overwritten'] += 1
            ev('overwritten', n=old['n'], stamp=old['stamp'], by_n=item['n'])

    def depth_cb(m):
        on_part('depth', key_of(m.header), {'encoding': m.encoding, 'frame_id': m.header.frame_id,
                                            'width': m.width, 'height': m.height,
                                            'is_bigendian': m.is_bigendian, 'step': m.step,
                                            'data': bytes(m.data)})

    def info_cb(m):
        on_part('info', key_of(m.header), {'frame_id': m.header.frame_id, 'width': m.width,
                                           'height': m.height, 'k': list(m.k)})

    def pose_cb(m):
        p, o = m.pose.position, m.pose.orientation
        on_part('pose', key_of(m.header), {'frame_id': m.header.frame_id, 'pos': [p.x, p.y, p.z],
                                           'quat_wxyz': [o.w, o.x, o.y, o.z]})

    def meta_cb(m):
        try:
            d = json.loads(m.data)
            k = tuple(d['stamp'])
        except Exception as e:                                       # noqa: BLE001
            ev('bad_meta', why=repr(e))
            return
        on_part('meta', stamp_key(*k), d)

    def clock_cb(m):
        sim_now['t'] = m.clock.sec + m.clock.nanosec * 1e-9

    nd.create_subscription(Image, '/wrist/aligned_depth_to_color/image_raw', depth_cb, qos)
    nd.create_subscription(CameraInfo, '/wrist/color/camera_info', info_cb, qos)
    nd.create_subscription(PoseStamped, '/wrist/pose', pose_cb, qos)
    nd.create_subscription(String, '/wrist/capture_meta', meta_cb, qos)
    nd.create_subscription(Clock, '/clock', clock_cb, 10)

    def worker():
        while not stop.is_set():
            it = slot.take(timeout=0.1)
            if it is None:
                continue
            w0 = time.time()
            det, ms = D1.detect(it['dep'], it['K'], it['T'], rng)
            order.append(it['n'])
            C['processed'] += 1
            lvl = 'L2' if det['L2'] else ('L1' if det['L1'] else ('L0' if det['L0'] else None))
            if lvl:
                C[lvl] += 1
            else:
                C['detect_reject'] += 1
            w1 = time.time()
            sim_age = None if sim_now['t'] is None else sim_now['t'] - it['t_cap']
            wall_age = None if it['src_wall'] is None else w1 - it['src_wall']
            obs = {'n': it['n'], 'stamp': it['stamp'], 'level': lvl, 'reject': det['reject'],
                   'reject_L2': det['reject_L2'], 'L1': det['L1'], 'L2': det['L2'],
                   'detect_ms': round(ms, 2), 'sim_age_s': sim_age, 'wall_age_s': wall_age}
            pub.publish(String(data=json.dumps(obs, ensure_ascii=False, default=_js)))
            C['published'] += 1
            ev('processed', **obs, proc_index=len(order) - 1, wait_wall_s=w0 - it['recv_wall'],
               det=det, T=it['T'].tolist())

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    print('[d1_shadow] 上線：凍結 detect()、待處理槽 1、配對容量 %d／逾時 %.1f s' % (a.pair_cap, a.pair_timeout_s),
          flush=True)

    def finish(why):
        stop.set()
        th.join(timeout=1.5)
        left = slot.drain()
        if left is not None:
            ev('shutdown_unprocessed', n=left['n'], stamp=left['stamp'])
        with pb_lock:
            un = pb.drain()
        for k, w, parts in un:
            ev('evicted', stamp=list(k), why=w, parts=parts)
        summ = {'why': why, 'seed': SEED, 'counters': C, 'worker_alive_at_exit': th.is_alive(),
                'unprocessed_in_slot': 0 if left is None else 1, 'unpaired_at_exit': len(un),
                'processing_order_n': order,
                'detector_sha256': __import__('hashlib').sha256(open(D1.__file__, 'rb').read()).hexdigest(),
                'params': D1.P}
        json.dump(summ, open(os.path.join(a.out, 'd1_shadow_summary.json'), 'w'), ensure_ascii=False,
                  indent=1, default=_js)
        log.close()
        print(f'[d1_shadow] 收尾（{why}）：{json.dumps(C, ensure_ascii=False)}', flush=True)

    got = {'sig': None}

    def on_sig(s, _f):
        got['sig'] = signal.Signals(s).name
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    try:
        while rclpy.ok() and got['sig'] is None:
            rclpy.spin_once(nd, timeout_sec=0.05)
            with pb_lock:
                ex = pb.expire(time.time())
            for k, why, parts in ex:
                C['evicted'] += 1
                ev('evicted', stamp=list(k), why=why, parts=parts)
    finally:
        finish(got['sig'] or 'rclpy_shutdown')
        try:
            nd.destroy_node()
            rclpy.shutdown()
        except Exception:                                            # noqa: BLE001
            pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
