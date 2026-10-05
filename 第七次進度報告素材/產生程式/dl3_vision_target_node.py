#!/usr/bin/env python3
"""DL3 幾何視覺目標節點：S4 shadow 節點（d1_shadow_node.py，凍結、不改）的複本，每格在凍結 D1 偵測之後
再跑 DL2 估計端（N-obs；失敗不回退 N-prior），發布 /dl3/handle_est（JSON）。**不讀真值、不發控制命令**；
任務節點只在 ALIGN 進入時依 dl3_latch 規則鎖定一次。規格 results/vision/DL3_vision_pregrasp_spec.md draft-2。

與 S4 節點的差異只在：(1) 每格處理改呼叫 process_frame()（可離線測試）；(2) 發布話題 /dl3/handle_est；
(3) 紀錄檔名 dl3_target.jsonl／dl3_target_summary.json；(4) 摘要附 DL2／DL1 程式雜湊與基準抓取參數。
以下沿用 S4 說明：

資料路徑（全部有界）：
  ROS 接收（每話題 KEEP_LAST 深度 2）→ PairBuffer（同擷取戳四段：depth／camera_info／pose／capture_meta；
  容量 3 組、逾時 1.0 s 牆鐘）→ check_contract（32FC1 m、optical frame、odom 位姿）→ LatestSlot（待處理最多一張，
  新影格覆蓋舊待處理影格）→ 工作執行緒 detect() → 發布 /d1/handle_obs（JSON；擷取戳不改寫）。
不訂閱也不發布任何控制話題。輸出年齡：模擬年齡＝輸出當下 /clock − 擷取戳；牆鐘年齡＝輸出牆鐘 − 來源擷取匹配牆鐘
（capture_meta.source_wall_t，同一台機器的 time.time()）。
紀錄 <out>/d1_shadow.jsonl：每個事件（received_part、paired、evicted、contract_reject、overwritten、processed、published、
shutdown_unprocessed）附影格序號 n 與擷取戳；處理順序、隨機種子（0）另記，供離線重現。
正常結束與 TERM 都有界收尾（工作執行緒最多等 1.5 s），寫 <out>/d1_shadow_summary.json。

    python3 -u evaluation/dl3_vision_target_node.py --out runs/<RUN>/dl3_target
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
import dl2_obs_to_target as E                                     # noqa: E402  DL2 估計端（凍結）

SEED = 0


def _js(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def baseline_reference():
    """設計抓取工具參考旋轉與基準抓取參數（與 DL2 相同來源：phf_01_M 停車位姿＋q_grasp 的 link_tcp FK）。"""
    sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    q = [-0.136412, 0.560, 1.297349, -0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813]
    Rg = K.fk(np.array(q, float), 'link_tcp')[:3, :3].copy()
    T_HG, a_H = E.baseline_grasp(Rg)
    return Rg, T_HG, a_H


def process_frame(dep, K, T, t_cap, n, rng, Rg, T_HG, a_H):
    """凍結 D1 偵測 → DL2 估計（N-obs）。回傳 (det, ms, est_msg)。不讀真值。"""
    det, ms = D1.detect(dep, K, T, rng)
    obs = {'path': 'G0', 'L1': det['L1'], 'L2': det['L2'], 'reject': det['reject'], 'reject_L2': det['reject_L2'],
           'depth': dep, 'K': K, 'T_cam': T, 't_obs': t_cap}
    r = E.estimate(obs, 'N-obs', Rg, T_HG, a_H, query_t=t_cap)
    msg = {'n': n, 't_obs': t_cap, 'est_ok': r['ok'], 'est_why': r.get('why'), 'prior_used': r['prior_used'],
           'geometry_contract': r['geometry_contract'], 'freshness': r.get('freshness'),
           'normal_diag': r.get('normal_diag')}
    if r['ok']:
        msg.update({'T_WH': np.asarray(r['T_WH']).tolist(), 'normal_W': r['normal_W'],
                    'T_WE_s': np.asarray(r['T_WE_s']).tolist(), 'candidate_index': r['candidate_index']})
    return det, ms, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--pair-cap', type=int, default=3)
    ap.add_argument('--pair-timeout-s', type=float, default=1.0)
    ap.add_argument('--qos-depth', type=int, default=2)
    ap.add_argument('--fault-inject-n', type=int, nargs='*', default=[],
                    help='僅測試用：處理這些影格序號時令偵測丟例外（驗證 worker_error 記錄）；預設不注入')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    log = open(os.path.join(a.out, 'dl3_target.jsonl'), 'w')
    Rg, T_HG, a_H = baseline_reference()
    log_lock = threading.Lock()
    C = {k: 0 for k in ('received_depth', 'received_info', 'received_pose', 'received_meta', 'paired',
                        'evicted', 'contract_reject', 'overwritten', 'processed', 'published',
                        'detect_reject', 'L2', 'L1', 'L0', 'worker_error')}

    def ev(kind, **kw):
        rec = {'ev': kind, 'wall_t': time.time(), **kw}
        with log_lock:
            log.write(json.dumps(rec, ensure_ascii=False, default=_js) + '\n')
            log.flush()

    rclpy.init()
    nd = Node('dl3_vision_target')
    qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=a.qos_depth,
                     reliability=QoSReliabilityPolicy.RELIABLE)
    pub = nd.create_publisher(String, '/dl3/handle_est', 10)
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
        ev('received', stamp=list(key), part=part, n=(msg.get('n') if part == 'meta' else None))
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

    inflight = {'item': None}
    fault = set(a.fault_inject_n or [])

    def worker():
        while not stop.is_set():
            it = slot.take(timeout=0.1)
            if it is None:
                continue
            inflight['item'] = it
            stage = 'detect'
            try:
                w0 = time.time()
                if it['n'] in fault:
                    raise RuntimeError(f'fault-inject n={it["n"]}')
                det, ms, est = process_frame(it['dep'], it['K'], it['T'], it['t_cap'], it['n'], rng, Rg, T_HG, a_H)
                order.append(it['n'])
                lvl = 'L2' if det['L2'] else ('L1' if det['L1'] else ('L0' if det['L0'] else None))
                w1 = time.time()
                out_sim = sim_now['t']
                sim_age = None if out_sim is None else out_sim - it['t_cap']
                # 牆鐘年齡＝模擬器「讀取並匹配影格」之後 → 本節點輸出；**不含**此前的算圖延遲
                wall_age = None if it['src_wall'] is None else w1 - it['src_wall']
                obs = {'n': it['n'], 'stamp': it['stamp'], 'level': lvl, 'reject': det['reject'],
                       'reject_L2': det['reject_L2'], 'L1': det['L1'], 'L2': det['L2'],
                       'detect_ms': round(ms, 2), 'out_sim_t': out_sim, 'sim_age_s': sim_age,
                       'wall_age_s': wall_age, **est}
                stage = 'publish'
                pub.publish(String(data=json.dumps(obs, ensure_ascii=False, default=_js)))
            except Exception as e:                                   # noqa: BLE001
                C['worker_error'] += 1
                ev('worker_error', n=it['n'], stamp=it['stamp'], stage=stage, why=repr(e))
                inflight['item'] = None
                continue
            C['processed'] += 1
            if obs.get('est_ok'):
                C['est_ok'] = C.get('est_ok', 0) + 1
            C['published'] += 1
            if lvl:
                C[lvl] += 1
            else:
                C['detect_reject'] += 1
            ev('processed', **obs, proc_index=len(order) - 1, wait_wall_s=w0 - it['recv_wall'],
               det=det, T=it['T'].tolist())
            inflight['item'] = None

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    print('[dl3_target] 上線：凍結 detect()、待處理槽 1、配對容量 %d／逾時 %.1f s' % (a.pair_cap, a.pair_timeout_s),
          flush=True)

    def finish(why):
        stop.set()
        th.join(timeout=1.5)
        busy = inflight['item'] if th.is_alive() else None
        if busy is not None:
            ev('shutdown_in_progress', n=busy['n'], stamp=busy['stamp'])
        left = slot.drain()
        if left is not None:
            ev('shutdown_unprocessed', n=left['n'], stamp=left['stamp'])
        with pb_lock:
            un = pb.drain()
        for k, w, parts in un:
            ev('evicted', stamp=list(k), why=w, parts=parts)
        summ = {'why': why, 'seed': SEED, 'counters': C, 'worker_alive_at_exit': th.is_alive(),
                'unprocessed_in_slot': 0 if left is None else 1, 'unpaired_at_exit': len(un),
                'in_progress_at_exit_n': None if busy is None else busy['n'],
                'fault_inject_n': sorted(fault),
                'wall_age_definition': '模擬器讀取匹配影格之後 → 節點輸出（不含算圖延遲）',
                'processing_order_n': order,
                'detector_sha256': __import__('hashlib').sha256(open(D1.__file__, 'rb').read()).hexdigest(),
                'params': D1.P}
        summ['dl2_estimator_sha256'] = __import__('hashlib').sha256(open(E.__file__, 'rb').read()).hexdigest()
        summ['T_HG'] = T_HG.tolist()
        summ['a_H'] = a_H.tolist()
        summ['R_ref_tool'] = Rg.tolist()
        summ['n_est_ok'] = C.get('est_ok', 0)
        json.dump(summ, open(os.path.join(a.out, 'dl3_target_summary.json'), 'w'), ensure_ascii=False,
                  indent=1, default=_js)
        log.close()
        print(f'[dl3_target] 收尾（{why}）：{json.dumps(C, ensure_ascii=False)}', flush=True)

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
