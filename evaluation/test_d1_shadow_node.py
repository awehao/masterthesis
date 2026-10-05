#!/usr/bin/env python3
"""D1 S4 節點整合測試（ROS，不起模擬器）：端到端配對、覆蓋對帳、逾時淘汰、契約拒絕、TERM 有界收尾。

以獨立 ROS_DOMAIN_ID 起 d1_shadow_node.py，由本測試扮演模擬器發布四段話題（S3 開發集影格），
快速連發以製造覆蓋；另送一組缺段（只有 depth ⇒ 逾時淘汰）與一組 16UC1（契約拒絕）；最後 SIGTERM。

    ROS_DOMAIN_ID=97 python3 evaluation/test_d1_shadow_node.py
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('ROS_DOMAIN_ID', '97')
fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


def main():
    import rclpy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    RUN = tempfile.mkdtemp(prefix='d1shadow_test_')
    out = os.path.join(RUN, 'd1_shadow')
    os.makedirs(out)
    pr = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'd1_shadow_node.py'), '--out', out],
                          stdout=open(os.path.join(out, 'node.log'), 'w'), stderr=subprocess.STDOUT)
    rclpy.init()
    nd = Node('d1_shadow_test_src')
    qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=2, reliability=QoSReliabilityPolicy.RELIABLE)
    p_dep = nd.create_publisher(Image, '/wrist/aligned_depth_to_color/image_raw', qos)
    p_ci = nd.create_publisher(CameraInfo, '/wrist/color/camera_info', qos)
    p_pose = nd.create_publisher(PoseStamped, '/wrist/pose', qos)
    p_meta = nd.create_publisher(String, '/wrist/capture_meta', qos)
    p_clk = nd.create_publisher(Clock, '/clock', 10)
    obs = []
    nd.create_subscription(String, '/d1/handle_obs', lambda m: obs.append(json.loads(m.data)), 10)
    # 等節點訂閱上線
    t0 = time.time()
    pubs = (p_dep, p_ci, p_pose, p_meta)
    while time.time() - t0 < 20 and not (all(p.get_subscription_count() > 0 for p in pubs)
                                         and nd.count_publishers('/d1/handle_obs') > 0):
        rclpy.spin_once(nd, timeout_sec=0.1)
    for _ in range(20):
        rclpy.spin_once(nd, timeout_sec=0.05)            # 讓 /d1/handle_obs 的訂閱也完成配對
    check('node_subscribed', all(p.get_subscription_count() > 0 for p in pubs))

    V = os.path.join(HERE, 'runs', 'd1_dev_f02P', 'wrist_v0')
    m = json.load(open(os.path.join(V, 'meta.json')))
    K = np.array(m['intrinsics_readback']['K'])
    frames = [f for f in m['frames'] if f['n'] in range(30, 45)]   # 近距（處理較久，連發易覆蓋）

    def stamp(t):
        ns = int(round(t * 1e9))
        return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)

    def pub(n, t, dep, pos, q, enc='32FC1', parts=('depth', 'info', 'pose', 'meta')):
        st = stamp(t)
        if 'depth' in parts:
            im = Image()
            im.header.stamp, im.header.frame_id = st, 'camera_color_optical_frame'
            im.height, im.width, im.is_bigendian = 480, 640, 0
            im.encoding = enc
            if enc == '32FC1':
                im.step, im.data = 640 * 4, dep.astype('<f4').tobytes()
            else:
                im.step, im.data = 640 * 2, np.zeros((480, 640), np.uint16).tobytes()
            p_dep.publish(im)
        if 'info' in parts:
            ci = CameraInfo()
            ci.header.stamp, ci.header.frame_id = st, 'camera_color_optical_frame'
            ci.height, ci.width = 480, 640
            ci.k = [float(v) for v in K.reshape(-1)]
            p_ci.publish(ci)
        if 'pose' in parts:
            ps = PoseStamped()
            ps.header.stamp, ps.header.frame_id = st, 'odom'
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in pos)
            ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = \
                (float(v) for v in q)
            p_pose.publish(ps)
        if 'meta' in parts:
            p_meta.publish(String(data=json.dumps({'n': n, 'stamp': [st.sec, st.nanosec],
                                                   'rendering_frame': [n, 1], 'source_wall_t': time.time()})))
        return [st.sec, st.nanosec]

    sent = {}
    srcinfo = {}                                     # n → (t, 影格 meta)：合成來源端紀錄供稽核
    t_sim = 100.0
    for i, f in enumerate(frames):
        d0 = np.load(os.path.join(V, 'frames', f'f{f["n"]:02d}_depth_m.npy'))
        d0 = d0[:, :, 0] if d0.ndim == 3 else d0
        t_sim += 0.2
        clk = Clock()
        clk.clock = stamp(t_sim)
        p_clk.publish(clk)
        sent[i + 1] = pub(i + 1, t_sim, d0, f['cam_pos_world'], f['cam_quat_wxyz_world'])
        srcinfo[i + 1] = (t_sim, f)
        for _ in range(3):
            rclpy.spin_once(nd, timeout_sec=0.005)       # 連發（約 15–20 ms 一組）⇒ 處理中被覆蓋
    # 缺段（只有 depth）⇒ 逾時淘汰
    t_sim += 0.2
    s901 = pub(901, t_sim, d0, frames[0]['cam_pos_world'], frames[0]['cam_quat_wxyz_world'], parts=('depth',))
    srcinfo[901] = (t_sim, frames[0])
    # 契約違反（16UC1）
    t_sim += 0.2
    s902 = pub(902, t_sim, d0, frames[0]['cam_pos_world'], frames[0]['cam_quat_wxyz_world'], enc='16UC1')
    srcinfo[902] = (t_sim, frames[0])
    t_end = time.time() + 3.0
    while time.time() < t_end:
        rclpy.spin_once(nd, timeout_sec=0.05)
    # 最後再連發兩組後立刻 TERM ⇒ 收尾時可能有未處理影格
    for j in (991, 992):
        t_sim += 0.2
        sent[j] = pub(j, t_sim, d0, frames[0]['cam_pos_world'], frames[0]['cam_quat_wxyz_world'])
        srcinfo[j] = (t_sim, frames[0])
        rclpy.spin_once(nd, timeout_sec=0.01)
    time.sleep(0.05)
    tk = time.time()
    pr.send_signal(signal.SIGTERM)
    try:
        pr.wait(timeout=3.0)
        dt_exit = time.time() - tk
    except subprocess.TimeoutExpired:
        pr.kill()
        dt_exit = None
    t_end = time.time() + 1.0
    while time.time() < t_end:
        rclpy.spin_once(nd, timeout_sec=0.05)            # 收 TERM 前已發布、尚在傳輸中的觀測
    check('term_bounded_exit_within_3s', dt_exit is not None and pr.returncode == 0, (dt_exit, pr.returncode))
    sp = os.path.join(out, 'd1_shadow_summary.json')
    check('summary_written', os.path.exists(sp))
    if not os.path.exists(sp):
        print(open(os.path.join(out, 'node.log')).read()[-2000:])
        return
    S = json.load(open(sp))
    C = S['counters']
    E = [json.loads(x) for x in open(os.path.join(out, 'd1_shadow.jsonl'))]
    print('counters', C, 'unprocessed', S['unprocessed_in_slot'], 'unpaired', S['unpaired_at_exit'])
    # 連發時 ROS 接收佇列（KEEP_LAST 2）可能在節點看到前就丟棄：以來源端影格為準對帳，不要求全部配對
    check('paired_plus_drops_bounded', C['paired'] <= len(sent) + 1, C['paired'])
    check('contract_reject_16UC1', C['contract_reject'] == 1
          and any(e['ev'] == 'contract_reject' and e['why'] == 'depth_encoding_16UC1' for e in E))
    check('timeout_eviction_of_partial', any(e['ev'] == 'evicted' and e['why'] == 'pair_timeout'
                                             and e['parts'] == ['depth'] for e in E))
    # 對帳：每個通過契約的影格必為 processed／overwritten／shutdown_unprocessed 之一，且恰好一次
    fate = {}
    by_stamp = {tuple(v): k for k, v in sent.items()}
    for e in E:
        if e['ev'] in ('processed', 'overwritten', 'shutdown_unprocessed'):
            fate.setdefault(e['n'], []).append(e['ev'])
        elif e['ev'] == 'evicted' and tuple(e['stamp']) in by_stamp:
            fate.setdefault(by_stamp[tuple(e['stamp'])], []).append('evicted:' + e['why'])
    for k in sent:
        fate.setdefault(k, []).append('transport_drop') if k not in fate else None
    drops = sorted(k for k, v in fate.items() if v == ['transport_drop'])
    print('transport_drop', drops)
    check('per_frame_fate_exactly_once', set(fate) == set(sent) and all(len(v) == 1 for v in fate.values()),
          {k: v for k, v in fate.items() if len(v) != 1})
    n_recv_full = sum(1 for v in fate.values() if v[0] in ('processed', 'overwritten', 'shutdown_unprocessed'))
    check('node_counters_close', n_recv_full + C['contract_reject'] == C['paired'],
          (n_recv_full, C['contract_reject'], C['paired']))
    check('overwrites_occurred', C['overwritten'] > 0, C['overwritten'])
    check('published_equals_processed', C['published'] == C['processed'])
    check('obs_topic_received_all_published', len(obs) == C['published'], (len(obs), C['published']))
    st_ok = all(e['stamp'] == sent[e['n']] for e in E if e['ev'] == 'processed')
    check('stamps_not_rewritten', st_ok)
    order = S['processing_order_n']
    check('processing_order_monotonic', order == sorted(order), order)
    ages = [o['wall_age_s'] for o in obs if o['wall_age_s'] is not None]
    check('wall_age_from_source_positive', ages and min(ages) > 0, ages[:3])
    check('summary_has_seed_and_sha', S['seed'] == 0 and len(S['detector_sha256']) == 64)
    # ---- 稽核：合成來源端紀錄（同一批送出影格；另加 1 筆 rejected、1 筆 no_new_frame）
    os.makedirs(os.path.join(RUN, 'wrist_live'))
    allst = {**sent, 901: s901, 902: s902}
    with open(os.path.join(RUN, 'wrist_live', 'capture.jsonl'), 'w') as fc, \
            open(os.path.join(RUN, 'wrist_live', 'truth.jsonl'), 'w') as ft:
        fc.write(json.dumps({'ev': 'rejected', 'why': 'no_pose_at_render_time', 't_after': 99.0}) + '\n')
        fc.write(json.dumps({'ev': 'no_new_frame', 't_after': 99.2}) + '\n')
        for n in sorted(allst):
            t, f = srcinfo[n]
            fc.write(json.dumps({'ev': 'published', 'n': n, 'stamp': allst[n], 'rendering_time': t,
                                 'read_minus_render_s': 0.0}) + '\n')
            ft.write(json.dumps({'n': n, 'stamp': allst[n], 't_cap': t,
                                 'handle_center_world_at_capture': f['handle_center_world_at_capture']}) + '\n')
    json.dump({'intrinsics_readback_K': K.tolist(), 'render_ms': None},
              open(os.path.join(RUN, 'wrist_live', 'meta.json'), 'w'))
    json.dump({'events': [{'phase': 'ALIGN', 'sim_t': 1e6}]}, open(os.path.join(RUN, 'task.json'), 'w'))
    rc = subprocess.run([sys.executable, os.path.join(HERE, 'd1_s4_audit.py'), RUN], capture_output=True, text=True)
    check('audit_runs', rc.returncode == 0, rc.stderr[-800:])
    A = json.load(open(os.path.join(RUN, 'analysis', 'd1_s4_audit.json')))
    fc_ = A['reconciliation']['per_frame_fate_counts']
    check('audit_fates_sum_to_source_published', sum(fc_.values()) == A['source']['published'] == len(allst),
          (fc_, A['source']['published'], len(allst)))
    check('audit_no_dup_no_unknown', not A['reconciliation']['frames_with_multiple_fates']
          and A['reconciliation']['n_unknown'] == 0, A['reconciliation'])
    # 連發時深度可能在 ROS 接收佇列被丟 ⇒ 其餘三段逾時淘汰；不要求恰 1 筆，但 901 必在其中
    check('audit_partial_and_contract_attributed', fc_.get('evicted:pair_timeout', 0) >= 1
          and fc_.get('contract_reject:depth_encoding_16UC1') == 1, fc_)
    check('audit_processed_matches_node', fc_.get('processed') == C['processed'], (fc_, C['processed']))
    check('audit_source_rejects_counted', A['source']['rejected'] == 1 and A['source']['no_new_frame'] == 1)
    check('audit_verdict_pipeline_pass', A['verdict']['通路與對帳'] == 'PASS', A['verdict'])
    check('audit_observation_L2_in_window', A['observation']['approach_window']['n_L2'] >= 1
          and A['verdict']['有效觀測（接近窗口 ≥ 1 格 L2）'] == 'PASS', A['observation']['approach_window'])
    check('audit_isolation_static', A['isolation']['only_obs_topic'] is True, A['isolation'])
    nd.destroy_node()
    rclpy.shutdown()


main()
print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
