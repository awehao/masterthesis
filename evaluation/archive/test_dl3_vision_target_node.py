#!/usr/bin/env python3
"""DL3 幾何視覺目標節點測試（ROS，不起模擬器）：沿用 S4 節點故障注入測試的合成來源（test_d1_shadow_node_fault.py）。

以 --fault-inject-n 1 起節點：第 1 格丟例外 ⇒ worker_error 記錄、工作執行緒存活、第 2 格照常處理並發布 /dl3/handle_est；
另檢查發布訊息含 DL2 估計欄位（est_ok、geometry_contract unchecked、t_obs＝擷取戳），以及 process_frame 的離線一致性。

    ROS_DOMAIN_ID=97 python3 evaluation/test_dl3_vision_target_node.py
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


def offline():
    import dl2_obs_to_target as E
    import dl3_vision_target_node as N
    Rg, T_HG, a_H = N.baseline_reference()
    V = os.path.join(HERE, 'runs', 'd1_dev_f02P', 'wrist_v0')
    m = json.load(open(os.path.join(V, 'meta.json')))
    K = np.array(m['intrinsics_readback']['K'])
    import dl0_autolabel as AL
    fr = {f['n']: f for f in AL.frames_of('d1_dev_f02P/wrist_v0')}[35]
    dep = AL.load_depth(fr['depth_path'])
    a = N.process_frame(dep, K, fr['T'], 3.0, 35, np.random.default_rng(0), Rg, T_HG, a_H)
    b = N.process_frame(dep, K, fr['T'], 3.0, 35, np.random.default_rng(0), Rg, T_HG, a_H)
    check('process_frame_deterministic', json.dumps(a[2], default=str) == json.dumps(b[2], default=str))
    det = a[0]
    r = E.estimate({'path': 'G0', 'L1': det['L1'], 'L2': det['L2'], 'reject': det['reject'], 'reject_L2': det['reject_L2'],
                    'depth': dep, 'K': K, 'T_cam': fr['T'], 't_obs': 3.0}, 'N-obs', Rg, T_HG, a_H, query_t=3.0)
    check('process_frame_equals_dl2_estimate', r['ok'] == a[2]['est_ok'] and (not r['ok'] or np.abs(np.array(a[2]['T_WH']) - r['T_WH']).max() == 0.0))
    check('process_frame_no_truth_keys', not any('handle_center' in k or 'truth' in k for k in a[2]))
    print('  frame 35 est', a[2]['est_ok'], a[2]['est_why'])


offline()


def main():
    import rclpy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    RUN = tempfile.mkdtemp(prefix='dl3target_fault_')
    out = os.path.join(RUN, 'dl3_target')
    os.makedirs(out)
    pr = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'dl3_vision_target_node.py'), '--out', out,
                           '--fault-inject-n', '1'],
                          stdout=open(os.path.join(out, 'node.log'), 'w'), stderr=subprocess.STDOUT)
    rclpy.init()
    nd = Node('dl3_target_fault_src')
    got = []
    nd.create_subscription(String, '/dl3/handle_est', lambda m: got.append(json.loads(m.data)), 10)
    qos = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=2, reliability=QoSReliabilityPolicy.RELIABLE)
    P = {'depth': nd.create_publisher(Image, '/wrist/aligned_depth_to_color/image_raw', qos),
         'info': nd.create_publisher(CameraInfo, '/wrist/color/camera_info', qos),
         'pose': nd.create_publisher(PoseStamped, '/wrist/pose', qos),
         'meta': nd.create_publisher(String, '/wrist/capture_meta', qos)}
    t0 = time.time()
    while time.time() - t0 < 20 and not all(p.get_subscription_count() > 0 for p in P.values()):
        rclpy.spin_once(nd, timeout_sec=0.1)
    check('node_subscribed', all(p.get_subscription_count() > 0 for p in P.values()))

    V = os.path.join(HERE, 'runs', 'd1_dev_f02P', 'wrist_v0')
    m = json.load(open(os.path.join(V, 'meta.json')))
    K = np.array(m['intrinsics_readback']['K'])
    f = next(x for x in m['frames'] if x['n'] == 35)
    d0 = np.load(os.path.join(V, 'frames', 'f35_depth_m.npy'))
    d0 = d0[:, :, 0] if d0.ndim == 3 else d0
    sent = {}
    for n, t in ((1, 10.0), (2, 10.2)):
        ns = int(round(t * 1e9))
        st = Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
        im = Image()
        im.header.stamp, im.header.frame_id = st, 'camera_color_optical_frame'
        im.height, im.width, im.is_bigendian, im.encoding, im.step = 480, 640, 0, '32FC1', 640 * 4
        im.data = d0.astype('<f4').tobytes()
        ci = CameraInfo()
        ci.header.stamp, ci.header.frame_id = st, 'camera_color_optical_frame'
        ci.height, ci.width, ci.k = 480, 640, [float(v) for v in K.reshape(-1)]
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = st, 'odom'
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in f['cam_pos_world'])
        ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = \
            (float(v) for v in f['cam_quat_wxyz_world'])
        P['depth'].publish(im)
        P['info'].publish(ci)
        P['pose'].publish(ps)
        P['meta'].publish(String(data=json.dumps({'n': n, 'stamp': [st.sec, st.nanosec], 'rendering_frame': [n, 1],
                                                  'source_wall_t': time.time()})))
        sent[n] = ([st.sec, st.nanosec], t)
        t1 = time.time() + 1.5                           # 等處理完再送下一格（不製造覆蓋）
        while time.time() < t1:
            rclpy.spin_once(nd, timeout_sec=0.05)
    t1 = time.time() + 1.0
    while time.time() < t1:
        rclpy.spin_once(nd, timeout_sec=0.05)
    pr.send_signal(signal.SIGTERM)
    try:
        pr.wait(timeout=3.0)
        ok_exit = pr.returncode == 0
    except subprocess.TimeoutExpired:
        pr.kill()
        ok_exit = False
    check('term_bounded_exit', ok_exit)
    S = json.load(open(os.path.join(out, 'dl3_target_summary.json')))
    E = [json.loads(x) for x in open(os.path.join(out, 'dl3_target.jsonl'))]
    we = [e for e in E if e['ev'] == 'worker_error']
    check('worker_error_logged_with_identity', len(we) == 1 and we[0]['n'] == 1 and we[0]['stage'] == 'detect'
          and 'fault-inject' in we[0]['why'], we)
    check('worker_survives_and_processes_next', any(e['ev'] == 'processed' and e['n'] == 2 for e in E))
    C = S['counters']
    check('summary_counts', C['worker_error'] == 1 and C['processed'] == 1 and C['published'] == 1
          and S['worker_alive_at_exit'] is False and S['fault_inject_n'] == [1], (C, S['worker_alive_at_exit']))
    check('published_one_message_for_frame2', len(got) == 1 and got[0]['n'] == 2, [g.get('n') for g in got])
    if got:
        g = got[0]
        check('message_has_estimate_fields', all(k in g for k in ('est_ok', 'est_why', 'prior_used', 'geometry_contract', 't_obs'))
              and g['geometry_contract'] == 'unchecked' and abs(g['t_obs'] - 10.2) < 1e-9, g)
        check('message_has_no_truth', not any('handle_center' in k or 'truth' in k for k in g))
    check('summary_has_estimator_hash', len(S.get('dl2_estimator_sha256', '')) == 64 and 'T_HG' in S)
    nd.destroy_node()
    rclpy.shutdown()


main()
print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
