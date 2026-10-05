#!/usr/bin/env python3
"""D1 S4 節點的工作執行緒例外測試（ROS，不起模擬器；Codex reviews/20261005_145312_reply.md 第 3 項）。

以 --fault-inject-n 1 起節點：第 1 格偵測丟例外 ⇒ 必須記 worker_error（含 n、stage、原因）、工作執行緒不退出、
第 2 格照常處理；收尾摘要 worker_error = 1、worker_alive_at_exit = False；稽核判通路 FAIL 並把第 1 格歸為 worker_error。

    ROS_DOMAIN_ID=97 python3 evaluation/test_d1_shadow_node_fault.py
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
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    RUN = tempfile.mkdtemp(prefix='d1shadow_fault_')
    out = os.path.join(RUN, 'd1_shadow')
    os.makedirs(out)
    pr = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'd1_shadow_node.py'), '--out', out,
                           '--fault-inject-n', '1'],
                          stdout=open(os.path.join(out, 'node.log'), 'w'), stderr=subprocess.STDOUT)
    rclpy.init()
    nd = Node('d1_shadow_fault_src')
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
    pr.send_signal(signal.SIGTERM)
    try:
        pr.wait(timeout=3.0)
        ok_exit = pr.returncode == 0
    except subprocess.TimeoutExpired:
        pr.kill()
        ok_exit = False
    check('term_bounded_exit', ok_exit)
    S = json.load(open(os.path.join(out, 'd1_shadow_summary.json')))
    E = [json.loads(x) for x in open(os.path.join(out, 'd1_shadow.jsonl'))]
    we = [e for e in E if e['ev'] == 'worker_error']
    check('worker_error_logged_with_identity', len(we) == 1 and we[0]['n'] == 1 and we[0]['stage'] == 'detect'
          and 'fault-inject' in we[0]['why'], we)
    check('worker_survives_and_processes_next', any(e['ev'] == 'processed' and e['n'] == 2 for e in E))
    C = S['counters']
    check('summary_counts', C['worker_error'] == 1 and C['processed'] == 1 and C['published'] == 1
          and S['worker_alive_at_exit'] is False and S['fault_inject_n'] == [1], (C, S['worker_alive_at_exit']))
    # 稽核：合成來源端紀錄 ⇒ 通路必須 FAIL，第 1 格歸為 worker_error:detect
    os.makedirs(os.path.join(RUN, 'wrist_live'))
    with open(os.path.join(RUN, 'wrist_live', 'capture.jsonl'), 'w') as fc, \
            open(os.path.join(RUN, 'wrist_live', 'truth.jsonl'), 'w') as ft:
        for n, (st, t) in sent.items():
            fc.write(json.dumps({'ev': 'published', 'n': n, 'stamp': st, 'rendering_time': t,
                                 'read_minus_render_s': 0.0}) + '\n')
            ft.write(json.dumps({'n': n, 'stamp': st, 't_cap': t,
                                 'handle_center_world_at_capture': f['handle_center_world_at_capture'],
                                 'cam_pos_world': f['cam_pos_world'], 'cam_quat_wxyz_world': f['cam_quat_wxyz_world']}) + '\n')
    json.dump({'intrinsics_readback_K': K.tolist()}, open(os.path.join(RUN, 'wrist_live', 'meta.json'), 'w'))
    json.dump({'events': [{'phase': 'ALIGN', 'sim_t': 1e6}]}, open(os.path.join(RUN, 'task.json'), 'w'))
    r = subprocess.run([sys.executable, os.path.join(HERE, 'd1_s4_audit.py'), RUN], capture_output=True, text=True)
    check('audit_runs', r.returncode == 0, r.stderr[-800:])
    A = json.load(open(os.path.join(RUN, 'analysis', 'd1_s4_audit.json')))
    fc_ = A['reconciliation']['per_frame_fate_counts']
    check('audit_worker_error_fate', fc_.get('worker_error:detect') == 1 and fc_.get('processed') == 1, fc_)
    check('audit_pipeline_fails_on_worker_error', A['verdict']['通路與對帳'].startswith('FAIL')
          and 'worker_error 1' in A['verdict']['通路與對帳'], A['verdict'])
    check('audit_counters_close', A['reconciliation']['counters']['counters_close'] is True, A['reconciliation']['counters'])
    nd.destroy_node()
    rclpy.shutdown()


main()
print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
