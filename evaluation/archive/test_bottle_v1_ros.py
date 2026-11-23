#!/usr/bin/env python3
"""V1 估計節點的 ROS 整合核對（不開 Isaac）：本程式當擷取端，發 /clock 與合成影格（test_bottle_v1_est.render，沿 bottle_p1_3 軌跡），估計節點是獨立程序。
核：同戳配對、估計回來的戳＝擷取戳、資料年齡（收到時的 /clock − 擷取戳）、壞訊息照樣回一筆失敗、連發舊影格時佇列有界且有丟棄計數。真值只在本程式評估。
    ROS_DOMAIN_ID=93 python3 evaluation/test_bottle_v1_ros.py
"""
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_ros_chain as RC                           # noqa: E402
TV, = RC.import_with_real_ros('test_bottle_v1_est')     # 它連帶匯入的 bottle_nav_chain 會裝 ros_stub，不能讓它蓋掉真 rclpy

import rclpy                                            # noqa: E402
from builtin_interfaces.msg import Time                 # noqa: E402
from geometry_msgs.msg import PoseStamped               # noqa: E402
from rosgraph_msgs.msg import Clock                     # noqa: E402
from sensor_msgs.msg import CameraInfo, Image           # noqa: E402
from std_msgs.msg import Float64MultiArray              # noqa: E402

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def stamp(t):
    ns = int(round(t * 1e9))
    return Time(sec=ns // 1000000000, nanosec=ns % 1000000000)


def R_to_wxyz(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def frames(n=6):
    """沿 p1_3 軌跡取幾個幾何可見的影格（1–2 m）。"""
    Z = np.load(os.path.join(HERE, 'runs', 'bottle_p1_3', 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    out = []
    for i in range(0, len(c['t']), 20):
        q = np.array([c[k][i] for k in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        T_wc = TV.G.C.K.fk(q, TV.CAM['parent']) @ TV.CAM['T']
        w, x, y, z = (c[k][i] for k in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz'))
        ax = np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])
        b0 = np.array([c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]])
        dist = float(np.linalg.norm(T_wc[:3, 3] - b0))
        if 1.0 <= dist <= 2.0:
            out.append((T_wc, b0, ax))
        if len(out) >= n:
            break
    return out


def main():
    TB = TV.TABLE1
    arg = f"{TB['top_z']},{TB['lo'][0]},{TB['lo'][1]},{TB['hi'][0]},{TB['hi'][1]}"
    node_p = subprocess.Popen([sys.executable, '-u', os.path.join(HERE, 'bottle_v1_est_node.py'), '--table', arg, '--ros-args', '-p', 'use_sim_time:=true'],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    rclpy.init()
    nd = rclpy.create_node('v1_test_src')
    pc = nd.create_publisher(Clock, '/clock', 10)
    pubs = dict(rgb=nd.create_publisher(Image, '/v1/color', 5), depth=nd.create_publisher(Image, '/v1/depth', 5),
                info=nd.create_publisher(CameraInfo, '/v1/camera_info', 5), pose=nd.create_publisher(PoseStamped, '/v1/cam_pose', 5))
    got = []
    clk = dict(t=10.0)
    nd.create_subscription(Float64MultiArray, '/v1/bottle_estimate', lambda m: got.append((clk['t'], list(m.data))), 10)

    def spin(sec):
        w = time.time()
        while time.time() - w < sec:
            c_ = Clock()
            c_.clock = stamp(clk['t'])
            pc.publish(c_)
            rclpy.spin_once(nd, timeout_sec=0.01)
            clk['t'] += 0.01

    def send(t, T_wc, D, which=('rgb', 'depth', 'info', 'pose'), bad=False):
        st = stamp(t)
        H, W = D.shape
        if 'rgb' in which:
            m = Image(height=H, width=W, encoding='rgb8', step=3 * W)
            m.header.stamp, m.header.frame_id, m.data = st, 'base_camera_color_optical_frame', bytes(H * W * 3)
            pubs['rgb'].publish(m)
        if 'depth' in which:
            m = Image(height=H, width=W, encoding='16UC1' if bad else '32FC1', step=4 * W)
            m.header.stamp, m.header.frame_id, m.data = st, 'base_camera_color_optical_frame', D.astype(np.float32).tobytes()
            pubs['depth'].publish(m)
        if 'info' in which:
            m = CameraInfo(height=H, width=W)
            m.header.stamp, m.k = st, [float(v) for v in TV.K.ravel()]
            pubs['info'].publish(m)
        if 'pose' in which:
            m = PoseStamped()
            m.header.stamp, m.header.frame_id = st, 'odom'
            q = R_to_wxyz(T_wc[:3, :3])
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in T_wc[:3, 3])
            m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = (float(v) for v in q)
            pubs['pose'].publish(m)

    spin(3.0)                                                   # 等探索
    F = frames()
    boxes = TV.scene_boxes()
    sent = []
    for k, (T_wc, b0, ax) in enumerate(F[:4]):
        D = TV.render(T_wc, boxes, (b0, ax, TV.G.BOTTLE['r'], TV.G.BOTTLE['h']), TV.CAM['near'], TV.CAM['far'])
        t_cap = round(clk['t'], 2)
        send(t_cap, T_wc, D)
        sent.append((t_cap, b0))
        spin(1.0)
    est = {round(g[1][0], 2): g for g in got}
    ok_n = sum(1 for t_, _ in sent if t_ in est and est[t_][1][1] == 1.0)
    chk('同戳配對：4 個完整影格都回估計且有效', ok_n == 4, (ok_n, [round(g[1][0], 2) for g in got], [s_[0] for s_ in sent]))
    if ok_n:
        e = [math.hypot(est[t_][1][2] - b0[0], est[t_][1][3] - b0[1]) * 1e3 for t_, b0 in sent if t_ in est]
        chk('回傳中心 xy 與合成真值一致（≤1 mm）', max(e) <= 1.0, e)
        age = [est[t_][0] - t_ for t_, _ in sent if t_ in est]
        print(f'   資料年齡（收到時 /clock − 擷取戳）{np.round(age, 3).tolist()} s；處理耗時 {[round(est[t_][1][11], 1) for t_, _ in sent if t_ in est]} ms')
        chk('資料年齡為正且有限', all(0 <= a_ < 5 for a_ in age), age)
    n0 = len(got)
    t_part = round(clk['t'], 2)
    send(t_part, F[0][0], np.zeros((480, 640)), which=('rgb', 'depth', 'info'))            # 缺位姿：不得配對
    spin(1.0)
    chk('缺一個話題的戳不產生估計', all(round(g[1][0], 2) != t_part for g in got[n0:]))
    t_bad = round(clk['t'], 2)
    send(t_bad, F[0][0], np.zeros((480, 640)), bad=True)
    spin(1.0)
    gb = [g for g in got if round(g[1][0], 2) == t_bad]
    chk('壞訊息（深度編碼錯）照樣回一筆失敗', len(gb) == 1 and gb[0][1][1] == 0.0 and gb[0][1][12] == 4.0, gb[:1])
    for k in range(8):                                          # 連發只有部分話題的舊影格：佇列有界、丟棄計數
        send(round(clk['t'] + 0.001 * k, 3), F[0][0], np.zeros((480, 640)), which=('rgb',))
    spin(0.5)
    D = TV.render(F[1][0], boxes, (F[1][1], F[1][2], TV.G.BOTTLE['r'], TV.G.BOTTLE['h']), TV.CAM['near'], TV.CAM['far'])
    t_last = round(clk['t'], 2)
    send(t_last, F[1][0], D)
    spin(1.0)
    gl = [g for g in got if round(g[1][0], 2) == t_last]
    chk('佇列有界：殘缺舊影格被丟棄並計數，之後完整影格仍能估計', len(gl) == 1 and gl[0][1][13] >= 5, gl[:1] and gl[0][1][13])
    nd.destroy_node()
    rclpy.shutdown()
    os.killpg(node_p.pid, 15)
    node_p.wait(timeout=5)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
