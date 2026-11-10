#!/usr/bin/env python3
"""第八次第一包：短程閉迴路用的二維假世界（Codex 20261110_183705 第 3 項）。取代 Isaac，其餘節點（追蹤器、封存 gmpc_node、平滑器、
Shield、planner_server、導航節點）全部照實跑：候選計畫 → GMPC → 平滑器 → Shield → 本節點運動模型。
世界：mission_room.sdf 已知牆與家具（mission_room_obstacles.known_boxes）＋關閉門扇＋未知箱體＋行人（mission_walkers.position，同 seed 檔、同世界時鐘）。
發布：/clock（模擬時間，以 --rtf 倍速推進）、/odom、TF odom→base_footprint、/joint_states（手臂 K2_READY，供 robot_state_publisher）、
  /scan_raw（經 scan_relay 成 /scan；frame lidar_link；360 束、10 Hz；二維射線對多邊形與圓求交，距離 0.12–10 m）。訂 /cmd_vel（車體速度），以一階延遲 TAU 與逐軸加速度限制積分（全向）。
不模擬接觸力：行人不會推人、機器人不會被擋住；圓盤與物體重疊照實記為安全包絡侵入（truth log）。這是與 Isaac 不同的地方，結果只作閉迴路行為比較。
真值 log 只寫檔（m8fw_truth.npz），不發布。
起跑同步：--go-file 指定時，ROS 時鐘照走（節點啟動需要），但行人世界時間停在 --t0、機器人不接受命令；旗標檔出現後世界時間才從 --t0 往下走（tw＝t0＋(t−t_go)）。
    python3 evaluation/m8_fakeworld_node.py --start x,y,yaw --t0 26.0 --out RUN_DIR [--rtf 1.0]
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mission_room_obstacles as MO  # noqa: E402
import mission_walkers as MW  # noqa: E402

import rclpy  # noqa: E402
from geometry_msgs.msg import TransformStamped, Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import JointState, LaserScan  # noqa: E402
from tf2_ros import TransformBroadcaster  # noqa: E402

DT, SCAN_DT = 0.01, 0.1
SCAN_N, SCAN_MIN, SCAN_MAX, R_MIN, R_MAX = 360, -3.14159, 3.14159, 0.12, 10.0
TAU = 0.10                                             # 底盤速度一階延遲（開發假設）
A_MAX = np.array([1.5, 1.0, 2.0])
ARM = [f'joint{i}' for i in range(1, 7)]
K2_READY = [0.0, 0.0, 0.035, 0.0, -1.5708, 0.0]


def box_poly(u):
    c, s = math.cos(math.radians(u['yaw_deg'])), math.sin(math.radians(u['yaw_deg']))
    sx, sy = u['size'][0], u['size'][1]
    return np.array([(u['x'] + c * px - s * py, u['y'] + s * px + c * py) for px, py in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))])


def segments():
    segs = []
    polys = [np.array([(r[0], r[2]), (r[1], r[2]), (r[1], r[3]), (r[0], r[3])]) for r in MO.known_boxes()]
    polys.append(np.array([(0.0, 0.0), (0.95, 0.0), (0.95, 0.04), (0.0, 0.04)]))          # 關閉門扇（門先驗）
    return polys


def ray_polys(o, D, polys):
    """D：(N,2) 單位向量；回傳各束最近交點距離。"""
    best = np.full(len(D), np.inf)
    for P in polys:
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            e = b - a
            den = D[:, 0] * e[1] - D[:, 1] * e[0]
            ok = np.abs(den) > 1e-12
            w = a - o
            t = np.where(ok, (w[0] * e[1] - w[1] * e[0]) / np.where(ok, den, 1), np.inf)
            u = np.where(ok, (w[0] * D[:, 1] - w[1] * D[:, 0]) / np.where(ok, den, 1), -1)
            hit = ok & (t > 0) & (u >= 0) & (u <= 1)
            best = np.where(hit & (t < best), t, best)
    return best


def ray_circle(o, D, c, r):
    w = o - c
    b = D @ w
    cc = w @ w - r * r
    disc = b * b - cc
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    return np.where((disc >= 0) & (t > 0), t, np.inf)


class FakeWorld(Node):
    def __init__(self, a):
        super().__init__('m8_fakeworld')
        self.a = a
        ob = json.load(open(a.obstacles))
        self.walkers = ob['walkers']
        self.polys = segments() + [box_poly(u) for u in ob['static_unknown']]
        x, y, th = (float(v) for v in a.start.split(','))
        self.q = np.array([x, y, th])
        self.v = np.zeros(3)
        self.cmd = np.zeros(3)
        self.t = float(a.t0)                                # ROS 時鐘
        self.tw, self.t_go = float(a.t0), (None if a.go_file else float(a.t0))   # 行人世界時間；開始時刻
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan_raw', 10)            # 同 Isaac 執行端：/scan_raw → scan_relay → /scan
        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Twist, '/cmd_vel', self._cmd, 10)
        self.next_scan = self.t
        self.log = []
        self.ang = SCAN_MIN + np.arange(SCAN_N) * (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)

    def _cmd(self, m):
        self.cmd = np.array([m.linear.x, m.linear.y, m.angular.z])

    def stamp(self):
        from builtin_interfaces.msg import Time
        s = Time()
        s.sec = int(math.floor(self.t))
        s.nanosec = int((self.t - math.floor(self.t)) * 1e9)
        return s

    def step(self):
        if self.t_go is None and os.path.exists(self.a.go_file):
            self.t_go = self.t
            self.get_logger().info(f'go：ROS 時鐘 {self.t:.2f}、世界時間 {self.tw:.2f}')
        if self.t_go is None:
            self.cmd = np.zeros(3)                          # 開始前不接受命令
        dv = (self.cmd - self.v) * (DT / TAU)
        dv = np.clip(dv, -A_MAX * DT, A_MAX * DT)
        self.v = self.v + dv
        c, s = math.cos(self.q[2]), math.sin(self.q[2])
        self.q = self.q + DT * np.array([c * self.v[0] - s * self.v[1], s * self.v[0] + c * self.v[1], self.v[2]])
        self.t += DT
        if self.t_go is not None:
            self.tw = float(self.a.t0) + (self.t - self.t_go)
        st = self.stamp()
        self.clock_pub.publish(Clock(clock=st))
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = st, 'odom', 'base_footprint'
        od.pose.pose.position.x, od.pose.pose.position.y = float(self.q[0]), float(self.q[1])
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = math.sin(self.q[2] / 2), math.cos(self.q[2] / 2)
        od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.angular.z = (float(v) for v in self.v)
        self.odom_pub.publish(od)
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = st, 'odom', 'base_footprint'
        tf.transform.translation.x, tf.transform.translation.y = float(self.q[0]), float(self.q[1])
        tf.transform.rotation.z, tf.transform.rotation.w = math.sin(self.q[2] / 2), math.cos(self.q[2] / 2)
        self.tf.sendTransform(tf)
        js = JointState()
        js.header.stamp, js.name, js.position = st, ARM + ['finger_joint1', 'finger_joint2'], K2_READY + [0.0, 0.0]
        self.js_pub.publish(js)
        W = [np.array(MW.position(w, self.tw)) for w in self.walkers]
        if self.t >= self.next_scan - 1e-9:
            self.next_scan += SCAN_DT
            o = self.q[:2]
            D = np.c_[np.cos(self.ang + self.q[2]), np.sin(self.ang + self.q[2])]
            r = ray_polys(o, D, self.polys)
            for w, c_ in zip(self.walkers, W):
                r = np.minimum(r, ray_circle(o, D, c_, float(w['radius'])))
            r = np.where((r >= R_MIN) & (r <= R_MAX), r, np.inf)
            sc = LaserScan()
            sc.header.stamp, sc.header.frame_id = st, 'lidar_link'
            sc.angle_min, sc.angle_max, sc.angle_increment = SCAN_MIN, SCAN_MAX, (SCAN_MAX - SCAN_MIN) / (SCAN_N - 1)
            sc.range_min, sc.range_max = R_MIN, R_MAX
            sc.ranges = [float(v) for v in r]
            self.scan_pub.publish(sc)
        if not self.log or self.t >= self.log[-1][0] + 0.05 - 1e-9:
            row = [self.t, self.tw, *self.q.tolist(), *self.v.tolist(), *self.cmd.tolist()]
            for c_ in W:
                row += [float(c_[0]), float(c_[1])]
            self.log.append(row)

    def save(self):
        cols = ['t', 'tw', 'base_x', 'base_y', 'base_yaw', 'v_x', 'v_y', 'v_w', 'cmd_x', 'cmd_y', 'cmd_w'] + [f"{w['name']}_{k}" for w in self.walkers for k in ('x', 'y')]
        np.savez_compressed(os.path.join(self.a.out, 'm8fw_truth.npz'), rows=np.array(self.log, float), cols=np.array(cols),
                            walkers=np.array(json.dumps(self.walkers)), static=np.array(json.dumps(json.load(open(self.a.obstacles))['static_unknown'])),
                            meta=np.array(json.dumps(dict(t_go=self.t_go, dt=DT, tau=TAU, a_max=A_MAX.tolist(), start=self.a.start, t0=self.a.t0, rtf=self.a.rtf,
                                                          note='二維假世界：不模擬接觸力；行人不推人'))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--t0', type=float, default=0.0)
    ap.add_argument('--obstacles', default=os.path.join(HERE, 'results', 'mission', 'mission_obstacles_seed1.json'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--rtf', type=float, default=1.0)
    ap.add_argument('--t-end', type=float, default=1e9, help='世界時間上限')
    ap.add_argument('--go-file', default='')
    a, _ = ap.parse_known_args()
    rclpy.init()
    w = FakeWorld(a)
    stop = {'v': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    wall0, sim0 = time.monotonic(), w.t
    try:
        while rclpy.ok() and not stop['v'] and w.tw < a.t_end:
            target = wall0 + (w.t - sim0) / a.rtf
            while time.monotonic() < target:
                rclpy.spin_once(w, timeout_sec=max(0.0, min(0.002, target - time.monotonic())))
            rclpy.spin_once(w, timeout_sec=0.0)
            w.step()
    except KeyboardInterrupt:
        pass
    finally:
        w.save()
        w.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
