#!/usr/bin/env python3
"""GUARD＋Shield 串接的真節點隔離鏈驅動與紀錄（不起 Isaac）。由 run_nav_chain_node_check.sh 啟動真節點後執行。
世界：機器人在原點朝 +x，前方一面牆（機器人座標 x＝xw）；依最終 /cmd_vel 的 vx 積分更新 xw（運動學閉迴路）。合成 /scan 360 束、10 Hz。
模式 full：驅動發 /cmd_vel_nav → 真 velocity_smoother → /cmd_vel_pre_shield → 真 Shield → /cmd_vel_smoothed → 真 GUARD → /cmd_vel。
模式 direct：不起平滑器，驅動直接發 /cmd_vel_pre_shield（可 40 Hz 連發，逼出「同一 GUARD 週期兩筆 Shield 輸入」）。
時間軸（牆鐘 s）：0–1 靜止｜1–7 朝牆全速（牆面起始 1.5 m）｜7–8 零、牆移遠｜8–12 輪級不可行組合（vx、vy、wz 同時貼上限，每秒換號）
  ｜12–14 掃描中斷（命令 0.25 前進）｜14–15 掃描恢復｜15–17 上游命令中斷｜17–20 恢復（0.2775, 0.1, 0.5）｜20–23（direct）40 Hz 交替兩個命令｜23–24 零。
輸出：npz＋json（所有話題附接收時刻；/cmd_vel 發布者清單）。"""
import json
import math
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray, String

MODE, OUT = sys.argv[1], sys.argv[2]
R = 0.30


class Driver(Node):
    def __init__(self):
        super().__init__('nav_chain_driver')
        self.cmd_topic = '/cmd_vel_nav' if MODE == 'full' else '/cmd_vel_pre_shield'
        self.pub_cmd = self.create_publisher(Twist, self.cmd_topic, 10)
        self.pub_scan = self.create_publisher(LaserScan, '/scan', QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1))
        self.rec = {k: [] for k in ('pre', 'smoothed', 'final', 'status', 'diag', 'sent', 'world')}
        self.create_subscription(Twist, '/cmd_vel_pre_shield', lambda m: self._tw('pre', m), 50)
        self.create_subscription(Twist, '/cmd_vel_smoothed', lambda m: self._tw('smoothed', m), 50)
        self.create_subscription(Twist, '/cmd_vel', self._final, 50)
        self.create_subscription(String, '/wheel_guard/status', lambda m: self.rec['status'].append((self.now(), m.data)), 50)
        self.create_subscription(Float32MultiArray, '/shield/diag', lambda m: self.rec['diag'].append((self.now(), *[float(v) for v in m.data])), 50)
        self.t0 = time.monotonic()
        self.xw, self.vx, self.t_int = 1.5 + R, 0.0, self.now()
        self.far_set = False
        self.k40 = 0
        self.create_timer(0.02, self._world)
        self.create_timer(0.10, self._scan)
        self.create_timer(0.05, self._cmd20)
        self.create_timer(0.025, self._cmd40)
        self.pubs_final = None

    def now(self):
        return time.monotonic() - self.t0 if hasattr(self, 't0') else 0.0

    def _tw(self, k, m):
        self.rec[k].append((self.now(), m.linear.x, m.linear.y, m.angular.z))

    def _final(self, m):
        self._tw('final', m)
        self.vx = float(m.linear.x)

    def _world(self):
        t = self.now()
        self.xw -= self.vx * (t - self.t_int)
        self.t_int = t
        if t >= 7.0 and not self.far_set:
            self.xw, self.far_set = 6.0, True
        self.rec['world'].append((t, self.xw))
        if self.pubs_final is None and t > 3.0:
            self.pubs_final = [(i.node_name, i.node_namespace) for i in self.get_publishers_info_by_topic('/cmd_vel')]

    def _scan(self):
        t = self.now()
        if 12.0 <= t < 14.0:
            return
        m = LaserScan()
        m.header.frame_id = 'base_footprint'
        m.header.stamp = self.get_clock().now().to_msg()
        n = 360
        m.angle_min, m.angle_increment, m.angle_max = -math.pi, 2 * math.pi / n, math.pi - 2 * math.pi / n
        m.range_min, m.range_max = 0.05, 8.0
        ang = m.angle_min + np.arange(n) * m.angle_increment
        c = np.cos(ang)
        r = np.where(c > 1e-3, self.xw / np.maximum(c, 1e-3), np.inf)
        r[r >= m.range_max] = float('inf')
        m.ranges = [float(v) for v in r]
        self.pub_scan.publish(m)

    def _want(self, t):
        if t < 1.0 or 7.0 <= t < 8.0 or t >= 23.0:
            return (0.0, 0.0, 0.0)
        if t < 7.0:
            return (0.2775, 0.0, 0.0)
        if t < 12.0:
            s = 1.0 if int(t) % 2 == 0 else -1.0
            return (0.2775 * s, 0.2775, 1.1327 * s)
        if t < 15.0:
            return (0.25, 0.0, 0.0)
        if t < 17.0:
            return None
        if t < 20.0:
            return (0.2775, 0.1, 0.5)
        return None

    def _send(self, u):
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = (float(v) for v in u)
        self.pub_cmd.publish(m)
        self.rec['sent'].append((self.now(), *u))

    def _cmd20(self):
        u = self._want(self.now())
        if u is not None:
            self._send(u)
        elif MODE == 'full' and 20.0 <= self.now() < 23.0:
            self._send((0.2, 0.0, 0.3))

    def _cmd40(self):
        t = self.now()
        if MODE == 'direct' and 20.0 <= t < 23.0:
            self.k40 += 1
            self._send((0.27, 0.0, 0.0) if self.k40 % 2 else (0.0, 0.27, 0.0))


def main():
    rclpy.init()
    n = Driver()
    while rclpy.ok() and n.now() < 24.5:
        rclpy.spin_once(n, timeout_sec=0.005)
    np.savez_compressed(OUT + '.npz', **{k: np.array(v, dtype=float) for k, v in n.rec.items() if k != 'status'})
    json.dump(dict(mode=MODE, publishers_cmd_vel=n.pubs_final, status=n.rec['status']), open(OUT + '.json', 'w'))
    print('driver done', MODE, {k: len(v) for k, v in n.rec.items()}, n.pubs_final, flush=True)
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
