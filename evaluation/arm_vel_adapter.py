"""Thin adapter: the safety filter's 9-vector to the arm's 6 joint velocities.

Format conversion and three checks, nothing else. No integration, no
smoothing, no filtering -- anything this node did to the numbers would be
attributed to the safety layer by the measurement downstream.

    /wholebody_safety/cmd_out   Float64MultiArray, 9
        [base_x, base_y, base_theta, joint1 .. joint6]   base in the WORLD frame
                    |   rotate the base pair into the body frame
                    v
    /wb_vel_cmd                 Float64MultiArray, 9      <- NOT the controller
        [vx_body, vy_body, wz, joint1 .. joint6]

ONE message, not two. The base part and the arm part come from a single
whole-body solve, and if they were published separately the gate downstream
could forward one and zero the other -- at which point the robot is executing a
command that no solver produced and that satisfies no constraint set. Keeping
them in one message makes them share a timestamp and a single staleness
decision.

The only arithmetic here is a planar rotation of the base velocity pair from
the world frame the filter solves in into the body frame the chassis plugin
takes. A rotation is exactly invertible and preserves magnitude: it changes the
representation, not the command. Nothing is clipped, smoothed or re-limited --
anything this node did to the numbers would be attributed to the safety layer
by the measurement downstream.

The controller is fed by the gate (arm_vel_gate.py), which is the only
publisher on the controller's command topic and on /cmd_vel. Two publishers on
either -- this one forwarding a live command while a watchdog sends zero --
race, and the robot does whichever arrived last.

It publishes ONLY when a new cmd_out arrives. Re-sending the last value on a
timer would make a dead safety node look alive to the gate, and the gate's
timeout would never fire. Staleness has to propagate, not be papered over.

    python3 evaluation/arm_vel_adapter.py
"""
from __future__ import annotations

import math
import sys
import time

import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray

ARM = [f'joint{i}' for i in range(1, 7)]
OUT = '/wb_vel_cmd'
# **實際消費端**的節點名。預設是 Gazebo 鏈的 ros2_control 控制器；
# Isaac 鏈沒有控制器，執行端直接吃 /wb_vel_cmd，所以要能指向它。
# 核對的對象應該是**真的會執行這些數字的那一端**。
CTRL_DEFAULT = '/lite6_vel_controller'


def verify_order(got, expect):
    """純函式：核對消費端回報的關節順序。**可不開 ROS 測試。**

    回傳 (ok, reason)。`got` 為 None 代表服務缺失或回應無效。
    """
    if got is None:
        return False, '消費端未回報 joints（服務缺失或回應無效）'
    if not isinstance(got, (list, tuple)):
        return False, f'joints 型別不對：{type(got).__name__}'
    got = list(got)
    if not all(isinstance(x, str) for x in got):
        return False, 'joints 內容不是字串陣列'
    if got == list(expect):
        return True, None
    if sorted(got) == sorted(expect):
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(got, expect)) if a != b]
        return False, f'順序不同（集合相同）：欄位 {diff}'
    return False, f'關節集合不同：消費端 {got} vs 濾波器 {list(expect)}'


class Adapter(Node):
    def __init__(self, ctrl=CTRL_DEFAULT):
        super().__init__('arm_vel_adapter')
        self.ctrl = str(ctrl)
        self.n_in = self.n_out = self.n_bad = 0
        self.last_in = 0.0
        self.yaw = None
        self.yaw_t = 0.0
        self.order_ok = self.check_order()
        from nav_msgs.msg import Odometry
        from rclpy.qos import qos_profile_sensor_data
        self.create_subscription(Odometry, '/odom', self.on_odom,
                                 qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray,
                                 '/wholebody_safety/cmd_out', self.on_cmd, 10)
        self.pub = self.create_publisher(Float64MultiArray, OUT, 10)
        self.diag = self.create_publisher(Float32MultiArray, '~/diag', 10)
        self.create_timer(0.5, self.report)

    def on_odom(self, m):
        q = m.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.yaw_t = time.monotonic()

    def check_order(self) -> bool:
        """Ask the controller which joints it drives, in which order.

        Nine numbers arriving in a message prove nothing about which joint each
        one belongs to. If the controller's list is ordered differently from
        the filter's, every joint gets someone else's velocity and the residual
        downstream still looks perfectly healthy -- there is no symptom until
        the arm moves somewhere unexpected.
        """
        got = None
        cli = self.create_client(
            __import__('rcl_interfaces.srv', fromlist=['GetParameters']).GetParameters,
            f'{self.ctrl}/get_parameters')
        if not cli.wait_for_service(timeout_sec=20.0):
            self.get_logger().error(
                f'{self.ctrl}/get_parameters 無法取得：不能核對關節順序')
        else:
            Req = __import__('rcl_interfaces.srv',
                             fromlist=['GetParameters']).GetParameters.Request
            req = Req(); req.names = ['joints']
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=20.0)
            r = fut.result()
            if r is not None and r.values:
                v = r.values[0]
                # 型別要對：必須是字串陣列，不能拿其他型別的欄位充數
                got = (list(v.string_array_value)
                       if getattr(v, 'string_array_value', None) else None)
        ok, why = verify_order(got, ARM)
        if not ok:
            self.get_logger().error(f'關節順序核對未通過（{self.ctrl}）：{why}')
            return False
        self.get_logger().info(f'關節順序已對 {self.ctrl} 核對通過：{got}')
        return True

    def on_cmd(self, msg):
        self.n_in += 1
        d = np.asarray(msg.data, dtype=float)
        if len(d) < 9 or not np.all(np.isfinite(d[:9])) or not self.order_ok:
            self.n_bad += 1
            return
        # No base yaw means no way to express the base command in the frame the
        # chassis takes. Dropping the base half and forwarding the arm half
        # would execute half of a whole-body solution, which satisfies nothing.
        if self.yaw is None or time.monotonic() - self.yaw_t > 0.3:
            self.n_bad += 1
            return
        c, s_ = math.cos(self.yaw), math.sin(self.yaw)
        vx, vy = float(d[0]), float(d[1])
        out = Float64MultiArray()
        out.data = [c * vx + s_ * vy,          # world -> body, a pure rotation
                    -s_ * vx + c * vy,
                    float(d[2])] + [float(x) for x in d[3:9]]
        self.pub.publish(out)
        self.n_out += 1
        self.last_in = time.monotonic()

    def report(self):
        m = Float32MultiArray()
        # 0 in 1 out 2 rejected 3 age of last accepted command, s
        age = time.monotonic() - self.last_in if self.last_in else -1.0
        m.data = [float(self.n_in), float(self.n_out), float(self.n_bad),
                  float(age)]
        self.diag.publish(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--consumer-node', default=CTRL_DEFAULT,
                    help='實際消費端的節點名（查其 joints 參數核對順序）。'
                         '預設為 Gazebo 鏈的 ros2_control 控制器；'
                         'Isaac 鏈請指定 /isaac_wholebody_sim')
    a, _ = ap.parse_known_args()
    rclpy.init()
    nd = Adapter(a.consumer_node)
    if not nd.order_ok:
        nd.get_logger().error('refusing to forward commands')
    try:
        rclpy.spin(nd)
    except KeyboardInterrupt:
        pass
    nd.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
