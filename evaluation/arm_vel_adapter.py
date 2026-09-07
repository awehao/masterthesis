"""Thin adapter: the safety filter's 9-vector to the arm's 6 joint velocities.

Format conversion and three checks, nothing else. No integration, no
smoothing, no filtering -- anything this node did to the numbers would be
attributed to the safety layer by the measurement downstream.

    /wholebody_safety/cmd_out   Float64MultiArray, 9
        [base_x, base_y, base_theta, joint1 .. joint6]
                    |   take the last six
                    v
    /arm_vel_cmd                Float64MultiArray, 6      <- NOT the controller

The controller is fed by the gate (arm_vel_gate.py), which is the only
publisher on the controller's command topic. Two publishers on that topic --
this one forwarding a live command while a watchdog sends zero -- race, and the
arm does whichever arrived last.

It publishes ONLY when a new cmd_out arrives. Re-sending the last value on a
timer would make a dead safety node look alive to the gate, and the gate's
timeout would never fire. Staleness has to propagate, not be papered over.

    python3 evaluation/arm_vel_adapter.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray

ARM = [f'joint{i}' for i in range(1, 7)]
OUT = '/arm_vel_cmd'
CTRL = '/lite6_vel_controller'


class Adapter(Node):
    def __init__(self):
        super().__init__('arm_vel_adapter')
        self.n_in = self.n_out = self.n_bad = 0
        self.last_in = 0.0
        self.order_ok = self.check_order()
        self.create_subscription(Float64MultiArray,
                                 '/wholebody_safety/cmd_out', self.on_cmd, 10)
        self.pub = self.create_publisher(Float64MultiArray, OUT, 10)
        self.diag = self.create_publisher(Float32MultiArray, '~/diag', 10)
        self.create_timer(0.5, self.report)

    def check_order(self) -> bool:
        """Ask the controller which joints it drives, in which order.

        Nine numbers arriving in a message prove nothing about which joint each
        one belongs to. If the controller's list is ordered differently from
        the filter's, every joint gets someone else's velocity and the residual
        downstream still looks perfectly healthy -- there is no symptom until
        the arm moves somewhere unexpected.
        """
        cli = self.create_client(
            __import__('rcl_interfaces.srv', fromlist=['GetParameters']).GetParameters,
            f'{CTRL}/get_parameters')
        if not cli.wait_for_service(timeout_sec=20.0):
            self.get_logger().error(
                f'{CTRL}/get_parameters unavailable: cannot verify joint order')
            return False
        Req = __import__('rcl_interfaces.srv', fromlist=['GetParameters']).GetParameters.Request
        req = Req(); req.names = ['joints']
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=20.0)
        if fut.result() is None or not fut.result().values:
            self.get_logger().error('no joints parameter from the controller')
            return False
        got = list(fut.result().values[0].string_array_value)
        if got != ARM:
            self.get_logger().error(
                f'JOINT ORDER MISMATCH: controller {got} vs filter {ARM}')
            return False
        self.get_logger().info(f'joint order verified against {CTRL}: {got}')
        return True

    def on_cmd(self, msg):
        self.n_in += 1
        d = np.asarray(msg.data, dtype=float)
        if len(d) < 9 or not np.all(np.isfinite(d[3:9])) or not self.order_ok:
            self.n_bad += 1
            return
        out = Float64MultiArray()
        out.data = [float(x) for x in d[3:9]]
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
    rclpy.init()
    nd = Adapter()
    if not nd.order_ok:
        nd.get_logger().error('refusing to forward commands')
    try:
        rclpy.spin(nd)
    except KeyboardInterrupt:
        pass
    nd.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
