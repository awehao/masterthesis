"""Output gate and watchdog: the ONLY publisher on BOTH command topics.

    /wb_vel_cmd  ->  [ freshness + validity ]  -+->  /lite6_vel_controller/commands
                                                +->  /cmd_vel  (base body twist)

Forwards a command only if it is fresh and finite. Otherwise it sends zero, at
a fixed rate, for as long as the fault lasts.

One decision covers both outputs
--------------------------------
Base and arm arrive in a single message and leave in the same tick, forwarded
together or zeroed together. If they were gated separately the robot could end
up driving the base from a live whole-body solution while the arm was being
held at zero by a watchdog -- a combination no solver produced, and one that
satisfies no constraint set, while every residual upstream still reports the
original solution as feasible.

Why a separate process, and why the only publisher
--------------------------------------------------
A watchdog that shares the topic with the thing it is guarding does not guard
it: both publish, the controller takes whichever message arrived last, and the
zero and the live command alternate at whatever rate each happens to run. One
publisher, one decision.

Why the timeout is monotonic wall time
--------------------------------------
Not the ROS clock. With use_sim_time the clock stops when the simulator does
and jumps when it catches up, so a watchdog on sim time cannot notice that the
process feeding it has died -- the very case it exists for.

What this does NOT establish
----------------------------
Sending zero is a STOP REQUEST. Whether the arm then decelerates within its
acceleration limit is a property of the controller and the hardware, measured
separately. This node's own failure is also outside what it can protect
against: if this process dies the controller keeps its last command, and
nothing here covers that.

    python3 evaluation/arm_vel_gate.py [--timeout 0.15] [--rate 50]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray

CTRL_CMD = '/lite6_vel_controller/commands'
BASE_CMD = '/cmd_vel'
N = 9                       # 3 base (body frame) + 6 arm
N_BASE, N_ARM = 3, 6


class Gate(Node):
    def __init__(self, timeout: float, rate: float, vmax: float,
                 base_vmax: float):
        super().__init__('arm_vel_gate')
        self.timeout = timeout
        self.vmax = vmax
        self.base_vmax = base_vmax
        self.v = np.zeros(N)
        self.t_cmd = 0.0                    # monotonic, 0 = never received
        self.zeroing = True
        self.t_fault = None                 # when the gate started sending zero
        self.n_fwd = self.n_zero = self.n_rej = 0
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd',
                                 self.on_cmd, 10)
        self.pub = self.create_publisher(Float64MultiArray, CTRL_CMD, 10)
        from geometry_msgs.msg import Twist
        self._Twist = Twist
        self.pub_base = self.create_publisher(Twist, BASE_CMD, 10)
        self.diag = self.create_publisher(Float32MultiArray, '~/diag', 10)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f'gate: sole publisher on {CTRL_CMD} and {BASE_CMD}, '
            f'timeout {timeout*1e3:.0f} ms (monotonic), '
            f'arm cap {vmax:.2f} rad/s, base cap {base_vmax:.2f}')

    def on_cmd(self, msg):
        d = np.asarray(msg.data, dtype=float)
        # Rejection is all-or-nothing for the same reason the gate is: half a
        # whole-body command is not a smaller command, it is a different one.
        if (len(d) != N or not np.all(np.isfinite(d))
                or np.abs(d[N_BASE:]).max() > self.vmax
                or np.abs(d[:N_BASE]).max() > self.base_vmax):
            self.n_rej += 1
            return
        self.v = d
        self.t_cmd = time.monotonic()

    def tick(self):
        stale = (self.t_cmd == 0.0
                 or (time.monotonic() - self.t_cmd) > self.timeout)
        out = np.zeros(N) if stale else self.v
        if stale and not self.zeroing:
            self.t_fault = time.monotonic()
            self.get_logger().warn('command stale: sending zero')
        if not stale and self.zeroing and self.t_cmd:
            self.get_logger().info('command fresh again: forwarding')
        self.zeroing = stale
        m = Float64MultiArray()
        m.data = [float(x) for x in out[N_BASE:]]
        self.pub.publish(m)
        t = self._Twist()
        t.linear.x, t.linear.y = float(out[0]), float(out[1])
        t.angular.z = float(out[2])
        self.pub_base.publish(t)
        if stale:
            self.n_zero += 1
        else:
            self.n_fwd += 1
        d = Float32MultiArray()
        # 0 forwarded 1 zeroed 2 rejected 3 age_s 4 zeroing 5 s_since_fault
        age = (time.monotonic() - self.t_cmd) if self.t_cmd else -1.0
        sf = (time.monotonic() - self.t_fault) if self.t_fault else -1.0
        d.data = [float(self.n_fwd), float(self.n_zero), float(self.n_rej),
                  float(age), 1.0 if stale else 0.0, float(sf)]
        self.diag.publish(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeout', type=float, default=0.15)
    ap.add_argument('--rate', type=float, default=50.0)
    ap.add_argument('--vmax', type=float, default=3.2,
                    help='reject any ARM component above this, rad/s')
    ap.add_argument('--base-vmax', type=float, default=1.2,
                    help='reject any BASE component above this, m/s or rad/s')
    a = ap.parse_args()
    rclpy.init()
    nd = Gate(a.timeout, a.rate, a.vmax, a.base_vmax)
    try:
        rclpy.spin(nd)
    except KeyboardInterrupt:
        pass
    # Best effort: leave the arm with a zero command rather than the last speed.
    try:
        m = Float64MultiArray(); m.data = [0.0] * N_ARM
        t = nd._Twist()
        for _ in range(5):
            nd.pub.publish(m)
            nd.pub_base.publish(t)
            time.sleep(0.01)
    except Exception:
        pass
    nd.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
