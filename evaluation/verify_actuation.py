"""Actuation self-test: does a velocity command reach the joints, and does
removing it stop them.

Four cases, in free space, before the safety layer is allowed anywhere near the
loop:

    forward     a small positive velocity on one joint
    reverse     the same magnitude negative
    zero        an explicit zero command
    silence     stop publishing entirely, and let the gate's watchdog act

Its own stop mechanism, not the one under test
----------------------------------------------
Small speed, short window, and a hard displacement limit: the moment a joint
has moved further than allowed, this script sends zero and ends the case. The
exit condition is never "it should stop by itself" -- that is the thing being
measured.

Two times are recorded and they are not the same thing:

    fault -> zero sent      when the gate published a zero command
    fault -> actually stopped   when |qdot| from /joint_states fell below a
                                threshold

A zero command is a request. Whether the deceleration that follows respects the
joint's acceleration limit is a separate question, and the peak is reported so
it can be judged rather than assumed.

    python3 evaluation/verify_actuation.py [--joint 3] [--vel 0.15]
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray

sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE  # noqa: E402

ARM = [f'joint{i}' for i in range(1, 7)]
STOPPED = 0.01          # rad/s below which the joint counts as stopped


class Probe(Node):
    def __init__(self, j: int, vel: float, limit: float):
        super().__init__('verify_actuation')
        self.j, self.vel, self.limit = j, vel, limit
        self.q = None
        self.qd = None
        self.t_js = 0.0
        self.hist = []
        self.gate_zero_t = None
        self.create_subscription(JointState, '/joint_states', self.on_js, 20)
        self.create_subscription(Float32MultiArray, '/arm_vel_gate/diag',
                                 self.on_gate, 20)
        self.pub = self.create_publisher(Float64MultiArray, '/arm_vel_cmd', 10)
        self.gate_zeroing = None

    def on_js(self, m):
        ix = {n: i for i, n in enumerate(m.name)}
        if not all(a in ix for a in ARM):
            return
        self.q = np.array([m.position[ix[a]] for a in ARM])
        if m.velocity and len(m.velocity) > max(ix[a] for a in ARM):
            self.qd = np.array([m.velocity[ix[a]] for a in ARM])
        self.t_js = time.monotonic()
        self.hist.append((self.t_js, self.q.copy(),
                          self.qd.copy() if self.qd is not None else None))

    def on_gate(self, m):
        z = m.data[4] > 0.5
        if z and self.gate_zeroing is False:
            self.gate_zero_t = time.monotonic()
        self.gate_zeroing = z

    def send(self, v):
        m = Float64MultiArray()
        d = np.zeros(6); d[self.j] = v
        m.data = [float(x) for x in d]
        self.pub.publish(m)

    def wait(self, sec):
        t0 = time.monotonic()
        while time.monotonic() - t0 < sec:
            rclpy.spin_once(self, timeout_sec=0.01)

    def run_case(self, label, vel, hold, then):
        """then: 'zero' send an explicit zero, 'silence' stop publishing."""
        if self.q is None:
            self.wait(3.0)
        q0 = self.q.copy()
        self.hist.clear()
        self.gate_zero_t = None
        t0 = time.monotonic()
        aborted = False
        while time.monotonic() - t0 < hold:
            self.send(vel)
            self.wait(0.02)
            if abs(self.q[self.j] - q0[self.j]) > self.limit:
                aborted = True
                break
        moved = float(self.q[self.j] - q0[self.j])
        vpeak = max((abs(h[2][self.j]) for h in self.hist if h[2] is not None),
                    default=float('nan'))
        t_fault = time.monotonic()
        if then == 'zero':
            t_end = time.monotonic()
            while time.monotonic() - t_end < 1.5:
                self.send(0.0)
                self.wait(0.02)
        else:
            self.wait(1.5)            # publish nothing at all
        # when did it actually stop
        t_stop = None
        for t, _, qd in self.hist:
            if qd is None or t < t_fault:
                continue
            if abs(qd[self.j]) < STOPPED:
                t_stop = t
                break
        # peak deceleration after the fault
        dec = 0.0
        prev = None
        for t, _, qd in self.hist:
            if qd is None or t < t_fault:
                continue
            if prev is not None and t > prev[0]:
                dec = max(dec, abs(qd[self.j] - prev[1]) / (t - prev[0]))
            prev = (t, qd[self.j])
        amax = LITE6_SAFE.max_acceleration[self.j]
        print(f"  {label:22} 命令 {vel:+.3f}  實際位移 {moved:+.4f} rad"
              f"  峰值實際速度 {vpeak:.4f} rad/s{'  [到位移上限中止]' if aborted else ''}")
        z = (self.gate_zero_t - t_fault) * 1e3 if self.gate_zero_t else float('nan')
        s = (t_stop - t_fault) * 1e3 if t_stop else float('nan')
        print(f"  {'':22} 故障→送零 {z:7.1f} ms   故障→實際停止 {s:7.1f} ms"
              f"   峰值減速度 {dec:6.2f} rad/s^2 (限 {amax:.2f})")
        return abs(moved) > 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--joint', type=int, default=3, help='1-6')
    ap.add_argument('--vel', type=float, default=0.15)
    ap.add_argument('--hold', type=float, default=2.0)
    ap.add_argument('--limit', type=float, default=0.45, help='rad, hard cap')
    a = ap.parse_args()
    j = a.joint - 1
    rclpy.init()
    nd = Probe(j, a.vel, a.limit)
    nd.wait(3.0)
    if nd.q is None:
        print('  ★ 沒有 /joint_states'); rclpy.shutdown(); return 1
    print(f"  joint{a.joint}  速度 {a.vel} rad/s  位移上限 {a.limit} rad\n")
    ok = []
    ok.append(nd.run_case('1 正向 + 明確送零', +a.vel, a.hold, 'zero'))
    nd.wait(1.0)
    ok.append(nd.run_case('2 反向 + 明確送零', -a.vel, a.hold, 'zero'))
    nd.wait(1.0)
    ok.append(nd.run_case('3 正向 + 停止發布', +a.vel, a.hold, 'silence'))
    nd.wait(1.0)
    ok.append(nd.run_case('4 零命令（不應動）', 0.0, a.hold, 'zero'))
    print(f"\n  致動成立（前三案有位移，第四案無位移）: "
          f"{'✓' if (ok[0] and ok[1] and ok[2] and not ok[3]) else '★ 未通過'}")
    nd.destroy_node(); rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
