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
        self.controller_zero_t = None
        self.fault_t = None
        self.reference_q = None
        self.create_subscription(Float64MultiArray,
                                 '/lite6_vel_controller/commands',
                                 self.on_controller, 20)

    def on_js(self, m):
        ix = {n: i for i, n in enumerate(m.name)}
        if not all(a in ix for a in ARM):
            return
        if len(m.position) <= max(ix[a] for a in ARM):
            return
        self.q = np.array([m.position[ix[a]] for a in ARM])
        self.qd = None
        if m.velocity and len(m.velocity) > max(ix[a] for a in ARM):
            self.qd = np.array([m.velocity[ix[a]] for a in ARM])
        self.t_js = time.monotonic()
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.hist.append((self.t_js, self.q.copy(),
                          self.qd.copy() if self.qd is not None else None, stamp))

    def on_controller(self, m):
        if self.fault_t is not None and self.controller_zero_t is None:
            if len(m.data) == 6 and np.all(np.isfinite(m.data)) \
                    and np.max(np.abs(m.data)) < 1e-9:
                self.controller_zero_t = time.monotonic()

    def check_state(self):
        if self.q is None or self.qd is None \
                or not np.all(np.isfinite(self.q)) \
                or not np.all(np.isfinite(self.qd)):
            raise RuntimeError('缺少有效的實際位置／速度')
        if time.monotonic() - self.t_js > 0.20:
            raise RuntimeError('JointState 超過 0.20 s 未更新')
        # This protocol has only been selected for joint3 with all other arm
        # joints at zero. These bounds are guards, not a collision certificate.
        if not -0.002 <= self.q[2] <= 0.40:
            raise RuntimeError(f'joint3 超過測試界限: {self.q[2]:.5f} rad')
        if self.reference_q is not None:
            other = [0, 1, 3, 4, 5]
            if np.max(np.abs(self.q[other] - self.reference_q[other])) > 0.01:
                raise RuntimeError('非測試關節移動超過 0.01 rad')

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

    def run_case(self, label, vel, hold, then, target=None):
        """then: 'zero' send an explicit zero, 'silence' stop publishing."""
        self.check_state()
        q0 = self.q.copy()
        self.hist.clear()
        self.gate_zero_t = None
        self.controller_zero_t = None
        self.fault_t = None
        t0 = time.monotonic()
        aborted = False
        reached = target is None
        deadline = hold if target is None else max(hold, abs(target - q0[self.j]) / abs(vel) + 3.0)
        while time.monotonic() - t0 < deadline:
            self.check_state()
            command = vel
            if target is not None:
                error = target - self.q[self.j]
                if abs(error) <= 0.003:
                    reached = True
                    break
                if error * vel <= 0:
                    raise RuntimeError('越過位置目標，停止自檢')
                command = float(np.clip(2.0 * error, -abs(vel), abs(vel)))
            self.send(command)
            self.wait(0.02)
            self.check_state()
            if abs(self.q[self.j] - q0[self.j]) > self.limit:
                aborted = True
                break
        if not reached:
            aborted = True
        moved = float(self.q[self.j] - q0[self.j])
        vpeak = max((abs(h[2][self.j]) for h in self.hist if h[2] is not None),
                    default=float('nan'))
        t_fault = time.monotonic()
        self.fault_t = t_fault
        # Even during silence the independent displacement and state-age
        # guards remain live. An abort always sends zero, never waits silently.
        while time.monotonic() - t_fault < 1.5:
            self.check_state()
            if abs(self.q[self.j] - q0[self.j]) > self.limit:
                aborted = True
            if then == 'zero' or aborted:
                self.send(0.0)
            self.wait(0.02)
        # when did it actually stop
        t_stop = None
        for t, _, qd, _ in self.hist:
            if qd is None or t < t_fault:
                continue
            if abs(qd[self.j]) < STOPPED:
                t_stop = t
                break
        # Peak deceleration after the fault, and -- just as important -- whether
        # this sampling could resolve it at all.
        #
        # /joint_states arrives at ~143 Hz with a 1 ms stamp quantum, so the
        # braking window is 6-8 ms wide. A stop that completes inside one
        # sample yields exactly vpeak/dt, which is a property of the sampling
        # rate, not of the joint: from 0.15 rad/s the only values a one-sample
        # drop can produce are 25.00, 21.43 and 18.75 rad/s^2, and the 19.98
        # limit sits between two of them. Such a reading decides nothing. It is
        # a LOWER bound on the true magnitude, reported as indeterminate rather
        # than as a pass or a failure.
        dec = 0.0
        dt_max = 0.0
        prev = None
        n_span = 0          # sample intervals the ramp-down actually spans
        braking = False
        for t, _, qd, stamp in self.hist:
            if qd is None:
                continue
            # Physical acceleration is differentiated in the joint sample's
            # time domain, not by DDS receipt intervals. Keep the last sample
            # before the fault so the first braking step is not omitted.
            if t >= t_fault and prev is not None and stamp > prev[0]:
                dt = stamp - prev[0]
                dec = max(dec, abs(qd[self.j] - prev[1]) / dt)
                dt_max = max(dt_max, dt)
                if braking or abs(prev[1]) >= STOPPED:
                    braking = True
                    if abs(prev[1]) >= STOPPED:
                        n_span += 1
            prev = (stamp, qd[self.j])
        amax = LITE6_SAFE.max_acceleration[self.j]
        # Smallest magnitude this sampling can tell apart from an instant drop.
        floor = vpeak / dt_max if dt_max > 0 else float('inf')
        resolved = n_span >= 2 and dec < floor * 0.9
        print(f"  {label:22} 命令 {vel:+.3f}  實際位移 {moved:+.4f} rad"
              f"  峰值實際速度 {vpeak:.4f} rad/s{'  [到位移上限中止]' if aborted else ''}")
        z = (self.controller_zero_t - t_fault) * 1e3 if self.controller_zero_t else float('nan')
        s = (t_stop - t_fault) * 1e3 if t_stop else float('nan')
        print(f"  {'':22} 故障→送零 {z:7.1f} ms   故障→實際停止 {s:7.1f} ms"
              f"   峰值減速度 {dec:6.2f} rad/s^2 (限 {amax:.2f})")
        if vpeak >= STOPPED:
            verdict = ('可分辨' if resolved else
                       '無法判定：降速未跨足夠樣本，'
                       '讀數是取樣分辨下限而非真實值')
            print(f"  {'':22} 降速横跨 {n_span} 個樣本間隔"
                  f"（最大 dt {dt_max*1e3:.1f} ms），"
                  f"可分辨下限 {floor:6.2f} rad/s^2 → {verdict}")
        self.check_state()
        stopped = t_stop is not None and np.max(np.abs(self.qd)) < STOPPED
        motion_ok = (moved * np.sign(vel) > 0.01 if vel != 0
                     else max(abs(h[1][self.j] - q0[self.j]) for h in self.hist) < 0.003)
        good = (motion_ok and stopped and self.controller_zero_t is not None
                and not aborted and reached)
        print(f'  結束 joint3={self.q[2]:.5f} rad  ' + ('PASS' if good else 'FAIL'), flush=True)
        return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--joint', type=int, default=3, help='1-6')
    ap.add_argument('--vel', type=float, default=0.15)
    ap.add_argument('--hold', type=float, default=2.0)
    ap.add_argument('--limit', type=float, default=0.45, help='rad, hard cap')
    ap.add_argument('--check-only', action='store_true', help='只核對 ROS 與初始狀態，不發命令')
    a = ap.parse_args()
    j = a.joint - 1
    if a.joint != 3 or not 0 < a.vel <= 0.15 or not 0 < a.hold <= 2.0 \
            or not 0 < a.limit <= 0.45:
        ap.error('本路徑只適用 joint3、0 < vel <= 0.15、0 < hold <= 2、0 < limit <= 0.45')
    rclpy.init()
    nd = Probe(j, a.vel, a.limit)
    nd.wait(3.0)
    armed = False
    try:
        nd.check_state()
        print(f'實際起始關節: {nd.q.tolist()}', flush=True)
        if np.max(np.abs(nd.q)) > 0.01 or np.max(np.abs(nd.qd)) >= STOPPED:
            raise RuntimeError('必須從全零附近且靜止開始；不自動轉場')
        # The probe replaces the adapter as gate input during this self-test.
        if nd.count_publishers('/arm_vel_cmd') != 1:
            raise RuntimeError('/arm_vel_cmd 存在其他發布者，不能競爭命令')
        if nd.count_publishers('/lite6_vel_controller/commands') != 1 \
                or nd.count_subscribers('/lite6_vel_controller/commands') < 1 \
                or nd.gate_zeroing is not True:
            raise RuntimeError('控制器／唯一輸出閘門尚未就緒或非零保持中')
        nd.reference_q = nd.q.copy()
        print('前置檢查通過：全零靜止、唯一閘門、控制器訂閱存在', flush=True)
        if a.check_only:
            return 0
        armed = True
        cases = [('1 正向至 +0.30', a.vel, a.hold, 'zero', 0.30),
                 ('2 反向回 +0.05', -a.vel, a.hold, 'zero', 0.05),
                 ('3 短正向 + 停止發布', a.vel, min(a.hold, 0.5), 'silence', None),
                 ('4 零命令（不應動）', 0.0, a.hold, 'zero', None)]
        for args in cases:
            if not nd.run_case(*args):
                print('★ 自檢未通過；停止後續案例', flush=True)
                return 1
        print('四案通過：方向、實際運動、送零及實際停止均已核對', flush=True)
        return 0
    except RuntimeError as exc:
        print(f'★ {exc}', flush=True)
        return 1
    finally:
        if armed:
            try:
                for _ in range(10):
                    nd.send(0.0)
                    nd.wait(0.02)
            except Exception:
                pass
        nd.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
