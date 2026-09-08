"""Cut one state feed at a time while commands keep flowing, and see what stops.

The question this answers
-------------------------
The gate's 150 ms watchdog measures how long since a COMMAND arrived. It cannot
see how old the state was that produced that command. So a feed can freeze while
something upstream keeps computing from the last good value and publishing at
full rate, and the watchdog stays happy the whole time. Whether the chain stops
in that case is a property of the checks INSIDE it, and those have to be tested
one feed at a time, because they are different checks with different thresholds
and one of them did not exist until it was looked for.

This publishes a constant cmd_in at a fixed rate from its own process and never
stops, so nothing here can be confused with the command path failing. Then it
cuts exactly one thing and records what the controller topics do.

  jointstate  deactivates joint_state_broadcaster
  tf          kills base_tf_bridge, so world -> base_footprint stops while
              /odom keeps flowing: the only consumer that notices is whatever
              checks the TRANSFORM's age
  odom        kills the /odom bridge, which stops both /odom and, downstream of
              it, the TF that is derived from it

    python3 evaluation/verify_state_staleness.py --fault tf
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray

REASON = {0: 'ok', 1: '無命令', 2: '命令過期', 3: '關節過期',
          4: '無運動學', 5: '無距離資料', 6: '無/過期 TF'}


def pids_matching(needle: str) -> list[int]:
    out = subprocess.run(['ps', '-eo', 'pid,args', '--no-headers'],
                         capture_output=True, text=True).stdout
    return [int(l.split()[0]) for l in out.splitlines()
            if needle in l and 'ps -eo' not in l]


class Tester(Node):

    def __init__(self, a):
        super().__init__('state_staleness_test')
        self.a = a
        self.cbg = ReentrantCallbackGroup()
        self.exec = MultiThreadedExecutor(num_threads=4)
        self.t_fault = None
        self.rows = []
        self.arm = None
        self.base_twist = None
        self.diag = None
        self.js_t = self.odom_t = None
        self.stopped = []
        self.create_subscription(Float64MultiArray,
                                 '/lite6_vel_controller/commands',
                                 self._on_arm, 20, callback_group=self.cbg)
        self.create_subscription(Twist, '/cmd_vel', self._on_base, 20,
                                 callback_group=self.cbg)
        self.create_subscription(Float32MultiArray, '/wholebody_safety/diag',
                                 self._on_diag, 20, callback_group=self.cbg)
        self.create_subscription(JointState, '/joint_states',
                                 self._on_js, 20, callback_group=self.cbg)
        self.create_subscription(Odometry, '/odom', self._on_odom,
                                 qos_profile_sensor_data, callback_group=self.cbg)
        self.pub = self.create_publisher(Float64MultiArray,
                                         '/wholebody_safety/cmd_in', 10)
        self.exec.add_node(self)

    def _on_arm(self, m):
        self.arm = np.asarray(m.data, float)

    def _on_base(self, m):
        self.base_twist = np.array([m.linear.x, m.linear.y, m.angular.z])

    def _on_diag(self, m):
        self.diag = list(m.data)

    def _on_js(self, m):
        self.js_t = time.monotonic()

    def _on_odom(self, m):
        self.odom_t = time.monotonic()

    def inject(self):
        """Fire the fault WITHOUT blocking the publishing loop.

        Run inline, `ros2 control set_controller_state` takes about 1.5 s, and
        for that whole time this process neither spins nor publishes. The first
        attempt measured a 1698 ms stop and a reason of "command stale" -- both
        of which were this test holding its own command back, not the chain
        reacting to the fault it was supposed to be testing. The command has to
        keep flowing across the injection or the experiment measures the wrong
        thing.
        """
        import threading
        f = self.a.fault
        if f == 'none':
            return 'none'
        if f == 'jointstate':
            threading.Thread(target=subprocess.run, daemon=True,
                             args=(['ros2', 'control', 'set_controller_state',
                                    'joint_state_broadcaster', 'inactive'],),
                             kwargs=dict(capture_output=True, text=True,
                                         timeout=30)).start()
            return 'joint_state_broadcaster -> inactive（背景執行）'
        # SIGSTOP, not SIGKILL. The stack launcher watches its children and
        # tears the whole stack down when one exits, so killing the TF bridge
        # also stopped /odom and the safety node, and the zeroing that followed
        # was the teardown rather than the check under test. SIGSTOP freezes the
        # process so it stops publishing while still existing, which is the
        # fault this is meant to inject: a feed that goes quiet, with everything
        # else running.
        needle = ('base_tf_bridge.py' if f == 'tf'
                  else 'parameter_bridge /odom_raw')
        ps = pids_matching(needle)
        for p in ps:
            subprocess.run(['kill', '-STOP', str(p)])
        self.stopped = ps
        return f'SIGSTOP {needle} pids={ps}'

    def run(self):
        a = self.a
        cmd = Float64MultiArray()
        cmd.data = [float(x) for x in a.cmd]
        t0 = time.monotonic()
        print(f'  持續發布 cmd_in = {a.cmd}（{a.rate:.0f} Hz），'
              f'{a.pre_s:.0f} s 後注入故障 [{a.fault}]', flush=True)
        while True:
            end = time.monotonic() + 1.0 / a.rate
            while time.monotonic() < end:
                self.exec.spin_once(timeout_sec=0.002)
            t = time.monotonic() - t0
            if self.t_fault is None and t >= a.pre_s:
                what = self.inject()
                self.t_fault = time.monotonic()
                print(f'  t={t:5.2f}s 注入：{what}', flush=True)
            # The command NEVER stops. That is the point.
            self.pub.publish(cmd)
            self.rows.append(dict(
                t=t,
                arm=None if self.arm is None else [float(x) for x in self.arm],
                base=None if self.base_twist is None
                     else [float(x) for x in self.base_twist],
                reason=None if self.diag is None else self.diag[1],
                tf_age=None if (self.diag is None or len(self.diag) < 19)
                       else self.diag[18],
                js_age=None if self.js_t is None else time.monotonic() - self.js_t,
                odom_age=None if self.odom_t is None
                         else time.monotonic() - self.odom_t))
            if t > a.pre_s + a.post_s:
                break
        for p in self.stopped:
            subprocess.run(['kill', '-CONT', str(p)])
        if self.stopped:
            print(f'  已 SIGCONT 恢復 {self.stopped}', flush=True)
        # leave zero behind
        z = Float64MultiArray()
        z.data = [0.0] * len(a.cmd)
        for _ in range(5):
            self.pub.publish(z)
            self.exec.spin_once(timeout_sec=0.01)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--fault', default='none',
                    choices=['none', 'jointstate', 'tf', 'odom'])
    ap.add_argument('--cmd', nargs=9, type=float,
                    default=[0.05, 0, 0, 0, 0.05, 0, 0, 0, 0])
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--pre-s', type=float, default=4.0)
    ap.add_argument('--post-s', type=float, default=6.0)
    a = ap.parse_args()

    rclpy.init()
    n = Tester(a)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 15 and n.arm is None:
        n.exec.spin_once(timeout_sec=0.05)
    try:
        n.run()
    except KeyboardInterrupt:
        pass
    R = n.rows
    tf = n.t_fault
    print(f'\n  {len(R)} 樣本')
    if tf is None:
        print('  沒有注入故障')
    else:
        rel = [r for r in R if r['t'] >= a.pre_s]
        def first_zero(key):
            for r in rel:
                v = r[key]
                if v is not None and max(abs(x) for x in v) < 1e-9:
                    return r['t'] - a.pre_s
            return None
        def feed_stop(key):
            prev = None
            for r in rel:
                v = r[key]
                if v is None:
                    continue
                if prev is not None and v < prev:      # age reset = still live
                    prev = v
                    continue
                if prev is not None and v > 0.2 and prev <= 0.2:
                    return r['t'] - a.pre_s - v
                prev = v
            return None
        za, zb = first_zero('arm'), first_zero('base')
        fs = feed_stop({'jointstate': 'js_age', 'tf': 'tf_age',
                        'odom': 'odom_age'}.get(a.fault, 'odom_age'))
        if fs is not None:
            print(f"  該 feed 實際停止於 故障請求後 {fs*1e3:.0f} ms")
        print(f"  故障 → 手臂命令歸零 {('%.0f ms' % (za*1e3)) if za is not None else '未歸零'}")
        print(f"  故障 → 底盤命令歸零 {('%.0f ms' % (zb*1e3)) if zb is not None else '未歸零'}")
        rs = sorted({int(r['reason']) for r in rel if r['reason'] is not None})
        print(f"  故障後 安全節點 reason：{[REASON.get(x, x) for x in rs]}")
        last = rel[-1]
        print(f"  結束時 手臂命令 {last['arm']}  底盤命令 {last['base']}")
        for k, lab in (('js_age', 'JointState'), ('odom_age', 'odom'),
                       ('tf_age', 'TF(安全節點回報)')):
            v = [r[k] for r in rel if r[k] is not None]
            if v:
                print(f"  {lab:18} 故障後最大年齡 {max(v)*1e3:7.0f} ms")
    n.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
