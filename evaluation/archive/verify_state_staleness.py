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
  odom        freezes the /odom bridge, which stops both /odom and, downstream
              of it, the TF that is derived from it
  distance    freezes arm_link_distance while JointState and TF stay live. This
              one is expected NOT to stop: stale points turn every row into
              NODATA and the filter applies a speed cap. That is a DEGRADATION,
              not a stop, and low speed is not by itself evidence of safety --
              with the obstacle distance unknown, slow is still enough to hit
              something. The test records what actually comes out, whether the
              robot keeps moving, and whether it recovers when the feed returns.

    python3 evaluation/verify_state_staleness.py --fault tf
"""
from __future__ import annotations

import argparse
import math
import os
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
        self.t_resume = None
        self.base_xy = None
        self.q = None
        self.base_stamp = None
        self.js_stamp = None
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
        ix = {n: i for i, n in enumerate(m.name)}
        a = [f'joint{i}' for i in range(1, 7)]
        if all(j in ix for j in a) and len(m.position) > max(ix[j] for j in a):
            self.q = [float(m.position[ix[j]]) for j in a]
        self.js_stamp = (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)

    def _on_odom(self, m):
        self.odom_t = time.monotonic()
        p = m.pose.pose.position
        self.base_xy = (float(p.x), float(p.y))
        self.base_stamp = (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)

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
        needle = {'tf': 'base_tf_bridge.py',
                  'odom': 'parameter_bridge /odom_raw',
                  'distance': 'ammr_wholebody_mpc/arm_link_distance'}[f]
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
                n_nodata=None if self.diag is None else self.diag[13],
                min_d=None if self.diag is None else self.diag[11],
                tf_age=None if (self.diag is None or len(self.diag) < 19)
                       else self.diag[18],
                base_xy=self.base_xy, q=self.q,
                base_stamp=self.base_stamp, js_stamp=self.js_stamp,
                js_age=None if self.js_t is None else time.monotonic() - self.js_t,
                odom_age=None if self.odom_t is None
                         else time.monotonic() - self.odom_t))
            if (a.recover_s > 0 and self.stopped
                    and self.t_resume is None and t > a.pre_s + a.post_s):
                for p_ in self.stopped:
                    subprocess.run(['kill', '-CONT', str(p_)])
                self.t_resume = time.monotonic()
                print(f'  t={t:5.2f}s 恢復：SIGCONT {self.stopped}', flush=True)
            if t > a.pre_s + a.post_s + a.recover_s:
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
                    choices=['none', 'jointstate', 'tf', 'odom', 'distance'])
    ap.add_argument('--cmd', nargs=9, type=float,
                    default=[0.05, 0, 0, 0, 0.05, 0, 0, 0, 0])
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--pre-s', type=float, default=4.0)
    ap.add_argument('--post-s', type=float, default=6.0)
    ap.add_argument('--out', default='',
                    help='save the per-cycle rows so the reason timeline can be '
                         'read afterwards; attribution needs it')
    ap.add_argument('--recover-s', type=float, default=0.0,
                    help='after post-s, resume the frozen feed and watch this '
                         'much longer')
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
        # Command zero and ACTUAL stop are different events, and the distance
        # between them is what a stop is worth. Measured from the feeds' own
        # timestamps, never from the loop clock.
        def motion_stop(t_from):
            seq = [r for r in R if r['t'] >= t_from and r['base_xy']
                   and r['q'] and r['base_stamp']]
            for i in range(4, len(seq)):
                a_, b_ = seq[i - 4], seq[i]
                dt_ = b_['base_stamp'] - a_['base_stamp']
                dj = b_['js_stamp'] - a_['js_stamp']
                if dt_ <= 0 or dj <= 0:
                    continue
                vb_ = math.hypot(b_['base_xy'][0] - a_['base_xy'][0],
                                 b_['base_xy'][1] - a_['base_xy'][1]) / dt_
                vq = max(abs(x - y) for x, y in zip(b_['q'], a_['q'])) / dj
                if vb_ < 2e-3 and vq < 2e-3:
                    return b_['t'], b_['base_xy'], b_['q']
            return None, None, None
        if za is not None:
            t_zero = a.pre_s + za
            r0 = next((r for r in R if r['t'] >= t_zero and r['base_xy']), None)
            t_stop, xy_stop, q_stop = motion_stop(t_zero)
            if r0 and t_stop is not None:
                dxy = math.hypot(xy_stop[0] - r0['base_xy'][0],
                                 xy_stop[1] - r0['base_xy'][1])
                dq = max(abs(x - y) for x, y in zip(q_stop, r0['q']))
                print(f"  命令歸零 → 實際停止 {(t_stop - t_zero)*1e3:.0f} ms"
                      f"；期間底盤位移 {dxy*1000:.1f} mm，最大關節位移 {dq*1e3:.1f} mrad")
            else:
                print('  命令歸零後未觀察到明確靜止（窗口不足或仍在動）')
        mv = [max(abs(x) for x in r['arm']) for r in rel if r['arm']]
        if mv:
            print(f"  故障後 手臂命令絕對值 中位 {np.median(mv):.4f} 最大 {max(mv):.4f}")
        bv = [max(abs(x) for x in r['base']) for r in rel if r['base']]
        if bv:
            print(f"  故障後 底盤命令絕對值 中位 {np.median(bv):.4f} 最大 {max(bv):.4f}")
        pre = [r for r in R if r['t'] < a.pre_s]
        for lab, seg in (('故障前', pre), ('故障後', rel)):
            av = [max(abs(x) for x in r['arm']) for r in seg if r['arm']]
            bb = [max(abs(x) for x in r['base']) for r in seg if r['base']]
            nd = [r['n_nodata'] for r in seg if r.get('n_nodata') is not None]
            if av and bb:
                print(f"  {lab}：手臂 {np.median(av):.4f}  底盤 {np.median(bb):.4f}"
                      f"  NODATA 列數 中位 {np.median(nd):.0f}" if nd else
                      f"  {lab}：手臂 {np.median(av):.4f}  底盤 {np.median(bb):.4f}")
        if n.t_resume is not None:
            t_rel = n.t_resume - n.t_fault + a.pre_s
            after = [r for r in R if r['t'] >= t_rel]
            back = next((r['t'] - t_rel for r in after
                         if r['arm'] and max(abs(x) for x in r['arm']) > 1e-9), None)
            rs = sorted({int(r['reason']) for r in after if r['reason'] is not None})
            print(f"  恢復後 → 命令再度非零 "
                  f"{('%.0f ms' % (back*1e3)) if back is not None else '未恢復'}"
                  f"；reason {[REASON.get(x, x) for x in rs]}")
        for k, lab in (('js_age', 'JointState'), ('odom_age', 'odom'),
                       ('tf_age', 'TF(安全節點回報)')):
            v = [r[k] for r in rel if r[k] is not None]
            if v:
                print(f"  {lab:18} 故障後最大年齡 {max(v)*1e3:7.0f} ms")
    if a.out:
        import json as _j
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        _j.dump(dict(args=vars(a), t_fault=n.t_fault, t_resume=n.t_resume,
                     rows=R), open(a.out, 'w'), ensure_ascii=False)
        print(f'  逐週期資料寫入 {a.out}')
    n.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
