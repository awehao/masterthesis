"""Minimal base input/output test: reach steady state, then stop. Two ways.

Why it is being redone
----------------------
The earlier stopping numbers were reported alongside an explanation that has now
been withdrawn: that gz's VelocityControl writes the body velocity directly and
therefore has no deceleration dynamics, so a zero command stops the base
instantly. Measurement says otherwise -- a held 0.10 m/s command produces about
0.030 m/s of actual motion, so the physics step is clearly doing something to
the commanded velocity. The observed zero displacement stands as an observation;
its CAUSE and the resolution it was measured at do not.

Every event gets its own timestamp, from the feed that carries it:

    last non-zero command     when the driver last published a moving command
    zero command published    when the zero (or the silence) began
    velocity decay            measured from odom, differenced on odom's stamps
    motion stopped            first sample below the stop threshold
    displacement              between each pair of the above

Measurement resolution is recorded with the result, because "0.00 mm" from a
feed that quantises at 1 mm and samples at 30 Hz is a different claim from the
same number out of a finer one.

This doubles as the frozen input/output record for the Isaac port: fixed scene,
fixed initial state, fixed physics step and publish rate, command and measured
trajectory both saved. Isaac re-runs the same INPUT and its response is compared;
it is not tuned to reproduce this simulator's numbers.

    python3 evaluation/gz_base_stop_test.py --mode zero
    python3 evaluation/gz_base_stop_test.py --mode silence
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class StopTest(Node):

    def __init__(self, a):
        super().__init__('gz_base_stop_test')
        self.a = a
        self.odom = []          # (stamp, x, y, wall)
        self.cmd = []           # (wall, vx)
        self.create_subscription(Odometry, '/odom', self._on_odom,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def _on_odom(self, m):
        self.odom.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                          float(m.pose.pose.position.x),
                          float(m.pose.pose.position.y),
                          time.monotonic()))

    def _pub(self, vx):
        t = Twist()
        t.linear.x = float(vx)
        self.pub.publish(t)
        self.cmd.append((time.monotonic(), float(vx)))

    def run(self):
        a = self.a
        dt = 1.0 / a.rate
        t0 = time.monotonic()
        nxt = t0
        # phase 1: drive to steady state
        while time.monotonic() - t0 < a.drive_s:
            rclpy.spin_once(self, timeout_sec=0.002)
            if time.monotonic() >= nxt:
                self._pub(a.vx)
                nxt += dt
        self.t_last_nonzero = time.monotonic()
        # phase 2: stop, one way or the other
        self.t_zero = None
        while time.monotonic() - t0 < a.drive_s + a.watch_s:
            rclpy.spin_once(self, timeout_sec=0.002)
            if a.mode == 'zero' and time.monotonic() >= nxt:
                if self.t_zero is None:
                    self.t_zero = time.monotonic()
                self._pub(0.0)
                nxt += dt
        if a.mode == 'silence':
            self.t_zero = self.t_last_nonzero      # publication simply ceased


def analyse(n, a) -> dict:
    O = np.array([(s, x, y, w) for s, x, y, w in n.odom if s > 0])
    st, x, y, wall = O[:, 0], O[:, 1], O[:, 2], O[:, 3]
    # speed differenced on ODOM's own stamps, never on the loop clock
    v = np.hypot(np.diff(x), np.diff(y)) / np.diff(st)
    tv = st[1:]
    # resolution of the measurement itself
    d_stamp = np.diff(st)
    step = np.abs(np.diff(x))
    step = step[step > 0]
    res = dict(
        odom_hz=float(len(st) / (st[-1] - st[0])),
        stamp_dt_p50=float(np.median(d_stamp)),
        stamp_dt_max=float(d_stamp.max()),
        pos_min_step_mm=float(step.min() * 1000) if len(step) else None,
        speed_floor_mm_s=float(step.min() / np.median(d_stamp) * 1000)
        if len(step) else None)

    # map wall-clock events onto the odom stamp axis
    def wall_to_stamp(w):
        i = int(np.argmin(np.abs(wall - w)))
        return float(st[i]), i
    s_nz, i_nz = wall_to_stamp(n.t_last_nonzero)
    s_z, i_z = wall_to_stamp(n.t_zero)
    steady = v[(tv > s_nz - a.steady_s) & (tv < s_nz)]
    # The threshold cannot be finer than what the feed can resolve. /odom here
    # samples at ~30 Hz and quantises position at ~0.39 mm, so speeds below
    # about 11.6 mm/s are indistinguishable from zero; asking whether the base
    # crossed 2 mm/s is a question this measurement cannot answer, and a "stop"
    # declared at that level would be an artefact. The threshold is therefore
    # raised to the resolvable floor and reported as such.
    floor = res['speed_floor_mm_s'] / 1000.0 if res['speed_floor_mm_s'] else 0.0
    thr = max(a.stop_v, floor * a.floor_k)
    out_thr = dict(requested=a.stop_v, floor=floor, used=thr,
                   limited_by_resolution=bool(thr > a.stop_v))
    after = np.flatnonzero((tv >= s_z) & (v < thr))
    i_stop = int(after[0]) + 1 if len(after) else None
    out = dict(
        mode=a.mode, cmd_vx=a.vx, rate=a.rate,
        steady_v=float(np.median(steady)) if len(steady) else None,
        tracking=float(np.median(steady) / abs(a.vx)) if len(steady) else None,
        t_last_nonzero=s_nz, t_zero_cmd=s_z, resolution=res,
        threshold=out_thr)
    if i_stop is not None:
        out['t_stop'] = float(st[i_stop])
        out['zero_to_stop_s'] = float(st[i_stop] - s_z)
        out['disp_zero_to_stop_mm'] = float(
            np.hypot(x[i_stop] - x[i_z], y[i_stop] - y[i_z]) * 1000)
        out['disp_nonzero_to_stop_mm'] = float(
            np.hypot(x[i_stop] - x[i_nz], y[i_stop] - y[i_nz]) * 1000)
    else:
        out['t_stop'] = None
    out['traj'] = [[float(s), float(px), float(py)] for s, px, py in
                   zip(st, x, y)]
    out['cmd'] = [[float(w), float(c)] for w, c in n.cmd]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='zero', choices=['zero', 'silence'])
    ap.add_argument('--vx', type=float, default=-0.10)
    ap.add_argument('--rate', type=float, default=50.0)
    ap.add_argument('--drive-s', type=float, default=8.0)
    ap.add_argument('--watch-s', type=float, default=5.0)
    ap.add_argument('--steady-s', type=float, default=3.0)
    ap.add_argument('--stop-v', type=float, default=2e-3)
    ap.add_argument('--floor-k', type=float, default=1.0,
                    help='stop threshold = max(stop_v, floor_k x resolvable floor)')
    ap.add_argument('--world', default='arm_barrier_test')
    ap.add_argument('--start', nargs=3, type=float, default=[0.0, 0.0, 0.0])
    ap.add_argument('--out', default='evaluation/results/gz_base_stop')
    a = ap.parse_args()

    subprocess.run(
        ['gz', 'service', '-s', f'/world/{a.world}/set_pose',
         '--reqtype', 'gz.msgs.Pose', '--reptype', 'gz.msgs.Boolean',
         '--timeout', '5000', '--req',
         f'name: "omni_bot", position: {{x: {a.start[0]}, y: {a.start[1]}, '
         f'z: {a.start[2]}}}, orientation: {{x:0,y:0,z:0,w:1}}'],
        capture_output=True, timeout=30)
    time.sleep(4)

    rclpy.init()
    n = StopTest(a)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10 and not n.odom:
        rclpy.spin_once(n, timeout_sec=0.05)
    if not n.odom:
        print('  沒有 /odom，先確認橋接'); rclpy.try_shutdown(); return 1
    print(f'  模式 {a.mode}：先以 {a.vx:+.3f} m/s 驅動 {a.drive_s:.0f} s 達穩態，'
          f'再' + ('發布零命令' if a.mode == 'zero' else '停止發布'), flush=True)
    n.run()
    r = analyse(n, a)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    path = f'{a.out}_{a.mode}.json'
    json.dump(r, open(path, 'w'), ensure_ascii=False)

    print(f"\n  穩態速度 {r['steady_v']:.4f} m/s   命令 {abs(a.vx):.4f}"
          f"   追蹤率 {r['tracking']*100:.1f}%")
    print(f"  量測解析度：odom {r['resolution']['odom_hz']:.1f} Hz，"
          f"時間戳間隔中位 {r['resolution']['stamp_dt_p50']*1e3:.1f} ms，"
          f"位置最小步進 {r['resolution']['pos_min_step_mm']:.4f} mm")
    th = r['threshold']
    print(f"                → 可分辨的最小速度約 "
          f"{r['resolution']['speed_floor_mm_s']:.2f} mm/s")
    print(f"  停止門檻：要求 {th['requested']*1000:.1f} mm/s，"
          f"實際採用 {th['used']*1000:.2f} mm/s"
          + ("（**受解析度限制而提高**）" if th['limited_by_resolution'] else ""))
    print(f"\n  最後非零命令 t={r['t_last_nonzero']:.3f}")
    print(f"  零命令／停止發布 t={r['t_zero_cmd']:.3f}")
    if r['t_stop'] is not None:
        print(f"  降到門檻以下 t={r['t_stop']:.3f}（零命令後 "
              f"{r['zero_to_stop_s']*1e3:.0f} ms）"
              + ("——在此取樣下與『停止』不可分辨"
                 if r['threshold']['limited_by_resolution'] else ""))
        print(f"  位移：零命令 → 停止 {r['disp_zero_to_stop_mm']:.2f} mm；"
              f"最後非零命令 → 停止 {r['disp_nonzero_to_stop_mm']:.2f} mm")
    else:
        print(f"  觀測窗內未低於停止門檻")
    print(f"\n  逐筆命令與實測軌跡已存 {path}")
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
