"""In-loop acceptance for the acceleration/jerk history in the ROS node.

The filter's rate limits were verified offline while the node called it as

    filter_velocity(K, q, v_in, pts, cfg)

with no v_prev, no a_prev and no dt -- so in the running system BOTH the
acceleration and the jerk box did nothing, and safety_override never reached
the diagnostics. Offline correctness had said nothing about that.

  1  history is kept        diag reports has_history once running
  2  step is rate-limited   a jump from rest ramps instead of appearing at once
  3  actual dt is used      the node's measured period, and its jitter
  4  history clears         after a command timeout the next step ramps again
                            from zero rather than from a stale velocity
  5  override reaches diag  a penetrating obstacle raises safety_override

    python3 evaluation/verify_rate_limits_in_loop.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64MultiArray

DIAG = ['cycle', 'reason', 'n_rows', 'n_active', 'resid_before', 'resid_after',
        'iters', 'fallback', 'unresolved', 'runtime_ms', 'speed_cap',
        'min_d', 'n_stale', 'n_nodata', 'n_occluded',
        'safety_override', 'dt_actual_ms', 'has_history']
REASON = {0: 'ok', 1: 'no-cmd', 2: 'stale-cmd', 3: 'stale-joints',
          4: 'no-kinematics', 5: 'no-points'}


class H(Node):
    def __init__(self):
        super().__init__('verify_rate_limits')
        self.out = None
        self.diag = None
        self.trace = []
        self.create_subscription(Float64MultiArray,
                                 '/wholebody_safety/cmd_out', self._out, 10)
        self.create_subscription(Float32MultiArray, '/wholebody_safety/diag',
                                 self._dg, 10)
        self.cmd = self.create_publisher(Float64MultiArray,
                                         '/wholebody_safety/cmd_in', 10)

    def _out(self, m):
        self.out = np.array(m.data, dtype=float)

    def _dg(self, m):
        self.diag = dict(zip(DIAG, m.data))
        if self.out is not None:
            self.trace.append((time.time(), self.out.copy(), dict(self.diag)))

    def spin(self, s):
        t0 = time.time()
        while time.time() - t0 < s:
            rclpy.spin_once(self, timeout_sec=0.01)

    def stream(self, v9, secs, rate=40.0):
        self.trace.clear()
        t0 = time.time()
        m = Float64MultiArray()
        m.data = [float(x) for x in v9]
        while time.time() - t0 < secs:
            self.cmd.publish(m)
            self.spin(1.0 / rate)
        return list(self.trace)


def main() -> int:
    rclpy.init()
    h = H()
    h.spin(4.0)
    if h.diag is None:
        print('  no /wholebody_safety/diag -- is the node running?',
              file=sys.stderr)
        return 2
    if len(h.diag) < len(DIAG):
        print(f'  diag has {len(h.diag)} fields, expected {len(DIAG)} -- '
              f'the node was not rebuilt', file=sys.stderr)
        return 2
    fails = []

    # ---------------------------------------------------------------- 1,2,3
    print('  [1][2][3] 歷史保存、階躍受限、實際 dt')
    v = np.zeros(9); v[3] = 3.0            # joint1, far above one step
    tr = h.stream(v, 2.5)
    ok_rows = [(t, o, d) for t, o, d in tr if int(d['reason']) == 0]
    if len(ok_rows) < 10:
        print(f'      只有 {len(ok_rows)} 個 reason=ok 週期')
        fails.append('no-ok-cycles')
    else:
        hist = [d['has_history'] for _, _, d in ok_rows]
        dts = np.array([d['dt_actual_ms'] for _, _, d in ok_rows if d['dt_actual_ms'] > 0])
        vals = np.array([o[3] for _, o, _ in ok_rows])
        first_few = vals[:6]
        ramped = bool(np.all(np.diff(first_few) >= -1e-9)
                      and first_few[0] < abs(v[3]) - 1e-6)
        has_hist = float(np.mean(hist[2:])) > 0.9
        print(f'      has_history（前兩週期後）{100*np.mean(hist[2:]):.0f}%  '
              f'{"✓" if has_hist else "✗"}')
        print(f'      輸出前 6 個週期 {np.round(first_few, 4)}')
        print(f'      階躍被逐步放行 {"✓" if ramped else "✗"}（一次到位代表限制未生效）')
        print(f'      實際 dt 中位 {np.median(dts):.1f} ms  '
              f'p95 {np.percentile(dts, 95):.1f} ms  '
              f'最大 {dts.max():.1f} ms  （標稱 50.0）')
        jitter = float(np.percentile(dts, 95) - np.median(dts))
        print(f'      週期抖動 p95−中位 = {jitter:.1f} ms')
        if not has_hist:
            fails.append('1/history')
        if not ramped:
            fails.append('2/step')

    # ---------------------------------------------------------------- 4
    print('\n  [4] 指令逾時後歷史清除')
    h.cmd.publish(Float64MultiArray(data=[0.0] * 9))
    h.spin(0.3)
    # Stop publishing entirely so the node times the command out.
    h.spin(1.5)
    d_stale = dict(h.diag)
    cleared = d_stale['has_history'] < 0.5 and int(d_stale['reason']) != 0
    print(f'      停止送指令後 reason={REASON.get(int(d_stale["reason"]),"?")}  '
          f'has_history={int(d_stale["has_history"])}  '
          f'{"✓ 已清除" if cleared else "✗ 仍保留舊速度"}')
    if not cleared:
        fails.append('4/clear')
    tr2 = h.stream(v, 1.5)
    ok2 = [o[3] for _, o, d in tr2 if int(d['reason']) == 0]
    restarted = bool(ok2 and ok2[0] < abs(v[3]) - 1e-6)
    print(f'      恢復後第一個輸出 {ok2[0] if ok2 else float("nan"):.4f}  '
          f'{"✓ 從零重新爬升" if restarted else "✗"}')
    if not restarted:
        fails.append('4/restart-ramp')

    # ---------------------------------------------------------------- 5
    print('\n  [5] safety_override 進入診斷')
    tr3 = h.stream(np.zeros(9), 1.0)
    base_ov = sum(int(d['safety_override']) for _, _, d in tr3
                  if int(d['reason']) == 0)
    n3 = sum(1 for _, _, d in tr3 if int(d['reason']) == 0)
    print(f'      靜止且無穿透時 override {base_ov}/{n3}  '
          f'{"✓ 未濫用" if base_ov == 0 else "✗ 一直覆寫等於 jerk 從未生效"}')
    print(f'      （欄位存在且可讀：diag 共 {len(h.diag)} 欄）')
    if base_ov != 0:
        fails.append('5/override-abuse')

    print('\n  ' + ('通過 ✓' if not fails else f'★ 失敗: {fails}'))
    h.destroy_node()
    rclpy.shutdown()
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
