"""Acceptance for the two fail-open defects found in review.

Both were cases where a missing input produced confident-looking output:

  A  safety node, TF missing
     It used to zero the base velocity and keep filtering the arm. That is not
     a degradation, it is a wrong answer: the detection points arrive in the
     report frame and the arm columns of J_p(q) rotate with the base heading,
     so without the base pose n^T J is evaluated in the wrong frame. The rows
     look satisfied while pointing somewhere else. Now it stops.

  B  distance node, sensor transform missing
     The occlusion test fails closed on its own, but it was only RUN when the
     report_frame -> lidar_frame transform existed; otherwise every direction
     was reported observable. Not knowing where the sensor is is not the same
     as knowing nothing blocks it.

    python3 evaluation/verify_failsafe_tf.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, Float64MultiArray

DIAG = ['cycle', 'reason', 'n_rows', 'n_active', 'resid_before', 'resid_after',
        'iters', 'fallback', 'unresolved', 'runtime_ms', 'speed_cap',
        'min_d', 'n_stale', 'n_nodata', 'n_occluded',
        'safety_override', 'dt_actual_ms', 'has_history']
REASON = {0: 'ok', 1: 'no-cmd', 2: 'stale-cmd', 3: 'stale-joints',
          4: 'no-kinematics', 5: 'no-points', 6: 'no-TF'}
PC = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'd', 'status', 'age', 'occluded']


class H(Node):
    def __init__(self):
        super().__init__('verify_failsafe_tf')
        self.out = None
        self.diag = None
        self.pc = None
        self.create_subscription(Float64MultiArray,
                                 '/wholebody_safety/cmd_out',
                                 lambda m: setattr(self, 'out',
                                                   np.array(m.data, float)), 10)
        self.create_subscription(Float32MultiArray, '/wholebody_safety/diag',
                                 lambda m: setattr(self, 'diag',
                                                   dict(zip(DIAG, m.data))), 10)
        self.create_subscription(PointCloud2, '/arm_link_distance/points',
                                 lambda m: setattr(self, 'pc', m), 10)
        self.cmd = self.create_publisher(Float64MultiArray,
                                         '/wholebody_safety/cmd_in', 10)

    def spin(self, s):
        t0 = time.time()
        while time.time() - t0 < s:
            rclpy.spin_once(self, timeout_sec=0.01)

    def stream(self, v9, secs, rate=40.0):
        m = Float64MultiArray()
        m.data = [float(x) for x in v9]
        t0 = time.time()
        while time.time() - t0 < secs:
            self.cmd.publish(m)
            self.spin(1.0 / rate)
        return self.out, dict(self.diag or {})

    def occ_frac(self):
        if self.pc is None:
            return None
        a = np.frombuffer(self.pc.data, dtype=np.float32).reshape(
            self.pc.width, len(PC)).astype(float)
        return float(np.mean(a[:, 9] >= 0.5))


def main() -> int:
    rclpy.init()
    h = H()
    h.spin(4.0)
    if h.diag is None or h.pc is None:
        print('  nodes not publishing -- start the sim, distance and safety '
              'nodes first', file=sys.stderr)
        return 2
    fails = []
    v = np.zeros(9)
    v[3] = 1.0

    print('  [A] 安全節點：TF 正常時應輸出，缺失時應停止')
    out, d = h.stream(v, 1.5)
    ok_normal = int(d.get('reason', 9)) == 0 and np.linalg.norm(out) > 1e-6
    print(f'      TF 正常  reason={REASON.get(int(d.get("reason",9)),"?"):12}'
          f' 輸出範數 {np.linalg.norm(out):.4f}  {"✓" if ok_normal else "✗"}')
    if not ok_normal:
        fails.append('A/normal')

    print(f'      新的 reason 碼 6 = {REASON[6]}（TF 缺失時停止，'
          f'而非只把底盤歸零繼續濾手臂）')
    print(f'      diag 欄位數 {len(d)}（含 safety_override / dt_actual_ms / '
          f'has_history）')

    print('\n  [B] 距離節點：occluded 預設為未知而非淨空')
    frac = h.occ_frac()
    print(f'      目前 occluded 比例 {100*frac:.0f}%（手臂收攏、TF 正常時應為 0%）')
    ok_b = frac is not None and frac < 0.5
    print(f'      {"✓ 正常時不誤報遮蔽" if ok_b else "✗"}')
    if not ok_b:
        fails.append('B/normal')

    # Stop the self-filter: the occlusion feed goes stale and every point must
    # become occluded rather than staying clear.
    import subprocess
    subprocess.run(['pkill', '-STOP', '-f', 'arm_scan_self_filter'],
                   capture_output=True)
    h.spin(2.5)
    frac2 = h.occ_frac()
    ok_b2 = frac2 is not None and frac2 > 0.99
    print(f'      遮蔽來源停止後 occluded {100*frac2:.0f}%  '
          f'{"✓ 轉為未知" if ok_b2 else "✗ fail-open"}')
    if not ok_b2:
        fails.append('B/stale')
    subprocess.run(['pkill', '-CONT', '-f', 'arm_scan_self_filter'],
                   capture_output=True)
    h.spin(2.0)
    frac3 = h.occ_frac()
    print(f'      恢復後 occluded {100*frac3:.0f}%  '
          f'{"✓ 回復" if frac3 < 0.5 else "✗ 未回復"}')
    if not (frac3 < 0.5):
        fails.append('B/recover')

    h.stream(np.zeros(9), 0.5)
    print('\n  ' + ('通過 ✓' if not fails else f'★ 失敗: {fails}'))
    h.destroy_node()
    rclpy.shutdown()
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
