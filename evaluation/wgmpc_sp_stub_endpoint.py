#!/usr/bin/env python3
"""設定點介面的**替身執行端**（無 Isaac、無 GPU）。

只為了測介面：發 /clock、/joint_states、/odom、/coman/arm_setpoint(+meta)、
/coman/applied_cmd。**不含任何物理** —— 它不是 Isaac 的替代品，
到達與保持不能用它判定。

可注入的反例由命令列控制：
  --ready-after S      S 秒（模擬時間）後才回報 ready
  --sp-offset-steps K  設定點的 sim_t 故意偏 K 個物理步（測同時刻配對）
  --fail-at S          S 秒後 exec_mode = 3（失效閂鎖）
  --api-false-at S     S 秒後 api_applied = False
  --sp-freeze-at S     S 秒後停發設定點（測 HOLD）
"""
from __future__ import annotations

import argparse
import json


import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
import tf2_ros

ARM = [f'joint{i}' for i in range(1, 7)]
PHYS_DT = 0.01
SP_COLS = (['physics_step_id', 'sim_t', 'ready']
           + [f'sp_{j}' for j in ARM] + ['exec_mode_code', 'api_applied'])


class Stub(Node):
    def __init__(self, a):
        super().__init__('wgmpc_sp_stub')
        self.a = a
        self.t = 0.0
        self.step = 0
        self.q = [0.0] * 6
        self.sp = [0.0] * 6
        self.sp_ready = False
        self.u_arm = [0.0] * 6
        self.clock = self.create_publisher(Clock, '/clock', 10)
        self.js = self.create_publisher(JointState, '/joint_states', 10)
        self.od = self.create_publisher(Odometry, '/odom', 10)
        self.spp = self.create_publisher(Float64MultiArray,
                                         '/coman/arm_setpoint', 10)
        self.spm = self.create_publisher(
            String, '/coman/arm_setpoint_meta',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.ap = self.create_publisher(Float64MultiArray,
                                        '/coman/applied_cmd', 10)
        self.apm = self.create_publisher(
            String, '/coman/applied_cmd_meta',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.tfb = tf2_ros.TransformBroadcaster(self)
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd',
                                 self._on_cmd, 10)
        self.n_cmd = 0
        self.n_nonzero_cmd = 0
        self.meta_sent = False
        self.create_timer(PHYS_DT, self._tick)

    def _on_cmd(self, m):
        if len(m.data) >= 9:
            self.u_arm = [float(x) for x in m.data[3:9]]
            self.n_cmd += 1
            if any(abs(float(x)) > 1e-12 for x in m.data):
                self.n_nonzero_cmd += 1

    def _tick(self):
        self.t += PHYS_DT
        self.step += 1
        t = self.t
        c = Clock()
        c.clock.sec = int(t)
        c.clock.nanosec = min(int(round((t - int(t)) * 1e9)), 999999999)
        self.clock.publish(c)
        stamp = c.clock
        # 設定點：ready 之後才建立，之後每物理步積分命令
        if (not self.sp_ready) and t >= self.a.ready_after:
            self.sp_ready = True
            self.sp = list(self.q)          # setpoint_init：由實測關節建立
        if self.sp_ready:
            self.sp = [p + r * PHYS_DT for p, r in zip(self.sp, self.u_arm)]
            # 一階追蹤（係數與 free4 辨識同量級，只為讓狀態會動）
            self.q = [x + 0.095 * (s - x) for x, s in zip(self.q, self.sp)]
        js = JointState()
        js.header.stamp = stamp
        js.name = list(ARM)
        js.position = [float(x) for x in self.q]
        js.velocity = [0.0] * 6
        self.js.publish(js)
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = 'odom'
        o.child_frame_id = 'base_footprint'
        o.pose.pose.orientation.w = 1.0
        self.od.publish(o)
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_footprint'
        tf.transform.rotation.w = 1.0
        self.tfb.sendTransform(tf)
        mode = 3 if (self.a.fail_at >= 0 and t >= self.a.fail_at) else 0
        api = not (self.a.api_false_at >= 0 and t >= self.a.api_false_at)
        # applied 回報（節點的 strict u_prev 需要它）
        am = Float64MultiArray()
        am.data = ([float(self.step), float(t), 0.0, 0.0, 0.0]
                   + [float(x) for x in self.u_arm]
                   + [float(mode), 0.01, float(self.n_cmd), 0.0,
                      1.0 if api else 0.0, float(self.n_cmd), float(t)])
        self.ap.publish(am)
        if not self.meta_sent:
            self.meta_sent = True
            self.apm.publish(String(data=json.dumps({'cols': [
                'physics_step_id', 'sim_t', 'bvx_body', 'bvy_body', 'wz',
                'qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6', 'exec_mode_code',
                'cmd_age_s', 'n_recv', 'n_rejected', 'api_applied',
                'src_recv_seq', 'src_recv_sim_t']}, ensure_ascii=False)))
            self.spm.publish(String(data=json.dumps({
                'cols': SP_COLS, 'joint_order': list(ARM),
                'sampling_instant': '**本步寫入後**（apply_action 之後）',
                'physics_dt_s': float(PHYS_DT),
            }, ensure_ascii=False)))
        if self.a.sp_freeze_at >= 0 and t >= self.a.sp_freeze_at:
            return                       # 停發設定點 ⇒ 節點應進 HOLD
        sm = Float64MultiArray()
        # **故意偏移 sim_t** 以測同時刻配對
        t_sp = t + self.a.sp_offset_steps * PHYS_DT
        sm.data = ([float(self.step), float(t_sp),
                    1.0 if self.sp_ready else 0.0]
                   + [float(x) if self.sp_ready else float('nan')
                      for x in self.sp]
                   + [float(mode), 1.0 if api else 0.0])
        self.spp.publish(sm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ready-after', type=float, default=0.3)
    ap.add_argument('--sp-offset-steps', type=int, default=0)
    ap.add_argument('--fail-at', type=float, default=-1.0)
    ap.add_argument('--api-false-at', type=float, default=-1.0)
    ap.add_argument('--sp-freeze-at', type=float, default=-1.0)
    ap.add_argument('--run-s', type=float, default=12.0)
    a = ap.parse_args()
    rclpy.init()
    nd = Stub(a)
    import time as _t
    t0 = _t.monotonic()
    while rclpy.ok() and _t.monotonic() - t0 < a.run_s:
        rclpy.spin_once(nd, timeout_sec=0.05)
    print(json.dumps({'n_cmd': nd.n_cmd, 'n_nonzero_cmd': nd.n_nonzero_cmd,
                      'sp_ready': nd.sp_ready, 'sim_t': round(nd.t, 3)},
                     ensure_ascii=False), flush=True)
    nd.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
