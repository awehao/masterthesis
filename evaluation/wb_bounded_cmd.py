"""有界測試命令源：固定幅度、固定斜升、固定持續時間。

**不繞過安全層。** 發布到 `/wholebody_safety/cmd_in`，經既有
`wholebody_safety_node → arm_vel_adapter → /wb_vel_cmd` 抵達 Isaac 執行端。
本檔**不發布** `/wb_vel_cmd`。

命令剖面（四段，時間全部事前固定）：

    零命令 → 斜升到低速 → 保持 → 斜降回零 → 零命令 → **停止發布**

`base` 模式的六個手臂分量**恆為零**；`arm` 模式的三個底盤分量恆為零。
不符模式的命令由接收端整筆拒收，本檔不負責裁切 —— 但本檔本來就不產生
不符模式的分量，兩邊獨立。

逾時測試的注意事項
------------------
**本檔停止發布，不一定讓 `/wb_vel_cmd` 停止更新**：安全層在命令過期後
會持續輸出零，adapter 也就繼續發。若 `/wb_vel_cmd` 仍在更新，
那只驗證了**上游的失效處置**，**不算接收端逾時測試**。
要測接收端逾時，必須明確中斷它的直接命令來源（停掉 adapter）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True, choices=['base', 'arm', 'sync'])
ap.add_argument('--out', required=True)
ap.add_argument('--rate-hz', type=float, default=20.0)
# --- 剖面：幅度、斜升、持續時間**事前固定** ---
ap.add_argument('--zero-lead-s', type=float, default=2.0, help='起始零命令')
ap.add_argument('--ramp-s', type=float, default=1.0, help='斜升／斜降各自長度')
ap.add_argument('--hold-s', type=float, default=3.0, help='低速保持')
ap.add_argument('--zero-tail-s', type=float, default=2.0, help='結束零命令')
ap.add_argument('--base-vx', type=float, default=0.03,
                help='底盤本體 +x 速度（m/s）。低速：執行端界限 0.05')
ap.add_argument('--arm-rate', type=float, default=0.05,
                help='joint2 速度（rad/s）。低速：執行端界限 1.0')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

import rclpy                                                   # noqa: E402
from rclpy.node import Node                                    # noqa: E402
from std_msgs.msg import Float64MultiArray                     # noqa: E402
from rosgraph_msgs.msg import Clock                            # noqa: E402

TOPIC = '/wholebody_safety/cmd_in'
T_LEAD = a.zero_lead_s
T_UP = T_LEAD + a.ramp_s
T_HOLD = T_UP + a.hold_s
T_DOWN = T_HOLD + a.ramp_s
T_END = T_DOWN + a.zero_tail_s


def profile(t):
    """回傳 (scale, 段名)；scale ∈ [0, 1]，斜升／斜降為線性。"""
    if t < T_LEAD:
        return 0.0, 'zero_lead'
    if t < T_UP:
        return (t - T_LEAD) / a.ramp_s, 'ramp_up'
    if t < T_HOLD:
        return 1.0, 'hold'
    if t < T_DOWN:
        return 1.0 - (t - T_HOLD) / a.ramp_s, 'ramp_down'
    if t < T_END:
        return 0.0, 'zero_tail'
    return None, 'stopped'


def vector(scale):
    v = [0.0] * 9
    if a.mode in ('base', 'sync'):
        v[0] = a.base_vx * scale          # 本體 +x
    if a.mode in ('arm', 'sync'):
        v[4] = a.arm_rate * scale         # joint2
    return v


class Src(Node):
    def __init__(self):
        super().__init__('wb_bounded_cmd')
        self.pub = self.create_publisher(Float64MultiArray, TOPIC, 10)
        self.sim_t = None
        self.create_subscription(Clock, '/clock', self._clk, 10)
        self.sent = []
        self.t0 = None
        self.done = False

    def _clk(self, m):
        self.sim_t = m.clock.sec + m.clock.nanosec * 1e-9

    def tick(self):
        if self.sim_t is None or self.done:
            return
        if self.t0 is None:
            self.t0 = self.sim_t
            print(f'[cmd] 起算 sim {self.t0:.3f}；剖面總長 {T_END:.1f} s',
                  flush=True)
        el = self.sim_t - self.t0
        scale, seg = profile(el)
        if scale is None:
            self.done = True
            print(f'[cmd] **停止發布** @ sim {self.sim_t:.3f}'
                  f'（已發 {len(self.sent)} 則）', flush=True)
            return
        v = vector(scale)
        self.pub.publish(Float64MultiArray(data=v))
        self.sent.append([round(self.sim_t, 4), round(el, 4), seg,
                          round(scale, 6)] + [round(x, 6) for x in v])


def main():
    rclpy.init()
    n = Src()
    print(f'[cmd] 模式 {a.mode} → {TOPIC}（**不直接發 /wb_vel_cmd**）', flush=True)
    print(f'[cmd] 剖面 零{a.zero_lead_s}s → 斜升{a.ramp_s}s → 保持{a.hold_s}s'
          f' → 斜降{a.ramp_s}s → 零{a.zero_tail_s}s → 停止發布', flush=True)
    print(f'[cmd] 幅度 base_vx {a.base_vx} m/s、arm_rate {a.arm_rate} rad/s',
          flush=True)
    period = 1.0 / a.rate_hz
    nxt = time.monotonic()
    w0 = time.monotonic()
    while rclpy.ok() and not n.done and time.monotonic() - w0 < 300.0:
        rclpy.spin_once(n, timeout_sec=0.005)
        if time.monotonic() >= nxt:
            n.tick()
            nxt += period
    json.dump({'schema': 'wb_bounded_cmd/1', 'mode': a.mode, 'topic': TOPIC,
               'profile_s': {'zero_lead': a.zero_lead_s, 'ramp': a.ramp_s,
                             'hold': a.hold_s, 'zero_tail': a.zero_tail_s,
                             'total': T_END},
               'amplitude': {'base_vx_mps': a.base_vx,
                             'arm_rate_rps': a.arm_rate},
               'rate_hz': a.rate_hz, 'n_sent': len(n.sent),
               'cols': ['sim_t', 'elapsed', 'segment', 'scale'] +
                       [f'v{i}' for i in range(9)],
               'sent': n.sent,
               'note': ('本檔停止發布不等於 /wb_vel_cmd 停止更新；'
                        '安全層在命令過期後會持續輸出零。接收端逾時測試'
                        '須另外中斷 adapter')},
              open(os.path.join(a.out, 'cmd_source.json'), 'w'),
              ensure_ascii=False)
    print(f'[cmd] -> {os.path.join(a.out, "cmd_source.json")}', flush=True)
    rclpy.try_shutdown()
    return 0


sys.exit(main())
