"""有界測試命令源：**階躍＋反向**，用來讓輪級限制實際觸發。

與 `wb_bounded_cmd.py`（斜坡剖面）並存，不改動該檔。

**為什麼要換剖面**：斜坡剖面在本階段幅度下，每個物理步的命令變化約
0.0003 m/s，遠低於單步輪級預算 0.0625 m/s —— λ 永遠是 1，
**跑再多趟也驗不到限制器**。

**幅度上限完全不動**：`vx = 0.045 ≤ 0.05`、`wz = 0.18 ≤ 0.2`
（執行端低速介面界限）、`dq2 = 0.05 rad/s`。改的是**剖面形狀**。
輪列值 |vx| + L|wz| = 0.045 + 0.245×0.18 = 0.0891 m/s；
反向時的變化 0.1782 m/s > 0.0625，必定觸發加速度限制。

座標約定與 `wb_bounded_cmd.py` 相同：底盤三分量在安全層 `report_frame`，
由 `arm_vel_adapter` 轉成本體座標後才到執行端。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--rate-hz', type=float, default=20.0)
ap.add_argument('--zero-lead-s', type=float, default=2.0)
ap.add_argument('--half-period-s', type=float, default=0.5)
ap.add_argument('--cycles', type=int, default=3)
ap.add_argument('--zero-tail-s', type=float, default=2.0)
ap.add_argument('--base-vx', type=float, default=0.045, help='≤ 0.05')
ap.add_argument('--base-wz', type=float, default=0.18, help='≤ 0.2')
ap.add_argument('--arm-rate', type=float, default=0.05)
ap.add_argument('--expect-base-yaw-deg', type=float, default=None)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

assert abs(a.base_vx) <= 0.05 + 1e-9, '不得超過低速介面界限 lin 0.05'
assert abs(a.base_wz) <= 0.2 + 1e-9, '不得超過低速介面界限 ang 0.2'

import rclpy                                                   # noqa: E402
from rclpy.node import Node                                    # noqa: E402
from std_msgs.msg import Float64MultiArray                     # noqa: E402
from rosgraph_msgs.msg import Clock                            # noqa: E402

TOPIC = '/wholebody_safety/cmd_in'
T_SQ = a.zero_lead_s + 2.0 * a.half_period_s * a.cycles
T_END = T_SQ + a.zero_tail_s


def profile(t):
    if t < a.zero_lead_s:
        return 0.0, 'zero_lead'
    if t < T_SQ:
        k = int((t - a.zero_lead_s) // a.half_period_s)
        return (1.0 if k % 2 == 0 else -1.0), f'step{k}'
    if t < T_END:
        return 0.0, 'zero_tail'
    return None, 'stopped'


class Src(Node):
    def __init__(self):
        super().__init__('wb_trigger_cmd')
        self.pub = self.create_publisher(Float64MultiArray, TOPIC, 10)
        self.sim_t = None
        self.create_subscription(Clock, '/clock', self._clk, 10)
        self.sent, self.t0, self.done = [], None, False

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
        sc, seg = profile(el)
        if sc is None:
            self.done = True
            print(f'[cmd] **停止發布** @ sim {self.sim_t:.3f}'
                  f'（已發 {len(self.sent)} 則）', flush=True)
            return
        v = [0.0] * 9
        v[0] = a.base_vx * sc
        v[2] = a.base_wz * sc
        v[4] = a.arm_rate * sc
        self.pub.publish(Float64MultiArray(data=v))
        self.sent.append([round(self.sim_t, 4), round(el, 4), seg, sc]
                         + [round(x, 6) for x in v])


def main():
    rclpy.init()
    n = Src()
    print(f'[cmd] **階躍＋反向**剖面 → {TOPIC}', flush=True)
    print(f'[cmd] 零{a.zero_lead_s}s → {a.cycles} 個週期'
          f'（每半週期 {a.half_period_s}s 反向）→ 零{a.zero_tail_s}s → 停止發布',
          flush=True)
    print(f'[cmd] 幅度 vx {a.base_vx}（界限 0.05）、wz {a.base_wz}（界限 0.2）、'
          f'joint2 {a.arm_rate} rad/s —— **界限未動，只改剖面形狀**', flush=True)
    if a.expect_base_yaw_deg is not None:
        print(f'[cmd] 起始底盤 yaw {a.expect_base_yaw_deg:+.3f}°', flush=True)
    period = 1.0 / a.rate_hz
    nxt = time.monotonic()
    w0 = time.monotonic()
    while rclpy.ok() and not n.done and time.monotonic() - w0 < 300.0:
        rclpy.spin_once(n, timeout_sec=0.005)
        if time.monotonic() >= nxt:
            n.tick()
            nxt += period
    json.dump({'schema': 'wb_trigger_cmd/1', 'topic': TOPIC,
               'profile': {'zero_lead': a.zero_lead_s,
                           'half_period': a.half_period_s,
                           'cycles': a.cycles, 'zero_tail': a.zero_tail_s,
                           'total': T_END},
               'amplitude': {'base_vx': a.base_vx, 'base_wz': a.base_wz,
                             'arm_rate': a.arm_rate},
               'bounds_unchanged': {'lin_mps': 0.05, 'ang_rps': 0.2,
                                    'note': '幅度上限未動；改的是剖面形狀'},
               'wheel_row_value_mps': round(abs(a.base_vx)
                                            + 0.245 * abs(a.base_wz), 6),
               'start_base_yaw_deg': a.expect_base_yaw_deg,
               'n_sent': len(n.sent),
               'cols': ['sim_t', 'elapsed', 'segment', 'scale']
                       + [f'v{i}' for i in range(9)],
               'sent': n.sent},
              open(os.path.join(a.out, 'cmd_source.json'), 'w'),
              ensure_ascii=False)
    print(f'[cmd] -> {os.path.join(a.out, "cmd_source.json")}', flush=True)
    rclpy.try_shutdown()
    return 0


sys.exit(main())
