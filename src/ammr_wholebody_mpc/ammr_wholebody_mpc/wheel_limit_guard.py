#!/usr/bin/env python3
"""最終輪級命令限制。設計見 evaluation/results/design_wheel_limit_chain.md。

保證（正常模式）：本節點**發布的命令序列**符合指定的輪速與輪加速度限制，
以**本節點自己上一次發布的命令**與**自己量到的發布間隔**為準。

不保證：實際輪子的物理加速度；異常停止／恢復模式的加速度限制（另列）。

為何在此層：velocity_smoother 只有每軸限制，看不到四輪共用的耦合列。實測
（evaluation/test_smoother_wheel.py）其輸出每軸全合規時，輪級加速度仍達
178.03 rad/s²，上限 125.0。任何位於它上游的保證都不是最終保證。
"""
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

SCHEMA = 'wheel_guard/1'

NORMAL = 'normal'                 # 前值已知合法、時間正常
STOP_UNVERIFIED = 'stop_unverified'   # 前值未知／時間異常／剛啟動，狀態不可信


def wheel_matrix(L):
    """r·ω = W·u，列序與 gmpc 的耦合約束一致。"""
    return np.array([[0.0, 1.0, L],
                     [-1.0, 0.0, L],
                     [0.0, -1.0, L],
                     [1.0, 0.0, L]])


def fit(target, prev, W, w_lim, a_lim, tol=1e-9):
    """沿 prev -> target 取最大可行 λ，同時滿足輪速與輪加速度。

    前提：prev 本身符合輪速限制。此時交集必非空（λ=0 即 prev）。
    前提不成立時回傳 (None, None, 'prev_infeasible') —— 由呼叫端處置，
    **不在此偷偷換基準**（舊實作換成零，使加速度保證換了對象）。
    """
    target = np.asarray(target, float)
    prev = np.asarray(prev, float)
    b = W @ prev
    if np.max(np.abs(b)) > w_lim + 1e-6:
        return None, None, 'prev_infeasible'
    d = W @ (target - prev)
    lam = 1.0
    for bi, di in zip(b, d):
        if abs(di) > tol:
            hi = (w_lim - bi) / di if di > 0 else (-w_lim - bi) / di
            lam = min(lam, max(hi, 0.0))
            lam = min(lam, a_lim / abs(di))
    lam = float(min(max(lam, 0.0), 1.0))
    return prev + lam * (target - prev), lam, 'ok'


class WheelLimitGuard(Node):

    def __init__(self):
        super().__init__('wheel_limit_guard')
        p = self.declare_parameter
        p('cmd_in_topic', '/cmd_vel_smoothed')
        p('cmd_out_topic', '/cmd_vel')
        p('odom_topic', '/odom')
        p('wheel_radius', 0.05)
        p('wheel_base_L', 0.245)
        p('wheel_w_max', 5.55)
        p('wheel_a_max', 125.0)
        p('publish_rate', 20.0)
        # 上游多久沒更新就把目標設為零（仍在正常模式內逐步減速）
        p('input_timeout', 0.25)
        # 自己的發布間隔超過這個值視為中斷：不可把中斷期間當成加速度預算
        p('dt_max', 0.50)
        # 恢復判定：odom 速度連續低於門檻這麼多筆，才認為機器人確實停住
        p('stopped_speed', 0.02)
        p('stopped_count', 10)
        # 冷啟動時是否已由啟動流程確認機器人靜止。預設 False ——
        # 不可預設為零，guard 重啟時機器人可能仍在執行舊命令。
        p('assume_stopped_at_start', False)

        g = lambda k: self.get_parameter(k).value          # noqa: E731
        self.r = float(g('wheel_radius'))
        self.L = float(g('wheel_base_L'))
        self.W = wheel_matrix(self.L)
        self.w_lim = self.r * float(g('wheel_w_max'))
        self.a_max = float(g('wheel_a_max'))
        self.input_timeout = float(g('input_timeout'))
        self.dt_max = float(g('dt_max'))
        self.stopped_speed = float(g('stopped_speed'))
        self.stopped_count = int(g('stopped_count'))

        be = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                        history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(Twist, str(g('cmd_out_topic')), 10)
        self.status_pub = self.create_publisher(String, '/wheel_guard/status', 10)
        self.create_subscription(Twist, str(g('cmd_in_topic')), self._in, 10)
        self.create_subscription(Odometry, str(g('odom_topic')), self._odom, be)

        self.target = np.zeros(3)
        self.t_in = None                 # 最後一次收到輸入的時間
        self.prev = None                 # 自己上一次發布的命令
        self.t_pub = None                # 自己上一次發布的時間
        self.mode = NORMAL if bool(g('assume_stopped_at_start')) else STOP_UNVERIFIED
        if self.mode is NORMAL:
            self.prev = np.zeros(3)
        self.slow_n = 0
        self.seq = 0
        self.faults = []
        self.create_timer(1.0 / float(g('publish_rate')), self._tick)
        self.get_logger().info(
            f'wheel_limit_guard: {g("cmd_in_topic")} -> {g("cmd_out_topic")}，'
            f'起始模式 {self.mode}，r·ω_max={self.w_lim:.4f} m/s，'
            f'α_max={self.a_max} rad/s²')

    # ------------------------------------------------------------------
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _in(self, m):
        self.target = np.array([m.linear.x, m.linear.y, m.angular.z], float)
        self.t_in = self._now()

    def _odom(self, m):
        v = m.twist.twist
        s = math.hypot(v.linear.x, v.linear.y) + abs(v.angular.z) * self.L
        self.slow_n = self.slow_n + 1 if s <= self.stopped_speed else 0

    # ------------------------------------------------------------------
    def _tick(self):
        now = self._now()
        self.seq += 1
        self.faults = []
        dt = None if self.t_pub is None else now - self.t_pub

        # 時間異常：倒退或中斷。不可把中斷期間當成加速度預算。
        if dt is not None and (dt <= 0.0 or dt > self.dt_max):
            self.faults.append('dt_invalid' if dt <= 0.0 else 'dt_gap')
            self.mode = STOP_UNVERIFIED

        if self.mode == STOP_UNVERIFIED:
            # 失效處置：直接送零。**不宣稱這次轉換滿足加速度限制。**
            out, lam, why = np.zeros(3), None, 'stop_unverified'
            if self.slow_n >= self.stopped_count:
                self.mode = NORMAL
                self.prev = np.zeros(3)
                self.faults.append('recovered_stop_confirmed')
            accel_claim = False
        else:
            # 上游逾時：把零設為目標，仍經輪級限制逐步減速。
            stale = (self.t_in is None) or (now - self.t_in > self.input_timeout)
            tgt = np.zeros(3) if stale else self.target
            if stale:
                self.faults.append('input_timeout')
            if dt is None:
                out, lam, why = np.zeros(3), 0.0, 'first_cycle'
                accel_claim = False          # 首筆無間隔可據
            else:
                a_lim = self.r * self.a_max * dt
                out, lam, why = fit(tgt, self.prev, self.W, self.w_lim, a_lim)
                if out is None:
                    # 前值不合法：不換基準，改進入獨立停止／恢復模式
                    self.faults.append('prev_infeasible')
                    self.mode = STOP_UNVERIFIED
                    out, lam, why = np.zeros(3), None, 'stop_unverified'
                    accel_claim = False
                else:
                    accel_claim = True

        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = (float(out[0]), float(out[1]),
                                               float(out[2]))
        self.pub.publish(m)
        w_out = (self.W @ out) / self.r
        st = dict(schema=SCHEMA, seq=self.seq, stamp=now, mode=self.mode,
                  action=why, lam=lam, dt=dt,
                  target=[float(v) for v in self.target],
                  out=[float(v) for v in out],
                  omega_max=float(np.max(np.abs(w_out))),
                  omega_limit=self.w_lim / self.r,
                  accel_guaranteed=accel_claim, faults=list(self.faults))
        s = String(); s.data = json.dumps(st, ensure_ascii=False)
        self.status_pub.publish(s)
        self.prev, self.t_pub = out, now


def main():
    rclpy.init()
    n = WheelLimitGuard()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
