#!/usr/bin/env python3
"""減速交接段：**第三個控制權擁有者**。

為什麼要它
----------
導航控制器的成本函數本質是「到達目標並煞停」，與「流暢滾過交棒點」在數學
目標上衝突。實測（`nav_handover_smooth_005749`）：交棒區內 18429 個物理步
裡，同時「在全身速度框內」且「還在動」的只有 **2 步＝0.02 s**，而導航是以
267 mm/s 衝到停車點再用自己的加速度上限剎到零。靠等它剛好慢慢經過，是結構
上不可行的。

本節點**不是 MPC**。進入交接區時路徑已鎖定為直線、定位是真值（本實驗的
明示前提），所以速度完全可以用**距離驅動的確定性輪廓**算出來：

    v(d) = clip( sqrt(v_roll² + 2·a·max(0, d − d_hold)), v_roll, v_cap )

距離驅動而不是時間驅動，是刻意的：迴圈時序抖動不會改變輪廓，同一個位置
永遠得到同一個速度。降到 v_roll 之後**維持**，所以交棒窗口長度是

    T = Δd_zone / v_roll

由設計決定。v_roll = 0.030 m/s、Δd = 0.20 m ⇒ 6.7 s（667 個物理步），
而不是 0.02 s。

不碰的東西
----------
導航控制器與它的成本函數、所有速度／加速度／輪級／關節限制。本節點的輸出
走**直寫路徑**（與導航同一條，不經命令鏈）—— 它起步時仍在 0.25 m/s 量級，
那是鏈上低速介面界限 0.05 m/s 的 5 倍。執行層對這條路徑套用導航自己的
ax/ay/az_max 作逐軸變化率上限，且**與導航共用同一個基準**（上一步真正寫出
去的那一筆），所以 nav→glide 的切換當步不會產生新的跳變。
"""
from __future__ import annotations

import argparse
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def along_speed(d, *, v_roll, decel, d_hold, d_stop, v_cap, v_latched=None):
    """沿路徑速度：**距離驅動**的平方根減速輪廓。

    `d` 是沿路徑到停車點的剩餘距離。距離驅動而不是時間驅動是刻意的 ——
    迴圈時序抖動不會改變輪廓，同一個位置永遠得到同一個速度。

      d > d_hold   : sqrt(v_roll² + 2·decel·(d − d_hold))，再夾到 v_cap
      d ≤ d_hold   : v_roll（**維持**，窗口長度 = (d_hold − d_stop)/v_roll）
      d ≤ d_stop   : 0 —— 交棒沒成立時不得繼續往櫃子推

    `v_latched` 是備妥當下的速度上界，用來保證**只減速不加速**。
    """
    if d <= d_stop:
        return 0.0
    over = max(0.0, d - d_hold)
    v = math.sqrt(v_roll * v_roll + 2.0 * decel * over)
    v = min(max(v, v_roll), v_cap)
    if v_latched is not None:
        v = min(v, v_latched)
    return v


class Glide(Node):
    def __init__(self, a):
        super().__init__('drawer_glide')
        self.a = a
        self.pub = self.create_publisher(Twist, a.topic, 10)
        # **深度 1**：狀態流要的是最新值，不是歷史。
        self.create_subscription(Odometry, '/odom', self._odom, 1)
        self.create_subscription(String, '/handover/state', self._hs, 1)
        # **套用回報**：唯一能說出「現在真正在跑多快」的來源。
        # 不用 /odom 位姿差分 —— 走 ROS 的取樣會被節點處理速率平均掉，
        # 實測差分把 85.1 mm/s 看成 4.3 mm/s。
        self.create_subscription(Float64MultiArray, '/coman/applied_cmd',
                                 self._ap, 1)
        self.pose = None
        self.hs = None
        self.v_applied = None
        self.v_latched = None       # 備妥當下的速度上界，只取一次
        self.latched_at_d = None
        self.n_pub = 0
        self.trace = []

    def _odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (float(m.pose.pose.position.x),
                     float(m.pose.pose.position.y), yaw,
                     float(m.header.stamp.sec)
                     + float(m.header.stamp.nanosec) * 1e-9)

    def _ap(self, m):
        d = list(m.data)
        if len(d) >= 5:
            self.v_applied = math.hypot(float(d[2]), float(d[3]))

    def _hs(self, m):
        try:
            self.hs = json.loads(m.data)
        except Exception:
            pass

    # ---------------------------------------------------------------- 輪廓
    def profile(self):
        """回傳 (vx_body, vy_body, wz, 診斷)。沒有位姿就回 None。"""
        if self.pose is None:
            return None
        a = self.a
        x, y, yaw, t = self.pose
        ux, uy = a.park[0] - a.start[0], a.park[1] - a.start[1]
        L = math.hypot(ux, uy)
        ux, uy = ux / L, uy / L
        rx, ry = x - a.start[0], y - a.start[1]
        along = rx * ux + ry * uy
        cross = -rx * uy + ry * ux
        d = L - along                     # 沿路徑到停車點的剩餘距離
        # 沿路徑速度：距離驅動的平方根減速輪廓，降到 v_roll 後維持
        # （核心算式抽成模組層的 along_speed，離線可測）。
        #
        # **減速段不加速**，上界在備妥當下**閂鎖一次**。
        #
        # 輪廓在 d=0.60 算出 350 mm/s，而接手當下底盤可能只有 149 mm/s
        #（實測 nav_glide_012356 先加速到 350 再減速）。所以要取上界。
        #
        # 但**每步重取套用速度會變成回授迴圈**：本節點輸出什麼，套用回報
        # 就是什麼，於是速度被凍住。實測 nav_glide2_012605 的 v_along 全程
        # 等於 v_applied，由 d=0.561 到 0.273 一路停在 153 mm/s，減速只能
        # 靠執行層的變化率上限硬降。
        #
        # 閂鎖之後輪廓隨 d 單調下降，`min(輪廓, 閂鎖值)` 就是「先維持接手
        # 速度、輪廓追上後照輪廓減速」—— 正是要的行為。
        # **只在備妥區內閂鎖。** 實測 nav_glide3_012738 在 d=4.6075 就鎖了，
        # 因為那時第一筆套用回報還是零速 ⇒ 鎖成 max(v_roll, 0) = 0.03，
        # 整段 0.59→0.30 用 30 mm/s 爬了 9.6 s。窗口很寬，但那是 bug 的
        # 副作用，不是輪廓算出來的，行為要與宣稱一致。
        if (self.v_latched is None and self.v_applied is not None
                and d <= a.arm_at):
            self.v_latched = max(a.v_roll, float(self.v_applied))
            self.latched_at_d = d
        v = along_speed(d, v_roll=a.v_roll, decel=a.decel, d_hold=a.d_hold,
                        d_stop=a.d_stop, v_cap=a.v_cap,
                        v_latched=self.v_latched)
        # 橫向修正：**有界**，不讓它變成速度的主項
        vc = min(max(-a.k_cross * cross, -a.v_cross_max), a.v_cross_max)
        # 偏航修正：**有界**，把停位朝向交給全身之前先收一部分
        wz = min(max(a.k_yaw * wrap(a.park_yaw - yaw), -a.wz_max), a.wz_max)
        # 世界座標 → 本體座標
        wx, wy = ux * v - uy * vc, uy * v + ux * vc
        c, s = math.cos(yaw), math.sin(yaw)
        return (wx * c + wy * s, -wx * s + wy * c, wz,
                {'d': d, 'cross': cross, 'v_along': v, 'v_cross': vc,
                 'v_applied': (None if self.v_applied is None
                               else round(float(self.v_applied), 5)),
                 'v_latched': (None if self.v_latched is None
                               else round(float(self.v_latched), 5))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default='/glide_vel')
    ap.add_argument('--start', default='-2.80,-3.20')
    ap.add_argument('--park', default='-0.136412,0.560')
    ap.add_argument('--park-yaw', type=float, default=1.2973)
    ap.add_argument('--arm-at', type=float, default=0.60,
                    help='離停車點這麼近就開始發命令（備妥，但還沒有控制權）')
    ap.add_argument('--v-roll', type=float, default=0.030,
                    help='滾動交棒速度；要在全身速度框內且高於滾動下界')
    ap.add_argument('--v-cap', type=float, default=0.35,
                    help='輸出上限；沿用導航的 vx_max，不放寬')
    ap.add_argument('--decel', type=float, default=0.50,
                    help='沿路徑減速度 [m/s²]；導航自己的 ax_max 是 1.5，'
                         '這裡取較緩的值，不超過它')
    ap.add_argument('--d-hold', type=float, default=0.25,
                    help='在這個剩餘距離達到 v_roll，之後維持')
    ap.add_argument('--d-stop', type=float, default=0.02,
                    help='剩這麼近就停 —— 交棒沒成立時不得繼續往櫃子推')
    ap.add_argument('--k-cross', type=float, default=0.30)
    ap.add_argument('--v-cross-max', type=float, default=0.010)
    ap.add_argument('--k-yaw', type=float, default=0.30)
    ap.add_argument('--wz-max', type=float, default=0.10)
    ap.add_argument('--rate', type=float, default=50.0)
    ap.add_argument('--timeout-s', type=float, default=300.0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    a.start = tuple(float(v) for v in a.start.split(','))
    a.park = tuple(float(v) for v in a.park.split(','))

    rclpy.init()
    nd = Glide(a)
    print(f'[glide] 備妥距離 {a.arm_at} m；輪廓 v_roll={a.v_roll} '
          f'decel={a.decel} d_hold={a.d_hold}；輸出上限 {a.v_cap}', flush=True)
    rep = {'scope': '距離驅動的確定性輪廓；直寫路徑；不碰導航控制器',
           'args': {k: v for k, v in vars(a).items() if k != 'out'},
           'events': [], 'trace': []}
    t1 = time.monotonic()
    armed = False
    dt = 1.0 / a.rate
    while rclpy.ok() and time.monotonic() - t1 < a.timeout_s:
        rclpy.spin_once(nd, timeout_sec=0.0)
        pr = nd.profile()
        if pr is not None:
            vx, vy, wz, dg = pr
            if not armed and dg['d'] <= a.arm_at:
                armed = True
                rep['events'].append({'armed_at_d': round(dg['d'], 4),
                                      'v_latched': dg['v_latched'],
                                      'latched_at_d': (
                                          None if nd.latched_at_d is None
                                          else round(nd.latched_at_d, 4))})
                print(f'[glide] **開始發命令** d={dg["d"]:.3f} m'
                      f'（還沒有控制權，會依控制權被拒）', flush=True)
            if armed:
                m = Twist()
                m.linear.x, m.linear.y, m.angular.z = vx, vy, wz
                nd.pub.publish(m)
                nd.n_pub += 1
                if nd.n_pub % 10 == 1:
                    rep['trace'].append(
                        {'d': round(dg['d'], 4),
                         'cross': round(dg['cross'], 5),
                         'v_along': round(dg['v_along'], 5),
                         'v_applied': dg['v_applied'],
                         'cmd': [round(vx, 5), round(vy, 5), round(wz, 5)],
                         'owner': (nd.hs or {}).get('owner')})
            if (nd.hs or {}).get('owner') == 'wholebody':
                rep['events'].append({'released_at_d': round(dg['d'], 4)})
                print(f'[glide] 全身已接手 ⇒ 交出，停止發命令 '
                      f'（d={dg["d"]:.3f} m）', flush=True)
                break
        time.sleep(dt)
    rep['n_pub'] = nd.n_pub
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print(f'[glide] 結束；發出 {nd.n_pub} 筆', flush=True)
    nd.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
