#!/usr/bin/env python3
"""r4（Codex 20261029_090630 選 B：滾動交棒）：--rolling ⇒ CRUISE 距停位 ≤ glide_r 起改 GLIDE（不進 PARK_ALIGN）：
  目標速度＝motm_coord.approach_vref（與展開節點同參數；v_creep 0、overshoot 0），自巡航速度以 a_dec_lin／a_dec_ang 平順減速（不直接裁切）、航向轉至停位朝向；
  實測（位姿差分）線速度 ∈ [ready_v_min, ready_v_max]、|w| ≤ ready_w、偏航誤差 ≤ ready_yaw_deg、距停位 ≥ ready_d_min ⇒ HANDOVER_READY（仍照 GLIDE 命令移動，
  等執行端滾動交棒）；計畫改折線 S→P0→停位（P0＝停位沿停位朝向後退 pre_d），使最後一段沿操作朝向直線進場（交棒時橫向偏離小）；/handover/state 轉 wholebody ⇒ HANDED_OVER（零）；GLIDE 期限 glide_timeout_s ⇒ GLIDE_TIMEOUT（零、任務端轉失敗）。命令仍經同一選擇器與護盾。
r3（Codex 20261029_082724）：(a) 起步前須有新鮮里程與有效掃描（最近一筆 /scan 牆鐘年齡 ≤ 1.0 s、frame_id 非空）才發計畫進 CRUISE；
  (b) 每 0.2 s 發 /nav1/state（JSON：seg、t、result、dist）供任務端把 NAV_TIMEOUT／ALIGN_TIMEOUT 轉成任務失敗受控收尾。
r2（door_nav1_door_2：GMPC 到達容差 goal_tolerance_xy 0.20 m ⇒ 停在 0.146 m 起保持零，r1 切換半徑 0.10 m 永遠未進 PARK_ALIGN ⇒ NAV_TIMEOUT）：
  切換半徑預設 0.10 → 0.25 m（須大於 GMPC 到達容差）；/scan 訂閱改感測器 QoS（scan_relay 為 BEST_EFFORT；r1 收 0 筆＝證據缺口，非護盾未收）。其餘不變。
NAV1 任務節點（Codex 20261029_081047／081705）：起點 → W1 停位 D1 的導航段；**唯一命令選擇器**＋明確分段的 PARK_ALIGN 末端對位。

命令鏈（本節點只在中間一格）：
    gmpc_node（GMPC＋horizon CBF）→ /cmd_vel_nav → velocity_smoother → /cmd_vel_pre_mux
    → **本節點選擇器**〔CRUISE：轉送平滑器輸出｜PARK_ALIGN：P 伺服｜ZERO：精確零〕→ /cmd_vel_pre_shield
    → scan_safety_shield（掃描一階 barrier 投影）→ /cmd_vel → 執行端（nav 權威）
本節點不寫 /cmd_vel（不繞過護盾）；同一時刻只有一個來源被輸出。

分段（單向，不回切）：
  - CRUISE：發直線計畫（odom 框；末點航向＝停位航向）；選擇器轉送平滑器輸出。距 D1 ≤ switch_r（r2 預設 0.25 m）⇒ PARK_ALIGN。
  - PARK_ALIGN：P 伺服（世界誤差轉本體；|v| ≤ v_max 0.03 m/s、|w| ≤ w_max 0.1 rad/s；逐軸加速度 ≤ a_lin 0.5 m/s²／a_ang 2.0 rad/s²）。
    到 ≤ tol_pos（5 mm）且 ≤ tol_yaw（1°）⇒ ZERO。期限 align_timeout_s（30 s，模擬時間）⇒ ALIGN_TIMEOUT（輸出零，記失敗）。
  - ZERO：持續輸出精確零。**交棒與否由執行端 PARK 閘門以實測速度、實際套用命令持續 0.5 s 判定**；本節點的到位只代表提出零命令。
  - 導航總期限 nav_timeout_s（150 s，模擬時間，自計畫發布起）未進 ZERO ⇒ NAV_TIMEOUT（輸出零）。
  - /handover/state 顯示控制者已轉 wholebody ⇒ 記錄並持續輸出零。
紀錄（每 1 s 及結束寫 --out）：分段時刻與位姿、計畫、選擇器輸出筆數、護盾輸出 /cmd_vel 的筆數與最後年齡、護盾改寫次數（輸出≠輸入）、
掃描 /scan 的筆數、frame_id 與最後年齡（護盾所用掃描的新鮮度證據）、PARK 閘門最後狀態、交棒狀態。
時間：模擬時間取 /odom header.stamp（執行端真值里程；本場景前提）。
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import time

import rclpy
import motm_coord as MC                                                # r4
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

VERSION = 'door_nav1_mission_node/r4'


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def clip(v, lo, hi):
    return max(lo, min(hi, v))


class Nav1(Node):
    def __init__(self, a):
        super().__init__('door_nav1_mission')
        self.a = a
        self.goal = tuple(float(v) for v in a.goal.split(','))
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.plan_pub = self.create_publisher(Path, '/plan', lat)
        self.out_pub = self.create_publisher(Twist, a.out_topic, 10)
        self.state_pub = self.create_publisher(String, '/nav1/state', 10)        # r3
        self.create_timer(0.2, self._pub_state)
        self.dist = None
        self._vp = None                                                          # r4：位姿差分速度
        self.vmeas = None
        self.create_subscription(Odometry, '/odom', self._odom, 10)
        self.create_subscription(Twist, a.in_topic, self._cin, 10)
        self.create_subscription(Twist, '/cmd_vel', self._shield_out, 10)
        self.create_subscription(LaserScan, a.scan_topic, self._scan, qos_profile_sensor_data)   # r2
        self.create_subscription(String, '/handover/state', self._ho, 10)
        self.create_subscription(String, '/park/gate', self._gate, 1)
        self.pose = None
        self.t = None
        self.cin = (0.0, 0.0, 0.0)
        self.cin_t = None
        self.seg = 'WAIT_ODOM'
        self.out_last = (0.0, 0.0, 0.0)
        self.out_last_t = None
        self.t_plan = None
        self.rep = {'version': VERSION, 'args': vars(a), 'goal': list(self.goal), 'segments': [], 'n_out': {'CRUISE': 0, 'PARK_ALIGN': 0, 'ZERO': 0, 'GLIDE': 0, 'HANDOVER_READY': 0},
                    'shield': {'n_out': 0, 'last_wall': None, 'n_modified': 0, 'max_abs_diff': 0.0},
                    'scan': {'topic': a.scan_topic, 'n': 0, 'frame_id': None, 'last_wall': None, 'last_stamp': None},
                    'in': {'topic': a.in_topic, 'n': 0}, 'gate_last': None, 'handover_last': None, 'align': {}, 'result': None}
        self.create_timer(1.0 / a.rate, self._tick)
        self.create_timer(1.0, self.save)

    # ---------------------------------------------------------------- 訂閱
    def _odom(self, m):
        self.t = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec
        p = m.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))
        cur = (self.t,) + self.pose                                             # r4：實測速度（0.1 s 以上的位姿差分）
        if self._vp is None:
            self._vp = cur
        elif cur[0] - self._vp[0] >= 0.1:
            dt = cur[0] - self._vp[0]
            self.vmeas = (math.hypot(cur[1] - self._vp[1], cur[2] - self._vp[2]) / dt, abs(wrap(cur[3] - self._vp[3])) / dt)
            self._vp = cur

    def _cin(self, m):
        self.cin = (m.linear.x, m.linear.y, m.angular.z)
        self.cin_t = self.t
        self.rep['in']['n'] += 1

    def _shield_out(self, m):
        s = self.rep['shield']
        s['n_out'] += 1
        s['last_wall'] = time.monotonic()
        d = max(abs(m.linear.x - self.out_last[0]), abs(m.linear.y - self.out_last[1]), abs(m.angular.z - self.out_last[2]))
        if d > 1e-9:
            s['n_modified'] += 1
            s['max_abs_diff'] = max(s['max_abs_diff'], d)

    def _scan(self, m):
        s = self.rep['scan']
        s['n'] += 1
        s['frame_id'] = m.header.frame_id
        s['last_wall'] = time.monotonic()
        s['last_stamp'] = m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec

    def _ho(self, m):
        try:
            self.rep['handover_last'] = json.loads(m.data)
        except ValueError:
            self.rep['handover_last'] = {'raw': m.data}

    def _gate(self, m):
        try:
            self.rep['gate_last'] = json.loads(m.data)
        except ValueError:
            self.rep['gate_last'] = {'raw': m.data}

    # ---------------------------------------------------------------- 計畫
    def _plan(self):
        sx, sy, _ = self.pose
        gx, gy, gyaw = self.goal
        verts = [(sx, sy), (gx, gy)]
        if self.a.rolling and self.a.pre_d > 0:                                   # r4：折線，最後一段沿停位朝向
            verts = [(sx, sy), (gx - self.a.pre_d * math.cos(gyaw), gy - self.a.pre_d * math.sin(gyaw)), (gx, gy)]
        p = Path()
        p.header.frame_id = 'odom'
        p.header.stamp = self.get_clock().now().to_msg()
        n_tot = 0
        for i in range(len(verts) - 1):
            (ax, ay), (bx, by) = verts[i], verts[i + 1]
            n = max(2, int(math.hypot(bx - ax, by - ay) / 0.05) + 1)
            yaw_seg = math.atan2(by - ay, bx - ax)
            for k in range(0 if i == 0 else 1, n):
                ps = PoseStamped()
                ps.header = p.header
                ps.pose.position.x = ax + (bx - ax) * k / (n - 1)
                ps.pose.position.y = ay + (by - ay) * k / (n - 1)
                last = (i == len(verts) - 2 and k == n - 1)
                yw = gyaw if last else yaw_seg
                ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yw / 2), math.cos(yw / 2)
                p.poses.append(ps)
                n_tot += 1
        self.plan_msg = p
        self.rep['plan'] = {'start': [sx, sy], 'goal': list(self.goal), 'vertices': [list(v) for v in verts], 'n': n_tot, 'frame': 'odom',
                            'kind': 'polyline' if len(verts) > 2 else 'straight_line'}

    def _go(self, seg, **kw):
        self.seg = seg
        self.rep['segments'].append(dict({'seg': seg, 't': self.t, 'pose': list(self.pose) if self.pose else None}, **kw))
        self.get_logger().info(f'[nav1] 分段 → {seg} @ sim {self.t}')
        print(f'[nav1] 分段 → {seg} @ sim {self.t:.2f} pose {self.pose}', flush=True)

    # ---------------------------------------------------------------- 主迴圈
    def _tick(self):
        a = self.a
        if self.pose is None or self.t is None:
            return
        if self.seg == 'WAIT_ODOM':
            sc = self.rep['scan']                                               # r3：起步前掃描須有效且新鮮
            if sc['last_wall'] is None or time.monotonic() - sc['last_wall'] > 1.0 or not sc['frame_id']:
                return
            self.rep['start_check'] = {'t': self.t, 'scan_frame': sc['frame_id'], 'scan_age_wall_s': time.monotonic() - sc['last_wall'],
                                       'scan_stamp': sc['last_stamp'], 'odom_t': self.t, 'pose': list(self.pose)}
            self._plan()
            self.t_plan = self.t
            self._go('CRUISE')
        if self.seg == 'CRUISE' and self.plan_pub.get_subscription_count() > 0 and (self.t - self.t_plan) < 5.0:
            self.plan_pub.publish(self.plan_msg)                               # 訂閱端 VOLATILE ⇒ 起始 5 s 內重發
        gx, gy, gyaw = self.goal
        x, y, th = self.pose
        ex, ey, eyaw = gx - x, gy - y, wrap(gyaw - th)
        dist = math.hypot(ex, ey)
        self.dist = dist
        ho = self.rep['handover_last'] or {}
        if self.seg not in ('ZERO', 'ALIGN_TIMEOUT', 'NAV_TIMEOUT', 'HANDED_OVER', 'GLIDE_TIMEOUT') and str(ho.get('owner', '')).lower().startswith('wholebody'):
            self._go('HANDED_OVER')
        if self.seg in ('CRUISE', 'PARK_ALIGN', 'GLIDE', 'HANDOVER_READY') and self.t - self.t_plan > a.nav_timeout_s:
            self._go('NAV_TIMEOUT', dist=dist)
            self.rep['result'] = 'nav_timeout'
        if a.rolling and self.seg == 'CRUISE' and dist <= a.glide_r:          # r4
            self._go('GLIDE', dist=dist, yaw_err_deg=math.degrees(eyaw), vmeas=self.vmeas)
            self.rep['glide'] = {'t0': self.t}
        if a.rolling and self.seg == 'GLIDE':
            vm = self.vmeas
            if vm is not None and a.ready_v_min <= vm[0] <= a.ready_v_max and vm[1] <= a.ready_w \
                    and abs(eyaw) <= math.radians(a.ready_yaw_deg) and dist >= a.ready_d_min:
                self._go('HANDOVER_READY', dist=dist, yaw_err_deg=math.degrees(eyaw), vmeas=vm)
                self.rep['glide'].update(t_ready=self.t, ready_dist=dist, ready_v=vm)
            elif self.t - self.rep['glide']['t0'] > a.glide_timeout_s:
                self._go('GLIDE_TIMEOUT', dist=dist, vmeas=vm)
                self.rep['result'] = 'glide_timeout'
        if not a.rolling and self.seg == 'CRUISE' and dist <= a.switch_r:
            self._go('PARK_ALIGN', dist=dist, yaw_err_deg=math.degrees(eyaw))
            self.rep['align']['t0'] = self.t
        if self.seg == 'PARK_ALIGN':
            if dist <= a.tol_pos and abs(eyaw) <= math.radians(a.tol_yaw_deg):
                self._go('ZERO', dist=dist, yaw_err_deg=math.degrees(eyaw))
                self.rep['align'].update(t_done=self.t, dist=dist, yaw_err_deg=math.degrees(eyaw))
                self.rep['result'] = 'zero_requested'
            elif self.t - self.rep['align']['t0'] > a.align_timeout_s:
                self._go('ALIGN_TIMEOUT', dist=dist, yaw_err_deg=math.degrees(eyaw))
                self.rep['result'] = 'align_timeout'
        if self.seg == 'CRUISE':
            u = self.cin if (self.cin_t is not None and self.t - self.cin_t <= 0.5) else (0.0, 0.0, 0.0)
        elif self.seg in ('GLIDE', 'HANDOVER_READY'):                           # r4：approach_vref 剖面＋平順減速
            (bx, by, bw), _ = MC.approach_vref(self.pose, self.goal, v_cap=a.motm_v_cap, a_ref=a.motm_a_ref, k_yaw=a.motm_k_yaw,
                                               w_cap=a.motm_w_cap, v_creep=a.motm_v_creep, overshoot=a.motm_overshoot)
            dt = (self.t - self.out_last_t) if self.out_last_t is not None else 1.0 / a.rate
            dt = max(dt, 1e-3)
            lx, ly, lw = self.out_last
            u = (clip(bx, lx - a.a_dec_lin * dt, lx + a.a_dec_lin * dt), clip(by, ly - a.a_dec_lin * dt, ly + a.a_dec_lin * dt),
                 clip(bw, lw - a.a_dec_ang * dt, lw + a.a_dec_ang * dt))
        elif self.seg == 'PARK_ALIGN':
            c, s = math.cos(th), math.sin(th)
            vx, vy = a.k_lin * (c * ex + s * ey), a.k_lin * (-s * ex + c * ey)
            n = math.hypot(vx, vy)
            if n > a.v_max:
                vx, vy = vx * a.v_max / n, vy * a.v_max / n
            w = clip(a.k_yaw * eyaw, -a.w_max, a.w_max)
            dt = (self.t - self.out_last_t) if self.out_last_t is not None else 1.0 / a.rate
            dt = max(dt, 1e-3)
            lx, ly, lw = self.out_last
            vx = clip(vx, lx - a.a_lin * dt, lx + a.a_lin * dt)
            vy = clip(vy, ly - a.a_lin * dt, ly + a.a_lin * dt)
            w = clip(w, lw - a.a_ang * dt, lw + a.a_ang * dt)
            u = (vx, vy, w)
        else:
            u = (0.0, 0.0, 0.0)
        if self.seg == 'ZERO' or self.seg in ('ALIGN_TIMEOUT', 'NAV_TIMEOUT', 'HANDED_OVER', 'GLIDE_TIMEOUT'):
            u = (0.0, 0.0, 0.0)
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = (float(v) for v in u)
        self.out_pub.publish(tw)
        self.out_last, self.out_last_t = u, self.t
        k = self.seg if self.seg in self.rep['n_out'] else 'ZERO'
        self.rep['n_out'][k] += 1

    def _pub_state(self):                                                    # r3
        self.state_pub.publish(String(data=json.dumps({'seg': self.seg, 't': self.t, 'result': self.rep['result'], 'dist': self.dist}, default=float)))

    def save(self):
        r = dict(self.rep)
        now = time.monotonic()
        r['shield_out_age_wall_s'] = None if r['shield']['last_wall'] is None else now - r['shield']['last_wall']
        r['scan_age_wall_s'] = None if r['scan']['last_wall'] is None else now - r['scan']['last_wall']
        r['seg_now'], r['t_now'], r['pose_now'] = self.seg, self.t, self.pose
        json.dump(r, open(self.a.out, 'w'), ensure_ascii=False, indent=1, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal', default='0.6,-0.5332,1.570796')
    ap.add_argument('--in-topic', default='/cmd_vel_pre_mux')
    ap.add_argument('--out-topic', default='/cmd_vel_pre_shield')
    ap.add_argument('--scan-topic', default='/scan')
    ap.add_argument('--switch-r', type=float, default=0.25)                   # r2：> gmpc goal_tolerance_xy 0.20
    ap.add_argument('--tol-pos', type=float, default=0.005)
    ap.add_argument('--tol-yaw-deg', type=float, default=1.0)
    ap.add_argument('--k-lin', type=float, default=1.0)
    ap.add_argument('--k-yaw', type=float, default=1.5)
    ap.add_argument('--v-max', type=float, default=0.03)
    ap.add_argument('--w-max', type=float, default=0.10)
    ap.add_argument('--a-lin', type=float, default=0.5)
    ap.add_argument('--a-ang', type=float, default=2.0)
    ap.add_argument('--align-timeout-s', type=float, default=30.0)
    ap.add_argument('--nav-timeout-s', type=float, default=150.0)
    ap.add_argument('--rolling', action='store_true')                         # r4
    ap.add_argument('--glide-r', type=float, default=0.45)
    ap.add_argument('--pre-d', type=float, default=0.6)
    ap.add_argument('--a-dec-lin', type=float, default=0.3)
    ap.add_argument('--a-dec-ang', type=float, default=1.0)
    ap.add_argument('--ready-v-min', type=float, default=0.012)
    ap.add_argument('--ready-v-max', type=float, default=0.030)
    ap.add_argument('--ready-w', type=float, default=0.05)
    ap.add_argument('--ready-yaw-deg', type=float, default=1.5)
    ap.add_argument('--ready-d-min', type=float, default=0.06)
    ap.add_argument('--glide-timeout-s', type=float, default=30.0)
    ap.add_argument('--motm-v-cap', type=float, default=0.030)
    ap.add_argument('--motm-a-ref', type=float, default=0.002)
    ap.add_argument('--motm-k-yaw', type=float, default=1.0)
    ap.add_argument('--motm-w-cap', type=float, default=0.10)
    ap.add_argument('--motm-v-creep', type=float, default=0.0)
    ap.add_argument('--motm-overshoot', type=float, default=0.0)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--out', required=True)
    a, _ = ap.parse_known_args()
    rclpy.init()
    nd = Nav1(a)
    stop = {'v': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(v=True))
    try:
        while rclpy.ok() and not stop['v']:
            rclpy.spin_once(nd, timeout_sec=0.02)
    except KeyboardInterrupt:
        pass
    finally:
        nd.save()
        nd.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
