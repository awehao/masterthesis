#!/usr/bin/env python3
# r11（2026-10-31；R1v3，Codex Q1）：r10 的版本化複本，只加 --start-turn：第二段導航接手當下，若車頭與近端路徑方向（--turn-look 0.3 m）差 > --turn-deg（45°），
#   先在 CRUISE 內部子狀態 START_TURN 原地轉（線速度依變化率降到 0、|ω| ≤ --turn-w、|α| ≤ --turn-a，自實寫命令起算），到 ≤ --turn-deg 才以當下輸出為種子
#   依 bumpless 變化率接到導航命令。/nav1/state 的 seg 仍是 CRUISE（另給 turn 欄），執行端原備妥核對不變。里程不新鮮（牆鐘 > --odom-stale-s）輸出零；
#   逾 --turn-timeout-s（模擬）或其兩倍牆鐘 ⇒ NAV_TIMEOUT（why start_turn_timeout，輸出零，任務端照既有規則受控中止）。記 rep['start_turn']。
# r10（2026-10-31；交棒銜接包）：r9 的版本化複本，只加 --bumpless：第二段（R1→導航）全身持有時選擇器輸出實際套用底盤值（待命，執行端拒收）；導航接手當下以實寫值為種子，依變化率接到導航命令，收斂才純轉送；資料不新鮮輸出零。
"""r9（door_m4r_10：系統慢時 GMPC 晚於 CRUISE 前 5 s 才匹配 /plan（非持久訂閱）⇒ 永遠 no_plan）：r8 的複本，只改計畫重發：
  CRUISE 期間除前 5 s 外，`/plan` 訂閱者數**增加**時再發一次（晚加入者收得到；不持續重發，避免每次重設 GMPC 單調進度）。記 rep["plan_resend"]。
r8（door_m4r_6：R2 停在固定點；GMPC reference_projection＝vertex，而 r5 的 _plan2 照頂點發布、未加密，pb1h 有一段 0.687 m）：r7 的版本化複本，只改 _plan2：
  給定折線每段加密到 ≤ PLAN2_DS（0.05 m，與第一段 _plan 一致），插點航向＝該段方向、末點＝goal 航向。其餘不變。
r7（door_m4r_4：第一段到達後 GMPC 仍追舊停位、內部 ξ_prev 貼輪速邊界 ⇒ 第二段開頭線段縮放凍結；Codex 20261029_230142 核可 A＋B）：r6 的版本化複本，只加：
  第一段 HANDED_OVER 當輪（尚未收到第二段任務時）**發一次空計畫**到 /plan（GMPC → no_plan：發零、ξ_prev 歸零），之後保持；
  第二段任務到時照 r5 發新計畫（空計畫只發一次、不會在第二段再發）。不動全身端任何狀態。記 rep['idle_plan']。
r6（door_m4r_2：R3 銜接只剩橫向 0.109 m）：r5 的版本化複本，只改**第二段** GLIDE：
  - approach_vref（已按目標朝向分解縱向／橫向）只在第二段改：k_lat 1.0 → --m2-k-lat 0.5 /s、橫向上限 v_lat_cap 0.010 → --m2-v-lat-cap 0.03 m/s（Codex 20261029_181857）；第一段不變；
  - 第二段 HANDOVER_READY 另需 |橫向| ≤ --m2-ready-lat-max（0.03 m）且沿線剩餘距離 d_s ≤ --m2-pre-max（0.40 m＝R3 前置區間）；原速度、偏航、d_s ≥ ready_d_min 照舊；
  - 橫向未收斂（> 0.03）且 d_s ≤ --m2-slow-ds（0.15 m）⇒ 沿線分量乘 clip((d_s − ready_d_min)/(slow_ds − ready_d_min), 0, 1)：不走過可接手區才等對位；
    合成命令仍經既有減速限幅、選擇器與護盾；期限照舊 ⇒ GLIDE_TIMEOUT。
  - /nav1/state 增欄 lat（第二段 GLIDE 時的橫向偏差）。
r5（mode 4 重設計 R2；Codex 20261029_145418／151139）：door_nav1_mission_node.py（r4）的版本化複本，**只加第二段任務**，第一段行為不變。
  第一段 HANDED_OVER 後保持輸出零並等 /nav1/mission（JSON：{"id": 2, "vertices": [[x, y], …], "goal": [x, y, yaw]}，TRANSIENT_LOCAL）。
  收到 ⇒ 計畫＝給定折線（每點航向＝該段切線、末點＝goal 航向；不經直線重算）、goal 改為第二段交回點，**立即進 CRUISE 並轉送平滑器輸出**：
  執行端此時仍由全身控制，導航命令會被拒；這是為了**交權當下已有有效導航命令**（執行端備妥判定看 /nav1/state 的 mission、seg、cin_age）。
  之後 GLIDE／HANDOVER_READY／GLIDE_TIMEOUT／NAV_TIMEOUT 規則同第一段（--rolling 的 approach_vref＋平順減速）。
  第二段 HANDED_OVER＝收到任務後**先看到控制者為 nav、之後再轉 wholebody**（第一段的 wholebody 狀態不算）。
  /nav1/state 增欄：mission（1／2）、cin_age（模擬秒；GMPC→平滑器輸出的年齡）、plan_n。
原 r4 說明：r4（Codex 20261029_090630 選 B：滾動交棒）：--rolling ⇒ CRUISE 距停位 ≤ glide_r 起改 GLIDE（不進 PARK_ALIGN）：
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
from std_msgs.msg import Float64MultiArray, String

VERSION = 'door_nav1_mission_node/r11'
PLAN2_DS = 0.05


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
        self.create_subscription(String, '/nav1/mission', self._mission, lat)              # r5
        self.mission = 1
        self.m2 = None
        self.m2_saw_nav = False
        self.pose = None
        self.t = None
        self.cin = (0.0, 0.0, 0.0)
        self.cin_t = None
        self.ap, self.ap_wall, self.bl = None, None, None
        self.odom_wall = None
        self.st = None                                                           # r11：START_TURN 子狀態
        self.create_subscription(Float64MultiArray, '/coman/applied_cmd', self._ap, 1)
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
        self.odom_wall = time.monotonic()
        p = m.pose.pose
        self.pose = (p.position.x, p.position.y, yaw_of(p.orientation))
        cur = (self.t,) + self.pose                                             # r4：實測速度（0.1 s 以上的位姿差分）
        if self._vp is None:
            self._vp = cur
        elif cur[0] - self._vp[0] >= 0.1:
            dt = cur[0] - self._vp[0]
            self.vmeas = (math.hypot(cur[1] - self._vp[1], cur[2] - self._vp[2]) / dt, abs(wrap(cur[3] - self._vp[3])) / dt)
            self._vp = cur

    def _ap(self, m):
        d = m.data
        if len(d) == 18 and all(math.isfinite(float(d[i])) for i in (1, 2, 3, 4)):
            self.ap, self.ap_wall = (float(d[2]), float(d[3]), float(d[4])), time.monotonic()

    def _blend(self, u):
        own = str((self.rep.get('handover_last') or {}).get('owner', '')).lower()
        fresh = self.ap is not None and time.monotonic() - self.ap_wall <= 0.3
        if own != 'nav':
            self.bl = None
            return self.ap if fresh else (0.0, 0.0, 0.0)
        if self.bl is None:
            if not fresh:
                return (0.0, 0.0, 0.0)
            self.bl = {'u': list(self.ap), 't': self.t, 't0': self.t, 'on': True}
            self.rep.setdefault('bumpless', []).append({'t': self.t, 'u0': list(self.ap)})
        if not self.bl['on']:
            return u
        dt = max(0.0, min(0.2, self.t - self.bl['t']))
        self.bl['t'] = self.t
        done = True
        for i, lim in ((0, self.a.bl_a_lin), (1, self.a.bl_a_lin), (2, self.a.bl_a_ang)):
            e = u[i] - self.bl['u'][i]
            self.bl['u'][i] += clip(e, -lim * dt, lim * dt)
            done = done and abs(u[i] - self.bl['u'][i]) <= lim * 0.05
        if done or self.t - self.bl['t0'] > 6.0:
            self.bl['on'] = False
            self.rep['bumpless'][-1].update(done_s=self.t - self.bl['t0'], converged=bool(done))
        return tuple(self.bl['u'])

    def _near_dir(self):                                                        # r11：自路徑最近點沿線 turn_look 的點的方向
        V = (self.rep.get('plan2') or {}).get('vertices') or []
        if len(V) < 2:
            return None
        x, y, _ = self.pose
        best, s_best, s0, cum = 1e9, 0.0, 0.0, [0.0]
        for (ax, ay), (bx, by) in zip(V[:-1], V[1:]):
            L = math.hypot(bx - ax, by - ay)
            u = clip(((x - ax) * (bx - ax) + (y - ay) * (by - ay)) / max(L * L, 1e-12), 0.0, 1.0)
            d = math.hypot(ax + u * (bx - ax) - x, ay + u * (by - ay) - y)
            if d < best:
                best, s_best = d, s0 + u * L
            s0 += L
            cum.append(s0)
        s = min(s_best + self.a.turn_look, s0)
        for i in range(len(V) - 1):
            if cum[i + 1] >= s:
                f = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)
                qx, qy = V[i][0] + f * (V[i + 1][0] - V[i][0]), V[i][1] + f * (V[i + 1][1] - V[i][1])
                return math.atan2(qy - y, qx - x)
        return math.atan2(V[-1][1] - y, V[-1][0] - x)

    def _turn(self, u_nav):                                                     # r11：START_TURN；回傳 None ⇒ 不在轉向
        a = self.a
        own = str((self.rep.get('handover_last') or {}).get('owner', '')).lower()
        if own != 'nav' or (self.st is not None and not self.st['on']):
            return None
        th = self.pose[2]
        d = self._near_dir()
        e = None if d is None else wrap(d - th)
        if self.st is None:
            if e is None or abs(e) <= math.radians(a.turn_deg):
                self.st = {'on': False, 'skipped': True, 't0': self.t, 'e0_deg': None if e is None else math.degrees(e)}
                self.rep['start_turn'] = dict(self.st)
                return None
            self.st = {'on': True, 't0': self.t, 'wall0': time.monotonic(), 'e0_deg': math.degrees(e), 'u0': list(self.out_last), 'n': 0}
            self.rep['start_turn'] = {k: v for k, v in self.st.items() if k != 'wall0'}
            print(f'[nav1] START_TURN 起 @ sim {self.t:.2f}：車頭與近端方向差 {math.degrees(e):+.1f}°', flush=True)
        st = self.st
        if self.t - st['t0'] > a.turn_timeout_s or time.monotonic() - st['wall0'] > 2.0 * a.turn_timeout_s:
            st['on'] = False
            self.rep['start_turn'].update(timeout=True, t_end=self.t, e_end_deg=None if e is None else math.degrees(e))
            self.rep['result'] = 'start_turn_timeout'
            self._go('NAV_TIMEOUT', why='start_turn_timeout', mission=self.mission)
            return (0.0, 0.0, 0.0)
        if self.odom_wall is None or time.monotonic() - self.odom_wall > a.odom_stale_s or e is None:
            self.rep['start_turn']['n_stale'] = self.rep['start_turn'].get('n_stale', 0) + 1
            return (0.0, 0.0, 0.0)
        if abs(e) <= math.radians(a.turn_deg):                                  # 到位：以當下輸出為種子接導航命令
            st['on'] = False
            self.bl = {'u': list(self.out_last), 't': self.t, 't0': self.t, 'on': True}
            self.rep.setdefault('bumpless', []).append({'t': self.t, 'u0': list(self.out_last), 'after_turn': True})
            self.rep['start_turn'].update(t_end=self.t, dur_s=self.t - st['t0'], e_end_deg=math.degrees(e), w_exit=self.out_last[2], n=st['n'])
            print(f'[nav1] START_TURN 完成 @ sim {self.t:.2f}（{self.t - st["t0"]:.2f} s，差 {math.degrees(e):+.1f}°）', flush=True)
            return None
        dt = clip(self.t - self.out_last_t, 0.0, 0.2) if self.out_last_t is not None else 1.0 / a.rate
        lx, ly, lw = self.out_last
        w_t = clip(a.turn_k * e, -a.turn_w, a.turn_w)
        st['n'] += 1
        return (clip(0.0, lx - a.bl_a_lin * dt, lx + a.bl_a_lin * dt), clip(0.0, ly - a.bl_a_lin * dt, ly + a.bl_a_lin * dt),
                clip(w_t, lw - a.turn_a * dt, lw + a.turn_a * dt))

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

    def _mission(self, m):                                                       # r5
        try:
            d = json.loads(m.data)
            ok = int(d.get('id')) == 2 and len(d['vertices']) >= 2 and len(d['goal']) == 3 \
                and all(math.isfinite(float(v)) for p in d['vertices'] for v in p) and all(math.isfinite(float(v)) for v in d['goal'])
        except (ValueError, KeyError, TypeError):
            ok = False
        if not ok:
            self.rep.setdefault('mission_rejected', []).append({'t': self.t, 'raw': m.data[:200]})
            return
        if self.m2 is None:
            self.m2 = d
            self.rep['mission2'] = {'t_rx': self.t, 'n_vertices': len(d['vertices']), 'goal': d['goal']}

    def _gate(self, m):
        try:
            self.rep['gate_last'] = json.loads(m.data)
        except ValueError:
            self.rep['gate_last'] = {'raw': m.data}

    # ---------------------------------------------------------------- 計畫
    def _plan2(self):                                                           # r5：給定折線
        verts0 = [(float(p[0]), float(p[1])) for p in self.m2['vertices']]
        gyaw = float(self.m2['goal'][2])
        verts = [verts0[0]]                                                        # r8：每段加密到 ≤ PLAN2_DS
        for (ax, ay), (bx, by) in zip(verts0[:-1], verts0[1:]):
            n = max(1, int(math.ceil(math.hypot(bx - ax, by - ay) / PLAN2_DS)))
            verts += [(ax + (bx - ax) * k / n, ay + (by - ay) * k / n) for k in range(1, n + 1)]
        p = Path()
        p.header.frame_id = 'odom'
        p.header.stamp = self.get_clock().now().to_msg()
        for i, (vx, vy) in enumerate(verts):
            if i < len(verts) - 1:
                yw = math.atan2(verts[i + 1][1] - vy, verts[i + 1][0] - vx)
            else:
                yw = gyaw
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x, ps.pose.position.y = vx, vy
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yw / 2), math.cos(yw / 2)
            p.poses.append(ps)
        self.plan_msg = p
        self.rep['plan2'] = {'start_pose': list(self.pose), 'vertices': [list(v) for v in verts], 'goal': list(self.m2['goal']), 'n': len(verts),
                             'n_given': len(verts0), 'densify_ds': PLAN2_DS,
                             'frame': 'odom', 'kind': 'given_polyline'}

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
        ho0 = self.rep['handover_last'] or {}
        if self.mission == 1 and self.seg == 'HANDED_OVER' and 'idle_plan' not in self.rep:                # r7：撤掉第一段計畫
            _ep = Path()
            _ep.header.frame_id = 'odom'
            _ep.header.stamp = self.get_clock().now().to_msg()
            self.plan_pub.publish(_ep)
            self.rep['idle_plan'] = {'t': self.t, 'n_sub': self.plan_pub.get_subscription_count()}
            print(f'[nav1] 第一段交棒後發空計畫（GMPC 閒置）@ sim {self.t:.2f}', flush=True)
        if self.mission == 1 and self.seg == 'HANDED_OVER' and self.m2 is not None:                 # r5：第二段起動
            self.mission = 2
            self.goal = tuple(float(v) for v in self.m2['goal'])
            self._plan2()
            self.t_plan = self.t
            self.rep['glide'] = None
            self._go('CRUISE', mission=2)
        if self.mission == 2 and str(ho0.get('owner', '')).lower() == 'nav':
            if not self.m2_saw_nav:
                self.rep['mission2']['t_owner_nav'] = self.t
            self.m2_saw_nav = True
        _nsub = self.plan_pub.get_subscription_count()                            # r9
        if self.seg == 'CRUISE' and _nsub > 0 and ((self.t - self.t_plan) < 5.0 or _nsub > getattr(self, '_nsub_prev', 0)):
            self.plan_pub.publish(self.plan_msg)                               # 訂閱端 VOLATILE ⇒ 起始 5 s 內重發；r9：訂閱者增加時再發
            if (self.t - self.t_plan) >= 5.0:
                self.rep.setdefault('plan_resend', []).append({'t': self.t, 'n_sub': _nsub, 'mission': self.mission})
        self._nsub_prev = _nsub
        gx, gy, gyaw = self.goal
        x, y, th = self.pose
        ex, ey, eyaw = gx - x, gy - y, wrap(gyaw - th)
        dist = math.hypot(ex, ey)
        self.dist = dist
        ho = self.rep['handover_last'] or {}
        if self.seg not in ('ZERO', 'ALIGN_TIMEOUT', 'NAV_TIMEOUT', 'HANDED_OVER', 'GLIDE_TIMEOUT') and str(ho.get('owner', '')).lower().startswith('wholebody') \
                and (self.mission == 1 or self.m2_saw_nav):                                       # r5：第二段須先看到 nav
            self._go('HANDED_OVER', mission=self.mission)
        if self.seg in ('CRUISE', 'PARK_ALIGN', 'GLIDE', 'HANDOVER_READY') and self.t - self.t_plan > a.nav_timeout_s:
            self._go('NAV_TIMEOUT', dist=dist)
            self.rep['result'] = 'nav_timeout'
        if a.rolling and self.seg == 'CRUISE' and dist <= a.glide_r:          # r4
            self._go('GLIDE', dist=dist, yaw_err_deg=math.degrees(eyaw), vmeas=self.vmeas)
            self.rep['glide'] = {'t0': self.t}
        if a.rolling and self.seg == 'GLIDE':
            vm = self.vmeas
            self.lat = abs(-ex * math.sin(gyaw) + ey * math.cos(gyaw))                                        # r6：到進場直線的橫向
            ds_ = ex * math.cos(gyaw) + ey * math.sin(gyaw)
            lat_ok = self.mission != 2 or (self.lat <= a.m2_ready_lat_max and ds_ <= a.m2_pre_max)
            if vm is not None and a.ready_v_min <= vm[0] <= a.ready_v_max and vm[1] <= a.ready_w \
                    and abs(eyaw) <= math.radians(a.ready_yaw_deg) and dist >= a.ready_d_min and lat_ok:
                self._go('HANDOVER_READY', dist=dist, yaw_err_deg=math.degrees(eyaw), vmeas=vm, lat=self.lat, mission=self.mission)
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
            if self.a.bumpless and self.mission == 2:
                ut = self._turn(u) if self.a.start_turn else None
                u = self._blend(u) if ut is None else ut
        elif self.seg in ('GLIDE', 'HANDOVER_READY'):                           # r4：approach_vref 剖面＋平順減速
            _vl = {'v_lat_cap': a.m2_v_lat_cap, 'k_lat': a.m2_k_lat} if self.mission == 2 else {}             # r6
            (bx, by, bw), _ = MC.approach_vref(self.pose, self.goal, v_cap=a.motm_v_cap, a_ref=a.motm_a_ref, k_yaw=a.motm_k_yaw,
                                               w_cap=a.motm_w_cap, v_creep=a.motm_v_creep, overshoot=a.motm_overshoot, **_vl)
            if self.mission == 2:                                                                             # r6：未對齊不走過可接手區
                _gx, _gy, _gyaw = self.goal
                _ds = (_gx - x) * math.cos(_gyaw) + (_gy - y) * math.sin(_gyaw)
                _lat = abs(-(_gx - x) * math.sin(_gyaw) + (_gy - y) * math.cos(_gyaw))
                if _lat > a.m2_ready_lat_max and _ds <= a.m2_slow_ds:
                    f = clip((_ds - a.ready_d_min) / max(a.m2_slow_ds - a.ready_d_min, 1e-6), 0.0, 1.0)
                    c_, s_ = math.cos(th), math.sin(th)
                    wx, wy = c_ * bx - s_ * by, s_ * bx + c_ * by
                    al = wx * math.cos(_gyaw) + wy * math.sin(_gyaw)
                    wx, wy = wx - (1.0 - f) * al * math.cos(_gyaw), wy - (1.0 - f) * al * math.sin(_gyaw)
                    bx, by = c_ * wx + s_ * wy, -s_ * wx + c_ * wy
                    self.rep.setdefault('m2_slow', {'n': 0})['n'] += 1
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
        if self.st is not None and self.st['on']:
            self.rep['start_turn']['w_max_abs'] = max(self.rep['start_turn'].get('w_max_abs', 0.0), abs(u[2]))
        tw = Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = (float(v) for v in u)
        self.out_pub.publish(tw)
        self.out_last, self.out_last_t = u, self.t
        k = self.seg if self.seg in self.rep['n_out'] else 'ZERO'
        self.rep['n_out'][k] += 1

    def _pub_state(self):                                                    # r3
        cin_age = None if (self.cin_t is None or self.t is None) else self.t - self.cin_t
        self.state_pub.publish(String(data=json.dumps({'seg': self.seg, 't': self.t, 'result': self.rep['result'], 'dist': self.dist,
                                                       'mission': self.mission, 'cin_age': cin_age, 'lat': getattr(self, 'lat', None),
                                                       'turn': None if self.st is None else ('turning' if self.st['on'] else ('skipped' if self.st.get('skipped') else 'done')),
                                                       'plan_n': len(self.plan_msg.poses) if getattr(self, 'plan_msg', None) else 0}, default=float)))

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
    ap.add_argument('--bumpless', action='store_true')
    ap.add_argument('--bl-a-lin', type=float, default=0.5)
    ap.add_argument('--bl-a-ang', type=float, default=1.2)
    ap.add_argument('--a-ang', type=float, default=2.0)
    ap.add_argument('--start-turn', action='store_true')                      # r11（需 --bumpless）
    ap.add_argument('--turn-deg', type=float, default=45.0)
    ap.add_argument('--turn-look', type=float, default=0.3)
    ap.add_argument('--turn-k', type=float, default=1.5)
    ap.add_argument('--turn-w', type=float, default=0.8)
    ap.add_argument('--turn-a', type=float, default=2.0)
    ap.add_argument('--turn-timeout-s', type=float, default=6.0)
    ap.add_argument('--odom-stale-s', type=float, default=0.3)
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
    ap.add_argument('--m2-v-lat-cap', type=float, default=0.03)                # r6
    ap.add_argument('--m2-ready-lat-max', type=float, default=0.03)            # r6
    ap.add_argument('--m2-k-lat', type=float, default=0.5)                     # r6
    ap.add_argument('--m2-pre-max', type=float, default=0.40)                  # r6
    ap.add_argument('--m2-slow-ds', type=float, default=0.15)                  # r6
    ap.add_argument('--motm-v-cap', type=float, default=0.030)
    ap.add_argument('--motm-a-ref', type=float, default=0.002)
    ap.add_argument('--motm-k-yaw', type=float, default=1.0)
    ap.add_argument('--motm-w-cap', type=float, default=0.10)
    ap.add_argument('--motm-v-creep', type=float, default=0.0)
    ap.add_argument('--motm-overshoot', type=float, default=0.0)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--out', required=True)
    a, _ = ap.parse_known_args()
    if a.start_turn and not a.bumpless:
        raise SystemExit('--start-turn 需要 --bumpless')
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
