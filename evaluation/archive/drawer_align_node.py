#!/usr/bin/env python3
"""ALIGN：由**實測把手位姿**算出接觸前的退讓目標，發給 W-GMPC。

為什麼要這個節點
----------------
求解節點原本只吃啟動參數的**固定**目標。抽屜實驗做不到：接觸前的退讓目標
要由實測把手位姿算出，而開啟／關閉段的目標會隨實測開度移動。本節點就是那個
目標的來源，走求解節點新加的 `--target-topic`。

目標怎麼算
----------
把手是沿世界 −y 的移動接頭（滑軌），**朝向不變**，而且是沿軸的圓柱 —— 繞自己
軸的轉角無意義。所以需要量測的只有**位置**，朝向沿用設計抓取姿態的 FK 結果：

    p_G_des = p_H_meas − (tcp_offset + standoff)·ŷ
    R_G_des = R_grasp        （設計抓取姿態下 FK 算出的朝向）

`tcp_offset` 是設計抓取關係：在停車位姿與設計抓取關節角下，用**求解器自己的
FK** 算出 TCP 位於把手中心的 −y 側 0.0147 m（＝規格裡的
`tcp_offset_along_tool_z`）。沒有另外編數字。

到達判定寫清楚
--------------
本節點用自己的 FK 從 `/joint_states` 與 `/odom` 算實測 TCP，獨立判定到達與
保持。求解節點也有它自己的判定（`task_error`）。**兩者是獨立的兩個量**：
一致才算到達站得住；不一致本身就是個發現，不得挑一個好看的報。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (          # noqa: E402
    WholeBodyKinematics)

ARM = [f'joint{i}' for i in range(1, 7)]


def so3_log_norm(R):
    """旋轉誤差的角度大小 [rad]。"""
    c = (float(np.trace(R)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


class Align(Node):
    def __init__(self, a, K):
        super().__init__('drawer_align')
        self.a = a
        self.K = K
        self.pub = self.create_publisher(Float64MultiArray, a.target_topic, 10)
        # **深度 1**：狀態流要的是最新值，不是歷史
        self.create_subscription(Float64MultiArray, '/drawer/handle_pose',
                                 self._handle, 1)
        self.create_subscription(JointState, '/joint_states', self._js, 1)
        self.create_subscription(Odometry, '/odom', self._odom, 1)
        self.create_subscription(String, '/handover/state', self._hs, 1)
        self.H = None          # (sim_t, 開度, 4x4)
        self.q_arm = None
        self.pose = None
        self.hs = None
        self.n_pub = 0
        self.n_handle_bad = 0
        self.handle_bad = {}

    def _handle(self, m):
        d = list(m.data)
        if len(d) != 18:
            self.n_handle_bad += 1
            w = f'長度 {len(d)} 不是 18'
            self.handle_bad[w] = self.handle_bad.get(w, 0) + 1
            return
        if not all(math.isfinite(float(x)) for x in d):
            self.n_handle_bad += 1
            self.handle_bad['含非有限值'] = (
                self.handle_bad.get('含非有限值', 0) + 1)
            return
        self.H = (float(d[0]), float(d[1]),
                  np.asarray(d[2:], float).reshape(4, 4))

    def _js(self, m):
        ix = {n: i for i, n in enumerate(m.name)}
        if all(j in ix for j in ARM):
            self.q_arm = [float(m.position[ix[j]]) for j in ARM]

    def _odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (float(m.pose.pose.position.x),
                     float(m.pose.pose.position.y), yaw)

    def _hs(self, m):
        try:
            self.hs = json.loads(m.data)
        except Exception:
            pass

    # ------------------------------------------------------------ 目標
    def target(self, standoff):
        """回傳 (4x4 目標, 診斷)；缺實測把手位姿就回 None。"""
        if self.H is None:
            return None, {'why': '還沒收到實測把手位姿'}
        p_H = self.H[2][:3, 3]
        off = float(self.a.tcp_offset_m) + float(standoff)
        T = np.eye(4)
        T[:3, :3] = self.a.R_grasp
        T[:3, 3] = (float(p_H[0]), float(p_H[1]) - off, float(p_H[2]))
        return T, {'opening_m': self.H[1], 'p_handle': [float(x) for x in p_H],
                   'standoff_m': float(standoff), 'offset_total_m': off}

    def measured_tcp(self):
        if self.pose is None or self.q_arm is None:
            return None
        q = np.array(list(self.pose) + list(self.q_arm), float)
        return self.K.fk(q, self.a.tcp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf',
                    default='evaluation/models/omni_bot_wholebody_expanded.urdf')
    ap.add_argument('--tcp', default='link_tcp')
    ap.add_argument('--target-topic', default='/drawer/tcp_target')
    ap.add_argument('--park', default='-0.136412,0.560,1.2973')
    ap.add_argument('--q-grasp',
                    default='-0.0321,0.9177,1.4853,-0.3580,-1.0325,-1.3813')
    ap.add_argument('--standoff-m', type=float, default=0.030,
                    help='接觸前退讓量（把手中心之外，再加上 tcp_offset）')
    ap.add_argument('--tcp-offset-m', type=float, default=None,
                    help='設計抓取關係；不給就用求解器 FK 由停車位姿＋'
                         '設計抓取關節角算出來，不編數字')
    # **判準沿用 WG2 既有值**，不另立一套
    ap.add_argument('--reach-pos-m', type=float, default=0.005)
    ap.add_argument('--reach-rot-rad', type=float, default=0.02)
    ap.add_argument('--hold-s', type=float, default=2.0)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--timeout-s', type=float, default=900.0,
                    help='要蓋過求解節點的壽命 —— 它先結束的話求解節點只剩'
                         '退路目標。實跑 nav_full_030409 就是這樣')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    park = tuple(float(v) for v in a.park.split(','))
    qg = [float(v) for v in a.q_grasp.split(',')]

    K = WholeBodyKinematics.from_urdf_file(a.urdf)
    # 設計抓取姿態的 FK ⇒ 朝向與 tcp_offset 都由這裡來
    T_grasp = K.fk(np.array(list(park) + qg, float), a.tcp)
    a.R_grasp = T_grasp[:3, :3].copy()
    if a.tcp_offset_m is None:
        # 把手中心（開度 0）在 TCP 的 +y 側；距離就是設計抓取關係
        a.tcp_offset_m = float(a.standoff_m) * 0.0   # 佔位，下面由把手實測定
        a.tcp_offset_from_fk = True
    else:
        a.tcp_offset_from_fk = False

    rclpy.init()
    nd = Align(a, K)
    rep = {'scope': ('接觸前退讓目標由**實測把手位姿**算出；朝向沿用設計'
                     '抓取姿態的 FK；到達判定本節點與求解節點**各自獨立**'),
           'park': list(park), 'q_grasp': qg,
           'R_grasp': [[float(x) for x in r] for r in a.R_grasp],
           'reach_pos_m': a.reach_pos_m, 'reach_rot_rad': a.reach_rot_rad,
           'hold_s': a.hold_s, 'standoff_m': a.standoff_m,
           'events': [], 'trace': []}
    # 等第一筆實測把手位姿，據此定出 tcp_offset（不編數字）
    t0 = time.monotonic()
    while rclpy.ok() and nd.H is None and time.monotonic() - t0 < 60.0:
        rclpy.spin_once(nd, timeout_sec=0.05)
    if nd.H is None:
        print('[align] **收不到實測把手位姿** ⇒ 中止', flush=True)
        if a.out:
            rep['abort'] = '收不到 /drawer/handle_pose'
            json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1)
        return 2
    if a.tcp_offset_from_fk:
        # 設計抓取姿態下，TCP 與**實測**把手中心在 y 上的差
        a.tcp_offset_m = float(nd.H[2][1, 3]) - float(T_grasp[1, 3])
        rep['tcp_offset_m'] = round(a.tcp_offset_m, 6)
        rep['tcp_offset_source'] = ('由求解器 FK 的設計抓取 TCP 與**實測**'
                                    '把手中心在 y 上的差算出')
        print(f'[align] tcp_offset = {a.tcp_offset_m*1000:.2f} mm'
              f'（由 FK ＋ 實測把手位姿算出）', flush=True)
    rep['tcp_offset_m'] = round(float(a.tcp_offset_m), 6)

    print(f'[align] 目標話題 {a.target_topic}；退讓 {a.standoff_m*1000:.0f} mm；'
          f'判準 ≤{a.reach_pos_m*1e3:.1f} mm / '
          f'{math.degrees(a.reach_rot_rad):.2f}°、保持 {a.hold_s:.1f} s',
          flush=True)
    dt = 1.0 / a.rate
    hold_t0 = None
    arrived = False
    t1 = time.monotonic()
    while rclpy.ok() and time.monotonic() - t1 < a.timeout_s:
        rclpy.spin_once(nd, timeout_sec=0.0)
        T, dg = nd.target(a.standoff_m)
        if T is not None:
            m = Float64MultiArray()
            m.data = [float(x) for x in T.reshape(-1)]
            nd.pub.publish(m)
            nd.n_pub += 1
            Tm = nd.measured_tcp()
            if Tm is not None:
                ep = float(np.linalg.norm(Tm[:3, 3] - T[:3, 3]))
                er = so3_log_norm(T[:3, :3].T @ Tm[:3, :3])
                sim_t = nd.H[0]
                ok = ep <= a.reach_pos_m and er <= a.reach_rot_rad
                if ok:
                    if hold_t0 is None:
                        hold_t0 = sim_t
                    elif sim_t - hold_t0 >= a.hold_s and not arrived:
                        arrived = True
                        rep['events'].append(
                            {'arrived_and_held': True, 'sim_t': sim_t,
                             'hold_s': round(sim_t - hold_t0, 3),
                             'err_pos_m': round(ep, 6),
                             'err_rot_rad': round(er, 6)})
                        print(f'[align] **到達並保持 {sim_t-hold_t0:.2f} s**'
                              f'（本節點獨立判定）'
                              f'；誤差 {ep*1000:.2f} mm / '
                              f'{math.degrees(er):.3f}°', flush=True)
                else:
                    # **離開容差就重新計時** —— 保持必須是連續的
                    if hold_t0 is not None:
                        rep['events'].append(
                            {'hold_reset': True, 'sim_t': sim_t,
                             'err_pos_m': round(ep, 6),
                             'err_rot_rad': round(er, 6)})
                    hold_t0 = None
                if nd.n_pub % 20 == 1:
                    rep['trace'].append(
                        {'sim_t': sim_t, 'opening_m': dg['opening_m'],
                         'err_pos_m': round(ep, 6),
                         'err_rot_rad': round(er, 6),
                         'in_tol': bool(ok),
                         'owner': (nd.hs or {}).get('owner')})
        time.sleep(dt)
    rep['n_pub'] = nd.n_pub
    rep['exit_why'] = ('rclpy 已關閉' if not rclpy.ok()
                       else f'壽命 {a.timeout_s:.0f} s 到期')
    rep['wall_s'] = round(time.monotonic() - t1, 2)
    print(f'[align] 結束原因：{rep["exit_why"]}（跑了 {rep["wall_s"]:.1f} s）',
          flush=True)
    rep['arrived_and_held'] = bool(arrived)
    rep['n_handle_bad'] = nd.n_handle_bad
    rep['handle_bad_reasons'] = dict(nd.handle_bad)
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1,
                  default=float)
    print(f'[align] 結束；發出 {nd.n_pub} 筆目標；'
          f'到達並保持 = {arrived}', flush=True)
    nd.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
