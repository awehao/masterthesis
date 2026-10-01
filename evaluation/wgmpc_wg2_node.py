"""WG2 的**薄執行介面**：把 WG1 的純數值核心接上既有安全鏈。

設計界線（依指導，**不複製 F18 那套雙 executor 框架**）
-------------------------------------------------------
* **單一 executor 所有權**：只有 `main()` 建立並 spin 一個
  `SingleThreadedExecutor`，節點只加入它一次。
  **不使用** `TransformListener(spin_thread=True)` ——
  那會把同一個節點加進第二個 executor，正是 F18 的成因。
  本節點也**完全不用 TF**：狀態只來自 `/joint_states` 與 `/odom`。
* **不可變狀態快照**：回呼只做一件事 —— 組出一個**新的** `Snap`
  並原子地指派給 `self._snap`。求解迴圈把它讀進區域變數一次，
  整輪不再讀 `self._snap`。**沒有任何跨回呼共用的可變陣列。**
  （F18 的 `_on_pts` 正是在兩次存取之間被換掉。）
* 核心保持純數值：本檔不含任何模型、成本或約束程式碼。

座標（**關鍵，避免雙重轉換**）
------------------------------
    核心輸出 u = (v_x^B, v_y^B, ω, q̇…)   —— **本體**座標
    /wholebody_safety/cmd_in              —— **世界**座標（安全鏈的慣例）
    adapter 再做 world → body 給 /wb_vel_cmd

所以本節點在發布前要做 **body → world**（乘 Rz(θ)）。
**少做這一步，adapter 的轉換就會把命令轉錯一次。**

命令的三個階段分開記錄（請求／修改後／實際套用）
-----------------------------------------------
    request  = 本節點發到 cmd_in 的（世界）
    modified = 安全層發到 cmd_out 的（世界）
    applied  = adapter 發到 /wb_vel_cmd 的（本體）

下一輪的 `u_prev` 取**實際套用**者（轉回本體），不是請求值；
取不到時依序退回 modified → request，並記錄來源。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.arm_pregrasp import ARM_JOINTS          # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import (                     # noqa: E402
    NQ, NU, WGMPCConfig, body_to_world, solve, task_error)
from ammr_wholebody_mpc.wholebody_kinematics import (            # noqa: E402
    WholeBodyKinematics)


@dataclass(frozen=True)
class Snap:
    """**不可變**狀態快照。回呼只建新的，不就地改。"""
    q: tuple           # 9 維
    sim_t: float       # 來源模擬時間
    recv_mono: float   # 收到時的單調牆鐘
    have_js: bool
    have_odom: bool


class WGMPCNode(Node):
    def __init__(self, a):
        super().__init__('wgmpc_wg2')
        # **同一時基**：狀態訊息的時間戳是模擬時間，
        # 所以本節點的時鐘也必須是模擬時間，否則
        # `sim_now() - snap.sim_t` 會變成「牆鐘 − 模擬時間」，
        # 每輪都判過期（實測：0 輪發布）。
        from rclpy.parameter import Parameter as _P
        self.set_parameters([_P('use_sim_time', _P.Type.BOOL, True)])
        self.a = a
        self.K = WholeBodyKinematics.from_urdf_file(a.urdf)
        self.cfg = WGMPCConfig(N=a.N, dt=1.0 / a.rate, tcp=a.tcp)
        # 兩半狀態各自保存**不可變** tuple；合成快照時才組 9 維
        self._arm = None          # (tuple6, sim_t, mono)
        self._base = None         # (tuple3, sim_t, mono)
        self._snap = None
        # 命令三階段，各自只存**最後一筆不可變 tuple**
        self._modified = None     # (tuple9_world, mono)
        self._applied = None      # (tuple9_body, mono)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.create_subscription(Float64MultiArray,
                                 '/wholebody_safety/cmd_out',
                                 self._on_modified, 10)
        self.create_subscription(Float64MultiArray, '/wb_vel_cmd',
                                 self._on_applied, 10)
        self.pub = self.create_publisher(Float64MultiArray,
                                         '/wholebody_safety/cmd_in', 10)
        self.log = []

    # ---------------------------------------------------------- 回呼
    def _on_js(self, m):
        ix = {n: i for i, n in enumerate(m.name)}
        if not all(j in ix for j in ARM_JOINTS):
            return
        if len(m.position) <= max(ix[j] for j in ARM_JOINTS):
            return
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self._arm = (tuple(float(m.position[ix[j]]) for j in ARM_JOINTS),
                     t, time.monotonic())
        self._compose()

    def _on_odom(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2.0 * (o.w * o.z + o.x * o.y),
                         1.0 - 2.0 * (o.y * o.y + o.z * o.z))
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self._base = ((float(p.x), float(p.y), float(yaw)), t, time.monotonic())
        self._compose()

    def _compose(self):
        """組出**新的**不可變快照並原子指派。不就地修改任何既有物件。"""
        a, b = self._arm, self._base
        if a is None or b is None:
            return
        self._snap = Snap(q=b[0] + a[0],
                          sim_t=min(a[1], b[1]),          # 取較舊者，不樂觀
                          recv_mono=min(a[2], b[2]),
                          have_js=True, have_odom=True)

    def _on_modified(self, m):
        if len(m.data) >= NU:
            self._modified = (tuple(float(x) for x in m.data[:NU]),
                              time.monotonic())

    def _on_applied(self, m):
        if len(m.data) >= NU:
            self._applied = (tuple(float(x) for x in m.data[:NU]),
                             time.monotonic())

    # ------------------------------------------------- u_prev 的來源
    def _u_prev(self, theta):
        """取**實際套用**者（本體）；退回 modified（世界→本體）→ request。"""
        now = time.monotonic()
        if self._applied is not None and now - self._applied[1] <= self.a.hist_age:
            return np.asarray(self._applied[0], float), 'applied'
        if self._modified is not None and now - self._modified[1] <= self.a.hist_age:
            w = np.asarray(self._modified[0], float)
            return body_to_world(-theta) @ w, 'modified_rotated'   # world→body
        return np.zeros(NU), 'zeros'

    # ---------------------------------------------------------- 迴圈
    def run(self, T_des) -> int:
        period = 1.0 / self.a.rate
        t0 = time.monotonic()
        slot = 0
        U_warm = None
        n_pub = n_drop_age = n_no_sol = 0
        n_miss = 0
        _prev_top = None
        while rclpy.ok():
            # **逐輪牆鐘**：核心的 timing_ms['total'] **不含**介面開銷
            # （發布、回呼處理、排程等待）⇒ 要另外量迴圈頂到頂的實際間隔。
            _top = time.monotonic()
            _cycle_wall = (None if _prev_top is None
                           else round((_top - _prev_top) * 1e3, 4))
            _prev_top = _top
            if time.monotonic() - t0 > self.a.duration_s:
                break
            snap = self._snap              # **讀一次**，整輪只用區域變數
            if snap is None:
                self._exec.spin_once(timeout_sec=0.01)
                continue
            age_in = self.sim_now() - snap.sim_t
            if age_in > self.a.max_input_age:
                # **求解前就過期**也要留紀錄，否則 log 看不到被擋下的輪次
                n_drop_age += 1
                self.log.append(dict(slot=slot, sim_t=snap.sim_t,
                                     age_in=round(age_in, 6), age_out=None,
                                     u_prev_src='n/a', ok=False,
                                     reason='stale_before_solve',
                                     published=False, timing_ms={'total': 0.0},
                                     cycle_wall_ms=_cycle_wall,
                                     dropped=f'求解前輸入已過期 '
                                             f'{age_in * 1e3:.0f} ms'))
                slot, _m = self._reschedule(t0, slot, period)
                n_miss += _m
                continue
            q0 = np.asarray(snap.q, float)
            u_prev, src = self._u_prev(float(q0[2]))
            r = solve(self.K, q0, u_prev, T_des, self.cfg, U_warm=U_warm)
            # **求解返回後重新檢查輸入年齡與任務有效性**
            self._exec.spin_once(timeout_sec=0.0)
            age_out = max(self.sim_now(), (self._snap.sim_t if self._snap
                                           else 0.0)) - snap.sim_t
            rec = dict(slot=slot, sim_t=snap.sim_t, age_in=round(age_in, 6),
                       age_out=round(age_out, 6), u_prev_src=src,
                       ok=bool(r.ok), reason=r.reason,
                       sqp_stop=r.sqp_stop_reason,
                       sqp_converged=bool(r.sqp_converged),
                       n_sqp=r.n_sqp_used, residual=r.max_residual,
                       timing_ms=r.timing_ms, cycle_wall_ms=_cycle_wall)
            if not r.ok:
                n_no_sol += 1
                rec['published'] = False
                self.log.append(rec)
                slot, _m = self._reschedule(t0, slot, period)
                n_miss += _m
                continue
            if age_out > self.a.max_input_age:
                # **過期解不發布，不重新蓋時間**
                n_drop_age += 1
                rec['published'] = False
                rec['dropped'] = f'輸入已過期 {age_out*1e3:.0f} ms'
                self.log.append(rec)
                slot, _m = self._reschedule(t0, slot, period)
                n_miss += _m
                continue
            # **body → world**，再發給安全鏈（adapter 之後會轉回 body）
            u_world = body_to_world(float(q0[2])) @ r.u0
            m = Float64MultiArray()
            m.data = [float(x) for x in u_world]
            self.pub.publish(m)
            n_pub += 1
            U_warm = r.U
            e = task_error(self.K, q0, T_des, self.cfg.tcp)
            rec.update(published=True,
                       request_world=[round(float(x), 8) for x in u_world],
                       request_body=[round(float(x), 8) for x in r.u0],
                       err_p=float(np.linalg.norm(e[:3])),
                       err_r=float(np.linalg.norm(e[3:])))
            self.log.append(rec)
            slot, _m = self._reschedule(t0, slot, period)
            n_miss += _m
        return dict(published=n_pub, dropped_stale=n_drop_age,
                    no_solution=n_no_sol, deadline_miss=int(n_miss),
                    rate_hz=self.a.rate)

    def _reschedule(self, t0, slot, period):
        """超時**跳過錯過的 slot**，不連續追趕（沿用 F17 的作法）。"""
        slot += 1
        target = t0 + slot * period
        now = time.monotonic()
        if now > target:
            slot += int((now - target) // period) + 1
            return slot, True
        while time.monotonic() < target:
            self._exec.spin_once(timeout_sec=0.002)
        return slot, False

    def sim_now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default=os.path.join(
        _HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--tcp', default='link_tcp')
    ap.add_argument('--N', type=int, default=5)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--target', nargs=3, type=float, required=True)
    ap.add_argument('--target-rot-deg', type=float, default=0.0)
    ap.add_argument('--duration-s', type=float, default=30.0)
    ap.add_argument('--max-input-age', type=float, default=0.2)
    ap.add_argument('--hist-age', type=float, default=0.5)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    rclpy.init()
    nd = WGMPCNode(a)
    # **單一 executor 所有權**：只有這裡建立，節點只加入一次
    ex = SingleThreadedExecutor()
    ex.add_node(nd)
    nd._exec = ex
    # 等狀態齊備
    t0 = time.monotonic()
    while nd._snap is None and time.monotonic() - t0 < 20.0:
        ex.spin_once(timeout_sec=0.05)
    if nd._snap is None:
        print('**狀態未齊備（缺 /joint_states 或 /odom）**', flush=True)
        ex.shutdown()
        nd.destroy_node()
        rclpy.try_shutdown()
        return 3
    q0 = np.asarray(nd._snap.q, float)
    T = nd.K.fk(q0, a.tcp).copy()
    T_des = np.eye(4)
    T_des[:3, 3] = np.asarray(a.target, float)
    T_des[:3, :3] = T[:3, :3]        # 首版只給位置目標＋保持起始姿態
    print(f'[wg2] N={a.N} dt={1.0/a.rate:.3f} 目標 {a.target}  '
          f'起始 TCP {np.round(T[:3,3],4).tolist()}', flush=True)
    try:
        stats = nd.run(T_des)
    finally:
        if a.out:
            json.dump({'args': vars(a), 'log': nd.log},
                      open(a.out, 'w'), ensure_ascii=False)
            print(f'[wg2] {len(nd.log)} 輪寫入 {a.out}', flush=True)
        ex.shutdown()
        nd.destroy_node()
        rclpy.try_shutdown()
    print(f'[wg2] {stats}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
