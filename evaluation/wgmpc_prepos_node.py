#!/usr/bin/env python3
"""階段 A 的**前置調姿**發布端：從實際零位走 j3 梯形軌跡。

為什麼走同一條管線
------------------
命令以**求解段封裝**發到 `/wgmpc/cmd_env`，因此會經過安全層、adapter 與執行端
E2 —— 和 W-GMPC 的命令走完全相同的路。前置調姿因此同樣受輪級限制、屏障與
設定點有效限位保護。若另開一條捷徑直接寫設定點，這段就不是受控移動段，
而紀錄裡也無法與階段 A 用同一套欄位比對。

**身分可辨**：本節點用自己的 `run_id`，所以事後能從封裝紀錄分辨每一筆命令
來自前置調姿還是 W-GMPC。兩者永不重疊（由交棒閘門保證），所以序號空間不需合併。

完成宣告
--------
走完軌跡後**繼續發零速率**一段時間（讓執行端的設定點停穩），再以 latched
字串宣告完成並解除自己。交棒閘門要的是**明確宣告**：沉默不等於完成，
因為發布端也可能是掛掉了。
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
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

from arm_traj import trapezoid                                  # noqa: E402
from ammr_wholebody_mpc import cmd_envelope as CE                # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import (            # noqa: E402
    WholeBodyKinematics)

ARM = [f'joint{i}' for i in range(1, 7)]
DONE_TOPIC = '/wgmpc/prepos_done'


class PreposNode(Node):
    def __init__(self, a):
        super().__init__('wgmpc_prepos')
        self.a = a
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        K = WholeBodyKinematics.from_urdf_file(a.urdf)
        lim = np.array(K.joint_limits())
        self.elo = lim[0, 3:] + a.joint_margin
        self.ehi = lim[1, 3:] - a.joint_margin
        self.goal = (np.array([float(v) for v in a.goal.split(',')])
                     if a.goal else 0.5 * (self.elo + self.ehi))
        if len(self.goal) != 6:
            raise SystemExit('--goal 必須是六個值')
        bad = [(i + 1, float(self.goal[i])) for i in range(6)
               if self.goal[i] < self.elo[i] or self.goal[i] > self.ehi[i]]
        if bad:
            raise SystemExit(f'**目標構型已在有效限位之外**：{bad}')
        self.run_id_n = CE.run_id_num(a.run_id)
        _lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(Float64MultiArray,
                                         '/wgmpc/cmd_env', 10)
        self.done_pub = self.create_publisher(String, DONE_TOPIC, _lat)
        self.create_subscription(JointState, '/joint_states', self._js, 10)
        self.q_meas = None
        self.q_meas_t = None
        self.seq = 0
        self.n_in_tol = 0
        self.r_prev = np.zeros(6)
        self.tail_left = int(round(a.tail_s * a.rate))
        self.e_hist = []
        self.published_any = False
        self.was_reached = None
        self.timed_out = False
        self.Q = None
        self.i = 0
        self.t0 = None
        self.n_tail = int(round(a.tail_s * a.rate))
        self.events = []

    def _js(self, m: JointState):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM):
            self.q_meas = np.array([float(d[j]) for j in ARM])
            self.q_meas_t = self._sim_t()

    def _sim_t(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def step_closed_loop(self):
        """**對實測角閉迴路**的點到點移動，不是開迴路播軌跡。

        為什麼改掉開迴路
        ----------------
        開迴路靠「發出的速率 × 控制週期」等於預期增量。那個等式在實跑裡不成立：
          * 按牆鐘發布時，RTF != 1 會讓積分按比例短少（r2 到 82.8%）
          * 改按模擬時鐘後仍短少 8.5%，而節律只差 3.9%、無凍結、無逾時
            ⇒ 還有未完全歸因的機制（命令被提前取代的嫌疑最大）
        閉迴路不需要那個等式成立：只要還有誤差就繼續給速率，到了就停。
        與節律、RTF、遺失步數都無關。

        速度與加速度上限**沿用凍結的預抓取軌跡限制**（0.35 / 0.7），
        所以運動本身的約束沒有放寬。
        """
        q = self.q_meas
        err = self.goal - q
        emax = float(np.abs(err).max())
        # 到達判定：連續 n 輪都在容差內才算（單輪可能是雜訊）
        if emax <= self.a.reach_tol:
            self.n_in_tol += 1
        else:
            self.n_in_tol = 0
        reached = self.n_in_tol >= self.a.reach_cycles
        if reached:
            self.tail_left -= 1
            r = np.zeros(6)                 # 到達後發零速率，讓設定點停穩
        else:
            r = np.clip(self.a.kp * err, -self.a.vel_max, self.a.vel_max)
            # 加速度上限：相對上一筆命令的變化量設限
            dr = r - self.r_prev
            lim = self.a.acc_max / self.a.rate
            r = self.r_prev + np.clip(dr, -lim, lim)
            r = np.clip(r, -self.a.vel_max, self.a.vel_max)
        self.r_prev = r.copy()
        self.e_hist.append((float(self._sim_t()), emax))
        if not self.published_any or reached != self.was_reached:
            self.get_logger().info(
                f'前置閉迴路：最大誤差 {emax*1e3:.2f} mrad'
                f'（容差 {self.a.reach_tol*1e3:.1f}）'
                f'{"、已到達 ⇒ 零速率尾段" if reached else ""}')
            self.was_reached = reached
        self.published_any = True
        self._publish(r)
        if reached and self.tail_left <= 0:
            return True
        if self.seq > self.a.max_cycles:
            self.get_logger().error(
                f'**前置調姿超過 {self.a.max_cycles} 輪仍未到達**'
                f'（最大誤差 {emax*1e3:.2f} mrad）')
            self.timed_out = True
            return True
        return False

    def _publish(self, r):
        u = [0.0, 0.0, 0.0] + [float(v) for v in r]
        m = Float64MultiArray()
        m.data = CE.encode(self.run_id_n, self.seq, CE.ST_SOLVER, self.seq,
                           -1, True, self._sim_t(), u, CE.K_NORMAL)
        self.pub.publish(m)
        _now = self._sim_t()
        if getattr(self, 'first_cmd_t', None) is None:
            self.first_cmd_t = _now
        self.last_cmd_t = _now
        self.seq += 1

    def _build(self):
        """以**實測零位**為起點產生軌跡。不以名義零位代替。"""
        q0 = self.q_meas.copy()
        Q, T = trapezoid(q0, self.goal, self.a.vel_max, self.a.acc_max,
                         self.a.rate)
        self.Q = Q
        self.T = T
        self.rate_cmd = np.vstack([np.diff(Q, axis=0) * self.a.rate,
                                   np.zeros((self.n_tail + 1, 6))])
        peak = float(np.abs(self.rate_cmd).max())
        if peak > self.a.arm_rate_max:
            raise SystemExit(f'**命令速率峰值 {peak:.4f} 超過執行端上限 '
                             f'{self.a.arm_rate_max}**：執行端會整筆拒收')
        self.get_logger().info(
            f'前置調姿：起點（實測）{np.round(q0, 5).tolist()} → '
            f'終點 {np.round(self.goal, 5).tolist()}；'
            f'{len(Q)} 個命令點、{T:.3f} s、速率峰值 {peak:.4f} rad/s'
            f'（上限 {self.a.arm_rate_max}）；尾段零速率 {self.a.tail_s} s')
        self.events.append({'ev': 'built', 'sim_t': self._sim_t(),
                            'q_start_measured': q0.tolist(),
                            'q_goal': self.goal.tolist(),
                            'n_points': int(len(Q)), 'duration_s': float(T),
                            'rate_peak_rad_s': peak})

    def tick(self):
        if self.q_meas is None:
            return False
        if self.Q is None:
            self._build()
            self.t0 = self._sim_t()
        return self.step_closed_loop()

    def tick_open_loop(self):
        """**保留但不再使用**的開迴路版本，供對照。"""
        if self.i >= len(self.rate_cmd):
            return True
        r = self.rate_cmd[self.i]
        # 底盤分量一律為零：前置調姿**不動底盤**
        u = [0.0, 0.0, 0.0] + [float(v) for v in r]
        m = Float64MultiArray()
        m.data = CE.encode(self.run_id_n, self.seq, CE.ST_SOLVER, self.seq,
                           -1, True, self._sim_t(), u, CE.K_NORMAL)
        self.pub.publish(m)
        _now = self._sim_t()
        if getattr(self, 'first_cmd_t', None) is None:
            self.first_cmd_t = _now
        self.last_cmd_t = _now
        self.seq += 1
        self.i += 1
        return False

    def declare_done(self):
        """**明確完成宣告**：帶上最後一筆命令的時間與序號，供交棒閘門核對。"""
        payload = {
            'source': 'wgmpc_prepos',
            'run_id': self.a.run_id,
            'n_commands': int(self.seq),
            'last_cmd_sim_t': float(getattr(self, 'last_cmd_t', float('nan'))),
            'declared_sim_t': float(self._sim_t()),
            'q_goal': self.goal.tolist(),
            'q_meas_at_done': (None if self.q_meas is None
                               else self.q_meas.tolist()),
            'tail_zero_rate_s': float(self.a.tail_s),
            'note': ('軌跡與尾段零速率都已發完；本節點隨即解除，'
                     '不再發任何命令'),
        }
        self.done_pub.publish(String(data=json.dumps(payload,
                                                     ensure_ascii=False)))
        payload['closed_loop'] = {
            'basis': '對實測關節角閉迴路（不是開迴路播軌跡）',
            'kp': self.a.kp, 'reach_tol_rad': self.a.reach_tol,
            'reach_cycles': self.a.reach_cycles,
            'vel_max_rps': self.a.vel_max, 'acc_max_rps2': self.a.acc_max,
            'timed_out': bool(self.timed_out),
            'final_max_err_rad': (None if not self.e_hist
                                  else self.e_hist[-1][1]),
            'n_cycles': int(self.seq),
        }
        payload['publish_cadence'] = {
            'basis': '模擬時鐘（不是牆鐘）',
            'nominal_period_sim_s': 1.0 / self.a.rate,
            'first_cmd_sim_t': getattr(self, 'first_cmd_t', None),
            'last_cmd_sim_t': getattr(self, 'last_cmd_t', None),
            'measured_mean_period_sim_s': (
                None if (getattr(self, 'first_cmd_t', None) is None
                         or self.seq < 2) else
                (self.last_cmd_t - self.first_cmd_t) / (self.seq - 1)),
        }
        self.events.append({'ev': 'done', **payload})
        self.get_logger().info(f'前置調姿完成宣告已發布（{DONE_TOPIC}）：'
                               f'{payload["n_commands"]} 筆命令，'
                               f'最後一筆 sim_t={payload["last_cmd_sim_t"]:.3f}')
        return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default=os.path.join(
        WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--run-id', default='prepos')
    ap.add_argument('--goal', default='',
                    help='六軸目標構型；空 = 有效限位中點')
    ap.add_argument('--vel-max', type=float, default=0.35)
    ap.add_argument('--acc-max', type=float, default=0.7)
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--arm-rate-max', type=float, default=1.0)
    ap.add_argument('--joint-margin', type=float, default=0.05)
    ap.add_argument('--kp', type=float, default=2.0,
                    help='閉迴路比例增益（rad/s per rad）')
    ap.add_argument('--reach-tol', type=float, default=0.005,
                    help='到達容差（rad）；比交棒的 prepos_goal_tol 更緊')
    ap.add_argument('--reach-cycles', type=int, default=5,
                    help='連續這麼多輪都在容差內才算到達')
    ap.add_argument('--max-cycles', type=int, default=600,
                    help='超過即判為未到達並照實宣告')
    ap.add_argument('--tail-s', type=float, default=1.0,
                    help='軌跡結束後繼續發零速率的時間（讓設定點停穩）')
    ap.add_argument('--wait-state-s', type=float, default=30.0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    rclpy.init()
    nd = PreposNode(a)
    rc = 0
    try:
        t_wait = time.monotonic()
        while rclpy.ok() and nd.q_meas is None:
            rclpy.spin_once(nd, timeout_sec=0.05)
            if time.monotonic() - t_wait > a.wait_state_s:
                nd.get_logger().error('**等不到 /joint_states**，中止')
                return 71
        # **按模擬時鐘發布，不是牆鐘。**
        # 執行端的設定點是**逐物理步**積分 rate*dt（模擬時間）。用 time.sleep
        # 按牆鐘發 20 Hz 時，只要 RTF != 1，每筆命令作用的物理步數就不是 5，
        # 積分結果按比例短少 —— 實跑兩趟：r1 到達 99.1%（RTF≈0.99）、
        # r2 只到 82.8%（RTF≈0.83，設定點停在 1.19 而目標是 1.437）。
        # W-GMPC 節點早已為同一原因改用模擬時鐘（並實跑確認 20.0 Hz）。
        period = 1.0 / a.rate
        t_last = None
        wall_guard = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(nd, timeout_sec=0.005)
            now = nd._sim_t()
            if now <= 0.0:
                # /clock 還沒動：不要在這裡空轉發命令
                if time.monotonic() - wall_guard > a.wait_state_s:
                    nd.get_logger().error('**模擬時鐘未前進**，中止')
                    return 72
                continue
            if t_last is not None and (now - t_last) < period - 1e-9:
                continue
            if nd.tick():
                break
            t_last = now if t_last is None else t_last + period
            # 模擬時鐘若跳躍（> 一個週期），對齊到現在，避免補發一串命令
            if now - t_last > period:
                t_last = now
            wall_guard = time.monotonic()
        payload = nd.declare_done()
        # 讓 latched 宣告確實送出
        for _ in range(20):
            rclpy.spin_once(nd, timeout_sec=0.02)
        if a.out:
            json.dump({'events': nd.events, 'done': payload},
                      open(a.out, 'w'), ensure_ascii=False, indent=1)
    except KeyboardInterrupt:
        rc = 130
    finally:
        nd.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
