#!/usr/bin/env python3
"""全身端：交棒前暖機 → 展開 → 交給 W-GMPC。

排序問題（先講清楚）
--------------------
交棒的預核要核「全身的首筆命令套得進去」，所以全身**必須在切換前就開始送
命令**。切換前它沒有控制權，命令會被執行端依控制權拒絕 —— 但命令鏈仍然
收下（n_recv 增加、快照更新），預核才有東西可核。

所以本節點分三段：

    暖機（切換前）  送「**維持實測底盤速度、手臂零速率**」。這讓交棒當步的
                   套用命令與導航最後一筆接近，換手不產生跳變；也讓預核有
                   一筆安全的首筆命令可核。
    展開（取得控制權後）  手臂自收攏走到夾持姿態。**j3 必須先往正向離開
                   限位** —— 收攏姿態 j3 = 0 距有效下限只有 0.011 rad，
                   而且往負向 0.05 rad 內就自碰（arm_initial_pose.yaml）。
                   底盤同時把最後一段距離與朝向誤差收掉。
    交棒給 W-GMPC  展開完成後宣告，由 W-GMPC 接手做接觸前到達。

**證據**：本節點逐物理步記錄套用命令與實測速度，切換前後各一段，
用來判斷換手是否連續（有沒有非預期零命令或跳變）。
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
from std_msgs.msg import Bool, Float64MultiArray, String

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from motm_coord import approach_vref                            # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.wholebody_kinematics import (            # noqa: E402
    WholeBodyKinematics)

ARM = [f'joint{i}' for i in range(1, 7)]
AP_COLS = ['physics_step_id', 'sim_t', 'bvx_body', 'bvy_body', 'wz',
           'qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6',
           'exec_mode_code', 'cmd_age_s', 'n_recv', 'n_rejected',
           'api_applied', 'src_recv_seq', 'src_recv_sim_t']


class WholeBody(Node):
    def __init__(self, a):
        super().__init__('drawer_wholebody')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.a = a
        self.pub = self.create_publisher(Float64MultiArray, '/wb_vel_cmd', 10)
        lat = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.done_pub = self.create_publisher(String, '/wholebody/state', lat)
        self.solver_ready = False
        self.create_subscription(Bool, '/wgmpc/ready', self._solver_ready, lat)
        # **深度 1**：這些是 50–100 Hz 的狀態流，節點處理得比它慢。
        # 用深度 10 會累積成數千步的落後 —— 實測看過 11000 步（110 s），
        # 以致要求的切換物理步早已過去。狀態流要的是**最新值**，不是歷史。
        self.create_subscription(Odometry, '/odom', self._odom, 1)
        self.create_subscription(JointState, '/joint_states', self._js, 1)
        self.create_subscription(Float64MultiArray, '/coman/applied_cmd',
                                 self._ap, 1)
        self.create_subscription(String, '/handover/state', self._hs, 1)
        # PARK_FIXED：執行端違規閂鎖 ⇒ 停止發布（預設不訂 ⇒ 既有行為不變）
        self.park_violation = None
        if getattr(a, 'park_fixed', False):
            self.create_subscription(String, '/park/violation', self._park_vio, 1)
            self.vio_pub = self.create_publisher(String, '/park/violation', 10)
        self.pose = None
        self.vb = None
        self.q = None
        self.q_t = None
        self.ap = None
        self.hs = None
        self.t_seen = 0.0
        self.trace = []          # 逐步：套用命令 ＋ 實測速度

    @staticmethod
    def _st(h):
        return float(h.stamp.sec) + float(h.stamp.nanosec) * 1e-9

    def _solver_ready(self, m):
        self.solver_ready = bool(m.data)

    def _note(self, t):
        if t > self.t_seen:
            self.t_seen = float(t)

    def _odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        t = self._st(m.header)
        self._note(t)
        p = (float(m.pose.pose.position.x), float(m.pose.pose.position.y),
             yaw, t)
        if self.pose is not None:
            dt = p[3] - self.pose[3]
            if dt > 0.005:
                dx, dy = p[0] - self.pose[0], p[1] - self.pose[1]
                c, s = math.cos(yaw), math.sin(yaw)
                self.vb = ((dx * c + dy * s) / dt, (-dx * s + dy * c) / dt,
                           (yaw - self.pose[2]) / dt)
        self.pose = p

    def _js(self, m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM):
            self.q = tuple(float(d[j]) for j in ARM)
            self.q_t = self._st(m.header)
            self._note(self.q_t)

    def _ap(self, m):
        if len(m.data) != len(AP_COLS):
            return
        c = {k: m.data[i] for i, k in enumerate(AP_COLS)}
        self.ap = c
        self._note(float(c['sim_t']))
        # **逐步證據**：套用命令 ＋ 當下實測速度 ＋ 誰在控制
        self.trace.append({
            'step': int(c['physics_step_id']), 'sim_t': float(c['sim_t']),
            'applied': [round(float(c[k]), 6) for k in
                        ('bvx_body', 'bvy_body', 'wz')],
            'applied_arm': [round(float(c[k]), 6) for k in
                            ('qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6')],
            'exec_mode': int(c['exec_mode_code']),
            'meas_vb': ([round(x, 6) for x in self.vb] if self.vb else None),
            'owner': (self.hs or {}).get('owner')})
        if len(self.trace) > 20000:
            self.trace = self.trace[-20000:]

    def _hs(self, m):
        try:
            self.hs = json.loads(m.data)
        except Exception:
            pass

    def _park_vio(self, m):
        try:
            self.park_violation = json.loads(m.data)
        except Exception:
            self.park_violation = {'why': m.data}

    def send(self, u9):
        m = Float64MultiArray()
        m.data = [float(x) for x in u9]
        self.pub.publish(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stow-q', default='0,0,0,0,-1.5707963,0')
    ap.add_argument('--park', default='-0.136412,0.56,1.297349')
    ap.add_argument('--urdf', default=os.path.join(
        WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--rate', type=float, default=20.0)
    ap.add_argument('--joint-margin', type=float, default=0.05)
    ap.add_argument('--v-base', type=float, default=0.030,
                    help='展開段的底盤速度上限（必須在全身速度框內）')
    ap.add_argument('--w-base', type=float, default=0.15)
    ap.add_argument('--qd-max', type=float, default=0.35,
                    help='展開段的關節速率上限')
    ap.add_argument('--j3-clear-rad', type=float, default=0.25,
                    help='j3 要先往正向離開限位到這個值，才動其他軸')
    ap.add_argument('--pos-tol', type=float, default=0.02)
    ap.add_argument('--yaw-tol', type=float, default=0.05)
    ap.add_argument('--q-tol', type=float, default=0.02)
    ap.add_argument('--timeout-s', type=float, default=240.0)
    ap.add_argument('--hold-timeout-s', type=float, default=120.0,
                    help='展開完成後保持交棒最多等這麼久（模擬時間）；'
                         '逾時就結束並記錄，不無限等')
    ap.add_argument('--out', default='')
    # **移動中操作（MotM）**：預設關閉 ⇒ 既有行為不變。
    ap.add_argument('--motm', action='store_true',
                    help='展開期間底盤照 √ 剖面朝停位走（與任務節點共用 '
                         'motm_coord.approach_vref）；手臂到位即完成，不等底盤；'
                         '保持交棒期間底盤**照剖面繼續走**，不送零')
    ap.add_argument('--motm-v-cap', type=float, default=0.030)
    ap.add_argument('--motm-a-ref', type=float, default=0.002)
    ap.add_argument('--motm-k-yaw', type=float, default=1.0)
    ap.add_argument('--motm-w-cap', type=float, default=0.10)
    # 接近時不降到零：以蠕行速度越過停位至多 overshoot（手臂補償），
    # 夾持遲遲不成立才在越過上限處停下。兩個節點必須用同一組值。
    ap.add_argument('--motm-v-creep', type=float, default=0.004)
    ap.add_argument('--motm-overshoot', type=float, default=0.030)
    ap.add_argument('--solver-handshake', action='store_true',
                    help='等求解器明確回報 ready 才交出；不以發布者數判斷')
    ap.add_argument('--park-fixed', action='store_true',
                    help='PARK_FIXED：暖機與展開的底盤三軸恆為零（不承接導航速度、不做停位修正）')
    a = ap.parse_args()
    if a.park_fixed and a.motm:
        print('[wb] **--park-fixed 與 --motm 不可同時指定** ⇒ 拒絕啟動', flush=True)
        return 3

    stow = np.array([float(v) for v in a.stow_q.split(',')])
    px, py, pyaw = (float(v) for v in a.park.split(','))
    K = WholeBodyKinematics.from_urdf_file(a.urdf)
    lim = np.array(K.joint_limits())
    elo, ehi = lim[0, 3:] + a.joint_margin, lim[1, 3:] - a.joint_margin
    # 夾持姿態：停位幾何算出的那一組（底盤 1 比 1 跟退時手臂維持它）
    Q_GRASP = np.array([-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813])

    rclpy.init()
    nd = WholeBody(a)
    rep = {'phase': 'WARMUP', 'events': [],
           # 參數落盤：--motm 等設定要能由檔案核對（horizon_config_check.py）
           'args': {k: v for k, v in vars(a).items() if k != 'out'},
           'q_grasp': Q_GRASP.tolist(), 'stow': stow.tolist()}
    t0 = time.monotonic()
    while rclpy.ok() and time.monotonic() - t0 < 30 and (
            nd.q is None or nd.vb is None):
        rclpy.spin_once(nd, timeout_sec=0.05)
    if nd.q is None:
        print('[wb] **收不到狀態** ⇒ 中止', flush=True)
        return 2
    print('[wb] 暖機：送「維持實測底盤速度、手臂零速率」', flush=True)

    phase = 'WARMUP'
    warm_src, warm_clipped = 'none', False
    hold_t0 = None
    dt = 1.0 / a.rate
    t1 = time.monotonic()
    j3_cleared = False
    while rclpy.ok() and time.monotonic() - t1 < a.timeout_s:
        rclpy.spin_once(nd, timeout_sec=0.0)
        if a.park_fixed and nd.park_violation is not None:
            rep['park_violation'] = nd.park_violation
            phase = 'PARK_VIOLATION'          # 結尾判失敗，不因已到 HOLD 而回傳成功
            print(f'[wb] 收到 PARK_FIXED 違規 ⇒ 停止發布：{nd.park_violation}', flush=True)
            break
        owner = (nd.hs or {}).get('owner')
        q = np.array(nd.q) if nd.q is not None else stow
        u = np.zeros(9)

        if phase == 'WARMUP':
            # **維持底盤速度**，手臂不動 —— 讓換手不產生跳變。
            #
            # 來源必須是**執行端自己的套用回報**，不是 /odom 位姿差分。
            # 位姿差分走 ROS，取樣會被節點處理速率平均掉：實測交棒當步底盤
            # 真正在跑 85.1 mm/s，差分估出來的卻讓暖機只送 4.3 mm/s，
            # 於是切換當步從 85.1 掉到 4.3 —— 一個物理步跳 80.8 mm/s。
            #
            # 用套用回報就沒有這個問題：命令鏈承接的 u_prev 與本節點送出的
            # 首筆請求是**同一個值**，λ 增量為零，交棒逐位元連續。
            src = 'nav_applied'
            nap = (nd.hs or {}).get('nav_applied')
            if a.park_fixed:
                # PARK_FIXED：執行端靜止閘門已保證套用底盤為零；不承接導航速度
                src, ub = 'park_fixed_zero', [0.0, 0.0, 0.0]
            elif nap is not None and nap.get('u_applied') is not None:
                ub = [float(x) for x in nap['u_applied'][:3]]
            elif nd.vb is not None:
                # 後備：回報還沒到。記下來源，不要混充
                src, ub = 'odom_diff', [float(x) for x in nd.vb]
            else:
                src, ub = 'none', [0.0, 0.0, 0.0]
            u[0] = float(np.clip(ub[0], -a.v_base, a.v_base))
            u[1] = float(np.clip(ub[1], -a.v_base, a.v_base))
            u[2] = float(np.clip(ub[2], -a.w_base, a.w_base))
            warm_src = src
            warm_clipped = any(abs(ub[i]) > (a.v_base if i < 2 else a.w_base)
                               + 1e-12 for i in range(3))
            if owner == 'wholebody':
                phase = 'UNFOLD'
                rep['events'].append({'sim_t': nd.t_seen, 'to': 'UNFOLD',
                                      'pose': list(nd.pose[:3]),
                                      'q': list(q)})
                rep['events'][-1]['warmup_base_src'] = warm_src
                rep['events'][-1]['warmup_clipped'] = bool(warm_clipped)
                rep['warmup_base_src'] = warm_src
                rep['warmup_clipped'] = bool(warm_clipped)
                print(f'[wb] **取得控制權** @ sim {nd.t_seen:.3f} ⇒ 展開；'
                      f'起步位姿 {[round(x, 4) for x in nd.pose[:3]]}；'
                      f'暖機底盤來源 {warm_src}'
                      f'{"（已夾進速度框）" if warm_clipped else ""}',
                      flush=True)
        elif phase == 'HOLD':
            # **保持交棒** —— 展開完成後不要立刻停發。
            # 停發會讓命令鏈在 max_cmd_age_s 之後走逾時路徑（n_frozen 累加、
            # 底盤經輪級減速），那在紀錄上看起來像故障，而且中間會有一段
            # 沒有新命令的空窗。這裡持續送零速率命令（設定點不變），
            # 直到 /wb_vel_cmd 上**出現第二個發布者**（求解節點上線）
            # 才交出並結束。發布者數是真的信號，不是計時猜測。
            u = np.zeros(9)
            if a.motm:
                (u[0], u[1], u[2]), _d = approach_vref(
                    nd.pose, (px, py, pyaw), v_cap=a.motm_v_cap,
                    a_ref=a.motm_a_ref, k_yaw=a.motm_k_yaw, w_cap=a.motm_w_cap,
                    v_creep=a.motm_v_creep, overshoot=a.motm_overshoot)
            n_pub_topic = nd.count_publishers('/wb_vel_cmd')
            if (nd.solver_ready if a.solver_handshake else n_pub_topic >= 2):
                rep['events'].append({'sim_t': nd.t_seen, 'to': 'RELEASED',
                                      'n_publishers': int(n_pub_topic),
                                      'pose': list(nd.pose[:3]),
                                      'q': list(q)})
                print(f'[wb] **求解節點已上線**（/wb_vel_cmd 發布者 '
                      f'{n_pub_topic}）⇒ 交出並結束 @ sim {nd.t_seen:.3f}',
                      flush=True)
                phase = 'RELEASED'
                rep['phase'] = 'RELEASED'
                # 最後一筆展開命令已發完，求解器收到本訊號後才准發命令。
                nd.done_pub.publish(String(data=json.dumps(
                    {'phase': 'RELEASED', 'sim_t': nd.t_seen})))
                break
        elif phase == 'UNFOLD':
            # ---- 手臂：**j3 先往正向離開限位**，再動其他軸 ----
            err = Q_GRASP - q
            if not j3_cleared and q[2] < a.j3_clear_rad:
                qd = np.zeros(6)
                qd[2] = min(a.qd_max, max(0.05, 2.0 * (a.j3_clear_rad - q[2])))
            else:
                if not j3_cleared:
                    j3_cleared = True
                    rep['events'].append({'sim_t': nd.t_seen,
                                          'j3_cleared_at': float(q[2])})
                    print(f'[wb] j3 已離開限位至 {q[2]:+.4f} rad，'
                          f'開始走其餘各軸', flush=True)
                qd = np.clip(2.0 * err, -a.qd_max, a.qd_max)
            # **不得把設定點推出有效限位**：逼近邊界時把該軸的速率收掉
            for i in range(6):
                nxt = q[i] + qd[i] * dt
                if nxt < elo[i] or nxt > ehi[i]:
                    qd[i] = 0.0
            u[3:] = qd
            # ---- 底盤：收掉最後一段距離與朝向誤差 ----
            ex, ey = px - nd.pose[0], py - nd.pose[1]
            c, s = math.cos(nd.pose[2]), math.sin(nd.pose[2])
            bx, by = ex * c + ey * s, -ex * s + ey * c
            eyaw = math.atan2(math.sin(pyaw - nd.pose[2]),
                              math.cos(pyaw - nd.pose[2]))
            if a.park_fixed:
                # **PARK_FIXED**：只動手臂；底盤三軸恆為零、不做停位修正（停位由閘門前的減速段完成）
                u[0] = u[1] = u[2] = 0.0
            elif a.motm:
                # **MotM**：底盤照剖面持續朝停位走，速度只在抵達時為零；
                # 展開完成不要求底盤到位 —— 剩下的距離由求解節點在對準與
                # 夾持期間走完（同一個剖面，交接時參考速度連續）。
                (u[0], u[1], u[2]), _d = approach_vref(
                    nd.pose, (px, py, pyaw), v_cap=a.motm_v_cap,
                    a_ref=a.motm_a_ref, k_yaw=a.motm_k_yaw, w_cap=a.motm_w_cap,
                    v_creep=a.motm_v_creep, overshoot=a.motm_overshoot)
            else:
                u[0] = float(np.clip(1.0 * bx, -a.v_base, a.v_base))
                u[1] = float(np.clip(1.0 * by, -a.v_base, a.v_base))
                u[2] = float(np.clip(1.0 * eyaw, -a.w_base, a.w_base))
            done_arm = float(np.abs(err).max()) <= a.q_tol
            done_base = a.motm or a.park_fixed or (math.hypot(ex, ey) <= a.pos_tol
                                                   and abs(eyaw) <= a.yaw_tol)
            if done_arm and done_base:
                phase = 'UNFOLD_DONE'
                sl = float(np.min(np.minimum(q - elo, ehi - q)))
                rep['unfold_done'] = {
                    'sim_t': nd.t_seen, 'q': list(q),
                    'q_err_max_rad': float(np.abs(err).max()),
                    'pos_err_m': math.hypot(ex, ey), 'yaw_err_rad': eyaw,
                    'meas_min_effective_slack_rad': sl,
                    'j3_rad': float(q[2])}
                print(f'[wb] **展開完成** 關節最大差 '
                      f'{float(np.abs(err).max()):.4f} rad、'
                      f'底盤 {math.hypot(ex, ey):.4f} m／'
                      f'{math.degrees(eyaw):+.2f}°、'
                      f'實測最小餘裕 {sl:+.4f} rad', flush=True)
        elif phase == 'UNFOLD_DONE':
            u[:] = 0.0
            if a.motm:
                (u[0], u[1], u[2]), _d = approach_vref(
                    nd.pose, (px, py, pyaw), v_cap=a.motm_v_cap,
                    a_ref=a.motm_a_ref, k_yaw=a.motm_k_yaw, w_cap=a.motm_w_cap,
                    v_creep=a.motm_v_creep, overshoot=a.motm_overshoot)
            m = String()
            m.data = json.dumps({'phase': 'UNFOLD_DONE',
                                 'sim_t': nd.t_seen}, default=float)
            nd.done_pub.publish(m)
            # 立刻進保持交棒：**不停發**，等求解節點上線
            phase = 'HOLD'
            hold_t0 = nd.t_seen
            rep['events'].append({'sim_t': nd.t_seen, 'to': 'HOLD'})
            print(f'[wb] 展開完成 ⇒ **保持交棒**（持續送零速率，'
                  f'等 /wb_vel_cmd 出現第二個發布者，最多 '
                  f'{a.hold_timeout_s:.0f} s）', flush=True)

        if a.park_fixed and float(np.max(np.abs(np.asarray(u, float)[:3]))) > 0.0:
            # PARK_FIXED 下不應生成非零底盤分量；若有 ⇒ 記錄並停止（不剪成零後繼續，避免藏住漏接）
            rep['park_base_cmd_generated'] = {'phase': phase, 'u_base': [float(x) for x in u[:3]],
                                              'sim_t': nd.t_seen}
            print(f'[wb] **PARK_FIXED 下生成了非零底盤命令** {u[:3]} @ {phase} ⇒ 停止', flush=True)
            # 通知其他命令來源（執行端、任務、減速段、mission）同一條違規閂鎖
            nd.vio_pub.publish(String(data=json.dumps(
                {'why': 'wholebody_base_cmd_nonzero', 'phase': phase,
                 'u_base': [float(x) for x in u[:3]], 'sim_t': nd.t_seen})))
            for _ in range(5):
                rclpy.spin_once(nd, timeout_sec=0.02)
            phase = 'PARK_VIOLATION'
            break
        nd.send(u)
        rep['phase'] = phase
        if phase == 'HOLD' and nd.t_seen - hold_t0 > a.hold_timeout_s:
            rep['events'].append({'sim_t': nd.t_seen,
                                  'hold_timeout': True,
                                  'n_publishers': int(
                                      nd.count_publishers('/wb_vel_cmd'))})
            print(f'[wb] **保持交棒逾時** {a.hold_timeout_s:.0f} s '
                  f'仍只有 {nd.count_publishers("/wb_vel_cmd")} 個發布者 ⇒ 結束',
                  flush=True)
            break
        time.sleep(max(0.0, dt * 0.5))

    rep['trace'] = nd.trace[-4000:]
    rep['final_phase'] = phase
    rep['final_pose'] = None if nd.pose is None else list(nd.pose[:3])
    rep['final_q'] = None if nd.q is None else list(nd.q)
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1,
                  default=float)
    # 展開完成即算成功；RELEASED 代表還額外把控制交給了求解節點
    ok = phase in ('UNFOLD_DONE', 'HOLD', 'RELEASED')
    rep['released_to_solver'] = (phase == 'RELEASED')
    print(f'[wb] {"展開完成" if ok else f"**停在 {phase}**"}', flush=True)
    nd.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass   # 已經關過就不是錯
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
