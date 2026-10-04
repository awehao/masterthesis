#!/usr/bin/env python3
"""任務節點：發計畫、監看、在備妥時要求**滾動交棒**。

範圍（明說，以免含糊）
----------------------
**控制器是真的** —— 第一階段的 `gmpc_node` 原樣使用，不改。
`/odom` 給**真值**（本實驗的明示前提），`/plan` 由本節點依已知靜態房間產生
—— **沒有**拉進 nav2 的 planner／map_server／amcl。房間是靜態已知的，
本實驗要回答的是 MotM，不是路徑規劃。

交棒
----
計畫**止於停位**，GMPC 本來就會朝最後一個位姿減速 —— 減速用的是它自己的
行為，不另外加機制。本節點只在**備妥條件成立時**要求轉移：

    底盤進入交棒區（距停位 ≤ handover_zone）
    且實測速度已降進全身的速度框（逐軸）

判定規則不在這裡重寫 —— 由 `wgmpc_handover.check_nav_handover` 給，
本節點只負責把觀測湊齊並送出請求。
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
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.wgmpc_handover import (                 # noqa: E402
    NH_OK, NH_REASON, NavHandoverConfig, NavHandoverState,
    check_nav_handover)
from ammr_wholebody_mpc.wholebody_kinematics import (            # noqa: E402
    WholeBodyKinematics)

ARM = [f'joint{i}' for i in range(1, 7)]
AP_COLS = ['physics_step_id', 'sim_t', 'bvx_body', 'bvy_body', 'wz',
           'qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6',
           'exec_mode_code', 'cmd_age_s', 'n_recv', 'n_rejected',
           'api_applied', 'src_recv_seq', 'src_recv_sim_t']
# 保留區（櫃體＋全開抽屜），由 drawer_unit.yaml 推得；與 test_drawer_room 同
RES = (-0.300, 0.300, 0.940, 1.675)


def box_dist(p, b):
    xlo, xhi, ylo, yhi = b
    return float(math.hypot(max(xlo - p[0], 0.0, p[0] - xhi),
                            max(ylo - p[1], 0.0, p[1] - yhi)))


class Mission(Node):
    def __init__(self, a):
        super().__init__('drawer_mission')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.a = a
        lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.plan_pub = self.create_publisher(Path, a.plan_topic, lat)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', lat)
        self.ho_pub = self.create_publisher(String, '/handover/request', 10)
        # **深度 1**：這些是 50–100 Hz 的狀態流，節點處理得比它慢。
        # 用深度 10 會累積成數千步的落後 —— 實測看過 11000 步（110 s），
        # 以致要求的切換物理步早已過去。狀態流要的是**最新值**，不是歷史。
        self.create_subscription(Odometry, '/odom', self._odom, 1)
        self.create_subscription(JointState, '/joint_states', self._js, 1)
        self.create_subscription(Float64MultiArray, '/coman/applied_cmd',
                                 self._ap, 1)
        self.create_subscription(String, '/handover/state', self._hs, 1)
        self.pose = None          # (x, y, yaw, t)
        self.prev_pose = None
        self.vb = None            # (vx_body, vy_body, wz)
        self.q = None
        self.q_t = None
        self.ap = None
        self.hs = None
        self.n_odom = 0
        # **以訊息自己的時間戳為準。** 節點時鐘（/clock）與訊息時間戳不同步時，
        # 用節點時鐘做差分會得到剛好 2 倍或 0.5 倍的假速度（實測看到 0.2775
        # 與 0.5551 交替），過期判定也會跟著誤判。這裡一律用訊息帶的 sim_t，
        # 並以「看過的最大 sim_t」當作現在。
        self.t_seen = 0.0

    @staticmethod
    def _stamp_s(h):
        return float(h.stamp.sec) + float(h.stamp.nanosec) * 1e-9

    def _note_t(self, t):
        if t > self.t_seen:
            self.t_seen = float(t)
        return self.t_seen

    def _sim_t(self):
        return self.t_seen

    def _odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        t = self._note_t(self._stamp_s(m.header))
        p = (float(m.pose.pose.position.x), float(m.pose.pose.position.y),
             yaw, self._stamp_s(m.header))
        if self.pose is not None:
            dt = p[3] - self.pose[3]
            # 太小的 dt 不要用 —— 差分會被放大成假速度
            if dt > 0.5 * 0.02:
                # **位姿差分 → 本體座標**：命令是本體的，比較的對象要一致
                dx, dy = p[0] - self.pose[0], p[1] - self.pose[1]
                c, s = math.cos(yaw), math.sin(yaw)
                self.vb = ((dx * c + dy * s) / dt,
                           (-dx * s + dy * c) / dt,
                           (yaw - self.pose[2]) / dt)
        self.prev_pose = self.pose
        self.pose = p
        self.n_odom += 1

    def _js(self, m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM):
            self.q = tuple(float(d[j]) for j in ARM)
            self.q_t = self._stamp_s(m.header)
            self._note_t(self.q_t)

    def _ap(self, m):
        if len(m.data) != len(AP_COLS):
            return
        c = {k: m.data[i] for i, k in enumerate(AP_COLS)}
        self.ap = c
        self._note_t(float(c['sim_t']))

    def _hs(self, m):
        try:
            self.hs = json.loads(m.data)
        except Exception:
            pass

    # ------------------------------------------------------------ 計畫
    def publish_plan(self, start, goal, spacing=0.05, frame='odom',
                     goal_yaw=None):
        p = Path()
        p.header.frame_id = frame
        p.header.stamp = self.get_clock().now().to_msg()
        s = np.array(start, float)
        g = np.array(goal, float)
        n = max(2, int(np.linalg.norm(g - s) / spacing) + 1)
        yaw_path = math.atan2(g[1] - s[1], g[0] - s[0])
        for k in range(n):
            ps = PoseStamped()
            ps.header = p.header
            q = s + (g - s) * (k / (n - 1))
            ps.pose.position.x = float(q[0])
            ps.pose.position.y = float(q[1])
            # **最後一個位姿用停位的 yaw**，不用路徑方向 —— 停位的朝向是
            # 手臂可達性算出來的，不是走過去的方向。先前兩者差了 14.8°。
            yw = (float(goal_yaw) if (goal_yaw is not None and k == n - 1)
                  else yaw_path)
            ps.pose.orientation.z = math.sin(yw / 2)
            ps.pose.orientation.w = math.cos(yw / 2)
            p.poses.append(ps)
        gp = PoseStamped()
        gp.header = p.header
        gp.pose = p.poses[-1].pose
        # **等訂閱者出現再發，而且重發。**
        # 本端發布是 TRANSIENT_LOCAL，但 gmpc 的 /plan 訂閱是預設 VOLATILE
        # （RELIABLE/KEEP_LAST/1，沒有設 durability）。兩者相容，所以**不會**
        # 報 QoS 不符，但 VOLATILE 訂閱者收不到 latch 的回放 —— 只收現場。
        # 「latched 所以發一次就好」這個假設因此是錯的：發布端比訂閱端先就緒
        # 時這一筆就掉了。實測 `nav_handover_ctrl5_002805` 整趟
        # state='no_plan' 1144 輪、底盤一步沒動，就是輸掉這場競態。
        n_sub = 0
        t0 = time.monotonic()
        while time.monotonic() - t0 < 10.0:
            n_sub = int(self.plan_pub.get_subscription_count())
            if n_sub > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        self.plan_n_sub_at_publish = n_sub
        for _ in range(5):
            self.plan_pub.publish(p)
            self.goal_pub.publish(gp)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.2)
        return len(p.poses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='-2.80,-3.20')
    ap.add_argument('--park', default='-0.136412,0.56')
    ap.add_argument('--plan-topic', default='/plan')
    ap.add_argument('--handover-zone', type=float, default=0.30)
    ap.add_argument('--slow-zone', type=float, default=0.80,
                    help='進入此距離後把導航的 v_nominal 降進全身的速度框，'
                         '讓交棒發生在**仍在移動**時')
    ap.add_argument('--v-min', type=float, default=0.010,
                    help='**滾動交棒的下界**：底盤至少要以這個速度在動才交棒。'
                         '設 0 則允許停車交棒')
    ap.add_argument('--slow-v', type=float, default=0.030,
                    help='慢速區的 v_nominal；必須在全身的速度框內')
    ap.add_argument('--park-yaw', type=float, default=1.297349)
    ap.add_argument('--switch-lead-steps', type=int, default=5,
                    help='切換步要排在觀測到的物理步之後多少步')
    ap.add_argument('--max-glide-retries', type=int, default=20,
                    help='減速段切換的重試上限；超過即中止並記錄')
    ap.add_argument('--glide-entry', type=float, default=0.60,
                    help='離停車點這麼近就把控制權轉給減速交接段')
    ap.add_argument('--switch-window-steps', type=int, default=400,
                    help='窗口預約的長度（物理步）。執行端在這個窗口內每一步'
                         '照原樣查條件，第一個成立的步才提交；不放寬條件')
    ap.add_argument('--switch-retry-after-steps', type=int, default=15,
                    help='排定的切換步過了這麼多步還沒發生就重試')
    ap.add_argument('--max-handover-retries', type=int, default=40,
                    help='重試上限；超過即中止並記錄')
    ap.add_argument('--stale-step-abort', type=int, default=200,
                    help='觀測到的物理步落後執行端超過這麼多就中止 —— '
                         '那表示訂閱累積了歷史，讀到的不是現況')
    ap.add_argument('--stow-q', default='0,0,0,0,-1.5707963,0')
    ap.add_argument('--timeout-s', type=float, default=300.0)
    ap.add_argument('--urdf', default=os.path.join(
        WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    start = tuple(float(v) for v in a.start.split(','))
    park = tuple(float(v) for v in a.park.split(','))
    stow = tuple(float(v) for v in a.stow_q.split(','))

    K = WholeBodyKinematics.from_urdf_file(a.urdf)
    lim = np.array(K.joint_limits())
    cfg = NavHandoverConfig(joint_lower=tuple(float(x) for x in lim[0, 3:]),
                            joint_upper=tuple(float(x) for x in lim[1, 3:]),
                            joint_margin=0.05,
                            handover_zone_m=a.handover_zone,
                            v_min_lin_mps=a.v_min)

    rclpy.init()
    nd = Mission(a)
    rep = {'start': start, 'park': park, 'handover_zone': a.handover_zone,
           'scope': ('控制器是真的 gmpc_node；/odom 為真值；'
                     '/plan 由本節點依已知靜態房間產生，未用 nav2 planner'),
           'events': []}
    t0 = time.monotonic()
    # 等 odom
    while rclpy.ok() and time.monotonic() - t0 < 30.0 and nd.n_odom == 0:
        rclpy.spin_once(nd, timeout_sec=0.05)
    if nd.n_odom == 0:
        print('[mission] **收不到 /odom** ⇒ 中止', flush=True)
        return 2
    n = nd.publish_plan(nd.pose[:2], park, goal_yaw=a.park_yaw)
    print(f'[mission] 計畫已發：{n} 點，{nd.pose[0]:.3f},{nd.pose[1]:.3f} '
          f'→ {park[0]:.3f},{park[1]:.3f}', flush=True)
    rep['plan_points'] = n
    rep['plan_n_sub_at_publish'] = getattr(nd, 'plan_n_sub_at_publish', None)
    if not rep['plan_n_sub_at_publish']:
        print('[mission] **發計畫時沒有任何訂閱者** —— gmpc 可能收不到，'
              '這一趟的 no_plan 要先排除', flush=True)

    requested = False
    glided = False
    slowed = False
    last_code = None
    from rcl_interfaces.srv import SetParameters
    from rcl_interfaces.msg import Parameter as ParamMsg, ParameterValue
    from rcl_interfaces.msg import ParameterType
    cli = nd.create_client(SetParameters, '/gmpc_controller/set_parameters')

    def set_v_nom(v):
        if not cli.wait_for_service(timeout_sec=2.0):
            return False
        pm = ParamMsg()
        pm.name = 'v_nominal'
        pm.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                  double_value=float(v))
        req = SetParameters.Request(parameters=[pm])
        fut = cli.call_async(req)
        t = time.monotonic()
        while rclpy.ok() and time.monotonic() - t < 3.0 and not fut.done():
            rclpy.spin_once(nd, timeout_sec=0.02)
        return bool(fut.done() and fut.result()
                    and fut.result().results[0].successful)
    t1 = time.monotonic()
    while rclpy.ok() and time.monotonic() - t1 < a.timeout_s:
        rclpy.spin_once(nd, timeout_sec=0.02)
        # 計畫在 publish_plan 裡已經等到訂閱者並重發過 5 次，這裡不再補發
        if nd.pose is None or nd.vb is None or nd.q is None or nd.ap is None:
            continue
        d_park = math.hypot(nd.pose[0] - park[0], nd.pose[1] - park[1])
        owner = (nd.hs or {}).get('owner')
        # **落後偵測**：執行端在 /handover/state 報自己的步數；
        # 兩者差太多表示訂閱累積了歷史，讀到的不是現況，要求的切換步會過期。
        exec_step = (nd.hs or {}).get('step')
        if exec_step is not None and nd.ap is not None:
            lag = int(exec_step) - int(nd.ap['physics_step_id'])
            rep['max_step_lag'] = max(rep.get('max_step_lag', 0), lag)
            if lag > a.stale_step_abort:
                rep['abort'] = (f'**觀測落後執行端 {lag} 個物理步** ⇒ '
                                f'讀到的不是現況，中止')
                print(f'[mission] {rep["abort"]}', flush=True)
                break
        if owner == 'wholebody':
            rep['handover_done'] = True
            rep['handover_step'] = (nd.hs or {}).get('step')
            print(f'[mission] **控制權已轉移給全身** @ step '
                  f'{rep["handover_step"]}', flush=True)
            break
        if not slowed and d_park <= a.slow_zone:
            okset = set_v_nom(a.slow_v)
            slowed = True
            rep['slow_zone_entered'] = {'d_park_m': round(d_park, 4),
                                        'v_nominal': a.slow_v, 'ok': okset}
            print(f'[mission] 進入慢速區 d={d_park:.3f} m ⇒ '
                  f'v_nominal → {a.slow_v}（{"成功" if okset else "**失敗**"}）',
                  flush=True)
        # ---- 轉給減速交接段 ----------------------------------------------
        # 導航負責大範圍逼近，減速段負責沿路徑把速度降進全身的速度框並
        # **維持**。兩者共用直寫路徑與同一個變化率上限基準，所以這一次切換
        # 的連續性是結構上保證的。
        if (not glided and owner == 'nav' and d_park <= a.glide_entry
                and exec_step is not None):
            at = int(exec_step) + a.switch_lead_steps
            msg = String()
            msg.data = json.dumps({'to': 'glide', 'at_step': at,
                                   'window_steps': a.switch_window_steps})
            nd.ho_pub.publish(msg)
            glided = True
            rep['glide_requested'] = {'d_park_m': round(d_park, 4),
                                      'at_step': at,
                                      'window_steps': a.switch_window_steps}
            print(f'[mission] d={d_park:.3f} m ⇒ **要求轉給減速交接段**'
                  f'（物理步 {at}，窗口 {a.switch_window_steps}）', flush=True)
            continue
        if not glided or owner == 'nav':
            # 還沒交給減速段之前不評估全身交棒 —— 導航還在全速逼近，
            # 評了也只會一直記「超出速度框」，把紀錄灌滿無用的列。
            #
            # **但要能重試。** 減速段的切換若被取消（窗口內它始終沒發出
            # 新鮮命令），控制權留在導航；只在這裡 continue 會讓整段安靜
            # 地卡住，跟先前「只請求一次就一直等」是同一個缺陷。
            # **要等執行端確實越過請求步才算失敗。**
            # 只看「pending 是 None」會在狀態還沒更新時就觸發 —— 實測請求
            # 排在物理步 3435，節點看到的執行端卻還是 3430，於是每一圈都
            # 重試一次，12 次用完就中止，而執行端根本還沒評估過。
            _gat = (rep.get('glide_requested') or {}).get('at_step')
            if (glided and owner == 'nav'
                    and (nd.hs or {}).get('pending') is None
                    and exec_step is not None and _gat is not None
                    and int(exec_step) > int(_gat)
                    + a.switch_retry_after_steps):
                rep['n_glide_retries'] = rep.get('n_glide_retries', 0) + 1
                if rep['n_glide_retries'] > a.max_glide_retries:
                    rep['abort'] = (f'**減速段切換重試 {a.max_glide_retries} '
                                    f'次仍未成功** ⇒ 中止')
                    print(f'[mission] {rep["abort"]}', flush=True)
                    break
                glided = False
                print(f'[mission] 減速段切換未發生（執行端 {exec_step}）⇒ '
                      f'**重試第 {rep["n_glide_retries"]} 次**；'
                      f'上一次未備妥：'
                      f'{(nd.hs or {}).get("not_ready_last")}', flush=True)
            continue

        # **切換可能被執行層取消**（切換當步的速度核對不過）。
        # 只請求一次就一直等，系統會安靜地卡住 —— 要能重試。
        if requested:
            # **窗口還開著就不是失敗。** 執行端在窗口內逐步重查條件，
            # 這時重發請求只會換來「已有排程」而把計數器推高，看起來像
            # 一直失敗。只有執行端自己把排程清掉（取消）才該重試。
            pend = (nd.hs or {}).get('pending')
            if pend is not None:
                rep['handover_pending_seen'] = pend
                rep['handover_n_steps_not_ready'] = (
                    (nd.hs or {}).get('n_steps_not_ready'))
                rep['handover_not_ready_last'] = (
                    (nd.hs or {}).get('not_ready_last'))
                continue
            if (exec_step is not None
                    and int(exec_step) > rep['handover_requested_at_step']
                    + a.switch_retry_after_steps):
                requested = False
                rep['n_handover_retries'] = rep.get('n_handover_retries', 0) + 1
                print(f'[mission] 切換未在第 '
                      f'{rep["handover_requested_at_step"]} 步發生'
                      f'（執行端已到 {exec_step}）⇒ **重試第 '
                      f'{rep["n_handover_retries"]} 次**', flush=True)
                if rep['n_handover_retries'] > a.max_handover_retries:
                    rep['abort'] = (f'**切換重試 {a.max_handover_retries} 次'
                                    f'仍未成功** ⇒ 中止')
                    print(f'[mission] {rep["abort"]}', flush=True)
                    break
            continue
        st = NavHandoverState(
            t_now=nd._sim_t(),
            # **現任的直寫控制者**：導航或減速交接段都算。
            # 導航的成本函數是「到達目標並煞停」，實測交棒區內可切窗口只有
            # 2 步（0.02 s）；減速段存在就是為了把那個窗口做寬。
            nav_in_control=(owner in ('nav', 'glide')), incumbent=owner,
            wgmpc_armed=False,
            nav_u_applied=tuple(float(nd.ap[k]) for k in
                                ('bvx_body', 'bvy_body', 'wz', 'qd1', 'qd2',
                                 'qd3', 'qd4', 'qd5', 'qd6')),
            nav_applied_t=float(nd.ap['sim_t']),
            nav_applied_step=int(nd.ap['physics_step_id']),
            base_vx_body=nd.vb[0], base_vy_body=nd.vb[1],
            base_yaw_rate_rps=nd.vb[2], base_pose_t=nd.pose[3],
            park_dist_m=d_park,
            reserved_clearance_m=box_dist(nd.pose[:2], RES) - 0.33,
            q_meas=nd.q, q_meas_t=nd.q_t, stow_q=stow,
            exec_mode=int(nd.ap['exec_mode_code']),
            n_recv=int(nd.ap['n_recv']),
            n_rejected=int(nd.ap['n_rejected']),
            exec_report_t=float(nd.ap['sim_t']), setpoint=None)
        v = check_nav_handover(st, cfg)
        if v.code != last_code:
            last_code = v.code
            rep['events'].append({'sim_t': round(nd._sim_t(), 3),
                                  'code': v.code, 'why': v.why,
                                  'park_dist_m': round(d_park, 4),
                                  'vb': [round(x, 5) for x in nd.vb]})
            print(f'[mission] d={d_park:.3f} m  v={nd.vb[0]:+.4f},'
                  f'{nd.vb[1]:+.4f},{nd.vb[2]:+.4f}  → 碼 {v.code} {v.why}',
                  flush=True)
        if v.ok:
            # 切換步要留夠的餘裕：請求經話題送到執行端也要時間。
            at = int(nd.ap['physics_step_id']) + a.switch_lead_steps
            # **窗口預約，不是指定單一步。**
            # 本節點看到的狀態落後執行端數十步（實測一次請求到達時執行端已
            # 多走 16 步），而條件成立的窗口只有一個 GMPC 週期寬（實測
            # 0.18 s＝18 個物理步）。指定單一步會一直錯過：三趟只中一趟。
            # 窗口**不放寬任何接手條件** —— 速度框、滾動下界、就緒檢查與
            # 預核都由執行端在每一步照原樣查，只是允許在窗口內逐步重試，
            # 提交的物理步一樣明確記錄。
            msg = String()
            msg.data = json.dumps({'to': 'wholebody', 'at_step': at,
                                   'window_steps': a.switch_window_steps})
            nd.ho_pub.publish(msg)
            requested = True
            rep['handover_requested_at_step'] = at
            rep['handover_window_steps'] = a.switch_window_steps
            rep['handover_requested_from_step'] = int(nd.ap['physics_step_id'])
            rep['handover_seed'] = v.detail.get('seed_for_executor')
            print(f'[mission] **備妥 ⇒ 要求在物理步 {at} 轉移控制權**；'
                  f'u_prev 來源 {v.detail["u_prev_source"]}', flush=True)
    rep['last_gate_code'] = last_code
    rep['last_gate_why'] = NH_REASON.get(last_code)
    rep['final_pose'] = nd.pose
    rep['final_vb'] = nd.vb
    rep['handover_state'] = nd.hs
    if a.out:
        json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1,
                  default=str)
    ok = bool(rep.get('handover_done'))
    print(f'[mission] {"交棒完成" if ok else "**交棒未完成**"}；'
          f'最後閘門 碼 {last_code} {NH_REASON.get(last_code)}', flush=True)
    nd.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass   # 已經關過就不是錯
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
