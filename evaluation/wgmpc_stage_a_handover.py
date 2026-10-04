#!/usr/bin/env python3
"""階段 A 的交棒閘門（執行期）：前置調姿 → W-GMPC。

判定規則在 `ammr_wholebody_mpc.wgmpc_handover`，與離線反例共用同一份。
本檔只負責**把觀測湊齊**：完成宣告、實測關節角、設定點、實際套用回報，
每一項都帶自己的時間戳。

通過 ⇒ 回傳碼 0，並把 `u_prev`（取自實際套用回報）寫進輸出 JSON，
供運行器以 `--u-prev-init` 傳給 W-GMPC。
不通過 ⇒ 回傳碼 72，**不讓 W-GMPC 上線**，運行器據此停止。
"""
from __future__ import annotations

import argparse
import json
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

from ammr_wholebody_mpc.wgmpc_handover import (                 # noqa: E402
    HandoverConfig, HandoverState, HO_REASON, check_handover)
from ammr_wholebody_mpc.wholebody_kinematics import (            # noqa: E402
    WholeBodyKinematics)

ARM = [f'joint{i}' for i in range(1, 7)]


class Gate(Node):
    def __init__(self, a):
        super().__init__('wgmpc_stage_a_handover')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.a = a
        self.done = None
        self.done_t = None
        self.q_meas = None
        self.q_meas_t = None
        self.sp = None
        self.sp_t = None
        self.u_app = None
        self.u_app_t = None
        self.api_applied = None
        self.exec_mode = None
        self.cmd_age_s = None
        self._ap_badlen = None
        self._sp_badlen = None
        _lat = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, a.done_topic, self._done, _lat)
        self.create_subscription(JointState, '/joint_states', self._js, 10)
        self.create_subscription(Float64MultiArray, a.setpoint_topic,
                                 self._sp, 10)
        self.create_subscription(Float64MultiArray, a.applied_topic,
                                 self._ap, 10)

    def _sim_t(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _done(self, m: String):
        try:
            self.done = json.loads(m.data)
        except Exception as e:
            self.get_logger().error(f'完成宣告無法解析：{e}')
            self.done = None
            return
        self.done_t = self._sim_t()

    def _js(self, m: JointState):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM):
            self.q_meas = tuple(float(d[j]) for j in ARM)
            self.q_meas_t = self._sim_t()

    def _sp(self, m: Float64MultiArray):
        # [step_id, sim_t, ready, sp1..sp6, exec_mode, ...]
        d = list(m.data)
        if len(d) < 11:
            self._sp_badlen = len(d)
            return
        ready = float(d[2]) >= 0.5
        if not ready:
            # **設定點尚未建立**：不當成「有值但是 NaN」，直接不更新
            return
        self.sp = tuple(float(x) for x in d[3:9])
        self.sp_t = float(d[1])          # 以**模擬時間**為時間戳，不用牆鐘

    # /coman/applied_cmd_meta 宣告的欄位順序（18 欄）
    AP_COLS = ['physics_step_id', 'sim_t', 'bvx_body', 'bvy_body', 'wz',
               'qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6',
               'exec_mode_code', 'cmd_age_s', 'n_recv', 'n_rejected',
               'api_applied', 'src_recv_seq', 'src_recv_sim_t']

    def _ap(self, m: Float64MultiArray):
        d = list(m.data)
        if len(d) != len(self.AP_COLS):
            # 欄位數不符 ⇒ **整筆不採用**。欄位順序是契約，長度不對就不能
            # 按位置取值（否則 api_applied 會取到別的欄位）。
            self._ap_badlen = len(d)
            return
        c = {k: d[i] for i, k in enumerate(self.AP_COLS)}
        self.u_app = tuple(float(c[k]) for k in
                           ('bvx_body', 'bvy_body', 'wz',
                            'qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6'))
        self.u_app_t = float(c['sim_t'])
        self.exec_mode = int(float(c['exec_mode_code']))
        self.api_applied = float(c['api_applied']) >= 0.5
        self.cmd_age_s = float(c['cmd_age_s'])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default=os.path.join(
        WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--done-topic', default='/wgmpc/prepos_done')
    # **完成宣告優先讀檔案。** latched 訊息只在發布端存活時服務後加入者；
    # 運行器為了保證「兩個來源不同時在線」會**先殺 prepos、再起本閘門**，
    # 於是那則宣告已經沒有發布者 —— 實測 done_received=False，而其餘三項
    # （實測角、設定點、套用回報）都是新鮮的。
    # 檔案裡的宣告與話題同一份內容（wgmpc_prepos_node 的 declare_done），
    # 所以「明確宣告，不是沉默」這個要求不受影響。
    ap.add_argument('--done-file', default='',
                    help='prepos.json；存在即以它為完成宣告的來源')
    # 前置來源**已解除**要可驗證，不是由呼叫端斷言。
    ap.add_argument('--prepos-pid', type=int, default=0,
                    help='前置節點的 PID；核對它已不存在')
    ap.add_argument('--setpoint-topic', default='/coman/arm_setpoint')
    ap.add_argument('--applied-topic', default='/coman/applied_cmd')
    ap.add_argument('--max-cmd-age-s', type=float, default=0.2)
    ap.add_argument('--quiet-s', type=float, default=0.4)
    ap.add_argument('--max-state-age-s', type=float, default=0.1)
    ap.add_argument('--joint-margin', type=float, default=0.05)
    ap.add_argument('--setpoint-meas-tol-rad', type=float, default=0.01)
    ap.add_argument('--prepos-goal-tol-rad', type=float, default=0.02,
                    help='前置調姿的到達容差；擋「命令發了但手臂沒動」')
    ap.add_argument('--timeout-s', type=float, default=60.0,
                    help='等觀測湊齊的牆鐘上限；逾時視為交棒失敗')
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    K = WholeBodyKinematics.from_urdf_file(a.urdf)
    lim = np.array(K.joint_limits())
    cfg = HandoverConfig(
        max_cmd_age_s=a.max_cmd_age_s, quiet_s=a.quiet_s,
        max_state_age_s=a.max_state_age_s, joint_margin=a.joint_margin,
        joint_lower=tuple(float(x) for x in lim[0, 3:]),
        joint_upper=tuple(float(x) for x in lim[1, 3:]),
        setpoint_meas_tol_rad=a.setpoint_meas_tol_rad,
        prepos_goal_tol_rad=a.prepos_goal_tol_rad, n_dof=9)
    cfg.validate()

    # ---- 完成宣告：檔案優先 ----
    done_file = None
    if a.done_file and os.path.exists(a.done_file):
        try:
            _d = json.load(open(a.done_file))
            done_file = _d.get('done')
            if not done_file:
                print(f'**{a.done_file} 沒有 done 區塊**', file=sys.stderr)
        except Exception as e:
            print(f'**{a.done_file} 無法解析**：{e!r}', file=sys.stderr)
    # ---- 前置來源是否真的解除 ----
    prepos_alive = None
    if a.prepos_pid > 0:
        prepos_alive = os.path.exists(f'/proc/{a.prepos_pid}')
        if prepos_alive:
            print(f'**前置節點 PID {a.prepos_pid} 仍存在** ⇒ 兩個來源會同時在線，'
                  f'拒絕交棒', file=sys.stderr)
            if a.out:
                json.dump({'verdict': 'fail', 'why': '前置來源尚未解除',
                           'prepos_pid': a.prepos_pid, 'prepos_alive': True},
                          open(a.out, 'w'), ensure_ascii=False, indent=1)
            return 72

    rclpy.init()
    nd = Gate(a)
    rep = {'quiet_s_required': a.quiet_s,
           'max_cmd_age_s': a.max_cmd_age_s,
           'joint_margin': a.joint_margin}
    rc = 72
    try:
        t0 = time.monotonic()
        verdict = None
        last_code = None
        while rclpy.ok() and time.monotonic() - t0 < a.timeout_s:
            rclpy.spin_once(nd, timeout_sec=0.05)
            _done = nd.done or done_file
            if _done is None:
                continue
            st = HandoverState(
                t_now=nd._sim_t(),
                prepos_done=True,
                prepos_last_cmd_t=_done.get('last_cmd_sim_t'),
                # 前置節點在宣告完成後即解除；運行器已先終止它。
                # 這裡以「宣告已收到 ＋ 運行器保證已終止」為依據，
                # **不是**以「沒聽到命令」推論。
                prepos_armed=False,
                wgmpc_armed=False,
                q_meas=nd.q_meas, q_meas_t=nd.q_meas_t,
                setpoint=nd.sp, setpoint_t=nd.sp_t,
                u_applied=nd.u_app, u_applied_t=nd.u_app_t,
                exec_mode=nd.exec_mode, api_applied=nd.api_applied,
                prepos_goal=tuple(_done['q_goal'])
                if _done.get('q_goal') else None)
            v = check_handover(st, cfg)
            last_code = v.code
            if v.ok:
                verdict = (v, st)
                break
        if verdict is None:
            rep['verdict'] = 'fail'
            rep['code'] = last_code
            rep['why'] = (HO_REASON.get(last_code, '等不到完成宣告')
                          if last_code is not None else
                          ('等不到完成宣告（話題與檔案都沒有）'
                           if done_file is None else '等不到有效狀態'))
            rep['observed'] = {
                'done_received_topic': nd.done is not None,
                'done_from_file': done_file is not None,
                'done_file': a.done_file or None,
                'prepos_pid': a.prepos_pid or None,
                'prepos_alive': prepos_alive,
                'q_meas': nd.q_meas, 'q_meas_t': nd.q_meas_t,
                'setpoint': nd.sp, 'setpoint_t': nd.sp_t,
                'u_applied': nd.u_app, 'u_applied_t': nd.u_app_t,
                'exec_mode': nd.exec_mode, 'api_applied': nd.api_applied,
                'applied_bad_field_count': getattr(nd, '_ap_badlen', None),
                'setpoint_bad_field_count': getattr(nd, '_sp_badlen', None)}
            print(f'[階段A] **交棒失敗**：{rep["why"]} '
                  f'（碼 {rep["code"]}）⇒ 不讓 W-GMPC 上線', flush=True)
            return 72
        v, st = verdict
        rep.update({
            'verdict': 'pass', 'code': v.code, 'why': v.why,
            'u_prev': list(v.u_prev), 'detail': v.detail,
            'prepos_done': nd.done or done_file,
            'done_source': ('topic' if nd.done is not None else
                            ('file' if done_file is not None else None)),
            'prepos_pid': a.prepos_pid or None,
            'prepos_alive_at_gate': prepos_alive,
            'exec_mode_at_handover': nd.exec_mode,
            'api_applied_at_handover': nd.api_applied,
            'q_meas_at_handover': list(st.q_meas),
            'setpoint_at_handover': list(st.setpoint),
            'handover_sim_t': st.t_now})
        print(f'[階段A] 交棒通過：安靜 {v.detail["quiet_s"]:.3f} s、'
              f'設定點與實測角最大差 '
              f'{v.detail["setpoint_meas_max_dev_rad"]*1e3:.3f} mrad、'
              f'u_prev 來源 {v.detail["u_prev_source"]}', flush=True)
        print(f'[階段A] u_prev = {np.round(v.u_prev, 6).tolist()}', flush=True)
        rc = 0
    except KeyboardInterrupt:
        rc = 130
    finally:
        if a.out:
            json.dump(rep, open(a.out, 'w'), ensure_ascii=False, indent=1,
                      default=str)
        nd.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
