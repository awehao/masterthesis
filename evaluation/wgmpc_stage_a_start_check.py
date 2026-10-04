#!/usr/bin/env python3
"""階段 A 的**起始構型核對**（執行期，START_MODE=spawn）。

這條路徑**不做前置調姿**：機器人直接生成在起始構型（`--init-arm-q`），
起始位置不去調 j3。

**它不是交棒判定，判準組成不同，兩者不可混用。** 交棒判定核的是
「前置來源已宣告完成、已解除、安靜夠久」；這裡沒有前置來源，所以那三條
不適用。輸出一律帶 `gate: start_posture`，報告不得寫成交棒通過。

能核的是（與交棒判定**判準組成不同**，不可混用）：
  * 實測關節角有效、新鮮、在有效限位內；
  * 實測角**真的等於要求的起始構型** —— 擋「生成參數沒生效」；
  * 設定點有效、在有效限位內、與實測角相符；
  * `u_prev` 取自**實際套用回報**（與交棒判定相同）；
  * 實際套用回報顯示手臂**確實靜止**（交棒判定不核這條）。

不能核的是：「沒有別的手臂命令來源」。運行器在 spawn 模式下不啟動前置
節點，那是**結構性質**（由 test_wgmpc_stage_a_runner.py 的不變量把住），
不是這個閘門量到的。

**修訂史**：初版假設此路徑下手臂「從未被命令過」，於是用初始靜止假設的
零值當 u_prev。實測否證 —— adapter 一上線就持續送零命令
（wgmpc_stage_a_spawn_151022：n_recv=289、exec_mode=0、設定點已建立，
碼 26 攔下）。既然取得到實際套用回報，就不需要那個假設，閘門也因此變強。
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
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))

from ammr_wholebody_mpc.wgmpc_handover import (                 # noqa: E402
    HandoverConfig, SS_REASON, StartPostureState, check_start_posture)
from ammr_wholebody_mpc.wholebody_kinematics import (            # noqa: E402
    WholeBodyKinematics)

ARM = [f'joint{i}' for i in range(1, 7)]

# /coman/applied_cmd 宣告的欄位順序（18 欄）。長度是契約：不符即整筆不採用。
AP_COLS = ['physics_step_id', 'sim_t', 'bvx_body', 'bvy_body', 'wz',
           'qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6',
           'exec_mode_code', 'cmd_age_s', 'n_recv', 'n_rejected',
           'api_applied', 'src_recv_seq', 'src_recv_sim_t']


class Gate(Node):
    def __init__(self, a):
        super().__init__('wgmpc_stage_a_start_check')
        self.set_parameters([rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.a = a
        self.q_meas = None
        self.q_meas_t = None
        self.sp = None
        self.sp_t = None
        self.sp_ready_seen = False
        self.u_app = None
        self.u_app_t = None
        self.exec_mode = None
        self.api_applied = None
        self.n_recv = None
        self.n_rejected = None
        self._ap_badlen = None
        self._sp_badlen = None
        self.create_subscription(JointState, '/joint_states', self._js, 10)
        self.create_subscription(Float64MultiArray, a.setpoint_topic,
                                 self._sp, 10)
        self.create_subscription(Float64MultiArray, a.applied_topic,
                                 self._ap, 10)

    def _sim_t(self):
        return self.get_clock().now().nanoseconds * 1e-9

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
        if float(d[2]) < 0.5:
            # **設定點尚未建立** ⇒ 不當成「有值但是 NaN」，直接不更新
            return
        self.sp = tuple(float(x) for x in d[3:9])
        self.sp_t = float(d[1])      # 以**模擬時間**為時間戳，不用牆鐘
        self.sp_ready_seen = True

    def _ap(self, m: Float64MultiArray):
        d = list(m.data)
        if len(d) != len(AP_COLS):
            # 欄位數不符 ⇒ **整筆不採用**（欄位順序是契約）
            self._ap_badlen = len(d)
            return
        c = {k: d[i] for i, k in enumerate(AP_COLS)}
        self.exec_mode = int(float(c['exec_mode_code']))
        self.n_recv = int(float(c['n_recv']))
        self.n_rejected = int(float(c['n_rejected']))
        self.api_applied = float(c['api_applied']) >= 0.5
        self.u_app = tuple(float(c[k]) for k in
                           ('bvx_body', 'bvy_body', 'wz',
                            'qd1', 'qd2', 'qd3', 'qd4', 'qd5', 'qd6'))
        self.u_app_t = float(c['sim_t'])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--urdf', default=os.path.join(
        WS, 'evaluation/models/omni_bot_wholebody_expanded.urdf'))
    ap.add_argument('--start-q', required=True,
                    help='要求的起始構型，六個逗號分隔值（= --init-arm-q）')
    ap.add_argument('--setpoint-topic', default='/coman/arm_setpoint')
    ap.add_argument('--applied-topic', default='/coman/applied_cmd')
    ap.add_argument('--max-state-age-s', type=float, default=0.1)
    ap.add_argument('--joint-margin', type=float, default=0.05)
    ap.add_argument('--start-tol-rad', type=float, default=0.02,
                    help='實測角與要求起始構型的容差；擋「生成參數沒生效」')
    ap.add_argument('--setpoint-meas-tol-rad', type=float, default=0.01)
    ap.add_argument('--rest-tol', type=float, default=1e-3,
                    help='套用回報的分量上限；起始位置必須是靜止起點')
    ap.add_argument('--timeout-s', type=float, default=60.0)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    try:
        start_q = tuple(float(v) for v in a.start_q.split(','))
    except Exception as e:
        print(f'**--start-q 無法解析**：{e!r}', file=sys.stderr)
        return 70
    if len(start_q) != 6:
        print(f'**--start-q 需要 6 個值，收到 {len(start_q)}**', file=sys.stderr)
        return 70

    K = WholeBodyKinematics.from_urdf_file(a.urdf)
    lim = np.array(K.joint_limits())
    cfg = HandoverConfig(
        max_state_age_s=a.max_state_age_s, joint_margin=a.joint_margin,
        joint_lower=tuple(float(x) for x in lim[0, 3:]),
        joint_upper=tuple(float(x) for x in lim[1, 3:]),
        setpoint_meas_tol_rad=a.setpoint_meas_tol_rad, n_dof=9)
    cfg.validate()

    rclpy.init()
    nd = Gate(a)
    rep = {'gate': 'start_posture',
           'not_a_handover': '判準組成與交棒判定不同，不可混用',
           'start_q': list(start_q),
           'start_tol_rad': a.start_tol_rad,
           'setpoint_meas_tol_rad': a.setpoint_meas_tol_rad,
           'rest_tol': a.rest_tol,
           'joint_margin': a.joint_margin}
    rc = 72
    try:
        t0 = time.monotonic()
        verdict = None
        last_code = None
        while rclpy.ok() and time.monotonic() - t0 < a.timeout_s:
            rclpy.spin_once(nd, timeout_sec=0.05)
            st = StartPostureState(
                t_now=nd._sim_t(), q_meas=nd.q_meas, q_meas_t=nd.q_meas_t,
                wgmpc_armed=False, start_q=start_q,
                setpoint=nd.sp, setpoint_t=nd.sp_t,
                u_applied=nd.u_app, u_applied_t=nd.u_app_t,
                exec_mode=nd.exec_mode, api_applied=nd.api_applied,
                n_recv=nd.n_recv, n_rejected=nd.n_rejected)
            v = check_start_posture(st, cfg, start_tol_rad=a.start_tol_rad,
                                    rest_tol=a.rest_tol)
            last_code = v.code
            if v.ok:
                verdict = (v, st)
                break
        if verdict is None:
            rep.update({
                'verdict': 'fail', 'code': last_code,
                'why': SS_REASON.get(last_code, '等不到有效狀態'),
                'observed': {
                    'q_meas': nd.q_meas, 'q_meas_t': nd.q_meas_t,
                    'setpoint_ready_seen': nd.sp_ready_seen,
                    'setpoint': nd.sp, 'setpoint_t': nd.sp_t,
                    'u_applied': nd.u_app, 'u_applied_t': nd.u_app_t,
                    'exec_mode': nd.exec_mode, 'api_applied': nd.api_applied,
                    'n_recv': nd.n_recv, 'n_rejected': nd.n_rejected,
                    'applied_bad_field_count': nd._ap_badlen,
                    'setpoint_bad_field_count': nd._sp_badlen}})
            print(f'[階段A] **起始構型核對失敗**：{rep["why"]} '
                  f'（碼 {rep["code"]}）⇒ 不讓 W-GMPC 上線', flush=True)
            return 72
        v, st = verdict
        rep.update({
            'verdict': 'pass', 'code': v.code, 'why': v.why,
            'u_prev': list(v.u_prev), 'detail': v.detail,
            'q_meas_at_gate': list(st.q_meas),
            # 下游（目標計算）沿用這個鍵名，與交棒路徑一致
            'handover_sim_t': st.t_now,
            'gate_sim_t': st.t_now})
        print(f'[階段A] 起始構型核對通過（**非交棒**）：'
              f'與要求構型最大差 {v.detail["start_max_dev_rad"]*1e3:.3f} mrad、'
              f'設定點與實測角最大差 '
              f'{v.detail["setpoint_meas_max_dev_rad"]*1e3:.3f} mrad、'
              f'有效限位最小餘裕 {v.detail["min_effective_slack_rad"]:+.6f} rad',
              flush=True)
        print(f'[階段A] 手臂靜止：套用回報最大分量 '
              f'{v.detail["arm_rest_max_abs_u"]:.2e}（上限 '
              f'{v.detail["rest_tol"]:.1e}）；執行端 '
              f'exec_mode={v.detail["exec_mode"]}、n_recv='
              f'{v.detail["n_recv"]}、n_rejected={v.detail["n_rejected"]}',
              flush=True)
        print(f'[階段A] u_prev = {np.round(v.u_prev, 6).tolist()}'
              f'（來源：{v.detail["u_prev_source"]} —— 取自實際套用回報）',
              flush=True)
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
