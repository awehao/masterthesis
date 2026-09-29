"""協同拉抽屜的**求解節點**：既有單步全身 QP ＋ 相位化的夾爪目標。

**重用**：運動學、ROS 介面、QP、guard、stop 與命令發布全部沿用
`wholebody_pregrasp.WholeBody`（凍結不動）。本檔只覆寫**目標從哪裡來**。

命令鏈路（**不繞過安全濾波**）：
    本節點 → /wholebody_safety/cmd_in → 安全濾波 → /wholebody_safety/cmd_out
           → arm_vel_adapter → /wb_vel_cmd → 協同執行端（E2 命令鏈）

目標如何生成（見 coman_pull_target.py）：
    接近段  夾爪目標 = 把手當下世界位姿 × 設計抓取關係的逆
    拉動段  夾爪目標 = 把手目標(s) × **連接當下**抓取關係的逆
    保持段  s 維持在行程
    退出段  由連接位姿沿工具 +z 退開

**尚未執行過任何趟次。** 底盤自由度、障礙物設定與接觸配對規則仍待裁決。
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from coman_pull_policy import PullTaskPolicy, TaskState            # noqa: E402
from coman_pull_target import PullTarget                          # noqa: E402

PHASES = ('APPROACH', 'ENGAGE_WAIT', 'PULL', 'HOLD', 'RELEASE_WAIT', 'RETREAT',
          'DONE')


def load_base():
    path = os.path.join(HERE, 'wholebody_pregrasp.py')
    spec = importlib.util.spec_from_file_location('wbp_base_pull', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['wbp_base_pull'] = mod
    spec.loader.exec_module(mod)
    return mod


def quat_R(q):
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def homog(p, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def build(M, cl):
    from std_msgs.msg import String

    class PullSolver(M.WholeBody):
        """只換目標來源；約束集沿用父類別（**真實障礙物，不是空場景**）。"""

        def __init__(self, a):
            super().__init__(a)
            self.phase = 'APPROACH'
            self.task = None            # 執行端的最新任務狀態
            self.pull = None            # PullTarget（連接後建立）
            self.t_pull0 = None
            self.grasp_offset = np.array([0.0, 0.0,
                                          -float(cl.tcp_offset_z)])
            self.R_des = np.array(json.loads(cl.grasp_rot), dtype=float)
            self.stroke = float(cl.stroke_m)
            self.pull_s = float(cl.pull_duration_s)
            self.retreat_m = float(cl.retreat_m)
            self.n_no_task = 0
            self.task_recv_sim = None
            # **與安全層同一份** d0 覆寫；不一致即上下游規則不同，趟次無效
            _pd = {}
            for _spec in [x for x in cl.pair_d0.split(',') if x.strip()]:
                _lk, _v = _spec.split(':')
                _fv = float(_v)
                if not (_fv > 0.0):
                    raise ValueError(f'pair_d0 值必須為正：{_spec!r}')
                _pd[_lk] = _fv
            self.cfg.d0_by_link = _pd
            if _pd:
                self.get_logger().warn(
                    f'**求解端逐連桿 d0 覆寫**（一般值 {self.cfg.d0}）：{_pd}')
            # **任務配時與新鮮度一律用模擬時間**；牆鐘只作程序監看（逾時／熱）
            from rclpy.parameter import Parameter
            self.set_parameters([Parameter('use_sim_time',
                                           Parameter.Type.BOOL, True)])
            self.create_subscription(String, '/coman/task_state',
                                     self._task, 10)
            self.get_logger().info(
                f'協同拉動求解節點：行程 {self.stroke*1000:.1f} mm、'
                f'拉動 {self.pull_s:.1f} s、退出 {self.retreat_m*1000:.0f} mm')

        # ---------------- 任務狀態 ----------------
        def sim_now(self) -> float:
            """**模擬時間**（use_sim_time ＋ 執行端發布的 /clock）。"""
            return self.get_clock().now().nanoseconds * 1e-9

        def _task(self, m):
            try:
                self.task = json.loads(m.data)
                self.task_recv_sim = self.sim_now()
            except Exception:           # noqa: BLE001
                self.task = None

        def handle_world(self):
            t = self.task
            if not t or 'handle_pos' not in t:
                return None
            return homog(np.array(t['handle_pos'], float),
                         quat_R(t['handle_quat']))

        def gripper_world(self):
            t = self.task
            if not t or 'gripper_pos' not in t:
                return None
            return homog(np.array(t['gripper_pos'], float),
                         quat_R(t['gripper_quat']))

        # ---------------- 目標 ----------------
        def approach_target(self):
            """接近段：由**把手當下位姿**與設計抓取關係反推夾爪目標。"""
            Th = self.handle_world()
            if Th is None:
                return None
            # 設計抓取：桿心位於工具 z = −tcp_offset_z，姿態為 R_des
            T = np.eye(4)
            T[:3, :3] = self.R_des
            T[:3, 3] = Th[:3, 3] - self.R_des @ self.grasp_offset
            return T

        def current_target(self, now_s):
            if self.phase in ('APPROACH', 'ENGAGE_WAIT'):
                return self.approach_target()
            if self.pull is None:
                return None
            if self.phase == 'PULL':
                return self.pull.gripper_target(now_s)
            if self.phase in ('HOLD', 'RELEASE_WAIT'):
                return self.pull.gripper_target(self.t_pull0 + self.pull_s)
            if self.phase == 'RETREAT':
                T = self.pull.gripper_target(self.t_pull0 + self.pull_s).copy()
                T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.array(
                    [0.0, 0.0, -self.retreat_m])
                return T
            return None

        # ---------------- 任務迴圈 ----------------
        def run(self, _T_des_ignored):
            """**另寫的任務迴圈**：父類別的 run() 是為固定目標寫的，
            誤差小於容差就完成 —— 拉動任務沿用會在**接近點就結束**。

            這裡重複了父迴圈的四件事：guard、solve、發布、deadline 排程與逐週期記錄；
            其餘（QP、運動學、停止處置）仍呼叫父類別，父檔**未修改**。
            相位與結束條件一律交給 `PullTaskPolicy`（已離線測試 16 項）。
            """
            import time as _t
            from std_msgs.msg import Float64MultiArray as _F64
            a = self.a
            self.base0 = self.base.copy()
            self.q_pref = np.array(a.posture, dtype=float)
            pol = PullTaskPolicy(pull_duration_s=self.pull_s,
                                 retreat_clear_m=float(cl.retreat_clear_m),
                                 tol_p=a.tol_p, tol_r=a.tol_r)
            period = 1.0 / a.rate
            t0 = _t.monotonic()
            slot = 0
            retreat_ref = None
            v_last = np.zeros(9)
            while True:
                why = self.guard()
                if why:
                    self.stop()
                    print(f'  中止（guard）：{why}', flush=True)
                    return False
                # **牆鐘逾時＝程序監看**，與任務配時分開；任務配時全部用模擬時間
                if _t.monotonic() - t0 > a.timeout_s:
                    self.stop()
                    print(f'  程序逾時（牆鐘）{a.timeout_s:.0f} s，'
                          f'相位 {pol.phase}', flush=True)
                    return False
                if self.task is None:
                    self.n_no_task += 1
                    self.stop()          # **沒有狀態就不發命令**
                    self.exec.spin_once(timeout_sec=0.01)
                    continue
                now_s = float(self.task.get('sim_t', 0.0))
                # 狀態年齡＝**模擬時間**之差（兩端同一時鐘源）
                age = (self.sim_now() - float(self.task.get('sim_t', 0.0))
                       if self.task_recv_sim is not None else float('inf'))
                Tg = self.gripper_world()
                if pol.phase == 'RETREAT' and retreat_ref is None and Tg is not None:
                    retreat_ref = (Tg[:3, 3].copy(), Tg[:3, :3] @ np.array([0, 0, 1.0]))
                r_signed = (float((Tg[:3, 3] - retreat_ref[0]) @ retreat_ref[1])
                            if (retreat_ref is not None and Tg is not None) else 0.0)
                st = TaskState(
                    sim_t=now_s, state_age_s=age,
                    attached=bool(self.task.get('attached')),
                    handover_pass=bool(self.task.get('handover_pass')),
                    hold_tracking_pass=bool(self.task.get('hold_tracking_pass')),
                    decouple_confirmed=bool(self.task.get('decouple_confirmed')),
                    pos_err_m=float(getattr(self, 'ep_last', 1.0)),
                    rot_err_rad=float(getattr(self, 'er_last', 1.0)),
                    cmd_max_abs=float(np.abs(v_last).max()),
                    retreat_signed_m=r_signed,
                    emergency=bool(self.task.get('emergency')))
                prev_phase = pol.phase
                dec = pol.step(st)
                if prev_phase == 'ENGAGE_WAIT' and dec['phase'] == 'PULL':
                    self.on_attached(now_s)
                self.phase = dec['phase']
                if dec['abort']:
                    self.stop()
                    print(f"  中止：{dec['abort']}（{dec['reason']}）", flush=True)
                    return False
                if dec['done']:
                    self.stop()
                    print(f"  完成：{dec['reason']}", flush=True)
                    return True
                tgt = self.current_target(now_s)
                if tgt is None:
                    self.stop()
                    self.exec.spin_once(timeout_sec=0.01)
                    continue
                try:
                    v, T, ep, er = super().solve(tgt)
                except RuntimeError as exc:
                    self.stop()
                    print(f'  中止（fail closed）：{exc}', flush=True)
                    return False
                self.ep_last, self.er_last = ep, er
                m = _F64()
                m.data = [float(x) for x in v]
                self.pub.publish(m)
                self.v_prev = v.copy()
                v_last = np.asarray(v, float)
                self.log.append(dict(t=now_s, phase=pol.phase, ep=float(ep),
                                     er=float(er),
                                     cmd=[float(x) for x in v]))
                slot += 1
                target = t0 + slot * period
                while _t.monotonic() < target:
                    self.exec.spin_once(timeout_sec=0.002)

        def on_attached(self, now_s):
            """連接當下建立 PullTarget：**用實際量到的**夾爪與把手位姿。"""
            Tg, Th = self.gripper_world(), self.handle_world()
            if Tg is None or Th is None:
                raise RuntimeError('連接時缺夾爪或把手位姿，無法建立拉動目標')
            axis = np.array(json.loads(cl.slide_axis), float)
            self.pull = PullTarget(Tg, Th, axis, self.stroke, self.pull_s, now_s)
            self.t_pull0 = now_s
            self.get_logger().info('已由連接當下的抓取關係建立拉動目標')

    return PullSolver


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--stroke-m', type=float, default=0.020)
    ap.add_argument('--pull-duration-s', type=float, default=4.0)
    ap.add_argument('--retreat-m', type=float, default=0.040)
    ap.add_argument('--tcp-offset-z', type=float, default=0.0147)
    ap.add_argument('--slide-axis', default='[0.0, -1.0, 0.0]')
    ap.add_argument('--pair-d0', default='',
                    help="逐連桿 d0 覆寫，格式 'link:value,link:value'；"
                         '必須與安全層參數一致')
    ap.add_argument('--retreat-clear-m', type=float, default=0.0233,
                    help='退出完成的實測門檻（沿退出起始方向的有號位移）')
    ap.add_argument('--grasp-rot',
                    default='[[-1,0,0],[0,0,1],[0,1,0]]')
    cl, rest = ap.parse_known_args()
    M = load_base()
    M.WholeBody = build(M, cl)
    argv = [sys.argv[0], '--solver', 'qp', '--out', cl.out] + rest
    old, sys.argv = sys.argv, argv
    try:
        return M.main()
    finally:
        sys.argv = old


if __name__ == '__main__':
    raise SystemExit(main())
