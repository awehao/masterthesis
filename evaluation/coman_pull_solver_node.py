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
            self.create_subscription(String, '/coman/task_state',
                                     self._task, 10)
            self.get_logger().info(
                f'協同拉動求解節點：行程 {self.stroke*1000:.1f} mm、'
                f'拉動 {self.pull_s:.1f} s、退出 {self.retreat_m*1000:.0f} mm')

        # ---------------- 任務狀態 ----------------
        def _task(self, m):
            try:
                self.task = json.loads(m.data)
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

        # ---------------- 相位推進與逐週期求解 ----------------
        def advance(self, now_s):
            """相位只由**執行端回報的量測與狀態機旗標**推進，不自行認定。"""
            t = self.task or {}
            if self.phase == 'APPROACH' and t.get('handover_pass'):
                self.phase = 'ENGAGE_WAIT'
            elif self.phase == 'ENGAGE_WAIT' and t.get('attached'):
                self.on_attached(now_s)
                self.phase = 'PULL'
            elif self.phase == 'PULL' and now_s - self.t_pull0 >= self.pull_s:
                self.phase = 'HOLD'
            elif self.phase == 'HOLD' and t.get('hold_tracking_pass'):
                self.phase = 'RELEASE_WAIT'
            elif self.phase == 'RELEASE_WAIT' and not t.get('attached', True):
                self.phase = 'RETREAT'

        def solve(self, _ignored):
            """父迴圈每週期呼叫一次；**目標由相位決定**，不是固定值。"""
            if self.task is None:
                self.n_no_task += 1
                raise RuntimeError('尚未收到執行端任務狀態，不發命令')
            now_s = float(self.task.get('sim_t', 0.0))
            self.advance(now_s)
            tgt = self.current_target(now_s)
            if tgt is None:
                raise RuntimeError(f'相位 {self.phase} 無可用目標（缺把手位姿）')
            return super().solve(tgt)

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
