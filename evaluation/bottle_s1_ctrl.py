#!/usr/bin/env python3
"""瓶子 S1 候選控制器：繞障 → 回到可操作區 → 移動抓取 → 持瓶進出口區（純邏輯、無 Isaac）。規格：規格/瓶子S1_候選規格.md。
以 S0bController 為基礎（判定、停止、求解、放行管沿用），只加 S1 才需要的部分：
  非直線底盤參考（x 單調的路徑 y(x)，偏航 0）；導航段手臂維持備妥姿態（底盤座標任務，與底盤解耦）；展臂由**位置事件**觸發（繞障延誤不會錯過）；
  成功必須實際進入出口區且底盤停下（不沿用 S0b 的持瓶後停住）。障礙來自 obstacles_fn：本版是真值注入 CBF，**不是感知導航**；無 Raw-scan Shield。
底盤速度：導航段 v_nav、接近段 v_c、持瓶段 v_carry，皆為工程設定，不是最佳化結果。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.se2 import from_xytheta                       # noqa: E402
import bottle_ref as BR                                               # noqa: E402
import bottle_s0b_ctrl as B                                           # noqa: E402

N, DT = B.N, B.DT


class DetourPath:
    """y(x)：在 [x1, x2] 間升餘弦側移 A，其餘 y＝0。x 單調，參考沿 x 前進。"""

    def __init__(self, x1, x2, A):
        self.x1, self.x2, self.A = float(x1), float(x2), float(A)

    def y(self, x):
        if x <= self.x1 or x >= self.x2:
            return 0.0
        return self.A * 0.5 * (1.0 - math.cos(2.0 * math.pi * (x - self.x1) / (self.x2 - self.x1)))

    def dy(self, x):
        if x <= self.x1 or x >= self.x2:
            return 0.0
        w = 2.0 * math.pi / (self.x2 - self.x1)
        return self.A * 0.5 * w * math.sin(w * (x - self.x1))


class S1Controller(B.S0bController):
    def __init__(self, K, ident, T_grasp, T_ready_rel, q_ready, table_top_z, bottle_radius, x_start, x_win, path, exit_c, exit_r, obstacles_fn,
                 x_exit_stop=None, v_nav=0.15, v_c=0.03, v_carry=0.10, x_slow=-0.60, **kw):
        x_stop = exit_c[0] if x_exit_stop is None else x_exit_stop
        super().__init__(K, ident, T_grasp, T_ready_rel, q_ready, table_top_z, bottle_radius, x_start, x_win, x_stop, v_c=v_c, obstacles_fn=obstacles_fn,
                         cbf_cfg=dict(B.CBF_MISSION, **kw.pop('cbf_extra', {})), **kw)
        self.path, self.exit_c, self.exit_r = path, np.asarray(exit_c, float), float(exit_r)
        self.v_nav, self.v_carry, self.x_slow = float(v_nav), float(v_carry), float(x_slow)
        self.x_pre = float(x_win[0]) + 0.05
        self.x_blend = self.x_pre - v_c * B.T_BLEND            # 展臂起點：以接近速度倒推，位置事件觸發
        self.st.t_pre, self.st.t_cruise0 = float('inf'), 0.0
        self.blending, self.mission_done, self.t_exit_in = False, False, None

    # ---- 參考
    def _ref_window(self, xs_eff):
        xs = xs_eff + self.v_b * DT * np.arange(N + 1)
        if self.st.phase == 'CARRY':                           # 持瓶段沿錨定的 y、θ
            return super()._ref_window(xs_eff)
        X_ref = np.array([from_xytheta(float(x), self.path.y(float(x)), 0.0) for x in xs])
        xi_ref = np.array([[self.v_b, self.path.dy(float(x)) * self.v_b, 0.0] for x in xs])
        return X_ref, xi_ref

    def _rel_task(self):
        return self.st.phase == 'CARRY' or (self.st.phase == 'CRUISE' and not self.blending)

    def _T_seq(self, q, t_eff):
        if self.st.phase == 'CRUISE' and not self.blending:    # 導航段：手臂維持備妥姿態（底盤座標）
            return np.stack([self.task.T_ready_rel] * (N + 1))
        return super()._T_seq(q, t_eff)

    def _speed(self, T_tcp, x):
        if x >= self.x_end:
            return max(self.v_b - B.A_REF * 2.0 * DT, 0.0)
        if self.st.phase == 'CARRY':
            return self.v_carry
        if not self.blending:
            v = self.v_nav if x < self.x_slow else self.v_c
            return v / math.sqrt(1.0 + self.path.dy(x) ** 2)   # v 是路徑速度；參考以 x 前進
        return B.heuristic_speed(self.task, self.st, self.t, T_tcp, x, self.x_we, self.v_c)

    # ---- 相位
    def _advance(self, T_tcp, close_done, q):
        if self.st.phase == 'CRUISE' and not self.blending and q[0] >= self.x_blend:
            if self.x_ws - 0.15 <= q[0] and B.Y_TUBE[0] <= q[1] <= B.Y_TUBE[1] and abs(q[2]) <= B.YAW_TUBE:
                self.blending = True
                self.st.t_cruise0, self.st.t_pre = self.t, self.t + B.T_BLEND
                self._ev('blend_start', x=float(q[0]), y=float(q[1]))
            else:
                self.n_gate += 1
        super()._advance(T_tcp, close_done, q)

    def step(self, t, q, **kw):
        out = super().step(t, q, **kw)
        q = np.asarray(q, float)
        if np.isfinite(q).all():
            in_zone = float(np.hypot(*(q[:2] - self.exit_c))) <= self.exit_r
            if in_zone and self.t_exit_in is None and self.st.phase == 'CARRY':
                self.t_exit_in = self.t
                self._ev('exit_enter', x=float(q[0]), y=float(q[1]))
            stopped_base = self._vb_meas(q) < 0.01 if getattr(self, 'q_prev2', None) is not None else False
            if self.done and in_zone and self.t_exit_in is not None and self.v_b == 0.0 and stopped_base and self.judge.stop is None and not self.mission_done:
                self.mission_done = True
                self._ev('mission_done', x=float(q[0]), y=float(q[1]))
        out['done'] = self.mission_done
        return out

    def report(self):
        r = super().report()
        r.update(exit_reached=self.t_exit_in is not None, t_exit_in=self.t_exit_in, mission_done=self.mission_done, blending=self.blending,
                 nav_source='truth_obstacle_to_cbf', shield=False, success_s0_judge=r['success'], success=bool(r['success'] and self.mission_done))
        return r


def exit_code(rep):
    """完整任務才回 0：抓持五項成立、實際進入出口區、底盤停下。"""
    if rep.get('stop') is None and rep.get('success') is True:
        return 0
    if rep.get('stop') is None:
        return 4                                               # 未中止也未完成（未抵出口、逾時）
    return 3 if (rep.get('stop_effect') or {}).get('ok') is True else 4
