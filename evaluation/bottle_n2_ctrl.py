#!/usr/bin/env python3
"""瓶子 N2 控制器：S1Controller 子類（凍結的 S0b／S1 不改），只換導航段的路徑與進度。
規劃段的進度 σ 是弧長，參考位姿與速度方向取自 PlannedPath；直線段入口後 σ＝x，行為同 S1。
參考領先限制改成比對機器人投影到路徑的 σ（S0b 另以實測 x 夾一次，σ ≤ x 時不會綁住，綁住也只是更保守）。
路徑監看失效（被新障礙佔到、代價圖過期）⇒ 受控停止，不退回工程繞行路線、不重規劃。
"""
import numpy as np

from ammr_wholebody_mpc.se2 import from_xytheta

import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
from bottle_s0_ctrl import NPH

DT, N = S1.DT, S1.N


class S1Planned(S1.S1Controller):
    def __init__(self, *a, monitor=None, **kw):
        super().__init__(*a, **kw)
        self.monitor, self.sig_robot, self.sig_log = monitor, float(self.s_ref), []

    def _ref_window(self, xs_eff):
        if self.st.phase == 'CARRY':
            return super()._ref_window(xs_eff)
        xs = xs_eff + self.v_b * DT * np.arange(N + 1)
        P, Tn = self.path.xy(xs), self.path.tangent(xs)
        X_ref = np.array([from_xytheta(float(p[0]), float(p[1]), 0.0) for p in P])
        xi_ref = np.column_stack([self.v_b * Tn[:, 0], self.v_b * Tn[:, 1], np.zeros(N + 1)])
        return X_ref, xi_ref

    def _solve(self, q):
        if self.st.phase != 'CARRY' and self.s_ref < self.path.x_lane:
            self.sig_robot = self.path.project(q[:2], self.sig_robot)
            self.s_ref = min(self.s_ref, self.sig_robot + B.S_LEAD)
            self.sig_log.append((round(self.t, 2), round(self.s_ref, 4), round(self.sig_robot, 4)))
        return super()._solve(q)

    def step(self, t, q, **kw):
        if self.monitor is not None and self.n_step % NPH == 0 and self.judge.stop is None:
            if self.monitor.why is not None:
                w = dict(self.monitor.why)                       # 複本：監看器自己的紀錄保持完整
                self.judge._stop(float(t), w.pop('why'), **w)
        return super().step(t, q, **kw)

    def report(self):
        r = super().report()
        m = self.monitor
        r.update(planned_path=dict(L=self.path.L, sig0=self.path.sig0, x_lane=self.path.x_lane, n_pts=len(self.path.P), pts=self.path.P[::5].round(4).tolist(),
                                   monitor=None if m is None else dict(n_check=m.n_check, max_cost=m.max_cost, why=m.why),
                                   lag_max=max((r_[2] - r_[1] for r_ in self.sig_log), default=None), lead_max=max((r_[1] - r_[2] for r_ in self.sig_log), default=None)))
        return r
