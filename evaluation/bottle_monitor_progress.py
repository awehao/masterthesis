#!/usr/bin/env python3
"""導航段路徑監看的獨立進度（Codex 20261101_093009；基礎設施修正，兩種策略共用，不是放寬避障條件）。
缺陷（bottle_w1_dyn2_g1／np1）：bottle_n2_ctrl.S1Planned 只在 s_ref < x_lane 時更新 sig_robot；參考領先先進直線段，sig_robot 凍結在入口前 2.6 cm（σ −0.926），
PathMonitor 之後每次代價圖更新都檢查機器人早已走過的殘段 [−0.926, −0.9)，移動物從後方斜穿時誤觸 path_blocked。
本物件只供監看用：以實測位置持續投影（單調不減，搜尋窗不往回），控制參考 s_ref、sig_robot、速度與 QP 都不動。
結束條件：投影 σ ≥ x_lane（沿末段抵達／通過入口，xy 在入口後是直線段 y＝0）且實測 y 在既有放行側向管 Y_TUBE（bottle_s0b_ctrl）內 ⇒ σ 設成 x_lane，
remaining_pts 為空、過期核對也停止（check_age 只在 σ < x_lane 時做）。量測非有限 ⇒ 不推進也不結束。
呼叫順序：每步先 update(實測)，再處理代價圖回呼（PlanRos.sig_fn）與 check_age。
"""
import numpy as np


class MonitorProgress:
    def __init__(self, path, y_tube):
        self.path, self.y_tube = path, (float(y_tube[0]), float(y_tube[1]))
        self.sig, self.done, self.t_done, self.n_bad = float(path.sig0), False, None, 0

    def update(self, q, t=None):
        if self.done:
            return self.sig
        q = np.asarray(q, float)
        if not np.isfinite(q[:2]).all():
            self.n_bad += 1
            return self.sig
        s = float(self.path.project(q[:2], self.sig, back=0.0, ahead=0.6))
        self.sig = max(self.sig, s)
        if self.sig >= self.path.x_lane - 1e-9 and self.y_tube[0] <= float(q[1]) <= self.y_tube[1]:
            self.sig, self.done, self.t_done = float(self.path.x_lane), True, t
        elif self.sig >= self.path.x_lane:
            self.sig = float(self.path.x_lane) - 1e-6                     # 投影過入口但側向不在管內：不結束，入口前最後一格仍檢查
        return self.sig

    def report(self):
        return dict(sig=self.sig, done=self.done, t_done=self.t_done, n_bad=self.n_bad, y_tube=self.y_tube,
                    note='導航段監看獨立進度：實測投影、抵達入口且在側向管內才結束；控制參考不受影響')
