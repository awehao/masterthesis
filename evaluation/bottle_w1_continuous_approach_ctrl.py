#!/usr/bin/env python3
"""Howard 2026-11-01：依進場進度展臂，不提前展完後保持備妥姿態。

版本化 SHROW 子類；不修改凍結控制器、QP、速度、安全線或視覺／相位閘。
原本的固定時間展臂在 x_blend 之前可能早已完成。本候選把同一條
q_tuck -> q_arm_ready 的關節參考參數化為 x_unfold -> x_blend 的進度，
抵達既有朝瓶起點才完成展臂。仍使用原 min-jerk，然後由父類朝瓶。
若底盤因安全限制停下，展臂也可能暫停；不宣稱任何情況皆不停頓。

時域內沿原導航參考預覽進度，已量測的展臂進度不因倒退而回退。
這不是新的抓取目標來源；備妥姿態仍是任務名目先驗。
預設不會被既有 runner 使用。continuous_unfold=False 完整走父類路徑。
"""
import numpy as np

import bottle_ref as BR
import bottle_w1_shrow_ctrl as SR


def approach_progress(x, start, finish):
    """未套 min-jerk 的位置進度；非有限資料或倒置區間直接拒絕。"""
    if not np.isfinite([x, start, finish]).all() or finish <= start:
        raise ValueError('continuous approach requires finite x and start < finish')
    return float(np.clip((x - start) / (finish - start), 0.0, 1.0))


class W1PlaceController(SR.W1PlaceController):
    group = 'G_SHROW_CONTINUOUS_APPROACH'

    def __init__(self, *a, continuous_unfold=True, **kw):
        super().__init__(*a, **kw)
        self.continuous_unfold = bool(continuous_unfold)
        self.approach_measured = 0.0
        self.approach_log = []
        if self.continuous_unfold and self.q_tuck is not None:
            approach_progress(self.x_unfold, self.x_unfold, self.x_blend)

    def step(self, t, q, **kw):
        if self.continuous_unfold and self.q_tuck is not None and self.st.phase == 'CRUISE' and not self.blending:
            x = float(np.asarray(q)[0])
            if np.isfinite(x):
                self.approach_measured = max(self.approach_measured, approach_progress(x, self.x_unfold, self.x_blend))
            elif self.judge.stop is None:
                self.judge._stop(float(t), 'continuous_approach_state_invalid')
        return super().step(t, q, **kw)

    def _unfolded(self):
        if not self.continuous_unfold or self.q_tuck is None:
            return super()._unfolded()
        return self.t_u0 is not None and self.approach_measured >= 1.0

    def _rel_target(self, tau):
        if not self.continuous_unfold or self.q_tuck is None:
            return super()._rel_target(tau)
        if self.t_u0 is None:
            progress = 0.0
        else:
            sig = self.s_ref + self.v_b * max(float(tau) - self.t, 0.0)
            # PlannedPath uses sigma, whereas DetourPath uses world x.
            x_ref = float(self.path.xy(sig)[0, 0]) if hasattr(self.path, 'xy') else float(sig)
            progress = max(self.approach_measured, approach_progress(x_ref, self.x_unfold, self.x_blend))
        qa = self.q_tuck + (self.q_arm_ready - self.q_tuck) * BR.min_jerk(progress)
        return self.K.fk(np.r_[0.0, 0.0, 0.0, qa], 'link_tcp')

    def _advance_inner(self, T_tcp, close_done, q):
        if self.continuous_unfold and self.st.phase == 'CRUISE' and not self.blending:
            self.approach_log.append((float(self.t), float(q[0]), self.approach_measured))
        return super()._advance_inner(T_tcp, close_done, q)

    def report(self):
        r = super().report()
        r['continuous_approach'] = dict(
            enabled=self.continuous_unfold,
            x_unfold=float(self.x_unfold), x_blend=float(self.x_blend),
            measured_progress=self.approach_measured, progress_log=self.approach_log,
            scope='只將固定時間展臂改為進場位置進度；預抓對位、兩路視覺、閉爪及安全判準不變',
            validation_scope='物理與渲染驗證狀態由各趟紀錄判定，本模組不自行宣告',
        )
        return r
