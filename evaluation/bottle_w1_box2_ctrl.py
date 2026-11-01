#!/usr/bin/env python3
"""窗口協調候選修正版（bottle_w1_box_ctrl 的版本化子類，2026-11-01；Codex 20261101_095306）。
bottle_w1_dyn2b_g1 停止的具體錯誤（bottle_w1_box_wheel_diag.py 重解）：耦合 QP 原始底盤命令與上一筆都貼在輪速邊界（殘差 +3e-11、+1.1e-9 rad/s，上限 5.55），
輪加速度遠低於上限；gmpc_wb.fit_to_wheels 的線段縮放因 (w_lim − b) 捨入為負而 λ＝0，命令被換回上一筆、標成後處理改動 ⇒ 退回停止。不是真正的限制違反。
本版只多一項：gmpc_wb.wheel_fit_boundary_tol＝1e-6 m/s（沿用 fit_to_wheels 既有的 tol 參數值）——原始命令在兩組輪級界內（含此容差）時不做後處理；
超出容差的真正違反照原線段縮放並被標成後處理改動、仍會退回停止（不放寬、不忽略 seq_consistent）。導航與停車兩個求解器都套用。
逐拍記錄回傳命令的輪速殘差（rad/s，正值＝在容差內接受的邊界量）。抬升誤差卡在約 22 mm 的問題另列，本版不處理。
"""
import dataclasses

import numpy as np

from ammr_wholebody_mpc import gmpc_wb as GW
import bottle_w1_box_ctrl as BX

TOL_WHEEL = 1e-6


class GMPCBox2(BX.GMPCBox):
    def solve(self, X_now, X_ref_win, xi_ref_win, xi_prev, obstacles=None, arm=None, xi_prev2=BX._TOP):
        r = super().solve(X_now, X_ref_win, xi_ref_win, xi_prev, obstacles, arm, xi_prev2)
        if xi_prev2 is BX._TOP:
            self.ctrl._wheel_log(r)
        return r


class W1PlaceController(BX.W1PlaceController):
    group = 'G_BOX2'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.nav = dataclasses.replace(self.nav, wheel_fit_boundary_tol=TOL_WHEEL)
        g = GMPCBox2(self.nav, self)
        self.mpc = self.mpc_nav = self.mpc_nav0 = g
        self.mpc_park = GW.GMPC(dataclasses.replace(self.nav, u_min=np.zeros(3), u_max=np.zeros(3)))
        self.wl = dict(n=0, n_pos=0, max_resid=-np.inf, actions={})

    def _wheel_log(self, r):
        w = GW.wheel_speeds(np.asarray(r.u_opt, float), self.nav)
        res = float(np.max(np.abs(w)) - self.nav.wheel_w_max)
        L = self.wl
        L['n'] += 1
        L['n_pos'] += int(res > 0.0)
        L['max_resid'] = max(L['max_resid'], res)
        a = str(r.accept_action)
        L['actions'][a] = L['actions'].get(a, 0) + 1

    def report(self):
        r = super().report()
        r['paired_strategy'] = dict(r['paired_strategy'], name='G_W1_window_box_wheeltol', wheel_fit_boundary_tol=TOL_WHEEL, wheel_log=self.wl)
        return r
