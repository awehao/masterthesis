#!/usr/bin/env python3
"""W1 持瓶段順暢候選（版本化子類，2026-11-01；Codex 20261101_082911 回覆、reviews/20261101_codex_experiment_execution.md 末節）。
原 G（bottle_w1_ctrl.W1PlaceController）不動；本類只在放置相位 CARRY_NAV 時，在同一個 GMPC QP 加底盤輸入二階差分成本
Σ Δ²u_kᵀ S2 Δ²u_k（gmpc_wb.S2；Δ²u_k = u_k − 2u_{k−1} + u_{k−2}），u_{−1}、u_{−2} 對齊預測起點之前兩筆已下達（在途）的底盤命令。
既有 S（一階 Δu 成本）在 G 為 0，本候選不動它；二階差分成本不是新方法，只是本平台的一個候選設定。
無因次化：S2_ii = λ · R_ii · (u_max,i / (a_max,i · dt))² —— 一步內打滿加速度的二階差分（a_max·dt）所付成本，等於 λ 倍「整個速度上限的偏差」在 R 下的成本。
λ 只有一個尺度參數，離線依序試 (0.03, 0.1, 0.3)，第一個符合事前要求者凍結（test_bottle_w1_s2.py）。
速度、加速度、幾何、安全線、手腕期限、抓持與放置證據都不改；命令不做事後濾波（安全核對仍針對同一 QP 的解）。
"""
import dataclasses

import numpy as np

from ammr_wholebody_mpc import gmpc_wb as GW
import bottle_w1_ctrl as W1

LAMBDAS = (0.03, 0.1, 0.3)
_TOP = object()


def s2_matrix(cfg, lam):
    d = np.diag(cfg.R) * (np.asarray(cfg.u_max, float) / (np.asarray(cfg.a_max, float) * cfg.dt)) ** 2
    return float(lam) * np.diag(d)


class GMPCJerk(GW.GMPC):
    """CARRY_NAV 時改用含 S2 的設定並帶入 u_{−2}；其他相位設定與輸入都和原 G 相同（S2 不加）。
    gmpc_wb 內部對 hold／耦合問題的遞迴呼叫會明確帶 xi_prev2，這裡只在最外層決定一次。"""

    def __init__(self, cfg, S2, ctrl):
        super().__init__(cfg)
        self.cfg0, self.cfg_s2, self.ctrl = cfg, dataclasses.replace(cfg, S2=np.array(S2, float)), ctrl

    def solve(self, X_now, X_ref_win, xi_ref_win, xi_prev, obstacles=None, arm=None, xi_prev2=_TOP):
        if xi_prev2 is _TOP:
            on = getattr(self.ctrl, 'pph', None) == 'CARRY_NAV' and bool(np.any(self.cfg_s2.S2))
            self.cfg = self.cfg_s2 if on else self.cfg0
            xi_prev2 = np.array(self.ctrl.u_prev2[:3], float) if on else None
            self.ctrl.n_s2 += int(on)
        return super().solve(X_now, X_ref_win, xi_ref_win, xi_prev, obstacles, arm, xi_prev2=xi_prev2)


class W1PlaceController(W1.W1PlaceController):
    group = 'G_S2'

    def __init__(self, *a, s2_lambda=0.0, **kw):
        super().__init__(*a, **kw)
        assert self.mpc is self.mpc_nav and self.mpc is self.mpc_nav0, '預期 S0b／S1／Place 的導航 GMPC 是同一物件'
        self.s2_lambda = float(s2_lambda)
        self.S2 = s2_matrix(self.nav, self.s2_lambda)
        g = GMPCJerk(self.nav, self.S2, self)
        self.mpc = self.mpc_nav = self.mpc_nav0 = g
        self.u_prev2, self._up_last, self.n_s2 = np.array(self.u_prev, float).copy(), None, 0

    def _solve(self, q):
        up = np.array(self.u_prev, float).copy()                  # u_{−1}＝最新一筆已下達（在途）命令；u_{−2}＝上一個控制週期的那一筆
        self.u_prev2 = up if self._up_last is None else self._up_last
        self._up_last = up
        return super()._solve(q)

    def report(self):
        r = super().report()
        r['paired_strategy'] = dict(name='G_W1_carry_S2', s2_lambda=self.s2_lambda, S2_diag=np.diag(self.S2).tolist(), n_s2_solves=self.n_s2,
                                    scope='同 G；只在 CARRY_NAV 加底盤輸入二階差分成本（同一 QP），其他相位與 G 相同',
                                    nondim='S2_ii = λ·R_ii·(u_max,i/(a_max,i·dt))²')
        return r
