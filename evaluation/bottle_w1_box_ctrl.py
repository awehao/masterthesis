#!/usr/bin/env python3
"""任務窗口與避障的預測協調候選（版本化子類，2026-11-01；Codex 20261101_094327 方向 A）。凍結 G（bottle_w1_ctrl）不動。
缺口：抓取窗口原本只靠進窗閘與實測越界後的 window_passed 停止，沒有進入預測求解；第二案（後方斜向追越）抬升中被 CBF 推出窗口。
本候選只在 CLOSE／LIFT（抓持後、CARRY 前；放置相位尚未開始）把已驗證的操作窗口——x ∈ [x_ws, x_we]、y ∈ Y_TUBE（bottle_s0b_ctrl）——
以 gmpc_wb.base_box 硬界限放進同一個時域 QP（第 1..N 步、含第一未來步；初態為在途前推後的生效時刻狀態，與 CBF、手臂同一問題）。
CARRY 之後解除。原偏航、角落准入、越窗停止判定、CBF（含鬆弛規則）、Shield、抓持證據、視覺期限、淨空門檻全部不改；x／y 矩形不是完整操作可行域。
回傳核對：每次求解後，把實際回傳的底盤命令（r.u_opt，含後處理與退回解）從生效時刻初態以 SE(2) 指數映射執行一個控制週期，位置須在窗口內（容差 1 mm）；
不滿足 ⇒ window_coord_violation 受控停止（不當作有效協調解）。回傳計畫的全時域精確積分只作診斷記錄：離線第二案時域末段（第 20 步）出現一階線性化誤差 2.2 mm，
而每拍只執行第一步、下一拍重解，以全時域判停會把線性化誤差當成違反。窗口列不可行時由 gmpc_wb 既有的煞停／退回路徑處理。
逐拍記錄：窗口是否開啟、預測窗口最小餘裕、回傳是否為退回解、抬升誤差（LIFT 時 TCP 對抬升位姿）。CBF 鬆弛與殘差照原 compact 紀錄。
"""
import dataclasses

import numpy as np

from ammr_wholebody_mpc import gmpc_wb as GW
from ammr_wholebody_mpc.se2 import exp_
import bottle_ref as BR
import bottle_s0b_ctrl as B
import bottle_w1_ctrl as W1

TOL = 1e-3
_TOP = object()


class GMPCBox(GW.GMPC):
    def __init__(self, cfg, ctrl):
        super().__init__(cfg)
        self.cfg0, self.ctrl = cfg, ctrl

    def solve(self, X_now, X_ref_win, xi_ref_win, xi_prev, obstacles=None, arm=None, xi_prev2=_TOP):
        if xi_prev2 is not _TOP:                                   # gmpc_wb 內部的 hold／耦合遞迴：沿用最外層已選定的設定
            return super().solve(X_now, X_ref_win, xi_ref_win, xi_prev, obstacles, arm, xi_prev2=xi_prev2)
        box = self.ctrl._box_now()
        self.cfg = self.cfg0 if box is None else dataclasses.replace(self.cfg0, base_box=box)
        r = super().solve(X_now, X_ref_win, xi_ref_win, xi_prev, obstacles, arm)
        self.ctrl._box_check(box, np.asarray(X_now, float), np.asarray(xi_ref_win, float), r)
        return r


class W1PlaceController(W1.W1PlaceController):
    group = 'G_BOX'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        assert self.mpc is self.mpc_nav and self.mpc is self.mpc_nav0, '預期 S0b／S1／Place 的導航 GMPC 是同一物件'
        g = GMPCBox(self.nav, self)
        self.mpc = self.mpc_nav = self.mpc_nav0 = g
        self.box_log, self.box_viol, self.n_box = [], None, 0

    def _box_now(self):
        if self.st.phase in ('CLOSE', 'LIFT') and self.pph is None:
            return ((float(self.x_ws), float(self.x_we)), (float(B.Y_TUBE[0]), float(B.Y_TUBE[1])))
        return None

    def _box_check(self, box, X_now, xi_ref_win, r):
        if box is None:
            return
        self.n_box += 1
        N, dt = self.nav.N, self.nav.dt
        plan = xi_ref_win[:N] + np.asarray(r.delta_xi_all, float).reshape(N, 3)
        X, P = X_now.copy(), []
        for k in range(N):
            X = X @ exp_(plan[k] * dt)
            P.append(X[:2, 2].copy())
        P = np.array(P)
        (xlo, xhi), (ylo, yhi) = box

        def over_of(Q):
            return np.maximum.reduce([xlo - Q[:, 0], Q[:, 0] - xhi, ylo - Q[:, 1], Q[:, 1] - yhi])
        over = over_of(P)                                          # 全時域精確積分：診斷用（後段含一階線性化誤差，下一拍會重解），只記錄
        X1 = X_now @ exp_(np.asarray(r.u_opt, float)[:3] * dt)     # 實際回傳的命令（含後處理／退回）從生效時刻執行一個控制週期
        over1 = float(over_of(X1[:2, 2][None, :])[0])
        ck = dict(getattr(r, 'arm_check', None) or {})
        lift_err = None
        if self.st.phase == 'LIFT' and getattr(self, 'T_tcp_last', None) is not None:
            lift_err = BR.pose_err(self.T_tcp_last, self.task.T_lift)[0]
        self.box_log.append((float(self.t), -over1, str(r.accept_action), ck.get('fallback'), lift_err, -float(over.max())))
        if over1 > TOL and self.box_viol is None and str(r.accept_action) != 'brake':   # brake＝gmpc_wb 既有煞停路徑（命令歸零），只記錄
            self.box_viol = dict(viol=over1, horizon_max_over=float(over.max()), accept=str(r.accept_action), fallback=ck.get('fallback'), status=str(r.status))

    def _solve(self, q):
        self.T_tcp_last = self.K.fk(np.asarray(q, float), 'link_tcp')
        u = super()._solve(q)
        if self.box_viol is not None and self.judge.stop is None:
            self.judge._stop(self.t, 'window_coord_violation', **self.box_viol)
            return None
        return u

    def report(self):
        r = super().report()
        L = np.array([(x[0], x[1], x[5]) for x in self.box_log], float).reshape(-1, 3)
        r['paired_strategy'] = dict(name='G_W1_window_box', n_box_solves=self.n_box, cmd_margin_min=None if not len(L) else float(L[:, 1].min()),
                                    horizon_margin_min=None if not len(L) else float(L[:, 2].min()),
                                    n_fallback_in_box=sum(1 for x in self.box_log if x[3]), violation=self.box_viol,
                                    box_log=[[round(x[0], 3), round(x[1], 5), x[2], x[3], None if x[4] is None else round(x[4], 5), round(x[5], 5)] for x in self.box_log],
                                    check='回傳命令一個控制週期精確核對（判定）＋全時域精確積分餘裕（診斷）',
                                    scope='凍結 G＋CLOSE／LIFT 時操作窗口 x∈[x_ws,x_we]、y∈Y_TUBE 作為同一時域 QP 的底盤位置硬界限（含在途前推初態）；回傳計畫精確積分核對；其餘不變')
        return r
