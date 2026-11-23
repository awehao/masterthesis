#!/usr/bin/env python3
"""沿預測軌跡重線性化候選（版本化，2026-11-01；Codex 20261101_122750 選 (i)）。以窗口協調修正版（bottle_w1_box2_ctrl：窗口硬界限＋輪級數值接受容差）為底，凍結 G 不動。
動機（bottle_w1_box2_infeas_diag.py）：CBF 避讓使底盤偏航誤差約 0.3 rad，繞參考線性化的誤差動態失準（一步解的時域位置與精確 SE(2) 積分差 7–24 mm），
窗口列「假可行」，後來真的煞不住（22.20 s 耦合不可行；停止過程實測越窗 4.2 mm）。
本候選只在 CLOSE／LIFT（窗口開啟時）每拍做一次重線性化（不迭代）：
  名目：上一拍回傳的底盤計畫（ξ_true＋δ，真參考基準）與手臂速度序列往後移一拍（末項重複）；底盤名目軌跡＝從生效時刻初態（在途前推後）以 SE(2) 指數映射精確 rollout。
  求解：gmpc_wb.solve(X_now, X̄, ū, …, cost_ref＝(真參考 X_ref, ξ_ref))——底盤動態、CBF、窗口、輪級、手臂 TCP Jacobian 都繞名目，成本仍對原參考任務
  （名目只是線性化點，不是新目標）；手臂以名目速度序列為 u_arm_nom。回傳的 δ 換回相對真參考，控制器其他部分不必改。
接受前核對（不只看下一步；核對不等於完整安全保證）：
  (a) 完整計畫：回傳底盤計畫自生效時刻精確積分，全時域位置在窗口內（容差 1 mm）；
  (b) 煞停餘裕：回傳命令執行一個控制週期，再各軸以 a_max 有界減速到 0（精確積分），全程在窗口內（容差 1 mm）。
  任一不過 ⇒ window_coord_violation（附 plan／stopping 類別）受控停止，趁停止路徑仍在窗內時就停。
輪級、手臂限制、CBF／鬆弛、Shield、窗口、期限都不放寬。逐拍記錄：上一拍預測的生效時刻狀態對本拍實際初態的誤差、兩項核對餘裕、抬升誤差、求解耗時。
"""
import dataclasses
import math
import time

import numpy as np

from ammr_wholebody_mpc import gmpc_wb as GW
from ammr_wholebody_mpc.se2 import exp_, to_xytheta
import bottle_ref as BR
import bottle_w1_box2_ctrl as BX2
import bottle_w1_box_ctrl as BX

TOL = 1e-3
_TOP = object()


def rollout(X0, U, dt):
    X, P = np.asarray(X0, float).copy(), [np.asarray(X0, float).copy()]
    for u in U:
        X = X @ exp_(np.asarray(u, float) * dt)
        P.append(X.copy())
    return np.array(P)


def box_over(P, box):
    (xlo, xhi), (ylo, yhi) = box
    return np.maximum.reduce([xlo - P[:, 0], P[:, 0] - xhi, ylo - P[:, 1], P[:, 1] - yhi])


class GMPCRelin(GW.GMPC):
    def __init__(self, cfg, ctrl):
        super().__init__(cfg)
        self.cfg0, self.ctrl = cfg, ctrl

    def solve(self, X_now, X_ref_win, xi_ref_win, xi_prev, obstacles=None, arm=None, xi_prev2=_TOP, cost_ref=None):
        if xi_prev2 is not _TOP:                                   # gmpc_wb 內部 hold／耦合遞迴
            return super().solve(X_now, X_ref_win, xi_ref_win, xi_prev, obstacles, arm, xi_prev2=xi_prev2, cost_ref=cost_ref)
        c = self.ctrl
        box = c._box_now()
        self.cfg = self.cfg0 if box is None else dataclasses.replace(self.cfg0, base_box=box)
        N, dt = self.cfg0.N, self.cfg0.dt
        X_now = np.asarray(X_now, float)
        xi_true = np.asarray(xi_ref_win, float)
        w0 = time.perf_counter()
        nom = c._nom
        if box is not None and nom is not None:
            Ub = np.vstack([nom['base'][1:], nom['base'][-1:], nom['base'][-1:]])[:N + 1]   # 移一拍，末項重複，補到 N+1 列
            Xbar = rollout(X_now, Ub[:N], dt)
            arm2 = arm
            if arm is not None and arm.get('mode') == 'active' and nom.get('arm') is not None:
                Ua = np.vstack([nom['arm'][1:], nom['arm'][-1:]])
                arm2 = dict(arm, u_arm_nom=Ua.ravel())
            r = super().solve(X_now, Xbar, Ub, xi_prev, obstacles, arm2, cost_ref=(np.asarray(X_ref_win, float), xi_true))
            delta_true = Ub[:N] + np.asarray(r.delta_xi_all, float).reshape(N, 3) - xi_true[:N]
            r = dataclasses.replace(r, delta_xi_all=delta_true)
            relin = True
        else:
            r = super().solve(X_now, X_ref_win, xi_ref_win, xi_prev, obstacles, arm)
            relin = False
        c._post(box, X_now, xi_true, r, relin, (time.perf_counter() - w0) * 1e3)
        return r


class W1PlaceController(BX2.W1PlaceController):
    group = 'G_RELIN'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        g = GMPCRelin(self.nav, self)
        self.mpc = self.mpc_nav = self.mpc_nav0 = g
        self._nom, self._pred_next, self.rl_log, self.n_relin = None, None, [], 0

    def _post(self, box, X_now, xi_true, r, relin, ms):
        N, dt = self.nav.N, self.nav.dt
        self._wheel_log(r)
        plan = xi_true[:N] + np.asarray(r.delta_xi_all, float).reshape(N, 3)
        pred_err = None
        if self._pred_next is not None:                            # 上一拍預測的本拍生效時刻狀態 vs 本拍實際初態
            d = to_xytheta(np.linalg.inv(self._pred_next) @ X_now)
            pred_err = [float(np.hypot(d[0], d[1])), float(abs(d[2]))]
        self._pred_next = rollout(X_now, plan[:1], dt)[1]
        ua = None if r.u_arm_seq is None else np.asarray(r.u_arm_seq, float).reshape(N, -1)
        self._nom = dict(base=plan.copy(), arm=ua)
        self.n_relin += int(relin)
        if box is None:
            return
        self.n_box += 1
        P_plan = rollout(X_now, plan, dt)[1:, :2, 2]
        over_plan = float(box_over(P_plan, box).max())
        u0 = np.asarray(r.u_opt, float)[:3]
        X1 = X_now @ exp_(u0 * dt)
        v, Xs, Ps = u0.copy(), X1.copy(), [X1[:2, 2].copy()]
        amax = np.asarray(self.nav.a_max, float) * dt
        for _ in range(200):                                       # 各軸有界減速到 0
            if np.all(np.abs(v) < 1e-12):
                break
            v = np.sign(v) * np.maximum(np.abs(v) - amax, 0.0)
            Xs = Xs @ exp_(v * dt)
            Ps.append(Xs[:2, 2].copy())
        over_stop = float(box_over(np.array(Ps), box).max())
        lift_err = BR.pose_err(self.T_tcp_last, self.task.T_lift)[0] if self.st.phase == 'LIFT' and getattr(self, 'T_tcp_last', None) is not None else None
        ck = dict(getattr(r, 'arm_check', None) or {})
        self.rl_log.append(dict(t=round(float(self.t), 3), relin=relin, plan_margin=-over_plan, stop_margin=-over_stop, pred_err=pred_err, lift_err=lift_err,
                                action=str(r.accept_action), fallback=ck.get('fallback'), ms=round(ms, 1)))
        self.box_log.append((float(self.t), -over_stop, str(r.accept_action), ck.get('fallback'), lift_err, -over_plan))
        if self.box_viol is None and str(r.accept_action) != 'brake' and (over_plan > TOL or over_stop > TOL):
            self.box_viol = dict(kind='plan' if over_plan > TOL else 'stopping', plan_over=over_plan, stop_over=over_stop, accept=str(r.accept_action),
                                 fallback=ck.get('fallback'), status=str(r.status), relin=relin)

    def _box_check(self, *a, **k):                                 # 由 _post 取代（父類的單步核對不再使用）
        return None

    def report(self):
        r = super().report()
        L = self.rl_log
        pe = [x['pred_err'][0] for x in L if x['pred_err'] is not None]
        r['paired_strategy'] = dict(r['paired_strategy'], name='G_W1_relin_window', n_relin=self.n_relin, rl_log=L[-120:],
                                    pred_err_max_m=max(pe) if pe else None, plan_margin_min=min((x['plan_margin'] for x in L), default=None),
                                    stop_margin_min=min((x['stop_margin'] for x in L), default=None),
                                    check='接受前：完整計畫精確積分＋一週期後有界減速路徑都在窗口內（容差 1 mm）；非完整安全保證')
        return r
