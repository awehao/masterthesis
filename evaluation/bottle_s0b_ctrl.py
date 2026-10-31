#!/usr/bin/env python3
"""瓶子 S0b 控制器：低速移動抓取（純邏輯、無 Isaac；Isaac 腳本與假受控對象測試共用同一份）。規格：規格/瓶子S0_物理抓持設計.md §11。
核心是 **gmpc_wb**（統一導航複本，導航 SE(2) 誤差問題＋手臂設定點模型＋TCP 任務，N＝20、dt 0.05，每週期兩次 QP），不是局部 FR 核心；
本趟結果才算這個耦合核心的物理驗證。場景無障礙，obstacles＝None，**CBF 未介入**。
判定、接觸資料、相位事件、抓持證據、手指與 d̂ 全部沿用 S0aController／S0Judge，只覆寫：
  求解（_solve）、未來參考（_T_seq）、相位推進（_advance：可達窗與 CARRY 參考錨定）、停止減速（_arm_decel：底盤用導航加速度界）、輸出（step：底盤命令走同一延遲佇列）。
底盤速度是**啟發式進度調速**（工程基線，只減速不加速，v_b ≤ v_c），不是 GMPC 最佳化出的進度協調。
任何 fallback（含 base_brake）一律走既有停止路徑，不把退回解當可繼續的命令。
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import gmpc_wb as GW                          # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp as S                     # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                  # noqa: E402
from ammr_wholebody_mpc.se2 import exp_, from_xytheta                 # noqa: E402
import bottle_ref as BR                                               # noqa: E402
import bottle_s0_ctrl as C                                            # noqa: E402

N, DT = 20, 0.05
T_CLOSE, T_EV, T_LIFT, T_LIFT_SETTLE = 0.5, 1.0, 1.0, 0.5
A_BASE = np.array([1.5, 1.0, 2.0])     # 同 nav_cfg a_max；停止時底盤用此界，不用手臂的 19.984
T_BLEND = 5.0                          # 展臂漸變只在抵達預抓前 5 s（全程漸變時目標超出可達，見 offline_wb_short.json）
W_ARM = dict(w_r=25.0, w_a=0.05, w_s=0.05, v_arm=0.9999, a_arm=C.A_ARM, margin=0.05)


def nav_cfg():
    return GW.GMPCConfig(N=N, dt=DT, u_min=np.array([-0.20, -0.25, -0.8]), u_max=np.array([0.35, 0.25, 0.8]), a_max=A_BASE.copy(),
                         Q=np.diag([10.0, 10.0, 5.0]), R=np.diag([0.5, 0.5, 0.2]), Qf=np.diag([50.0, 50.0, 25.0]), wheel_coupling=True)


def heuristic_speed(task, st, t, T_tcp, x, x_we, v_c):
    """啟發式進度調速（工程基線）：依剩餘任務時間限制底盤速度，使抬升完成前不越窗；只減速不加速。"""
    if BR.PHASES.index(st.phase) > BR.PHASES.index('LIFT'):
        return float(v_c)
    t_rem = {'CRUISE': max(st.t_pre - t, 0.0) + 0.3 + task.t_insert, 'PREGRASP': BR.pose_err(T_tcp, task.T_pre)[0] / 0.05 + 0.3 + task.t_insert,
             'INSERT': max(task.t_insert - (t - st.t_phase), 0.0) + 0.2}.get(st.phase, 0.0)
    need = (max(T_LIFT - (t - st.t_phase), 0.0) + T_LIFT_SETTLE if st.phase == 'LIFT' else
            max(T_CLOSE + T_EV - (t - st.t_phase), 0.0) + T_LIFT + T_LIFT_SETTLE if st.phase == 'CLOSE' else T_CLOSE + T_EV + T_LIFT + T_LIFT_SETTLE)
    return float(np.clip((x_we - 0.02 - x) / max(t_rem + need, 1e-3), 0.0, v_c))


class S0bController(C.S0aController):
    def __init__(self, K, ident, T_grasp, T_ready_rel, q_ready, table_top_z, bottle_radius, x_start, x_win, x_end, v_c=0.03, x_pre=None, t_blend=T_BLEND, lift_m=0.05, reg=None, required=()):
        x_pre = x_win[0] + 0.05 if x_pre is None else x_pre
        t_pre = max((x_pre - x_start) / v_c, 4.0)
        super().__init__(K, ident, T_grasp, T_ready_rel, q_ready, table_top_z, bottle_radius, t_reach=t_pre, lift_m=lift_m, reg=reg, required=required)
        self.cfg.base_fixed = False
        self.st.t_cruise0 = max(t_pre - t_blend, 0.0)
        self.nav = nav_cfg()
        self.mpc = GW.GMPC(self.nav)
        self.x_ws, self.x_we, self.x_end, self.v_c = float(x_win[0]), float(x_win[1]), float(x_end), float(v_c)
        self.s_ref, self.v_b, self.n_tick = float(x_start), float(v_c), 0
        self.n_fallback, self.v_b_hist = 0, []
        self.end_t, self.end_ok = None, None
        self.mpc_fail_inject = None                        # 測試注入：callable(t)->dict，覆寫 arm_check 內容以模擬 fallback

    # ---- 參考
    def _plan_at(self, t_eff):
        xs = self.s_ref + self.v_b * C.D_STEPS * C.PHYS
        return (np.array([t_eff - 1.0, t_eff + 30.0]), np.array([xs - self.v_b, xs + 30.0 * self.v_b]), 0.0)

    def _T_seq(self, q, t_eff):
        T_seq, _ = BR.horizon(self.task, self.st, self._plan_at(t_eff), t_eff, N, DT)
        return T_seq

    def _speed(self, T_tcp, x):
        if self.done or x >= self.x_end:
            return 0.0
        return heuristic_speed(self.task, self.st, self.t, T_tcp, x, self.x_we, self.v_c)

    # ---- 相位
    def _advance(self, T_tcp, close_done, q):
        ph0 = self.st.phase
        BR.step_phase(self.task, self.st, self.t, T_tcp, grasp_evidence_ok=self.judge.t_grasp is not None, close_done=close_done, base_meas=np.array([q[0], 0.0, 0.0]))
        if ph0 != 'CARRY' and self.st.phase == 'CARRY':
            self.s_ref = float(q[0])                       # 參考錨定到實測 x（y、θ 取參考），否則 T_carry_rel 與 plan 相差 s_ref−x，目標瞬跳
        if BR.PHASES.index(self.st.phase) < BR.PHASES.index('CARRY') and q[0] > self.x_we:
            self._trigger_stop('window_passed')

    # ---- 求解
    def _solve(self, q):
        t0 = time.perf_counter()
        cfg, st = self.cfg, self.st
        if self.n_tick % 2 == 0:
            v_new = self._speed(self.K.fk(q, 'link_tcp'), float(q[0]))
            self.s_ref += (self.v_b - v_new) * C.D_STEPS * C.PHYS   # 改速自命令生效時刻起算：否則前推項 v_b·τ 瞬變，參考倒跳、底盤收尾時倒退
            self.v_b = v_new
        self.n_tick += 1
        self.v_b_hist.append(self.v_b)
        qp, sp = q.copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, cfg, 1)
        t_eff = self.t + C.D_STEPS * C.PHYS
        T_seq = self._T_seq(q, t_eff)
        if self.fail_inject is not None and self.fail_inject(self.t):
            return None
        xs_eff = self.s_ref + self.v_b * C.D_STEPS * C.PHYS
        xi_ref = np.tile([self.v_b, 0.0, 0.0], (N + 1, 1))
        X_ref = [from_xytheta(xs_eff, 0.0, 0.0)]
        for j in range(N):
            X_ref.append(X_ref[-1] @ exp_(xi_ref[j] * DT))
        w_p = C.W_NEAR if st.phase in ('PREGRASP', 'INSERT', 'CLOSE', 'LIFT') else C.W_FAR
        arm = dict(mode='active', K=self.K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=cfg.arm_model, T_seq=T_seq, w_p=w_p,
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, **W_ARM)
        r = self.mpc.solve(X_now=from_xytheta(*qp[:3]), X_ref_win=np.array(X_ref), xi_ref_win=xi_ref, xi_prev=self.u_prev[:3], arm=arm)
        ck = dict(r.arm_check or {})
        if self.mpc_fail_inject is not None:
            ck.update(self.mpc_fail_inject(self.t) or {})
        self.solve_ms.append((time.perf_counter() - t0) * 1e3)
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False):
            self.n_fallback += 1
            self._ev('wb_fallback', why=ck.get('fallback'), failed=ck.get('failed'))
            self._trigger_stop(f"wb_fallback:{ck.get('fallback')}")
            return None
        self.s_ref += self.v_b * DT
        return np.r_[np.asarray(r.u_opt, float), r.u_arm_seq[0]]

    # ---- 停止：手臂有界減速同 S0a；底盤用導航加速度界
    def _arm_decel(self):
        da = np.r_[A_BASE, np.full(6, C.A_ARM)] * self.cfg.dt
        return self.u_last + np.clip(-self.u_last, -da, da)

    def step(self, t, q, **kw):
        ua = self.pipe[0].copy()                           # 本步生效的命令（S0a step 內 append 後 pop(0) 取同一筆）
        out = super().step(t, q, **kw)
        out['base_twist'] = ua[:3]
        if self.done and self.judge.stop is None:          # 完成後收尾：底盤參考已歸零，核對確實停下
            if self.end_t is None:
                self.end_t = self.t
                self._ev('end_tail')
            elif self.end_ok is None and self.t - self.end_t >= 1.0 and self.q_last is not None:
                self.end_ok = dict(t=self.t, base_speed=self._vb_meas(q), ok=bool(self._vb_meas(q) < 0.01))
        self.q_prev2 = np.asarray(q, float).copy()
        return out

    def _vb_meas(self, q):
        p = getattr(self, 'q_prev2', None)
        return float('nan') if p is None else float(np.hypot(*(np.asarray(q, float)[:2] - p[:2]))) / C.PHYS

    def report(self):
        r = super().report()
        vb = np.array(self.v_b_hist) if self.v_b_hist else np.zeros(1)
        r.update(core='gmpc_wb', obstacles=None, window=[self.x_ws, self.x_we], n_fallback=self.n_fallback, end_effect=self.end_ok, v_c=self.v_c, v_b_min=float(vb.min()), v_b_max=float(vb.max()), s_ref_end=self.s_ref, n_tick=self.n_tick)
        return r


def exit_code(rep):
    """只有完整成功且底盤確實停下才回 0；其餘沿用 S0a（3 已停止且有效、4 未驗證／無效）。"""
    if rep.get('success') is True and rep.get('stop') is None:
        return 0 if (rep.get('end_effect') or {}).get('ok') is True else 4
    return C.exit_code(rep)
