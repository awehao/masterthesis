#!/usr/bin/env python3
"""瓶子 S0b 控制器：低速移動抓取（純邏輯、無 Isaac；Isaac 腳本與假受控對象測試共用同一份）。規格：規格/瓶子S0_物理抓持設計.md §11。
核心是 **gmpc_wb**（統一導航複本，導航 SE(2) 誤差問題＋手臂設定點模型＋TCP 任務，N＝20、dt 0.05，每週期兩次 QP），不是局部 FR 核心；
本趟結果才算這個耦合核心的物理驗證。場景無障礙，obstacles＝None，**CBF 未介入**。
判定、接觸資料、相位事件、抓持證據、手指與 d̂ 全部沿用 S0aController／S0Judge，只覆寫：
  求解（_solve）、未來參考（_T_seq）、相位推進（_advance：可達窗與 CARRY 參考錨定）、停止減速（_arm_decel：底盤用導航加速度界）、輸出（step：底盤命令走同一延遲佇列）。
底盤速度是**啟發式進度調速**（工程基線，只減速不加速，v_b ≤ v_c），不是 GMPC 最佳化出的進度協調。
任何 fallback（含 base_brake）一律走既有停止路徑，不把退回解當可繼續的命令。
bottle_s0b_1（git e06af3481）之後新增，供離線 CBF＋手臂案例用，obstacles_fn 為 None 時行為不變（test_bottle_s0b 回歸）：
  obstacles_fn（障礙，於命令生效時刻取值）、完整 SE(2) 錨定（CARRY 參考取實測 x、y、θ）、參考領先上限 S_LEAD、arm_mode＝'hold' 對照與 v_replay、log_cbf、逐週期耗時。
"""
from __future__ import annotations

import dataclasses
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
S_LEAD = 0.02                          # 參考不得領先實測超過此距離（m）：底盤被 CBF 擋住時 s_ref 不累積，否則放行後追趕（0.05 時夾持中 0.09 m/s）。bottle_s0b_1 全程 x−s_ref 峰 5.19 mm，此界不作用
YAW_TUBE = math.radians(3.0)           # 放行偏航：採樣 y∈{−0.04,0,0.10}、±2/3/5/10°；±5° 起開始失敗。工程准入，非可行性證明；角落見 X_CORNER
X_CORNER = 0.02                        # 放行管排除角落的 x 上限
A_REF = 0.2                            # 完成或到終點後參考降速斜率（m/s²）
Y_TUBE = (-0.04, 0.10)                 # 放行 INSERT 的底盤側向管：bottle_branch 全窗連續可解的範圍（y ±0.05 步進掃描；y≥0.15 或 ≤−0.10 失敗）
T_BLEND = 5.0                          # 展臂漸變只在抵達預抓前 5 s（全程漸變時目標超出可達，見 offline_wb_short.json）
W_ARM = dict(w_r=25.0, w_a=0.05, w_s=0.05, v_arm=0.9999, a_arm=C.A_ARM, margin=0.05)
CBF_MISSION = dict(cbf_alpha=0.5, cbf_safe_margin=0.69, cbf_prune_range=1.2, cbf_near_steps=6, cbf_far_stride=2)   # 同 run_mission_roundtrip4.sh 導航參數（靜態邊距 0.47 由障礙 margin 鍵給）


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
    def __init__(self, K, ident, T_grasp, T_ready_rel, q_ready, table_top_z, bottle_radius, x_start, x_win, x_end, v_c=0.03, x_pre=None, t_blend=T_BLEND, lift_m=0.05, reg=None, required=(),
                 obstacles_fn=None, cbf_cfg=None, arm_mode='active', v_replay=None, log_cbf=False, qp_solver='gi'):
        x_pre = x_win[0] + 0.05 if x_pre is None else x_pre
        t_pre = max((x_pre - x_start) / v_c, 4.0)
        super().__init__(K, ident, T_grasp, T_ready_rel, q_ready, table_top_z, bottle_radius, t_reach=t_pre, lift_m=lift_m, reg=reg, required=required)
        self.cfg.base_fixed = False
        self.st.t_cruise0 = max(t_pre - t_blend, 0.0)
        self.nav = dataclasses.replace(nav_cfg(), **(cbf_cfg or {}))
        self.mpc = GW.GMPC(self.nav)
        self.mpc_nc = GW.GMPC(self.nav) if log_cbf else None   # 同狀態移除 CBF 重解用；獨立物件，不干擾主求解器的暖啟動
        self.obstacles_fn, self.arm_mode, self.v_replay, self.log_cbf = obstacles_fn, arm_mode, v_replay, log_cbf
        self.y_ref, self.th_ref, self.cbf_log = 0.0, 0.0, []
        self.done_anchored, self.n_gate, self.qp_solver, self.qp_log = False, 0, qp_solver, []
        self.u_arm_plan, self.base_plan, self.no_inflight = None, None, False
        self.compact = None                                # list 時逐週期存實際命令、CBF 殘差、slack 與重解輸入（反事實重解在實跑後做）
        self.x_ws, self.x_we, self.x_end, self.v_c = float(x_win[0]), float(x_win[1]), float(x_end), float(v_c)
        self.s_ref, self.v_b, self.n_tick = float(x_start), float(v_c), 0
        self.n_fallback, self.v_b_hist = 0, []
        self.end_t, self.end_ok = None, None
        self.mpc_fail_inject = None                        # 測試注入：callable(t)->dict，覆寫 arm_check 內容以模擬 fallback

    # ---- 參考
    def _plan_at(self, t_eff):
        xs = self.s_ref + self.v_b * C.D_STEPS * C.PHYS
        return (np.array([t_eff - 1.0, t_eff + 30.0]), np.array([xs - self.v_b, xs + 30.0 * self.v_b]), self.y_ref, self.th_ref)

    def _T_seq(self, q, t_eff):
        if self.st.phase == 'CARRY':                       # 持瓶相對底盤固定：T_seq 是底盤座標目標，配 arm['tcp_rel_base']
            return np.stack([self.st.T_carry_rel] * (N + 1))
        T_seq, _ = BR.horizon(self.task, self.st, self._plan_at(t_eff), t_eff, N, DT)
        return T_seq

    def _ref_window(self, xs_eff):
        c_, s_ = math.cos(self.th_ref), math.sin(self.th_ref)
        xi_ref = np.tile([self.v_b * c_, -self.v_b * s_, 0.0], (N + 1, 1))   # 參考沿世界 +x、偏航固定（與 plan 一致）
        X_ref = [from_xytheta(xs_eff, self.y_ref, self.th_ref)]
        for j in range(N):
            X_ref.append(X_ref[-1] @ exp_(xi_ref[j] * DT))
        return np.array(X_ref), xi_ref

    def _rel_task(self):
        return self.st.phase == 'CARRY'

    def _arm_extra(self):
        return {}

    def _speed(self, T_tcp, x):
        if self.v_replay is not None:
            return float(self.v_replay[min(self.n_tick, len(self.v_replay) - 1)])
        if self.done or x >= self.x_end:
            return max(self.v_b - A_REF * 2.0 * DT, 0.0)   # 參考依斜率降速：瞬間歸零時任何過衝都被高增益拉回成倒退
        return heuristic_speed(self.task, self.st, self.t, T_tcp, x, self.x_we, self.v_c)

    # ---- 相位
    def _in_tube(self, q):
        if q[1] < 0.0 and q[2] < 0.0 and q[0] > X_CORNER:  # 已知不可行角落：y≈−0.04、負偏航時抬升位姿在 x≥0.14 失敗；INSERT 到抬升完成底盤再走約 0.1 m
            return False
        return self.x_ws <= q[0] <= self.x_we and Y_TUBE[0] <= q[1] <= Y_TUBE[1] and abs(q[2]) <= YAW_TUBE

    def _advance(self, T_tcp, close_done, q):
        ph0 = self.st.phase
        if ph0 == 'PREGRASP' and not self._in_tube(q):
            self.n_gate += 1                               # 底盤不在驗證過的管內：不放行 INSERT，TCP 維持預抓
            return
        BR.step_phase(self.task, self.st, self.t, T_tcp, grasp_evidence_ok=self.judge.t_grasp is not None, close_done=close_done, base_meas=q[:3].copy())
        if ph0 != 'CARRY' and self.st.phase == 'CARRY':
            self.s_ref, self.y_ref, self.th_ref = (float(v) for v in q[:3])   # 底盤參考錨定到實測 SE(2)：繞障後 y、θ 不再是 0，不把底盤拉回原線
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
        self.s_ref = min(self.s_ref, float(q[0]) + S_LEAD)
        self.n_tick += 1
        self.v_b_hist.append(self.v_b)
        qp, sp = q.copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, cfg, 1)
        if self.no_inflight:                               # 消融：求解初態用當下量測，不沿佇列前推；實際延遲、佇列、參考時標與安全核對都不變
            qp, sp = q.copy(), self.s.copy()
        if self.done and self.v_b == 0.0 and not self.done_anchored:      # 降速完成：參考落回命令生效時刻的預測位置，否則領先的參考位姿落在 keep-out 內，CBF 列把底盤往回推
            self.s_ref, self.done_anchored = float(qp[0]), True
        t_eff = self.t + C.D_STEPS * C.PHYS
        T_seq = self._T_seq(qp, t_eff)
        if self.fail_inject is not None and self.fail_inject(self.t):
            return None
        xs_eff = self.s_ref + self.v_b * C.D_STEPS * C.PHYS
        X_ref, xi_ref = self._ref_window(xs_eff)
        w_p = C.W_NEAR if st.phase in ('PREGRASP', 'INSERT', 'CLOSE', 'LIFT') else C.W_FAR
        arm = dict(mode='active', K=self.K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=cfg.arm_model, T_seq=T_seq, w_p=w_p,
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=self._rel_task(), qp_solver=self.qp_solver, **W_ARM, **self._arm_extra()) if self.arm_mode == 'active' else dict(mode='hold')
        obs = None if self.obstacles_fn is None else self.obstacles_fn(t_eff)
        solve_kw = dict(X_now=from_xytheta(*qp[:3]), X_ref_win=X_ref, xi_ref_win=xi_ref, xi_prev=self.u_prev[:3], arm=arm)
        r = self.mpc.solve(obstacles=obs, **solve_kw)
        ck = dict(r.arm_check or {})
        if self.mpc_fail_inject is not None:
            ck.update(self.mpc_fail_inject(self.t) or {})
        self.solve_ms.append((time.perf_counter() - t0) * 1e3)
        self.u_arm_plan = None if r.u_arm_seq is None else np.array(r.u_arm_seq, float)
        self.base_plan = xi_ref[:N] + np.asarray(r.delta_xi_all, float)
        self.qp_log.append(dict(ck.get('qp') or dict(solver='fallback' if ck.get('fallback') else 'hold')))
        self.ob_rows = max(getattr(self, 'ob_rows', 0), int(ck.get('ob_rows') or 0))
        if ck.get('ob_by_body'):
            bb = self.__dict__.setdefault('ob_body_rows', {})
            for k_, v_ in ck['ob_by_body'].items():
                bb[k_] = dict(ticks=bb.get(k_, {}).get('ticks', 0) + 1, rows_max=max(bb.get(k_, {}).get('rows_max', 0), int(v_)))
        if ck.get('ob_planes'):
            pl = self.__dict__.setdefault('ob_plane_count', {})
            for k_ in ck['ob_planes']:
                pl[k_] = pl.get(k_, 0) + 1
        if self.compact is not None and self.arm_mode == 'active' and r.u_arm_seq is not None:
            self.compact.append(dict(obs=obs, t=self.t, phase=BR.PHASES.index(st.phase), qp=qp.copy(), sp=sp.copy(), u_prev=self.u_prev.copy(), X_ref=X_ref.copy(), xi_ref=xi_ref.copy(), T_seq=np.array(T_seq), rel=bool(arm['tcp_rel_base']), w_p=float(w_p),
                                     bias=np.array(cfg.arm_model.bias, float), u=np.r_[np.asarray(r.u_opt, float), r.u_arm_seq[0]], cbf_active=int(r.cbf_active), min_h=float(r.min_h), eps0=float(r.eps0), resid_ns=float(r.cbf_resid_post_noslack), resid_s=float(r.cbf_resid_post_slack)))
        if self.log_cbf and obs:
            u_nc = np.asarray(self.mpc_nc.solve(obstacles=None, **solve_kw).u_opt, float)
            u_op = np.asarray(r.u_opt, float)
            self.cbf_log.append(dict(t=self.t, phase=st.phase, cbf_active=int(r.cbf_active), min_h=float(r.min_h), eps0=float(r.eps0), resid_ns=float(r.cbf_resid_post_noslack), resid_s=float(r.cbf_resid_post_slack),
                                     evaluated_on=r.cbf_resid_evaluated_on, status=r.status, accept=r.accept_action, du=float(np.linalg.norm(u_op - u_nc)), u=u_op.tolist(), u_nc=u_nc.tolist(), fallback=ck.get('fallback')))
        if self.arm_mode == 'hold':
            if r.accept_action == 'brake' or r.status != 'solved':
                self.n_fallback += 1
                self._trigger_stop('wb_fallback:hold_' + str(r.accept_action))
                return None
            self.s_ref += self.v_b * DT
            return np.r_[np.asarray(r.u_opt, float), np.zeros(6)]
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
        ms = np.array(self.solve_ms) if self.solve_ms else np.zeros(1)
        cnt, why = {}, {}
        for e in self.qp_log:
            cnt[e['solver']] = cnt.get(e['solver'], 0) + 1
            if e.get('gi_why'):
                why[e['gi_why']] = why.get(e['gi_why'], 0) + 1
        os_ms = [e['osqp_ms'] for e in self.qp_log if e.get('osqp_ms') is not None]
        r.update(qp_solver=self.qp_solver, qp_counts=cnt, gi_fail=why, backup_ms_max=max(os_ms, default=None), qp_log=[[e.get('solver'), e.get('gi_why'), round(float(e.get('gi_ms') or 0.0), 2), None if e.get('osqp_ms') is None else round(float(e['osqp_ms']), 2)] for e in self.qp_log])
        r.update(n_gate=self.n_gate, core='gmpc_wb', obstacles=None if self.obstacles_fn is None else 'obstacles_fn', arm_mode=self.arm_mode, solve_ms_all=[round(float(v), 2) for v in self.solve_ms], n_over_50ms=int((ms > 50.0).sum()), window=[self.x_ws, self.x_we], n_fallback=self.n_fallback, end_effect=self.end_ok, v_c=self.v_c, v_b_min=float(vb.min()), v_b_max=float(vb.max()), s_ref_end=self.s_ref, n_tick=self.n_tick)
        return r


def exit_code(rep):
    """只有完整成功且底盤確實停下才回 0；其餘沿用 S0a（3 已停止且有效、4 未驗證／無效）。"""
    if rep.get('success') is True and rep.get('stop') is None:
        return 0 if (rep.get('end_effect') or {}).get('ok') is True else 4
    return C.exit_code(rep)
