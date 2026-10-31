#!/usr/bin/env python3
"""瓶子 S0a 控制器（純邏輯、無 Isaac；Isaac 腳本與假受控對象測試共用同一份）。規格：規格/瓶子S0_物理抓持設計.md。
底盤停住（base_fixed）；手臂以 wgmpc_core_sp_fr（局部未來參考核心，非統一導航）20 Hz 求解，參考與相位用 bottle_ref（只由事件推進；無抓持證據不預覽抬升）。
**命令延遲 8 物理步是注入設定**（取 rt_43 求解端補償值），非實測 ROS 延遲；求解前以在途命令與同一設定點模型前推。
d̂：閉爪前以 IntervalOffsetObserver 更新（補償重力下垂），閉爪請求起凍結（往返任務教訓：接觸中不更新）。
停止處置（**本腳本自有，不是既有命令鏈的重用**，由 test_bottle_s0a.py 另驗）：新命令以 a_arm 有界減速至 0；在途命令照常生效；底盤零；手指維持當下命令；
  之後核對停止有效性（1 s 內手臂關節速度 < 1e-3 rad/s、底盤 < 0.01 m/s）。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S                     # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp_fr as FR                 # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE                  # noqa: E402
import bottle_ref as BR                                               # noqa: E402
import bottle_s0_logic as L                                           # noqa: E402
import offset_interval as OINT                                        # noqa: E402

PHYS, NPH, D_STEPS = 0.01, 5, 8
GRASP_DEPTH = 0.004      # TCP 越過瓶軸 4 mm：瓶軸落在指片（TCP 後 1.3–21.3 mm）內，瓶近面離手指根部塊（TCP 後 21.3 mm）約 2.3 mm
FINGER_OPEN, FINGER_CLOSED, T_CLOSE_RAMP = 0.0089, 0.0, 0.5
W_FAR, W_NEAR = 50.0, 200.0   # PREGRASP～LIFT 用 200（同 offline GRASP_W；本案例調參結果）
A_ARM = 19.984


def make_cfg(ident):
    am = S.ArmSetpointModel(alpha=np.asarray(ident['alpha'], float), bias=np.asarray(ident['bias_rad'], float), phys_dt=ident['phys_dt_measured_s'])
    c = S.WGMPCConfigSP(N=5, dt=0.05, arm_model=am, w_s=1e-3, w_a=0.05, w_s_arm=0.05, row_scaling=True)
    c.w_p, c.w_vref = W_FAR, 0.0
    c.base_fixed = True
    return c


class S0aController:
    def __init__(self, K, ident, T_grasp, T_ready_rel, q_ready, table_top_z, bottle_radius, t_reach=4.0, lift_m=0.05, t_hold_end=6.0, reg=None, required=()):
        self.K, self.cfg = K, make_cfg(ident)
        self.task = BR.BottleTask(T_grasp=np.array(T_grasp, float), lift_m=lift_m, T_ready_rel=np.array(T_ready_rel, float))
        self.st = BR.PhaseState(t_cruise0=0.0, t_pre=float(t_reach))
        self.judge = L.S0Judge(table_top_z, bottle_radius, reg=reg, required=required)
        self.s = np.array(q_ready, float)                  # 手臂設定點（積分結果）
        self.pipe = [np.zeros(9) for _ in range(D_STEPS)]
        self.u_prev, self.u_last, self.U_warm = np.zeros(9), np.zeros(9), None
        self.obs = OINT.IntervalOffsetObserver(self.cfg.arm_model.alpha, self.cfg.arm_model.phys_dt, 0.5, 0.05)
        self.q_anchor, self.s_seq = None, []
        self.bias0 = np.asarray(self.cfg.arm_model.bias, float).copy()
        self.n_step, self.t = 0, 0.0
        self.f_cmd = FINGER_OPEN
        self.t_close_req = None
        self.fail_inject = None                            # 測試注入：callable(t)->bool 回 True 時求解視為失敗
        self.stopped_t = None
        self.q_last = None
        self.stop_ok = None
        self.t_hold_end = float(t_hold_end)
        self.done = False
        self.n_solve_fail, self.solve_ms = 0, []
        self.events = []
        self.lim_lo, self.lim_hi = np.asarray(LITE6_SAFE.lower, float) + 0.05, np.asarray(LITE6_SAFE.upper, float) - 0.05

    # ---- 內部
    def _ev(self, what, **kw):
        self.events.append(dict(t=round(self.t, 3), what=what, **kw))

    def _plan(self, q):
        return (np.array([self.t - 1.0, self.t + 60.0]), np.array([q[0], q[0]]), float(q[1]))

    def _trigger_stop(self, why):
        self.judge._stop(self.t, why)

    # 掛鉤（受控碰觸 ProbeController 覆寫；判定與停止路徑共用）
    def _T_seq(self, q, t_eff):
        T_seq, _ = BR.horizon(self.task, self.st, self._plan(q), t_eff, self.cfg.N, self.cfg.dt)
        return T_seq

    def _advance(self, T_tcp, close_done, q):
        BR.step_phase(self.task, self.st, self.t, T_tcp, grasp_evidence_ok=self.judge.t_grasp is not None, close_done=close_done, base_meas=q[:3])

    def _solve(self, q):
        cfg, st = self.cfg, self.st
        cfg.w_p = W_NEAR if st.phase in ('PREGRASP', 'INSERT', 'CLOSE', 'LIFT') else W_FAR
        qp, sp = q.copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, cfg, 1)
        t_eff = self.t + D_STEPS * PHYS
        T_seq = self._T_seq(q, t_eff)
        if self.fail_inject is not None and self.fail_inject(self.t):
            return None
        r = FR.solve_sp_fr(self.K, S.make_z(qp, sp), self.u_prev, T_seq, cfg, U_warm=self.U_warm)
        self.solve_ms.append(float(r.timing_ms.get('total', float('nan'))))
        if not r.ok or not np.isfinite(r.u0).all():
            return None
        self.U_warm = r.U
        return np.asarray(r.u0, float)

    def _offset_update(self, q):
        """閉爪前更新 d̂；閉爪請求起凍結。"""
        qa = q[3:]
        if self.t_close_req is None and self.q_anchor is not None and len(self.s_seq) >= 1:
            self.obs.update_setpoints(self.q_anchor, self.s_seq[:-1], qa)     # 區間 (k, k+1] 的 s_j，j＝k…k+1−1
            self.cfg.arm_model.bias = self.bias0 + np.asarray(self.cfg.arm_model.alpha, float) * np.asarray(self.obs.d, float)
        self.q_anchor, self.s_seq = qa.copy(), [self.s.copy()]

    def _arm_decel(self):
        da = A_ARM * self.cfg.dt
        return self.u_last + np.clip(-self.u_last, -da, da)

    # ---- 每物理步
    def step(self, t, q, f_fingers=None, t_contact=None, T_bottle=None, t_bottle=None, f_table=None, forbidden=None, penetration=None):
        """回傳 dict(arm_target(6), finger_target, base_twist(3), phase, stopped, done)。"""
        self.t = float(t)
        q = np.asarray(q, float)
        T_tcp = self.K.fk(q, 'link_tcp') if np.isfinite(q).all() else None
        m = L.Meas(t=self.t, f_fingers=f_fingers, t_contact=t_contact, T_bottle=T_bottle, t_bottle=t_bottle, f_table=f_table,
                   forbidden=forbidden or {}, penetration=penetration, T_tcp=T_tcp)
        if T_tcp is None:
            self._trigger_stop('joint_state_nonfinite')
        self.judge.update(m)
        tick = self.n_step % NPH == 0
        if tick and T_tcp is not None and self.judge.stop is None:
            self._offset_update(q)
            ph0 = self.st.phase
            close_done = self.t_close_req is not None and self.t - self.t_close_req >= T_CLOSE_RAMP
            self._advance(T_tcp, close_done, q)
            if self.st.phase != ph0:
                self._ev('phase', to=self.st.phase)
            if self.st.phase == 'CLOSE' and self.t_close_req is None:
                self.t_close_req = self.t
                self.judge.on_close_request(self.t, m)
                self._ev('close_request')
            if self.st.phase == 'LIFT':
                self.judge.on_lift_start(self.t)
            if self.judge.res['hold_ok'] is True and not self.done:
                self.done = True
                self._ev('done')
        # 控制命令
        if self.judge.stop is not None:
            if self.stopped_t is None:
                self.stopped_t = self.t
                self._ev('stop', u_arm_max=float(np.abs(self.u_last[3:]).max()), **{k: v for k, v in self.judge.stop.items() if k != 't'})
            if tick:
                self.u_last = self._arm_decel()
        elif tick:
            u = None if T_tcp is None else self._solve(q)
            if u is None:
                self.n_solve_fail += 1
                self._trigger_stop('solver_fail')
                self.stopped_t = self.t
                self._ev('stop', why='solver_fail', u_arm_max=float(np.abs(self.u_last[3:]).max()))
                self.u_last = self._arm_decel()
                self.U_warm = None
            else:
                self.u_last = u
            self.u_prev = self.u_last.copy()
        # 注入延遲：本步進佇列，佇列最舊者生效（設定點積分）
        self.pipe.append(self.u_last.copy())
        ua = self.pipe.pop(0)
        nxt = self.s + PHYS * ua[3:]
        if np.any(nxt < self.lim_lo) or np.any(nxt > self.lim_hi):            # 設定點限位：不積分越界
            nxt = np.clip(nxt, self.lim_lo, self.lim_hi)
            if self.judge.stop is None:
                self._trigger_stop('setpoint_limit')
        self.s = nxt
        self.s_seq.append(self.s.copy())
        # 手指：閉爪請求後斜坡到閉合；停止時維持當下命令
        if self.t_close_req is not None and self.judge.stop is None:
            f = min((self.t - self.t_close_req) / T_CLOSE_RAMP, 1.0)
            self.f_cmd = FINGER_OPEN + (FINGER_CLOSED - FINGER_OPEN) * f
        # 停止有效性
        if self.stopped_t is not None and self.q_last is not None and T_tcp is not None:
            dq = float(np.abs(q[3:] - self.q_last[3:]).max()) / PHYS
            vb = float(np.hypot(*(q[:2] - self.q_last[:2]))) / PHYS            # 實測底盤速度（位姿差分），不以下零命令代替
            if self.t - self.stopped_t >= 1.0 and self.stop_ok is None:
                self.stop_ok = dict(t=self.t, arm_rate=dq, base_speed=vb, ok=bool(dq < 1e-3 and vb < 0.01))
        self.q_last = q.copy()
        self.n_step += 1
        return dict(arm_target=self.s.copy(), finger_target=float(self.f_cmd), base_twist=np.zeros(3), phase=self.st.phase,
                    stopped=self.judge.stop is not None, done=self.done)

    def report(self):
        r = self.judge.report()
        sm = np.array([x for x in self.solve_ms if math.isfinite(x)])
        r.update(events=self.events, stop_effect=self.stop_ok, n_solve_fail=self.n_solve_fail, d_hat=[float(x) for x in self.obs.d],
                 solve_ms=dict(p50=float(np.percentile(sm, 50)), p95=float(np.percentile(sm, 95)), max=float(sm.max())) if sm.size else None,
                 phase=self.st.phase, injected_delay_steps=D_STEPS, grasp_depth_m=GRASP_DEPTH)
        return r


def ready_from_pre(base, T_pre, back=0.14, up=0.08):
    """備妥姿態：以預抓解為種子，對預抓位姿沿工具軸後退 back、上移 up 求 IK ⇒ 與抓取解同一 IK 分支（避免翻腕）。回傳 (q_ready, T_ready_rel)。"""
    import bottle_reach_scan as RS
    gp = RS.ik(base, T_pre, RS.SEEDS)
    if not gp[0]:
        raise ValueError('預抓位姿 IK 失敗')
    Tr = T_pre.copy()
    Tr[:3, 3] += -back * T_pre[:3, 2] + np.array([0.0, 0.0, up])
    g = RS.ik(base, Tr, [gp[2]])
    if not g[0]:
        raise ValueError('備妥位姿 IK 失敗')
    Tb = np.eye(4)
    c_, s_ = math.cos(base[2]), math.sin(base[2])
    Tb[:3, :3] = [[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]
    Tb[:3, 3] = [base[0], base[1], 0.0]
    return np.asarray(g[2], float), np.linalg.inv(Tb) @ Tr


def S0_K():
    """與離線一致的全身運動學（evaluation/models/omni_bot_wholebody_expanded.urdf）。"""
    from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics
    return WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))


def grasp_target(xb, yb, h):
    """側向抓取目標：TCP 越過瓶軸 GRASP_DEPTH（沿工具軸 +y）。"""
    T = BR.grasp_pose(xb, yb, h)
    T[:3, 3] += GRASP_DEPTH * T[:3, 2]
    return T


def exit_code(rep):
    """只有完整成功回 0；中止、逾時未完成、停止有效性未通過都非零（Codex 20261031_185332 (3)）。"""
    if rep.get('success') is True and rep.get('stop') is None:
        return 0
    return 3 if (rep.get('stop_effect') or {}).get('ok') is True else 4     # 3：已停止且停止有效；4：停止未驗證／無效


__all__ = ['S0aController', 'grasp_target', 'make_cfg', 'exit_code', 'math']
