#!/usr/bin/env python3
"""遠距整合：進線段與命令路由的可執行骨架（Codex 20261116：(a)）。離線、運動學層；不是 ROS 接線完成，也不是全身避障驗證。
1. Route：底盤與手臂各自一個命令擁有者（base: NAV→GRASP；arm: HOLD→TRACK→GRASP），全域 STOP 優先於兩者；非擁有者的命令被拒並計數。
2. select_pregrasp：先「可連續到達」再排序——以當下實測手臂構型為起點，逐候選核：
   準備段（現構型→追蹤窗口起點的 IK 解，關節直線、速率上限 V_PREP）、移動追蹤窗口（沿通過線每 2.5 cm 的連續 IK：限位、可操作度、所需關節速率、淨空）、
   插入（抓取點 s: S_PRE→0）；全過才合格；合格者依準備時間排序、可操作度次之。開門可行性不在此核（明列待核）。回傳含搜尋耗時。
3. ArmPreTracker：導航仍擁有底盤時，手臂提前追「把手估計後退 S_PRE」的世界目標。底盤運動取**實測位姿**與 Shield／GUARD **實際放行**的命令（在途管線），
   不追導航提案；命令受速率／加速度限制，超限就以上限追（記落後量），IK 失敗或越限位 ⇒ 手臂保持並回報未就緒。安全層煞車（放行命令變小）時手臂跟著實際運動走。
4. merge_plan：導航末段的目標＝通過線上抓取點前 L_IN 的進線點（朝向＝通過線朝向、速度方向＝通過方向）；交接區＝距抓取點 [W_MIN, W_MAX]；
   只在控制拍邊界交接（抓持控制器建構時重設拍相位）；不到點停車。"""
import math
import time

import numpy as np

import door_far_link as FL
import door_fast_kin as DFK

V_PREP, V_ARM, A_ARM = 0.5, 1.0, 20.0                                # 準備段關節速率上限（rad/s）；手臂命令速率／加速度上限（同控制器）
L_IN, W_MIN, W_MAX = 0.60, 0.05, 0.25                                # 進線點距抓取點；交接區（沿線距離，m）
TRACK_STEP = 0.025


class Route:
    def __init__(self):
        self.base, self.arm, self.stop, self.log, self.n_rejected = 'NAV', 'HOLD', False, [], 0

    def allow(self, part, src, t):
        own = self.base if part == 'base' else self.arm
        if self.stop or src != own:
            self.n_rejected += 1
            self.log.append(dict(t=round(float(t), 3), what='cmd_rejected', part=part, src=src, owner='STOP' if self.stop else own))
            return False
        return True

    def give(self, part, to, t, ok=True, why=None):
        if self.stop or not ok:
            self.log.append(dict(t=round(float(t), 3), what='switch_refused', part=part, to=to, why='global_stop' if self.stop else why))
            return False
        frm = self.base if part == 'base' else self.arm
        setattr(self, part, to)
        self.log.append(dict(t=round(float(t), 3), what='switch', part=part, frm=frm, to=to))
        return True

    def global_stop(self, t, why):
        self.stop = True
        self.log.append(dict(t=round(float(t), 3), what='global_stop', why=why))


def _wrap(a):
    return (np.asarray(a, float) + math.pi) % (2 * math.pi) - math.pi


def select_pregrasp(T_WH, q_now, T_at, s_pre, K, clear_fn, thr, sig_floor, lo, hi, v_pass):
    t0 = time.perf_counter()
    cands = []
    _, stat = FL.pregrasp_from_estimate(T_WH, T_at, s_pre, K, clear_fn=clear_fn, thr=thr, sig_floor=sig_floor, lo=lo, hi=hi, collect=cands)
    t_gen = time.perf_counter() - t0
    Tp = np.asarray(T_at(T_WH, s_pre), float)
    ok, rej = [], {}
    for c in cands:
        d, p0, yaw = c['d'], c['p0'], c['yaw']
        # 追蹤窗口：由抓取點往回到 W_MAX，連續 IK
        seed, Q, why = c['q_pregrasp'], [], None
        for al in np.arange(0.0, W_MAX + 1e-9, TRACK_STEP):
            b = np.r_[p0 - d * al, yaw]
            q, r, sm = DFK.ik_closure_fc(Tp, b, seed)
            if not (r < 1e-6 and np.isfinite(q).all()) or np.abs(q - seed).max() > 0.5:
                why = 'track_ik'; break
            if np.any(q < lo) or np.any(q > hi):
                why = 'track_joint_limit'; break
            if sm < sig_floor:
                why = 'track_manipulability'; break
            cl = clear_fn(np.r_[b, q])
            if any(cl[k] < thr[k] for k in ('env', 'base', 'self')):
                why = 'track_clearance'; break
            Q.append(q); seed = q
        if why is None:
            Q = np.array(Q)
            rate = float(np.abs(np.diff(Q, axis=0)).max() / (TRACK_STEP / v_pass)) if len(Q) > 1 else 0.0
            if rate > V_ARM:
                why = 'track_rate'
        if why is None:                                              # 插入：抓取點 s 由 s_pre→0
            seed = c['q_pregrasp']
            for s_ in np.linspace(s_pre, 0.0, 5):
                q, r, sm = DFK.ik_closure_fc(np.asarray(T_at(T_WH, s_), float), c['base'], seed)
                if not (r < 1e-6) or np.any(q < lo) or np.any(q > hi) or sm < sig_floor:
                    why = 'insert_ik_or_limit'; break
                seed = q
        if why is None:                                              # 準備段：現構型 → 窗口起點解（關節直線）
            q_start = Q[-1]
            dq = _wrap(q_start - np.asarray(q_now[3:], float))
            qa0 = np.asarray(q_now[3:], float)
            if np.any(qa0 + dq < lo) or np.any(qa0 + dq > hi):
                why = 'prep_wraps_joint_limit'
            else:
                for u_ in (0.25, 0.5, 0.75):
                    cl = clear_fn(np.r_[q_now[:3], qa0 + u_ * dq])
                    if cl['self'] < thr['self']:
                        why = 'prep_self_clearance'; break
        if why is not None:
            rej[why] = rej.get(why, 0) + 1
            continue
        ok.append(dict(c, q_window_start=q_start.copy(), prep_time_s=float(np.abs(dq).max() / V_PREP), prep_move_max_rad=float(np.abs(dq).max()), track_rate_max=rate, sigma_track_min=float(c['sigma_min'])))
    ok.sort(key=lambda x: (round(x['prep_time_s'], 2), -x['sigma_min']))
    info = dict(gen=stat, n_feasible_static=len(cands), n_ok=len(ok), rejected=rej, t_gen_s=round(t_gen, 2), t_total_s=round(time.perf_counter() - t0, 2), door_opening_feasibility='待核（本選擇未核開門段）')
    return (ok[0] if ok else None), info


class ArmPreTracker:
    """手臂提前追蹤（底盤由導航／安全鏈控制）。每控制拍呼叫 step(q_meas, base_pipe, u_base_released, s_arm, T_target)。
    base_pipe＝實際放行、仍在途的底盤命令（本體座標，逐物理步）；u_base_released＝本拍安全鏈實際放行的底盤命令。"""

    def __init__(self, tau, off_fn, dt, phys, nph, lo, hi):
        self.tau, self.off_fn, self.dt, self.phys, self.nph, self.lo, self.hi = np.asarray(tau, float), off_fn, dt, phys, nph, lo, hi
        self.q_prev, self.u_prev, self.lag_max, self.n_clip, self.ready = None, np.zeros(6), 0.0, 0, False

    def step(self, q, base_pipe, u_base, s_arm, s_pipe_end, T_target, seed):
        b = np.asarray(q[:3], float).copy()
        for ub in list(base_pipe) + [u_base] * self.nph:              # 在途命令走完＋本拍放行命令作用一拍後的底盤位姿
            c, s_ = math.cos(b[2]), math.sin(b[2])
            b = b + self.phys * np.array([c * ub[0] - s_ * ub[1], s_ * ub[0] + c * ub[1], ub[2]])
        q1, r, sm = DFK.ik_closure_fc(T_target, b, seed)
        if not (r < 1e-6 and np.isfinite(q1).all()) or np.any(q1 < self.lo) or np.any(q1 > self.hi):
            self.ready = False
            return dict(ok=False, why='ik_or_limit', u=np.zeros(6), q1=None)
        qd = np.zeros(6) if self.q_prev is None else (q1 - self.q_prev) / self.dt
        S1 = q1 + self.tau * qd - self.off_fn(np.r_[b, q1])
        u = (S1 - np.asarray(s_pipe_end, float)) / self.dt
        u_c = np.clip(u, -V_ARM, V_ARM)
        u_c = np.clip(u_c, self.u_prev - A_ARM * self.dt, self.u_prev + A_ARM * self.dt)
        clipped = bool(np.abs(u_c - u).max() > 1e-9)
        self.n_clip += int(clipped)
        self.q_prev, self.u_prev = q1.copy(), u_c.copy()
        err = float(np.abs(q1 - np.asarray(q[3:], float)).max())
        self.lag_max = max(self.lag_max, err)
        self.ready = (not clipped) and err <= 3 * FL.ARM_TRACK_TOL
        return dict(ok=True, u=u_c, q1=q1, clipped=clipped, sigma=float(sm), track_err=err)


def merge_plan(pg):
    """導航末段目標與交接區。回 dict(p_in, yaw, d, window)。"""
    return dict(p_in=pg['p0'] - pg['d'] * L_IN, yaw=pg['yaw'], d=pg['d'].copy(), p_grasp=pg['p0'].copy(), window=(W_MIN, W_MAX), handover='控制拍邊界；移動中；不到點停車')
