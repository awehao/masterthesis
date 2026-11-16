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


def select_pregrasp(T_WH, q_now, T_at, s_pre, K, clear_fn, thr, sig_floor, lo, hi, v_pass, standoffs=None, alongs=None):
    t0 = time.perf_counter()
    cands = []
    _, stat = FL.pregrasp_from_estimate(T_WH, T_at, s_pre, K, clear_fn=clear_fn, thr=thr, sig_floor=sig_floor, lo=lo, hi=hi, collect=cands, standoffs=standoffs, alongs=alongs)
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
        # 誤差分列（Codex 20261116：q1 是約 0.13 s 後的構型，不能直接和現在的實測比）：
        #   track_err      ＝現在實測關節 對 現在底盤位姿下的目標 IK 解（交接判定用的量，同 FL.ARM_TRACK_TOL 的定義）
        #   track_err_pred ＝生效時刻的預測關節（設定點沿在途命令走完＋本拍命令一拍、關節一階模型）對 q1
        qn, rn, _ = DFK.ik_closure_fc(T_target, np.asarray(q[:3], float), np.asarray(q[3:], float))
        err = float(np.abs(qn - np.asarray(q[3:], float)).max()) if (rn < 1e-6 and np.isfinite(qn).all()) else float('nan')
        a_, sp_, al_ = np.asarray(q[3:], float).copy(), np.asarray(s_arm, float).copy(), self.phys / self.tau
        n_p = max(len(base_pipe), 1)
        dsp = (np.asarray(s_pipe_end, float) - sp_) / n_p
        off_ = np.asarray(self.off_fn(np.r_[b, q1]), float)             # 靜態偏置（關節收斂到 設定點＋偏置）；與命令計算用同一個模型
        for _ in range(n_p):
            sp_ = sp_ + dsp; a_ = a_ + al_ * (sp_ + off_ - a_)
        for _ in range(self.nph):
            sp_ = sp_ + self.phys * u_c; a_ = a_ + al_ * (sp_ + off_ - a_)
        err_pred = float(np.abs(a_ - q1).max())
        if np.isfinite(err):
            self.lag_max = max(self.lag_max, err)
        self.ready = (not clipped) and np.isfinite(err) and err <= 3 * FL.ARM_TRACK_TOL
        return dict(ok=True, u=u_c, q1=q1, clipped=clipped, sigma=float(sm), track_err=err, track_err_pred=err_pred)


def merge_plan(pg):
    """導航末段目標與交接區。回 dict(p_in, yaw, d, window)。"""
    return dict(p_in=pg['p0'] - pg['d'] * L_IN, yaw=pg['yaw'], d=pg['d'].copy(), p_grasp=pg['p0'].copy(), window=(W_MIN, W_MAX), handover='控制拍邊界；移動中；不到點停車')


S_OBS = 0.33                                                         # 觀察追蹤：TCP 目標＝抓取位姿沿接近軸後退 S_OBS（腕部相機看得到把手中心與兩端的最小後退量 0.31 m 加餘量）
OBS_ALONG = (0.25, -0.10)                                            # 觀察追蹤已核的沿線範圍（距抓取點；負值＝已通過把手正前方）。0.275 m 處可操作度 0.027<0.03、更遠構不到
STOP_ALONG = -0.05                                                   # 取像趟：通過把手正前方 0.05 m 後開始正常停止（TCP 始終在後退 S_OBS 的觀察位，不進無視覺段）


def observe_target(T_WH, T_at, a_H):
    T = np.asarray(T_at(T_WH, 0.0), float).copy()
    T[:3, 3] += S_OBS * (np.asarray(T_WH, float)[:3, :3] @ np.asarray(a_H, float))
    return T


def observe_branch(T_WH, pg, q_now, base_prep_path, T_at, a_H, clear_fn, thr, sig_floor, lo, hi, v_pass):
    """觀察追蹤的連續分支與完整核對（候選；不是已驗證）。pg＝通過線（p0, d, yaw）。base_prep_path＝準備段期間底盤預計經過的位姿序列（由導航路徑取樣）。
    追蹤窗口取樣由 W_MAX＋2.5 cm 到 −2.5 cm（含停止時多走的餘量：v²/2a 約數 mm）；每點核 IK 連續、限位、可操作度（整段 ≥ 下限）、淨空（env／base／self）、所需關節速率。
    準備段：現構型→窗口起點解的關節直線，9 個取樣點配底盤路徑同步取樣，核限位與淨空（env／base／self）。
    回 (dict 或 None, 資訊)。"""
    import itertools
    Tt = observe_target(T_WH, T_at, a_H)
    alongs = np.arange(OBS_ALONG[0], OBS_ALONG[1] - 1e-9, -TRACK_STEP)   # 已核範圍；作業範圍取其內縮一格（起點 0.225 m、停止起點 −0.05 m），兩端各留 2.5–5 cm 給啟動前推誤差與停止行程
    best, n_try, rej = None, 0, {}
    for sd in itertools.product((-0.6, 0.0, 0.6), (-0.3, 0.2), (0.6, 1.4), (-1.5, 0.0, 1.5), (-0.7, 0.7), (0.0,)):
        n_try += 1
        seed, Q, why, sig, mc = np.array(sd), [], None, np.inf, {k: np.inf for k in ('env', 'base', 'self')}
        for al in alongs:
            b = np.r_[pg['p0'] - pg['d'] * al, pg['yaw']]
            q, r, sm = DFK.ik_closure_fc(Tt, b, seed)
            if not Q:
                q = _wrap(q)
            if not (r < 1e-6 and np.isfinite(q).all()) or (Q and np.abs(q - seed).max() > 0.5):
                why = 'track_ik'; break
            if np.any(q < lo) or np.any(q > hi):
                why = 'track_joint_limit'; break
            if sm < sig_floor:
                why = 'track_manipulability'; break
            cl = clear_fn(np.r_[b, q])
            if any(cl[k] < thr[k] for k in mc):
                why = 'track_clearance'; break
            sig = min(sig, sm); mc = {k: min(mc[k], float(cl[k])) for k in mc}
            Q.append(q); seed = q
        if why is None:
            Q = np.array(Q)
            rate = float(np.abs(np.diff(Q, axis=0)).max() / (TRACK_STEP / v_pass))
            if rate > V_ARM:
                why = 'track_rate'
        if why is None:
            qa0 = np.asarray(q_now[3:], float)
            dq = Q[0] - qa0                                           # 關節直線（不繞圈）：兩端都在限位內 ⇒ 途中也在限位內
            if np.any(qa0 < lo) or np.any(qa0 > hi):
                why = 'prep_start_outside_limit'
            else:
                for i_, u_ in enumerate(np.linspace(0.0, 1.0, 9)):
                    bp = np.asarray(base_prep_path[min(int(round(u_ * (len(base_prep_path) - 1))), len(base_prep_path) - 1)], float)
                    cl = clear_fn(np.r_[bp, qa0 + u_ * dq])
                    if any(cl[k] < thr[k] for k in ('env', 'base', 'self')):
                        why = 'prep_clearance:' + ','.join(k for k in ('env', 'base', 'self') if cl[k] < thr[k]); break
                    mc = {k: min(mc[k], float(cl[k])) for k in mc}
        if why is not None:
            rej[why] = rej.get(why, 0) + 1
            continue
        c = dict(q_start_op=Q[1].copy(), q_window_start=Q[0].copy(), q_window=Q, alongs=alongs.copy(), sigma_min=float(sig), min_clear=mc, track_rate_max=rate, prep_move_max_rad=float(np.abs(dq).max()), prep_time_s=float(np.abs(dq).max() / V_PREP), T_target=Tt)
        if best is None or (round(c['prep_time_s'], 2), -c['sigma_min']) < (round(best['prep_time_s'], 2), -best['sigma_min']):
            best = c
    return best, dict(n_seeds=n_try, rejected=rej, status='候選（離線幾何與運動學核對；偵測成立與否、實際追蹤誤差未驗）')


def track_start_decision(q, base_pipe, pg, arm_q, br, phys):
    """追蹤啟動判定（用在途管線前推後的底盤位姿）：回 ('start'｜'wait'｜'refuse', 資訊)。
    start：前推後已進入已核的可達窗口（沿線距離 ≤ 窗口起點）、側向與朝向在線上、手臂已準備到窗口起點解附近。
    refuse：前推後已進入窗口但手臂未就緒或不在線上 ⇒ 不得進入伸臂／抓持區，呼叫端須協同停止（底盤與手臂）。"""
    b = np.asarray(q[:3], float).copy()
    for ub in base_pipe:
        c, s_ = math.cos(b[2]), math.sin(b[2])
        b = b + phys * np.array([c * ub[0] - s_ * ub[1], s_ * ub[0] + c * ub[1], ub[2]])
    r = b[:2] - pg['p0']
    along, lat = float(-(r @ pg['d'])), float(r @ pg['n'])
    yaw_e = float(_wrap(b[2] - pg['yaw']))
    arm_err = float(np.abs(np.asarray(arm_q, float) - br['q_start_op']).max())     # 手臂須已準備到「作業起點」的解（已核範圍邊緣對沿線位置很敏感，不拿邊緣解當起點）
    info = dict(along_pred=round(along, 4), lat_pred_mm=round(lat * 1e3, 2), yaw_err_deg=round(math.degrees(yaw_e), 3), arm_err_rad=round(arm_err, 4))
    a_start = float(br['alongs'][1])                              # 作業起點＝已核範圍內縮一格
    if along > a_start:
        return 'wait', info
    ok = abs(lat) <= FL.LINE_TOL['lateral_m'] and abs(yaw_e) <= FL.LINE_TOL['yaw_rad'] and arm_err <= 0.05
    return ('start' if ok else 'refuse'), info


def select_pregrasp_first(T_WH, q_now, T_at, s_pre, K, clear_fn, thr, sig_floor, lo, hi, v_pass, budget_s=8.0):
    """限時、找到合格候選即止的搜尋（Codex 20261116：背景計算要進時間線）。驗收條件與 select_pregrasp 完全相同，只是逐格（側向距離由 0.45 m 往外、沿線偏移由 0 往外）
    依序試，第一個通過全部條件的就回傳——**不是最優**。超過 budget_s 仍無解 ⇒ 回 None（呼叫端不得伸臂）。"""
    t0 = time.perf_counter()
    so_all = sorted(np.arange(0.36, 0.5601, 0.01), key=lambda x: abs(x - 0.45))
    al_all = sorted(np.arange(-0.12, 0.1201, 0.02), key=lambda x: abs(x))
    n = 0
    for so in so_all:
        for al_ in al_all:
            if time.perf_counter() - t0 > budget_s:
                return None, dict(why='budget_exceeded', n_cells=n, t_total_s=round(time.perf_counter() - t0, 2))
            n += 1
            pg, info = select_pregrasp(T_WH, q_now, T_at, s_pre, K, clear_fn, thr, sig_floor, lo, hi, v_pass, standoffs=[so], alongs=[al_])
            if pg is not None:
                return pg, dict(mode='first_feasible（非最優）', n_cells=n, t_total_s=round(time.perf_counter() - t0, 2), last=info)
    return None, dict(why='no_feasible', n_cells=n, t_total_s=round(time.perf_counter() - t0, 2))
