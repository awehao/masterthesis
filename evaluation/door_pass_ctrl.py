#!/usr/bin/env python3
"""開門後：放手→退出→收臂→穿門（Codex 20261117_193621、後續裁定；Howard：導航到操作同一個控制器）。
同一個 gmpc_wb 實例（開門執行層的 self.mpc，即抓持段建立的共用實例）；不重設求解器、不清管線（設定點、在途管線、前一筆命令原樣承接）。
階段（全部以實測完成條件推進；名目剖面時間自該階段開始起算）：
  UNBIND   命令＝零（綁定中的靜止閉鏈：底盤零速、手臂設定點不動 ⇒ 閉鏈自然成立，不產生任何會破壞閉鏈的動作）；
           out['release'] 帶解除請求（版本＝本控制器、請求拍）⇒ 控制伺服器發 /d2/ctl unbind ⇒ 執行端停用 FixedJoint 並回報 /d2/ack unbound（實際生效步）
           ⇒ 快照 unbound ⇒ 才進下一階段。逾時 UNBIND_TO_S ⇒ 停止。
  FINGERS  手指張開命令；命令仍零；兩指實測開度 ≥ FINGER_OK 才進退出；逾時 ⇒ 停止。
  RETREAT  TCP 沿抓取接近軸反向退出 S_RET（幾何同 DoorExitCtrl 的 Δ(s)＝T_at(把手, s)−T_at(把手, 0)；方向在階段開始時以底盤座標固定）；
           W-GMPC 手臂 active、tcp_rel_base；底盤參考＝錨定位姿。完成＝名目時間到且 TCP 相對底盤實際位移 ≥ 0.9·S_RET。
  STOW     TCP 相對底盤位姿混接到收臂姿態（起始導航構型 Q_STOW 的正運動學）；q_post＝Q_STOW 只引導分支。完成＝名目時間到且 TCP 相對誤差 ≤ 10 mm、3°。
  DRIVE    手臂 hold；底盤參考與障礙物＝導航前端（Smac；門扇不排除，照一般障礙物進 CBF）。
           完成（場景專用判定）＝導航到點（halted）≥1 s、命令小、底盤里程 y＞DONE_Y_MIN（已過門口）⇒ done ⇒ 有界減速 ⇒ finished。
尾段（Codex 裁定 2）：每拍候選＝本拍命令一拍＋同一停法（stop_traj_exit：真管線、有界減速）到停穩，逐取樣點核全身淨空（clear_door_fast，同控制伺服器實測監看的門模型、
  門角＝實測）與手臂設定點限位；通過 ⇒ 該軌跡就是本段提交的已核尾段（candidate_extra1）；不過 ⇒ 不提交候選、改有界減速，減速軌跡同樣核過才給尾段；
  核不過 ⇒ 不給尾段 ⇒ 執行器沿用上一份已核尾段。"""
import math
import os
import time

import numpy as np

import door_motm_grasp_ctrl as G
import door_clear_door_fast as CF

XO, XE, K, B, C0, S, W = G.XO, G.XE, G.K, G.B, G.C0, G.S, G.W
NPH, PHYS, DT, D_STEPS = G.NPH, G.PHYS, G.DT, G.D_STEPS
FINGER_OPEN, FINGER_OK = 0.0089, float(os.environ.get('PASS_FINGER_OK', '0.0085'))
UNBIND_TO_S, FINGER_TO_S = 3.0, 3.0
T_RET, S_RET = float(os.environ.get('PASS_T_RET', '1.5')), float(os.environ.get('PASS_S_RET', '0.10'))
T_STOW = float(os.environ.get('PASS_T_STOW', '3.0'))
PH_EXTRA_TO_S = 3.0                                                  # 名目時間後仍未達完成條件的寬限；逾時 ⇒ 停止
Q_STOW = np.array([float(x) for x in os.environ.get('PASS_Q_STOW', '-0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018').split(',')])
W_Q_STOW = float(os.environ.get('PASS_W_Q', '2.0'))
DONE_HOLD_S = 1.0
DONE_Y_MIN = float(os.environ.get('PASS_DONE_Y_MIN', '0.3'))         # 場景專用：門口在 y＝0、客廳側 y＞0
KEYS = ('env', 'base', 'self', 'shell_bar')


def _T3(b):
    c, s = math.cos(b[2]), math.sin(b[2])
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = b[:2]
    return T


def _log_R(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
    return w * th


def _exp_R(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


class DoorPassCtrl:
    def __init__(self, src, t0, q_end, phi_end):
        self.mpc, self.cfg = src.mpc, src.cfg
        self.s = np.asarray(src.s, float).copy()
        self.pipe = [np.asarray(u, float).copy() for u in src.pipe]
        self.u_last, self.u_prev = np.zeros(9), np.asarray(getattr(src, 'u_prev', src.u_last), float).copy()
        self.lim_lo, self.lim_hi = np.asarray(src.lim_lo, float), np.asarray(src.lim_hi, float)
        self.nav_in = getattr(src, 'nav_in', None)
        self.bias_obs = getattr(src, 'bias_obs', None)
        self.T_WH0 = np.asarray(src.T_WH_bind, float).copy()
        self.t, self.n_step = float(t0), 0
        self.phase, self.t_ph0 = 'UNBIND', float(t0)
        self.stop, self.abort, self.done, self.finished = None, None, False, False
        self.release, self.unbound, self.finger_cmd, self.finger_meas = True, False, None, None
        self.phi_now = float(phi_end)
        self.tail_cert_mode = False
        self._rec_tail = {1: []}
        self.u_from_base, self.ramp_j = np.asarray(self.pipe[-1][:3], float).copy(), 0
        self.events, self.wg_log, self.tick_log, self.sweep_log = [], [], [], []
        self.clear_min = {k: 9.0 for k in KEYS}
        self.b_anchor = np.asarray(q_end[:3], float).copy()
        Rg, T_HG, a_H = G.PG.baseline_reference()
        T_at = lambda T, s_: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s_))['T_WE'], float)   # noqa: E731
        Th = src.handle_T(float(phi_end))
        self.dvec_w = T_at(Th, S_RET)[:3, 3] - T_at(Th, 0.0)[:3, 3]  # 世界座標退出位移（長 S_RET，沿接近軸遠離把手）
        self.T_rel_r0, self.T_rel_s0 = None, None
        self.T_rel_stow = K.fk(np.r_[np.zeros(3), Q_STOW], 'link_tcp')
        self.t_halt, self.t_still = None, None
        self._ev('pass_start', phi_end_deg=round(math.degrees(float(phi_end)), 3), dvec_w=[round(float(x), 4) for x in self.dvec_w], q_stow=Q_STOW.tolist())

    def _ev(self, what, **kw):
        self.events.append(dict(t=round(self.t, 3), what=what, **kw))

    def _stop(self, why, **kw):
        if self.stop is None:
            self.stop = dict(why=why, t=round(self.t, 3), **kw)
            self._ev('pass_stop', why=why, **kw)

    def _goto(self, ph, **kw):
        self.phase, self.t_ph0 = ph, self.t
        self._ev('pass_phase', phase=ph, **kw)

    def _pred(self, q):
        qp, sp = np.asarray(q, float).copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, self.cfg, 1)
        return qp, sp

    def _sweep(self, q, u, extra, j0, u_from, rec):
        """同一停法前推到停穩（候選 extra＝1：本拍命令一拍後減速）；逐取樣點核全身淨空與設定點限位。回 (違反 dict 或 None, 取樣數)。"""
        Q, _, _ = XE.stop_traj_exit(q, self.s, self.pipe, u, u_from, j0, self.cfg, extra, rec=rec)
        if rec is not None and any(np.any(r_[0] < self.lim_lo - 1e-9) or np.any(r_[0] > self.lim_hi + 1e-9) for r_ in rec):
            return dict(why='setpoint_limit'), 0
        n, prev = 0, None
        for qs in Q:
            if prev is not None and float(np.abs(qs - prev).max()) < 1e-7:
                continue                                              # 停穩後構型不變（同一點不重算）
            prev = qs
            c = CF.clear_door_fast(np.r_[qs, self.phi_now], self.T_WH0)
            n += 1
            for k in KEYS:
                self.clear_min[k] = min(self.clear_min[k], float(c[k]))
            bad = [k for k in KEYS if c[k] < W.THR[k]]
            if bad:
                return dict(why='clearance', bad=bad, at={k: round(float(c[k]), 4) for k in KEYS}, who={k: str((c.get('who') or {}).get(k)) for k in bad}), n
        return None, n

    def _stop_traj(self, q, u, e, j0, u_from):
        """執行器在本段末另做停止尾段時呼叫（停止模式）：同一停法＋同一淨空核對；不過 ⇒ 回違反（不給尾段，執行器沿用上一份已核尾段）。"""
        rec = None if self._rec_tail is None else self._rec_tail.get(e)
        bad, n = self._sweep(q, u, e, j0, u_from, rec)
        self.sweep_log.append((round(self.t, 3), 'stop_tail', n, None if bad is None else bad.get('why')))
        return None, bad, None

    def _rel_target(self, t_abs):
        if self.phase == 'RETREAT':
            u = XE.minjerk(min(max((t_abs - self.t_ph0) / T_RET, 0.0), 1.0))[0]
            T = self.T_rel_r0.copy()
            T[:3, 3] = T[:3, 3] + u * (_T3(self.b_anchor)[:3, :3].T @ self.dvec_w)
            return T
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / T_STOW, 0.0), 1.0))[0]
        A, Bm = self.T_rel_s0, self.T_rel_stow
        T = np.eye(4)
        T[:3, 3] = (1 - u) * A[:3, 3] + u * Bm[:3, 3]
        T[:3, :3] = A[:3, :3] @ _exp_R(u * _log_R(A[:3, :3].T @ Bm[:3, :3]))
        return T

    def _solve(self, q):
        qp, sp = self._pred(q)
        t_eff = self.t + D_STEPS * PHYS
        X, xi, obs, umin0, src = G.MovingGraspCtrl.nav_window(self, q, qp)
        if obs is None:
            return None, 'nav_obstacles_stale'
        if self.phase != 'DRIVE':                                      # 操作段：底盤參考＝錨定位姿，障礙物照導航前端
            X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
            xi = [np.zeros(3)] * (XO.N + 1)
            umin0, src = float('nan'), 'anchor'
        arm = dict(mode='hold')
        if self.phase in ('RETREAT', 'STOW'):
            Ts = [self._rel_target(t_eff + k * DT) for k in range(XO.N + 1)]
            arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts), w_p=C0.W_FAR,
                       lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper, tcp_rel_base=True, qp_solver='gi', **B.W_ARM)
            if self.phase == 'STOW' and W_Q_STOW > 0.0:
                arm.update(q_post=np.tile(Q_STOW, (XO.N, 1)), w_q=W_Q_STOW)
        mpc = self.mpc
        u_min_saved = np.asarray(mpc.cfg.u_min, float).copy()
        if math.isfinite(umin0):
            mpc.cfg.u_min = np.array([umin0, u_min_saved[1], u_min_saved[2]])
        w0 = time.perf_counter()
        try:
            r = mpc.solve(X_now=XO.from_xytheta(*qp[:3]), X_ref_win=np.array(X), xi_ref_win=np.array(xi), xi_prev=self.u_prev[:3], arm=arm, obstacles=obs)
        finally:
            mpc.cfg.u_min = u_min_saved
        self.wg_log.append((round(self.t, 3), self.phase, round((time.perf_counter() - w0) * 1e3, 2), str(r.status), src, len(obs), int(r.cbf_active)))
        if not np.isfinite(np.asarray(r.u_opt, float)).all():
            return None, 'wgmpc_nonfinite'
        if arm['mode'] == 'hold':
            return np.r_[np.asarray(r.u_opt, float), np.zeros(6)], src
        ck = dict(r.arm_check or {})
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None or r.status != 'solved':
            return None, 'wgmpc:%s/%s/%s' % (r.status, ck.get('fallback'), ck.get('failed'))
        return np.r_[np.asarray(r.u_opt, float), np.asarray(r.u_arm_seq[0], float)], src

    def _advance(self, q):
        """實測完成條件推進階段（q＝本拍預測構型；門角、手指＝實測）。"""
        el = self.t - self.t_ph0
        if self.phase == 'UNBIND':
            if self.unbound:
                self.finger_cmd = FINGER_OPEN
                self._goto('FINGERS')
            elif el > UNBIND_TO_S:
                self._stop('unbind_ack_timeout')
        elif self.phase == 'FINGERS':
            fm = self.finger_meas
            if fm is not None and min(fm) >= FINGER_OK:
                W.FINGER_Q = float(np.mean(fm))
                self.T_rel_r0 = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
                self._goto('RETREAT', finger=[round(float(x), 5) for x in fm])
            elif el > FINGER_TO_S:
                self._stop('finger_open_timeout', finger=None if fm is None else [round(float(x), 5) for x in fm])
        elif self.phase == 'RETREAT':
            T_now = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
            moved = float(np.dot(T_now[:3, 3] - self.T_rel_r0[:3, 3], _T3(self.b_anchor)[:3, :3].T @ self.dvec_w)) / S_RET
            if el >= T_RET and moved >= 0.9 * S_RET:
                self.T_rel_s0 = T_now
                self._goto('STOW', retreat_m=round(moved, 4))
            elif el > T_RET + PH_EXTRA_TO_S:
                self._stop('retreat_not_reached', retreat_m=round(moved, 4))
        elif self.phase == 'STOW':
            T_now = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
            ep = float(np.linalg.norm(T_now[:3, 3] - self.T_rel_stow[:3, 3]))
            er = math.degrees(float(np.linalg.norm(_log_R(T_now[:3, :3].T @ self.T_rel_stow[:3, :3]))))
            if el >= T_STOW and ep <= 0.010 and er <= 3.0:
                self._goto('DRIVE', stow_err_mm=round(ep * 1e3, 2), stow_err_deg=round(er, 2))
            elif el > T_STOW + PH_EXTRA_TO_S:
                self._stop('stow_not_reached', stow_err_mm=round(ep * 1e3, 2), stow_err_deg=round(er, 2))

    def step(self, t, q, phi_meas, finger_meas=None):
        self.t = float(t)
        q = np.asarray(q, float)
        self.phi_now = float(phi_meas)
        if finger_meas is not None:
            self.finger_meas = (float(finger_meas[0]), float(finger_meas[1]))
        if self.n_step % NPH == 0:
            w_t = time.perf_counter()
            u_from = np.asarray(self.pipe[-1][:3], float).copy()
            if self.stop is None and not self.done:
                self._advance(q)
            cand = None
            if self.stop is None and not self.done:
                if self.phase in ('UNBIND', 'FINGERS'):
                    cand = np.zeros(9)                               # 綁定中／放手中：靜止（閉鏈自然成立）
                else:
                    u, why = self._solve(q)
                    if u is None:
                        self._stop(why)
                    else:
                        cand = u
                if cand is not None and self.phase == 'DRIVE':
                    ni = self.nav_in
                    halted = ni is not None and not ni.get('stale') and bool(ni.get('halted'))
                    self.t_halt = (self.t_halt if self.t_halt is not None else self.t) if halted else None
                    if self.t_halt is not None and self.t - self.t_halt >= DONE_HOLD_S and float(np.abs(cand[:3]).max()) < 0.01 and float(q[1]) > DONE_Y_MIN:
                        self.done = True
                        self._ev('pass_done', base=[round(float(x), 4) for x in q[:3]])
                        cand = None
            if cand is not None:                                     # 候選＋完整尾段核對（通過 ⇒ 該軌跡即本段提交的已核尾段）
                rec = self._rec_tail.get(1) if self._rec_tail is not None else None
                if rec is not None:
                    rec.clear()
                bad, n = self._sweep(q, cand, 1, 0, u_from, rec)
                self.sweep_log.append((round(self.t, 3), 'candidate', n, None if bad is None else bad.get('why')))
                if bad is not None:
                    if rec is not None:
                        rec.clear()
                    self._stop('candidate_tail_rejected', **bad)
                else:
                    self.u_last = cand
            if self.stop is not None or self.done:
                da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * self.cfg.dt
                self.u_last = self.u_last + np.clip(-self.u_last, -da, da)
                quiet = not np.any(self.u_last) and not any(np.any(u_) for u_ in self.pipe)
                self.t_still = (self.t_still if self.t_still is not None else self.t) if quiet else None
                if self.t_still is not None and self.t - self.t_still >= 0.3 and not self.finished:
                    self.finished = True
                    self._ev('pass_finished', done=self.done, stop=None if self.stop is None else self.stop.get('why'))
            self.u_prev = self.u_last.copy()
            self.u_from_base, self.ramp_j = u_from, 0
            self.tick_log.append((round(self.t, 3), round((time.perf_counter() - w_t) * 1e3, 3), self.phase))
        cmd = XO.ramp_cmd(self.u_from_base, self.u_last, self.ramp_j)
        self.ramp_j += 1
        self.pipe.append(cmd)
        ua = self.pipe.pop(0)
        nxt = self.s + PHYS * ua[3:]
        if np.any(nxt < self.lim_lo) or np.any(nxt > self.lim_hi):
            nxt = np.clip(nxt, self.lim_lo, self.lim_hi)
            self._stop('setpoint_limit')
        self.s = nxt
        self.n_step += 1
        return dict(arm_target=self.s.copy(), base_twist=ua[:3].copy(), finger=self.finger_cmd, ready=False, ready_ver=None, stopped=self.stop is not None, abort=False, done=self.done)
