#!/usr/bin/env python3
"""開門後：放手→退出→收臂→穿門（Codex 20261117_193621；Howard：導航到操作同一個控制器）。
同一個 gmpc_wb 實例（開門執行層的 self.mpc，即抓持段建立的共用實例）；不重設求解器、不清管線（設定點、在途管線、前一筆命令原樣承接）。
階段（自開門完成的下一拍起算）：
  HOLD     0.5 s：底盤參考＝原地，手臂 hold（同一求解）。
  RELEASE  0.8 s：解除綁定（out['release']＝True ⇒ 控制伺服器發 /d2/ctl unbind；Isaac 移除 FixedJoint），手指張開；手臂 hold、底盤原地。
  RETREAT  T_RET：TCP 沿抓取接近軸反向退出 S_RET（幾何同 DoorExitCtrl 的 Δ(s)＝T_at(把手, s)−T_at(把手, 0)；方向在退出開始時以底盤座標固定），
           W-GMPC 手臂 active、tcp_rel_base（TCP 目標掛在底盤上）；底盤參考原地。
  STOW     T_STOW：TCP 相對底盤位姿混接到收臂姿態（起始導航構型 Q_STOW 的正運動學）；姿態偏好 q_post＝Q_STOW 只引導分支。
  DRIVE    手臂 hold；底盤參考與障礙物＝導航前端（Smac 路徑到穿門目標；不再排除門扇附近障礙點：門扇由 LiDAR 與 CBF 照一般障礙物處理）。
           導航到點（halted）且整機慢 ⇒ done ⇒ 有界減速停下 ⇒ finished（不再提交新段，執行器播完已核尾段後停穩驗收）。
停止：求解失敗／障礙物資料過期 ⇒ 有界減速（同抓持控制器的停止減速）；每拍尾段＝stop_traj_exit（同一停法、真管線）。
不宣稱：尾段只做命令減速，未做全身淨空掃掠（實測淨空由控制伺服器每拍監看，違反即中止）。"""
import math
import os
import time

import numpy as np

import door_motm_grasp_ctrl as G

XO, XE, K, B, C0, S = G.XO, G.XE, G.K, G.B, G.C0, G.S
NPH, PHYS, DT, D_STEPS = G.NPH, G.PHYS, G.DT, G.D_STEPS
FINGER_OPEN = 0.0089
T_HOLD, T_REL = 0.5, 0.8
T_RET, S_RET = float(os.environ.get('PASS_T_RET', '1.5')), float(os.environ.get('PASS_S_RET', '0.10'))
T_STOW = float(os.environ.get('PASS_T_STOW', '3.0'))
Q_STOW = np.array([float(x) for x in os.environ.get('PASS_Q_STOW', '-0.0128,-0.0519,0.086,0.0116,-1.6249,-0.0018').split(',')])
W_Q_STOW = float(os.environ.get('PASS_W_Q', '2.0'))
DONE_HOLD_S = 1.0
DONE_Y_MIN = float(os.environ.get('PASS_DONE_Y_MIN', '0.3'))         # 穿門完成：底盤已到門後（里程 y＞此值；門口在 y＝0，客廳側 y＞0）；避免中途點到點被當成完成


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
        self.u_last, self.u_prev = np.asarray(src.u_last, float).copy(), np.asarray(getattr(src, 'u_prev', src.u_last), float).copy()
        self.lim_lo, self.lim_hi = np.asarray(src.lim_lo, float), np.asarray(src.lim_hi, float)
        self.nav_in = getattr(src, 'nav_in', None)
        self.t, self.t_p0, self.n_step = float(t0), float(t0), 0
        self.phase, self.stop, self.abort, self.done, self.finished, self.release = 'HOLD', None, None, False, False, False
        self.finger_cmd = None
        self.tail_cert_mode = False
        self._rec_tail = {1: []}
        self.u_from_base, self.ramp_j = np.asarray(self.pipe[-1][:3], float).copy(), 0
        self.events, self.wg_log, self.tick_log = [], [], []
        self.b_anchor = np.asarray(q_end[:3], float).copy()
        Rg, T_HG, a_H = G.PG.baseline_reference()
        T_at = lambda T, s_: np.asarray(G.E.OT.object_target(T, T_HG, a_H, float(s_))['T_WE'], float)   # noqa: E731
        Th = src.handle_T(float(phi_end))
        self.dvec_w = T_at(Th, S_RET)[:3, 3] - T_at(Th, 0.0)[:3, 3]  # 世界座標退出位移（長 S_RET，沿接近軸遠離把手）
        self.T_rel_r0 = None                                           # 退出開始時 TCP 相對底盤位姿
        self.T_rel_s0 = None                                           # 收臂開始時 TCP 相對底盤位姿
        self.T_rel_stow = np.linalg.inv(_T3(np.zeros(3))) @ K.fk(np.r_[np.zeros(3), Q_STOW], 'link_tcp')
        self.t_halt, self.t_still = None, None
        self._ev('pass_start', phi_end_deg=round(math.degrees(float(phi_end)), 3), dvec_w=[round(float(x), 4) for x in self.dvec_w], q_stow=Q_STOW.tolist())

    def _ev(self, what, **kw):
        self.events.append(dict(t=round(self.t, 3), what=what, **kw))

    def _stop(self, why, **kw):
        if self.stop is None:
            self.stop = dict(why=why, t=round(self.t, 3), **kw)
            self._ev('pass_stop', why=why, **kw)

    def _pred(self, q):
        qp, sp = np.asarray(q, float).copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, self.cfg, 1)
        return qp, sp

    def _stop_traj(self, q, u, e, j0, u_from):
        rec = None if self._rec_tail is None else self._rec_tail.get(e)
        Q, cmds, k_dec = XE.stop_traj_exit(q, self.s, self.pipe, u, u_from, j0, self.cfg, e, rec=rec)
        return Q, None, k_dec

    def _phase_at(self, te):
        if te < T_HOLD:
            return 'HOLD'
        if te < T_HOLD + T_REL:
            return 'RELEASE'
        if te < T_HOLD + T_REL + T_RET:
            return 'RETREAT'
        if te < T_HOLD + T_REL + T_RET + T_STOW:
            return 'STOW'
        return 'DRIVE'

    def _rel_target(self, te, qp):
        """手臂任務（底盤座標）：退出直線、收臂混接。"""
        t1 = T_HOLD + T_REL
        if te < t1 + T_RET:
            u = XE.minjerk(min(max((te - t1) / T_RET, 0.0), 1.0))[0]
            T = self.T_rel_r0.copy()
            Rb = _T3(self.b_anchor)[:3, :3]
            T[:3, 3] = T[:3, 3] + u * (Rb.T @ self.dvec_w)
            return T
        if self.T_rel_s0 is None:
            return self.T_rel_r0
        u = XE.minjerk(min(max((te - t1 - T_RET) / T_STOW, 0.0), 1.0))[0]
        A, Bm = self.T_rel_s0, self.T_rel_stow
        T = np.eye(4)
        T[:3, 3] = (1 - u) * A[:3, 3] + u * Bm[:3, 3]
        T[:3, :3] = A[:3, :3] @ _exp_R(u * _log_R(A[:3, :3].T @ Bm[:3, :3]))
        return T

    def _solve(self, q):
        qp, sp = self._pred(q)
        te0 = self.t + D_STEPS * PHYS - self.t_p0
        X, xi, obs, umin0, src = G.MovingGraspCtrl.nav_window(self, q, qp)
        if obs is None:
            return None, 'nav_obstacles_stale'
        ph = self._phase_at(te0)
        if ph != 'DRIVE':                                              # 操作段：底盤參考原地（錨定位姿），障礙物照導航前端
            X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
            xi = [np.zeros(3)] * (XO.N + 1)
            umin0, src = float('nan'), 'anchor'
        arm = dict(mode='hold')
        if ph in ('RETREAT', 'STOW'):
            if self.T_rel_r0 is None:
                self.T_rel_r0 = np.linalg.inv(_T3(qp[:3])) @ K.fk(qp, 'link_tcp')
            if ph == 'STOW' and self.T_rel_s0 is None:
                self.T_rel_s0 = self._rel_target(T_HOLD + T_REL + T_RET - 1e-9, qp)
            Ts = [self._rel_target(te0 + k * DT, qp) for k in range(XO.N + 1)]
            arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts), w_p=C0.W_FAR,
                       lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper, tcp_rel_base=True, qp_solver='gi', **B.W_ARM)
            if ph == 'STOW' and W_Q_STOW > 0.0:
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
        self.wg_log.append((round(self.t, 3), ph, round((time.perf_counter() - w0) * 1e3, 2), str(r.status), src, len(obs), int(r.cbf_active)))
        if not np.isfinite(np.asarray(r.u_opt, float)).all():
            return None, 'wgmpc_nonfinite'
        if arm['mode'] == 'hold':
            return np.r_[np.asarray(r.u_opt, float), np.zeros(6)], src
        ck = dict(r.arm_check or {})
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None or r.status != 'solved':
            return None, 'wgmpc:%s/%s/%s' % (r.status, ck.get('fallback'), ck.get('failed'))
        return np.r_[np.asarray(r.u_opt, float), np.asarray(r.u_arm_seq[0], float)], src

    def step(self, t, q, phi_meas, finger_meas=None):
        self.t = float(t)
        q = np.asarray(q, float)
        if self.n_step % NPH == 0:
            w_t = time.perf_counter()
            te = self.t - self.t_p0
            ph = self._phase_at(te)
            if ph != self.phase:
                self._ev('pass_phase', phase=ph)
                self.phase = ph
            if ph != 'HOLD' and not self.release:
                self.release = True                                  # 解除綁定（控制伺服器見到即發 unbind；之後每段都帶，避免遺失）
                self.finger_cmd = FINGER_OPEN
            if self.stop is None and not self.done:
                u, why = self._solve(q)
                if u is None:
                    self._stop(why)
                else:
                    self.u_last = u
                if ph == 'DRIVE' and self.stop is None:
                    ni = self.nav_in
                    halted = ni is not None and not ni.get('stale') and bool(ni.get('halted'))
                    self.t_halt = (self.t_halt if self.t_halt is not None else self.t) if halted else None
                    slow = float(np.abs(self.u_last[:3]).max()) < 0.01
                    if self.t_halt is not None and self.t - self.t_halt >= DONE_HOLD_S and slow and te > T_HOLD + T_REL + T_RET + T_STOW + 2.0 and float(q[1]) > DONE_Y_MIN:
                        self.done = True
                        self._ev('pass_done', base=[round(float(x), 4) for x in q[:3]])
            if self.stop is not None or self.done:
                da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * self.cfg.dt
                self.u_last = self.u_last + np.clip(-self.u_last, -da, da)
                quiet = not np.any(self.u_last) and not any(np.any(u_) for u_ in self.pipe)
                self.t_still = (self.t_still if self.t_still is not None else self.t) if quiet else None
                if self.t_still is not None and self.t - self.t_still >= 0.3 and not self.finished:
                    self.finished = True
                    self._ev('pass_finished', done=self.done, stop=None if self.stop is None else self.stop.get('why'))
            self.u_prev = self.u_last.copy()
            self.u_from_base, self.ramp_j = np.asarray(self.pipe[-1][:3], float).copy(), 0
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
