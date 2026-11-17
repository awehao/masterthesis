#!/usr/bin/env python3
"""穿門後：茶几圓柱體移動抓取→抬升→持物返回起點（Howard 11/17 完整任務；Claude 自行裁定 20261117_2315，Codex 用量用完期間）。
同一個 gmpc_wb 實例（承接穿門控制器的 mpc、設定點、在途管線、前一筆命令、導航前端輸入；不重設、不清管線、任務切換不強制停車）。
停止尾段、候選核對、步進骨架沿用 DoorPassCtrl（每拍候選＋同一停法到停穩的全身淨空核對；不過不給尾段）。
階段（實測完成條件推進）：
  APPROACH  監督送的抓取底盤位姿＝導航目標（Smac）；TCP 參考＝接近開始時 TCP 相對底盤位姿（掛在參考底盤上）→ 沿接近軸的抓取線（d_line→0）混接；
            導航到點後底盤收斂到抓取底盤位姿。完成＝名目時間到且 TCP 對抓取位姿 ≤ 5 mm／3°（實測構型）。
  CLOSE     底盤原地、手臂保持抓取位姿、手指命令斜降到 0。
  CONFIRM   夾住證據＝兩指接觸力各 ≥ CONTACT_N 且手指未全閉（≥ FINGER_EMPTY），連續 EVID_S；期限 EVID_TO_S ⇒ 停止（未夾到）。
  LIFT      TCP 垂直抬升 LIFT_M（最小急動 T_LIFT）。
  VERIFY    抬升後仍有夾住證據連續 EVID_S ⇒ pick_lifted；否則停止（掉落）。
  CARRY     TCP 相對底盤混接到收臂姿態（持物）；q_post 只引導分支。
  RETURN    手臂 hold；底盤參考＋障礙物＝導航前端（監督送起點目標，經開著的門）。完成（場景專用）＝導航到點 ≥1 s、命令小、底盤里程 y＜RET_Y_MAX ⇒ done ⇒ 有界減速 ⇒ finished。
不宣稱：持物期間淨空模型不含圓柱本體（只核機器人與場景）；抓持＝Isaac 真實摩擦接觸（不綁定、不把圓柱固定到 TCP）；成功判定只用接觸力與手指開度（真值只做驗收記錄）。"""
import math
import os
import time

import numpy as np

import door_pass_ctrl as P

G, XO, XE, K, B, C0, S, W = P.G, P.XO, P.XE, P.K, P.B, P.C0, P.S, P.W
NPH, PHYS, DT, D_STEPS = P.NPH, P.PHYS, P.DT, P.D_STEPS
_T3, _log_R, _exp_R = P._T3, P._log_R, P._exp_R
D_LINE = float(os.environ.get('PICK_D_LINE', '0.08'))
T_SETTLE = float(os.environ.get('PICK_T_SETTLE', '1.0'))
GRASP_DEPTH = float(os.environ.get('PICK_GRASP_DEPTH', '0.004'))     # 瓶子線 bottle_s0_ctrl.GRASP_DEPTH
T_CLOSE, EVID_S, EVID_TO_S = 0.6, 0.3, 3.0
CONTACT_N = float(os.environ.get('PICK_CONTACT_N', '0.5'))
FINGER_EMPTY = float(os.environ.get('PICK_FINGER_EMPTY', '0.0015'))   # 手指開度低於此 ⇒ 判定夾空（全閉）
LIFT_M, T_LIFT = float(os.environ.get('PICK_LIFT_M', '0.05')), 1.2
T_CARRY = 3.0
RET_Y_MAX = float(os.environ.get('PICK_RET_Y_MAX', '-2.3'))           # 場景專用：起點 (1.5, −2.8)
PH_TO_S = 4.0


class TablePickCtrl(P.DoorPassCtrl):
    def __init__(self, src, t0, app):
        """src＝穿門控制器（DRIVE 中）；app＝監督送的取物封包：T_grasp（odom 4×4）、base_goal（odom x,y,yaw）、T（接近時長）、T_b、t_l0。"""
        self.mpc, self.cfg = src.mpc, src.cfg
        self.s = np.asarray(src.s, float).copy()
        self.pipe = [np.asarray(u, float).copy() for u in src.pipe]
        self.u_last, self.u_prev = np.asarray(src.u_last, float).copy(), np.asarray(src.u_prev, float).copy()
        self.lim_lo, self.lim_hi = src.lim_lo, src.lim_hi
        self.nav_in = getattr(src, 'nav_in', None)
        self.T_WH0, self.phi_now = src.T_WH0, src.phi_now
        self.t, self.n_step = float(t0), 0
        self.stop, self.abort, self.done, self.finished = None, None, False, False
        self.release, self.unbound, self.finger_cmd, self.finger_meas, self.finger_force = True, True, P.FINGER_OPEN, src.finger_meas, None
        self.tail_cert_mode = False
        self._rec_tail = {1: []}
        self.u_from_base, self.ramp_j = np.asarray(self.pipe[-1][:3], float).copy(), 0
        self.events, self.wg_log, self.tick_log, self.sweep_log = [], [], [], []
        self.clear_min = {k: 9.0 for k in P.KEYS}
        self.T_rel_stow = src.T_rel_stow
        self.t_halt, self.t_still = None, None
        self.T_grasp = np.asarray(app['T_grasp'], float).copy()
        self.T_grasp[:3, 3] = self.T_grasp[:3, 3] + GRASP_DEPTH * self.T_grasp[:3, 2]   # 同瓶子線：TCP 沿工具軸越過圓柱軸 4 mm，軸心落在指片（TCP 後 1.3–21.3 mm）內（第十二趟 TCP 在軸心上 ⇒ 指尖夾弦、抬升滑落）
        self.app_prof = dict(base_goal=list(app['base_goal']), door=None)   # nav_window：導航到點後收斂到抓取底盤位姿
        self.T_app, self.T_b, self.t_l0 = float(app['T']), float(app['T_b']), float(app['t_l0'])
        self.T_rel0, self.b_anchor, self.n_ev, self.T_lift0, self.T_rel_s0 = None, None, 0, None, None
        self.ret_sent = False
        self.phase, self.t_ph0 = 'APPROACH', float(t0)
        self._ev('pick_start', T_grasp=[round(float(x), 4) for x in self.T_grasp[:3, 3]], base_goal=[round(float(x), 4) for x in app['base_goal']], T=round(self.T_app, 3))

    def _stop(self, why, /, **kw):
        if 'why' in kw:
            kw['detail_why'] = kw.pop('why')                          # 違反 dict 自帶 why（如 clearance）⇒ 改名，避免與停止原因重複
        if self.stop is None:
            self.stop = dict(why=why, t=round(self.t, 3), **kw)
            self._ev('pick_stop', why=why, phase=self.phase, **kw)

    def _goto(self, ph, **kw):
        self.phase, self.t_ph0 = ph, self.t
        self._ev('pick_phase', phase=ph, **kw)

    def _T_line(self, s_):
        T = self.T_grasp.copy()
        T[:3, 3] = T[:3, 3] - s_ * T[:3, 2]                            # 沿接近軸（TCP z）後退 s
        return T

    def _app_target(self, t_abs, Xk):
        """接近剖面：參考底盤上的起始相對位姿 → 抓取線（d_line→0，接近線提早 t_settle 到底）。"""
        t = t_abs - self.t_ph0
        Te = self.T_app - T_SETTLE
        u = min(max((t - self.t_l0) / max(Te - self.t_l0, 1e-6), 0.0), 1.0)
        s_ = D_LINE * (1.0 - XE.minjerk(u)[0]) if t < Te else 0.0
        Tl = self._T_line(max(s_, 0.0))
        beta = XE.minjerk(min(max(t / max(self.T_b, 1e-6), 0.0), 1.0))[0]
        if beta >= 1.0 - 1e-12:
            return Tl
        c_, s2 = math.cos(Xk[2]), math.sin(Xk[2])
        Tb = np.eye(4); Tb[:2, :2] = [[c_, -s2], [s2, c_]]; Tb[:2, 3] = Xk[:2]
        Tb = Tb @ self.T_rel0
        T = np.eye(4)
        T[:3, 3] = (1 - beta) * Tb[:3, 3] + beta * Tl[:3, 3]
        T[:3, :3] = Tb[:3, :3] @ _exp_R(beta * _log_R(Tb[:3, :3].T @ Tl[:3, :3]))
        return T

    def _lift_target(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / T_LIFT, 0.0), 1.0))[0]
        T = self.T_lift0.copy()
        T[2, 3] += u * LIFT_M
        return T

    def _carry_target(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / T_CARRY, 0.0), 1.0))[0]
        A, Bm = self.T_rel_s0, self.T_rel_stow
        T = np.eye(4)
        T[:3, 3] = (1 - u) * A[:3, 3] + u * Bm[:3, 3]
        T[:3, :3] = A[:3, :3] @ _exp_R(u * _log_R(A[:3, :3].T @ Bm[:3, :3]))
        return T

    def _held(self):
        f, fm = self.finger_force, self.finger_meas
        return f is not None and fm is not None and min(f) >= CONTACT_N and min(fm) >= FINGER_EMPTY

    def _advance(self, q):
        el = self.t - self.t_ph0
        if self.T_rel0 is None:
            self.T_rel0 = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
        if self.phase == 'APPROACH':
            Tc = K.fk(q, 'link_tcp')
            ep = float(np.linalg.norm(Tc[:3, 3] - self.T_grasp[:3, 3]))
            er = math.degrees(float(np.linalg.norm(_log_R(self.T_grasp[:3, :3].T @ Tc[:3, :3]))))
            if el >= self.T_app and ep <= 0.005 and er <= 3.0:
                self.b_anchor = np.asarray(q[:3], float).copy()
                self.finger_cmd = P.FINGER_OPEN
                self._goto('CLOSE', ep_mm=round(ep * 1e3, 2), er_deg=round(er, 2))
            elif el > self.T_app + PH_TO_S:
                self._stop('approach_not_reached', ep_mm=round(ep * 1e3, 2), er_deg=round(er, 2))
        elif self.phase == 'CLOSE':
            self.finger_cmd = max(0.0, P.FINGER_OPEN * (1.0 - el / T_CLOSE))
            if el >= T_CLOSE:
                self.finger_cmd, self.n_ev = 0.0, 0
                self._goto('CONFIRM')
        elif self.phase == 'CONFIRM':
            self.n_ev = self.n_ev + 1 if self._held() else 0
            if self.n_ev * DT >= EVID_S - 1e-9:
                self.T_lift0 = K.fk(q, 'link_tcp')
                self._goto('LIFT', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas], force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
            elif el > EVID_TO_S:
                self._stop('grasp_evidence_timeout', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas], force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
        elif self.phase == 'LIFT':
            if el >= T_LIFT:
                self.n_ev = 0
                self._goto('VERIFY')
        elif self.phase == 'VERIFY':
            self.n_ev = self.n_ev + 1 if self._held() else 0
            if self.n_ev * DT >= EVID_S - 1e-9:
                self.T_rel_s0 = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
                self._ev('pick_lifted', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas], force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
                self._goto('CARRY')
            elif el > EVID_TO_S:
                self._stop('lost_after_lift', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas], force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
        elif self.phase == 'CARRY':
            T_now = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
            ep = float(np.linalg.norm(T_now[:3, 3] - self.T_rel_stow[:3, 3]))
            if el >= T_CARRY and ep <= 0.010:
                self._goto('RETURN', carry_err_mm=round(ep * 1e3, 2))
            elif el > T_CARRY + PH_TO_S:
                self._stop('carry_not_reached', carry_err_mm=round(ep * 1e3, 2))
            if not self._held():
                self._stop('lost_during_carry')
        elif self.phase == 'RETURN':
            if not self._held():
                self._stop('lost_during_return')

    def _solve(self, q):
        qp, sp = self._pred(q)
        t_eff = self.t + D_STEPS * PHYS
        X, xi, obs, umin0, src = G.MovingGraspCtrl.nav_window(self, q, qp)
        if obs is None:
            return None, 'nav_obstacles_stale'
        if self.phase in ('CLOSE', 'CONFIRM', 'LIFT', 'VERIFY', 'CARRY'):   # 接觸／持物操作段：底盤原地
            X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
            xi = [np.zeros(3)] * (XO.N + 1)
            umin0, src = float('nan'), 'anchor'
        if self.phase == 'RETURN':
            self.app_prof = None                                      # 返回段：照導航前端（不再收斂到抓取位）
        arm = dict(mode='hold')
        Xp = [(float(Xk[0, 2]), float(Xk[1, 2]), math.atan2(float(Xk[1, 0]), float(Xk[0, 0]))) for Xk in X]
        if self.phase in ('APPROACH', 'CLOSE', 'CONFIRM', 'LIFT', 'VERIFY'):
            if self.phase == 'APPROACH':
                Ts = [self._app_target(t_eff + k * DT, Xp[k]) for k in range(XO.N + 1)]
            elif self.phase in ('LIFT', 'VERIFY'):
                Ts = [self._lift_target(t_eff + k * DT) if self.phase == 'LIFT' else self._lift_target(self.t_ph0 + T_LIFT + 1.0) for k in range(XO.N + 1)]
            else:
                Ts = [self.T_grasp] * (XO.N + 1)
            arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts),
                       w_p=C0.W_NEAR if self.phase != 'APPROACH' else C0.W_FAR, lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper, tcp_rel_base=False, qp_solver='gi', **B.W_ARM)
        elif self.phase == 'CARRY':
            Ts = [self._carry_target(t_eff + k * DT) for k in range(XO.N + 1)]
            arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts), w_p=C0.W_FAR,
                       lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper, tcp_rel_base=True, qp_solver='gi', q_post=np.tile(P.Q_STOW, (XO.N, 1)), w_q=P.W_Q_STOW, **B.W_ARM)
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

    def step(self, t, q, phi_meas, finger_meas=None, finger_force=None):
        if finger_force is not None:
            self.finger_force = (float(finger_force[0]), float(finger_force[1]))
        if self.n_step % NPH == 0 and self.phase == 'RETURN' and self.stop is None and not self.done:   # 返回完成判定（場景專用）
            ni = self.nav_in
            halted = ni is not None and not ni.get('stale') and bool(ni.get('halted'))
            self.t_halt = (self.t_halt if self.t_halt is not None else float(t)) if halted else None
            if self.t_halt is not None and float(t) - self.t_halt >= P.DONE_HOLD_S and float(q[1]) < RET_Y_MAX and float(np.abs(self.u_last[:3]).max()) < 0.01:
                self.done = True
                self._ev('pick_return_done', base=[round(float(x), 4) for x in q[:3]])
        return super().step(t, q, phi_meas, finger_meas=finger_meas)
