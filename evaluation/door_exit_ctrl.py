#!/usr/bin/env python3
"""開門後退出、收臂、繞門扇與穿門的執行層（設計 r1／Codex 20261113_010828）。假世界與（日後）Isaac 共用同一入口 step()。

段別（依序；每段參考走完且停穩才進下一段）：
  X1  退出：沿**當下**工具接近軸反向，Δ(s)＝T_at(把手(φ₀), s)−T_at(把手(φ₀), 0)，s：0→S_PRE（最小急動時間律），姿態保持；底盤保持；
      手臂＝命令生效時刻預測狀態的 IK（以前一解為種子），設定點＝q_ik＋τ·q̇_ik−off(q)。
  X1b 後退：**保持關節構型**（手臂設定點不動），底盤沿本體 −x 後退 BACK（梯形速度）。
  X2  收臂：關節空間最小急動插值到既有行車姿態，底盤保持。
  X3  繞門扇＋X4 穿門：照給定路徑點的運動學跟隨（**A* 理想跟隨，代替導航堆疊**；只能報幾何／運動學串接），偏航策略＝先原地轉到 90° 再平移；手臂保持行車姿態。
偏置 off(q)：X1 起點實測 (q₀−s₀)（解綁張爪後、靜止）為錨，加 URDF 重力變化／kp（同開門 A 的定義；只取一次）。
門：獨立被動狀態（不由 TCP 反推）；每拍核實測門角在帶 [φ₀−δφ, φ₀＋(ω₀＋δω)/c_lo·(1−e^{−c_lo t})＋δφ] 內，出帶 ⇒ 停止（door_out_of_band）。
每拍核：實測淨空（門扇／立柱／橫桿轉到實測門角，手指實際開度；兩指—橫桿只在 X1 起始包覆區豁免，間隙達門檻即永久撤除）；
        候選提交前「現在停／晚一拍停」掃掠（真管線、真減速、門角帶兩端）；手臂速度／加速度、本體速度框、輪面速度與加速度。
停止：任何違反 ⇒ 受控停止（底盤有界減速、手臂設定點凍結；不再前進）；停止有效性＝管線清空、底盤與手臂停穩。"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_open_exec_ctrl as XO                                     # noqa: E402
import door_body_model as BM                                         # noqa: E402
import door_limits as LIM                                            # noqa: E402
import door_insert_ctrl as DI                                        # noqa: E402

W, K, C0, B, S, PG = XO.W, XO.K, XO.C0, XO.B, XO.S, XO.W.DA.PG
E = DI.R4.E
PHYS, NPH, D_STEPS, DT = XO.PHYS, XO.NPH, XO.D_STEPS, XO.DT
KEYS = XO.KEYS
FINGER_OPEN = 0.0089
I_CZ = 8.0 * (0.95 ** 2 + 0.04 ** 2) / 12.0
I_H = I_CZ + 8.0 * (0.475 ** 2 + 0.02 ** 2)
C_EFF = 2.0 * I_CZ / I_H
C_LO = 0.8 * C_EFF
D_OMEGA, D_PHI = math.radians(0.01), math.radians(0.01)
SETTLE = dict(arm_rps=0.02, base_mps=0.005, yaw_rps=0.01, n_ticks=4)
V_NAV, A_NAV, W_TURN, A_TURN = 0.25, 0.5, 0.5, 1.0                  # 跟隨段（運動學代替導航）：平移速度／加速度、轉向角速度／角加速度
BACK, V_BACK, A_BACK = 0.25, 0.10, 0.5
STOP_EXTRAS = (0, 1)                                                 # 現在停／晚一拍停


def minjerk(u):
    u = min(max(u, 0.0), 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u), (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4)


def trap(t, dist, v, a):
    """梯形速度：回 (位移, 速度)。"""
    ta = v / a
    if dist < v * ta:                                               # 三角形
        ta = math.sqrt(dist / a); v = a * ta
    tc = (dist - v * ta) / v
    T = 2 * ta + tc
    if t <= 0:
        return 0.0, 0.0
    if t < ta:
        return 0.5 * a * t * t, a * t
    if t < ta + tc:
        return 0.5 * a * ta * ta + v * (t - ta), v
    if t < T:
        r = T - t
        return dist - 0.5 * a * r * r, a * r
    return dist, 0.0


def trap_T(dist, v, a):
    ta = v / a
    if dist < v * ta:
        return 2 * math.sqrt(dist / a)
    return 2 * ta + (dist - v * ta) / v


class DoorExitCtrl:
    def __init__(self, ident, T_WH_bind, phi_bind, s, pipe, u_last, u_prev, bias, finger_q, phi0, omega0, path_xy, q_stow, joint_kp=600.0, sweep_stride=3):
        self.cfg = C0.make_cfg(ident)
        self.cfg.base_fixed = False
        self.cfg.arm_model.bias = np.asarray(bias, float).copy()
        self.T_WH_bind, self.phi_bind = np.asarray(T_WH_bind, float), float(phi_bind)
        W.set_model(phi0=self.phi_bind, finger_q=float(finger_q))
        self.finger_q = float(finger_q)
        self.s = np.asarray(s, float).copy()
        self.pipe = [np.asarray(u, float).copy() for u in pipe]
        self.u_last, self.u_prev = np.asarray(u_last, float).copy(), np.asarray(u_prev, float).copy()
        self.u_from_base, self.ramp_j = np.asarray(self.pipe[-1][:3], float).copy(), NPH
        self.phi0, self.omega0 = float(phi0), float(omega0)
        self.path = np.asarray(path_xy, float)
        self.q_stow = np.asarray(q_stow, float)
        self.joint_kp = float(joint_kp)
        self.lim_lo, self.lim_hi = np.asarray(XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(XO.LITE6_SAFE.upper, float) - 0.05
        al = np.asarray(self.cfg.arm_model.alpha, float)
        self.tau = PHYS / al
        self.va = float(B.W_ARM['v_arm'])
        Rg, self.T_HG, self.a_H = PG.baseline_reference()
        self.T_at = lambda T, s_: np.asarray(E.OT.object_target(T, self.T_HG, self.a_H, float(s_))['T_WE'], float)   # noqa: E731
        self.S_EXIT = float(PG.S_PRE)
        self.t, self.n_step = 0.0, 0
        self.phase, self.t_phase, self.ph_hist = 'X1', 0.0, [('X1', 0.0)]
        self.anchor, self.off0, self.G0, self.bodies = None, None, None, BM.arm_bodies(K)
        self.q_ik_prev, self.q_ref0, self.b_ref0, self.Tn0, self.Th0 = None, None, None, None, None
        self.finger_exempt, self.finger_gap_prev, self.finger_release_t = True, None, None
        self.stop, self.stop_t, self.stop_ok, self.done = None, None, None, False
        self.settle_n, self.q_tick, self.rate = 0, None, dict(arm=0.0, base=0.0, yaw=0.0)
        self.mins = {k: np.inf for k in KEYS}
        self.mins.update(finger_bar_after_release=np.inf, stop_sweep=np.inf)
        self.lim = dict(arm_rate=0.0, arm_acc=0.0, wheel_v=0.0, wheel_a=0.0, body_v=[0.0, 0.0, 0.0], body_a=[0.0, 0.0, 0.0])
        self.events, self.rec = [], []
        self.sweep_stride = int(sweep_stride)
        self.u_body_prev = np.zeros(3)
        self.nav = None
        self._ref_done = False

    # ---- 門角帶
    def band(self, t):
        hi = (abs(self.omega0) + D_OMEGA) / C_LO * (1.0 - math.exp(-C_LO * t))
        lo = 0.0 if self.omega0 - D_OMEGA >= 0.0 else -(D_OMEGA - self.omega0) / C_LO * (1.0 - math.exp(-C_LO * t))
        return self.phi0 + lo - D_PHI, self.phi0 + hi + D_PHI

    def _ev(self, what, **kw):
        self.events.append(dict(t=round(self.t, 3), what=what, **kw))

    def _stop(self, why, **kw):
        if self.stop is None:
            self.stop, self.stop_t = dict(t=round(self.t, 3), why=why, phase=self.phase, **kw), self.t
            self._ev('stop', why=why, phase=self.phase, **kw)

    def off(self, q9):
        return self.off0 + (BM.arm_gravity(K, q9, self.bodies) - self.G0) / self.joint_kp

    # ---- 淨空
    def finger_bar(self, q9, phi):
        _, bar = PG.handle_shapes(W.handle_pose(self.T_WH_bind, phi))
        d = np.inf
        for n in ('uflite_finger1', 'uflite_finger2'):
            T = K.fk(q9, n)
            P = (T[:3, :3] @ DI.DOC.CL[n].T).T + T[:3, 3] + (1.0 if n.endswith('1') else -1.0) * self.finger_q * T[:3, 1]
            d = min(d, float(PG.bar_dist(P, bar).min()))
        return d

    def clear(self, q9, phis):
        out = {k: np.inf for k in KEYS}
        out['finger_bar'] = np.inf
        for ph in phis:
            c = W.clear_door(np.r_[q9, ph], self.T_WH_bind)
            for k in KEYS:
                out[k] = min(out[k], float(c[k]))
            out['finger_bar'] = min(out['finger_bar'], self.finger_bar(q9, ph))
        return out

    def _bad(self, c, during_stop=False):
        bad = [k for k in KEYS if c[k] < W.THR[k]]
        if not self.finger_exempt and c['finger_bar'] < W.THR['env']:
            bad.append('finger_bar')
        return bad

    # ---- 預測（管線清空後的狀態）
    def _pred(self, q):
        qp, sp = np.asarray(q, float).copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, self.cfg, 1)
        return qp, sp

    def _ik(self, b, T, seed):
        q = np.asarray(seed, float).copy()
        for _ in range(30):
            q9 = np.r_[b, q]
            Tc = K.fk(q9, 'link_tcp')
            dR = T[:3, :3] @ Tc[:3, :3].T
            e = np.r_[T[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
            if np.linalg.norm(e) < 1e-10:
                break
            J = K.jacobian(q9, 'link_tcp')[:, 3:9]
            q = q + np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
        return q, float(np.linalg.norm(e))

    # ---- 各段命令（本體速度 3＋手臂設定點速度 6）
    def _cmd(self, q, qp, sp):
        te = self.t + D_STEPS * PHYS + DT - self.t_phase                  # 本拍命令生效後一拍末（相對段起點）
        u = np.zeros(9)
        if self.phase == 'X1':
            T1 = 1.0
            a, _ = minjerk(te / T1)
            a0, _ = minjerk((te - DT) / T1)
            Tt = self.Tn0.copy()
            Tt[:3, 3] = self.Tn0[:3, 3] + (self.T_at(self.Th0, a * self.S_EXIT)[:3, 3] - self.T_at(self.Th0, 0.0)[:3, 3])
            T0_ = self.Tn0.copy()
            T0_[:3, 3] = self.Tn0[:3, 3] + (self.T_at(self.Th0, a0 * self.S_EXIT)[:3, 3] - self.T_at(self.Th0, 0.0)[:3, 3])
            seed = self.q_ik_prev if self.q_ik_prev is not None else qp[3:]
            q1, r1 = self._ik(qp[:3], Tt, seed)
            q0_, r0 = self._ik(qp[:3], T0_, q1)
            if max(r1, r0) > 1e-6:
                return None, 'ik_fail'
            self.q_ik_prev = q1
            S1 = q1 + self.tau * (q1 - q0_) / DT - self.off(np.r_[qp[:3], q1])
            u[3:] = (S1 - sp) / DT
            done_ref = te >= T1
        elif self.phase == 'X1b':
            Tb = trap_T(BACK, V_BACK, A_BACK)
            _, v = trap(te - 0.5 * DT, BACK, V_BACK, A_BACK)
            u[0] = -v
            done_ref = te >= Tb + DT
        elif self.phase == 'X2':
            span = float(np.abs(self.q_stow - self.q_ref0).max())
            T2 = max(3.5, span / 0.45)
            a, da = minjerk(te / T2)
            qr = self.q_ref0 + a * (self.q_stow - self.q_ref0)
            qd = (da / T2) * (self.q_stow - self.q_ref0)
            S1 = qr + self.tau * qd - self.off(np.r_[qp[:3], qr])
            u[3:] = (S1 - sp) / DT
            done_ref = te >= T2
        else:                                                           # X3＋X4：先轉向 90°，再沿路徑
            n = self.nav
            yaw_t = math.pi / 2
            if n['mode'] == 'turn':
                dyaw = (yaw_t - qp[2] + math.pi) % (2 * math.pi) - math.pi
                w = float(np.clip(1.5 * dyaw, -W_TURN, W_TURN))
                wp = self.u_body_prev[2]
                w = float(np.clip(w, wp - A_TURN * DT, wp + A_TURN * DT))
                u[2] = w
                if abs(dyaw) < math.radians(0.5) and abs(wp) < 0.02:
                    n['mode'], n['t0'] = 'move', self.t
                done_ref = False
            else:
                tt = self.t + D_STEPS * PHYS + 0.5 * DT - n['t0']
                sd, vd = trap(tt, n['L'], V_NAV, A_NAV)
                i = int(np.searchsorted(n['cum'], sd)); i = min(max(i, 1), len(n['cum']) - 1)
                f = (sd - n['cum'][i - 1]) / max(n['cum'][i] - n['cum'][i - 1], 1e-9)
                pd = n['P'][i - 1] + f * (n['P'][i] - n['P'][i - 1])
                tg = (n['P'][i] - n['P'][i - 1]) / max(np.linalg.norm(n['P'][i] - n['P'][i - 1]), 1e-9)
                vw = vd * tg + 1.0 * (pd - qp[:2])
                c, s_ = math.cos(qp[2]), math.sin(qp[2])
                u[0], u[1] = c * vw[0] + s_ * vw[1], -s_ * vw[0] + c * vw[1]
                dyaw = (yaw_t - qp[2] + math.pi) % (2 * math.pi) - math.pi
                u[2] = float(np.clip(1.5 * dyaw, -W_TURN, W_TURN))
                done_ref = sd >= n['L'] - 1e-6 and np.linalg.norm(n['P'][-1] - qp[:2]) < 0.01
            # 本體速度框與加速度（有界）
        lo, hi = LIM.BODY_V_MIN, LIM.BODY_V_MAX
        u[:3] = np.clip(u[:3], lo, hi)
        da = LIM.A_BODY * DT
        u[:3] = np.clip(u[:3], self.u_body_prev - da, self.u_body_prev + da)
        return u, done_ref

    def _check_cmd(self, u):
        bad = []
        ar = float(np.abs(u[3:]).max())
        aa = float(np.abs(u[3:] - self.u_prev[3:]).max()) / DT
        wv = float(np.abs(LIM.WM @ u[:3]).max())
        wa = float(np.abs(LIM.WM @ (u[:3] - self.u_prev[:3])).max()) / DT
        self.lim['arm_rate'], self.lim['arm_acc'] = max(self.lim['arm_rate'], ar), max(self.lim['arm_acc'], aa)
        self.lim['wheel_v'], self.lim['wheel_a'] = max(self.lim['wheel_v'], wv), max(self.lim['wheel_a'], wa)
        self.lim['body_v'] = [max(a_, abs(b_)) for a_, b_ in zip(self.lim['body_v'], u[:3])]
        self.lim['body_a'] = [max(a_, abs(b_) / DT) for a_, b_ in zip(self.lim['body_a'], u[:3] - self.u_prev[:3])]
        if ar > self.va + 1e-9:
            bad.append(dict(what='arm_rate', v=round(ar, 4)))
        if aa > C0.A_ARM + 1e-9:
            bad.append(dict(what='arm_acc', v=round(aa, 3)))
        if wv > LIM.W_LIM + 1e-9:
            bad.append(dict(what='wheel_v', v=round(wv, 4)))
        if wa > LIM.WA_LIM + 1e-9:
            bad.append(dict(what='wheel_a', v=round(wa, 3)))
        return bad

    def _sweep(self, q, u):
        """候選提交前：提交 u 後「現在停（extra 0）／晚一拍停（extra 1）」，門角取帶兩端。"""
        for e in STOP_EXTRAS:
            Q = XO.stop_traj_ramp(q, self.s, self.pipe, u, self._ramp_end(), 0, self.cfg, e)
            idx = list(range(0, len(Q), self.sweep_stride)) + [len(Q) - 1]
            for i in idx:
                lo, hi = self.band(self.t - self.t_x1 + i * XO.C21.SAMPLE_EVERY * PHYS)   # 門角帶取該取樣點的時刻
                c = self.clear(Q[i], (lo, hi))
                self.mins['stop_sweep'] = min(self.mins['stop_sweep'], min(c[k] - W.THR[k] for k in KEYS))
                b_ = [k for k in KEYS if c[k] < W.THR[k]]
                if (not self.finger_exempt) and c['finger_bar'] < W.THR['env']:
                    b_.append('finger_bar')
                if b_:
                    return dict(extra=e, i=i, bad=b_)
        return None

    def _ramp_end(self):
        return np.asarray(self.pipe[-1][:3], float).copy()

    def step(self, t, q, phi_meas):
        self.t = float(t)
        q = np.asarray(q, float)
        tick = self.n_step % NPH == 0
        if tick and self.off0 is None:
            self.off0, self.G0 = q[3:] - self.s, BM.arm_gravity(K, q, self.bodies)
            self.anchor = dict(q0=q.tolist(), s0=self.s.tolist(), off0=self.off0.tolist(), joint_kp=self.joint_kp)
            self.Tn0 = K.fk(q, 'link_tcp')
            self.Th0 = W.handle_pose(self.T_WH_bind, self.phi0)
            self.t_x1 = self.t
        if tick:
            if self.q_tick is not None:
                self.rate = dict(arm=float(np.abs(q[3:] - self.q_tick[3:]).max()) / DT, base=float(np.linalg.norm(q[:2] - self.q_tick[:2])) / DT, yaw=abs(q[2] - self.q_tick[2]) / DT)
            self.q_tick = q.copy()
            # 監看：門角帶、實測淨空
            lo, hi = self.band(self.t - self.t_x1)
            if not (lo <= phi_meas <= hi):
                self._stop('door_out_of_band', phi_deg=round(math.degrees(phi_meas), 4), band_deg=[round(math.degrees(lo), 4), round(math.degrees(hi), 4)])
            c = self.clear(q, (phi_meas,))
            for k in KEYS:
                self.mins[k] = min(self.mins[k], c[k])
            if self.finger_exempt:
                if self.finger_gap_prev is not None and c['finger_bar'] < self.finger_gap_prev - 1e-4:
                    self._stop('finger_gap_worsening', gap_mm=round(c['finger_bar'] * 1e3, 3))
                self.finger_gap_prev = c['finger_bar']
                if c['finger_bar'] >= W.THR['env']:
                    self.finger_exempt, self.finger_release_t = False, self.t
                    self._ev('finger_exemption_released', gap_mm=round(c['finger_bar'] * 1e3, 3))
            else:
                self.mins['finger_bar_after_release'] = min(self.mins['finger_bar_after_release'], c['finger_bar'])
            b_ = self._bad(c)
            if b_:
                self._stop('clearance', bad=b_, at={k: round(float(v), 4) for k, v in c.items()})
            self.rec.append([self.t, KEYS.index('env') * 0 + c['env'], c['base'], c['self'], c['shell_bar'], c['finger_bar'], phi_meas, lo, hi, ['X1', 'X1b', 'X2', 'X3X4', 'STOP'].index(self.phase if self.stop is None else 'STOP')])
            # 段別推進
            if self.stop is None:
                slow = self.rate['arm'] <= SETTLE['arm_rps'] and self.rate['base'] <= SETTLE['base_mps'] and self.rate['yaw'] <= SETTLE['yaw_rps']
                if getattr(self, '_ref_done', False) and slow and not any(np.any(u_) for u_ in self.pipe) and not np.any(self.u_last):
                    self.settle_n += 1
                else:
                    self.settle_n = 0
                if self.settle_n >= SETTLE['n_ticks']:
                    nxt = dict(X1='X1b', X1b='X2', X2='X3X4', X3X4='DONE')[self.phase]
                    self._ev('phase_done', phase=self.phase)
                    if nxt == 'DONE':
                        self.done = True
                        self._stop('exit_pass_done')
                    else:
                        self.phase, self.t_phase, self._ref_done, self.settle_n = nxt, self.t, False, 0
                        self.ph_hist.append((nxt, round(self.t, 3)))
                        if nxt == 'X2':
                            self.q_ref0 = q[3:].copy()
                        if nxt == 'X3X4':
                            P = np.vstack([q[:2], self.path])
                            cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
                            self.nav = dict(mode='turn', P=P, cum=cum, L=float(cum[-1]), t0=None)
            # 命令
            if self.stop is None:
                qp, sp = self._pred(q)
                u, info = self._cmd(q, qp, sp)
                if u is None:
                    self._stop(info)
                else:
                    if not self._ref_done and bool(info):
                        self._ref_done = True
                    if self._ref_done:                                   # 參考走完：命令歸零（停穩後換段）
                        u = np.zeros(9)
                    bad = self._check_cmd(u)
                    if bad:
                        self._stop('cmd_limit', bad=bad)
                    else:
                        sw = self._sweep(q, u)
                        if sw is not None:
                            self._stop('candidate_stop_violation', **sw)
                        else:
                            self.u_last = u
            if self.stop is not None:                                    # 受控停止：底盤有界減速、手臂設定點凍結
                da = np.r_[LIM.A_BODY * DT, np.full(6, np.inf)]
                u = self.u_last.copy()
                u[3:] = 0.0
                u[:3] = u[:3] + np.clip(-u[:3], -da[:3], da[:3])
                self.u_last = u
                if self.stop_ok is None and self.rate['arm'] <= SETTLE['arm_rps'] and self.rate['base'] <= SETTLE['base_mps'] and self.rate['yaw'] <= SETTLE['yaw_rps'] and not any(np.any(u_) for u_ in self.pipe) and not np.any(u):
                    self.stop_ok = dict(t=round(self.t, 3), since_stop_s=round(self.t - self.stop_t, 3), ok=True)
            self.u_prev = self.u_last.copy()
            self.u_body_prev = self.u_last[:3].copy()
            self.u_from_base, self.ramp_j = self._ramp_end(), 0
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
        return dict(arm_target=self.s.copy(), base_twist=ua[:3].copy(), stopped=self.stop is not None, done=self.done)

    def report(self):
        Rr = np.array(self.rec, float) if self.rec else np.zeros((1, 10))
        return dict(stop=self.stop, done=self.done, stop_effect=self.stop_ok, phases=self.ph_hist, events=self.events, anchor=self.anchor,
                    min_clear={k: round(float(v), 4) for k, v in self.mins.items()}, finger_release_t=self.finger_release_t, limits={k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in self.lim.items()},
                    door_band_end_deg=[round(math.degrees(v), 4) for v in self.band(self.t - getattr(self, 't_x1', 0.0))], door_meas_end_deg=round(math.degrees(float(Rr[-1, 6])), 4) if len(Rr) else None, t_end=round(self.t, 3))
