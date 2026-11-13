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
import time
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
STOP_EXTRAS = (1, 2)                                                 # 候選提交後：下一拍停／再晚一拍停（現在停另在求解前以實際管線與斜坡相位核）
STOP_STILL_S = 0.3                                                   # 停止有效：連續靜止且管線清空 0.3 s
WAIT = dict(omega_dps=0.02, hold_s=0.3, win_s=0.3, max_s=10.0)      # r4 條件式等待（Codex 20261113_090540）：門角速度（0.3 s 窗最小平方斜率）連續 0.3 s ≤0.02°/s 才放行 X1；上限 10 s，逾時拒絕退出
GAP_TOL = 1e-4                                                       # 兩指間隙不惡化容差（對「至今最大間隙」比，不逐拍累積）


def stop_traj_exit(q, s, pipe, u_last, u_from_base, j0, cfg, extra_ticks=0):
    """與 door_open_exec_ctrl.stop_traj_ramp 同一停法（每控制拍有界減速：底盤 A_BASE、手臂 A_ARM；底盤分量拍內逐物理步斜坡；真管線），
    另回 (取樣構型, 逐物理步送進管線的命令, 第一次減速的物理步序號)，供時序核對。j0＝目前控制拍內已過的物理步數（拍首＝NPH ⇒ 立即進入新一拍）。"""
    qp, sp = np.asarray(q, float).copy(), np.asarray(s, float).copy()
    pp = [np.asarray(u, float).copy() for u in pipe]
    u, uf, j = np.asarray(u_last, float).copy(), np.asarray(u_from_base, float).copy(), int(j0)
    da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * cfg.dt
    out, cmds, k, quiet, k_dec = [qp.copy()], [], 0, 0, None
    while quiet < XO.C21.SETTLE_STEPS and k < 2000:
        if j >= NPH:
            uf, j = XO.ramp_cmd(uf, u, NPH - 1)[:3], 0
            if k >= extra_ticks * NPH:
                if k_dec is None:
                    k_dec = k
                u = u + np.clip(-u, -da, da)
        cmd = XO.ramp_cmd(uf, u, j)
        pp.append(cmd)
        cmds.append(cmd.copy())
        qp, sp = S.plant_phys_step(qp, sp, pp.pop(0), cfg, 1)
        k += 1
        j += 1
        quiet = quiet + 1 if (not np.any(cmd) and not any(np.any(x) for x in pp)) else 0
        if k % XO.C21.SAMPLE_EVERY == 0:
            out.append(qp.copy())
    return np.array(out), cmds, k_dec


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
    def __init__(self, ident, T_WH_bind, phi_bind, s, pipe, u_last, u_prev, bias, finger_q, phi0, omega0, path_xy, q_stow, joint_kp=600.0, sweep_stride=3, scope_end=None, wait_enable=False, anchor_pred_bias=True, pool=0):
        self._pool = None                                            # r5（G3）：停止掃掠淨空分給預建工作程序；0＝r4
        if int(pool) > 0:
            import door_sweep_pool as DSP
            self._pool = DSP.SweepPool(int(pool))
        self.pool_ms = []
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
        self.wait_enable = bool(wait_enable)                         # r4：W 段預設停用（Codex 20261113_091401；等待策略在兩個實錄起點未過現行檢查）
        self.anchor_pred_bias = bool(anchor_pred_bias)               # 預測模型偏置以錨定時實測 q−s 初始化（局部預測模型初始化變更；門檻不變）
        self.bias_inherited = np.asarray(bias, float).copy()
        self.phase, self.t_phase, self.ph_hist = ('W', 0.0, [('W', 0.0)]) if self.wait_enable else ('X1', 0.0, [('X1', 0.0)])
        self.phi_hist, self.wait_ok_n, self.wait_info = [], 0, None
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
        self.abort, self.finger_gap_max, self.phi_tick, self.still_n, self.sweep_timing, self.stop_sev, self.stop_kinds = None, None, None, 0, [], None, set()
        self.inject = None                                           # 測試用（外部停止等）
        self.scope_end = scope_end                                   # 'X2'：X2 完成即在控制器內範圍停止，不進 X3X4、不產生任何 X3X4 命令（Codex 20261113_020612）
        self.cmd_log = []                                            # 每控制拍（t, 段, 提交命令 9）——驗證無 X3X4 命令

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

    def close_pool(self):
        if self._pool is not None:
            self._pool.close()

    def warmup(self, q, phi_meas):
        """r6：交接前預熱（複本空跑一拍、本控制器狀態不動；同開門執行層 r10）。回複本那一拍耗時 ms。"""
        import copy
        pool, self._pool = self._pool, None
        try:
            tmp = copy.deepcopy(self)
        finally:
            self._pool = pool
        tmp._pool = None
        fq_save, phi0_save = W.FINGER_Q, W.PHI0
        t0 = time.perf_counter()
        try:
            tmp.step(0.0, np.asarray(q, float).copy(), float(phi_meas))
        finally:
            W.FINGER_Q, W.PHI0 = fq_save, phi0_save
        return (time.perf_counter() - t0) * 1e3

    def _sweep_pool(self, q, u, extras, j0, u_from, tag):
        """r5：同 _sweep 的樣本、門角帶與判定順序；淨空與兩指—橫桿距離由工作程序算（同函式）。逾時／程序失敗／缺結果 ⇒ 模擬中止，不當通過。"""
        items, X, tim = [], [], {}
        for e in extras:
            Q, cmds, k_dec = stop_traj_exit(q, self.s, self.pipe, u, u_from, j0, self.cfg, e)
            tim[e] = dict(t=round(self.t, 3), tag=tag, extra=e, j0=int(j0), k_decel=k_dec, n_cmd=len(cmds))
            idx = list(range(0, len(Q), self.sweep_stride)) + [len(Q) - 1]
            for i in idx:
                lo, hi = self.band(self.t - self.t_x1 + i * XO.C21.SAMPLE_EVERY * PHYS)
                items.append((e, i, len(X)))
                X.append(np.r_[Q[i], lo])
                X.append(np.r_[Q[i], hi])
        h = self._pool.submit(self.T_WH_bind, W.FINGER_Q, W.PHI0, np.array(X), finger_bar_q=self.finger_q)
        res, why, tms = self._pool.collect(h, XO.POOL_TIMEOUT_S)
        self.pool_ms.append(tms)
        if res is None:
            self._abort('sweep_unavailable', pool_why=why, tag=tag)
            return dict(tag=tag, pool_unavailable=why)
        seen = set()
        for e, i, j in items:
            if e not in seen:
                seen.add(e)
                self.sweep_timing.append(tim[e])
            c = {k: np.inf for k in KEYS}
            c['finger_bar'] = np.inf
            for r in (res[j], res[j + 1]):
                for k in KEYS:
                    c[k] = min(c[k], float(r[k]))
                c['finger_bar'] = min(c['finger_bar'], r['finger_bar'])
            self.mins['stop_sweep'] = min(self.mins['stop_sweep'], min(c[k] - W.THR[k] for k in KEYS))
            b_ = [k for k in KEYS if c[k] < W.THR[k]]
            if c['finger_bar'] < (0.0 if self.finger_exempt else W.THR['env']):
                b_.append('finger_bar')
            if b_:
                return dict(tag=tag, extra=e, i=i, bad=b_)
        return None

    def _sweep(self, q, u, extras, j0, u_from, tag):
        if self._pool is not None:
            return self._sweep_pool(q, u, extras, j0, u_from, tag)
        """停止掃掠：從目前管線與斜坡相位，依序執行 u（extras 拍後開始有界減速）直到停穩；每取樣點核淨空（門角取帶在該時刻的兩端）
        與兩指規則（例外中只要求不穿入，撤除後要求門檻）。回 None 或違反資訊；並記錄減速起點（時序核對）。"""
        for e in extras:
            Q, cmds, k_dec = stop_traj_exit(q, self.s, self.pipe, u, u_from, j0, self.cfg, e)
            self.sweep_timing.append(dict(t=round(self.t, 3), tag=tag, extra=e, j0=int(j0), k_decel=k_dec, n_cmd=len(cmds)))
            idx = list(range(0, len(Q), self.sweep_stride)) + [len(Q) - 1]
            for i in idx:
                lo, hi = self.band(self.t - self.t_x1 + i * XO.C21.SAMPLE_EVERY * PHYS)   # 門角帶取該取樣點的時刻
                c = self.clear(Q[i], (lo, hi))
                self.mins['stop_sweep'] = min(self.mins['stop_sweep'], min(c[k] - W.THR[k] for k in KEYS))
                b_ = [k for k in KEYS if c[k] < W.THR[k]]
                if c['finger_bar'] < (0.0 if self.finger_exempt else W.THR['env']):
                    b_.append('finger_bar')
                if b_:
                    return dict(tag=tag, extra=e, i=i, bad=b_)
        return None

    def _ramp_end(self):
        return np.asarray(self.pipe[-1][:3], float).copy()

    def _abort(self, why, **kw):
        if self.abort is None:
            self.abort = dict(t=round(self.t, 3), why=why, phase=self.phase, **kw)
            self._ev('abort', why=why, **kw)
            self.stop_ok = dict(t=round(self.t, 3), ok=None, status='unverified（模擬中止）')

    def _severity(self, c):
        """停止中比較用的嚴重度（越小越糟）：兩指間隙與各淨空餘裕的最小值。"""
        return min(c['finger_bar'], min(c[k] - W.THR[k] for k in KEYS))

    def _anchor(self, q, phi, omega):
        """以當下實測狀態建立退出參考：偏置錨定 off0＝q−s、重力基準、TCP 與把手基準、門角帶起點；
        停止預測模型的偏置也改用同一實測值（bias＝α·off0；自由空間辨識偏置在此構型不成立——full_1 預測穩態與實測差 5.3 mrad）。"""
        al_ = np.asarray(self.cfg.arm_model.alpha, float)
        pre = dict(pipe_empty=bool(not any(np.any(u_) for u_ in self.pipe)), u_last_zero=bool(not np.any(self.u_last)))   # 取錨前置：管線清空、無命令（靜止與無鎖存由執行器在放行前確認）
        self.off0, self.G0 = q[3:] - self.s, BM.arm_gravity(K, q, self.bodies)
        if self.anchor_pred_bias:
            self.cfg.arm_model.bias = al_ * self.off0                 # 只在取錨時設一次；之後不因掃掠失敗再調
        self.phi0, self.omega0, self.t_x1 = float(phi), float(omega), self.t
        self.anchor = dict(t=round(self.t, 3), q0=q.tolist(), s0=self.s.tolist(), off0=self.off0.tolist(), joint_kp=self.joint_kp, phi0_deg=round(math.degrees(phi), 4), omega0_dps=round(math.degrees(omega), 4), precondition=pre,
                           pred_bias_inherited=self.bias_inherited.tolist(), pred_bias_used=np.asarray(self.cfg.arm_model.bias, float).tolist(), equiv_offset_inherited_mrad=[round(float(v) * 1e3, 3) for v in self.bias_inherited / al_],
                           equiv_offset_used_mrad=[round(float(v) * 1e3, 3) for v in np.asarray(self.cfg.arm_model.bias, float) / al_], anchored_pred_bias=self.anchor_pred_bias)
        if not all(pre.values()):
            self._stop('anchor_precondition', **pre)
        self.Tn0 = K.fk(q, 'link_tcp')
        self.Th0 = W.handle_pose(self.T_WH_bind, self.phi0)

    def _omega_est(self):
        """門角速度：最近 WAIT.win_s 內每控制拍門角的最小平方斜率（不足窗長回 None）。"""
        n = int(round(WAIT['win_s'] / DT)) + 1
        if len(self.phi_hist) < n:
            return None
        return float(np.polyfit(np.arange(n) * DT, np.array(self.phi_hist[-n:]), 1)[0])

    def _monitor_finger(self, gap, gap_band=None):
        """兩指—橫桿：例外只免 5 mm 餘裕、不免非穿透；不惡化以「至今最大間隙」為基準（不逐拍累積）；
        撤除例外要求量測門角與門角帶兩端的間隙都達門檻（與停止掃掠同一判準），撤除後永久。回違反原因或 None。"""
        if self.finger_exempt:
            if gap < 0.0:
                return 'finger_penetration'
            if self.finger_gap_max is not None and gap < self.finger_gap_max - GAP_TOL:
                return 'finger_gap_worsening'
            self.finger_gap_max = gap if self.finger_gap_max is None else max(self.finger_gap_max, gap)
            if min(gap, gap if gap_band is None else gap_band) >= W.THR['env']:
                self.finger_exempt, self.finger_release_t = False, self.t
                self._ev('finger_exemption_released', gap_mm=round(gap * 1e3, 3))
            return None
        self.mins['finger_bar_after_release'] = min(self.mins['finger_bar_after_release'], gap)
        return 'finger_bar' if gap < W.THR['env'] else None

    def step(self, t, q, phi_meas):
        self.t = float(t)
        q = np.asarray(q, float)
        if self.abort is not None:                                   # 模擬中止：不再產生命令
            return dict(arm_target=self.s.copy(), base_twist=np.zeros(3), stopped=True, done=False, abort=True)
        tick = self.n_step % NPH == 0
        if tick and self.off0 is None:
            self._anchor(q, self.phi0, self.omega0)
        if tick:
            if self.q_tick is not None:
                self.rate = dict(arm=float(np.abs(q[3:] - self.q_tick[3:]).max()) / DT, base=float(np.linalg.norm(q[:2] - self.q_tick[:2])) / DT, yaw=abs(q[2] - self.q_tick[2]) / DT)
            self.q_tick = q.copy()
            # 監看：門角帶、實測淨空、兩指規則（停止期間照樣；停止中再違反 ⇒ 模擬中止）
            viol = []
            lo, hi = self.band(self.t - self.t_x1)
            out_band = not (lo <= phi_meas <= hi)
            c = self.clear(q, (phi_meas,))
            for k in KEYS:
                self.mins[k] = min(self.mins[k], c[k])
            gb = min(self.finger_bar(q, lo), self.finger_bar(q, hi)) if self.finger_exempt else None
            fv = self._monitor_finger(c['finger_bar'], gb)
            if fv:
                viol.append((fv, dict(gap_mm=round(c['finger_bar'] * 1e3, 3))))
            b_ = [k for k in KEYS if c[k] < W.THR[k]]
            if b_:
                viol.append(('clearance', dict(bad=b_, at={k: round(float(v), 4) for k, v in c.items()})))
            if out_band:
                # 舊帶失效：以當下狀態重建帶（門角、量測角速度），並以新帶核「現在停」；不成立 ⇒ 中止
                om = 0.0 if self.phi_tick is None else (phi_meas - self.phi_tick) / DT
                old = [round(math.degrees(lo), 4), round(math.degrees(hi), 4)]
                self.phi0, self.omega0, self.t_x1 = float(phi_meas), float(om), self.t
                sw = self._sweep(q, self.u_last, (0,), self.ramp_j, self.u_from_base, 'rebanded_stop_now')
                if sw is not None:
                    self._abort('door_out_of_band_unverifiable', phi_deg=round(math.degrees(phi_meas), 4), old_band_deg=old, sweep=sw)
                else:
                    viol.append(('door_out_of_band', dict(phi_deg=round(math.degrees(phi_meas), 4), old_band_deg=old, new_band_origin_deg=round(math.degrees(phi_meas), 4), omega_dps=round(math.degrees(om), 4))))
            self.phi_tick = float(phi_meas)
            self.phi_hist.append(float(phi_meas))
            if self.stop is not None and viol:                          # 停止中：觸發停止的同一違反若未惡化視為持續狀態；新的或惡化的違反 ⇒ 中止
                sev = self._severity(c)
                viol = [v for v in viol if not (v[0] in self.stop_kinds and self.stop_sev is not None and sev >= self.stop_sev - GAP_TOL)]
            if viol and self.abort is None:
                if self.stop is None:
                    self._stop(viol[0][0], **viol[0][1])
                    self.stop_sev, self.stop_kinds = self._severity(c), {v[0] for v in viol}
                else:
                    self._abort('violation_during_stop', viol=[v[0] for v in viol], stop_why=self.stop['why'])
            if self.abort is not None:
                return dict(arm_target=self.s.copy(), base_twist=np.zeros(3), stopped=True, done=False, abort=True)
            if self.inject is not None:
                self.inject(self)
            if self.phase == 'W' and self.stop is None:                   # 條件式等待：命令為零、監看與現在停掃掠照常；達條件 ⇒ 以當下狀態重新錨定並放行 X1
                om = self._omega_est()
                tw = self.t - self.t_phase
                self.wait_ok_n = self.wait_ok_n + 1 if (om is not None and abs(om) <= math.radians(WAIT['omega_dps'])) else 0
                if self.wait_ok_n * DT >= WAIT['hold_s'] - 1e-9:
                    self._anchor(q, phi_meas, om)
                    self.wait_info = dict(wait_s=round(tw, 3), omega_release_dps=round(math.degrees(om), 5))
                    self._ev('wait_released', **self.wait_info)
                    self.phase, self.t_phase = 'X1', self.t
                    self.ph_hist.append(('X1', round(self.t, 3)))
                elif tw >= WAIT['max_s']:
                    self.wait_info = dict(wait_s=round(tw, 3), omega_dps=None if om is None else round(math.degrees(om), 5), timeout=True)
                    self._stop('wait_timeout_exit_refused', **self.wait_info)
            self.rec.append([self.t, c['env'], c['base'], c['self'], c['shell_bar'], c['finger_bar'], phi_meas, lo, hi, ['W', 'X1', 'X1b', 'X2', 'X3X4', 'STOP'].index(self.phase if self.stop is None else 'STOP')])
            # 段別推進
            if self.stop is None:
                slow = self.rate['arm'] <= SETTLE['arm_rps'] and self.rate['base'] <= SETTLE['base_mps'] and self.rate['yaw'] <= SETTLE['yaw_rps']
                if getattr(self, '_ref_done', False) and slow and not any(np.any(u_) for u_ in self.pipe) and not np.any(self.u_last):
                    self.settle_n += 1
                else:
                    self.settle_n = 0
                if self.settle_n >= SETTLE['n_ticks']:
                    nxt = dict(W='W', X1='X1b', X1b='X2', X2='X3X4', X3X4='DONE')[self.phase]
                    self._ev('phase_done', phase=self.phase)
                    if nxt == 'X3X4' and self.scope_end == 'X2':
                        self._stop('scope_end_after_X2')
                    elif nxt == 'DONE':
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
            # 求解前「現在停」：目前命令、實際管線與斜坡相位
            if self.stop is None:
                sw0 = self._sweep(q, self.u_last, (0,), self.ramp_j, self.u_from_base, 'stop_now')
                if sw0 is not None:
                    self._stop('stop_now_violation', **sw0)
            # 命令
            if self.stop is None and self.phase == 'W':
                self.u_last = np.zeros(9)
            elif self.stop is None:
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
                        sw = self._sweep(q, u, STOP_EXTRAS, 0, self._ramp_end(), 'candidate')
                        if sw is not None:
                            self._stop('candidate_stop_violation', **sw)
                        else:
                            self.u_last = u
            if self.stop is not None:                                    # 受控停止：與預測同一停法（每拍有界減速：底盤 A_BASE、手臂 A_ARM），停止命令也過限制核對
                da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * self.cfg.dt
                u = self.u_last + np.clip(-self.u_last, -da, da)
                badl = self._check_cmd(u)
                if badl:
                    self._abort('stop_cmd_limit', bad=badl)
                    return dict(arm_target=self.s.copy(), base_twist=np.zeros(3), stopped=True, done=False, abort=True)
                self.u_last = u
            self.u_prev = self.u_last.copy()
            self.u_body_prev = self.u_last[:3].copy()
            self.cmd_log.append((round(self.t, 3), self.phase if self.stop is None else 'STOP:' + self.phase, self.u_last.copy()))
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
        if self.stop is not None and self.stop_ok is None and self.abort is None:
            still = (self.rate['arm'] <= SETTLE['arm_rps'] and self.rate['base'] <= SETTLE['base_mps'] and self.rate['yaw'] <= SETTLE['yaw_rps']
                     and not any(np.any(u_) for u_ in self.pipe) and not np.any(self.u_last))
            self.still_n = self.still_n + 1 if still else 0
            if self.still_n * PHYS >= STOP_STILL_S - 1e-9:
                self.stop_ok = dict(t=round(self.t, 3), since_stop_s=round(self.t - self.stop_t, 3), still_s=STOP_STILL_S, ok=True)
        self.n_step += 1
        return dict(arm_target=self.s.copy(), base_twist=ua[:3].copy(), stopped=self.stop is not None, done=self.done, abort=self.abort is not None)

    def report(self):
        Rr = np.array(self.rec, float) if self.rec else np.zeros((1, 10))
        return dict(stop=self.stop, abort=self.abort, done=self.done, stop_effect=self.stop_ok, phases=self.ph_hist, events=self.events, anchor=self.anchor,
                    wait=self.wait_info, sweep_timing_sample=self.sweep_timing[:6], sweep_timing_n=len(self.sweep_timing), scope_end=self.scope_end,
                    x3x4_cmd_ticks=sum(1 for _, ph, u in self.cmd_log if ph == 'X3X4' and np.any(u)), cmd_ticks=len(self.cmd_log),
                    min_clear={k: round(float(v), 4) for k, v in self.mins.items()}, finger_release_t=self.finger_release_t, limits={k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v]) for k, v in self.lim.items()},
                    door_band_end_deg=[round(math.degrees(v), 4) for v in self.band(self.t - getattr(self, 't_x1', 0.0))], door_meas_end_deg=round(math.degrees(float(Rr[-1, 6])), 4) if len(Rr) else None, t_end=round(self.t, 3))
