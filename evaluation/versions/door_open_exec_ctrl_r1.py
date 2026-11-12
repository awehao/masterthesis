#!/usr/bin/env python3
"""門機構約束開門：執行層（Codex 20261112 綁定冒煙 3 裁定的五點）。**機構約束離線規劃＋閉鏈誤差回授追蹤**——不是在線機構約束 MPC。
不沿用任何既有開門控制／時間排程／關節序列。綁定（理想抓持）成立後才啟用。

輸入
  plan      door_wbopt 的密集可執行參考 Xd（底盤 3、手臂 6、門角 1；間隔 h），起點＝綁定瞬間實測狀態。
  T_WH_bind 綁定瞬間的把手位姿、phi_bind 綁定瞬間門角、T_HE 綁定瞬間實測抓持變換（之後固定不變）。
  handover  從插入控制器原樣承接：手臂設定點 s、8 步在途命令管線、上一命令、手臂模型偏置。不重置、不清管線。
每控制拍（0.05 s）
  1 進度調節（明確公式）：參考相位 σ 的前進率 g＝clip((LEAD_MAX −(φ_ref(σ)−φ_實測))/LEAD_BAND, 0, 1)，σ ← σ＋dt·g。
    參考超前實測門角越多就越慢，超前到 LEAD_MAX 時停住等門；不讓參考與機構脫節後硬拉。g＝0 持續 STALL_S ⇒ 停止 door_not_following。
  2 預測視窗（k＝0…N）：σ_k＝σ_生效＋k·dt·g。
      底盤參考 X_ref[k]＝b_ref(σ_k)、本體速度前饋＝g·R(θ_ref)ᵀ·db_ref/dσ；
      預測門角 φ̂_k＝φ_ref(σ_k)；TCP 目標 T_seq[k]＝Rz[−(φ̂_k−φ_bind)]·T_WH_bind·T_HE（世界座標）；
      手臂線性化名目＝參考關節經設定點模型反算（同 jnom；求解器回傳的就是完整命令，不再另加前饋）。
  3 **同一個全身問題**（gmpc_wb.GMPC，底盤不固定＋手臂 active）同時輸出底盤與手臂命令；輪級耦合在求解器內，求解後不再改命令。
  4 提交前核停止路徑（含**轉動中的門**）：對「現在停」（求解前）與「候選命令後下一拍停／再晚一拍停」各預測整段機器人軌跡（真管線、真減速），
    每個取樣點由把手點位置反算當時門角 φ_i，核：預測閉鏈殘差（徑向／高度／姿態；綁定下它代表內應力而非可見誤差）與全身對 φ_i 門扇的淨空（閉合手指開度）。
    違反 ⇒ 候選不提交、受控停止。
  5 監看（鎖存停止）：實測閉鏈殘差、設定點閉鏈殘差（手臂被命令去的位置對機構允許位置＝內應力指標）、實測淨空、進度停滯、求解失敗、外部超力旗標。
停止（r2；兩種，分開）：
  路徑停止（預設：監看觸發、外部要求、完成）——把進度率 g 以 1/T_STOP 的斜率降到 0，期間**繼續用同一個全身求解追蹤參考**；底盤與手臂沿已核過的
    機構相容路徑一起減速，設定點為補償落後而超前的量隨 g 一起收回。g＝0 且各速度低於門檻（或逾時）後才轉入命令歸零。綁定保持。
  緊急停止（求解失敗、狀態非有限、超力、路徑停止中再出錯）——命令逐拍有界減速、在途命令走完。這種停法**不保證閉鏈相容**（底盤立刻停、手臂還會走完超前量；
    名目起步第 2 拍的預測閉鏈偏差 10–18 mm），所以同時設 request_unbind：執行器必須先解除綁定。
  每拍仍預測緊急停止路徑：淨空違反（對轉動中的門）⇒ 候選命令不提交；預測閉鏈偏差只記錄（它就是緊急停止要先解綁的理由）。
停止有效性分核：管線清空、底盤、手臂、門角速度。
完成：σ 到參考終點、實測門角 ≥ φ_goal−END_PHI_TOL、各速度低於門檻連續 END_N 拍。
數值皆為事前登錄的工程設定；未校準的傳遞力只作模擬保護。"""
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault('M8_SCENE_REG', '20261112_m8_house_scene.json')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_wbopt as W                                             # noqa: E402

DA = W.DA
GW, K, C0, B, C21, S = DA.GW, DA.K, DA.C0, DA.B, DA.C21, DA.S
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE               # noqa: E402
from ammr_wholebody_mpc.se2 import from_xytheta                    # noqa: E402

PHYS, NPH, D_STEPS, N, DT = C0.PHYS, C0.NPH, C0.D_STEPS, B.N, B.DT
LEAD_MAX, LEAD_BAND, STALL_S = math.radians(3.0), math.radians(1.0), 2.0
CL_TOL = dict(pos_m=0.010, rot_deg=5.0)                            # 實測／預測閉鏈殘差上限
STRESS_TOL = dict(pos_m=0.030, rot_deg=8.0)                        # 設定點閉鏈殘差上限（內應力指標）
END = dict(phi_tol=math.radians(0.5), n_ticks=10, wait_s=4.0, arm_rps=0.02, base_mps=0.005, yaw_rps=0.01, door_rps=0.01)
T_STOP, PATH_STOP_MAX_S = 0.5, 3.0                                # 路徑停止：g 由 1 降到 0 的時間、最長等待
EMERGENCY = ('solver_fail', 'state_nonfinite', 'overforce', 'setpoint_limit', 'path_stop_timeout')
STOP_STILL = dict(arm_rps=2e-3, base_mps=0.003, yaw_rps=0.003, door_rps=0.005, hold_s=0.3, timeout_s=4.0)
KEYS = ('env', 'base', 'self', 'shell_bar')


def rot_deg(Ta, Tb):
    Re = Ta[:3, :3].T @ Tb[:3, :3]
    v = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    return math.degrees(math.atan2(float(np.linalg.norm(v)), 0.5 * (float(np.trace(Re)) - 1.0)))


class DoorOpenExec:
    def __init__(self, ident, Xd, h, T_WH_bind, phi_bind, T_HE, phi_goal, handover, finger_q, sweep_stride=2):
        self.cfg = C0.make_cfg(ident)
        self.cfg.base_fixed = False
        self.cfg.arm_model.bias = np.asarray(handover['bias'], float).copy()
        self.mpc = GW.GMPC(B.nav_cfg())
        self.Xd, self.h = np.asarray(Xd, float), float(h)
        self.T_WH_bind, self.phi_bind, self.T_HE, self.phi_goal = np.asarray(T_WH_bind, float), float(phi_bind), np.asarray(T_HE, float), float(phi_goal)
        W.set_model(phi0=self.phi_bind, finger_q=finger_q)
        self.T_EH = np.linalg.inv(self.T_HE)
        pH = self.T_WH_bind[:3, 3]
        self.ang_bind, self.r_handle, self.z_handle = math.atan2(pH[1], pH[0]), float(np.hypot(pH[0], pH[1])), float(pH[2])
        self.s = np.asarray(handover['s'], float).copy()
        self.pipe = [np.asarray(u, float).copy() for u in handover['pipe']]
        assert len(self.pipe) == D_STEPS
        self.u_last, self.u_prev = np.asarray(handover['u_last'], float).copy(), np.asarray(handover['u_prev'], float).copy()
        self.lim_lo, self.lim_hi = np.asarray(LITE6_SAFE.lower, float) + 0.05, np.asarray(LITE6_SAFE.upper, float) - 0.05
        self.sig_end = (len(self.Xd) - 1) * self.h
        self.sigma, self.g, self.t, self.n_step = 0.0, 0.0, 0.0, 0
        self.stop, self.done, self.stopped_t, self.stop_ok, self._still = None, False, None, None, 0
        self.events, self.rec, self.fail_inject = [], [], None
        self.solve_ms, self.sweep_ms, self.tick_ms = [], [], []
        self.q_tick, self.phi_tick, self.rate = None, None, dict(base=0.0, yaw=0.0, arm=0.0, door=0.0)
        self.q_last, self.phi_last, self._stall, self._n_end, self.t_ref_end = None, None, 0.0, 0, None
        self.sweep_stride = int(sweep_stride)
        self.mins = dict(act={k: np.inf for k in KEYS}, stop={k: np.inf for k in KEYS}, n_sweep=0, stop_closure_pos=0.0, stop_closure_rot=0.0)
        self.overforce = None
        self.stop_mode, self.request_unbind, self.t_path_stop, self.n_emerg_closure = None, False, None, 0
        self._ev('start', sigma_end=round(self.sig_end, 2), phi_bind_deg=round(math.degrees(self.phi_bind), 3), pipe_nonzero=int(sum(bool(np.any(u)) for u in self.pipe)))

    def _ev(self, what, **kw):
        self.events.append(dict(t=round(float(self.t), 3), what=what, **kw))

    def _stop(self, why, **kw):
        if self.stop is None:
            self.stop = dict(t=round(float(self.t), 3), why=why, sigma=round(self.sigma, 3), **kw)
            self.stop_mode = 'cmd' if why in EMERGENCY else 'path'
            self.request_unbind = self.stop_mode == 'cmd'
            self.stop['mode'] = self.stop_mode
            self.t_path_stop = self.t
            self._ev('stop', why=why, mode=self.stop_mode, **kw)
        elif self.stop_mode == 'path' and why in EMERGENCY:          # 路徑停止途中再出錯 ⇒ 升級為緊急停止
            self.stop_mode, self.request_unbind = 'cmd', True
            self.stop['escalated'] = why
            self._ev('stop_escalated', why=why)

    # ---- 參考
    def ref(self, sig):
        x = min(max(sig, 0.0), self.sig_end) / self.h
        i = min(int(x), len(self.Xd) - 2)
        a = x - i
        return (1 - a) * self.Xd[i] + a * self.Xd[i + 1]

    def dref(self, sig):
        if sig <= 0.0 or sig >= self.sig_end:
            return np.zeros(10)
        i = min(int(sig / self.h), len(self.Xd) - 2)
        return (self.Xd[i + 1] - self.Xd[i]) / self.h

    def handle_T(self, phi):
        return W.handle_pose(self.T_WH_bind, phi)

    def phi_of(self, q9):
        """由機器人構型反算門角（把手點繞鉸鏈的方位角）；回 (φ, 把手點)。"""
        pH = (K.fk(q9, 'link_tcp') @ self.T_EH)[:3, 3]
        d = math.atan2(pH[1], pH[0]) - self.ang_bind
        d = (d + math.pi) % (2 * math.pi) - math.pi
        return self.phi_bind - d, pH

    def closure_at(self, q9, phi):
        """構型 q9 對門角 φ 的閉鏈殘差：位置（m）、姿態（°）。"""
        Tc, Td = K.fk(q9, 'link_tcp'), self.handle_T(phi) @ self.T_HE
        return float(np.linalg.norm(Tc[:3, 3] - Td[:3, 3])), rot_deg(Td, Tc)

    def model_q(self, q):
        """設定點在模型下的穩態構型（s＋bias/α）：手臂「被命令去的位置」。與實測門角下的把手比＝內應力指標（r1 直接用 s，把靜態偏置誤當應力）。"""
        return np.r_[q[:3], self.s + np.asarray(self.cfg.arm_model.bias, float) / np.asarray(self.cfg.arm_model.alpha, float)]

    # ---- 停止路徑（含轉動的門）
    def _sweep(self, q, u, extras, tag):
        for e in extras:
            Q = C21.stop_traj(np.asarray(q, float), self.s, self.pipe, u, self.cfg, e)
            idx = list(range(0, len(Q), self.sweep_stride))
            if idx[-1] != len(Q) - 1:
                idx.append(len(Q) - 1)
            for i in idx:
                if not np.isfinite(Q[i]).all():
                    return dict(extra=e, i=i, nonfinite=True)
                phi_i, _ = self.phi_of(Q[i])
                ep, er = self.closure_at(Q[i], phi_i)
                self.mins['stop_closure_pos'], self.mins['stop_closure_rot'] = max(self.mins['stop_closure_pos'], ep), max(self.mins['stop_closure_rot'], er)
                c = W.clear_door(np.r_[Q[i], phi_i], self.T_WH_bind)
                self.mins['n_sweep'] += 1
                for k in KEYS:
                    self.mins['stop'][k] = min(self.mins['stop'][k], c[k])
                bad = [k for k in KEYS if c[k] < W.THR[k]]
                if ep > CL_TOL['pos_m'] or er > CL_TOL['rot_deg']:
                    self.n_emerg_closure += 1                         # 只記錄：緊急停止路徑的閉鏈不相容（⇒ 緊急停止要先解綁）
                if bad or not (-0.02 <= phi_i <= W.PHI_MAX + 0.02):
                    return dict(extra=e, i=i, tag=tag, clear_bad=bad, closure_pos_mm=round(ep * 1e3, 3), closure_rot_deg=round(er, 3), phi_deg=round(math.degrees(phi_i), 2), at={k: round(float(c[k]), 4) for k in KEYS})
        return None

    # ---- 求解（同一個全身問題）
    def _solve(self, q, phi_meas):
        w0 = time.perf_counter()
        cfg = self.cfg
        qp, sp = q.copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, cfg, 1)
        g = self.g
        sig_eff = self.sigma + D_STEPS * PHYS * g
        X, xi, Ts, qr = [], [], [], []
        for k in range(N + 1):
            sk = sig_eff + k * DT * g
            x, dx = self.ref(sk), self.dref(sk)
            c, s_ = math.cos(x[2]), math.sin(x[2])
            X.append(from_xytheta(float(x[0]), float(x[1]), float(x[2])))
            xi.append([g * (c * dx[0] + s_ * dx[1]), g * (-s_ * dx[0] + c * dx[1]), g * dx[2]])
            Ts.append(self.handle_T(float(x[9])) @ self.T_HE)        # 預測門角 φ̂_k＝φ_ref(σ_k)
            qr.append((x[3:9], g * dx[3:9]))
        if self.fail_inject is not None and self.fail_inject(self.t):
            return None
        al = np.asarray(cfg.arm_model.alpha, float)
        tau, off = PHYS / al, np.asarray(cfg.arm_model.bias, float) / al
        va, sk_, u_nom = float(B.W_ARM['v_arm']), sp.copy(), []
        for k in range(N):
            s_ref = qr[k + 1][0] + tau * qr[k + 1][1] - off
            u = np.clip((s_ref - sk_) / DT, -va, va)
            u_nom.append(u)
            sk_ = sk_ + DT * u
        arm = dict(u_arm_nom=np.array(u_nom), mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=cfg.arm_model, T_seq=np.array(Ts), w_p=C0.W_NEAR,
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=False, qp_solver='gi', **B.W_ARM)
        r = self.mpc.solve(X_now=from_xytheta(*qp[:3]), X_ref_win=np.array(X), xi_ref_win=np.array(xi), xi_prev=self.u_prev[:3], arm=arm, obstacles=None)
        self.solve_ms.append((time.perf_counter() - w0) * 1e3)
        ck = dict(r.arm_check or {})
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None:
            return None
        u = np.r_[np.asarray(r.u_opt, float), r.u_arm_seq[0]]
        return u if np.isfinite(u).all() else None

    def _decel(self):
        da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * self.cfg.dt
        return self.u_last + np.clip(-self.u_last, -da, da)

    def step(self, t, q, phi_meas):
        """每物理步。q＝實測 (底盤 3＋手臂 6)、phi_meas＝實測門角。回 dict(arm_target, base_twist, stopped, done)。"""
        self.t = float(t)
        q = np.asarray(q, float)
        tick = self.n_step % NPH == 0
        w_t = time.perf_counter()
        if not (np.isfinite(q).all() and math.isfinite(phi_meas)):
            self._stop('state_nonfinite')
        if tick:
            if self.q_tick is not None:
                self.rate = dict(base=float(np.linalg.norm(q[:2] - self.q_tick[:2])) / self.cfg.dt, yaw=abs(q[2] - self.q_tick[2]) / self.cfg.dt, arm=float(np.abs(q[3:] - self.q_tick[3:]).max()) / self.cfg.dt, door=abs(phi_meas - self.phi_tick) / self.cfg.dt)
            self.q_tick, self.phi_tick = q.copy(), float(phi_meas)
        if tick and self.stop is not None and self.stop_mode == 'path':   # 路徑停止期間仍記錄（閉鏈殘差、淨空）
            x_ref = self.ref(self.sigma)
            ep, er = self.closure_at(q, phi_meas)
            sp_, sr_ = self.closure_at(self.model_q(q), phi_meas)
            c = W.clear_door(np.r_[q, phi_meas], self.T_WH_bind)
            for k in KEYS:
                self.mins['act'][k] = min(self.mins['act'][k], c[k])
            Tm, Tr = K.fk(q, 'link_tcp'), K.fk(x_ref[:9], 'link_tcp')
            self.rec.append([self.t, self.sigma, self.g, phi_meas, x_ref[9], ep, er, sp_, sr_, float(np.linalg.norm(q[:2] - x_ref[:2])), abs(q[2] - x_ref[2]), float(np.linalg.norm(Tm[:3, 3] - Tr[:3, 3])), *[c[k] for k in KEYS],
                             float(np.abs(self.u_last[3:]).max()), float(np.hypot(self.u_last[0], self.u_last[1]))])
        if tick and self.stop is None:
            x_ref = self.ref(self.sigma)
            lead = float(x_ref[9] - phi_meas)
            self.g = float(np.clip((LEAD_MAX - lead) / LEAD_BAND, 0.0, 1.0)) if self.sigma < self.sig_end else 0.0
            ep, er = self.closure_at(q, phi_meas)
            sp_, sr_ = self.closure_at(self.model_q(q), phi_meas)
            c = W.clear_door(np.r_[q, phi_meas], self.T_WH_bind)
            for k in KEYS:
                self.mins['act'][k] = min(self.mins['act'][k], c[k])
            Tm, Tr = K.fk(q, 'link_tcp'), K.fk(x_ref[:9], 'link_tcp')
            self.rec.append([self.t, self.sigma, self.g, phi_meas, x_ref[9], ep, er, sp_, sr_, float(np.linalg.norm(q[:2] - x_ref[:2])), abs(q[2] - x_ref[2]), float(np.linalg.norm(Tm[:3, 3] - Tr[:3, 3])), *[c[k] for k in KEYS],
                             float(np.abs(self.u_last[3:]).max()), float(np.hypot(self.u_last[0], self.u_last[1]))])
            bad = [k for k in KEYS if c[k] < W.THR[k]]
            if self.overforce is not None:
                self._stop('overforce', **self.overforce)
            elif ep > CL_TOL['pos_m'] or er > CL_TOL['rot_deg']:
                self._stop('closure_residual', pos_mm=round(ep * 1e3, 3), rot_deg=round(er, 3))
            elif sp_ > STRESS_TOL['pos_m'] or sr_ > STRESS_TOL['rot_deg']:
                self._stop('setpoint_closure_stress', pos_mm=round(sp_ * 1e3, 3), rot_deg=round(sr_, 3))
            elif bad:
                self._stop('clearance', bad=bad, at={k: round(float(c[k]), 4) for k in KEYS})
            else:
                self._stall = self._stall + self.cfg.dt if (self.g <= 1e-9 and self.sigma < self.sig_end) else 0.0
                if self._stall > STALL_S:
                    self._stop('door_not_following', lead_deg=round(math.degrees(lead), 3))
            if self.stop is None and self.sigma >= self.sig_end:
                if self.t_ref_end is None:
                    self.t_ref_end = self.t
                    self._ev('ref_end', phi_meas_deg=round(math.degrees(phi_meas), 3))
                good = phi_meas >= self.phi_goal - END['phi_tol'] and self.rate['arm'] <= END['arm_rps'] and self.rate['base'] <= END['base_mps'] and self.rate['yaw'] <= END['yaw_rps'] and self.rate['door'] <= END['door_rps']
                self._n_end = self._n_end + 1 if good else 0
                if self._n_end >= END['n_ticks']:
                    self.done = True
                    self._stop('open_done', phi_meas_deg=round(math.degrees(phi_meas), 3))
                elif self.t - self.t_ref_end > END['wait_s']:
                    self._stop('end_not_reached', phi_meas_deg=round(math.degrees(phi_meas), 3), rates={k: round(v, 5) for k, v in self.rate.items()})
            if self.stop is None:
                w0 = time.perf_counter()
                b_ = self._sweep(q, self.u_last, (0,), 'stop_now')
                self.sweep_ms.append((time.perf_counter() - w0) * 1e3)
                if b_ is not None:
                    self._stop('stop_sweep_violation', **b_)
        path = self.stop is not None and self.stop_mode == 'path'
        if path and tick:                                           # 路徑停止：g 斜降，繼續追蹤；g＝0 且慢下來（或逾時）後轉命令歸零
            self.g = max(self.g - self.cfg.dt / T_STOP, 0.0)
            slow = self.rate['arm'] <= END['arm_rps'] and self.rate['base'] <= END['base_mps'] and self.rate['yaw'] <= END['yaw_rps']
            if (self.g <= 0.0 and slow) or self.t - self.t_path_stop > PATH_STOP_MAX_S:
                if not (self.g <= 0.0 and slow):
                    self._stop('path_stop_timeout')
                self.stop_mode = 'cmd' if self.stop_mode == 'cmd' else 'settled'
            else:
                u = self._solve(q, phi_meas)
                if u is None:
                    self._stop('solver_fail')
                else:
                    self.u_last = u
                    self.sigma = min(self.sigma + self.cfg.dt * self.g, self.sig_end)
                    self.u_prev = self.u_last.copy()
        if self.stop is not None and self.stop_mode in ('cmd', 'settled'):
            if self.stopped_t is None:
                self.stopped_t = self.t
            if tick:
                self.u_last = self._decel()
                self.u_prev = self.u_last.copy()
        elif self.stop is None and tick:
            u = self._solve(q, phi_meas)
            if u is None:
                self._stop('solver_fail')
            else:
                w0 = time.perf_counter()
                b_ = self._sweep(q, u, (1, 2), 'candidate')
                self.sweep_ms[-1] += (time.perf_counter() - w0) * 1e3
                if b_ is not None:
                    self._stop('candidate_stop_violation', **b_)
            if self.stop is None:
                self.u_last = u
                self.sigma = min(self.sigma + self.cfg.dt * self.g, self.sig_end)
                self.u_prev = self.u_last.copy()
            elif self.stop_mode == 'cmd':
                self.stopped_t = self.t
                self.u_last = self._decel()
                self.u_prev = self.u_last.copy()
        if tick:
            self.tick_ms.append((time.perf_counter() - w_t) * 1e3)
        self.pipe.append(self.u_last.copy())
        ua = self.pipe.pop(0)
        nxt = self.s + PHYS * ua[3:]
        if np.any(nxt < self.lim_lo) or np.any(nxt > self.lim_hi):
            nxt = np.clip(nxt, self.lim_lo, self.lim_hi)
            self._stop('setpoint_limit')
        self.s = nxt
        if self.stopped_t is not None and self.q_last is not None and self.stop_ok is None:
            dq = float(np.abs(q[3:] - self.q_last[3:]).max()) / PHYS
            vb = float(np.hypot(*(q[:2] - self.q_last[:2]))) / PHYS
            wy = abs(q[2] - self.q_last[2]) / PHYS
            dd = abs(phi_meas - self.phi_last) / PHYS
            empty = not np.any(self.u_last) and not any(np.any(u) for u in self.pipe)
            st = dict(pipe=bool(empty), arm=dq < STOP_STILL['arm_rps'], base=vb < STOP_STILL['base_mps'] and wy < STOP_STILL['yaw_rps'], door=dd < STOP_STILL['door_rps'])
            self._still = self._still + 1 if all(st.values()) else 0
            if self._still * PHYS >= STOP_STILL['hold_s'] or self.t - self.stopped_t >= STOP_STILL['timeout_s']:
                self.stop_ok = dict(t=round(self.t, 3), since_stop_s=round(self.t - self.stopped_t, 3), parts=st, arm_rate=dq, base_speed=vb, yaw_rate=wy, door_rate=dd, ok=bool(self._still * PHYS >= STOP_STILL['hold_s']))
        self.q_last, self.phi_last = q.copy(), float(phi_meas)
        self.n_step += 1
        return dict(arm_target=self.s.copy(), base_twist=ua[:3].copy(), stopped=self.stop is not None, done=self.done)

    def report(self):
        Rr = np.array(self.rec) if self.rec else np.zeros((1, 18))

        def st(col, sc=1.0):
            return dict(max=round(float(Rr[:, col].max()) * sc, 4), p95=round(float(np.percentile(Rr[:, col], 95)) * sc, 4), end=round(float(Rr[-1, col]) * sc, 4))

        def pct(a):
            a = np.array(a or [0.0])
            return dict(p50=round(float(np.percentile(a, 50)), 1), p95=round(float(np.percentile(a, 95)), 1), max=round(float(a.max()), 1))
        return dict(stop=self.stop, done=self.done, stop_effect=self.stop_ok, events=self.events, sigma_end=round(self.sig_end, 2), sigma=round(self.sigma, 3),
                    closure_measured=dict(pos_mm=st(5, 1e3), rot_deg=st(6)), setpoint_closure=dict(pos_mm=st(7, 1e3), rot_deg=st(8)), base_track_m=st(9), yaw_track_rad=st(10), tcp_vs_ref_mm=st(11, 1e3),
                    lead_deg_max=round(math.degrees(float((Rr[:, 4] - Rr[:, 3]).max())), 3), g_min=round(float(Rr[:, 2].min()), 3), g_lt1_frac=round(float((Rr[:, 2] < 1 - 1e-9).mean()), 3),
                    min_clear_actual={k: round(float(v), 4) for k, v in self.mins['act'].items()}, min_clear_stop_sweep={k: round(float(v), 4) for k, v in self.mins['stop'].items()},
                    emergency_stop_path=dict(closure_pos_mm_max=round(self.mins['stop_closure_pos'] * 1e3, 3), closure_rot_deg_max=round(self.mins['stop_closure_rot'], 3), n_cfg_over_tol=self.n_emerg_closure, note='緊急（命令減速）停止路徑的預測閉鏈偏差；超過容差＝該停法閉鏈不相容 ⇒ 緊急停止先解綁'), request_unbind=self.request_unbind, stop_mode=self.stop_mode, n_sweep_cfg=self.mins['n_sweep'],
                    arm_cmd_max=round(float(Rr[:, 16].max()), 4), base_cmd_max=round(float(Rr[:, 17].max()), 4), arm_sat_s=round(float((Rr[:, 16] >= float(B.W_ARM['v_arm']) - 1e-3).sum()) * self.cfg.dt, 2),
                    phi_end_deg=round(math.degrees(float(Rr[-1, 3])), 3), t_end=round(float(self.t), 2),
                    timing=dict(solve_ms=pct(self.solve_ms), sweep_ms=pct(self.sweep_ms), whole_tick_ms=pct(self.tick_ms)),
                    params=dict(T_STOP=T_STOP, PATH_STOP_MAX_S=PATH_STOP_MAX_S, EMERGENCY=list(EMERGENCY), LEAD_MAX_deg=3.0, LEAD_BAND_deg=1.0, STALL_S=STALL_S, CL_TOL=CL_TOL, STRESS_TOL=STRESS_TOL, END={k: (round(math.degrees(v), 3) if k == 'phi_tol' else v) for k, v in END.items()}, STOP_STILL=STOP_STILL, thresholds=W.THR, finger_q=W.FINGER_Q))
