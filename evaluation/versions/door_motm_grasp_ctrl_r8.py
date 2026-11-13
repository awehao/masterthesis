#!/usr/bin/env python3
"""移動中抓持控制器（MotM 開門第一包（2）（3）的受測對象；設計 r1 §1B）。假世界與（日後）Isaac 共用同一入口 step()。

底盤：沿世界方向 d 以 v 通過（本體速度命令＝世界速度轉本體；偏航與側向位置比例回授），經 8 步管線＋拍內斜坡，不停車。
手臂：每控制拍以「命令生效後一拍末」的預測底盤（管線清空後狀態＋本拍底盤命令積分）解 IK，使 TCP 落在
      T_at(把手估計, s(t))——s 由 S_PRE（預抓）經最小急動降到 0（抓取）；之後保持 s＝0（閉爪、確認期間持續追蹤，不凍結）。
      設定點＝q_ik＋τ·q̇_ik−off(q)，off(q)＝靜態偏置估計（上游觀測器值）＋重力變化／kp（同開門 A 的定義）。
段別（以時間＋條件推進）：HOLD（預抓位保持，s＝S_PRE）→ INSERT → CLOSE（命令兩指閉合斜坡）→ CONFIRM（相對運動條件連續成立）→ READY（可綁定；繼續追蹤）。
放行判準＝夾爪相對把手估計的位置／姿態誤差與相對速度在門檻內，連續 CONFIRM_S（每拍取最大值，不以首尾差抵消中間振盪）；不要求底盤停。
監看與停止：實測淨空（門扇／立柱／橫桿＝門角實測；兩指—橫桿在 INSERT 起為指定接觸只要求不穿入）、窗口邊界（底盤沿 d 的位置超出可抓窗口 ⇒ 停止）、
            現在停／候選下一拍與再晚一拍停止掃掠（door_exit_ctrl.stop_traj_exit，同一停法）、手臂速度／加速度、輪面速度／加速度。
            違反 ⇒ 受控停止（同一停法有界減速）；停止中新違反 ⇒ 中止。
真值只在受控對象與評估端；控制器只讀估計。
r2（Codex 20261113_120619：READY 是可撤銷的當下資格，不是永久相位標籤）：下游接受交接（accept_handover）之前，READY 每拍重判——
  相對位姿／速度（同 CONFIRM 門檻）、估計新鮮度（最新採用影格的擷取時刻距今 ≤ READY_Q.fresh_s）、剩餘窗口（「現在停」預測停止終點沿 d 仍在窗口內）、
  停止可行（現在停掃掠通過，原本就每拍做）。相對運動或新鮮度不過 ⇒ 撤銷 READY 回 CONFIRM（重新累積）；停止終點出窗口或首次 READY 後
  accept_max_s 內未被接受 ⇒ 受控停止。CONFIRM 也要求估計新鮮。窗口邊界檢查在 READY 中照常（r1 對 READY 豁免，已移除）。
  每個合格拍給一個 ready_ver；accept_handover(ver) 只接受「當拍」版本，舊版本拒絕。READY 只代表模型內的放行條件，不代表實際雙指接觸或移動綁定已驗證。
r3（G3）：pool＝工作程序數（0＝r2）：停止掃掠的淨空與兩指—橫桿距離分給預建工作程序（同函式、同樣本、原順序判定）；逾時／程序失敗／缺結果 ⇒ 模擬中止。
  中止那一步起不再出命令（r2 的最後回傳把 abort 固定為 False，中止要到下一步才生效；已改）。
r4（Codex 20261113_124213 §1）：閉爪開始鎖定有時間戳的接觸任務參考 lock＝(T_est, φ_meas, t, est_ver)——不是鎖關節、也不是鎖夾爪相對底盤；
  之後手臂目標＝T_at(Rz[−(φ_meas−φ_lock)]·T_lock, s)（門動就按機構更新），底盤照走、手臂依生效時刻的預測底盤補償；新影像只做一致性監看
  （機構校正後的融合估計對鎖定參考差 > FUSE 門檻 ⇒ contact_ref_inconsistent 停止），不灌進接觸中的運動目標。READY 的相對判準改對鎖定參考。
  閉爪開始≠抓持成立（接觸、開度、相對運動放行照舊另核）。
  接觸後停止（CLOSE／CONFIRM／READY）改「接觸停法」：底盤命令照原本每拍有界減速，手臂每拍仍以預測底盤解 IK 追鎖定把手（s＝0）——不是底盤、手臂各自減速；
  停止掃掠在接觸段用同一停法向前模擬（每拍 IK、命令限制、淨空），驗不過（IK 失敗、限制、未停穩）⇒ 違反；實際停止中命令驗不過 ⇒ 模擬中止。不自動張爪、不解綁。
  接觸停法停穩判準：底盤命令與管線為零、手臂命令 ≤ CSTOP.arm_cmd、各速率低於門檻連續 0.3 s。
r6（Codex 20261113_130916 §2）：
  相對速度改用同一拍的「把手參考—TCP 相對變換」差分（rel_motion）：門與夾爪一起動、相對不變 ⇒ 0；TCP 不動、把手動 ⇒ 有相對運動。
  （r1–r5 用 TCP 世界位姿差分，只在門靜止時等於相對速度。）
  滑脫快速拒絕通道（鎖定後）：最近 FAST.win_s 內擷取的原始影格（不經融合閘門），按擷取時刻的門角機構校正到現在門角後取中位，
  對鎖定參考差 > FUSE 門檻 ⇒ contact_slip_suspect（接觸停法）。融合估計仍供規劃與一般監看；快速通道只拒絕、不改運動目標。
  訊號來源（照實）：門角＝step() 的 phi_meas（假世界＝受控對象真值；Isaac＝門鉸鏈關節讀值，是模擬量測，實機需另行估計）；
  把手影格＝est_fn（假世界＝真值＋偏差＋雜訊＋延遲＋遮擋；Isaac＝腕部相機偵測）；本控制器不使用接觸力訊號（指尖接觸由執行器另核）。
r7（Codex 20261113_130916 §4）：deadline_ms（None＝r6）。拍首計時，提交候選時已超過 ⇒ 不提交（晚到不放行）；本拍「現在停」掃掠已通過
  （鎖定後即接觸停法的向前模擬：IK、命令限制、停穩、淨空）⇒ 以該停法停止（tick_deadline_miss），否則模擬中止。tick_log 分記正常拍與停止拍耗時。
  測試掛鉤：deadline_inject（強制本拍晚到）、fb_inject（使本拍備援視為未驗證）。
r8（Codex 20261113_133147 §3）：close_gate（可呼叫，回 True 才准從 INSERT 進入 CLOSE）——下游首段未及時可用就不開始閉爪；
  INSERT 時間到而閘門未開 ⇒ close_gate_not_ready 受控停止（尚未接觸，用原停法）。None＝r7。
r5：fastkin＝True 時 IK 改 FastChain（door_fast_kin.ik_closure_fc 同迭代，解差 4e-14 級）、重力改 FastGravity2（差 9e-15 N·m）；預設 False＝r4。"""
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_exit_ctrl as XE                                          # noqa: E402

XO, W, K, C0, B, S, PG, E, LIM, BM = XE.XO, XE.W, XE.K, XE.C0, XE.B, XE.S, XE.PG, XE.E, XE.LIM, XE.BM
PHYS, NPH, D_STEPS, DT, KEYS = XE.PHYS, XE.NPH, XE.D_STEPS, XE.DT, XE.KEYS
SCHED = dict(hold_s=0.3, ins_s=1.0, close_s=0.5, confirm_s=0.3)
CONFIRM = dict(pos_m=0.005, rot_deg=3.0, lin_mps=0.005, ang_rps=0.02)
FUSE = dict(win_s=1.0, gate_m=0.005, gate_deg=3.0, min_frames=5)
FAST = dict(win_s=0.2, min_frames=3)                                 # r6 事前登錄：滑脫快速通道視窗與最少影格（門檻沿用 FUSE.gate_m／gate_deg）     # 把手估計融合：近 win_s 影格的穩健平均（把手在世界靜止）；離群門檻
FINGER_CLOSED = 0.0017
CSTOP = dict(arm_cmd=0.005, max_steps=2000)                         # r4 事前登錄：接觸停法停穩的手臂命令上限（rad/s）、預測步數上限
READY_Q = dict(fresh_s=0.3, accept_max_s=0.5)                       # r2 事前登錄：估計新鮮度上限（30 Hz、延遲 0.1 s 下約 0.13 s）、首次 READY 後等待下游接受的上限


class MovingGraspCtrl:
    def __init__(self, ident, est_fn, d_world, v_pass, window_s, s, pipe, u_last, u_prev, off_static, finger_open=0.0089, joint_kp=600.0, sweep_stride=3, sched=None, pool=0, fastkin=False, deadline_ms=None):
        self.fastkin = bool(fastkin)
        self.deadline_ms = None if deadline_ms is None else float(deadline_ms)
        self.deadline_inject, self.fb_inject = None, None
        self.n_deadline_miss, self.tick_log, self._fb_ok = 0, [], False
        self.close_gate = None                                       # r8
        self._pool = None
        if int(pool) > 0:
            import door_sweep_pool as DSP
            self._pool = DSP.SweepPool(int(pool))
        self.pool_ms = []
        self.cfg = C0.make_cfg(ident)
        self.cfg.base_fixed = False
        al = np.asarray(self.cfg.arm_model.alpha, float)
        self.cfg.arm_model.bias = al * np.asarray(off_static, float)
        self.tau, self.va = PHYS / al, float(B.W_ARM['v_arm'])
        self.est_fn = est_fn                                         # 回 (T_WH 估計, 版本, 擷取時刻) 或 None
        self.d = np.asarray(d_world, float) / np.linalg.norm(d_world)
        self.v = float(v_pass)
        self.win = tuple(window_s)                                   # 沿 d 的可抓窗口（相對 window 原點的位移，m）
        self.s = np.asarray(s, float).copy()
        self.pipe = [np.asarray(u, float).copy() for u in pipe]
        self.u_last, self.u_prev = np.asarray(u_last, float).copy(), np.asarray(u_prev, float).copy()
        self.u_from_base, self.ramp_j = np.asarray(self.pipe[-1][:3], float).copy(), NPH
        self.off0 = np.asarray(off_static, float).copy()
        self.joint_kp = float(joint_kp)
        self.bodies = BM.arm_bodies(K)
        import door_fast_kin as DFK
        self._DFK = DFK
        self._grav2 = DFK.FastGravity2(K, self.bodies) if self.fastkin else None
        self.G0 = None
        self.finger_q = float(finger_open)
        self.finger_open = float(finger_open)
        self.sweep_stride = int(sweep_stride)
        self.sched = dict(SCHED, **(sched or {}))
        Rg, self.T_HG, self.a_H = PG.baseline_reference()
        self.T_at = lambda T, s_: np.asarray(E.OT.object_target(T, self.T_HG, self.a_H, float(s_))['T_WE'], float)   # noqa: E731
        self.S_PRE = float(PG.S_PRE)
        self.lim_lo, self.lim_hi = np.asarray(XO.LITE6_SAFE.lower, float) + 0.05, np.asarray(XO.LITE6_SAFE.upper, float) - 0.05
        self.t, self.n_step, self.t0 = 0.0, 0, None
        self.phase, self.t_phase, self.ph_hist = 'HOLD', 0.0, [('HOLD', 0.0)]
        self.b_origin, self.yaw_ref = None, None
        self.q_ik_prev, self.T_est, self.est_ver = None, None, None
        self.frames, self.fuse = {}, dict(used=0, rejected=0, last_ver=None)
        self.stop, self.stop_t, self.stop_ok, self.abort, self.still_n = None, None, None, None, 0
        self.confirm_n, self.ready_t = 0, None
        self.ready_now, self.ready_ver, self.n_ready_ver, self.accepted, self.accept_t = False, None, 0, False, None
        self.n_revoke, self.stop_end_s, self.est_age = 0, None, None
        self.lock, self.phi_now, self.lock_incons_max = None, None, dict(pos_mm=0.0, deg=0.0)
        self.cstop = dict(used=False, n_ticks=0)
        self.raw_frames, self.phi_hist, self.fast = [], [], dict(max_pos_mm=0.0, max_deg=0.0, n_eval=0)
        self.rel_prev_T = None
        self.rec, self.events, self.cmd_log = [], [], []
        self.mins = {k: np.inf for k in KEYS}
        self.mins.update(finger_bar=np.inf, stop_sweep=np.inf)
        self.lim = dict(arm_rate=0.0, arm_acc=0.0, wheel_v=0.0, wheel_a=0.0)
        self.rate = dict(arm=0.0, base=0.0, yaw=0.0)
        self.q_tick, self.rel_prev = None, None
        self.u_body_prev = np.asarray(self.u_last[:3], float).copy()

    # ---- 共用
    def _ev(self, what, **kw):
        self.events.append(dict(t=round(self.t, 3), what=what, **kw))

    def _stop(self, why, **kw):
        if self.stop is None:
            self.stop, self.stop_t = dict(t=round(self.t, 3), why=why, phase=self.phase, **kw), self.t
            self._ev('stop', why=why, phase=self.phase, **kw)

    def _abort(self, why, **kw):
        if self.abort is None:
            self.abort = dict(t=round(self.t, 3), why=why, phase=self.phase, **kw)
            self.stop_ok = dict(ok=None, status='unverified（模擬中止）')

    def _gravity(self, q9):
        return self._grav2(q9) if self._grav2 is not None else BM.arm_gravity(K, q9, self.bodies)

    def off(self, q9):
        return self.off0 + (self._gravity(q9) - self.G0) / self.joint_kp

    def s_of(self, t_rel):
        """TCP 沿接近軸的退讓量 s（S_PRE→0）：HOLD 段 S_PRE、INSERT 段最小急動、之後 0。t_rel 自控制器起點。"""
        h, i = self.sched['hold_s'], self.sched['ins_s']
        if t_rel <= h:
            return self.S_PRE
        a, _ = XE.minjerk((t_rel - h) / i)
        return self.S_PRE * (1.0 - a)

    def handle_ref(self, phi=None):
        """r4：鎖定前＝融合估計；鎖定後＝鎖定參考按門角機構更新（鉸鏈在世界原點，同 door_wbopt.handle_pose）。"""
        if self.lock is None:
            return self.T_est
        ph = self.phi_now if phi is None else phi
        return W.Rz4(-(float(ph) - self.lock['phi'])) @ self.lock['T']

    def target(self, t_rel):
        return self.T_at(self.handle_ref(), self.s_of(t_rel))

    def _arm_track(self, qp, sp, uf, ub, te, seed):
        """命令生效後一拍末的預測底盤（qp 起、本拍底盤命令斜坡）解 IK → 設定點 → 手臂命令。回 (u_arm 或 None, 殘差, q1)。"""
        b = np.asarray(qp[:3], float).copy()
        uf = np.asarray(uf, float)
        for jj in range(NPH):
            vv = uf + min(jj + 1, NPH) / NPH * (ub - uf)
            cb, sb = math.cos(b[2]), math.sin(b[2])
            b = b + PHYS * np.array([cb * vv[0] - sb * vv[1], sb * vv[0] + cb * vv[1], vv[2]])
        q1, r1 = self._ik(b, self.target(te), seed)
        q0_, r0 = self._ik(qp[:3], self.target(te - DT), q1)
        if max(r1, r0) > 1e-6 or float(np.abs(q1 - seed).max()) > 0.5:
            return None, max(r1, r0), None
        S1 = q1 + self.tau * (q1 - q0_) / DT - self.off(np.r_[b, q1])
        return (S1 - sp) / DT, max(r1, r0), q1

    def _lim_bad(self, u, u_prev):
        bad = []
        if float(np.abs(u[3:]).max()) > self.va + 1e-9:
            bad.append('arm_rate')
        if float(np.abs(u[3:] - u_prev[3:]).max()) / DT > C0.A_ARM + 1e-9:
            bad.append('arm_acc')
        if float(np.abs(LIM.WM @ u[:3]).max()) > LIM.W_LIM + 1e-9:
            bad.append('wheel_v')
        if float(np.abs(LIM.WM @ (u[:3] - u_prev[:3])).max()) / DT > LIM.WA_LIM + 1e-9:
            bad.append('wheel_a')
        return bad

    def _contact_stop_traj(self, q, u0, extra, tr, j0, u_from):
        """r4：接觸停法的向前模擬（真管線、拍內斜坡、每拍：底盤有界減速、手臂以預測底盤 IK 追鎖定把手）。回 (Q 取樣, 違反或 None, 減速起點)。"""
        qp, sp = np.asarray(q, float).copy(), self.s.copy()
        pp = [np.asarray(x, float).copy() for x in self.pipe]
        u, uf, j = np.asarray(u0, float).copy(), np.asarray(u_from, float).copy(), int(j0)
        u_prev = u.copy()
        dab = B.A_BASE * self.cfg.dt
        out, k, quiet, k_dec, n_tick = [qp.copy()], 0, 0, None, 0
        seed = self.q_ik_prev if self.q_ik_prev is not None else qp[3:]
        while quiet < XO.C21.SETTLE_STEPS and k < CSTOP['max_steps']:
            if j >= NPH:
                uf, j = XO.ramp_cmd(uf, u, NPH - 1)[:3], 0
                if k >= extra * NPH:
                    if k_dec is None:
                        k_dec = k
                    ub = u[:3] + np.clip(-u[:3], -dab, dab)
                    q2, s2 = qp.copy(), sp.copy()
                    for x in pp:
                        q2, s2 = S.plant_phys_step(q2, s2, x, self.cfg, 1)
                    te = tr + k * PHYS + D_STEPS * PHYS + DT
                    ua, res, q1 = self._arm_track(q2, s2, pp[-1][:3], ub, te, seed)
                    if ua is None:
                        return np.array(out), dict(what='contact_stop_ik_fail', k=k, resid=res), k_dec
                    un = np.r_[ub, ua]
                    bad = self._lim_bad(un, u_prev)
                    if bad:
                        return np.array(out), dict(what='contact_stop_limit', k=k, bad=bad), k_dec
                    seed, u_prev, u = q1, un, un
                n_tick += 1
            cmd = XO.ramp_cmd(uf, u, j)
            pp.append(cmd)
            qp, sp = S.plant_phys_step(qp, sp, pp.pop(0), self.cfg, 1)
            k += 1
            j += 1
            settled = (not np.any(cmd[:3]) and not any(np.any(x[:3]) for x in pp) and float(np.abs(cmd[3:]).max()) <= CSTOP['arm_cmd'])
            quiet = quiet + 1 if settled else 0
            if k % XO.C21.SAMPLE_EVERY == 0:
                out.append(qp.copy())
        if quiet < XO.C21.SETTLE_STEPS:
            return np.array(out), dict(what='contact_stop_not_settled', k=k), k_dec
        return np.array(out), None, k_dec

    def _pred(self, q):
        qp, sp = np.asarray(q, float).copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, self.cfg, 1)
        return qp, sp

    def _ik(self, b, T, seed):
        if self.fastkin:
            q_, r_, _ = self._DFK.ik_closure_fc(T, np.asarray(b, float), seed)
            return q_, r_
        q = np.asarray(seed, float).copy()
        res = np.inf
        for _ in range(30):
            q9 = np.r_[b, q]
            Tc = K.fk(q9, 'link_tcp')
            dR = T[:3, :3] @ Tc[:3, :3].T
            e = np.r_[T[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
            res = float(np.linalg.norm(e))
            if res < 1e-10:
                break
            J = K.jacobian(q9, 'link_tcp')[:, 3:9]
            q = q + np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
        return q, res

    def clear(self, q9, phi, exempt):
        W.FINGER_Q = self.finger_q
        c = W.clear_door(np.r_[q9, phi], self.T_WH_model)
        c['finger_bar'] = XE.DoorExitCtrl.finger_bar(self, q9, phi)
        return c

    def _bad(self, c, exempt):
        """exempt：'insert'＝兩指—橫桿只要求不穿入（張開插入）；'contact'＝閉爪起兩指—橫桿為指定接觸（同靜止版 contact=True），只核夾爪殼；None＝全部照門檻。"""
        b_ = [k for k in KEYS if c[k] < W.THR[k]]
        if exempt is None and c['finger_bar'] < W.THR['env']:
            b_.append('finger_bar')
        elif exempt == 'insert' and c['finger_bar'] < 0.0:
            b_.append('finger_bar')
        return b_

    def close_pool(self):
        if self._pool is not None:
            self._pool.close()

    def _stop_traj(self, q, u, e, j0, u_from):
        """r4：接觸段（已鎖定）用接觸停法；其餘用原停法。回 (Q, 違反或 None, k_dec)。"""
        if self.lock is not None:
            return self._contact_stop_traj(q, u, e, self.t - self.t0, j0, u_from)
        Q, cmds, k_dec = XE.stop_traj_exit(q, self.s, self.pipe, u, u_from, j0, self.cfg, e)
        return Q, None, k_dec

    def _sweep_pool(self, q, u, extras, j0, u_from, phi, exempt, tag):
        items, X, ends = [], [], {}
        for e in extras:
            Q, vtraj, k_dec = self._stop_traj(q, u, e, j0, u_from)
            if vtraj is not None:
                return dict(tag=tag, extra=e, traj=vtraj, k_decel=k_dec)
            if tag == 'stop_now' and e == 0:
                self.stop_end_s = float((np.asarray(Q[-1][:2]) - self.b_origin[:2]) @ self.d)
            idx = list(range(0, len(Q), self.sweep_stride)) + [len(Q) - 1]
            for i in idx:
                items.append((e, i, len(X), k_dec))
                X.append(np.r_[Q[i], phi])
        W.FINGER_Q = self.finger_q
        h = self._pool.submit(self.T_WH_model, W.FINGER_Q, W.PHI0, np.array(X), finger_bar_q=self.finger_q)
        res, why, tms = self._pool.collect(h, XO.POOL_TIMEOUT_S)
        self.pool_ms.append(tms)
        if res is None:
            self._abort('sweep_unavailable', pool_why=why, tag=tag)
            return dict(tag=tag, pool_unavailable=why)
        for e, i, j, k_dec in items:
            c = res[j]
            self.mins['stop_sweep'] = min(self.mins['stop_sweep'], min(c[k] - W.THR[k] for k in KEYS if k != 'shell_bar'))
            b_ = self._bad(c, exempt)
            if b_:
                return dict(tag=tag, extra=e, i=i, bad=b_, k_decel=k_dec)
        return None

    def _sweep(self, q, u, extras, j0, u_from, phi, exempt, tag):
        if self._pool is not None:
            return self._sweep_pool(q, u, extras, j0, u_from, phi, exempt, tag)
        for e in extras:
            Q, vtraj, k_dec = self._stop_traj(q, u, e, j0, u_from)
            if vtraj is not None:
                return dict(tag=tag, extra=e, traj=vtraj, k_decel=k_dec)
            if tag == 'stop_now' and e == 0:
                self.stop_end_s = float((np.asarray(Q[-1][:2]) - self.b_origin[:2]) @ self.d)   # r2：現在停的停止終點沿 d
            idx = list(range(0, len(Q), self.sweep_stride)) + [len(Q) - 1]
            for i in idx:
                c = self.clear(Q[i], phi, exempt)
                self.mins['stop_sweep'] = min(self.mins['stop_sweep'], min(c[k] - W.THR[k] for k in KEYS if k != 'shell_bar'))
                b_ = self._bad(c, exempt)
                if b_:
                    return dict(tag=tag, extra=e, i=i, bad=b_, k_decel=k_dec)
        return None

    def _check_cmd(self, u):
        bad = []
        ar = float(np.abs(u[3:]).max())
        aa = float(np.abs(u[3:] - self.u_prev[3:]).max()) / DT
        wv = float(np.abs(LIM.WM @ u[:3]).max())
        wa = float(np.abs(LIM.WM @ (u[:3] - self.u_prev[:3])).max()) / DT
        for k_, v_ in (('arm_rate', ar), ('arm_acc', aa), ('wheel_v', wv), ('wheel_a', wa)):
            self.lim[k_] = max(self.lim[k_], v_)
        if ar > self.va + 1e-9:
            bad.append(dict(what='arm_rate', v=round(ar, 4)))
        if aa > C0.A_ARM + 1e-9:
            bad.append(dict(what='arm_acc', v=round(aa, 3)))
        if wv > LIM.W_LIM + 1e-9:
            bad.append(dict(what='wheel_v', v=round(wv, 4)))
        if wa > LIM.WA_LIM + 1e-9:
            bad.append(dict(what='wheel_a', v=round(wa, 3)))
        return bad

    @staticmethod
    def rel_motion(Trel_prev, Trel):
        """r6：相對變換差分 ⇒ (線速度 m/s, 角速度 rad/s)，以把手座標表示（與世界運動無關）。"""
        lin = float(np.linalg.norm(Trel[:3, 3] - Trel_prev[:3, 3])) / DT
        dR = Trel_prev[:3, :3].T @ Trel[:3, :3]
        return lin, math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))) / DT

    def _rel(self, q):
        """控制器視角的夾爪相對把手（估計）誤差：位置 m、姿態 °（TCP 對 T_at(估計, s＝0)）。"""
        Tc, Tg = K.fk(q, 'link_tcp'), self.T_at(self.handle_ref(), 0.0)      # r4：鎖定後對鎖定參考
        dR = Tg[:3, :3].T @ Tc[:3, :3]
        return float(np.linalg.norm(Tc[:3, 3] - Tg[:3, 3])), math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))), Tc

    def step(self, t, q, phi_meas):
        self.t = float(t)
        q = np.asarray(q, float)
        if self.abort is not None:
            return dict(arm_target=self.s.copy(), base_twist=np.zeros(3), finger=self.finger_q, ready=False, stopped=True, abort=True)
        tick = self.n_step % NPH == 0
        w_t = time.perf_counter()
        if tick:
            self._fb_ok = False
            if self.t0 is None:
                self.t0, self.b_origin, self.yaw_ref = self.t, q[:3].copy(), float(q[2])
                self.G0 = self._gravity(q)
            est = self.est_fn(self.t)
            self.phi_hist.append((self.t, float(phi_meas)))
            if est is not None and (not self.raw_frames or self.raw_frames[-1][0] != est[1]):
                self.raw_frames.append((est[1], float(est[2]), np.asarray(est[0], float).copy()))   # r6：原始影格（不經融合閘門）
                self.raw_frames = [f_ for f_ in self.raw_frames if f_[1] >= self.t - 2.0]
            if est is not None and est[1] not in self.frames:
                Tn = np.asarray(est[0], float)
                if self.T_est is not None and len(self.frames) >= FUSE['min_frames']:
                    dR = self.T_est[:3, :3].T @ Tn[:3, :3]
                    dang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
                    if np.linalg.norm(Tn[:3, 3] - self.T_est[:3, 3]) > FUSE['gate_m'] or dang > FUSE['gate_deg']:
                        self.fuse['rejected'] += 1
                        Tn = None
                if Tn is not None:
                    self.frames[est[1]] = (float(est[2]), Tn)
            keep = {k_: v_ for k_, v_ in self.frames.items() if v_[0] >= self.t - FUSE['win_s'] - 1e-9}
            if keep:
                Ts = [v_[1] for v_ in keep.values()]
                Tm = np.eye(4)
                Tm[:3, 3] = np.mean([T_[:3, 3] for T_ in Ts], axis=0)
                U_, _, Vt_ = np.linalg.svd(np.sum([T_[:3, :3] for T_ in Ts], axis=0))
                Tm[:3, :3] = U_ @ np.diag([1.0, 1.0, np.linalg.det(U_ @ Vt_)]) @ Vt_
                self.T_est, self.est_ver = Tm, max(keep)
                self.fuse.update(used=len(keep), last_ver=max(keep))
            if self.T_est is None:
                self._stop('no_estimate')
            self.est_age = None if self.est_ver is None else self.t - float(self.frames[self.est_ver][0])
            fresh = self.est_age is not None and self.est_age <= READY_Q['fresh_s'] + 1e-9
            self.phi_now = float(phi_meas)
            self.T_WH_model = self.handle_ref()                        # 淨空模型的把手＝估計（r4：鎖定後＝鎖定參考）；門角 φ 由實測給
            W.set_model(phi0=float(phi_meas), finger_q=self.finger_q)
            if self.lock is not None and self.T_est is not None:       # r4：新影像只做一致性監看
                Hr = self.handle_ref()
                dpos = float(np.linalg.norm(self.T_est[:3, 3] - Hr[:3, 3]))
                dR_ = Hr[:3, :3].T @ self.T_est[:3, :3]
                dang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR_) - 1) / 2))))
                self.lock_incons_max = dict(pos_mm=max(self.lock_incons_max['pos_mm'], dpos * 1e3), deg=max(self.lock_incons_max['deg'], dang))
                if (dpos > FUSE['gate_m'] or dang > FUSE['gate_deg']) and self.stop is None:
                    self._stop('contact_ref_inconsistent', dpos_mm=round(dpos * 1e3, 3), dang_deg=round(dang, 3))
                rec_ = [f_ for f_ in self.raw_frames if f_[1] >= self.t - FAST['win_s'] - 1e-9 and f_[1] >= self.lock['t'] - 1e-9]
                if len(rec_) >= FAST['min_frames']:                    # r6：滑脫快速通道
                    def phi_at(tc):
                        cand = [p_ for t_, p_ in self.phi_hist if t_ <= tc + 1e-9]
                        return cand[-1] if cand else self.phi_hist[0][1]
                    Ps, Rs_ = [], []
                    for _, tc, Tf in rec_:
                        Ta = W.Rz4(-(float(phi_meas) - phi_at(tc))) @ Tf  # 擷取時門角 → 現在門角
                        Ps.append(Ta[:3, 3]); Rs_.append(Ta[:3, :3])
                    pm = np.median(np.array(Ps), axis=0)
                    dps = float(np.linalg.norm(pm - Hr[:3, 3]))
                    dgs = sorted(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Hr[:3, :3].T @ R_) - 1) / 2)))) for R_ in Rs_)[len(Rs_) // 2]
                    self.fast = dict(max_pos_mm=max(self.fast['max_pos_mm'], dps * 1e3), max_deg=max(self.fast['max_deg'], dgs), n_eval=self.fast['n_eval'] + 1)
                    if (dps > FUSE['gate_m'] or dgs > FUSE['gate_deg']) and self.stop is None:
                        self._stop('contact_slip_suspect', dpos_mm=round(dps * 1e3, 3), dang_deg=round(dgs, 3), n_frames=len(rec_))
            if self.q_tick is not None:
                self.rate = dict(arm=float(np.abs(q[3:] - self.q_tick[3:]).max()) / DT, base=float(np.linalg.norm(q[:2] - self.q_tick[:2])) / DT, yaw=abs(q[2] - self.q_tick[2]) / DT)
            self.q_tick = q.copy()
            tr = self.t - self.t0
            exempt = 'insert' if self.phase == 'INSERT' else ('contact' if self.phase in ('CLOSE', 'CONFIRM', 'READY') else None)
            # 監看
            c = self.clear(q, phi_meas, exempt)
            for k in KEYS:
                self.mins[k] = min(self.mins[k], c[k])
            self.mins['finger_bar'] = min(self.mins['finger_bar'], c['finger_bar'])
            sd = float((q[:2] - self.b_origin[:2]) @ self.d)
            viol = []
            b_ = self._bad(c, exempt)
            if b_:
                viol.append(('clearance', dict(bad=b_, at={k: round(float(v), 4) for k, v in c.items() if k != 'who'})))
            if not (self.win[0] - 1e-9 <= sd <= self.win[1] + 1e-9):
                viol.append(('outside_window', dict(s_along=round(sd, 4), window=self.win)))
            ep, er, Tc = self._rel(q)
            Tg_now = self.T_at(self.handle_ref(), 0.0)
            Trel = np.linalg.inv(Tg_now) @ Tc                          # r6：把手參考（s＝0 抓取位）—TCP 相對變換（同一拍）
            if self.rel_prev_T is not None:
                lin, ang = self.rel_motion(self.rel_prev_T, Trel)
            else:
                lin = ang = np.inf
            self.rel_prev_T = Trel
            self.rel_prev = Tc
            self.rec.append([self.t, tr, sd, ep, er, lin, ang, c['env'], c['self'], c['shell_bar'], c['finger_bar'], ['HOLD', 'INSERT', 'CLOSE', 'CONFIRM', 'READY', 'STOP'].index(self.phase if self.stop is None else 'STOP')])
            if viol and self.abort is None:
                if self.stop is None:
                    self._stop(viol[0][0], **viol[0][1])
                elif viol[0][0] != self.stop['why']:
                    self._abort('violation_during_stop', viol=[v_[0] for v_ in viol])
                    return dict(arm_target=self.s.copy(), base_twist=np.zeros(3), finger=self.finger_q, ready=False, stopped=True, abort=True)
            # 段別
            if self.stop is None:
                h, i, cl, cf = (self.sched[k] for k in ('hold_s', 'ins_s', 'close_s', 'confirm_s'))
                if self.phase == 'HOLD' and tr >= h:
                    self._phase('INSERT')
                elif self.phase == 'INSERT' and tr >= h + i and self.close_gate is not None and not self.close_gate(self):
                    self._stop('close_gate_not_ready')                 # r8：下游首段未就緒 ⇒ 不閉爪
                elif self.phase == 'INSERT' and tr >= h + i:
                    self._phase('CLOSE')
                    self.lock = dict(T=self.T_est.copy(), phi=float(phi_meas), t=round(self.t, 3), est_ver=self.est_ver)   # r4：閉爪開始鎖定接觸任務參考
                    self._ev('contact_ref_locked', phi_deg=round(math.degrees(float(phi_meas)), 4), est_ver=self.est_ver)
                elif self.phase == 'CLOSE' and tr >= h + i + cl:
                    self._phase('CONFIRM')
                rel_ok = ep <= CONFIRM['pos_m'] and er <= CONFIRM['rot_deg'] and lin <= CONFIRM['lin_mps'] and ang <= CONFIRM['ang_rps']
                if self.phase == 'READY':                              # r2：READY 每拍重判（交接被接受前）
                    why_r = [w_ for w_, ok_ in (('rel_motion', rel_ok), ('stale_estimate', fresh)) if not ok_]
                    if why_r:
                        self.n_revoke += 1
                        self._ev('ready_revoked', why=why_r, ep_mm=round(ep * 1e3, 3), er_deg=round(er, 3), lin_mmps=round(lin * 1e3, 3), est_age_s=None if self.est_age is None else round(self.est_age, 3))
                        self._phase('CONFIRM')
                        self.confirm_n = 0
                if self.ready_t is not None and not self.accepted and self.t - self.ready_t > READY_Q['accept_max_s'] + 1e-9:
                    self._stop('accept_timeout', since_first_ready_s=round(self.t - self.ready_t, 3), phase_now=self.phase)   # 首次 READY 起算，撤銷回 CONFIRM 也照計
                if self.phase == 'CONFIRM':
                    ok = rel_ok and fresh
                    # 相對速度：夾爪對把手（把手靜止於世界 ⇒ 夾爪世界速度）；每拍取值、連續成立才累積
                    self.confirm_n = self.confirm_n + 1 if ok else 0
                    if self.confirm_n * DT >= cf - 1e-9:
                        self._phase('READY')
                        if self.ready_t is None:
                            self.ready_t = self.t                      # 首次 READY（接受等待上限由此起算，撤銷不重置）
            if self.phase in ('CLOSE', 'CONFIRM', 'READY'):
                self.finger_q = max(FINGER_CLOSED, self.finger_q - (self.finger_open - FINGER_CLOSED) * DT / self.sched['close_s'])
            # 現在停
            if self.stop is None:
                sw0 = self._sweep(q, self.u_last, (0,), self.ramp_j, self.u_from_base, phi_meas, exempt, 'stop_now')
                if sw0 is not None:
                    self._stop('stop_now_violation', **sw0)
                else:
                    self._fb_ok = self.fb_inject is None or bool(self.fb_inject(self))
            if self.stop is None and self.phase == 'READY' and self.stop_end_s is not None and self.stop_end_s > self.win[1] + 1e-9:
                self._stop('window_insufficient', stop_end_s=round(self.stop_end_s, 4), window=self.win)   # r2：現在停會停到窗口外
            self.ready_now = bool(self.stop is None and self.phase == 'READY')
            if self.ready_now:
                self.n_ready_ver += 1
                self.ready_ver = self.n_ready_ver
            else:
                self.ready_ver = None
            if self.stop is None:
                qp, sp = self._pred(q)
                yaw_err = (self.yaw_ref - qp[2] + math.pi) % (2 * math.pi) - math.pi
                lat = float(np.cross(np.r_[self.d, 0.0], np.r_[qp[:2] - self.b_origin[:2], 0.0])[2])
                nrm = np.array([-self.d[1], self.d[0]])
                vw = self.v * self.d - 1.0 * lat * nrm
                cth, sth = math.cos(qp[2]), math.sin(qp[2])
                ub = np.array([cth * vw[0] + sth * vw[1], -sth * vw[0] + cth * vw[1], 1.5 * yaw_err])
                ub = np.clip(ub, LIM.BODY_V_MIN, LIM.BODY_V_MAX)
                ub = np.clip(ub, self.u_body_prev - LIM.A_BODY * DT, self.u_body_prev + LIM.A_BODY * DT)
                # 預測底盤：管線清空後再以本拍命令（斜坡）走一拍
                te = tr + D_STEPS * PHYS + DT
                seed = self.q_ik_prev if self.q_ik_prev is not None else qp[3:]
                ua_, res_, q1 = self._arm_track(qp, sp, np.asarray(self.pipe[-1][:3], float).copy(), ub, te, seed)
                if ua_ is None:
                    self._stop('ik_fail', resid=res_)
                else:
                    self.q_ik_prev = q1
                    u = np.r_[ub, ua_]
                    bad = self._check_cmd(u)
                    if bad:
                        self._stop('cmd_limit', bad=bad)
                    else:
                        sw = self._sweep(q, u, XE.STOP_EXTRAS, 0, np.asarray(self.pipe[-1][:3], float), phi_meas, exempt, 'candidate')
                        late = self.deadline_ms is not None and ((time.perf_counter() - w_t) * 1e3 > self.deadline_ms or (self.deadline_inject is not None and self.deadline_inject(self)))
                        if sw is not None:
                            self._stop('candidate_stop_violation', **sw)
                        elif late:                                     # r7：晚到候選不提交
                            self.n_deadline_miss += 1
                            el = round((time.perf_counter() - w_t) * 1e3, 1)
                            if self._fb_ok:
                                self._stop('tick_deadline_miss', elapsed_ms=el, contact=self.lock is not None)
                            else:
                                self._stop('tick_deadline_miss', elapsed_ms=el, contact=self.lock is not None)
                                self._abort('tick_deadline_miss_unverified', elapsed_ms=el)
                        else:
                            self.u_last = u
            if self.stop is not None and self.lock is not None and self.abort is None:   # r4：接觸停法（底盤有界減速、手臂照追鎖定把手）
                self.cstop['used'] = True
                self.cstop['n_ticks'] += 1
                qp, sp = self._pred(q)
                ub = self.u_last[:3] + np.clip(-self.u_last[:3], -B.A_BASE * self.cfg.dt, B.A_BASE * self.cfg.dt)
                seed = self.q_ik_prev if self.q_ik_prev is not None else qp[3:]
                ua_, res_, q1 = self._arm_track(qp, sp, np.asarray(self.pipe[-1][:3], float).copy(), ub, tr + D_STEPS * PHYS + DT, seed)
                if ua_ is None:
                    self._abort('contact_stop_ik_fail', resid=res_)
                else:
                    u = np.r_[ub, ua_]
                    bad = self._lim_bad(u, self.u_prev)
                    if bad:
                        self._abort('contact_stop_cmd_limit', bad=bad)
                    else:
                        self.q_ik_prev = q1
                        self.u_last = u
            elif self.stop is not None:
                da = np.r_[B.A_BASE, np.full(6, C0.A_ARM)] * self.cfg.dt
                u = self.u_last + np.clip(-self.u_last, -da, da)
                self.u_last = u
            self.u_prev = self.u_last.copy()
            self.u_body_prev = self.u_last[:3].copy()
            self.cmd_log.append((round(self.t, 3), self.phase if self.stop is None else 'STOP', self.u_last.copy()))
            self.u_from_base, self.ramp_j = np.asarray(self.pipe[-1][:3], float).copy(), 0
            self.tick_log.append((round(self.t, 3), round((time.perf_counter() - w_t) * 1e3, 3), self.stop is not None))
        cmd = XO.ramp_cmd(self.u_from_base, self.u_last, self.ramp_j)
        self.ramp_j += 1
        self.pipe.append(cmd)
        ua = self.pipe.pop(0)
        nxt = self.s + PHYS * ua[3:]
        if np.any(nxt < self.lim_lo) or np.any(nxt > self.lim_hi):
            nxt = np.clip(nxt, self.lim_lo, self.lim_hi)
            self._stop('setpoint_limit')
        self.s = nxt
        if self.stop is not None and self.stop_ok is None:
            if self.cstop['used']:                                     # r4：接觸停法停穩＝底盤命令與管線為零、手臂命令 ≤ CSTOP.arm_cmd、速率低
                cmd_ok = not np.any(self.u_last[:3]) and not any(np.any(u_[:3]) for u_ in self.pipe) and float(np.abs(self.u_last[3:]).max()) <= CSTOP['arm_cmd'] and all(float(np.abs(u_[3:]).max()) <= CSTOP['arm_cmd'] for u_ in self.pipe)
            else:
                cmd_ok = not any(np.any(u_) for u_ in self.pipe) and not np.any(self.u_last)
            still = (self.rate['arm'] <= 0.02 and self.rate['base'] <= 0.005 and self.rate['yaw'] <= 0.01 and cmd_ok)
            self.still_n = self.still_n + 1 if still else 0
            if self.still_n * PHYS >= 0.3 - 1e-9:
                self.stop_ok = dict(t=round(self.t, 3), ok=True)
        self.n_step += 1
        if self.stop is not None:
            self.ready_now, self.ready_ver = False, None
        if self.abort is not None:                                   # r3：中止當步起不出命令
            return dict(arm_target=self.s.copy(), base_twist=np.zeros(3), finger=self.finger_q, ready=False, ready_ver=None, stopped=True, abort=True)
        return dict(arm_target=self.s.copy(), base_twist=ua[:3].copy(), finger=self.finger_q, ready=self.ready_now, ready_ver=self.ready_ver, stopped=self.stop is not None, abort=False)

    def accept_handover(self, ver):
        """下游接受交接：只接受當拍的 READY 版本（舊版本、已撤銷、已停止 ⇒ 拒絕）。接受後本控制器不再出命令。"""
        ok = bool(self.ready_now and self.stop is None and ver is not None and ver == self.ready_ver)
        self._ev('accept_handover' if ok else 'accept_refused', ver=ver, current=self.ready_ver)
        if ok:
            self.accepted, self.accept_t = True, self.t
        return ok

    def _phase(self, ph):
        self.phase, self.t_phase = ph, self.t
        self.ph_hist.append((ph, round(self.t - self.t0, 3)))
        self._ev('phase', to=ph)

    # 讓 door_exit_ctrl.finger_bar 可直接用（需要 T_WH_bind、finger_q）
    @property
    def T_WH_bind(self):
        return self.T_WH_model

    def report(self):
        Rr = np.array(self.rec, float) if self.rec else np.zeros((1, 12))
        return dict(fuse=self.fuse, stop=self.stop, abort=self.abort, stop_effect=self.stop_ok, phases=self.ph_hist, ready_t_rel=None if self.ready_t is None else round(self.ready_t - self.t0, 3),
                    fast_slip=self.fast,
                    contact_lock=None if self.lock is None else dict(t=self.lock['t'], phi_deg=round(math.degrees(self.lock['phi']), 4), est_ver=self.lock['est_ver']), lock_inconsistency_max=self.lock_incons_max, contact_stop=self.cstop,
                    accepted=self.accepted, accept_t_rel=None if self.accept_t is None else round(self.accept_t - self.t0, 3), n_ready_revoked=self.n_revoke, n_ready_ver=self.n_ready_ver,
                    min_clear={k: round(float(v), 4) for k, v in self.mins.items()}, limits={k: round(float(v), 4) for k, v in self.lim.items()}, events=self.events,
                    s_along_end=round(float(Rr[-1, 2]), 4))
