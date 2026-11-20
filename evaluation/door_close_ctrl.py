#!/usr/bin/env python3
"""回程關門（Howard 11/19「都差不多可以開始做關門了」「回程推開穿過後再關門」）：從臥室側夾回門扇把手、整車繞鉸鏈共轉把門帶回關閉、放手退出、返回。
同一個 gmpc_wb 實例（承接前一個取物控制器：mpc、設定點、在途管線、前一筆命令、導航前端輸入）；停止尾段、候選核對沿用 DoorPassCtrl（淨空含門扇，門角＝實測）。
把手位姿：去程開門綁定時的視覺把手位姿 T_WH0（門角 PHI0）＋門機構（鉸鏈軸）＋**本段門角**（門扇深度偵測 door_leaf_detect，監督融合後隨封包給），
  T_grasp＝W.handle_pose(T_WH0, φ_c)·T_HE（T_HE＝去程綁定時的實測抓持變換，同一種抓法）。
階段：
  UNFOLD    關節空間展開到封包給的抓持構型 q_c（＝去程開門結束的抓持構型繞鉸鏈共轉到 φ_c；手臂關節相同）；底盤照導航前端往抓持底盤位。
  APPROACH  同 TablePickCtrl（TCP 沿接近軸到 T_grasp，到位 ≤5 mm／3°）。
  CLOSE／CONFIRM  手指閉合；把手夾持證據同抽屜（開度帶＋合力）⇒ door_bind（執行端 FixedJoint 夾爪↔門扇）。
  BIND      等執行端綁定生效回報。
  SWING     共轉關門：門角參考 φ_ref(t)＝φ_c → PHI_GOAL 最小急動（T_SWING）；底盤參考＝綁定瞬間底盤位姿繞鉸鏈（原點、+z）轉 −(φ_ref−φ_c)；
            手臂相對底盤保持（tcp_rel_base、q_post＝綁定瞬間構型）⇒ 剛體共轉、閉鏈自然成立（離線：第二十六趟 85° 構型共轉到 2°，淨空／輪速／限位全過）。
            完成＝名目時間到且實測門角 ≤ PHI_GOAL＋PHI_TOL ⇒ door_release。
  UNBIND／FINGERS／RETREAT／STOW  同穿門段（解除回報、張爪 ≥8.5 mm、沿接近軸退 0.10、關節收臂）⇒ 事件 pick_door_closed ⇒ RETURN（監督送起點目標）。
不宣稱：門扇開度由視覺量（形狀模型＝已知門）；抓持與綁定為理想連接；共轉關門手臂不動，非最佳化軌跡；門無閂，放手後可能漂移（另記真值）。"""
import math
import os

import numpy as np

import door_pick_ctrl as PK
import door_pass_ctrl as P

G, XO, XE, K, B, C0, S, W = P.G, P.XO, P.XE, P.K, P.B, P.C0, P.S, P.W
NPH, PHYS, DT, D_STEPS = P.NPH, P.PHYS, P.DT, P.D_STEPS
_T3, _log_R, _exp_R = P._T3, P._log_R, P._exp_R
PHI_GOAL = math.radians(float(os.environ.get('DC_PHI_GOAL_DEG', '2.0')))
PHI_TOL = math.radians(float(os.environ.get('DC_PHI_TOL_DEG', '1.0')))
CL_R_TOL = float(os.environ.get('DC_CL_R_TOL', '0.008'))             # 共轉停止軌跡閉鏈半徑容差（m）
T_SWING_MIN = float(os.environ.get('DC_T_SWING_MIN', '3.0'))          # 共轉時長下限（s）；實際時長在綁定瞬間依輪速／輪加速度上限算（Howard 11/20：不寫死 14 s）
SWING_V_FRAC, SWING_A_FRAC = float(os.environ.get('DC_SWING_V_FRAC', '0.8')), float(os.environ.get('DC_SWING_A_FRAC', '0.3'))   # 留給追蹤與停止軌跡的餘裕
UNF_QD = float(os.environ.get('DC_UNF_QD', '0.6'))
H_FQ, H_FQ_TOL, H_FSUM = 0.0018, 0.0006, 1.0                         # 門把手同 26 mm 橫桿（同抽屜判準）


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def corotate_base(b, dphi):
    """底盤位姿繞鉸鏈（原點）跟著門轉：門角變化 dphi（門扇轉 −dphi）⇒ 位置轉 −dphi、偏航 −dphi。"""
    R = _Rz(-dphi)
    p = R @ np.array([float(b[0]), float(b[1]), 0.0])
    return np.array([p[0], p[1], float(b[2]) - dphi])


class DoorCloseCtrl(PK.TablePickCtrl):
    def __init__(self, src, t0, app):
        """app（odom）：T_grasp、base_goal、q_c（抓持構型手臂關節）、phi_c（本段門角，視覺）、T、T_b、t_l0。"""
        de = getattr(src, 'door_end', None)
        phi_c = float(app['phi_c'])
        b_c = corotate_base(de['b'], phi_c - de['phi'])               # 開門結束抓持構型繞鉸鏈共轉到視覺門角
        app = dict(app, base_goal=[float(x) for x in b_c], T_grasp=(W.handle_pose(src.T_WH0, phi_c) @ np.asarray(src.T_HE, float)).tolist())
        super().__init__(src, t0, app)
        self.motm = None
        self.door_geom = app.get('door')                               # 門幾何（里程）：綁定共轉段放進 app_prof ⇒ door_filter 排除門扇點（s97d）
        self.app_tol_z, self.app_v = None, None
        self.T_grasp = np.asarray(app['T_grasp'], float).copy()       # 不加圓柱的 GRASP_DEPTH（同一種門把抓法）
        self.phi_c = phi_c
        self.q_c = np.asarray(de['q'], float).copy()
        self.door_bind, self.door_bound, self.release, self.unbound = False, False, False, False
        self.q_u0, self.T_u, self.b_s0, self.q_s0, self.T_rel_s = None, None, None, None, None
        self.phase = 'UNFOLD'
        self._ev('pick_door_close_start', phi_c_deg=round(math.degrees(self.phi_c), 2), T_grasp=[round(float(x), 4) for x in self.T_grasp[:3, 3]])

    def _held_bar(self):
        f, fm = self.finger_force, self.finger_meas
        return f is not None and fm is not None and all(abs(float(x) - H_FQ) <= H_FQ_TOL for x in fm) and float(sum(f)) >= H_FSUM

    def _unf_q(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / max(self.T_u, 1e-6), 0.0), 1.0))[0]
        return (1.0 - u) * self.q_u0 + u * self.q_c

    def _swing_T(self, q, dphi):
        """最小急動門角剖面的最短時長：共轉時底盤體速度與門角速度 ω 成正比（位置繞鉸鏈轉、偏航 −ω），
        輪速＝WM·體速度 ⇒ 每單位 ω 的最大輪速 w_u；峰值 ω＝1.875Δφ/T、峰值 α＝5.774Δφ/T² 分別受輪速、輪加速度上限（乘餘裕比例）限制。"""
        LIM = P.G.LIM
        p = np.asarray(q[:2], float); th = float(q[2])
        vw = np.array([p[1], -p[0]])                                  # ω＝1 時世界速度（corotate_base：位置轉 −Δφ）
        c_, s_ = math.cos(th), math.sin(th)
        tw = np.array([c_ * vw[0] + s_ * vw[1], -s_ * vw[0] + c_ * vw[1], -1.0])
        w_u = float(np.abs(np.asarray(LIM.WM) @ tw).max())
        om_max, al_max = SWING_V_FRAC * float(LIM.W_LIM) / w_u, SWING_A_FRAC * float(LIM.WA_LIM) / w_u
        d = abs(float(dphi))
        return max(T_SWING_MIN, 1.875 * d / om_max, math.sqrt(5.774 * d / al_max))

    def _phi_ref(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / self.T_swing, 0.0), 1.0))[0]
        return self.phi_c + u * (PHI_GOAL - self.phi_c)

    def _advance(self, q):
        el = self.t - self.t_ph0
        if self.phase == 'UNFOLD':
            if self.q_u0 is None:
                self.q_u0 = np.asarray(q[3:], float).copy()
                self.T_u = 1.875 * float(np.abs(self.q_c - self.q_u0).max()) / UNF_QD + 0.5
            if el >= self.T_u and float(np.abs(np.asarray(q[3:], float) - self.q_c).max()) <= 0.03:
                self.T_rel0 = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
                self._goto('APPROACH', unfold_s=round(el, 2))
            elif el > self.T_u + PK.PH_TO_S:
                self._stop('door_close_unfold_not_reached')
            return
        if self.phase == 'CONFIRM':
            self.n_ev = self.n_ev + 1 if self._held_bar() else 0
            if self.n_ev * DT >= PK.EVID_S - 1e-9:
                self.door_bind = True                                 # 執行端 FixedJoint 夾爪↔門扇（理想抓持，同去程）
                self.b_anchor = np.asarray(q[:3], float).copy()
                self._goto('BIND', finger=[round(float(x), 5) for x in self.finger_meas], force=[round(float(x), 3) for x in self.finger_force])
            elif el > PK.EVID_TO_S:
                self._stop('door_close_grasp_evidence_timeout', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas])
            return
        if self.phase == 'BIND':
            if self.door_bound:
                Tc_ = K.fk(q, 'link_tcp')                              # 綁定瞬間以實際抓持重新對準淨空模型的把手（同開門綁定；s87d：沿用去程把手位姿 ⇒ 共轉中夾爪殼–橫桿壓在 5 mm 門檻）
                T_WH_now = Tc_ @ np.linalg.inv(np.asarray(self.T_HE, float))
                self.T_WH0 = W.Rz4(float(self.phi_now) - W.PHI0) @ T_WH_now
                if self.finger_meas is not None:
                    W.FINGER_Q = (float(self.finger_meas[0]), float(self.finger_meas[1]))
                self.b_s0, self.q_s0 = np.asarray(q[:3], float).copy(), np.asarray(q[3:], float).copy()
                self.T_rel_s = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
                self.phi_c = float(self.phi_now)                       # 綁定瞬間實測門角起算
                self.T_swing = self._swing_T(q, PHI_GOAL - self.phi_c)   # 依限制算的最短共轉時長
                if self.door_geom is not None:
                    self.app_prof = dict(base_goal=list(self.app_prof['base_goal']) if self.app_prof else None, door=self.door_geom)   # 綁定後門扇是操作對象：門扇點不進 CBF（門的碰撞由門專屬淨空核對負責，同去程）
                self._goto('SWING', phi_deg=round(math.degrees(self.phi_c), 2), T_swing=round(self.T_swing, 2))
            elif el > P.UNBIND_TO_S:
                self._stop('door_bind_ack_timeout')
            return
        if self.phase == 'SWING':
            if el >= self.T_swing and float(self.phi_now) <= PHI_GOAL + PHI_TOL:
                self.release = True
                self.b_anchor = np.asarray(q[:3], float).copy()
                self._ev('pick_door_swung', phi_deg=round(math.degrees(float(self.phi_now)), 2))
                self._goto('UNBIND')
            elif el > self.T_swing + PK.PH_TO_S:
                self._stop('door_close_not_reached', phi_deg=round(math.degrees(float(self.phi_now)), 2))
            if not self._held_bar():
                self._stop('lost_handle_during_close')
            return
        if self.phase == 'UNBIND':
            if self.unbound:
                self.finger_cmd = P.FINGER_OPEN
                self._goto('FINGERS')
            elif el > P.UNBIND_TO_S:
                self._stop('door_unbind_ack_timeout')
            return
        if self.phase == 'FINGERS':
            fm = self.finger_meas
            if fm is not None and min(fm) >= P.FINGER_OK:
                Tc = K.fk(q, 'link_tcp')
                self.T_rel_r0 = np.linalg.inv(_T3(q[:3])) @ Tc
                self.dvec_b = _T3(self.b_anchor)[:3, :3].T @ (-P.S_RET * Tc[:3, 2])
                self._goto('RETREAT')
            elif el > P.FINGER_TO_S:
                self._stop('finger_open_timeout')
            return
        if self.phase == 'RETREAT':
            T_now = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
            moved = float(np.dot(T_now[:3, 3] - self.T_rel_r0[:3, 3], self.dvec_b)) / P.S_RET
            if el >= P.T_RET and moved >= 0.9 * P.S_RET:
                self.binned = True                                    # RETURN 段不再要求持物證據
                self._ev('pick_door_closed', phi_deg=round(math.degrees(float(self.phi_now)), 2))
                self._bin_start(q, 'BIN_STOW')                        # 關節收臂回導航姿勢 ⇒ pick_stowed ⇒ RETURN
            elif el > P.T_RET + P.PH_EXTRA_TO_S:
                self._stop('door_close_retreat_not_reached', retreat_m=round(moved, 4))
            return
        super()._advance(q)                                           # APPROACH／CLOSE／BIN_STOW／RETURN

    def _sweep(self, q, u, extra, j0, u_from, rec):
        """綁定共轉中（SWING）：停止軌跡上門扇跟著夾爪轉（理想連接＝剛性），每個取樣點的門角由該點 TCP 繞鉸鏈（原點）的方位角換算
        φ_s＝φ_now−(ang_s−ang_now)（同 corotate_base 的方向）；s87d／s94d：沿用固定 φ_now ⇒ 停止軌跡中夾爪相對靜止的橫桿移動，
        隨共轉加速壓到 shell_bar 5 mm 門檻。其他段照原核對。"""
        if self.phase != 'SWING' or not self.door_bound:
            return super()._sweep(q, u, extra, j0, u_from, rec)
        P_ = P
        Q, _, _ = P_.XE.stop_traj_exit(q, self.s, self.pipe, u, u_from, j0, self.cfg, extra, rec=rec)
        if rec is not None and any(np.any(r_[0] < self.lim_lo - 1e-9) or np.any(r_[0] > self.lim_hi + 1e-9) for r_ in rec):
            return dict(why='setpoint_limit'), 0
        p0 = K.fk(np.asarray(q, float), 'link_tcp')[:3, 3]
        a0 = math.atan2(float(p0[1]), float(p0[0]))
        r0 = float(np.hypot(p0[0], p0[1]))
        n, prev = 0, None
        for qs in Q:
            if prev is not None and float(np.abs(qs - prev).max()) < 1e-7:
                continue
            prev = qs
            ps = K.fk(np.asarray(qs, float), 'link_tcp')[:3, 3]
            da = (math.atan2(float(ps[1]), float(ps[0])) - a0 + math.pi) % (2 * math.pi) - math.pi
            c = P_.CF.clear_door_fast(np.r_[qs, float(self.phi_now) - da], self.T_WH0)
            n += 1
            for k in P_.KEYS:
                self.clear_min[k] = min(self.clear_min[k], float(c[k]))
            bad = [k for k in P_.KEYS if k != 'shell_bar' and c[k] < P_.W.THR[k]]   # 綁定中橫桿與夾爪剛性相連，殼–桿距離恆定，不核（s123d：停止軌跡平移／轉動減速率不同 ⇒ 以位置換算門角時誤報 5 mm）
            r_dev = abs(float(np.hypot(ps[0], ps[1])) - r0)                 # 閉鏈相容：停止軌跡上夾爪離鉸鏈距離不變（否則綁定與鉸鏈互拉）
            if r_dev > CL_R_TOL:
                return dict(why='closure', r_dev_mm=round(r_dev * 1e3, 2)), n
            if bad:
                return dict(why='clearance', bad=bad, at={k: round(float(c[k]), 4) for k in P_.KEYS}, who={k: str((c.get('who') or {}).get(k)) for k in bad}), n
        return None, n

    def _solve(self, q):
        if self.phase in ('APPROACH', 'CLOSE') or self.phase.startswith('BIN_') or self.phase == 'RETURN':
            return super()._solve(q)
        qp, sp = self._pred(q)
        t_eff = self.t + D_STEPS * PHYS
        X, xi, obs, umin0, src = G.MovingGraspCtrl.nav_window(self, q, qp)
        if obs is None:
            return None, 'nav_obstacles_stale'
        q_post, tcp_rel, w_p = None, True, C0.W_NEAR
        if self.phase == 'UNFOLD':
            qs = [self._unf_q(t_eff + k * DT) for k in range(XO.N + 1)]
            Ts = [K.fk(np.r_[np.zeros(3), qk], 'link_tcp') for qk in qs]
            q_post, w_p = np.array(qs[1:]), C0.W_FAR
        elif self.phase == 'SWING':
            X = [XO.from_xytheta(*corotate_base(self.b_s0, self._phi_ref(t_eff + k * DT) - self.phi_c)) for k in range(XO.N + 2)]
            xi = []                                                   # 參考體速度＝相鄰參考位姿的相對運動 /DT（s102d：xi 全零 ⇒ 速度被壓、位置落後 0.2 m、偏航落後，夾爪相對把手轉）
            for k in range(XO.N + 1):
                T1 = np.linalg.inv(X[k]) @ X[k + 1]
                xi.append(np.array([T1[0, 2] / DT, T1[1, 2] / DT, math.atan2(T1[1, 0], T1[0, 0]) / DT]))
            X = X[:XO.N + 1]
            Ts = [self.T_rel_s] * (XO.N + 1)
            q_post = np.tile(self.q_s0, (XO.N, 1))
            umin0, src = float('nan'), 'corotate'
            obs = [o_ for o_ in obs if not o_.get('static')]               # 綁定共轉：牆／門框由全身淨空核對（env）負責，靜態牆點不進 CBF（s99d：牆點擋住底盤繞鉸鏈的位置、只有偏航跟上 ⇒ 夾爪相對把手轉、門又被拉開）；行人等動態障礙照舊
        else:                                                         # CONFIRM、BIND、UNBIND、FINGERS、RETREAT：底盤錨定
            X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
            xi = [np.zeros(3)] * (XO.N + 1)
            umin0, src = float('nan'), 'anchor'
            if self.phase == 'RETREAT':
                u_ = lambda tt: XE.minjerk(min(max((tt - self.t_ph0) / P.T_RET, 0.0), 1.0))[0]   # noqa: E731
                Ts = []
                for k in range(XO.N + 1):
                    T_ = self.T_rel_r0.copy(); T_[:3, 3] = T_[:3, 3] + u_(t_eff + k * DT) * self.dvec_b; Ts.append(T_)
            else:
                Tw = K.fk(q, 'link_tcp') if self.T_rel_s is None else _T3(self.b_anchor) @ self.T_rel_s
                Ts = [np.linalg.inv(_T3(self.b_anchor)) @ (self.T_grasp if self.phase in ('CONFIRM', 'BIND') else Tw)] * (XO.N + 1)
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts), w_p=w_p,
                   lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper, tcp_rel_base=tcp_rel, qp_solver='gi', **B.W_ARM)
        if q_post is not None:
            arm.update(q_post=q_post, w_q=5.0)
        mpc = self.mpc
        u_min_saved = np.asarray(mpc.cfg.u_min, float).copy()
        if math.isfinite(umin0):
            mpc.cfg.u_min = np.array([umin0, u_min_saved[1], u_min_saved[2]])
        import time
        w0 = time.perf_counter()
        try:
            r = mpc.solve(X_now=XO.from_xytheta(*qp[:3]), X_ref_win=np.array(X), xi_ref_win=np.array(xi), xi_prev=self.u_prev[:3], arm=arm, obstacles=obs)
        finally:
            mpc.cfg.u_min = u_min_saved
        self.wg_log.append((round(self.t, 3), 'door_close:' + self.phase, round((time.perf_counter() - w0) * 1e3, 2), str(r.status), src, len(obs), int(r.cbf_active), PK._obs_near(obs, qp), [round(float(X[0][0, 2] - qp[0]), 4), round(float(X[0][1, 2] - qp[1]), 4), round(float(math.atan2(X[0][1, 0], X[0][0, 0]) - qp[2] + math.pi) % (2 * math.pi) - math.pi, 4)]))   # 底盤參考誤差（x, y, 偏航）
        if not np.isfinite(np.asarray(r.u_opt, float)).all():
            return None, 'wgmpc_nonfinite'
        ck = dict(r.arm_check or {})
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None or r.status != 'solved':
            return None, 'wgmpc:%s/%s/%s' % (r.status, ck.get('fallback'), ck.get('failed'))
        return np.r_[np.asarray(r.u_opt, float), np.asarray(r.u_arm_seq[0], float)], src
