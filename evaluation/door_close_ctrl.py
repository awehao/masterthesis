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
T_SWING = float(os.environ.get('DC_T_SWING', '14.0'))
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
        super().__init__(src, t0, app)
        self.motm = None
        self.app_tol_z, self.app_v = None, None
        self.T_grasp = np.asarray(app['T_grasp'], float).copy()
        self.phi_c = float(app['phi_c'])
        self.q_c = np.asarray(app['q_c'], float).copy()
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

    def _phi_ref(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / T_SWING, 0.0), 1.0))[0]
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
                self.b_s0, self.q_s0 = np.asarray(q[:3], float).copy(), np.asarray(q[3:], float).copy()
                self.T_rel_s = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
                self.phi_c = float(self.phi_now)                       # 綁定瞬間實測門角起算
                self._goto('SWING', phi_deg=round(math.degrees(self.phi_c), 2))
            elif el > P.UNBIND_TO_S:
                self._stop('door_bind_ack_timeout')
            return
        if self.phase == 'SWING':
            if el >= T_SWING and float(self.phi_now) <= PHI_GOAL + PHI_TOL:
                self.release = True
                self.b_anchor = np.asarray(q[:3], float).copy()
                self._ev('pick_door_swung', phi_deg=round(math.degrees(float(self.phi_now)), 2))
                self._goto('UNBIND')
            elif el > T_SWING + PK.PH_TO_S:
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
            X = [XO.from_xytheta(*corotate_base(self.b_s0, self._phi_ref(t_eff + k * DT) - self.phi_c)) for k in range(XO.N + 1)]
            xi = [np.zeros(3)] * (XO.N + 1)
            Ts = [self.T_rel_s] * (XO.N + 1)
            q_post = np.tile(self.q_s0, (XO.N, 1))
            umin0, src = float('nan'), 'corotate'
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
        self.wg_log.append((round(self.t, 3), 'door_close:' + self.phase, round((time.perf_counter() - w0) * 1e3, 2), str(r.status), src, len(obs), int(r.cbf_active)))
        if not np.isfinite(np.asarray(r.u_opt, float)).all():
            return None, 'wgmpc_nonfinite'
        ck = dict(r.arm_check or {})
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None or r.status != 'solved':
            return None, 'wgmpc:%s/%s/%s' % (r.status, ck.get('fallback'), ck.get('failed'))
        return np.r_[np.asarray(r.u_opt, float), np.asarray(r.u_arm_seq[0], float)], src
