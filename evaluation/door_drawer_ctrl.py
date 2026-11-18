#!/usr/bin/env python3
"""穿門後：開抽屜→取筆→返回（Howard 11/18「從開門銜接開抽屜」、選定「開門→開抽屜→取筆→返回」；Claude 自行裁定，Codex 用量用完期間）。
同一個 gmpc_wb 實例（承接穿門控制器：mpc、設定點、在途管線、前一筆命令、導航前端輸入；不重設、不清管線、任務切換不強制停車）。
停止尾段、候選核對、步進骨架沿用 DoorPassCtrl／TablePickCtrl（每拍候選＋同一停法到停穩的全身淨空核對；淨空模型含抽屜方盒，開度＝本控制器的量測估計）。
任務段（self.task）與階段：
  handle   APPROACH → CLOSE → CONFIRM（同取物：兩指接觸力＋未全閉，連續 0.3 s）⇒ dr_attach（執行端 FixedJoint 夾爪↔抽屜，理想抓持，同門把）
           PULL     底盤參考與 TCP 參考同沿拉出方向平移 OPEN_M（最小急動 T_PULL；全身拉開）；開度估計＝TCP 沿拉出方向位移（運動學，非真值）。
                    完成＝名目時間到且 TCP 對參考 ≤ 5 mm ⇒ dr_release（附開度估計）。
           UNBIND   命令零，等執行端解除 FixedJoint 的回報（dr_unbound）。
           FINGERS  手指張開，兩指實測 ≥ 8.5 mm。
           RETREAT  TCP 沿接近軸反向退 S_RET、並在 35% 名目時間後開始抬高 RET_LIFT（越過面板頂；底盤錨定）⇒ 事件 pick_dr_released（開度）⇒ WAIT（保持）等監督送觀察封包。
  observe  APPROACH（目標＝俯視觀察位姿、底盤收斂到觀察底盤位）→ OBSERVE（保持；事件 pick_dr_observing）等監督送夾筆封包。
  marker   同 TablePickCtrl：APPROACH → CLOSE → CONFIRM（⇒ attach，執行端 FixedJoint 夾爪↔筆）→ LIFT → VERIFY（⇒ pick_lifted）→ CARRY → RETURN。
不宣稱：把手位置用場景登錄（已知家具）＋理想里程，不是視覺；抓持（把手、筆）在夾住證據成立後為理想連接；淨空模型不含筆本體。"""
import math
import os

import numpy as np

import door_pick_ctrl as PK
import door_pass_ctrl as P
import drawer_scene as D

G, XO, XE, K, B, C0, S, W = P.G, P.XO, P.XE, P.K, P.B, P.C0, P.S, P.W
NPH, PHYS, DT, D_STEPS = P.NPH, P.PHYS, P.DT, P.D_STEPS
_T3, _log_R, _exp_R = P._T3, P._log_R, P._exp_R
D.install(W)
T_PULL = float(os.environ.get('DR_T_PULL', '4.0'))
PULL_TOL = 0.005
WAIT_TO_S = float(os.environ.get('DR_WAIT_TO_S', '20.0'))             # 等監督封包（選位在監督背景執行緒）
OBS_TOL_M, OBS_TOL_DEG = 0.010, 5.0
RET_LIFT = float(os.environ.get('DR_RET_LIFT', '0.18'))              # 放手退出時另抬高（越過抽屜面板頂再去俯視；s39 直接混接 ⇒ 手指掃到面板被停）
T_RET_DR = float(os.environ.get('DR_T_RET', '2.5'))
OBS_QD = float(os.environ.get('DR_OBS_QD', '0.6'))                    # 俯視關節空間移動峰速上限（rad/s）
W_Q_OBS = float(os.environ.get('DR_W_Q_OBS', '5.0'))
W_Q_APP = float(os.environ.get('DR_W_Q_APP', '0.0'))                  # 把手接近的關節姿態引導權重（s52：1.0 時引導解與實際底盤位不符 ⇒ 接近誤差 110 mm 未到位；預設關）
OBS_BUMP = float(os.environ.get('DR_OBS_BUMP', '0.05'))              # 俯視接近路徑上拱（m）
CARRY_POS = np.array([float(x) for x in os.environ.get('DR_CARRY_POS', '0.30,0.0,0.70').split(',')])   # 持筆收臂位（底盤座標）：原收臂點 (0.36, 0, 0.588) 在拉開的抽屜正上方、低於面板頂 0.60 ⇒ 返回後退會撞面板；改高 0.70（離線 σ 0.117）
T_CARRY_DR = float(os.environ.get('DR_T_CARRY', '4.0'))
LIFT_DELAY = 0.35                                                     # 抬升自退出名目時間的 35% 起（先水平離開橫桿）


class DrawerCtrl(PK.TablePickCtrl):
    def __init__(self, src, t0, app):
        """app（odom）：T_grasp＝把手抓取 TCP（已含越過桿心量）、base_goal、T、T_b、t_l0、pull（拉出方向 x,y）、open_m。"""
        super().__init__(src, t0, app)
        self.T_grasp = np.asarray(app['T_grasp'], float).copy()      # 把手：不再加圓柱的 GRASP_DEPTH（越過桿心量已在資產 tcp_offset_along_tool_z）
        self.task = 'handle'
        self.pull = np.r_[np.asarray(app['pull'], float)[:2], 0.0]
        self.pull = self.pull / max(float(np.linalg.norm(self.pull)), 1e-9)
        self.open_m = float(app.get('open_m', 0.20))
        self.T_pull = float(app.get('T_pull', T_PULL))
        self.dr_attach, self.dr_release, self.dr_unbound, self.q_open = False, False, False, 0.0
        self.T_p0, self.b_p0, self.T_hold, self.T_rel_r0, self.dvec_b, self.T_obs0 = None, None, None, None, None, None
        self.q_obs, self.q_obs0, self.q_arm_now = None, None, None
        D.set_open(0.0)
        self.q_h = np.asarray(app['q_grasp'], float).copy() if app.get('q_grasp') is not None else None   # 把手抓取關節解（選位時算；s51 底盤偏航差 0.47 rad 時手腕走 j5 −1.76 ⇒ 自身淨空 4.6 mm）
        self.q_h0 = None
        if self.q_h is not None and W_Q_APP > 0.0:
            self.app_w_q = W_Q_APP
            self.app_qpost = self._handle_qpost
        self._ev('pick_dr_start', T_grasp=[round(float(x), 4) for x in self.T_grasp[:3, 3]], pull=[round(float(x), 4) for x in self.pull[:2]], open_m=self.open_m)

    # ---- 監督送的下一段封包（observe／marker）
    def next_app(self, app, t0):
        kind = app.get('kind')
        self.app_qpost = None                                         # 把手接近的姿態引導只用在把手段
        if kind == 'observe' and self.task == 'handle' and self.phase == 'WAIT':
            self.task = 'observe'
            self.T_grasp = np.asarray(app['T_obs'], float).copy()
            self.T_obs0 = None if self.T_hold is None else self.T_hold.copy()   # 起點＝退出終點（世界）
            self.q_obs = np.asarray(app['q_obs'], float).copy() if app.get('q_obs') is not None else None   # 俯視構型（關節空間過去；s42）
            self.q_obs0 = None
        elif kind == 'marker' and self.task == 'observe' and self.phase == 'OBSERVE':
            self.task = 'marker'
            self.T_grasp = np.asarray(app['T_grasp'], float).copy()   # 已含越過筆軸 4 mm（drawer_reach.grasp_T；s44 這裡再加一次 ⇒ TCP 在筆軸下 9.5 mm，替身判未夾到）
        else:
            self._ev('pick_dr_app_ignored', kind=kind, task=self.task, phase=self.phase)
            return
        self.app_prof = dict(base_goal=list(app['base_goal']), door=None)
        self.T_app, self.T_b, self.t_l0 = float(app['T']), float(app['T_b']), float(app['t_l0'])
        if kind == 'observe' and self.q_obs is not None and self.q_arm_now is not None:   # 關節空間時長：最大關節位移 ÷ OBS_QD（最小急動峰速 1.875×）
            self.T_app = max(self.T_app, 1.875 * float(np.abs(self.q_obs - self.q_arm_now).max()) / OBS_QD + PK.T_SETTLE)
        self.T_rel0 = None
        self.t = float(t0)
        self._goto('APPROACH', task=self.task, base_goal=[round(float(x), 4) for x in app['base_goal']], T=round(self.T_app, 3))

    def _obs_q(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / max(self.T_app - PK.T_SETTLE, 1e-6), 0.0), 1.0))[0]
        return (1.0 - u) * self.q_obs0 + u * self.q_obs

    def _obs_rel(self, t_abs):
        return K.fk(np.r_[np.zeros(3), self._obs_q(t_abs)], 'link_tcp')    # 相對底盤（關節直線插值的正向運動學）

    def _app_target(self, t_abs, Xk):
        if self.task != 'observe':
            return super()._app_target(t_abs, Xk)
        if self.T_obs0 is None:
            return self.T_grasp
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / max(self.T_app - PK.T_SETTLE, 1e-6), 0.0), 1.0))[0]   # 世界座標：退出終點 → 俯視位（不掛在移動底盤上；s40 掛底盤 ⇒ 底盤前進把夾爪帶向面板）
        A, Bm = self.T_obs0, self.T_grasp
        T = np.eye(4)
        T[:3, 3] = (1 - u) * A[:3, 3] + u * Bm[:3, 3] + OBS_BUMP * math.sin(math.pi * u) * np.array([0.0, 0.0, 1.0])
        T[:3, :3] = A[:3, :3] @ _exp_R(u * _log_R(A[:3, :3].T @ Bm[:3, :3]))
        return T

    def _carry_target(self, t_abs):
        if self.task != 'marker':
            return super()._carry_target(t_abs)
        r = min(max((t_abs - self.t_ph0) / T_CARRY_DR, 0.0), 1.0)
        uz = XE.minjerk(min(r / 0.6, 1.0))[0]
        ux = XE.minjerk(min(max((r - 0.4) / 0.6, 0.0), 1.0))[0]         # 先抬高、後水平（s47 直線斜移 ⇒ link_eef 擦面板頂 5.0 mm 被停）
        A = self.T_rel_s0
        T = np.eye(4)
        T[:2, 3] = (1 - ux) * A[:2, 3] + ux * CARRY_POS[:2]
        T[2, 3] = (1 - uz) * A[2, 3] + uz * CARRY_POS[2]                # 只移位置（同圓柱）；姿態保持朝下
        T[:3, :3] = A[:3, :3]
        return T

    def _T_line(self, s_):
        if self.task == 'observe':
            return self.T_grasp.copy()                                # 觀察位：不走抓取線
        return super()._T_line(s_)

    def _handle_qpost(self, t_abs):
        if self.q_h0 is None:
            self.q_h0 = self.q_arm_now.copy() if self.q_arm_now is not None else self.q_h.copy()
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / max(self.T_app - PK.T_SETTLE, 1e-6), 0.0), 1.0))[0]
        return (1.0 - u) * self.q_h0 + u * self.q_h

    def _pull_s(self, t_abs):
        return self.open_m * XE.minjerk(min(max((t_abs - self.t_ph0) / self.T_pull, 0.0), 1.0))[0]

    def _pull_target(self, t_abs):
        T = self.T_p0.copy()
        T[:3, 3] = T[:3, 3] + self._pull_s(t_abs) * self.pull
        return T

    def _ret_target(self, t_abs):
        r = min(max((t_abs - self.t_ph0) / T_RET_DR, 0.0), 1.0)
        ux = XE.minjerk(r)[0]
        uz = XE.minjerk(min(max((r - LIFT_DELAY) / (1.0 - LIFT_DELAY), 0.0), 1.0))[0]
        T = self.T_rel_r0.copy()
        T[:3, 3] = T[:3, 3] + ux * self.dvec_b + uz * np.array([0.0, 0.0, RET_LIFT])
        return T

    def _advance(self, q):
        el = self.t - self.t_ph0
        if self.task == 'handle':
            if self.phase == 'CONFIRM':
                self.n_ev = self.n_ev + 1 if self._held() else 0
                if self.n_ev * DT >= PK.EVID_S - 1e-9:
                    self.T_p0 = K.fk(q, 'link_tcp')
                    self.b_p0 = np.asarray(q[:3], float).copy()
                    self.dr_attach = True                             # 夾住證據成立 ⇒ 執行端 FixedJoint 夾爪↔抽屜（理想抓持，同門把）
                    self._goto('PULL', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas],
                               force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
                elif el > PK.EVID_TO_S:
                    self._stop('handle_grasp_evidence_timeout', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas],
                               force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
                return
            if self.phase == 'PULL':
                Tc = K.fk(q, 'link_tcp')
                self.q_open = max(0.0, float(np.dot(Tc[:3, 3] - self.T_p0[:3, 3], self.pull)))
                D.set_open(self.q_open)
                ep = float(np.linalg.norm(Tc[:3, 3] - self._pull_target(self.t_ph0 + self.T_pull)[:3, 3]))
                if el >= self.T_pull and ep <= PULL_TOL:
                    self.b_anchor = np.asarray(q[:3], float).copy()
                    self.T_hold = Tc.copy()
                    self.dr_release = True
                    self._ev('pick_dr_pulled', q_open=round(self.q_open, 4), err_mm=round(ep * 1e3, 2))
                    self._goto('UNBIND')
                elif el > self.T_pull + PK.PH_TO_S:
                    self._stop('pull_not_reached', err_mm=round(ep * 1e3, 2), q_open=round(self.q_open, 4))
                if not self._held():
                    self._stop('lost_handle_during_pull')
                return
            if self.phase == 'UNBIND':
                if self.dr_unbound:
                    self.finger_cmd = P.FINGER_OPEN
                    self._goto('FINGERS')
                elif el > P.UNBIND_TO_S:
                    self._stop('drawer_unbind_ack_timeout')
                return
            if self.phase == 'FINGERS':
                fm = self.finger_meas
                if fm is not None and min(fm) >= P.FINGER_OK:
                    W.FINGER_Q = float(np.mean(fm))
                    Tc = K.fk(q, 'link_tcp')
                    self.T_rel_r0 = np.linalg.inv(_T3(q[:3])) @ Tc
                    self.dvec_b = _T3(self.b_anchor)[:3, :3].T @ (-P.S_RET * Tc[:3, 2])   # 沿接近軸（工具 z）反向
                    self._goto('RETREAT', finger=[round(float(x), 5) for x in fm])
                elif el > P.FINGER_TO_S:
                    self._stop('finger_open_timeout', finger=None if fm is None else [round(float(x), 5) for x in fm])
                return
            if self.phase == 'RETREAT':
                T_now = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
                d_ = T_now[:3, 3] - self.T_rel_r0[:3, 3]
                moved = float(np.dot(d_, self.dvec_b)) / P.S_RET
                lifted = float(d_[2])
                if el >= T_RET_DR and moved >= 0.9 * P.S_RET and lifted >= 0.9 * RET_LIFT:
                    self.T_hold = K.fk(q, 'link_tcp')
                    self._ev('pick_dr_released', q_open=round(self.q_open, 4), retreat_m=round(moved, 4), lift_m=round(lifted, 4))
                    self._goto('WAIT')
                elif el > T_RET_DR + P.PH_EXTRA_TO_S:
                    self._stop('retreat_not_reached', retreat_m=round(moved, 4), lift_m=round(lifted, 4))
                return
            if self.phase == 'WAIT':
                if el > WAIT_TO_S:
                    self._stop('observe_app_timeout')
                return
        if self.task == 'observe':
            if self.phase == 'APPROACH':
                if self.T_rel0 is None:
                    self.T_rel0 = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
                if self.q_obs is not None and self.q_obs0 is None:
                    self.q_obs0 = np.asarray(q[3:], float).copy()
                Tc = K.fk(q, 'link_tcp')
                if self.q_obs is not None:                            # 關節空間俯視：完成判定用相對底盤誤差（底盤由導航收斂）
                    Tt = K.fk(np.r_[np.zeros(3), self.q_obs], 'link_tcp'); Tr = np.linalg.inv(_T3(q[:3])) @ Tc
                    ep = float(np.linalg.norm(Tr[:3, 3] - Tt[:3, 3])); er = math.degrees(float(np.linalg.norm(_log_R(Tt[:3, :3].T @ Tr[:3, :3]))))
                else:
                    ep = float(np.linalg.norm(Tc[:3, 3] - self.T_grasp[:3, 3]))
                    er = math.degrees(float(np.linalg.norm(_log_R(self.T_grasp[:3, :3].T @ Tc[:3, :3]))))
                if el >= self.T_app and ep <= OBS_TOL_M and er <= OBS_TOL_DEG:
                    self.b_anchor = np.asarray(q[:3], float).copy()
                    self._ev('pick_dr_observing', ep_mm=round(ep * 1e3, 2), er_deg=round(er, 2), base=[round(float(x), 4) for x in q[:3]])
                    self._goto('OBSERVE')
                elif el > self.T_app + PK.PH_TO_S:
                    self._stop('observe_not_reached', ep_mm=round(ep * 1e3, 2), er_deg=round(er, 2))
                return
            if self.phase == 'OBSERVE':
                if el > WAIT_TO_S:
                    self._stop('marker_app_timeout')
                return
        if self.task == 'marker' and self.phase == 'CARRY':          # 持筆收臂：完成判定用本段時長（兩段式剖面）
            T_now = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
            ep = float(np.linalg.norm(T_now[:3, 3] - self._carry_target(self.t_ph0 + T_CARRY_DR)[:3, 3]))
            if not self._held():
                self._stop('lost_during_carry')
            elif el >= T_CARRY_DR and ep <= 0.010:
                self._ev('pick_dr_carried', carry_err_mm=round(ep * 1e3, 2))   # 監督據此送返回目標（伺服器轉發只帶事件名）
                self._goto('RETURN', carry_err_mm=round(ep * 1e3, 2))
            elif el > T_CARRY_DR + PK.PH_TO_S:
                self._stop('carry_not_reached', carry_err_mm=round(ep * 1e3, 2))
            return
        super()._advance(q)                                          # handle 的 APPROACH／CLOSE；marker 其餘段（同圓柱）

    def _solve(self, q):
        if self.task == 'marker' and self.phase == 'CARRY':          # 持筆收臂用近距權重（s48：W_FAR 姿態追蹤弱 ⇒ TCP 不動而夾爪傾向面板，link_eef 4.8 mm 被停）
            w_far = C0.W_FAR
            C0.W_FAR = C0.W_NEAR
            try:
                return super()._solve(q)
            finally:
                C0.W_FAR = w_far
        if self.task == 'marker' or (self.task == 'handle' and self.phase in ('APPROACH', 'CLOSE')):
            return super()._solve(q)
        qp, sp = self._pred(q)
        t_eff = self.t + D_STEPS * PHYS
        X, xi, obs, umin0, src = G.MovingGraspCtrl.nav_window(self, q, qp)
        if obs is None:
            return None, 'nav_obstacles_stale'
        Xp = [(float(Xk[0, 2]), float(Xk[1, 2]), math.atan2(float(Xk[1, 0]), float(Xk[0, 0]))) for Xk in X]
        tcp_rel, q_post = False, None
        if self.task == 'observe' and self.q_obs is not None and self.q_obs0 is not None and self.phase in ('APPROACH', 'OBSERVE'):
            if self.phase == 'APPROACH':
                Ts = [self._obs_rel(t_eff + k * DT) for k in range(XO.N + 1)]
                q_post = np.array([self._obs_q(t_eff + (k + 1) * DT) for k in range(XO.N)])
            else:
                X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
                xi = [np.zeros(3)] * (XO.N + 1)
                Ts = [K.fk(np.r_[np.zeros(3), self.q_obs], 'link_tcp')] * (XO.N + 1)
                q_post = np.tile(self.q_obs, (XO.N, 1))
                umin0, src = float('nan'), 'anchor'
            tcp_rel, w_p = True, (C0.W_FAR if self.phase == 'APPROACH' else C0.W_NEAR)
        elif self.task == 'observe' and self.phase == 'APPROACH':
            Ts = [self._app_target(t_eff + k * DT, Xp[k]) for k in range(XO.N + 1)]
            w_p = C0.W_FAR
        else:
            if self.phase == 'PULL':                                 # 全身拉開：底盤參考與 TCP 參考同沿拉出方向
                ss = [self._pull_s(t_eff + k * DT) for k in range(XO.N + 1)]
                X = [XO.from_xytheta(self.b_p0[0] + s_ * self.pull[0], self.b_p0[1] + s_ * self.pull[1], self.b_p0[2]) for s_ in ss]
                xi = [np.zeros(3)] * (XO.N + 1)
                Ts = [self._pull_target(t_eff + k * DT) for k in range(XO.N + 1)]
            else:
                X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
                xi = [np.zeros(3)] * (XO.N + 1)
                if self.phase == 'RETREAT':
                    Ts = [self._ret_target(t_eff + k * DT) for k in range(XO.N + 1)]
                    tcp_rel = True
                elif self.phase in ('OBSERVE',):
                    Ts = [self.T_grasp] * (XO.N + 1)
                else:                                                 # CONFIRM（handle）、WAIT：保持
                    Ts = [self.T_hold if self.T_hold is not None else self.T_grasp] * (XO.N + 1)
            umin0, src = float('nan'), 'anchor' if self.phase != 'PULL' else 'pull'
            w_p = C0.W_NEAR if self.phase in ('CONFIRM', 'PULL', 'OBSERVE') else C0.W_FAR
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts), w_p=w_p,
                   lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper, tcp_rel_base=tcp_rel, qp_solver='gi', **B.W_ARM)
        if q_post is not None:
            arm.update(q_post=q_post, w_q=W_Q_OBS)
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
        self.wg_log.append((round(self.t, 3), self.task + ':' + self.phase, round((time.perf_counter() - w0) * 1e3, 2), str(r.status), src, len(obs), int(r.cbf_active)))
        if not np.isfinite(np.asarray(r.u_opt, float)).all():
            return None, 'wgmpc_nonfinite'
        ck = dict(r.arm_check or {})
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None or r.status != 'solved':
            return None, 'wgmpc:%s/%s/%s' % (r.status, ck.get('fallback'), ck.get('failed'))
        return np.r_[np.asarray(r.u_opt, float), np.asarray(r.u_arm_seq[0], float)], src


    def step(self, t, q, phi_meas, finger_meas=None, finger_force=None):
        self.q_arm_now = np.asarray(q[3:], float).copy()                  # 俯視關節空間時長用
        return super().step(t, q, phi_meas, finger_meas=finger_meas, finger_force=finger_force)
