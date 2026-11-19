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
PICK_CARRY = os.environ.get('PICK_CARRY', '1') == '1'                 # Howard 11/18「夾到東西沒有收手臂，伸出來很危險」⇒ 預設收臂（理想連接後收臂不再滑落）
CARRY_V, CARRY_WZ = float(os.environ.get('PICK_CARRY_V', '0.2775')), float(os.environ.get('PICK_CARRY_WZ', '1.1327'))   # 持物底盤速度框（m/s、rad/s）；夾到後理想連接 ⇒ 預設＝硬體上限（不收緊）
RET_Y_MAX = float(os.environ.get('PICK_RET_Y_MAX', '-2.3'))           # 場景專用：起點 (1.5, −2.8)
PH_TO_S = 4.0
APP_TOL_Z = float(os.environ['PICK_APP_TOL_Z']) if os.environ.get('PICK_APP_TOL_Z') else None   # 圓柱（直立、側夾）接近完成：水平 ≤5 mm、沿圓柱軸（世界 z）≤ 此值；未設＝原 5 mm 球（第二十七趟 Isaac：東側伸遠，TCP 穩定低 9 mm、水平 <1 mm ⇒ approach_not_reached）
APP_V = float(os.environ['PICK_APP_V']) if os.environ.get('PICK_APP_V') else None   # 接近段底盤速度框（任務限制；s60／s68／s69：沿茶几邊 0.19 m/s 接近，停止軌跡底盤淨空 0.098–0.099 被停）；未設＝原行為
BIN = os.environ.get('D2_BIN', '0') == '1'                            # Howard 11/19「車子後面放一個盒子 放夾取物」：抬起確認後放進車尾收納盒（rear_bin.py）→ 收臂（相機朝前）→ 返回／下一件
BIN_QD = float(os.environ.get('D2_BIN_QD', '0.8'))                    # 放盒關節空間移動峰速上限（rad/s；最小急動峰速 1.875×）
BIN_MOVE = os.environ.get('D2_BIN_MOVE', '1') == '1'                  # 放盒／收臂中底盤照導航前端（限速框 BIN_V/BIN_WZ）；0＝底盤錨定
BIN_V, BIN_WZ = float(os.environ.get('D2_BIN_V', '0.15')), float(os.environ.get('D2_BIN_WZ', '0.5'))
DETACH_TO_S = 3.0
MOTM_CLOSE_LATE_S = float(os.environ.get('MOTM_CLOSE_LATE_S', '0.5'))   # 移動中夾取：名目到點後可延後閉爪的上限（底盤仍在走；超過 ⇒ motm_window_missed）
MOTM_W_Q = float(os.environ.get('MOTM_W_Q', '1.0'))                    # 移動中夾取接近段關節姿態引導權重（目前構型→選位抓取解 q_grasp；s74：只給 TCP 目標 ⇒ j2 滑到 −2.02、link4–link6 自身 4.5 mm）
MOTM_EVID_TO_S = float(os.environ.get('MOTM_EVID_TO_S', '0.8'))         # 移動中夾取：閉爪後夾住證據期限（窗口有限）
if BIN:
    import rear_bin as RBIN
    import door_clear_door_fast as _CF_



def xi_from_X(X, dt):
    """參考體速度＝相鄰參考位姿相對運動 /dt（最後一點沿用前一個）；移動參考不可配 xi＝0（s102d：速度被壓、追不上）。"""
    xi = []
    for k in range(len(X) - 1):
        T1 = np.linalg.inv(X[k]) @ X[k + 1]
        xi.append(np.array([T1[0, 2] / dt, T1[1, 2] / dt, math.atan2(T1[1, 0], T1[0, 0]) / dt]))
    xi.append(xi[-1].copy() if xi else np.zeros(3))
    return xi


def _obs_near(obs, q):
    """（只記錄）最近的非靜態障礙物：表面距（扣半徑）、速度 vx, vy（世界）；s91d 診斷用。"""
    best = None
    for o in obs or []:
        if o.get('static'):
            continue
        d = math.hypot(float(o['x']) - float(q[0]), float(o['y']) - float(q[1])) - float(o.get('radius', 0.0))
        if best is None or d < best[0]:
            best = (round(d, 3), round(float(o.get('vx', 0.0)), 3), round(float(o.get('vy', 0.0)), 3))
    return best

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
        self.T_HE, self.door_end = getattr(src, 'T_HE', None), getattr(src, 'door_end', None)   # 回程關門用（沿控制器鏈傳遞）
        self.t_halt, self.t_still = None, None
        self.T_grasp = np.asarray(app['T_grasp'], float).copy()
        self.T_grasp[:3, 3] = self.T_grasp[:3, 3] + GRASP_DEPTH * self.T_grasp[:3, 2]   # 同瓶子線：TCP 沿工具軸越過圓柱軸 4 mm，軸心落在指片（TCP 後 1.3–21.3 mm）內（第十二趟 TCP 在軸心上 ⇒ 指尖夾弦、抬升滑落）
        self.app_prof = dict(base_goal=list(app['base_goal']), door=None)   # nav_window：導航到點後收斂到抓取底盤位姿
        self.T_app, self.T_b, self.t_l0 = float(app['T']), float(app['T_b']), float(app['t_l0'])
        self.T_rel0, self.b_anchor, self.n_ev, self.T_lift0, self.T_rel_s0 = None, None, 0, None, None
        self.ret_sent, self.attach = False, False
        self.obj_release, self.obj_detached, self.binned, self.q_b0, self.T_bm, self.bin_target = False, False, False, None, None, None
        if BIN:
            RBIN.install(_CF_)                                        # 盒體進自碰撞淨空（取物控制器建立時才裝；開門／穿門段不變）
        self.phase, self.t_ph0 = 'APPROACH', float(t0)
        self.motm = app.get('motm')                                   # Howard 11/19 移動中夾取：dict(b0, e, v_g, s0)（里程）；None＝原停車夾取
        self.m_s0, self.m_t0, self.T_rel_l0 = None, None, None
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

    def _motm_t_arm(self, q):
        """手臂展開到抓取解所需時間：最大關節位移 ÷ 0.6 rad/s（最小急動峰速 1.875×）＋0.5 s，至少 t_arm（展開與底盤移動重疊、不停車）。"""
        t_a = float(self.motm.get('t_arm', 4.0))
        if self.motm.get('q_grasp') is not None:
            t_a = max(t_a, 1.875 * float(np.abs(np.asarray(self.motm['q_grasp'], float) - np.asarray(q[3:], float)).max()) / 0.6 + 0.5)
        return t_a

    def _motm_s(self, t_abs):
        return self.m_s0 + float(self.motm['v_g']) * (t_abs - self.m_t0)

    def _motm_ref(self, t_eff):
        """經過路線參考：底盤沿 e 等速 v_g、朝向固定（全向底盤橫移）；回 (X 視窗, xi 視窗)。"""
        b0, e, v = np.asarray(self.motm['b0'], float), np.asarray(self.motm['e'], float), float(self.motm['v_g'])
        th = float(b0[2]); c_, s_ = math.cos(th), math.sin(th)
        vb = np.array([c_ * v * e[0] + s_ * v * e[1], -s_ * v * e[0] + c_ * v * e[1], 0.0])   # 世界速度轉本體
        X = [XO.from_xytheta(b0[0] + self._motm_s(t_eff + k * DT) * e[0], b0[1] + self._motm_s(t_eff + k * DT) * e[1], th) for k in range(XO.N + 1)]
        return X, [vb.copy() for _ in range(XO.N + 1)]

    def _motm_qpost(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.m_t0) / max(self.T_app - T_SETTLE, 1e-6), 0.0), 1.0))[0]
        return (1.0 - u) * self.q_m0 + u * self.q_mg

    def _lift_target(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / T_LIFT, 0.0), 1.0))[0]
        T = self.T_lift0.copy()
        T[2, 3] += u * LIFT_M
        return T

    def _carry_target(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / T_CARRY, 0.0), 1.0))[0]
        A, Bm = self.T_rel_s0, self.T_rel_stow
        T = np.eye(4)
        T[:3, 3] = (1 - u) * A[:3, 3] + u * Bm[:3, 3]                   # 只移位置：夾爪相對底盤姿態保持抬升時（圓柱維持直立；第十四趟混到收臂姿態 ⇒ 手腕繞接近軸轉、圓柱轉橫滑落）
        T[:3, :3] = A[:3, :3]
        return T

    def _held(self):
        f, fm = self.finger_force, self.finger_meas
        return f is not None and fm is not None and min(f) >= CONTACT_N and min(fm) >= FINGER_EMPTY

    def _advance(self, q):
        el = self.t - self.t_ph0
        if self.T_rel0 is None:
            self.T_rel0 = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
        if self.motm is not None and self.m_s0 is None and self.phase == 'APPROACH':   # 進線前：底盤先到路線起點（位置＋偏航）再開始等速橫移
            b0, e = np.asarray(self.motm['b0'], float), np.asarray(self.motm['e'], float)   # （s76：偏航差 2.9 rad 邊轉邊走擦桌角；第三十一趟 Isaac：只轉不移、離線 0.31 m ⇒ 進線偏航漂、淨空 0.100）
            s_st = min(float(np.dot(np.asarray(q[:2], float) - b0[:2], e)), float(self.motm.get('s0', -0.05)) - 0.10, -float(self.motm['v_g']) * self._motm_t_arm(q))
            b_st = np.array([b0[0] + s_st * e[0], b0[1] + s_st * e[1], float(b0[2])])
            dyaw_ = abs((float(q[2]) - float(b0[2]) + math.pi) % (2 * math.pi) - math.pi)
            dpos_ = float(np.hypot(*(np.asarray(q[:2], float) - b_st[:2])))
            if dyaw_ > 0.10 or dpos_ > 0.04:
                self.b_anchor = b_st
                self._goto('ALIGN', dyaw=round(dyaw_, 3), dpos=round(dpos_, 3), s_start=round(s_st, 3))
                return
        if self.phase == 'ALIGN':
            dyaw_ = abs((float(q[2]) - float(self.b_anchor[2]) + math.pi) % (2 * math.pi) - math.pi)
            dpos_ = float(np.hypot(*(np.asarray(q[:2], float) - self.b_anchor[:2])))
            if dyaw_ <= 0.10 and dpos_ <= 0.04:
                self.T_rel0 = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
                self._goto('APPROACH', aligned_s=round(self.t - self.t_ph0, 2))
            elif self.t - self.t_ph0 > 15.0:
                self._stop('motm_align_timeout', dyaw=round(dyaw_, 3), dpos=round(dpos_, 3))
            return
        if self.motm is not None and self.m_s0 is None:               # 移動中夾取時程：自目前底盤在路線上的投影起算，名目到點＝底盤經過 s＝0
            b0, e = np.asarray(self.motm['b0'], float), np.asarray(self.motm['e'], float)
            self.m_s0 = min(float(np.dot(np.asarray(q[:2], float) - b0[:2], e)), float(self.motm.get('s0', -0.05)) - 0.10,
                            -float(self.motm['v_g']) * self._motm_t_arm(q))   # 手臂展開至少 t_arm（s73）
            self.m_t0 = self.t
            self.T_app = (0.0 - self.m_s0) / float(self.motm['v_g'])
            self.T_b, self.t_l0 = max(1.0, self.T_app - 1.5), max(0.5, self.T_app - 2.5)
            if self.motm.get('q_grasp') is not None and MOTM_W_Q > 0.0:   # 底盤經過 s＝0 時恰在選位底盤位 ⇒ 選位抓取解即該時刻構型
                self.q_m0, self.q_mg = np.asarray(q[3:], float).copy(), np.asarray(self.motm['q_grasp'], float)
                self.app_qpost, self.app_w_q = self._motm_qpost, MOTM_W_Q
            self._ev('pick_motm_start', s0=round(self.m_s0, 4), T_app=round(self.T_app, 3), v_g=self.motm['v_g'])
        if self.phase == 'APPROACH':
            Tc = K.fk(q, 'link_tcp')
            ep = float(np.linalg.norm(Tc[:3, 3] - self.T_grasp[:3, 3]))
            er = math.degrees(float(np.linalg.norm(_log_R(self.T_grasp[:3, :3].T @ Tc[:3, :3]))))
            tz_ = getattr(self, 'app_tol_z', APP_TOL_Z)
            if tz_ is not None:                                       # 分方向：沿圓柱軸的高度差不影響側夾
                d_ = Tc[:3, 3] - self.T_grasp[:3, 3]
                ep_ok = float(np.linalg.norm(d_[:2])) <= 0.005 and abs(float(d_[2])) <= tz_
            else:
                ep_ok = ep <= 0.005
            if el >= self.T_app and ep_ok and er <= 3.0:
                self.b_anchor = np.asarray(q[:3], float).copy()
                self.finger_cmd = P.FINGER_OPEN
                self._goto('CLOSE', ep_mm=round(ep * 1e3, 2), er_deg=round(er, 2))
            elif self.motm is not None and el > self.T_app + MOTM_CLOSE_LATE_S:
                self._stop('motm_window_missed', ep_mm=round(ep * 1e3, 2), er_deg=round(er, 2), s=round(self._motm_s(self.t), 4))
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
                self.T_rel_l0 = np.linalg.inv(_T3(q[:3])) @ self.T_lift0     # 移動中夾取：連接後抬升改相對底盤（物體已理想連接，不再受世界固定可達窗限制）
                self.attach = True                                    # Howard 11/18「夾到就改成理想連接」：夾住證據成立 ⇒ 請執行端以 FixedJoint 固定圓柱（同門把理想綁定）
                self._goto('LIFT', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas], force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
            elif el > (MOTM_EVID_TO_S if self.motm is not None else EVID_TO_S):
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
                if BIN and not getattr(self, 'bin_carry_first', False):
                    self._bin_start(q, 'BIN_MOVE')
                    return
                if BIN:                                               # 抽屜筆：先兩段式收臂抬過面板頂再轉去放盒（s63：抬升位直接轉 ⇒ link_eef 掃抽屜面板 5 mm 被停）
                    self._goto('CARRY')
                    return
                self._goto('CARRY' if PICK_CARRY else 'RETURN')               # 預設不收臂：手臂相對底盤保持抬升位姿（第十五趟收臂一開始手腕重配置 ⇒ 接觸瞬斷滑落；瓶子線持瓶也是手臂不動、底盤移動）
            elif el > EVID_TO_S:
                self._stop('lost_after_lift', finger=None if self.finger_meas is None else [round(float(x), 5) for x in self.finger_meas], force=None if self.finger_force is None else [round(float(x), 3) for x in self.finger_force])
        elif self.phase == 'CARRY':
            T_now = np.linalg.inv(_T3(q[:3])) @ K.fk(q, 'link_tcp')
            ep = float(np.linalg.norm(T_now[:3, 3] - self._carry_target(self.t_ph0 + T_CARRY)[:3, 3]))
            if el >= T_CARRY and ep <= 0.010:
                self._goto('RETURN', carry_err_mm=round(ep * 1e3, 2))
            elif el > T_CARRY + PH_TO_S:
                self._stop('carry_not_reached', carry_err_mm=round(ep * 1e3, 2))
            if not self._held():
                self._stop('lost_during_carry')
        elif self.phase in ('BIN_MOVE', 'BIN_REL', 'BIN_OPEN', 'BIN_DROP', 'BIN_STOW'):
            self._advance_bin(q, el)
        elif self.phase == 'RETURN':
            if not self.binned and not self._held():
                self._stop('lost_during_return')

    # ---- 車尾收納盒（D2_BIN）
    def _bin_start(self, q, ph):
        self.q_b0 = np.asarray(q[3:], float).copy()
        self.bin_target = RBIN.q_place()[0] if ph == 'BIN_MOVE' else P.Q_STOW
        self.T_bm = 1.875 * float(np.abs(self.bin_target - self.q_b0).max()) / BIN_QD + 0.5
        self._goto(ph, T=round(self.T_bm, 2), q_target=[round(float(x), 4) for x in self.bin_target])

    def _bin_q(self, t_abs):
        u = XE.minjerk(min(max((t_abs - self.t_ph0) / max(self.T_bm, 1e-6), 0.0), 1.0))[0]
        return (1.0 - u) * self.q_b0 + u * self.bin_target

    def _advance_bin(self, q, el):
        fm = self.finger_meas
        if self.phase in ('BIN_MOVE', 'BIN_STOW'):
            err = float(np.abs(np.asarray(q[3:], float) - self.bin_target).max())
            if self.phase == 'BIN_MOVE' and not self._held():
                self._stop('lost_during_bin_move', finger=None if fm is None else [round(float(x), 5) for x in fm])
            elif el >= self.T_bm and err <= 0.03:
                if self.phase == 'BIN_MOVE':
                    self.obj_release = True                           # 請執行端解除夾爪↔物體理想連接（FixedJoint）
                    self._goto('BIN_REL', err=round(err, 4))
                else:
                    self._ev('pick_stowed', err=round(err, 4), base=[round(float(x), 4) for x in q[:3]])   # 監督據此送下一件／返回目標
                    self._goto('RETURN', err=round(err, 4))
            elif el > self.T_bm + PH_TO_S:
                self._stop('bin_move_not_reached' if self.phase == 'BIN_MOVE' else 'bin_stow_not_reached', err=round(err, 4))
        elif self.phase == 'BIN_REL':
            if self.obj_detached:
                self.attach = False                                   # 連接已解除 ⇒ 同拍撤回連接請求（s66：撤回晚於伺服器重置 ⇒ 重送 attach_cyl，Isaac 會把物體焊回夾爪）
                self.finger_cmd = P.FINGER_OPEN
                self._goto('BIN_OPEN')
            elif el > DETACH_TO_S:
                self._stop('obj_detach_ack_timeout')
        elif self.phase == 'BIN_OPEN':
            if fm is not None and min(fm) >= P.FINGER_OK:
                self._goto('BIN_DROP', finger=[round(float(x), 5) for x in fm])
            elif el > P.FINGER_TO_S:
                self._stop('finger_open_timeout', finger=None if fm is None else [round(float(x), 5) for x in fm])
        elif self.phase == 'BIN_DROP':
            if el >= RBIN.DROP_WAIT_S:
                self.binned, self.attach = True, False
                self._ev('pick_binned')
                self._bin_start(q, 'BIN_STOW')

    def _solve(self, q):
        if self.phase in ('BIN_MOVE', 'BIN_REL', 'BIN_OPEN', 'BIN_DROP', 'BIN_STOW'):
            return self._solve_bin(q)
        qp, sp = self._pred(q)
        t_eff = self.t + D_STEPS * PHYS
        X, xi, obs, umin0, src = G.MovingGraspCtrl.nav_window(self, q, qp)
        if obs is None:
            return None, 'nav_obstacles_stale'
        if self.phase == 'ALIGN':                                     # 原地對齊：底盤位置不動、轉到路線朝向；手臂保持
            X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
            xi = [np.zeros(3)] * (XO.N + 1)
            umin0, src = float('nan'), 'align'
        elif self.motm is not None and self.m_s0 is not None and self.phase in ('APPROACH', 'CLOSE', 'CONFIRM', 'LIFT', 'VERIFY'):   # 移動中夾取：底盤沿經過路線不停
            X, xi = self._motm_ref(t_eff)
            umin0, src = float('nan'), 'motm_line'
        elif self.phase in ('CLOSE', 'CONFIRM', 'LIFT', 'VERIFY', 'CARRY'):   # 接觸／持物操作段：底盤原地
            X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
            xi = [np.zeros(3)] * (XO.N + 1)
            umin0, src = float('nan'), 'anchor'
        if self.phase == 'RETURN':
            self.app_prof = None                                      # 返回段：照導航前端（不再收斂到抓取位）
            if src == 'nav':                                          # 會車選邊（s92d：行人 walker_r1 正面迎來、牆側不夠 ⇒ 被推著倒退後接觸）
                P.yield_apply(self, qp, X, obs)
        arm = dict(mode='hold')
        Xp = [(float(Xk[0, 2]), float(Xk[1, 2]), math.atan2(float(Xk[1, 0]), float(Xk[0, 0]))) for Xk in X]
        if self.phase in ('APPROACH', 'CLOSE', 'CONFIRM', 'LIFT', 'VERIFY'):
            if self.phase == 'APPROACH':
                Ts = [self._app_target(t_eff + k * DT, Xp[k]) for k in range(XO.N + 1)]
            elif self.phase in ('LIFT', 'VERIFY') and self.motm is not None:   # 相對底盤抬升
                def _lr(tt):
                    T_ = self.T_rel_l0.copy(); T_[2, 3] += LIFT_M * XE.minjerk(min(max((tt - self.t_ph0) / T_LIFT, 0.0), 1.0))[0]; return T_
                Ts = [_lr(t_eff + k * DT) if self.phase == 'LIFT' else _lr(self.t_ph0 + T_LIFT + 1.0) for k in range(XO.N + 1)]
            elif self.phase in ('LIFT', 'VERIFY'):
                Ts = [self._lift_target(t_eff + k * DT) if self.phase == 'LIFT' else self._lift_target(self.t_ph0 + T_LIFT + 1.0) for k in range(XO.N + 1)]
            else:
                Ts = [self.T_grasp] * (XO.N + 1)
            arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts),
                       w_p=C0.W_NEAR if self.phase != 'APPROACH' else C0.W_FAR, lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper,
                       tcp_rel_base=bool(self.motm is not None and self.phase in ('LIFT', 'VERIFY')), qp_solver='gi', **B.W_ARM)
            qpf = getattr(self, 'app_qpost', None)                       # 選用：接近段關節姿態引導（抽屜把手；圓柱流程不設 ⇒ 不變）
            if qpf is not None and self.phase == 'APPROACH':
                arm.update(q_post=np.array([qpf(t_eff + (k + 1) * DT) for k in range(XO.N)]), w_q=float(getattr(self, 'app_w_q', 1.0)))
        elif self.phase == 'CARRY':
            Ts = [self._carry_target(t_eff + k * DT) for k in range(XO.N + 1)]
            arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts), w_p=C0.W_FAR,
                       lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper, tcp_rel_base=True, qp_solver='gi', **B.W_ARM)
        mpc = self.mpc
        u_min_saved = np.asarray(mpc.cfg.u_min, float).copy()
        u_max_saved = np.asarray(mpc.cfg.u_max, float).copy()
        if math.isfinite(umin0):
            mpc.cfg.u_min = np.array([umin0, u_min_saved[1], u_min_saved[2]])
        if self.phase == 'APPROACH' and getattr(self, 'app_v', APP_V) is not None:   # 接近段速度框（只圓柱；抽屜控制器設 None）
            lim_ = np.array([self.app_v if hasattr(self, 'app_v') else APP_V] * 2 + [0.6])
            mpc.cfg.u_max = np.minimum(np.asarray(mpc.cfg.u_max, float), lim_)
            mpc.cfg.u_min = np.maximum(np.asarray(mpc.cfg.u_min, float), -lim_)
        nv_ = P.near_v(qp[:3]) if self.phase in ('APPROACH', 'RETURN', 'CARRY', 'ALIGN') else None
        if nv_ is not None:                                           # 近家具減速（同穿門段）
            lim_ = np.array([nv_, nv_, 9.0])
            mpc.cfg.u_max = np.minimum(np.asarray(mpc.cfg.u_max, float), lim_)
            mpc.cfg.u_min = np.maximum(np.asarray(mpc.cfg.u_min, float), -lim_)
        if self.phase in ('RETURN', 'CARRY'):                          # 持物速度框（任務限制，非導航參數）：第十六趟伸臂持物時底盤原地轉 0.43 rad/s ⇒ 圓柱沿運動方向滑出
            lim_ = np.array([CARRY_V, CARRY_V, CARRY_WZ])
            mpc.cfg.u_max = np.minimum(np.asarray(mpc.cfg.u_max, float), lim_)   # 不覆蓋前面的近家具減速
            mpc.cfg.u_min = np.maximum(np.asarray(mpc.cfg.u_min, float), -lim_)
        w0 = time.perf_counter()
        try:
            r = mpc.solve(X_now=XO.from_xytheta(*qp[:3]), X_ref_win=np.array(X), xi_ref_win=np.array(xi), xi_prev=self.u_prev[:3], arm=arm, obstacles=obs)
        finally:
            mpc.cfg.u_min, mpc.cfg.u_max = u_min_saved, u_max_saved
        self.wg_log.append((round(self.t, 3), self.phase, round((time.perf_counter() - w0) * 1e3, 2), str(r.status), src, len(obs), int(r.cbf_active), _obs_near(obs, qp)))
        if not np.isfinite(np.asarray(r.u_opt, float)).all():
            return None, 'wgmpc_nonfinite'
        if arm['mode'] == 'hold':
            return np.r_[np.asarray(r.u_opt, float), np.zeros(6)], src
        ck = dict(r.arm_check or {})
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None or r.status != 'solved':
            return None, 'wgmpc:%s/%s/%s' % (r.status, ck.get('fallback'), ck.get('failed'))
        return np.r_[np.asarray(r.u_opt, float), np.asarray(r.u_arm_seq[0], float)], src

    def _solve_bin(self, q):
        """放盒／收臂：手臂關節空間剖面（相對底盤，tcp_rel_base）；底盤照導航前端（限速框）或錨定。"""
        qp, sp = self._pred(q)
        t_eff = self.t + D_STEPS * PHYS
        X, xi, obs, umin0, src = G.MovingGraspCtrl.nav_window(self, q, qp)
        if obs is None:
            return None, 'nav_obstacles_stale'
        self.app_prof = None
        if not BIN_MOVE:
            X = [XO.from_xytheta(*self.b_anchor)] * (XO.N + 1)
            xi = [np.zeros(3)] * (XO.N + 1)
            umin0, src = float('nan'), 'anchor'
        if self.phase in ('BIN_MOVE', 'BIN_STOW'):
            qs = [self._bin_q(t_eff + k * DT) for k in range(XO.N + 1)]
        else:
            qs = [self.bin_target] * (XO.N + 1)                       # 解除／張爪／落下：保持放置構型
        Ts = [K.fk(np.r_[np.zeros(3), qk], 'link_tcp') for qk in qs]
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts),
                   w_p=C0.W_FAR, lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper, tcp_rel_base=True, qp_solver='gi', q_post=np.array(qs[1:]), w_q=5.0, **B.W_ARM)
        mpc = self.mpc
        u_min_saved = np.asarray(mpc.cfg.u_min, float).copy()
        u_max_saved = np.asarray(mpc.cfg.u_max, float).copy()
        if math.isfinite(umin0):
            mpc.cfg.u_min = np.array([umin0, u_min_saved[1], u_min_saved[2]])
        nv_ = P.near_v(qp[:3])
        lim_ = np.array([min(BIN_V, nv_ if nv_ is not None else 9.0)] * 2 + [BIN_WZ])   # 放盒中底盤速度框（任務限制）＋近家具減速
        mpc.cfg.u_max = np.minimum(u_max_saved, lim_)
        mpc.cfg.u_min = np.maximum(np.asarray(mpc.cfg.u_min, float), -lim_)
        w0 = time.perf_counter()
        try:
            r = mpc.solve(X_now=XO.from_xytheta(*qp[:3]), X_ref_win=np.array(X), xi_ref_win=np.array(xi), xi_prev=self.u_prev[:3], arm=arm, obstacles=obs)
        finally:
            mpc.cfg.u_min, mpc.cfg.u_max = u_min_saved, u_max_saved
        self.wg_log.append((round(self.t, 3), self.phase, round((time.perf_counter() - w0) * 1e3, 2), str(r.status), src, len(obs), int(r.cbf_active), _obs_near(obs, qp)))
        if not np.isfinite(np.asarray(r.u_opt, float)).all():
            return None, 'wgmpc_nonfinite'
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
