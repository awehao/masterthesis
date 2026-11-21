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
r12（Codex 20261113_144548；Isaac 名目趟 2：手指目標＝閉合開度模型值 1.7 mm ⇒ 未夾緊、單指推門）：
  手指**命令**與碰撞**幾何**分開——命令在 CLOSE 段由張開斜降到 0（同停車版夾緊；之後保持 0）；幾何用兩指各自實測開度（step 的 finger_meas；
  假世界無手指物理時用 max(命令, FINGER_CLOSED) 模型）。CLOSE 段的監看與停止／候選掃掠同時以實測開度與預測閉合開度（FINGER_CLOSED）兩種幾何評估取最小。
  交接被接受前門角自鎖定起變化 >0.5°（同 BindGate 基準）⇒ door_moved_before_bind 停止；停止後 4 s 內未停穩（管線、底盤、手臂、門）⇒ 模擬中止（不無限追門）。
r11：「現在停」淨空先派工、派工期間主程序算追蹤命令（投機計算，收到現在停結果後才用；不過即丟棄，q_ik_prev 只在通過後更新）；
  接觸停法模擬的管線全零檢查改隨佇列維護旗標（同一布林）。命令與 r10 逐位元相同（假世界核對）。
r10：_arm_track 的 IK 對（q1、q0）與重力偏置以「輸入位元組完全相同」為鍵快取（底盤預測位姿、目標位姿、種子；每拍清空）——
  接觸停法向前模擬在底盤停下後反覆以同一輸入解 IK；命中時結果與重算逐位元相同（Isaac 冒煙 3：接觸段首拍 50.0 ms 觸發期限）。
r9（Howard 20261113 13:49／Codex 20261113_141451）：前導接近——HOLD 段可位於窗口之前（未接觸、預抓保持，同一底盤律與監看），
  INSERT 起才要求在窗口內；pool 可傳入外部共用的計算池（不由本控制器關閉）。
r8（Codex 20261113_133147 §3）：close_gate（可呼叫，回 True 才准從 INSERT 進入 CLOSE）——下游首段未及時可用就不開始閉爪；
  INSERT 時間到而閘門未開 ⇒ close_gate_not_ready 受控停止（尚未接觸，用原停法）。None＝r7。
r5：fastkin＝True 時 IK 改 FastChain（door_fast_kin.ik_closure_fc 同迭代，解差 4e-14 級）、重力改 FastGravity2（差 9e-15 N·m）；預設 False＝r4。"""
import math
import os
import sys
import time

import numpy as np
import door_fric_ff as FF

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


def door_filter(obs, ap, phi):
    """接近開始後（ap 帶 door）：只排除「屬於門扇本體」的障礙——動態追蹤障礙且中心落在門扇（繞鉸鏈轉 −φ 的線段、寬 w）上：
    垂直距離 ≤ 0.06 m、沿門扇 0.02 ≤ a ≤ w−0.02。靜態牆點（含門框、鄰牆）一律保留；未能關聯到門扇的障礙一律保留（Codex 20261117_173459）。"""
    if not obs or ap is None or ap.get('door') is None:
        return obs
    dd = ap['door']; h = np.asarray(dd['hinge'], float); u = np.asarray(dd['dir'], float); u = u / np.linalg.norm(u)
    ph = 0.0 if phi is None else float(phi)
    c_, s_ = math.cos(-ph), math.sin(-ph)
    ue = np.array([c_ * u[0] - s_ * u[1], s_ * u[0] + c_ * u[1]]); un = np.array([-ue[1], ue[0]])
    out, n_drop = [], 0
    for o_ in obs:
        p_ = np.array([o_['x'], o_['y']]) - h
        a_, d_ = float(p_ @ ue), abs(float(p_ @ un))
        if (not o_.get('static', False)) and d_ <= 0.06 and 0.02 <= a_ <= float(dd['w']) - 0.02:
            n_drop += 1
            continue
        out.append(o_)
    return out


BIAS_OBS = os.environ.get('GRASP_BIAS_OBS', '1') == '1'                 # 手臂等效偏置線上觀測（接近開始後、閉爪前）
BIAS_OBS_TAU = float(os.environ.get('GRASP_BIAS_OBS_TAU', '0.5'))
BIAS_OBS_MAX = float(os.environ.get('GRASP_BIAS_OBS_MAX', '0.02'))      # 新增殘差補償限幅（rad／軸）
CLOSE_WAIT_S = float(os.environ.get('GRASP_CLOSE_WAIT_S', '2.0'))   # 插入名目時間到後閉爪準備窗口上限（模擬秒；0＝原單拍判定）
BG_V = 0.06                                                          # 導航到點後收斂到抓取底盤位姿的參考速度上限（m/s）


def _T3_(p):
    c_, s_ = math.cos(p[2]), math.sin(p[2])
    T = np.eye(4); T[:2, :2] = [[c_, -s_], [s_, c_]]; T[:2, 3] = p[:2]
    return T


class MovingGraspCtrl:
    def __init__(self, ident, est_fn, d_world, v_pass, window_s, s, pipe, u_last, u_prev, off_static, finger_open=0.0089, joint_kp=600.0, sweep_stride=3, sched=None, pool=0, fastkin=False, deadline_ms=None, fric_ff=None, lock_budget_fn=None):
        self.fastkin = bool(fastkin)
        self.deadline_ms = None if deadline_ms is None else float(deadline_ms)
        self.deadline_inject, self.fb_inject = None, None
        self.n_deadline_miss, self.tick_log, self._fb_ok = 0, [], False
        self.close_gate = None                                       # r8
        self.close_wait_t0, self.close_delay = None, 0.0               # 閉爪準備窗口
        self._ikc = {}                                               # r10：每拍清空
        self._pool, self._pool_owned = None, False
        if not isinstance(pool, (int, float)):                       # r9：外部共用計算池
            self._pool = pool
        elif int(pool) > 0:
            import door_sweep_pool as DSP
            self._pool, self._pool_owned = DSP.SweepPool(int(pool)), True
        self.pool_ms = []
        self.cfg = C0.make_cfg(ident)
        self.cfg.base_fixed = False
        al = np.asarray(self.cfg.arm_model.alpha, float)
        self.cfg.arm_model.bias = al * np.asarray(off_static, float)
        self.tau, self.va = PHYS / al, float(B.W_ARM['v_arm'])
        self.fric_ff = fric_ff                                         # r13：庫倫摩擦前饋的 τ_f（N·m；None＝不加，逐值同 r12）
        # r14（Codex 20261116 A3）：鎖定任務參考模式。lock_budget_fn(t, q9, ep, er) → dict(pos_m, ang_deg, ok, ...)；給了就以「鎖定預算在界內」取代影像新鮮度條件——
        #   est_fn 仍回原始時間戳的鎖定參考（不偽造新鮮度）；預算每拍重算；超界：READY 撤銷、未閉爪則拒絕閉爪（停止）、已接觸（CLOSE 之後）走既有接觸停法。相對運動、確認時間、BindGate 不變。None＝r13 行為。
        self.lock_budget_fn, self.lock_budget, self.lock_budget_max = lock_budget_fn, None, None
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
        self.finger_q = float(finger_open)                           # r12：碰撞幾何用的開度（純量或兩指 tuple）
        self.finger_cmd = float(finger_open)                         # r12：手指命令（CLOSE 起斜降到 0）
        self._finger_meas = None
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
        """TCP 沿接近軸的退讓量 s（S_PRE→0）：HOLD 段 S_PRE、INSERT 段最小急動、之後 0。t_rel 自控制器起點。
        D2 承接（ins_prof 有值；預設 None＝原行為）：hold_s 應為 0，插入段＝自交權當下 (s0, ṡ0, s̈0) 到 (0,0,0) 的五次多項式（時長 ins_s），承接連續接近的進度，不重播 HOLD／INSERT。"""
        h, i = self.sched['hold_s'], self.sched['ins_s']
        ip = getattr(self, 'ins_prof', None)
        if ip is not None:
            u = min(max(t_rel - h, 0.0), i)
            s0, v0, a0 = float(ip['s0']), float(ip['v0']), float(ip['a0'])
            T2, T3, T4, T5 = i * i, i ** 3, i ** 4, i ** 5
            c3 = (20 * (0.0 - s0) - 12 * v0 * i - 3 * a0 * T2) / (2 * T3)
            c4 = (30 * (s0 - 0.0) + 16 * v0 * i + 3 * a0 * T2) / (2 * T4)
            c5 = (12 * (0.0 - s0) - 6 * v0 * i - a0 * T2) / (2 * T5)
            if u >= i - 1e-12:
                return 0.0                                            # 終點精確為 0（c13：五次式在 u＝i 算出 −1e−18，被 object_target 的 s≥0 檢查拒絕）
            return max(0.0, s0 + v0 * u + 0.5 * a0 * u * u + c3 * u ** 3 + c4 * u ** 4 + c5 * u ** 5)
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
        ap = getattr(self, 'app_prof', None)
        if ap is None:
            return self.T_at(self.handle_ref(), self.s_of(t_rel))
        return self._app_target(ap, float(t_rel))

    def _app_target(self, ap, t):
        """W-GMPC 接近剖面（主線修正：交權提前到進走廊；這只是 TCP 參考，命令由 W-GMPC 解）：
        Tl(t)＝沿接近軸的抓取線（s：d_line 於 [t_l0, T] 最小急動降到 0，之前 d_line、之後 0）；
        Tb(t)＝交權當下 TCP 相對底盤的位姿掛在名目底盤（b0 沿 d 以 v 前進）上；
        參考＝Tb→Tl 的混接（位置線性、姿態測地線；權重 β＝minjerk(t/T_b)）。"""
        t_abs = t
        t = t - float(ap.get('t0', 0.0))                               # 單一控制器：剖面時間自「接近開始」起算
        T, tl0, Tb_ = float(ap['T']), float(ap['t_l0']), float(ap['T_b'])
        Te = T - float(ap.get('t_settle', 0.6))                      # 接近線提早 t_settle 到底、之後 s＝0 保持（c31：剛到底即閉爪時 TCP 落後 3.4 mm，鎖定預算 6.8 mm＞5 mm）
        u = min(max((t - tl0) / max(Te - tl0, 1e-6), 0.0), 1.0)
        sf = float(ap.get('s_final', 0.0))                             # 最終抓取深度留量（外殼—橫桿名目 6.8 mm 太緊：開門中停止路徑吃掉約 2 mm；u8／s4）
        s_ = sf + (float(ap['d_line']) - sf) * (1.0 - XE.minjerk(u)[0]) if t < Te else sf
        Tl = self.T_at(self.handle_ref(), max(s_, 0.0))
        beta = XE.minjerk(min(max(t / max(Tb_, 1e-6), 0.0), 1.0))[0]
        if beta >= 1.0 - 1e-12:
            return Tl
        xc = getattr(self, '_xref_cache', None)
        if ap.get('b0') is None and xc is not None:                     # 單一控制器：起點 TCP 掛在導航參考底盤上（不假設沿門直線）
            kk = int(min(max(round((t_abs - xc[0]) / DT), 0), len(xc[1]) - 1))
            b0 = np.asarray(xc[1][kk], float); p2 = b0[:2]
        else:
            b0 = np.asarray(ap['b0'], float)
            p2 = b0[:2] + self.d * self.v * max(t, 0.0)
        c_, s2 = math.cos(b0[2]), math.sin(b0[2])
        Tb = np.eye(4); Tb[:2, :2] = [[c_, -s2], [s2, c_]]; Tb[:2, 3] = p2
        Tb = Tb @ np.asarray(ap['T_rel0'], float)
        R = Tb[:3, :3].T @ Tl[:3, :3]
        cth = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)); th = math.acos(cth)
        Rb = np.eye(3)
        if th > 1e-9:
            w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(th))
            Kx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]); a_ = beta * th
            Rb = np.eye(3) + math.sin(a_) * Kx + (1 - math.cos(a_)) * Kx @ Kx
        Tt = np.eye(4); Tt[:3, :3] = Tb[:3, :3] @ Rb; Tt[:3, 3] = (1 - beta) * Tb[:3, 3] + beta * Tl[:3, 3]
        return Tt

    def _arm_track(self, qp, sp, uf, ub, te, seed):
        """命令生效後一拍末的預測底盤（qp 起、本拍底盤命令斜坡）解 IK → 設定點 → 手臂命令。回 (u_arm 或 None, 殘差, q1)。"""
        b = np.asarray(qp[:3], float).copy()
        uf = np.asarray(uf, float)
        for jj in range(NPH):
            vv = uf + min(jj + 1, NPH) / NPH * (ub - uf)
            cb, sb = math.cos(b[2]), math.sin(b[2])
            b = b + PHYS * np.array([cb * vv[0] - sb * vv[1], sb * vv[0] + cb * vv[1], vv[2]])
        s1_, s0_ = self.s_of(te), self.s_of(te - DT)
        key = (b.tobytes(), np.asarray(qp[:3], float).tobytes(), np.asarray(seed, float).tobytes(), s1_, s0_, self.handle_ref().tobytes())
        hit = self._ikc.get(key)
        if hit is None:
            q1, r1 = self._ik(b, self.target(te), seed)
            q0_, r0 = self._ik(qp[:3], self.target(te - DT), q1)
            offv = self.off(np.r_[b, q1]) if not (max(r1, r0) > 1e-6 or float(np.abs(q1 - seed).max()) > 0.5) else None
            hit = (q1, r1, q0_, r0, offv)
            self._ikc[key] = hit
        q1, r1, q0_, r0, offv = hit
        if max(r1, r0) > 1e-6 or float(np.abs(q1 - seed).max()) > 0.5:
            return None, max(r1, r0), None
        S1 = q1 + self.tau * (q1 - q0_) / DT - offv + FF.ff((q1 - q0_) / DT, self.fric_ff, self.joint_kp)
        return (S1 - sp) / DT, max(r1, r0), q1

    def _track(self, q, tr):
        if getattr(self, 'unified', False):
            return self._track_unified(q, tr)
        if getattr(self, 'wgmpc', False):
            return self._track_wgmpc(q, tr)
        return self._track_ik(q, tr)

    def init_app_rel(self, qp):
        """接近開始的第一拍：起點 TCP 相對底盤位姿（以本拍預測構型），監看與求解共用。"""
        ap_ = getattr(self, 'app_prof', None)
        if ap_ is not None and ap_.get('T_rel0') is None:
            ap_['T_rel0'] = (np.linalg.inv(_T3_(np.asarray(qp, float)[:3])) @ K.fk(np.asarray(qp, float), 'link_tcp')).tolist()

    def shared_mpc(self):
        """單一控制器的求解器實例（Codex 20261117_161844）：以導航送來的求解設定（CBF、權重、限制）建一次，抓持與開門共用；之後只換任務。"""
        if getattr(self, 'mpc', None) is None:
            ni = getattr(self, 'nav_in', None) or {}
            cf = ni.get('cfg')
            fields = {f.name for f in XO.dataclasses.fields(XO.GR.GMPCConfig)}
            if cf:
                kw = {k_: (np.asarray(v_, float) if isinstance(v_, list) else v_) for k_, v_ in cf.items() if k_ in fields}
                rc = XO.GR.GMPCConfig(**kw)
                self.cfg_src = 'nav_node'
            else:
                nc = B.nav_cfg()
                rc = XO.GR.GMPCConfig(**{f.name: getattr(nc, f.name) for f in XO.dataclasses.fields(nc)})
                self.cfg_src = 'bottle_nav_cfg（導航設定尚未收到）'
            rc.cmd_ramp_w = (0.4, 0.6) if XO.BASE_RAMP else None
            rc.wheel_w_max, rc.wheel_a_max = rc.wheel_w_max * XO.WHEEL_FRAC[0], rc.wheel_a_max * XO.WHEEL_FRAC[1]
            self.mpc = XO.GR.GMPC(rc)
            self.wg_log = []
        return self.mpc

    def nav_window(self, q, qp):
        """門把移動中抓取（Howard 11/21；app_prof 帶 line 時）：障礙物照導航前端，底盤參考改為「併線＋沿門前走廊等速經過抓取底盤位」（同茶几 motm_merge），
        不收斂到站位、不停；走廊盡頭前指數減速。交接（READY 被接受）後由開門控制接手（首段規劃已含初速度）。"""
        X, xi, obs, umin0, src = MovingGraspCtrl._nav_window_base(self, q, qp)   # 取物／抽屜／關門控制器以類別函式呼叫（self 不是本類）
        ap_ = getattr(self, 'app_prof', None)
        ln_ = None if (ap_ is None or not isinstance(self, MovingGraspCtrl)) else ap_.get('line')
        if ln_ is None or obs is None or self.stop is not None or self.phase not in ('HOLD', 'INSERT', 'CLOSE', 'CONFIRM', 'READY'):
            return X, xi, obs, umin0, src
        import motm_merge as MM
        tr = self.t - self.t0
        v_ = float(ln_['v_g']); b0 = np.asarray(ln_['b0'], float); e = np.asarray(ln_['e'], float)[:2]; e = e / max(float(np.linalg.norm(e)), 1e-9)
        L_ = getattr(self, '_dline', None)
        if L_ is None:
            cq_, sq_ = math.cos(float(q[2])), math.sin(float(q[2])); up_ = np.asarray(self.u_prev[:2], float)
            vw_ = np.array([cq_ * up_[0] - sq_ * up_[1], sq_ * up_[0] + cq_ * up_[1]])
            G_ = MM.merge_geom(q[:2], vw_ if float(np.linalg.norm(vw_)) > 0.03 else np.zeros(2), b0, e, s_cap=float(ln_.get('s_cap', -0.2)))
            v0_ = float(np.clip(np.linalg.norm(vw_), 0.0, 0.2))
            law_ = MM.time_law(G_, v0_, v_, float(ln_.get('s_slow', -0.3)), v_hi_cap=float(ln_.get('v_hi', 0.15)))
            L_ = self._dline = dict(G=G_, law=law_, tr0=tr, y0=float(q[2]), sig_rot=G_['Lc'] + 0.2, sig_end=MM.sig_of_s(G_, float(ln_.get('s_end', 0.25))), w=max(v_, 0.02) * 1.0)
            self._ev('door_line_start', lat=round(G_['lat'], 3), s_p=round(G_['s_p'], 3), s_m=round(G_['s_m'], 3), Lc=round(G_['Lc'], 3), v0=round(v0_, 3),
                     t_pass=round(MM.t_at_sig(law_, MM.sig_of_s(G_, 0.0), v_), 2), T_app=round(float(ap_['T']), 2))
        G_, law_ = L_['G'], L_['law']

        def sig_(tt):
            sg = MM.sig_at_t(law_, max(0.0, tt - L_['tr0']), v_)
            s1 = L_['sig_end'] - L_['w']
            return sg if sg <= s1 else L_['sig_end'] - L_['w'] * math.exp(-(sg - s1) / L_['w'])   # 走廊盡頭平滑停
        te0 = tr + D_STEPS * PHYS
        Xn = []
        for k in range(XO.N + 1):
            sg = sig_(te0 + k * DT); p_ = MM.at(G_, sg)
            Xn.append(XO.from_xytheta(float(p_[0]), float(p_[1]), MM.yaw_at(G_, sg, L_['y0'], b0[2], L_['sig_rot'])))
        xin = []
        for k in range(len(Xn) - 1):
            T1 = np.linalg.inv(Xn[k]) @ Xn[k + 1]
            xin.append(np.array([T1[0, 2] / DT, T1[1, 2] / DT, math.atan2(T1[1, 0], T1[0, 0]) / DT]))
        xin.append(xin[-1].copy())
        return Xn, xin, obs, float('nan'), 'door_line'

    def _nav_window_base(self, q, qp):
        """導航前端本拍輸入（相對擷取當時機器人位姿）→ 世界參考視窗、障礙物、速度下限；過期 ⇒ 在預測位姿保持（零速度）。"""
        N = XO.N
        ni = getattr(self, 'nav_in', None)
        if ni is not None and not ni.get('stale') and ni.get('halted'):   # 導航到點／無計畫：障礙物照原規則持續提供；參考由任務端（收斂到抓取底盤位姿或原地）
            c0, s0 = math.cos(q[2]), math.sin(q[2]); obs_h = []
            for o_ in ni.get('obs', []):
                d_ = dict(x=float(q[0] + c0 * o_[0] - s0 * o_[1]), y=float(q[1] + s0 * o_[0] + c0 * o_[1]), radius=float(o_[2]), vx=float(c0 * o_[3] - s0 * o_[4]), vy=float(s0 * o_[3] + c0 * o_[4]))
                if math.isfinite(o_[5]):
                    d_['margin'] = float(o_[5])
                if o_[6] > 0.5:
                    d_['static'] = True
                obs_h.append(d_)
            self._obs_last = (self.t, door_filter(obs_h, getattr(self, 'app_prof', None), getattr(self, 'phi_now', None)))
            ni = None
        if ni is not None and not ni.get('stale'):
            X0 = np.eye(3); c0, s0 = math.cos(q[2]), math.sin(q[2]); X0[:2, :2] = [[c0, -s0], [s0, c0]]; X0[:2, 2] = q[:2]
            rel = np.asarray(ni['X_rel'], float).reshape(-1, 3); xin = np.asarray(ni['xi_ref'], float).reshape(-1, 3)
            sh = int(round(D_STEPS * PHYS / DT))                       # 導航視窗起點＝擷取時刻；本求解初態在管線之後 ⇒ 對齊移 sh 步
            X, xi = [], []
            for k in range(N + 1):
                j = min(k + sh, len(rel) - 1)
                cr, sr = math.cos(rel[j, 2]), math.sin(rel[j, 2])
                Rj = np.eye(3); Rj[:2, :2] = [[cr, -sr], [sr, cr]]; Rj[:2, 2] = rel[j, :2]
                P = X0 @ Rj
                X.append(XO.from_xytheta(float(P[0, 2]), float(P[1, 2]), math.atan2(P[1, 0], P[0, 0])))
                xi.append(xin[min(j, len(xin) - 1)])
            obs = []
            for o_ in ni.get('obs', []):
                ox, oy = q[0] + c0 * o_[0] - s0 * o_[1], q[1] + s0 * o_[0] + c0 * o_[1]
                d_ = dict(x=float(ox), y=float(oy), radius=float(o_[2]), vx=float(c0 * o_[3] - s0 * o_[4]), vy=float(s0 * o_[3] + c0 * o_[4]))
                if math.isfinite(o_[5]):
                    d_['margin'] = float(o_[5])
                if o_[6] > 0.5:
                    d_['static'] = True
                obs.append(d_)
            obs = door_filter(obs, getattr(self, 'app_prof', None), getattr(self, 'phi_now', None))
            self._obs_last = (self.t, obs)
            return X, xi, obs, float(ni.get('u_min0', float('nan'))), 'nav'
        ol = getattr(self, '_obs_last', None)
        obs = ol[1] if (ol is not None and self.t - ol[0] <= 0.5) else None   # 過期（>0.5 s）⇒ None：呼叫端停止，不在無障礙資料下續算
        ap_ = getattr(self, 'app_prof', None)
        bg = None if ap_ is None else ap_.get('base_goal')
        if bg is not None:                                             # 導航到點（前端停止更新）後：同一控制器把底盤帶到抓取底盤位姿（慢速直線、減速收斂；不假設沿門）
            bg = np.asarray(bg, float); X, xi = [], []
            p = np.asarray(qp[:3], float).copy()
            for k in range(N + 1):
                d_ = bg[:2] - p[:2]; dist = float(np.linalg.norm(d_)); dth = (bg[2] - p[2] + math.pi) % (2 * math.pi) - math.pi
                v = min(BG_V, 0.8 * dist / max(DT, 1e-6) * DT / 0.5)   # 約 0.5 s 時間常數減速、上限 BG_V
                w = max(-0.3, min(0.3, dth / 0.5))
                X.append(XO.from_xytheta(float(p[0]), float(p[1]), float(p[2])))
                vw_ = (v * d_ / dist) if dist > 1e-6 else np.zeros(2)
                c_, s_ = math.cos(p[2]), math.sin(p[2])
                xi.append(np.array([c_ * vw_[0] + s_ * vw_[1], -s_ * vw_[0] + c_ * vw_[1], w]))
                p = p + DT * np.array([vw_[0], vw_[1], w])
            return X, xi, obs, float('nan'), 'base_goal'
        X = [XO.from_xytheta(*qp[:3])] * (N + 1)
        return X, [np.zeros(3)] * (N + 1), obs, float('nan'), 'hold'

    def _track_unified(self, q, tr):
        """單一控制器（Codex 20261117_161844；Howard：導航到操作同一個控制器）：每拍同一個 gmpc_wb 實例；底盤參考與障礙物＝導航前端原樣輸入，
        CBF 全階段；手臂：導航時保持，接近開始後依剖面加入 TCP 任務（觀測觸發），閉爪後 s＝0。底盤不假設沿門直線。回 (ub, ua, 殘差或原因, None)。"""
        mpc = self.shared_mpc()
        w0 = time.perf_counter()
        qp, sp = self._pred(q)
        te0 = tr + D_STEPS * PHYS
        X, xi, obs, umin0, src = self.nav_window(q, qp)
        if obs is None:
            return np.zeros(3), None, 'nav_obstacles_stale', None             # 障礙物資料過期 ⇒ 停止（原停止處置；Codex 20261117_173459）
        self._xref_cache = (te0, [(float(Xk[0, 2]), float(Xk[1, 2]), math.atan2(float(Xk[1, 0]), float(Xk[0, 0]))) for Xk in X])
        ap_ = getattr(self, 'app_prof', None)
        if ap_ is None:
            arm = dict(mode='hold')
            Ts = None
        else:
            if ap_.get('T_rel0') is None:
                ap_['T_rel0'] = (np.linalg.inv(_T3_(qp[:3])) @ K.fk(qp, 'link_tcp')).tolist()
            Ts = [self.target(te0 + k * DT) for k in range(XO.N + 1)]
            in_blend = (te0 - float(ap_.get('t0', 0.0))) < float(ap_['T_b'])
            arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts), w_p=C0.W_FAR if in_blend else C0.W_NEAR,
                       lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper, tcp_rel_base=False, qp_solver='gi', **B.W_ARM)
            if ap_.get('line') is not None and getattr(self, 'w_post', 0.0) > 0.0 and in_blend:   # 走廊經過（s155d：沿走廊移動中逐拍接續 IK ⇒ j6 一路轉到限位）：固定引導到「抓取底盤位 b0 上的門旁抓取解」
                if getattr(self, '_q_line', None) is None:
                    b0_ = np.asarray(ap_['line']['b0'], float)
                    qg_, rg_ = self._ik(b0_, self.T_at(self.handle_ref(), float(ap_.get('s_final', 0.0))), np.array([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157]))
                    self._q_line = np.asarray(qg_, float).copy() if (rg_ < 1e-6 and np.isfinite(qg_).all()) else False
                    self._ev('door_line_qpost', ok=self._q_line is not False, resid=round(float(rg_), 7), q=None if self._q_line is False else [round(float(x), 3) for x in self._q_line])
                if self._q_line is not False:
                    arm.update(q_post=np.tile(self._q_line, (XO.N, 1)), w_q=float(self.w_post))
            elif in_blend and getattr(self, 'w_post', 0.0) > 0.0:       # 門旁分支姿態偏好（只引導分支）：時域末端目標在末端參考底盤上的門旁 IK
                seed_ = self.q_post if getattr(self, 'q_post', None) is not None else np.array([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157])
                bN = np.array(self._xref_cache[1][-1])
                qN, rN = self._ik(bN, Ts[-1], seed_)
                if rN < 1e-6 and np.isfinite(qN).all() and float(np.abs(qN - seed_).max()) <= 0.5:
                    self.q_post = np.asarray(qN, float).copy()
                if getattr(self, 'q_post', None) is not None:
                    arm.update(q_post=np.tile(self.q_post, (XO.N, 1)), w_q=float(self.w_post))
        u_min_saved = np.asarray(mpc.cfg.u_min, float).copy()
        if math.isfinite(umin0):
            mpc.cfg.u_min = np.array([umin0, u_min_saved[1], u_min_saved[2]])   # 導航繞行速度下限（同導航節點）
        try:
            r = mpc.solve(X_now=XO.from_xytheta(*qp[:3]), X_ref_win=np.array(X), xi_ref_win=np.array(xi), xi_prev=self.u_prev[:3], arm=arm, obstacles=obs)
        finally:
            mpc.cfg.u_min = u_min_saved
        ck = dict(r.arm_check or {})
        ms = (time.perf_counter() - w0) * 1e3
        e_tcp = None if Ts is None else round(float(np.linalg.norm(K.fk(qp, 'link_tcp')[:3, 3] - Ts[0][:3, 3])) * 1e3, 3)
        self.wg_log.append((round(self.t, 3), round(ms, 2), e_tcp, str(r.status), src, 0 if obs is None else len(obs), int(r.cbf_active)))
        if not np.isfinite(np.asarray(r.u_opt, float)).all():
            return np.zeros(3), None, 'wgmpc_nonfinite', None
        if ap_ is None:
            return np.asarray(r.u_opt, float), np.zeros(6), 0.0, None   # 導航期：手臂保持（同一求解，arm hold）
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None or r.status != 'solved':
            return np.asarray(r.u_opt, float), None, 'wgmpc:%s/%s/%s' % (r.status, ck.get('fallback'), ck.get('failed')), None
        return np.asarray(r.u_opt, float), np.asarray(r.u_arm_seq[0], float), 0.0, None

    def _track_wgmpc(self, q, tr):
        """W-GMPC 主導（Howard 20261117 主線修正）：底盤與手臂命令由同一個 gmpc_wb 全身問題解出，求解後不改。
        初態＝在途管線前推後的預測狀態（命令生效時刻）；底盤參考＝沿門方向 d 的直線（過 b_origin、偏航 yaw_ref），以 v 前進；
        TCP 序列＝當下排程目標 target(t_eff＋k·dt)（插入剖面 s(t)；閉爪後 s＝0）。手臂名目由求解器熱啟動（同瓶子鏈），不另解 IK。
        求解失敗、回退、序列不一致 ⇒ 回 (None 手臂)，由呼叫端走原停止處置。回 (ub, ua, 殘差或原因, None)。"""
        if getattr(self, 'mpc', None) is None:
            nc = B.nav_cfg()
            rc = XO.GR.GMPCConfig(**{f.name: getattr(nc, f.name) for f in XO.dataclasses.fields(nc)})
            rc.cmd_ramp_w = (0.4, 0.6) if XO.BASE_RAMP else None
            qb = np.asarray(getattr(self, 'base_q_diag', (100.0, 10.0, 50.0)), float)   # 本體 x（對門側向）與偏航加重、沿門（本體 y）照原值（c32／c33：原 10/10/5 時底盤被拖著補 TCP 誤差，側向偏 8 cm）
            rc.Q = np.diag(qb); rc.Qf = np.diag(5.0 * qb)
            rc.wheel_w_max, rc.wheel_a_max = rc.wheel_w_max * XO.WHEEL_FRAC[0], rc.wheel_a_max * XO.WHEEL_FRAC[1]
            self.mpc = XO.GR.GMPC(rc)
            self.wg_log = []
        w0 = time.perf_counter()
        qp, sp = self._pred(q)
        te0 = tr + D_STEPS * PHYS
        along = float((qp[:2] - self.b_origin[:2]) @ self.d) if getattr(self, 'app_prof', None) is None else self.v * te0   # 接近剖面：沿門位置錨定時間計畫（與 TCP 參考的名目底盤同一時間軸；c34：未錨定時底盤被往前推、READY 超過抓取點 7.7 cm）
        th = float(self.yaw_ref)
        vw = self.v * self.d
        c_, s_ = math.cos(th), math.sin(th)
        xi_b = [c_ * vw[0] + s_ * vw[1], -s_ * vw[0] + c_ * vw[1], 0.0]
        X, xi, Ts = [], [], []
        for k in range(XO.N + 1):
            pos = self.b_origin[:2] + self.d * (along + self.v * k * DT)
            X.append(XO.from_xytheta(float(pos[0]), float(pos[1]), th)); xi.append(xi_b)
            Ts.append(self.target(te0 + k * DT))
        arm = dict(mode='active', K=K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=self.cfg.arm_model, T_seq=np.array(Ts), w_p=C0.W_NEAR,
                   lo=XO.LITE6_SAFE.lower, hi=XO.LITE6_SAFE.upper, tcp_rel_base=False, qp_solver='gi', **B.W_ARM)
        ap_ = getattr(self, 'app_prof', None)
        in_blend = ap_ is not None and te0 < float(ap_['T_b'])
        if in_blend:
            arm['w_p'] = C0.W_FAR                                    # 混接段（遠）：TCP 權重同瓶子鏈遠段 50；近段 200（c32：200 時底盤被拖著補 TCP 誤差，側向偏 9 cm）
        if in_blend and getattr(self, 'w_post', 0.0) > 0.0:                        # 門旁分支姿態偏好（c29：從導航姿態追 TCP 收斂到另一分支，開門首段不可行）：時域末端目標在末端名目底盤上的門旁 IK（種子＝上一解或門旁構型）
            seed_ = self.q_post if getattr(self, 'q_post', None) is not None else np.array([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157])
            bN = np.r_[self.b_origin[:2] + self.d * (along + self.v * XO.N * DT), th]
            qN, rN = self._ik(bN, Ts[-1], seed_)
            if rN < 1e-6 and np.isfinite(qN).all() and float(np.abs(qN - seed_).max()) <= 0.5:
                self.q_post = np.asarray(qN, float).copy()
            if getattr(self, 'q_post', None) is not None:
                arm.update(q_post=np.tile(self.q_post, (XO.N, 1)), w_q=float(self.w_post))
        r = self.mpc.solve(X_now=XO.from_xytheta(*qp[:3]), X_ref_win=np.array(X), xi_ref_win=np.array(xi), xi_prev=self.u_prev[:3], arm=arm, obstacles=None)
        ck = dict(r.arm_check or {})
        ms = (time.perf_counter() - w0) * 1e3
        Tc = K.fk(qp, 'link_tcp')
        self.wg_log.append((round(self.t, 3), round(ms, 2), round(float(np.linalg.norm(Tc[:3, 3] - Ts[0][:3, 3])) * 1e3, 3), str(r.status), str(ck.get('fallback'))))
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False) or r.u_arm_seq is None or r.status != 'solved':
            return np.asarray(r.u_opt, float), None, 'wgmpc:%s/%s/%s' % (r.status, ck.get('fallback'), ck.get('failed')), None
        return np.asarray(r.u_opt, float), np.asarray(r.u_arm_seq[0], float), 0.0, None

    def _track_ik(self, q, tr):
        """r11：本拍追蹤命令（純計算，不改狀態）：底盤沿 d 以 v＋側向／偏航回授（加速度限），手臂以命令生效後一拍末的預測底盤解 IK。"""
        qp, sp = self._pred(q)
        yaw_err = (self.yaw_ref - qp[2] + math.pi) % (2 * math.pi) - math.pi
        lat = float(np.cross(np.r_[self.d, 0.0], np.r_[qp[:2] - self.b_origin[:2], 0.0])[2])
        nrm = np.array([-self.d[1], self.d[0]])
        vw = self.v * self.d - 1.0 * lat * nrm
        cth, sth = math.cos(qp[2]), math.sin(qp[2])
        ub = np.array([cth * vw[0] + sth * vw[1], -sth * vw[0] + cth * vw[1], 1.5 * yaw_err])
        ub = np.clip(ub, LIM.BODY_V_MIN, LIM.BODY_V_MAX)
        ub = np.clip(ub, self.u_body_prev - LIM.A_BODY * DT, self.u_body_prev + LIM.A_BODY * DT)
        te = tr + D_STEPS * PHYS + DT
        seed = self.q_ik_prev if self.q_ik_prev is not None else qp[3:]
        ua_, res_, q1 = self._arm_track(qp, sp, np.asarray(self.pipe[-1][:3], float).copy(), ub, te, seed)
        return ub, ua_, res_, q1

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

    def _rec_for(self, e):
        """r15（非阻塞執行）：工作程序設 self._rec_tail＝{extra: []} 時，停止模擬逐步輸出寫進對應清單（不影響任何判定）。"""
        rt = getattr(self, '_rec_tail', None)
        return None if rt is None else rt.get(e)

    def _contact_stop_traj(self, q, u0, extra, tr, j0, u_from, rec=None):
        """r4：接觸停法的向前模擬（真管線、拍內斜坡、每拍：底盤有界減速、手臂以預測底盤 IK 追鎖定把手）。回 (Q 取樣, 違反或 None, 減速起點)。"""
        qp, sp = np.asarray(q, float).copy(), self.s.copy()
        pp = [np.asarray(x, float).copy() for x in self.pipe]
        u, uf, j = np.asarray(u0, float).copy(), np.asarray(u_from, float).copy(), int(j0)
        u_prev = u.copy()
        dab = B.A_BASE * self.cfg.dt
        out, k, quiet, k_dec, n_tick = [qp.copy()], 0, 0, None, 0
        nzq = [bool(np.any(x[:3])) for x in pp]                      # r11：管線各筆底盤是否非零（與 pp 同步進出）
        n_nz = sum(nzq)
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
            nzc = bool(np.any(cmd[:3]))
            nzq.append(nzc)
            n_nz += int(nzc) - int(nzq.pop(0))
            pc_ = pp.pop(0)
            qp, sp = S.plant_phys_step(qp, sp, pc_, self.cfg, 1)
            if rec is not None:
                rec.append((sp.copy(), pc_[:3].copy()))
            k += 1
            j += 1
            settled = (not nzc and n_nz == 0 and float(np.abs(cmd[3:]).max()) <= CSTOP['arm_cmd'])
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

    def _geoms(self):
        """r12：要評估的手指幾何——實測（或模型）；CLOSE 段另加預測閉合開度。"""
        g = [self.finger_q]
        if self.phase == 'CLOSE':
            g.append((FINGER_CLOSED, FINGER_CLOSED))
        return g

    def clear(self, q9, phi, exempt):
        out = None
        fq_save = self.finger_q
        try:
            for gq in self._geoms():
                self.finger_q = gq
                W.FINGER_Q = gq
                c = W.clear_door(np.r_[q9, phi], self.T_WH_model)
                c['finger_bar'] = XE.DoorExitCtrl.finger_bar(self, q9, phi)
                out = c if out is None else {k: (min(out[k], c[k]) if k != 'who' else out[k]) for k in out}
        finally:
            self.finger_q = fq_save
            W.FINGER_Q = fq_save
        return out

    def _bad(self, c, exempt):
        """exempt：'insert'＝兩指—橫桿只要求不穿入（張開插入）；'contact'＝閉爪起兩指—橫桿為指定接觸（同靜止版 contact=True），只核夾爪殼；None＝全部照門檻。"""
        b_ = [k for k in KEYS if c[k] < W.THR[k]]
        if exempt is None and c['finger_bar'] < W.THR['env']:
            b_.append('finger_bar')
        elif exempt == 'insert' and c['finger_bar'] < 0.0:
            b_.append('finger_bar')
        return b_

    def close_pool(self):
        if self._pool is not None and self._pool_owned:
            self._pool.close()

    def _stop_traj(self, q, u, e, j0, u_from):
        """r4：接觸段（已鎖定）用接觸停法；其餘用原停法。回 (Q, 違反或 None, k_dec)。"""
        if self.lock is not None:
            return self._contact_stop_traj(q, u, e, self.t - self.t0, j0, u_from, rec=self._rec_for(e))
        Q, cmds, k_dec = XE.stop_traj_exit(q, self.s, self.pipe, u, u_from, j0, self.cfg, e, rec=self._rec_for(e))
        return Q, None, k_dec

    def _sweep_pool(self, q, u, extras, j0, u_from, phi, exempt, tag):
        sub = self._sweep_pool_submit(q, u, extras, j0, u_from, phi, exempt, tag)
        if sub[0] == 'viol':
            return sub[1]
        return self._sweep_pool_finish(sub[1], sub[2], exempt, tag)

    def _sweep_pool_submit(self, q, u, extras, j0, u_from, phi, exempt, tag):
        """r11：準備（停止軌跡）＋派工。回 ('viol', 違反) 或 ('h', items, handle)。"""
        items, X, ends = [], [], {}
        for e in extras:
            Q, vtraj, k_dec = self._stop_traj(q, u, e, j0, u_from)
            if vtraj is not None:
                return ('viol', dict(tag=tag, extra=e, traj=vtraj, k_decel=k_dec))
            if (tag == 'stop_now' and e == 0) or (tag == 'candidate' and e == 1 and getattr(self, 'tail_cert_mode', False)):
                self.stop_end_s = float((np.asarray(Q[-1][:2]) - self.b_origin[:2]) @ self.d)
            idx = list(range(0, len(Q), self.sweep_stride)) + [len(Q) - 1]
            for i in idx:
                items.append((e, i, len(X), k_dec))
                X.append(np.r_[Q[i], phi])
        hs = []
        gqs = self._geoms()
        if len(gqs) > 1 and getattr(self, 'multi_geom', False):       # 非阻塞版（11/17）：多組手指幾何一個工作、各項取最小（door_clear_door_fast_multi，與逐組工作再合併逐值相同）
            hs.append(self._pool.submit(self.T_WH_model, dict(multi=list(gqs)), W.PHI0, np.array(X), finger_bar_q=list(gqs)))
        else:
            for gq in gqs:                                           # r12：每種手指幾何一個工作
                hs.append(self._pool.submit(self.T_WH_model, gq, W.PHI0, np.array(X), finger_bar_q=gq))
        W.FINGER_Q = self.finger_q
        return ('h', items, hs)

    def _sweep_pool_finish(self, items, hs, exempt, tag):
        res = None
        for h in hs:
            r_, why, tms = self._pool.collect(h, XO.POOL_TIMEOUT_S)
            self.pool_ms.append(tms)
            if r_ is None:
                res = None
                break
            res = r_ if res is None else [{k: (min(a[k], b[k]) if k != 'who' else a[k]) for k in a} for a, b in zip(res, r_)]
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
            if (tag == 'stop_now' and e == 0) or (tag == 'candidate' and e == 1 and getattr(self, 'tail_cert_mode', False)):
                self.stop_end_s = float((np.asarray(Q[-1][:2]) - self.b_origin[:2]) @ self.d)   # r2：現在停的停止終點沿 d；尾段證書模式＝本拍候選 e＝1（與計算池版本一致）
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

    def bias_obs_update(self, hist):
        """手臂等效偏置觀測（Codex 第八趟裁定；offset-free MPC 的擾動估測方向，不宣稱無偏保證）。
        hist＝[(n, q_arm, s_reported)] 連續 NPH＋1 筆實測；第 n→n+1 步作用的設定點＝訊息 n+1 回報的設定點。
        IntervalOffsetObserver（樣本外：先用更新前估計預測再更新；τ 0.5 s）；新增殘差補償 δ＝d−d0 每軸限幅 ±BIAS_OBS_MAX；
        只在接近開始後、閉爪前更新（閉爪開始即凍結，承接到開門）；估計寫進共同模型 cfg.arm_model.bias＝α·d（QP、延遲前推、停止核對同一份）。"""
        if not BIAS_OBS or getattr(self, 'app_prof', None) is None or self.lock is not None or self.phase in ('CLOSE', 'CONFIRM', 'READY') or self.stop is not None:
            return
        n1 = int(hist[-1][0])
        if getattr(self, '_bo_last_n', None) is not None and n1 <= self._bo_last_n:
            return                                                    # 同一快照只更新一次
        self._bo_last_n = n1
        al = np.asarray(self.cfg.arm_model.alpha, float)
        if getattr(self, '_bo', None) is None:
            import offset_interval as OI
            self._bo_d0 = np.asarray(self.cfg.arm_model.bias, float) / al
            self._bo = OI.IntervalOffsetObserver(al, PHYS, tau_s=BIAS_OBS_TAU, max_rad=0.3, d0=self._bo_d0)
            self.bias_log = []
        q0, q1 = hist[0][1], hist[-1][1]
        s_seq = [h_[2] for h_ in hist[1:]]
        rec = self._bo.update_setpoints(q0, s_seq, q1)
        if rec.get('used') == 'updated':
            dd = np.clip(self._bo.d - self._bo_d0, -BIAS_OBS_MAX, BIAS_OBS_MAX)
            self._bo.d = self._bo_d0 + dd
            self.cfg.arm_model.bias = al * self._bo.d
        self.bias_log.append((round(self.t, 3), rec.get('used'), None if rec.get('pred_residual') is None else round(float(np.abs(rec['pred_residual']).max()) * 1e3, 4), [round(float(x) * 1e3, 3) for x in (self._bo.d - self._bo_d0)]))

    def _s_nom(self):
        """名目抓姿的接近軸深度留量（m）：接近剖面的 s_final；無剖面＝0。"""
        ap = getattr(self, 'app_prof', None)
        return float(ap.get('s_final', 0.0)) if ap is not None else 0.0

    def _rel(self, q):
        """控制器視角的夾爪相對把手（估計）誤差：位置 m、姿態 °（TCP 對 T_at(估計, s＝0)）。"""
        Tc, Tg = K.fk(q, 'link_tcp'), self.T_at(self.handle_ref(), self._s_nom())      # r4：鎖定後對鎖定參考；名目抓姿含深度留量（Codex 192427：與接近目標一致）
        dR = Tg[:3, :3].T @ Tc[:3, :3]
        return float(np.linalg.norm(Tc[:3, 3] - Tg[:3, 3])), math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))), Tc

    def step(self, t, q, phi_meas, finger_meas=None):
        self.t = float(t)
        self._finger_meas = None if finger_meas is None else (float(finger_meas[0]), float(finger_meas[1]))   # r12：兩指實測開度（幾何用）
        if self._finger_meas is not None:
            self.finger_q = self._finger_meas
        q = np.asarray(q, float)
        if self.abort is not None:
            return dict(arm_target=self.s.copy(), base_twist=np.zeros(3), finger=self.finger_cmd, ready=False, stopped=True, abort=True)
        tick = self.n_step % NPH == 0
        w_t = time.perf_counter()
        if tick:
            self._fb_ok = False
            self._ikc = {}
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
            keep = {k_: v_ for k_, v_ in self.frames.items() if v_[0] >= self.t - FUSE['win_s'] - 1e-9} if self.lock_budget_fn is None else dict(self.frames)   # r14 鎖定模式：鎖定參考不因融合時間窗過期而被丟掉（時間戳照原樣）
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
            if self.lock is not None and not self.accepted and self.stop is None and abs(float(phi_meas) - self.lock['phi']) > math.radians(0.5):
                self._stop('door_moved_before_bind', door_change_deg=round(math.degrees(float(phi_meas) - self.lock['phi']), 4))   # r12：同 BindGate 基準
            if self.stop is not None and self.stop_ok is None and self.abort is None and self.t - self.stop_t > 4.0:
                self._abort('stop_not_settled_4s', since_stop_s=round(self.t - self.stop_t, 3))   # r12：不無限追門；停止有效性未驗證
            fresh = self.est_age is not None and self.est_age <= READY_Q['fresh_s'] + 1e-9
            self.phi_now = float(phi_meas)
            self.T_WH_model = self.handle_ref()                        # 淨空模型的把手＝估計（r4：鎖定後＝鎖定參考）；門角 φ 由實測給
            W.set_model(phi0=float(phi_meas))
            W.FINGER_Q = self.finger_q
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
            if sd > self.win[1] + 1e-9 or (sd < self.win[0] - 1e-9 and self.phase != 'HOLD'):   # r9：HOLD（預抓保持、未接觸）可在窗口前（前導接近）
                viol.append(('outside_window', dict(s_along=round(sd, 4), window=self.win)))
            ep, er, Tc = self._rel(q)
            Tg_now = self.T_at(self.handle_ref(), self._s_nom())
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
                    return dict(arm_target=self.s.copy(), base_twist=np.zeros(3), finger=self.finger_cmd, ready=False, stopped=True, abort=True)
            # 段別
            if self.lock_budget_fn is not None:                        # r14：鎖定預算每拍重算（手臂項用本控制器的 TCP 相對位姿）
                Ts_ = self.target(tr)                                     # 手臂項＝TCP 對「當下排程目標」（HOLD／INSERT 時是後退 s(t) 的位姿，不是抓取位；far_door_4 試跑誤算成 33 mm）
                dRs = Ts_[:3, :3].T @ Tc[:3, :3]
                ep_s = float(np.linalg.norm(Tc[:3, 3] - Ts_[:3, 3])); er_s = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dRs) - 1) / 2))))
                self.lock_budget = self.lock_budget_fn(self.t, q, ep_s, er_s)
                fresh = bool(self.lock_budget.get('ok'))
                if self.lock_budget_max is None or self.lock_budget['pos_m'] > self.lock_budget_max['pos_m']:
                    self.lock_budget_max = dict(self.lock_budget, t=round(self.t, 3), phase=self.phase)
                if not fresh and self.stop is None and self.phase in ('CLOSE', 'CONFIRM', 'READY') and not self.accepted:
                    self._stop('lock_budget_exceeded_in_contact', budget={k_: self.lock_budget.get(k_) for k_ in ('pos_m', 'ang_deg')})   # 已接觸 ⇒ 既有接觸停法
            if self.stop is None:
                h, i, cl, cf = (self.sched[k] for k in ('hold_s', 'ins_s', 'close_s', 'confirm_s'))
                bud_bad = self.phase == 'INSERT' and tr >= h + i and self.lock_budget_fn is not None and not fresh
                gate_bad = self.phase == 'INSERT' and tr >= h + i and self.close_gate is not None and not self.close_gate(self)
                if (bud_bad or gate_bad) and CLOSE_WAIT_S > 0.0:      # 閉爪準備窗口（Codex 20261117 第七趟裁定）：保持 s_final、繼續閉迴路追蹤、不閉爪；總期限自第一次未過起算、不重置
                    if self.close_wait_t0 is None:
                        self.close_wait_t0 = self.t
                        self._ev('close_wait_start', budget_ok=not bud_bad, gate_ok=not gate_bad, budget={k_: self.lock_budget.get(k_) for k_ in ('pos_m', 'ang_deg')} if self.lock_budget else None)
                    if self.t - self.close_wait_t0 > CLOSE_WAIT_S + 1e-9:
                        if bud_bad:
                            self._stop('lock_budget_exceeded_no_close', budget={k_: self.lock_budget.get(k_) for k_ in ('pos_m', 'ang_deg')}, waited_s=round(self.t - self.close_wait_t0, 3))
                        else:
                            self._stop('close_gate_not_ready', waited_s=round(self.t - self.close_wait_t0, 3))
                elif bud_bad:
                    self._stop('lock_budget_exceeded_no_close', budget={k_: self.lock_budget.get(k_) for k_ in ('pos_m', 'ang_deg')})     # 超預算不閉爪
                elif self.phase == 'HOLD' and tr >= h:
                    self._phase('INSERT')
                elif gate_bad:
                    self._stop('close_gate_not_ready')                 # r8：下游首段未就緒 ⇒ 不閉爪
                elif self.phase == 'INSERT' and tr >= h + i:
                    if self.close_wait_t0 is not None:
                        self.close_delay = self.t - self.close_wait_t0  # 後續排程（CLOSE→CONFIRM）以實際閉爪開始重算
                        self._ev('close_wait_done', waited_s=round(self.close_delay, 3))
                    self._phase('CLOSE')
                    self.lock = dict(T=self.T_est.copy(), phi=float(phi_meas), t=round(self.t, 3), est_ver=self.est_ver)   # r4：閉爪開始鎖定接觸任務參考
                    self._ev('contact_ref_locked', phi_deg=round(math.degrees(float(phi_meas)), 4), est_ver=self.est_ver)
                elif self.phase == 'CLOSE' and tr >= h + i + cl + self.close_delay:
                    self._phase('CONFIRM')
                rel_ok = ep <= CONFIRM['pos_m'] and er <= CONFIRM['rot_deg'] and lin <= CONFIRM['lin_mps'] and ang <= CONFIRM['ang_rps']
                if self.phase == 'READY':                              # r2：READY 每拍重判（交接被接受前）
                    why_r = [w_ for w_, ok_ in (('rel_motion', rel_ok), ('stale_estimate' if self.lock_budget_fn is None else 'lock_budget', fresh)) if not ok_]
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
                self.finger_cmd = max(0.0, self.finger_cmd - self.finger_open * DT / self.sched['close_s'])   # r12：命令斜降到 0（夾緊）
            if self._finger_meas is None:                              # 假世界：無手指物理 ⇒ 幾何＝max(命令, 閉合開度模型)
                self.finger_q = max(FINGER_CLOSED, self.finger_cmd)
            # 現在停（r11：有計算池時先派工，派工期間投機計算追蹤命令）
            exempt = 'insert' if self.phase == 'INSERT' else ('contact' if self.phase in ('CLOSE', 'CONFIRM', 'READY') else None)   # r12：段別轉換後重算（閉爪段掃掠含閉合幾何 ⇒ 兩指—橫桿為指定接觸）
            spec = None
            if self.stop is None and getattr(self, 'tail_cert_mode', False):
                pass                                                   # 非阻塞尾段證書模式（只在非阻塞執行器明確開啟）：晚到時執行的是上一段已核的 extra＝1 尾段，現在停（e＝0）軌跡從不執行 ⇒ 不重算；停止終點檢查改用本拍候選 e＝1 的軌跡終點
            elif self.stop is None and self._pool is not None:
                sn = self._sweep_pool_submit(q, self.u_last, (0,), self.ramp_j, self.u_from_base, phi_meas, exempt, 'stop_now')
                if sn[0] == 'viol':
                    self._stop('stop_now_violation', **sn[1])
                else:
                    spec = self._track(q, tr)
                    sw0 = self._sweep_pool_finish(sn[1], sn[2], exempt, 'stop_now')
                    if sw0 is not None:
                        self._stop('stop_now_violation', **sw0)
                        spec = None
                    else:
                        self._fb_ok = self.fb_inject is None or bool(self.fb_inject(self))
            elif self.stop is None:
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
                ub, ua_, res_, q1 = spec if spec is not None else self._track(q, tr)
                if ua_ is None:
                    self._stop('ik_fail', resid=res_)
                else:
                    self.q_ik_prev = q1
                    u = np.r_[ub, ua_]
                    bad = self._check_cmd(u)
                    if bad:
                        self._stop('cmd_limit', bad=bad)
                    else:
                        cert = getattr(self, 'tail_cert_mode', False)
                        sw = self._sweep(q, u, (1,) if cert else XE.STOP_EXTRAS, 0, np.asarray(self.pipe[-1][:3], float), phi_meas, exempt, 'candidate')   # 尾段證書模式：只核實際會執行的 extra＝1（e＝2「同命令再留一拍」在非阻塞下不會發生）
                        if cert and sw is None and self.phase == 'READY' and self.stop_end_s is not None and self.stop_end_s > self.win[1] + 1e-9:
                            sw = dict(tag='candidate', why='window_insufficient', stop_end_s=round(self.stop_end_s, 4), window=self.win)
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
            return dict(arm_target=self.s.copy(), base_twist=np.zeros(3), finger=self.finger_cmd, ready=False, ready_ver=None, stopped=True, abort=True)
        return dict(arm_target=self.s.copy(), base_twist=ua[:3].copy(), finger=self.finger_cmd, ready=self.ready_now, ready_ver=self.ready_ver, stopped=self.stop is not None, abort=False)

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
