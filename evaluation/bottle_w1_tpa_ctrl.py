#!/usr/bin/env python3
"""TPA v2：時間參數化的整段接近（Codex 20261111_013508 第 3 項；Howard 11/11 00:18）。版本化子類，基線＝bottle_w1_r1a_ctrl（多種子分支、攔下名目備妥）。
（v1 在世界座標做 TCP 五次 Hermite：起點速度＝底盤速度、時長＝底盤到位時間 9 s ⇒ 初速項 V0·T 約 3 m，軌跡衝過目標再折回，離線 IK 大量失敗；改 v2。）
計畫承接時（手臂在 U 展開中或 H 觀察保持）立即生成一條關節空間整段軌跡，經相對底盤參考執行，到 t_pre 直接切世界座標 PREGRASP，中間不經任何固定姿態、不在預抓構型零速停住：
  起點＝命令生效時刻的預測狀態：_stop_rest 的在途命令走完構型 qp 的手臂部分（含視向偏移）、最後手臂命令速度 u_last[3:9]。
  終點＝底盤到 x_pre（目標專屬車道、偏航 0）時本趟量測目標預抓位姿 T_pre 的 IK 解（以工作者回傳的分支 q_pre_lo 為種子）；
        終點關節速度 q̇_end＝J⁻¹·(−v_c x̂, 0)：TCP 在世界座標靜止、底盤以 v_c 前進 ⇒ 接 PREGRASP／INSERT（世界座標）時速度連續。
        預抓偏移（沿工具軸 6 cm）只是軌跡終端的幾何條件。
  形狀：q(t)＝q0＋v0·tb·φ(t/tb)＋m(t)（traj_make）：起點速度在 tb＝max(0.5, 3|v0|/a_arm) 內平滑衰減（多走 v0·tb/2，不隨 T 放大），
        主體 m 為由靜止到 (q1, q̇1) 的五次；各端加速度為零。
  時長 T：自 max(底盤以名目剖面到 x_pre 的時間, 1.0) 起以 0.1 s 加長，直到全程 |q̇| ≤ v_arm、|q̈| ≤ a_arm、關節離 LITE6_SAFE ≥ 0.05。
  底盤：x < x_pre 前沿名目剖面；手臂需要更久時把 x_slow 前的巡航速度降到 v_nav' 使到位時間＝T（base_profile）。
        底盤早到 x_pre 而手臂未切換時，S1 既有規則減速等手臂（_unfolded 在切換前為假，同時擋住 S1 在 x_blend 另起舊漸變）。
  執行：t < t_pre 時相對底盤參考＝FK(q(τ))；預覽超過 t_pre 的部分以「底盤沿車道 v_c 前進」換算世界固定 T_pre 的相對位姿；
        t ≥ t_pre 切 blending＝True、PREGRASP（世界 T_pre；既有容差與放行側向管閘門放行 INSERT）。
限制：軌跡依時間參數化；底盤被動態障礙拖慢時，到 t_pre 底盤未到 x_pre ⇒ 世界目標 T_pre 由全身 GMPC 追（底盤也被拉動），INSERT 仍受 PREGRASP 容差與側向管閘門。
"""
import math

import numpy as np

import bottle_s0_ctrl as C0
import bottle_w1_r1a_ctrl as R1A
from bottle_s0b_ctrl import W_ARM

LIM_M = 0.05


def _phi(s, d=0):
    """起點速度衰減形狀：φ(0)＝0、φ'(0)＝1、φ''(0)＝0、φ'(1)＝0、φ''(1)＝0、φ(1)＝1/2。"""
    if s >= 1.0:
        return (0.5, 0.0, 0.0)[d]
    return (s - s ** 3 + 0.5 * s ** 4, 1.0 - 3 * s ** 2 + 2 * s ** 3, -6 * s + 6 * s ** 2)[d]


def quintic_coef(q0, v0, q1, v1, T):
    """每軸五次：q(0)＝q0、q'(0)＝v0、q''(0)＝0、q(T)＝q1、q'(T)＝v1、q''(T)＝0。回傳係數 (6, n)，q(t)＝Σ c_k t^k。"""
    q0, v0, q1, v1 = (np.asarray(x, float) for x in (q0, v0, q1, v1))
    A = np.array([[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4], [6 * T, 12 * T ** 2, 20 * T ** 3]])
    b = np.vstack([q1 - q0 - v0 * T, v1 - v0, np.zeros_like(q0)])
    c345 = np.linalg.solve(A, b)
    return np.vstack([q0, v0, np.zeros_like(q0), c345])


def quintic_eval(C, t, d=0):
    ks = np.arange(6)
    if d == 0:
        w = t ** ks
    elif d == 1:
        w = np.r_[0.0, [k * t ** (k - 1) for k in ks[1:]]]
    else:
        w = np.r_[0.0, 0.0, [k * (k - 1) * t ** (k - 2) for k in ks[2:]]]
    return w @ C


def traj_make(q0, v0, q1, v1, T, tb):
    """q(t)＝q0＋v0·tb·φ(t/tb)＋m(t)：起點速度 v0 在 tb 內平滑衰減（位移 v0·tb/2，不隨 T 放大），主體 m 由靜止的五次到 (q1, v1)。
    （單一五次帶 v0 時，長時長下初速項的超越量約 0.19·v0·T：v0 0.9 rad/s、T 9 s 會多走 1.5 rad——v1 世界 Hermite 失敗的同一原因。）"""
    q0, v0 = np.asarray(q0, float), np.asarray(v0, float)
    tb = min(float(tb), float(T))
    C = quintic_coef(np.zeros_like(q0), np.zeros_like(q0), np.asarray(q1, float) - q0 - 0.5 * v0 * tb, v1, T)
    return dict(q0=q0, v0=v0, tb=tb, C=C, T=float(T))


def traj_eval(P, t, d=0):
    t = min(max(float(t), 0.0), P['T'])
    s = t / P['tb']
    if d == 0:
        return P['q0'] + P['v0'] * P['tb'] * _phi(s) + quintic_eval(P['C'], t)
    if d == 1:
        return P['v0'] * _phi(s, 1) + quintic_eval(P['C'], t, 1)
    return P['v0'] / P['tb'] * _phi(s, 2) + quintic_eval(P['C'], t, 2)


def tb_of(v0, a_lim):
    return max(0.5, 3.0 * float(np.max(np.abs(v0))) / float(a_lim))


def plan_joint(q0, v0, q1, v1, T_min, v_lim, a_lim, lo, hi, step=0.1, T_max=30.0):
    """從 T_min 起加長到全程速度、加速度、關節裕度都成立。回傳 (軌跡, 峰值比) 或 (None, 原因)。"""
    T = max(float(T_min), 0.5)
    tb = tb_of(v0, a_lim)
    while T <= T_max:
        P = traj_make(q0, v0, q1, v1, T, tb)
        ts = np.linspace(0.0, T, max(121, int(T / 0.02) + 1))
        Q = np.array([traj_eval(P, t) for t in ts])
        V = np.array([traj_eval(P, t, 1) for t in ts])
        A = np.array([traj_eval(P, t, 2) for t in ts])
        vr, ar = float(np.max(np.abs(V) / v_lim)), float(np.max(np.abs(A) / a_lim))
        mg = float(min((Q - lo).min(), (hi - Q).min()))
        if vr <= 1.0 and ar <= 1.0 and mg >= LIM_M:
            return P, dict(v_ratio=round(vr, 3), a_ratio=round(ar, 3), joint_margin=round(mg, 4), tb=round(P['tb'], 3))
        T += step
    return None, 'no_feasible_duration'


def end_state(K, base_end, T_pre, q_seed, v_c, arm_idx):
    """底盤在 base_end 時 T_pre 的 IK 解與「TCP 世界靜止、底盤沿 +x 以 v_c 前進」的關節速度。"""
    from ammr_wholebody_mpc.arm_pregrasp import solve_ik
    qf = np.r_[base_end, q_seed]
    r = solve_ik(K, qf, T_pre)
    if not (r.ok and r.pos_err < 1e-3 and r.rot_err < 1e-2):
        return None, None, 'ik_end'
    J = K.jacobian(r.q, 'link_tcp')[:, arm_idx]                       # 6×6（線速度、角速度；世界）
    c_, s_ = math.cos(base_end[2]), math.sin(base_end[2])
    tw = np.r_[-v_c * c_, -v_c * s_, 0.0, 0.0, 0.0, 0.0]               # 手臂須抵消底盤前進
    try:
        qd = np.linalg.solve(J, tw)
    except np.linalg.LinAlgError:
        return None, None, 'jacobian_singular'
    return r.q[arm_idx], qd, None


def base_time(sig, x_pre, x_slow, v_nav, v_c, v0):
    """名目剖面（x < x_slow：v_nav，其後 v_c；以 A_SLEW 加減速）下從 σ 到 x_pre 的時間。"""
    import bottle_s1_ctrl as S1
    t, x, v, dt, a = 0.0, float(sig), max(float(v0), 0.0), 0.05, S1.A_SLEW
    for _ in range(4000):
        if x >= x_pre:
            break
        vt = v_nav if x < x_slow else v_c
        v = min(vt, v + a * dt) if v < vt else max(vt, v - a * dt)
        x += v * dt
        t += dt
    return t


def base_profile(sig, x_pre, x_slow, v_nav, v_c, v0, T, v_floor=0.02):
    """手臂需要比底盤久（T > 名目到位時間）時降低底盤速度使到位時間＝T：先降 x_slow 前的巡航速度（下限 v_c），仍太早再把巡航與接近速度一起降（下限 v_floor）。
    回傳 (v_nav', v_c', 到位時間)。降到下限仍早到 ⇒ 到位時間 < T，由 S1 既有規則在 x_pre 減速等手臂。"""
    t_nom = base_time(sig, x_pre, x_slow, v_nav, v_c, v0)
    if T <= t_nom + 0.05:
        return float(v_nav), float(v_c), t_nom
    if base_time(sig, x_pre, x_slow, v_c, v_c, v0) >= T:
        lo, hi = float(v_c), float(v_nav)
        for _ in range(30):
            m = 0.5 * (lo + hi)
            lo, hi = (m, hi) if base_time(sig, x_pre, x_slow, m, v_c, v0) > T else (lo, m)
        return hi, float(v_c), base_time(sig, x_pre, x_slow, hi, v_c, v0)
    lo, hi = float(v_floor), float(v_c)
    if base_time(sig, x_pre, x_slow, lo, lo, v0) < T:
        return lo, lo, base_time(sig, x_pre, x_slow, lo, lo, v0)
    for _ in range(30):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if base_time(sig, x_pre, x_slow, m, m, v0) > T else (lo, m)
    return hi, hi, base_time(sig, x_pre, x_slow, hi, hi, v0)


def plan_tpa(K, qa0, qd0, base_end, T_pre, q_seed, sig, x_pre, x_slow, v_nav, v_c, v_b, arm_idx, lo, hi):
    """完整規劃（控制器與離線核對共用）：終點 IK／速度 → 關節軌跡 → 底盤剖面；接近速度被降時以 v_c' 重算終點速度再規劃一次。
    回傳 dict 或 (None, 原因)。"""
    q1, qd1, why = end_state(K, base_end, T_pre, q_seed, v_c, arm_idx)
    if q1 is None:
        return None, why
    t_base = base_time(sig, x_pre, x_slow, v_nav, v_c, v_b)
    P, info = plan_joint(qa0, qd0, q1, qd1, max(t_base, 1.0), W_ARM['v_arm'], C0.A_ARM, lo, hi)
    if P is None:
        return None, str(info)
    v_nav_eff, v_c_eff, t_arr = base_profile(sig, x_pre, x_slow, v_nav, v_c, v_b, P['T'])
    replan = False
    if v_c_eff < v_c - 1e-9:
        q1b, qd1b, why = end_state(K, base_end, T_pre, q_seed, v_c_eff, arm_idx)
        if q1b is None:
            return None, why
        P2, info2 = plan_joint(qa0, qd0, q1b, qd1b, P['T'], W_ARM['v_arm'], C0.A_ARM, lo, hi)
        if P2 is None:
            return None, str(info2)
        q1, qd1, P, info, replan = q1b, qd1b, P2, info2, True
        v_nav_eff, v_c_eff, t_arr = base_profile(sig, x_pre, x_slow, v_nav, v_c, v_b, P['T'])
    return dict(P=P, info=info, q1=q1, qd1=qd1, t_base=t_base, v_nav_eff=v_nav_eff, v_c_eff=v_c_eff, t_arrive=t_arr, replan=replan), None


START_PH = ('U', 'H')                                                   # 承接時手臂正在展開（U，含速度）或觀察保持（H）都直接接整段軌跡


class W1PlaceController(R1A.W1PlaceController):
    group = R1A.W1PlaceController.group + '_TPA2'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.tpa = None
        self._arm_idx = [self.K.dof_names.index(f'joint{i}') for i in range(1, 7)]

    def _gp_apply(self, g, res, q):
        super()._gp_apply(g, res, q)
        if self.r1['applied'] and self.tpa is None and self.judge.stop is None and self.vh_seq['ph'] in START_PH:
            self._tpa_start(np.asarray(q, float))

    def _vstep(self, q):
        if self.tpa is not None:
            return
        if self.vh_seq['ph'] in START_PH and self.r1['applied'] and self.judge.stop is None:
            self._tpa_start(np.asarray(q, float))
            return
        super(R1A.W1PlaceController, self)._vstep(q)                     # 跳過 r1a 的 R 段（由 TPA 取代）；TUCK／U／H 與視向照基線

    def _tpa_start(self, q):
        from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L
        qp, _, _ = self._stop_rest(q)
        qa0 = np.asarray(qp[3:9], float)                                 # 命令生效時刻構型（含視向偏移、在途命令走完）
        qd0 = np.asarray(self.u_last, float)[3:9]                        # 最後送出的手臂速度命令（設定點模型下即生效後的關節速度）
        y_lane = float(self.tlane['y_lane']) if self.tlane else 0.0
        base_end = np.array([float(self.x_pre), y_lane, 0.0])
        sig = float(getattr(self, 'sig_robot', qp[0]))
        t_lat = C0.D_STEPS * C0.PHYS
        sig_e = sig + float(self.v_b) * t_lat
        R, why = plan_tpa(self.K, qa0, qd0, base_end, np.array(self.task.T_pre, float), np.asarray(self.q_arm_ready, float), sig_e, float(self.x_pre), float(self.x_slow),
                          float(self.v_nav), float(self.v_c), float(self.v_b), self._arm_idx, L.lower, L.upper)
        if R is None:
            self.judge._stop(self.t, 'tpa_' + str(why))
            return
        P, info, q1, qd1, t_base = R['P'], R['info'], R['q1'], R['qd1'], R['t_base']
        T, v_nav_eff, v_c_eff, t_arr = P['T'], R['v_nav_eff'], R['v_c_eff'], R['t_arrive']
        t_e = float(self.t) + t_lat
        self.tpa = dict(t_e=t_e, T=T, P=P, q0=qa0, qd0=qd0, q1=q1, qd1=qd1, base_end=base_end, t_base=round(t_base, 3), t_arrive=round(t_arr, 3), v_nav_eff=round(v_nav_eff, 4),
                        v_c_eff=round(v_c_eff, 4), replan=R['replan'],
                        info=info, switched=False, sig=round(sig, 4), ph0=self.vh_seq['ph'], T_pre=np.array(self.task.T_pre, float),
                        dist_nominal_ready_end=round(float(np.abs(q1 - self.q_ready_nominal).max()), 4))
        self.d_static = np.array(self.obs.d, float)
        self.st.t_cruise0, self.st.t_pre = t_e, t_e + T
        self.vh_seq = dict(self.vh_seq, ph='TPA')
        self._ev('tpa_start', T=round(T, 3), t_base=round(t_base, 3), v_nav_eff=round(v_nav_eff, 4), v_c_eff=round(v_c_eff, 4), v_ratio=info['v_ratio'], a_ratio=info['a_ratio'],
                 qd0_max=round(float(np.abs(qd0).max()), 4), qd1_max=round(float(np.abs(qd1).max()), 4), sig=round(sig, 4), x_pre=round(float(self.x_pre), 4), ph0=self.tpa['ph0'])

    def _rel_target(self, tau):
        P = self.tpa
        if P is None:
            return super()._rel_target(tau)
        s = tau - P['t_e']
        if s <= P['T']:
            return self.K.fk(np.r_[0.0, 0.0, 0.0, traj_eval(P['P'], s)], 'link_tcp')
        be = P['base_end'] + np.array([float(P['v_c_eff']) * (s - P['T']), 0.0, 0.0])   # 預覽超過 t_pre：世界固定 T_pre 換到沿車道前進的底盤座標
        c_, s_ = math.cos(be[2]), math.sin(be[2])
        Tb = np.eye(4)
        Tb[:3, :3] = [[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]
        Tb[:3, 3] = [be[0], be[1], 0.0]
        return np.linalg.inv(Tb) @ P['T_pre']

    def _unfolded(self):
        if self.tpa is not None:
            return bool(self.tpa['switched'])                             # TPA 中擋住 S1 在 x_blend 另起舊漸變（會重設 t_pre 並改走 T_ready_rel→T_pre 內插）；底盤早到 x_pre 時 S1 規則減速等手臂
        return super()._unfolded()

    def _advance_inner(self, T_tcp, close_done, q):
        P = self.tpa
        if P is not None and not P['switched'] and P.get('time_switch', True) and self.st.phase == 'CRUISE' and float(self.t) >= P['t_e'] + P['T'] - 1e-9:   # TPAS 子類停用（time_switch＝False）
            P['switched'] = True
            self.blending = True                                          # 切世界座標：step_phase 於 t ≥ t_pre 進 PREGRASP（世界固定 T_pre）
            self._ev('tpa_switch', x=round(float(q[0]), 4), y=round(float(q[1]), 4))
        super()._advance_inner(T_tcp, close_done, q)

    def _v_target(self, T_tcp, x):
        if self.tpa is not None and self.st.phase == 'CRUISE' and not self.blending and x < float(self.x_pre):
            v = self.tpa['v_nav_eff'] if x < self.x_slow else self.tpa['v_c_eff']
            return v / math.sqrt(1.0 + self.path.dy(x) ** 2)
        return super()._v_target(T_tcp, x)

    def report(self):
        r = super().report()
        P = self.tpa
        if P is None:
            r['tpa'] = None
        else:
            r['tpa'] = {k: (np.round(v, 5).tolist() if isinstance(v, np.ndarray) else v) for k, v in P.items() if k != 'P'}
            r['tpa']['traj'] = dict(tb=P['P']['tb'], C=np.round(P['P']['C'], 6).tolist())
        return r
