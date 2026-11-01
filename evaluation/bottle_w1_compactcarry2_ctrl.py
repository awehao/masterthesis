#!/usr/bin/env python3
"""緊湊持物修正版 v2（Codex 20261101_201934 兩項修正；版本化子類）。bottle_w1_compactcarry_ctrl（v1）原樣保留，是 bottle_w1_compact_nominal1 那趟的版本。
緊湊姿態、清出後才收近、PLACE_APPROACH 轉回原側持放置姿態、停車與放置都沿用 v1；只改下列兩項。
1 過渡分段與時長：三段各自起訖速度為零的 min-jerk——收近（側持 → 緊湊，TCP 直線）、轉回 A（xy 不動只升高到側持高度）、轉回 B（高度保持只外伸到側持 xy）。
  v1 的轉回是單一 min-jerk 分段「先升再伸」，中點參考速度由向上約 0.254 m/s 換成水平約 0.415 m/s（向量跳變約 0.487 m/s）；v2 兩段交界速度都是零。
  轉回只從緊湊穩態開始：收近未完成就進 PLACE_APPROACH 時先走完收近再轉回，不從半途起轉回（避免起點速度不為零）。
  時長由 bottle_w1_compact_timing.py 沿整段路徑 IK 延續的 max|dq/ds|、max|d²q/ds²| 對 v_arm／a_arm 求所需值（取較大者、進位到 0.05 s）：
  收近 3.80 s、轉回 A 2.00 s、轉回 B 3.80 s（皆由速度限制決定）；v1 的 2.22 s 只由端點 Δq 推定，沿路徑的峰值設定點速率約 1.70 rad/s，超過 v_arm 0.9999。
2 第一桌淨空閘（取代 v1 只核模型瓶心一點）：核對點＝整段降低路徑（s 0→1，21 點）上的完整瓶身（側面＋端蓋取樣）與夾爪包圍盒（3×3×3 格點），加上實測構型的同一組點；
  底盤位姿集合＝實測、在途預測（命令管線）、上一拍 GMPC 底盤預測時域（N 步）、沿運送路徑前方 V_AHEAD×剩餘過渡時間（保持目前側向偏移與實測偏航）。
  判準：凡低於第一桌桌面頂＋Z_PAD 的點，在每一個位姿下水平上都位於桌面投影外擴 EDGE_PAD 之外。位姿集合與路徑取樣交叉全核，不依時間對齊：
  底盤停在集合中任一位姿、手臂停在路徑任一點都已涵蓋（停止路徑），在途與預測時域內的側向偏移也已涵蓋。
  收近開始前每拍核對，全過才開始；收近中每拍重核，不過 ⇒ 受控停止 compact_table_clear_lost。任何非有限輸入：開始前不放行，收近中受控停止 compact_clear_invalid。
  閘只針對第一桌（運送起點）；第二桌由轉回時先升後伸與既有持瓶淨空核對處理，不在本閘內。
不變：抓持、滑移、禁止接觸、淨空、CBF、Shield、第一拍列、資料期限、放置判準。0.219 m 等包絡數值只指瓶身＋夾爪，不是整支手臂。
"""
import math
import time

import numpy as np

import bottle_ref as BR
import bottle_s0b_ctrl as B
import bottle_s1_geom as G
import bottle_w1_compactcarry_ctrl as CC
import bottle_w1_continuous_approach_ctrl as CA

TCP_C = CC.TCP_C
T_TO, T_UP, T_OUT = 3.80, 2.00, 3.80
_TOP = G.table_boxes()[0]
TOP_Z = float(G.S1_TABLE['top_z'])
TABLE1_C, TABLE1_HALF = _TOP[1][:2].copy(), _TOP[2][:2] / 2.0
EDGE_PAD, Z_PAD = 0.05, 0.02
V_AHEAD = 0.30                                          # 前方距離用的速度上界（v_carry 0.25 m/s 之上）
NS = 21
GRIP = ('link6', 'link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')


def rect_out_dist(X, Y, c=None, half=None):
    """點到第一桌桌面投影（矩形）的水平有號距離：外面為正，裡面為負。c／half 未給＝呼叫當下的模組常數。"""
    c = TABLE1_C if c is None else c
    half = TABLE1_HALF if half is None else half
    dx = np.abs(X - c[0]) - half[0]
    dy = np.abs(Y - c[1]) - half[1]
    out = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    return np.where((dx <= 0) & (dy <= 0), np.maximum(dx, dy), out)


def grip_grid_tcp(qa):
    """夾爪連桿點雲在 TCP 座標的包圍盒，取 3×3×3 格點（剛性，與手臂構型無關）。"""
    q0 = np.r_[0.0, 0.0, 0.0, np.asarray(qa, float)]
    Ti = np.linalg.inv(G.C.K.fk(q0, 'link_tcp'))
    W = G.link_world(q0, G.Q_OPEN)
    P = np.vstack([v for k, v in W.items() if k in GRIP])
    Pt = (Ti[:3, :3] @ P.T).T + Ti[:3, 3]
    lo, hi = Pt.min(axis=0), Pt.max(axis=0)
    ax = [np.linspace(lo[i], hi[i], 3) for i in range(3)]
    return np.array([(x, y, z) for x in ax[0] for y in ax[1] for z in ax[2]])


def clear_margin(P_base, poses, P_world=None, t1=None):
    """低於桌面頂＋Z_PAD 的點在所有位姿下到桌面投影的最小水平距離減 EDGE_PAD；沒有低點 ⇒ +inf。非有限 ⇒ None。
    t1＝(桌面頂, 投影中心, 半尺寸)；未給＝呼叫當下的模組常數 TOP_Z／TABLE1_C／TABLE1_HALF（舊版本行為）。"""
    if not (np.isfinite(P_base).all() and np.isfinite(poses).all() and (P_world is None or np.isfinite(P_world).all())):
        return None
    top, c1, h1 = (TOP_Z, TABLE1_C, TABLE1_HALF) if t1 is None else t1
    m = math.inf
    low = P_base[P_base[:, 2] < top + Z_PAD]
    if len(low):
        c, s = np.cos(poses[:, 2:3]), np.sin(poses[:, 2:3])
        X = poses[:, 0:1] + c * low[:, 0] - s * low[:, 1]
        Y = poses[:, 1:2] + s * low[:, 0] + c * low[:, 1]
        m = min(m, float(rect_out_dist(X, Y, c1, h1).min()) - EDGE_PAD)
    if P_world is not None:
        lw = P_world[P_world[:, 2] < top + Z_PAD]
        if len(lw):
            m = min(m, float(rect_out_dist(lw[:, 0], lw[:, 1], c1, h1).min()) - EDGE_PAD)
    return m


class W1PlaceController(CA.W1PlaceController):
    group = 'G_SHROW_CONT_COMPACT2'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.cc, self.cc_ev, self.cc_log = None, [], []
        self.gate_log, self.gate_ms, self.gate_trig, self.gate_mon_min = [], [], None, math.inf
        self._Pt = None

    def _ccd(self, name):
        """過渡時長：實例參數 cc_t_to／cc_t_up／cc_t_out（不可變計畫）優先；未設＝呼叫當下的本模組常數（舊版本與暫換常數的子類行為不變）。"""
        v = getattr(self, 'cc_' + name.lower(), None)
        return float(v) if v is not None else globals()[name]

    def _t1(self):
        """第一桌（收近閘／外伸閘用）：實例參數 cc_t1＝(桌面頂, 投影中心, 半尺寸) 優先；未設＝呼叫當下的模組常數。"""
        v = getattr(self, 'cc_t1', None)
        return v if v is not None else (TOP_Z, TABLE1_C, TABLE1_HALF)

    # ---- 幾何
    def _pts_tcp(self):
        if self._Pt is None:
            BP = G.bottle_points(n_ang=12, n_z=5)
            Tr = np.asarray(self.judge.T_rel, float)
            Pb = (Tr[:3, :3] @ BP.T).T + Tr[:3, 3]
            self._Pt = np.vstack([Pb, grip_grid_tcp(self.q_arm_ready)])
        return self._Pt

    def _sweep_base(self, T_side):
        Pt = self._pts_tcp()
        R, pa, pc = T_side[:3, :3], T_side[:3, 3], TCP_C
        Pr = (R @ Pt.T).T
        return np.vstack([Pr + (pa + (pc - pa) * s) for s in np.linspace(0.0, 1.0, NS)])

    def _poses(self, q, t_rem):
        q = np.asarray(q, float)
        P = [q[:3]]
        qp, _ = self._inflight(q)
        P.append(np.asarray(qp[:3], float))
        if self.base_plan is not None:
            x, y, th = (float(v) for v in qp[:3])
            for vx, vy, w in np.asarray(self.base_plan, float):
                x += (math.cos(th) * vx - math.sin(th) * vy) * B.DT
                y += (math.sin(th) * vx + math.cos(th) * vy) * B.DT
                th += w * B.DT
                P.append(np.array([x, y, th]))
        if self.cpath is not None and self.csig is not None:
            off = q[:2] - self.cpath.xy(self.csig)[0]
            L = V_AHEAD * max(float(t_rem), 0.0)
            sig = self.csig + np.linspace(0.0, L, max(2, int(L / 0.02) + 1))
            pa = self.cpath.xy(sig) + off
            P.extend(np.column_stack([pa, np.full(len(pa), q[2])]))
        return np.array(P, float)

    def _gate(self, q, t_rem):
        t0 = time.perf_counter()
        try:
            if self.judge.T_rel is None or not np.isfinite(np.asarray(q, float)).all() or not np.isfinite(np.asarray(self.judge.T_rel, float)).all():
                return None, 0
            T_side = np.asarray(self.st.T_carry_rel, float)
            Tw = self.K.fk(np.asarray(q, float), 'link_tcp')
            Pt = self._pts_tcp()
            Pw = (Tw[:3, :3] @ Pt.T).T + Tw[:3, 3]
            poses = self._poses(q, t_rem)
            return clear_margin(self._sweep_base(T_side), poses, Pw, self._t1()), len(poses)
        finally:
            self.gate_ms.append((time.perf_counter() - t0) * 1e3)

    # ---- 參考
    def _cc_T(self, t):
        f = self.cc
        pa, pc = f['T_side'][:3, 3], TCP_C
        T = f['T_side'].copy()
        if f['t_ret'] is None:
            s = BR.min_jerk(min(max((t - f['t0']) / self._ccd('T_TO'), 0.0), 1.0))
            T[:3, 3] = pa + (pc - pa) * s
            return T
        tau = t - f['t_ret']
        if tau <= self._ccd('T_UP'):
            s = BR.min_jerk(min(max(tau / self._ccd('T_UP'), 0.0), 1.0))
            T[:3, 3] = [pc[0], pc[1], pc[2] + (pa[2] - pc[2]) * s]
        else:
            u = BR.min_jerk(min((tau - self._ccd('T_UP')) / self._ccd('T_OUT'), 1.0))
            T[:3, 3] = [pc[0] + (pa[0] - pc[0]) * u, pc[1] + (pa[1] - pc[1]) * u, pa[2]]
        return T

    def _advance_inner(self, T_tcp, close_done, q):
        super()._advance_inner(T_tcp, close_done, q)
        if self.judge.stop is not None:
            return
        t = float(self.t)
        if self.cc is None:
            if self.pph == 'CARRY_NAV' and self.judge.t_grasp is not None and self.judge.t_lift_ok is not None:
                m, n = self._gate(q, self._ccd('T_TO'))
                if self.n_step % 5 == 0 or (m is not None and m >= 0.0):
                    self.gate_log.append((round(t, 3), 'pre', None if m is None else round(m, 4), n))
                if m is not None and m >= 0.0:
                    self.cc = dict(stage='to_compact', t0=t, t_ret=None, T_side=np.array(self.st.T_carry_rel, float))
                    self.gate_trig = dict(t=t, margin=m, n_poses=n)
                    self._ev('compact_start', gate_margin=round(m, 4), n_poses=n)
                    self.cc_ev.append(('to_compact', t))
            return
        f = self.cc
        if f['stage'] == 'to_compact':
            t_rem = f['t0'] + self._ccd('T_TO') - t
            m, n = self._gate(q, t_rem)
            if self.n_step % 5 == 0 or m is None or m < 0.0:
                self.gate_log.append((round(t, 3), 'mon', None if m is None else round(m, 4), n))
            if m is None:
                self._pstop('compact_clear_invalid')
                return
            self.gate_mon_min = min(self.gate_mon_min, m)
            if m < 0.0:
                self._pstop('compact_table_clear_lost', margin=round(m, 4))
                return
            if t_rem <= 1e-9:
                f['stage'] = 'compact'
                self._ev('compact_reached')
                self.cc_ev.append(('compact', t))
        if f['stage'] == 'compact' and self.pph == 'PLACE_APPROACH':
            f.update(stage='return', t_ret=t)
            self._ev('compact_return')
            self.cc_ev.append(('return', t))
        elif f['stage'] == 'return':
            tau = t - f['t_ret']
            if 'raised' not in f and tau >= self._ccd('T_UP') - 1e-9:
                f['raised'] = t
                self._ev('compact_raised')
                self.cc_ev.append(('raised', t))
            if tau >= self._ccd('T_UP') + self._ccd('T_OUT') - 1e-9:
                f['stage'] = 'side'
                self._ev('compact_returned')
                self.cc_ev.append(('side', t))
        if f['stage'] in ('to_compact', 'compact', 'return') and self.pph == 'PLACE_APPROACH':
            self.n_still_p = 0                                   # 收近未完成或轉回放置姿態完成前不累計停車靜止
        if self.n_step % 5 == 0:
            self.cc_log.append((round(t, 3), f['stage']))

    def _T_seq_place(self, qp, t_eff):
        T, rel = super()._T_seq_place(qp, t_eff)
        if self.cc is None or self.pph not in ('CARRY_NAV', 'PLACE_APPROACH', 'PARKED'):
            return T, rel
        return np.stack([self._cc_T(t_eff + k * B.DT) for k in range(B.N + 1)]), rel

    def report(self):
        r = super().report()
        gm = np.array(self.gate_ms) if self.gate_ms else np.zeros(1)
        r['compact_carry'] = dict(
            version='v2', events=self.cc_ev, tcp_compact=TCP_C.tolist(), T_to=T_TO, T_up=T_UP, T_out=T_OUT, log=self.cc_log,
            gate=dict(edge_pad=EDGE_PAD, z_pad=Z_PAD, top_z=TOP_Z, v_ahead=V_AHEAD, n_sweep=NS, trigger=self.gate_trig,
                      monitor_min=None if not math.isfinite(self.gate_mon_min) else self.gate_mon_min, ms_max=float(gm.max()), ms_mean=float(gm.mean()), n=len(self.gate_ms), log=self.gate_log),
            scope='v2：三段各自起訖速度為零的過渡（時長依整段路徑關節速度／加速度）；第一桌淨空閘＝整段降低路徑的完整瓶身＋夾爪點 × 實測／在途／預測時域／路徑前方位姿；停車與放置沿用原算法')
        return r
