#!/usr/bin/env python3
"""瓶子第二桌放置控制器（設計：第七次進度報告素材/規格/瓶子第二桌放置_最小設計.md draft-2）。
PlaceController(S1Planned)：抓取到 LIFT 完全沿用 S1Planned（N2）；LIFT 開始時以實測狀態解停車位、請求持瓶段路徑；CARRY 開始時核對後採用。
持瓶以後（self.pph 不為 None）由本類別的 _solve_place 求解：進度全用弧長 σ，參考推進與領先限制都在這裡，不經 S0b 的 x 夾限與 done 的 x 錨定。
放置相位 self.pph：CARRY_NAV → PLACE_APPROACH → PARKED → LOWER → RELEASE → RETREAT → HOLD → DONE；st.phase 停在 'CARRY'（BR 相位機不改）。
同一個全身 GMPC：停車後改用底盤命令上下限為 0 的同設定 GMPC（同 S1Parked），不是另一個控制器。
數值都是事前凍結的候選判準（未驗證）；第一版瓶位姿、底盤位姿、第二桌位置用真值。
"""
import copy
import dataclasses
import math
import time

import numpy as np

from ammr_wholebody_mpc import gmpc_wb as GW
from ammr_wholebody_mpc import wgmpc_core_sp as SP
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE
from ammr_wholebody_mpc.se2 import exp_, from_xytheta

import bottle_n2_ctrl as N2C
import bottle_n2_path as NP
import bottle_ref as BR
import bottle_s0_ctrl as C0
import bottle_s0_logic as L
import bottle_s0b_ctrl as B
import bottle_s1_ctrl as S1
import bottle_s1_geom as G

DT, N = S1.DT, S1.N
PLACE = dict(c=(2.4, -0.39), size=(0.80, 0.50), top=0.60, target=(2.4, -0.58), lane_y=-1.0, entry_back=0.6, theta=0.0)
REG = dict(park_v=0.01, park_w=0.02, park_s=0.5, park_pos=0.020, park_yaw_deg=1.0, park_bottle=0.015,
           v_fast=0.02, v_slow=0.005, slow_above=0.010, extra_down=0.010, sup_fz=0.98, sup_h_ratio=0.3, sup_gap=0.003, sup_s=0.3, tilt_deg=3.0, edge_m=0.010,
           rel_open=0.8, rel_f=0.05, rel_s=0.5, rel_deadline=2.0, ret_back=0.06, ret_t1=1.5, ret_t2=3.4, ret_clear=0.010,
           hold_s=3.0, st_disp=0.005, st_tilt=3.0, st_v=0.001, st_fz_ratio=0.9, place_tol_report=0.030,
           join_lat=0.05, conn_len=0.8, conn_turn_deg=3.0, a_lat=0.5, a_axis=(1.5, 1.0), kappa_win=0.3, smooth_iters=100, ref_jump=0.05, v_ref_jump=0.075, a_dec=0.15, t_mult=3.0, stale_s=0.1, v_win=0.1)
PPH = ('CARRY_NAV', 'PLACE_APPROACH', 'PARKED', 'LOWER', 'RELEASE', 'RETREAT', 'HOLD', 'DONE')


class PlaceJudge(L.S0Judge):
    """鬆爪准入後：手指脫離接觸改成預期事件（失抓、滑移不再判）；禁止接觸、穿透、資料有效性照舊。"""

    def release(self, t):
        self.released_t = float(t)
        self.r = dict(self.r, loss_s=float('inf'))
        self.T_rel_kept, self.T_rel = self.T_rel, None


class CarryPath(NP.PlannedPath):
    """持瓶段：規劃段弧長 σ，入口 x_lane 之後是 y＝lane_y、朝 +x 的指定直線（θ_park＝0）。"""

    def __init__(self, P, x_lane, lane_y):
        self.lane_y = float(lane_y)
        super().__init__(P, x_lane)

    def xy(self, sig):
        sig = np.atleast_1d(np.asarray(sig, float))
        out = super().xy(sig)
        lane = sig >= self.x_lane
        out[lane, 1] = self.lane_y
        return out


def bezier(p0, t0, p3, t3, ds=NP.DS):
    d = float(np.linalg.norm(p3 - p0)) / 3.0
    c1, c2 = p0 + d * t0, p3 - d * t3
    u = np.linspace(0.0, 1.0, max(int(3 * d / ds), 8))[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * p3


def max_turn_deg(P):
    T = np.diff(NP.resample(P), axis=0)
    a = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
    return float(np.degrees(np.abs(np.diff(a))).max()) if len(a) > 1 else 0.0


def smooth_path(P, iters, lam=0.5):
    """固定兩端的拉普拉斯平滑（0.02 m 取樣，100 次約等於 0.2 m 的平滑半徑）。會切角，呼叫端須重新核對代價與幾何。"""
    Q = NP.resample(P).copy()
    for _ in range(int(iters)):
        Q[1:-1] += lam * (Q[:-2] + Q[2:] - 2.0 * Q[1:-1])
    return NP.resample(Q)


def build_carry(raw, p_pred, dir_pred, x_entry, lane_y, reg=REG, v_max=0.25):
    """規劃原始路徑 → 平滑 → 承接接段＋規劃段＋接直線段。回傳 (CarryPath, 資訊) 或 (None, 原因)。
    轉角每 20 mm 只報告（draft-3：Codex 20261101_040958 改用物理量驗收，見 ref_profile_check）。"""
    raw0 = NP.resample(raw)
    raw = smooth_path(raw0, reg['smooth_iters'])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
    d = np.linalg.norm(raw - p_pred, axis=1)
    k = int(np.argmin(d[s <= 1.0])) if (s <= 1.0).any() else int(np.argmin(d))
    lat = float(d[k])
    if lat > reg['join_lat']:
        return None, dict(why='carry_join_lateral', lateral=lat)
    k2 = int(np.searchsorted(s, s[k] + reg['conn_len']))
    if k2 >= len(raw) - 2:
        return None, dict(why='carry_path_too_short', L=float(s[-1]))
    t3 = raw[k2 + 1] - raw[k2]
    t3 = t3 / max(np.linalg.norm(t3), 1e-9)
    conn = bezier(np.asarray(p_pred, float), np.asarray(dir_pred, float), raw[k2], t3)
    turn = max_turn_deg(conn)
    P = np.vstack([conn, raw[k2 + 1:]])
    J = join_lane_y(P, x_entry, lane_y, reg['conn_len'])
    return CarryPath(J, x_entry, lane_y), dict(lateral=lat, conn_turn_deg=turn, n=len(J), turn_max_deg=max_turn_deg(J), raw_turn_max_deg=max_turn_deg(raw0),
                                               smooth_shift_max=float(np.linalg.norm(raw - NP.resample(raw0)[np.linspace(0, len(raw0) - 1, len(raw)).round().astype(int)], axis=1).max()) if len(raw0) else None)


def curvature_grid(cp, win):
    """規劃段（σ0 到直線入口）的離散曲率，取 ±win/2 內最大值；直線段 0。"""
    sig = np.arange(cp.sig0, cp.x_lane + 1e-9, NP.DS)
    T = np.diff(cp.xy(sig), axis=0)
    h = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
    k = np.r_[0.0, np.abs(np.diff(h)) / NP.DS, 0.0]
    w = max(int(round(win / 2 / NP.DS)), 1)
    km = np.array([k[max(i - w, 0):i + w + 1].max() for i in range(len(k))])
    return sig, km


def speed_target(s, v_b, rem, vcap_fn, v_carry, a_dec, slew):
    v = min(v_carry, math.sqrt(2.0 * a_dec * max(rem, 0.0)), vcap_fn(s))
    v = float(np.clip(v, v_b - slew, v_b + slew)) if rem > 0.02 else min(v, v_b)
    return 0.0 if rem <= 1e-4 or v < 1e-4 else v


def ref_profile_check(cp, s0, v0, s_park, vcap_fn, v_carry, reg):
    """用控制器同一套速度更新（每 2 週期改速）模擬參考到停車，逐控制週期算參考速度向量的變化：逐軸加速度、橫向加速度。"""
    s, v, k, W, alat = s0, v0, 0, [], 0.0
    sig_k, km = curvature_grid(cp, reg['kappa_win'])
    slew = S1.A_SLEW * 2.0 * DT
    while k < 4000:
        if k % 2 == 0:
            v_new = speed_target(s, v, s_park - s, vcap_fn, v_carry, reg['a_dec'], slew)
            s += (v - v_new) * C0.D_STEPS * C0.PHYS
            v = v_new
        s += v * DT
        se = s + v * C0.D_STEPS * C0.PHYS
        W.append(v * cp.tangent(se)[0])
        alat = max(alat, v * v * float(np.interp(se, sig_k, km, right=0.0)))
        k += 1
        if v == 0.0 and s >= s_park - 0.02:
            break
    A = np.abs(np.diff(np.array(W), axis=0)) / DT
    ax, ay = (float(A[:, 0].max()), float(A[:, 1].max())) if len(A) else (0.0, 0.0)
    ok = ax <= reg['a_axis'][0] + 1e-9 and ay <= reg['a_axis'][1] + 1e-9 and alat <= reg['a_lat'] + 1e-9
    return ok, dict(a_x_max=ax, a_y_max=ay, a_lat_max=alat, n=len(W), v_cap_min=float(min(vcap_fn(x_) for x_ in sig_k)))


def join_lane_y(P, x_lane, lane_y, L_=NP.JOIN_LEN):
    """NP.join_lane 的 y 參數版：末段 L_ 換成三次 Bézier，終點 (x_lane, lane_y)、切線 +x。"""
    P = NP.resample(P)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    i = min(max(int(np.searchsorted(s, max(s[-1] - L_, 0.0))), 1), len(P) - 1)
    t0 = P[i] - P[i - 1]
    t0 = t0 / max(np.linalg.norm(t0), 1e-9)
    B_ = bezier(P[i - 1], t0, np.array([x_lane, lane_y]), np.array([1.0, 0.0]))
    return NP.resample(np.vstack([P[:i - 1], B_]))


def bottle_bottom_z(T_b, r):
    s = math.sqrt(max(1.0 - float(T_b[2, 2]) ** 2, 0.0))
    return float(T_b[2, 3]) - r * s


def tilt_deg(T_b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(T_b[2, 2])))))


class PlaceController(N2C.S1Planned):
    def __init__(self, *a, place=None, carry_planner=None, reg=None, **kw):
        super().__init__(*a, **kw)
        self.place, self.preg = dict(PLACE, **(place or {})), dict(REG, **(reg or {}))
        self.carry_planner = carry_planner                  # request(start_xyyaw, goal_xy, t)；poll()→None｜dict(ok, P, frame, grid, why)
        self.judge.__class__ = PlaceJudge                   # 同一個判定物件，只加鬆爪後的語義切換
        self.mpc_nav = self.mpc
        self.mpc_park = GW.GMPC(dataclasses.replace(self.nav, u_min=np.zeros(3), u_max=np.zeros(3)))
        self.pph, self.t_pph, self.cpath, self.csig, self.cmon = None, None, None, None, None
        self.park, self.h_lift, self.carry_req, self.carry_info = None, None, None, None
        self.d_cmd, self.v_desc, self.d_freeze, self.t_sup0, self.n_sup = 0.0, 0.0, None, None, 0
        self.t_release, self.n_rel, self.T_lower = None, 0, None
        self.T_b_rel0, self.n_still_p, self.place_ev, self.st_log = None, 0, {}, []
        self.f_place, self.q_fingers, self.T_ret0, self.q_ret0 = None, None, None, None
        self.vcap = lambda s_: float('inf')
        self.bottle_mass = 0.20                             # 資產 bottle_s0.yaml；跑法讀回後覆寫
        self.t_ok = {}

    # ---- 共用
    def _set(self, ph, **info):
        self.pph, self.t_pph = ph, self.t
        self.place_ev[ph] = dict(t=round(self.t, 3), **info)
        self._ev('place_phase', to=ph, **{k: v for k, v in info.items() if k not in ('t', 'what')})

    def _pstop(self, why, **kw):
        if self.judge.stop is None:
            self.judge._stop(self.t, why, **kw)

    def _inflight(self, q):
        qp, sp = np.asarray(q, float).copy(), self.s.copy()
        for u_in in self.pipe:
            qp, sp = SP.plant_phys_step(qp, sp, u_in, self.cfg, 1)
        return qp, sp

    def _park_solve(self, q):
        """θ_park 固定，用完整座標變換：瓶底心 xy＝放置點 ⇒ 底盤 (x, y)。回傳 (P_park, 抬升高度)。"""
        Tbt = self.K.fk(np.r_[0.0, 0.0, 0.0, q[3:]], 'link_tcp') @ self.judge.T_rel
        th = self.place['theta']
        c_, s_ = math.cos(th), math.sin(th)
        p = Tbt[:2, 3]
        xy = np.array(self.place['target']) - np.array([c_ * p[0] - s_ * p[1], s_ * p[0] + c_ * p[1]])
        return np.r_[xy, th], float(Tbt[2, 3]) - self.place['top']

    # ---- 相位
    def _advance_inner(self, T_tcp, close_done, q):
        super()._advance_inner(T_tcp, close_done, q)
        if self.judge.stop is not None:
            return
        st, R = self.st, self.preg
        if st.phase == 'LIFT' and self.carry_req is None and self.judge.T_rel is not None:
            self.park, _ = self._park_solve(q)                          # LIFT 開始：瓶還在第一桌，只取停車 xy 當規劃目標
            goal = (self.park[0] - self.place['entry_back'], float(self.park[1]))
            qv = getattr(self, 'q_prev2', None)                             # 起點只是規劃猜測：實測位姿＋世界座標實測速度向量×LIFT 名目時長
            vw = np.zeros(2) if qv is None else (np.asarray(q[:2], float) - np.asarray(qv[:2], float)) / C0.PHYS
            start = np.r_[np.asarray(q[:2], float) + vw * float(self.task.t_lift), q[2]]
            self.carry_req = dict(t_req=self.t, start=start.tolist(), q_meas=q[:3].tolist(), v_world=vw.tolist(), goal=goal, park=self.park.tolist())
            self.carry_planner.request(tuple(start), goal, self.t)
            self._ev('carry_plan_request', **self.carry_req)
        if st.phase == 'CARRY' and self.pph is None:
            res = self.carry_planner.poll()
            if res is None or not res.get('ok'):
                self._pstop('carry_plan_missing' if res is None else 'carry_plan_invalid', detail=None if res is None else res.get('why'))
                return
            self.carry_info = dict(plan=res.get('info'))                # 被拒也留下原始路徑與規劃資訊
            qp, _ = self._inflight(q)
            v = np.asarray(self.base_plan[0][:2], float) if self.base_plan is not None else np.array([1.0, 0.0])
            c_, s_ = math.cos(qp[2]), math.sin(qp[2])
            dw = np.array([c_ * v[0] - s_ * v[1], s_ * v[0] + c_ * v[1]])
            dw = dw / np.linalg.norm(dw) if np.linalg.norm(dw) > 1e-3 else np.array([math.cos(qp[2]), math.sin(qp[2])])
            ok, why = NP.validate_plan(res['frame'], res['P'], np.array(self.carry_req['start'][:2]), np.array(self.carry_req['goal']), res['grid']) if res.get('grid') is not None else (res['frame'] == 'map', 'frame')
            if not ok:
                self._pstop('carry_plan_rejected', detail=why)
                return
            self.park, _ = self._park_solve(q)                          # CARRY 開始：抬升後的實際構型重算停車位，直線段 y＝停車 y
            cp, info = build_carry(res['P'], qp[:2], dw, self.carry_req['goal'][0], float(self.park[1]), R, v_max=self.v_carry)
            if cp is None:
                self._pstop(info.pop('why'), **info)
                return
            full = np.vstack([cp.P, cp.xy(np.arange(cp.x_lane, float(self.park[0]) + 1e-9, NP.DS))])   # 規劃段＋接段＋直線段到停車位
            if res.get('grid') is not None and NP.blocked(res['grid'].cost(full)).any():
                self._pstop('carry_plan_rejected', detail='承接後（含直線段到停車位）有內切以上或未知格')
                return
            csig = cp.project(q[:2], cp.sig0, back=0.0, ahead=0.3)       # 實測位置的投影（領先量同 S0b 以實測為準）
            lead = min(max(self.s_ref - float(q[0]), 0.0), B.S_LEAD)         # 承接前參考對實測的領先量，沿用
            ref_old = np.array([self.s_ref + self.v_b * C0.D_STEPS * C0.PHYS, self.y_ref])   # 承接前的參考（生效時刻）
            s_new = csig + lead
            jump = float(np.linalg.norm(cp.xy(s_new + self.v_b * C0.D_STEPS * C0.PHYS)[0] - ref_old))
            if jump > R['ref_jump']:
                self._pstop('carry_ref_discontinuous', jump=jump)
                return
            v_old = self.v_b * np.array([math.cos(self.th_ref), math.sin(self.th_ref)])
            v_new = self.v_b * cp.tangent(s_new + self.v_b * C0.D_STEPS * C0.PHYS)[0]
            dv = float(np.linalg.norm(v_new - v_old))
            if dv > R['v_ref_jump']:
                self._pstop('carry_ref_discontinuous', dv=dv)
                return
            sig_k, km = curvature_grid(cp, R['kappa_win'])
            vcap = lambda s_, sk=sig_k, kk=km: min(self.v_carry, math.sqrt(R['a_lat'] / max(float(np.interp(s_, sk, kk, right=0.0)), 1e-9)))   # noqa: E731  局部降速：曲率處橫向加速度 ≤ a_lat
            ok, prof = ref_profile_check(cp, s_new, self.v_b, float(self.park[0]), vcap, self.v_carry, R)
            if not ok:
                self._pstop('carry_ref_accel', **prof)
                return
            info.update(ref_jump=jump, v_ref_jump=dv, lead=lead, v_b=self.v_b, v_meas=float(self._vb_meas(q)), profile=prof)
            self.cpath, self.carry_info, self.csig, self.vcap = cp, dict(info, plan=res.get('info')), csig, vcap
            self.s_ref, self.done_anchored = s_new, True                     # 參考速度 v_b 沿用承接前的值：位置與速度都連續
            self.th_ref = self.place['theta']
            if res.get('monitor') is not None:
                self.cmon = res['monitor'](cp)
            self._set('CARRY_NAV', sig0=round(self.csig, 4), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()})
            return
        if self.pph is None:
            return
        sig_park = float(self.park[0])
        tp = self.t - self.t_pph
        if self.cmon is not None and self.pph in ('CARRY_NAV', 'PLACE_APPROACH') and self.cmon.why is not None:
            w = dict(self.cmon.why)
            self._pstop(w.pop('why'), **w)
            return
        if self.pph == 'CARRY_NAV' and self.s_ref >= self.cpath.x_lane:
            self._set('PLACE_APPROACH')
        elif self.pph == 'PLACE_APPROACH':
            if self.v_b == 0.0 and self.s_ref >= sig_park - 1e-9:
                vb, w = self._vb_meas(q), abs(float(q[2] - self.q_prev2[2])) / C0.PHYS if getattr(self, 'q_prev2', None) is not None else float('inf')
                ok = vb < R['park_v'] and w < R['park_w']
                self.n_still_p = self.n_still_p + C0.NPH if ok else 0
                if self.n_still_p * C0.PHYS >= R['park_s'] - 1e-9:
                    e_pos = float(np.hypot(q[0] - self.park[0], q[1] - self.park[1]))
                    e_yaw = abs(math.degrees(math.atan2(math.sin(q[2] - self.park[2]), math.cos(q[2] - self.park[2]))))
                    Tb = self.T_b_now
                    e_b = float(np.hypot(Tb[0, 3] - self.place['target'][0], Tb[1, 3] - self.place['target'][1])) if Tb is not None else float('inf')
                    if e_pos > R['park_pos'] or e_yaw > R['park_yaw_deg'] or e_b > R['park_bottle']:
                        self._pstop('park_misaligned', e_pos=e_pos, e_yaw=e_yaw, e_bottle=e_b)
                        return
                    self.T_lower = T_tcp.copy()                             # 下降起點：當下實測 TCP 世界位姿
                    self.h_lift = bottle_bottom_z(Tb, self.judge.br) - self.place['top']   # 實測瓶底離第二桌面
                    self._set('PARKED', e_pos=round(e_pos, 4), e_yaw=round(e_yaw, 3), e_bottle=round(e_b, 4))
                    self._set('LOWER', h_lift=round(self.h_lift, 4))
        elif self.pph == 'LOWER':
            qp, _ = self._inflight(q)
            if self.d_freeze is None:
                Tb_pred = self.K.fk(qp, 'link_tcp') @ self.judge.T_rel
                h_pred = bottle_bottom_z(Tb_pred, self.judge.br) - self.place['top']
                self.v_desc = R['v_fast'] if h_pred > R['slow_above'] else R['v_slow']
                self.d_cmd = min(self.d_cmd + self.v_desc * DT, self.h_lift + R['extra_down'])
                if self.d_cmd >= self.h_lift + R['extra_down'] - 1e-9 and self.t_sup0 is None:
                    self._pstop('no_support', d_cmd=self.d_cmd, h_lift=self.h_lift)
            if self.t_sup0 is not None and (self.t - self.t_sup0) >= R['sup_s'] - 1e-9 and self.n_sup * C0.PHYS >= R['sup_s'] - 1e-9:
                self.T_b_rel0 = None if self.T_b_now is None else np.array(self.T_b_now)
                self.judge.release(self.t)
                self.t_release = self.t
                self._set('RELEASE', d_cmd=round(self.d_cmd, 4))
        elif self.pph == 'RELEASE':
            if self.n_rel * C0.PHYS >= R['rel_s'] - 1e-9:
                self.T_ret0, self.q_ret0 = T_tcp.copy(), np.asarray(q[3:], float).copy()
                self._set('RETREAT')
            elif tp > R['rel_deadline'] + 0.5:
                self._pstop('release_not_confirmed', n_rel=self.n_rel)
        elif self.pph == 'RETREAT':
            if tp >= R['ret_t1'] and self.q_ret1 is None:
                self.q_ret1 = np.asarray(q[3:], float).copy()
            if tp >= R['ret_t1'] + R['ret_t2']:
                d = self._arm_bottle_clear(q)
                if d < R['ret_clear']:
                    self._pstop('retreat_clearance', d=d)
                    return
                self._set('HOLD', clear=round(d, 4))
        elif self.pph == 'HOLD' and tp >= R['hold_s']:
            ok, met = self._place_success()
            self.place_metrics = met
            if ok:
                self.mission_done = True
                self._set('DONE')
                self._ev('mission_done', x=float(q[0]), y=float(q[1]))
            else:
                self._pstop('place_not_stable', **{k: v for k, v in met.items() if k != 'ok'})
        tp = self.t - self.t_pph                                         # 相位可能剛切換：重算
        if self.pph in self._limits() and tp > self._limits()[self.pph]:
            self._pstop('place_phase_timeout', phase=self.pph, t_phase=tp)

    q_ret1 = None

    def _limits(self):
        R, m = self.preg, self.preg['t_mult']
        h = self.h_lift or 0.05
        t_low = max(h - R['slow_above'], 0.0) / R['v_fast'] + (R['slow_above'] + R['extra_down']) / R['v_slow']
        return dict(PLACE_APPROACH=m * max(self.place['entry_back'] / 0.1, 6.0), LOWER=m * t_low, RELEASE=R['rel_deadline'] + 1.0, RETREAT=m * (R['ret_t1'] + R['ret_t2']), HOLD=R['hold_s'] + 2.0)

    def _arm_bottle_clear(self, q):
        Tb = self.T_b_now
        if Tb is None:
            return float('nan')
        P = np.vstack(list(G.link_world(np.asarray(q, float), self.f_cmd).values()))
        Pb = (Tb[:3, :3].T @ (P - Tb[:3, 3]).T).T
        return float(G.cyl_dist(Pb, 0.0, 0.0, 0.0, G.BOTTLE['h'], self.judge.br).min())

    def _place_success(self):
        R, L_ = self.preg, self.st_log
        if not L_:
            return False, dict(ok=False, why='no_log')
        P = np.array([e[1] for e in L_])
        tt = np.array([e[0] for e in L_])
        valid = bool(np.isfinite(P).all() and all(math.isfinite(e[2]) for e in L_))
        disp = float(np.linalg.norm(P[:, :3] - P[0, :3], axis=1).max()) if valid else float('nan')
        tilt = float(max(e[2] for e in L_)) if valid else float('nan')
        tail = (tt >= tt[-1] - 1.0)
        k_ = max(int(round(R['v_win'] / C0.PHYS)), 1)                        # 末 1 s 逐 0.1 s 段速度的最大值（不讓往返互相抵消）
        Pt = P[tail]
        v_end = float((np.linalg.norm(Pt[k_:] - Pt[:-k_], axis=1) / (k_ * C0.PHYS)).max()) if valid and len(Pt) > k_ else float('nan')
        hold = [e for e in L_ if self.place_ev.get('HOLD') and e[0] >= self.place_ev['HOLD']['t']]
        fz_h = np.array([e[3] for e in hold], float)
        fz_min = float(fz_h.min()) if len(fz_h) and np.isfinite(fz_h).all() else float('nan')   # 全窗須有效：有 NaN 就不成立
        mg = float(self.bottle_mass) * 9.81
        x, y = P[-1, 0], P[-1, 1]
        cx, cy = self.place['c']
        sx, sy = self.place['size']
        edge = min(x - (cx - sx / 2), (cx + sx / 2) - x, y - (cy - sy / 2), (cy + sy / 2) - y)
        off = float(np.hypot(x - self.place['target'][0], y - self.place['target'][1]))
        met = dict(disp=disp, tilt=tilt, v_end=v_end, fz_min=fz_min, fz_need=R['st_fz_ratio'] * mg, edge=edge, edge_need=self.judge.br + R['edge_m'], offset=off)
        ok = valid and disp <= R['st_disp'] and tilt <= R['st_tilt'] and v_end < R['st_v'] and fz_min >= R['st_fz_ratio'] * mg and edge >= self.judge.br + R['edge_m']
        met['valid'] = valid
        met['ok'] = bool(ok)
        return bool(ok), met

    def _inside(self, T_b):
        cx, cy = self.place['c']
        sx, sy = self.place['size']
        return min(T_b[0, 3] - (cx - sx / 2), (cx + sx / 2) - T_b[0, 3], T_b[1, 3] - (cy - sy / 2), (cy + sy / 2) - T_b[1, 3]) >= self.judge.br + self.preg['edge_m']

    # ---- 每物理步：支撐、鬆爪、穩定量測
    def step(self, t, q, f_place=None, q_fingers=None, **kw):
        self.f_place, self.q_fingers = f_place, q_fingers
        R = self.preg
        T_b = kw.get('T_bottle')
        ok_b = T_b is not None and np.isfinite(T_b).all()
        ok_f = f_place is not None and np.isfinite(np.asarray(f_place, float)).all()
        ok_q = q_fingers is not None and np.isfinite(np.asarray(q_fingers, float)).all() and kw.get('f_fingers') is not None and np.isfinite(np.asarray(kw['f_fingers'], float)).all()
        # 瓶↔第二桌面依相位准入（走既有禁止接觸判定）：LOWER 前任何接觸都禁止；LOWER 起只允許像瓶底支撐的接觸（法向向上、水平小、瓶底貼桌面、在放置區內）
        forb = dict(kw.get('forbidden') or {})
        if not ok_f:
            forb['bottle_table2'] = float('nan')
        else:
            fv = np.asarray(f_place, float)
            fn = float(np.linalg.norm(fv))
            if self.pph not in ('LOWER', 'RELEASE', 'RETREAT', 'HOLD', 'DONE'):   # 放好之後瓶仍靠第二桌支撐
                forb['bottle_table2'] = fn
            else:
                bottom = ok_b and fv[2] > 0.0 and float(np.hypot(fv[0], fv[1])) <= R['sup_h_ratio'] * fv[2] + self.judge.r['forbid_n'] \
                    and bottle_bottom_z(T_b, self.judge.br) - self.place['top'] <= R['sup_gap'] and self._inside(T_b)
                forb['bottle_table2'] = 0.0 if bottom else fn
        kw['forbidden'] = forb
        if self.pph in ('LOWER', 'RELEASE', 'RETREAT', 'HOLD') and self.judge.stop is None:   # 放置量測缺失：連續計數歸零；過期即停（同既有資料失效規則）
            for k_, ok_ in (('place_force', ok_f and ok_b), ('fingers', ok_q)):
                if ok_:
                    self.t_ok[k_] = float(t)
                elif float(t) - self.t_ok.setdefault(k_, float(t)) > R['stale_s']:
                    self.judge._stop(float(t), f'data_stale_{k_}')
            if not (ok_f and ok_b):
                self.n_sup = 0
            if not ok_q:
                self.n_rel = 0
        if self.pph == 'LOWER' and self.judge.stop is None and ok_b and ok_f:
            fz, fh = float(f_place[2]), float(np.hypot(f_place[0], f_place[1]))
            gap = bottle_bottom_z(T_b, self.judge.br) - self.place['top']
            inside = self._inside(T_b)
            good = fz >= R['sup_fz'] and fh <= R['sup_h_ratio'] * fz and gap <= R['sup_gap'] and tilt_deg(T_b) <= R['tilt_deg'] and inside
            if good and self.d_freeze is None:
                self.d_freeze, self.t_sup0 = self.d_cmd, float(t)          # 首次有效支撐：不再加深
                self._ev('support_first', fz=fz, fh=fh, gap=gap, d_cmd=self.d_cmd)
            self.n_sup = self.n_sup + 1 if good else 0
        if self.pph == 'RELEASE' and ok_q:
            opened = min(q_fingers) >= R['rel_open'] * C0.FINGER_OPEN
            free = max(kw['f_fingers']) < R['rel_f']
            self.n_rel = self.n_rel + 1 if (opened and free) else 0
        if self.t_release is not None:
            fz = float(f_place[2]) if ok_f else float('nan')
            self.st_log.append((float(t), np.r_[T_b[:3, 3]] if ok_b else np.full(3, np.nan), tilt_deg(T_b) if ok_b else float('nan'), fz))
        out = super().step(t, q, **kw)
        if self.t_release is not None:
            f = min((float(t) - self.t_release) / C0.T_CLOSE_RAMP, 1.0)
            self.f_cmd = C0.FINGER_CLOSED + (C0.FINGER_OPEN - C0.FINGER_CLOSED) * f
            out['finger_target'] = float(self.f_cmd)
        out['done'] = self.mission_done
        out['place_phase'] = self.pph
        return out

    # ---- 求解
    def _solve(self, q):
        if self.pph is None:
            return super()._solve(q)
        return self._solve_place(q)

    def _place_speed(self):
        R = self.preg
        if self.pph not in ('CARRY_NAV', 'PLACE_APPROACH'):
            return 0.0
        return speed_target(self.s_ref, self.v_b, float(self.park[0]) - self.s_ref, self.vcap, self.v_carry, R['a_dec'], S1.A_SLEW * 2.0 * DT)

    def _T_seq_place(self, qp, t_eff):
        R, ph = self.preg, self.pph
        if ph in ('CARRY_NAV', 'PLACE_APPROACH', 'PARKED'):
            return np.stack([self.st.T_carry_rel] * (N + 1)), True
        if ph in ('LOWER', 'RELEASE'):
            d0 = self.d_cmd if self.d_freeze is None else self.d_freeze
            T = []
            for k in range(N + 1):
                d = d0 if (ph == 'RELEASE' or self.d_freeze is not None) else min(d0 + self.v_desc * DT * k, self.h_lift + R['extra_down'])
                Tk = self.T_lower.copy()
                Tk[2, 3] -= d
                T.append(Tk)
            return np.stack(T), False
        if ph == 'RETREAT':
            T = []
            for k in range(N + 1):
                tau = t_eff + k * DT - self.t_pph
                if tau <= R['ret_t1'] or self.q_ret1 is None:
                    Tk = self.T_ret0.copy()
                    Tk[:3, 3] -= R['ret_back'] * BR.min_jerk(min(max(tau / R['ret_t1'], 0.0), 1.0)) * self.T_ret0[:3, 2]
                    T.append(Tk)
                else:
                    s_ = BR.min_jerk(min((tau - R['ret_t1']) / R['ret_t2'], 1.0))
                    qa = self.q_ret1 + (self.q_tuck - self.q_ret1) * s_
                    T.append(self.K.fk(np.r_[qp[:3], qa], 'link_tcp'))          # 用實際底盤位姿，不用名目停車位
            return np.stack(T), False
        return np.stack([self.K.fk(np.r_[0.0, 0.0, 0.0, self.q_tuck], 'link_tcp')] * (N + 1)), True

    def _ref_window_place(self, xs_eff):
        th = self.place['theta']
        if self.pph not in ('CARRY_NAV', 'PLACE_APPROACH'):
            X = from_xytheta(float(self.park[0]), float(self.park[1]), th)
            return np.array([X] * (N + 1)), np.zeros((N + 1, 3))
        xs = xs_eff + self.v_b * DT * np.arange(N + 1)
        P, Tn = self.cpath.xy(xs), self.cpath.tangent(xs)
        c_, s_ = math.cos(th), math.sin(th)
        vw = self.v_b * Tn
        xi = np.column_stack([c_ * vw[:, 0] + s_ * vw[:, 1], -s_ * vw[:, 0] + c_ * vw[:, 1], np.zeros(N + 1)])   # 世界速度 → 本體
        return np.array([from_xytheta(float(p[0]), float(p[1]), th) for p in P]), xi

    def _solve_place(self, q):
        """S0b._solve 的弧長版：參考推進、領先限制都用 σ；停車後換底盤命令為 0 的同設定 GMPC。"""
        t0 = time.perf_counter()
        cfg, R = self.cfg, self.preg
        if self.n_tick % 2 == 0:
            v_new = self._place_speed()
            self.s_ref += (self.v_b - v_new) * C0.D_STEPS * C0.PHYS
            self.v_b = v_new
        if self.pph in ('CARRY_NAV', 'PLACE_APPROACH'):
            self.csig = self.cpath.project(q[:2], self.csig)
            self.s_ref = min(self.s_ref, self.csig + B.S_LEAD)
            if self.v_b == 0.0 and self.s_ref >= float(self.park[0]) - 0.02:
                self.s_ref = float(self.park[0])
        self.n_tick += 1
        self.v_b_hist.append(self.v_b)
        qp, sp = self._inflight(q)
        t_eff = self.t + C0.D_STEPS * C0.PHYS
        T_seq, rel = self._T_seq_place(qp, t_eff)
        xs_eff = self.s_ref + self.v_b * C0.D_STEPS * C0.PHYS
        X_ref, xi_ref = self._ref_window_place(xs_eff)
        parked = self.pph not in ('CARRY_NAV', 'PLACE_APPROACH')
        self.mpc = self.mpc_park if parked else self.mpc_nav
        w_p = C0.W_NEAR if self.pph in ('LOWER', 'RELEASE', 'RETREAT') else C0.W_FAR
        arm = dict(mode='active', K=self.K, tcp='link_tcp', q_arm=qp[3:], s_arm=sp, u_arm_prev=self.u_prev[3:], model=cfg.arm_model, T_seq=T_seq, w_p=w_p,
                   lo=LITE6_SAFE.lower, hi=LITE6_SAFE.upper, tcp_rel_base=rel, qp_solver=self.qp_solver, **B.W_ARM)
        obs = None if self.obstacles_fn is None else self.obstacles_fn(t_eff)
        r = self.mpc.solve(obstacles=obs, X_now=from_xytheta(*qp[:3]), X_ref_win=X_ref, xi_ref_win=xi_ref, xi_prev=self.u_prev[:3], arm=arm)
        ck = dict(r.arm_check or {})
        self.solve_ms.append((time.perf_counter() - t0) * 1e3)
        self.u_arm_plan = None if r.u_arm_seq is None else np.array(r.u_arm_seq, float)
        self.base_plan = xi_ref[:N] + np.asarray(r.delta_xi_all, float)
        self.qp_log.append(dict(ck.get('qp') or dict(solver='fallback' if ck.get('fallback') else 'hold')))
        if self.compact is not None and r.u_arm_seq is not None:
            self.compact.append(dict(obs=obs, t=self.t, phase=BR.PHASES.index(self.st.phase), qp=qp.copy(), sp=sp.copy(), u_prev=self.u_prev.copy(), X_ref=X_ref.copy(), xi_ref=xi_ref.copy(), T_seq=np.array(T_seq), rel=bool(rel), w_p=float(w_p),
                                     bias=np.array(cfg.arm_model.bias, float), u=np.r_[np.asarray(r.u_opt, float), r.u_arm_seq[0]], cbf_active=int(r.cbf_active), min_h=float(r.min_h), eps0=float(r.eps0),
                                     resid_ns=float(r.cbf_resid_post_noslack), resid_s=float(r.cbf_resid_post_slack)))
        if ck.get('fallback') or ck.get('failed') or not ck.get('seq_consistent', False):
            self.n_fallback += 1
            self._ev('wb_fallback', why=ck.get('fallback'), failed=ck.get('failed'))
            self._trigger_stop(f"wb_fallback:{ck.get('fallback')}")
            return None
        self.s_ref += self.v_b * DT
        return np.r_[np.asarray(r.u_opt, float), r.u_arm_seq[0]]

    def report(self):
        r = super().report()
        r.update(place=dict(pph=self.pph, events=self.place_ev, carry_req=self.carry_req, carry=self.carry_info, park=None if self.park is None else [float(v) for v in self.park],
                            h_lift=self.h_lift, d_freeze=self.d_freeze, metrics=getattr(self, 'place_metrics', None), reg=self.preg, place=self.place,
                            carry_path=None if self.cpath is None else self.cpath.P.round(5).tolist(), monitor=None if self.cmon is None else dict(n_check=self.cmon.n_check, max_cost=self.cmon.max_cost, why=self.cmon.why)))
        return r
