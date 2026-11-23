#!/usr/bin/env python3
"""第八次第一包候選 r4：穿越時窗層 v2（Codex 20261110_194328 核可 A）。純邏輯、不依賴 ROS；m8_window.py（r3）保留作紀錄。
相對 v1 的修正：
1 移動物預測：逐航跡記錄速度歷史。剛出現（< T_STABLE）、速度變化大（1 s 內 > DV_UNSTABLE）、或確認航跡但速度 < V_MOVER（停下／起步中）
  一律視為「不穩」：有方向（速度 ≥ 0.05）時沿該方向以事前宣告的速度上界 V_UB 0.8 m/s 外推；沒有方向時以 V_UB 向四周擴張的圓
  （擴張只算到 T_OMNI 3 s，每週期重算；否則 12 s 會蓋滿整個房間）。
  穩定的移動物照估計速度等速外推。V_UB 是本開發場景的適用範圍先驗（全案例共用），不讀各行人的生成速度、路線或折返時刻，也不代表一般行人。
2 機器人通過時間：每個路徑點給 [t_lb, t_ub]。t_lb＝延遲 T_DELAY_LB＋以 A_LB 加速到 V_LB；t_ub＝延遲 T_DELAY＋以 A_UB 加速到 V_UB_R
  ＋沿路徑轉向角／W_TURN（保守上界：涵蓋轉向、輪級限制、Shield 與跟隨落後的開發假設）。
3 衝突改逐點時空核對：路徑點 i 的機器人佔用 [t_lb−DT_MARGIN, t_ub＋DT_MARGIN] 與該點被任一移動物覆蓋的時間有交集＝衝突。
  COMMIT 要求自機器人到離開所有覆蓋的整段都沒有衝突（不再只比走出時間與行人到達目前位置的時間）。
其餘（失效關閉、WAIT 停點、EXIT 側移契約、STOP_BLOCKED、WAIT_TIMEOUT、T_CLEAR）同 v1；參數都是開發假設，DT_MARGIN 不加大。
"""
import math
from collections import deque

import numpy as np

V_MOVER, V_UB, T_LONG, DTAU = 0.15, 0.80, 12.0, 0.1
T_STABLE, DV_UNSTABLE, V_DIR_MIN = 1.5, 0.15, 0.05
T_OMNI = 3.0                                            # 方向不明時擴張圓的決策時域（每週期重算；開發假設：機器人反應能力為秒級）
R_BODY, R_BASE, MARGIN, SIG0, SIG_V = 0.30, 0.39, 0.15, 0.05, 0.05
T_DELAY_LB, A_LB, V_LB = 0.20, 1.0, 0.35                # 最早到達（樂觀）
T_DELAY, A_UB, V_UB_R, W_TURN = 0.35, 0.30, 0.22, 0.6   # 最晚通過（保守）
A_BRAKE = 0.5
DT_MARGIN, TRACK_MAX_AGE, T_CLEAR, T_WAIT_MAX = 1.0, 0.5, 1.0, 60.0
WAIT_GAP, EXIT_EXTRA, RES = 0.20, 0.20, 0.05
T_LAT0, V_LAT = 1.0, 0.22                               # 側移保守上界（契約實測 0.25 m/s 斜率，取 0.22）


def rz(tau):
    return R_BODY + R_BASE + MARGIN + SIG0 + SIG_V * np.asarray(tau, float)


def d_brake(v):
    v = abs(float(v))
    return v * T_DELAY + v * v / (2.0 * A_BRAKE)


def _profile(s, v0, delay, a, vmax):
    s = np.maximum(np.asarray(s, float), 0.0)
    v0 = min(max(float(v0), 0.0), vmax)
    s_acc = (vmax ** 2 - v0 ** 2) / (2.0 * a)
    t_acc = np.where(s <= s_acc, (-v0 + np.sqrt(v0 * v0 + 2.0 * a * np.minimum(s, s_acc))) / a, (vmax - v0) / a)
    return delay + t_acc + np.maximum(s - s_acc, 0.0) / vmax


def robot_window(path, rp, v0):
    seg = np.r_[np.linalg.norm(path[0] - rp), np.linalg.norm(np.diff(path, axis=0), axis=1)]
    s = np.cumsum(seg)
    d = np.diff(np.vstack([rp[None, :], path]), axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    dh = np.abs(np.angle(np.exp(1j * np.diff(np.r_[h[0], h]))))
    turn = np.cumsum(dh) / W_TURN
    return s, _profile(s, v0, T_DELAY_LB, A_LB, V_LB), _profile(s, v0, T_DELAY, A_UB, V_UB_R) + turn


def t_lat(d):
    return T_LAT0 + max(float(d), 0.0) / V_LAT


class TrackHistory:
    def __init__(self):
        self.h = {}

    def update(self, t, tracks):
        for x in tracks:
            if not x.get('confirmed'):
                continue
            e = self.h.setdefault(x['tid'], dict(t0=t, sp=deque(maxlen=40)))
            e['sp'].append((t, math.hypot(x['vx'], x['vy'])))
        for k in [k for k, e in self.h.items() if t - e['sp'][-1][0] > 5.0]:
            del self.h[k]

    def kind(self, t, x):
        e = self.h.get(x['tid'])
        sp = math.hypot(x['vx'], x['vy'])
        if e is None or t - e['t0'] < T_STABLE:
            return 'unstable'
        old = [v for tt, v in e['sp'] if t - tt >= 1.0]
        if sp < V_MOVER or (old and abs(sp - old[-1]) > DV_UNSTABLE):
            return 'unstable'
        return 'stable'


def cover_times(P, x, kind):
    """各點被覆蓋的時間（N×K 布林）與 τ。"""
    taus = np.arange(0.0, T_LONG + 1e-9, DTAU)
    P = np.atleast_2d(P)
    p0, v = np.array([x['x'], x['y']]), np.array([x['vx'], x['vy']])
    sp = float(np.linalg.norm(v))
    if kind == 'stable':
        C = p0[None, :] + taus[:, None] * v[None, :]
        R = rz(taus)
    elif sp >= V_DIR_MIN:
        C = p0[None, :] + taus[:, None] * (v / sp * max(sp, V_UB))[None, :]
        R = rz(taus)
    else:
        C = np.repeat(p0[None, :], len(taus), axis=0)
        R = rz(taus) + V_UB * np.minimum(taus, T_OMNI)
    D = np.hypot(P[:, None, 0] - C[None, :, 0], P[:, None, 1] - C[None, :, 1])
    return D < R[None, :], taus


def densify(a, b, res=RES):
    n = max(2, int(math.ceil(float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / (res / 2.0))) + 1)
    return np.asarray(a, float)[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (np.asarray(b, float) - np.asarray(a, float))[None, :]


class WindowLayer2:
    def __init__(self, blocked=None):
        self.mode, self.t_noconf, self.t_wait0 = 'CRUISE', None, None
        self.blocked = blocked or (lambda p: True)
        self.hold, self.side, self.log = None, None, []
        self.hist = TrackHistory()

    def _set(self, t, mode, changed=False, **info):
        if mode != self.mode or changed:
            self.log.append(dict(t=round(float(t), 3), mode=mode, **info))
        if mode == 'WAIT' and self.mode != 'WAIT':
            self.t_wait0 = t
        if mode not in ('WAIT', 'WAIT_TIMEOUT'):
            self.t_wait0 = None
        self.mode = mode

    def _ok_line(self, a, b):
        return not any(self.blocked(p) for p in densify(a, b))

    def _point_conflicts(self, path, t_lb, t_ub, M):
        """逐點：回傳 (衝突布林陣列, 是否被任一移動物在 T_LONG 內覆蓋)。"""
        conf = np.zeros(len(path), bool)
        cov_any = np.zeros(len(path), bool)
        for x, k in M:
            cov, taus = cover_times(path, x, k)
            cov_any |= cov.any(axis=1)
            for i in np.flatnonzero(cov.any(axis=1)):
                tt = taus[cov[i]]
                if (tt >= t_lb[i] - DT_MARGIN).any() and (tt <= t_ub[i] + DT_MARGIN).any() and ((tt >= t_lb[i] - DT_MARGIN) & (tt <= t_ub[i] + DT_MARGIN)).any():
                    conf[i] = True
        return conf, cov_any

    def _first_arrival(self, p, M):
        """原地站著最早何時被覆蓋；方向不明（停下）的物體只看 T_OMNI 內（每週期重算），有方向的看 T_LONG。"""
        best = None
        for x, k in M:
            cov, taus = cover_times(p, x, k)
            if k == 'unstable' and math.hypot(x['vx'], x['vy']) < V_DIR_MIN:
                cov = cov & (taus <= T_OMNI)[None, :]
            j = np.flatnonzero(cov[0])
            if len(j) and (best is None or taus[j[0]] < best[0]):
                best = (float(taus[j[0]]), x, k)
        return best

    def _exit(self, rp, x, M, t_avail):
        v = np.array([x['vx'], x['vy']])
        if np.linalg.norm(v) < V_DIR_MIN:
            return None, {'any': 'threat_has_no_direction'}
        vh = v / np.linalg.norm(v)
        nrm = np.array([-vh[1], vh[0]])
        lat = float((rp - np.array([x['x'], x['y']])) @ nrm)
        first = self.side if self.side is not None else (math.copysign(1.0, lat) if abs(lat) > 1e-6 else 1.0)
        why = {}
        for sd in (first, -first):
            need = float(rz(t_avail)) - sd * lat + EXIT_EXTRA
            goal = rp + sd * nrm * need
            if t_lat(need) + DT_MARGIN >= t_avail:
                why[sd] = f'too_slow t_lat {t_lat(need):.2f} ≥ {t_avail - DT_MARGIN:.2f}'
                continue
            if not self._ok_line(rp, goal):
                why[sd] = 'static'
                continue
            if self._first_arrival(goal, M) is not None:
                why[sd] = 'goal_in_band'
                continue
            return dict(kind='exit', goal=goal.round(3).tolist(), side=sd, need=round(need, 3), t_lat=round(t_lat(need), 2), t_avail=round(t_avail, 2)), None
        return None, why

    def update(self, t, pose, vel, path, tracks, tracks_age):
        rp = np.array(pose[:2], float)
        if tracks_age is None or tracks_age > TRACK_MAX_AGE:
            self._set(t, 'HOLD_STALE', age=None if tracks_age is None else round(float(tracks_age), 3))
            self.hold = None
            return dict(mode='HOLD_STALE')
        self.hist.update(t, tracks)
        M = [(x, self.hist.kind(t, x)) for x in tracks if x.get('confirmed')]
        vel = np.asarray(vel, float)
        sp = float(np.linalg.norm(vel))
        path = np.asarray(path, float) if len(path) else rp[None, :]
        hd = path[min(len(path) - 1, 4)] - rp
        v_along = max(0.0, float(vel @ hd) / max(np.linalg.norm(hd), 1e-9))
        s, t_lb, t_ub = robot_window(path, rp, v_along)
        conf, cov_any = self._point_conflicts(path, t_lb, t_ub, M)
        fa = self._first_arrival(rp, M)
        n_unst = sum(1 for _, k in M if k == 'unstable')
        if fa is not None:                                                    # 在帶內
            ta, x, k = fa
            nc = np.flatnonzero(~cov_any)
            i_out = int(nc[0]) if len(nc) else len(path) - 1
            if not conf[:i_out + 1].any():
                self._set(t, 'COMMIT', i_out=i_out, s_out=round(float(s[i_out]), 2), t_out_ub=round(float(t_ub[i_out]), 2), t_arrive=round(ta, 2), threat_kind=k)
                self.hold, self.side = None, None
                return dict(mode='COMMIT')
            opt, why = self._exit(rp, x, M, ta)
            if opt is not None:
                changed = self.mode != 'EXIT' or self.hold is None or np.linalg.norm(np.array(self.hold['goal']) - np.array(opt['goal'])) > 0.10
                if changed:
                    self.hold, self.side = opt, opt['side']
                self._set(t, 'EXIT', changed=changed, **self.hold)
                return dict(mode='EXIT', hold=self.hold, changed=changed)
            self.hold, self.side = None, None
            self._set(t, 'STOP_BLOCKED', why=why, threat_kind=k, t_arrive=round(ta, 2), note='候選評估的 COMMIT／EXIT 均未獲接受；停止不保證行人不撞上來')
            return dict(mode='STOP_BLOCKED', why=why)
        self.side = None
        if not conf.any():
            if self.mode != 'CRUISE':
                self.t_noconf = self.t_noconf if self.t_noconf is not None else t
                if t - self.t_noconf >= T_CLEAR:
                    self._set(t, 'CRUISE', n_unstable=n_unst)
                    self.hold = None
            return dict(mode=self.mode, hold=self.hold)
        self.t_noconf = None
        if self.mode == 'WAIT' and self.t_wait0 is not None and t - self.t_wait0 > T_WAIT_MAX:
            self._set(t, 'WAIT_TIMEOUT', waited=round(t - self.t_wait0, 1))
            return dict(mode='WAIT_TIMEOUT')
        i_c = int(np.flatnonzero(conf)[0])
        db = d_brake(sp)
        j = None
        for jj in range(i_c - 1, -1, -1):
            if s[jj] < db or s[jj] > s[i_c] - WAIT_GAP + 1e-9 or cov_any[jj]:
                continue
            pts = [rp] + list(path[:jj + 1])
            if all(self._ok_line(pts[q], pts[q + 1]) for q in range(len(pts) - 1)):
                j = jj
                break
        if j is not None:
            opt = dict(kind='wait', stop=path[j].round(3).tolist(), i_stop=int(j), s_stop=round(float(s[j]), 3), i_conflict=i_c, n_unstable=n_unst)
        else:
            p_stop = rp + (vel / sp) * db if sp > 1e-6 else rp.copy()
            if self._first_arrival(p_stop, M) is None and self._ok_line(rp, p_stop):
                opt = dict(kind='stop_now', stop=p_stop.round(3).tolist(), i_stop=-1, i_conflict=i_c, note='前方沒有可用停點；實際停在煞停位置，不是瞬間停住')
            else:
                self._set(t, 'STOP_BLOCKED', why='no_wait_point_and_stop_in_band', note='停止不保證行人不撞上來')
                self.hold = None
                return dict(mode='STOP_BLOCKED', why='no_wait_point_and_stop_in_band')
        changed = self.mode != 'WAIT' or self.hold is None or self.hold.get('kind') != opt['kind'] or (opt['kind'] == 'wait' and self.hold.get('stop') != opt['stop'])
        self.hold = opt
        self._set(t, 'WAIT', changed=changed, **opt)
        return dict(mode='WAIT', hold=opt, changed=changed)
