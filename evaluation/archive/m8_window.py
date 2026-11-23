#!/usr/bin/env python3
"""第八次第一包候選 r3：穿越時窗層（Codex 20261110_190030 選 C）。純邏輯、不依賴 ROS。
核心：在預測衝突區外等待，取得足夠通過時窗才進入，不是進入後反覆被趕開。輸入只有線上量（位姿、實測速度、剩餘路徑、確認航跡與年齡、
blocked(p)＝代價圖查詢，缺失／過期由呼叫端回 True）；不讀行人生成路線、週期或折返時刻。參數全是開發假設：
  移動物：確認航跡且速度 ≥ V_MOVER；等速外推 T_LONG；佔用半徑 rz(τ)＝R_BODY＋R_BASE＋MARGIN＋SIG0＋SIG_V·τ（已含底盤半徑）。
  衝突區：剩餘路徑上在某 τ∈[0, T_LONG] 被 rz(τ) 覆蓋的第一段連續點 [i_a, i_b]；行人佔用時間＝這些點被覆蓋的 τ 範圍 [tm_in, tm_out]。
  機器人佔用時間：自當下速度、延遲 T_DELAY、加速 A_ACC 到 V_NOM，走到 s(i_a)（底盤前緣進入，rz 已含半徑）與 s(i_b)（整個底盤離開）；
    前後各加 DT_MARGIN。兩區間重疊＝衝突。
決策（每週期重算）：
  HOLD_STALE  航跡年齡 > TRACK_MAX_AGE 或沒有航跡訊息（不放行）。
  在帶內（機器人目前位置在 T_LONG 內會被覆蓋）：
    COMMIT    往前走完該衝突區（到 s(i_b)）的時間＋DT_MARGIN ＜ 行人到達機器人目前位置的時間 ⇒ 照原路徑走（不停）。
    EXIT      否則以側移離開（垂直該行人速度，計畫朝向＝當下車頭；執行時間依 plan_yaw 契約核對 t_lat(d)＝1.0＋d／0.25），
              時間來得及、離開線與終點靜態可行、終點不在任何帶內才採用；兩側都不行 ⇒ STOP_BLOCKED（既有保護；停止不保證行人不撞上來）。
  不在帶內但路徑前方衝突：
    WAIT      停點＝s(i_a)−WAIT_GAP（≥ 煞停距離、不在任何帶內、沿途靜態可行）；做不到 ⇒ 若「往前穿越」不衝突就 COMMIT，
              否則若煞停位置在帶外就原地停（kind＝stop_now，實際停在煞停位置），否則 EXIT／STOP_BLOCKED。
    WAIT 連續超過 T_WAIT_MAX ⇒ WAIT_TIMEOUT（記未完成，不強行穿越）。
  CRUISE      無衝突且不在帶內，連續 T_CLEAR 秒。
"""
import math

import numpy as np

V_MOVER, T_LONG, DTAU = 0.15, 12.0, 0.1
R_BODY, R_BASE, MARGIN, SIG0, SIG_V = 0.30, 0.39, 0.15, 0.05, 0.05
V_NOM, A_ACC, A_BRAKE, T_DELAY = 0.30, 0.5, 0.5, 0.35
DT_MARGIN, TRACK_MAX_AGE, T_CLEAR, T_WAIT_MAX = 1.0, 0.5, 1.0, 60.0
WAIT_GAP, EXIT_EXTRA, RES = 0.20, 0.20, 0.05
T_LAT0, V_LAT = 1.0, 0.25                               # plan_yaw 側移契約（m8contract_py_lat：0.5 m 3.03 s、0.8 m 4.23 s）


def rz(tau):
    return R_BODY + R_BASE + MARGIN + SIG0 + SIG_V * np.asarray(tau, float)


def d_brake(v):
    v = abs(float(v))
    return v * T_DELAY + v * v / (2.0 * A_BRAKE)


def t_reach(s, v0):
    """自目前速度 v0 起（延遲後）以 A_ACC 加速到 V_NOM，走完弧長 s 的時間。"""
    s = max(float(s), 0.0)
    v0 = min(max(float(v0), 0.0), V_NOM)
    s_acc = (V_NOM ** 2 - v0 ** 2) / (2.0 * A_ACC)
    if s <= s_acc:
        return T_DELAY + (-v0 + math.sqrt(v0 * v0 + 2.0 * A_ACC * s)) / A_ACC
    return T_DELAY + (V_NOM - v0) / A_ACC + (s - s_acc) / V_NOM


def t_lat(d):
    return T_LAT0 + max(float(d), 0.0) / V_LAT


def movers(tracks):
    return [x for x in tracks if x.get('confirmed') and math.hypot(x['vx'], x['vy']) >= V_MOVER]


def _cover(P, m):
    """各點被移動物覆蓋的時間（N×K 布林）與 τ 陣列。"""
    taus = np.arange(0.0, T_LONG + 1e-9, DTAU)
    C = np.array([m['x'], m['y']])[None, :] + taus[:, None] * np.array([m['vx'], m['vy']])[None, :]
    P = np.atleast_2d(P)
    D = np.hypot(P[:, None, 0] - C[None, :, 0], P[:, None, 1] - C[None, :, 1])
    return D < rz(taus)[None, :], taus


def first_arrival(p, m):
    cov, taus = _cover(p, m)
    k = np.flatnonzero(cov[0])
    return None if not len(k) else float(taus[k[0]])


def zone(path, m):
    cov, taus = _cover(path, m)
    hit = cov.any(axis=1)
    idx = np.flatnonzero(hit)
    if not len(idx):
        return None
    i_a = int(idx[0])
    i_b = i_a
    while i_b + 1 < len(path) and hit[i_b + 1]:
        i_b += 1
    occ = cov[i_a:i_b + 1].any(axis=0)
    k = np.flatnonzero(occ)
    return dict(i_a=i_a, i_b=i_b, tm_in=float(taus[k[0]]), tm_out=float(taus[k[-1]]), open_end=bool(k[-1] == len(taus) - 1))


def densify(a, b, res=RES):
    n = max(2, int(math.ceil(float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / (res / 2.0))) + 1)
    return np.asarray(a, float)[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (np.asarray(b, float) - np.asarray(a, float))[None, :]


class WindowLayer:
    def __init__(self, blocked=None):
        self.mode, self.t_noconf, self.t_wait0 = 'CRUISE', None, None
        self.blocked = blocked or (lambda p: True)
        self.hold, self.side, self.log = None, None, []

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

    def _exit(self, rp, m, M, t_avail):
        v = np.array([m['vx'], m['vy']])
        vh = v / max(np.linalg.norm(v), 1e-9)
        nrm = np.array([-vh[1], vh[0]])
        lat = float((rp - np.array([m['x'], m['y']])) @ nrm)
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
            if any(first_arrival(goal, mm) is not None for mm in M):
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
        M = movers(tracks)
        vel = np.asarray(vel, float)
        sp = float(np.linalg.norm(vel))
        path = np.asarray(path, float) if len(path) else rp[None, :]
        s = np.cumsum(np.r_[np.linalg.norm(path[0] - rp), np.linalg.norm(np.diff(path, axis=0), axis=1)])
        heading = path[min(len(path) - 1, 4)] - rp
        v_along = max(0.0, float(vel @ heading) / max(np.linalg.norm(heading), 1e-9))
        inside = [(m, first_arrival(rp, m)) for m in M]
        inside = [(m, ta) for m, ta in inside if ta is not None]
        zs = []
        for m in M:
            z = zone(path, m)
            if z is None:
                continue
            r_in, r_out = t_reach(s[z['i_a']], v_along) - DT_MARGIN, t_reach(s[z['i_b']], v_along) + DT_MARGIN
            if not (r_out < z['tm_in'] or (r_in > z['tm_out'] and not z['open_end'])):
                zs.append((z, m, r_in, r_out))
        if inside:
            m, ta = min(inside, key=lambda x: x[1])
            z = zone(path, m)
            if z is not None and t_reach(s[z['i_b']], v_along) + DT_MARGIN < ta and not any(zz[1] is not m for zz in zs):
                self._set(t, 'COMMIT', s_out=round(float(s[z['i_b']]), 2), t_out=round(t_reach(s[z['i_b']], v_along), 2), t_arrive=round(ta, 2))
                self.hold, self.side = None, None
                return dict(mode='COMMIT')
            opt, why = self._exit(rp, m, M, ta)
            if opt is not None:
                changed = self.mode != 'EXIT' or self.hold is None or np.linalg.norm(np.array(self.hold['goal']) - np.array(opt['goal'])) > 0.10
                if changed:
                    self.hold, self.side = opt, opt['side']
                self._set(t, 'EXIT', changed=changed, **self.hold)
                return dict(mode='EXIT', hold=self.hold, changed=changed)
            self.hold, self.side = None, None
            self._set(t, 'STOP_BLOCKED', why=why, note='停止不保證行人不撞上來')
            return dict(mode='STOP_BLOCKED', why=why)
        self.side = None
        if not zs:
            if self.mode != 'CRUISE':
                self.t_noconf = self.t_noconf if self.t_noconf is not None else t
                if t - self.t_noconf >= T_CLEAR:
                    self._set(t, 'CRUISE')
                    self.hold = None
            return dict(mode=self.mode, hold=self.hold)
        self.t_noconf = None
        z, m, r_in, r_out = min(zs, key=lambda x: x[0]['i_a'])
        if self.mode == 'WAIT' and self.t_wait0 is not None and t - self.t_wait0 > T_WAIT_MAX:
            self._set(t, 'WAIT_TIMEOUT', waited=round(t - self.t_wait0, 1))
            return dict(mode='WAIT_TIMEOUT')
        db = d_brake(sp)
        s_w = s[z['i_a']] - WAIT_GAP
        cand = [j for j in range(z['i_a']) if db <= s[j] <= s_w + 1e-9]
        j = None
        for jj in reversed(cand):
            if any(first_arrival(path[jj], mm) is not None for mm in M):
                continue
            pts = [rp] + list(path[:jj + 1])
            if all(self._ok_line(pts[k], pts[k + 1]) for k in range(len(pts) - 1)):
                j = jj
                break
        if j is not None:
            opt = dict(kind='wait', stop=path[j].round(3).tolist(), i_stop=int(j), s_stop=round(float(s[j]), 3), i_a=z['i_a'], tm=[round(z['tm_in'], 2), round(z['tm_out'], 2)],
                       r=[round(r_in, 2), round(r_out, 2)])
        else:
            p_stop = rp + (vel / sp) * db if sp > 1e-6 else rp.copy()
            if not any(first_arrival(p_stop, mm) is not None for mm in M) and self._ok_line(rp, p_stop):
                opt = dict(kind='stop_now', stop=p_stop.round(3).tolist(), i_stop=-1, note='前方沒有可用停點；實際停在煞停位置，不是瞬間停住')
            else:
                opt, why = self._exit(rp, m, M, z['tm_in'])
                if opt is None:
                    self._set(t, 'STOP_BLOCKED', why=why, note='停止不保證行人不撞上來')
                    self.hold = None
                    return dict(mode='STOP_BLOCKED', why=why)
                self.hold, self.side = opt, opt['side']
                self._set(t, 'EXIT', changed=True, **opt)
                return dict(mode='EXIT', hold=opt, changed=True)
        changed = self.mode != 'WAIT' or self.hold is None or self.hold.get('kind') != opt['kind'] or (opt['kind'] == 'wait' and self.hold.get('stop') != opt['stop'])
        self.hold = opt
        self._set(t, 'WAIT', changed=changed, **opt)
        return dict(mode='WAIT', hold=opt, changed=changed)
