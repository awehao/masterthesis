#!/usr/bin/env python3
"""第八次第一包候選：預測通行衝突層 r2（Codex 20261110_182724 第 2 項；r2 依 20261110_183705 修正）。純邏輯、不依賴 ROS。
輸入只有線上量：機器人位姿與實測速度、剩餘計畫路徑、追蹤器確認航跡（位置、速度）與其年齡、代價圖是否可用（blocked(p) 由呼叫端給）。
不讀行人的預設路線、折返時刻或真值速度。以下參數全是**開發假設**（不是不確定性界，也不為過案掃參）：
  V_MOVER 0.15 m/s 以上算移動物；等速外推；掃掠半徑 rc(τ)＝R_BODY 0.30＋R_BASE 0.39＋MARGIN 0.15＋SIG0 0.05＋SIG_V 0.05·τ（隨預測時間增大）。
  T_H 5 s：路徑衝突時間比對的時域；T_LONG 12 s：「站在這裡會不會被掃到」看的前方路線長度（半徑一樣隨 τ 增大）。
  機器人沿路徑以 V_NOM 0.30 m/s、沿離開線以 V_EXIT 0.25 m/s 估到達時間，另加 T_DELAY 0.35 s（感知＋追蹤＋命令鏈延遲）。
  煞停距離 d_brake(v)＝v·T_DELAY＋v²／(2·A_BRAKE 0.5)。
失效關閉：航跡年齡 > TRACK_MAX_AGE 0.5 s 或沒有航跡訊息 ⇒ HOLD_STALE（不放行）；blocked(p) 在代價圖缺失／過期時由呼叫端回 True。
決策（每週期重算、重驗；同側／同停點只在仍通過驗證時沿用，以防抖）：
  先算「若現在開始煞車」的實際停止位置 p_stop。
  EXIT  p_stop 落在某移動物的前方路線帶內（原地等也會被掃到）⇒ 垂直該移動物速度離開。選邊只作排序：沿用中的一側優先；
        否則橫向偏移 < LAT_TIE 時往路徑延續側、否則往已偏的一側。每一側都驗：離開線依 RES 加密查 blocked、離開途中各點對所有移動物的
        時間掃掠、終點不在任何前方路線帶。都不過 ⇒ STOP_BLOCKED（呼叫端照既有保護處理；停止不保證行人不撞上來）。
  WAIT  路徑在 T_H 內與某移動物時間掃掠衝突 ⇒ 停點＝第一個衝突點之前、弧長 ≥ d_brake、不在任何前方路線帶、到停點途中不衝突、
        沿途與停點不被 blocked 的最遠路徑點；沒有 ⇒ 若 p_stop 安全就 WAIT（kind＝stop_now，實際停在 p_stop，不是瞬間停住），否則改評估 EXIT。
  CRUISE 無衝突、p_stop 不在前方路線帶，且連續 T_CLEAR 1 s。
"""
import math

import numpy as np

V_MOVER, T_H, T_LONG, DT_WIN, T_CLEAR = 0.15, 5.0, 12.0, 1.5, 1.0
R_BODY, R_BASE, MARGIN, SIG0, SIG_V = 0.30, 0.39, 0.15, 0.05, 0.05
V_NOM, V_EXIT, A_BRAKE, T_DELAY = 0.30, 0.25, 0.5, 0.35
TRACK_MAX_AGE = 0.5
EXIT_EXTRA, LAT_TIE = 0.20, 0.25
RES = 0.05                                               # 加密查代價的間距（＝地圖解析度）
DTAU = 0.1


def rc(tau):
    return R_BODY + R_BASE + MARGIN + SIG0 + SIG_V * np.asarray(tau, float)


def d_brake(v):
    v = abs(float(v))
    return v * T_DELAY + v * v / (2.0 * A_BRAKE)


def movers(tracks):
    return [x for x in tracks if x.get('confirmed') and math.hypot(x['vx'], x['vy']) >= V_MOVER]


def _centers(m, t1):
    taus = np.arange(0.0, t1 + 1e-9, DTAU)
    return taus, np.array([m['x'], m['y']])[None, :] + taus[:, None] * np.array([m['vx'], m['vy']])[None, :]


def in_band(P, m, t1=T_LONG):
    """點站著不動，在 [0, t1] 內是否會落入移動物掃掠（半徑隨 τ 增大）。"""
    taus, C = _centers(m, t1)
    P = np.atleast_2d(P)
    D = np.hypot(P[:, None, 0] - C[None, :, 0], P[:, None, 1] - C[None, :, 1])
    return (D < rc(taus)[None, :]).any(axis=1)


def traj_hits(P, tau_p, m):
    """機器人軌跡點 P（N×2）於時刻 tau_p（N）是否與移動物時間掃掠衝突（|τ−τ_p| ≤ DT_WIN）。回傳第一個衝突索引或 None。"""
    taus, C = _centers(m, T_H + DT_WIN)
    R = rc(taus)
    for i in range(len(P)):
        if tau_p[i] > T_H + DT_WIN:
            break
        k = np.abs(taus - tau_p[i]) <= DT_WIN
        if k.any() and (np.hypot(C[k, 0] - P[i, 0], C[k, 1] - P[i, 1]) < R[k]).any():
            return i
    return None


def densify(a, b, res=RES):
    n = max(2, int(math.ceil(float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / (res / 2.0))) + 1)
    return np.asarray(a, float)[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (np.asarray(b, float) - np.asarray(a, float))[None, :]


class ConflictLayer:
    def __init__(self, blocked=None):
        self.mode, self.t_noconf = 'CRUISE', None
        self.blocked = blocked or (lambda p: True)           # 預設失效關閉
        self.hold, self.side, self.rev = None, None, 0
        self.log = []

    def _set(self, t, mode, **info):
        if mode != self.mode or info.get('changed'):
            self.log.append(dict(t=round(float(t), 3), mode=mode, **{k: v for k, v in info.items() if k != 'changed'}))
        if mode != self.mode:
            self.rev += 1
        self.mode = mode

    def _static_ok(self, pts):
        return not any(self.blocked(p) for p in pts)

    def _exit_option(self, rp, m, sd, M):
        v = np.array([m['vx'], m['vy']])
        vh = v / max(np.linalg.norm(v), 1e-9)
        nrm = np.array([-vh[1], vh[0]])
        lat = float((rp - np.array([m['x'], m['y']])) @ nrm)
        tau_pass = float(np.clip(((rp - np.array([m['x'], m['y']])) @ vh) / max(np.linalg.norm(v), 1e-9), 0.0, T_LONG))   # 行人走到機器人這一段的時間
        need = float(rc(tau_pass)) - sd * lat + EXIT_EXTRA                                   # 與終點驗證用同一個隨時間增大的半徑
        goal = rp + sd * nrm * need
        line = densify(rp, goal)
        if not self._static_ok(line):
            return None, 'static'
        tau = T_DELAY + np.linalg.norm(line - rp, axis=1) / V_EXIT
        if any(traj_hits(line, tau, mm) is not None for mm in M if mm is not m):
            return None, 'other_mover'
        k_end = traj_hits(line[-1:], tau[-1:], m)
        if k_end is not None or any(in_band(goal, mm)[0] for mm in M):
            return None, 'goal_in_band'
        return dict(kind='exit', goal=goal.round(3).tolist(), side=sd, lat=round(lat, 3), need=round(need, 3), tau_pass=round(tau_pass, 2),
                    mover=[round(m['x'], 3), round(m['y'], 3), round(m['vx'], 3), round(m['vy'], 3)]), None

    def update(self, t, pose, vel, path, tracks, tracks_age):
        """pose＝(x, y, yaw)；vel＝實測世界速度 (vx, vy)；path＝剩餘路徑（自投影點起）；tracks_age＝最新航跡訊息年齡（None＝沒有）。"""
        rp = np.array(pose[:2], float)
        if tracks_age is None or tracks_age > TRACK_MAX_AGE:
            self._set(t, 'HOLD_STALE', age=None if tracks_age is None else round(float(tracks_age), 3))
            self.hold = None
            return dict(mode='HOLD_STALE', reason='tracks_stale')
        M = movers(tracks)
        vel = np.asarray(vel, float)
        sp = float(np.linalg.norm(vel))
        p_stop = rp + (vel / sp) * d_brake(sp) if sp > 1e-6 else rp.copy()
        path = np.asarray(path, float)
        if len(path) == 0:
            path = rp[None, :]
        seg = np.r_[np.linalg.norm(path[0] - rp), np.linalg.norm(np.diff(path, axis=0), axis=1)]
        s = np.cumsum(seg)
        tau_path = T_DELAY + s / V_NOM
        inside = [m for m in M if in_band(p_stop, m)[0]]
        conf = []
        for m in M:
            i = traj_hits(path, tau_path, m)
            if i is not None:
                conf.append((i, m))
        if not conf and not inside:
            if self.mode != 'CRUISE':
                self.t_noconf = self.t_noconf if self.t_noconf is not None else t
                if t - self.t_noconf >= T_CLEAR:
                    self._set(t, 'CRUISE')
                    self.hold, self.side = None, None
            return dict(mode=self.mode, n_movers=len(M), hold=self.hold, rev=self.rev)
        self.t_noconf = None

        def try_exit(threat):
            v = np.array([threat['vx'], threat['vy']])
            vh = v / max(np.linalg.norm(v), 1e-9)
            nrm = np.array([-vh[1], vh[0]])
            lat = float((rp - np.array([threat['x'], threat['y']])) @ nrm)
            ahead = path[min(len(path) - 1, int(1.0 / RES))] - rp
            ap = float(ahead @ nrm)
            first = self.side if (self.mode == 'EXIT' and self.side is not None) else ((math.copysign(1.0, ap) if abs(ap) > 1e-6 else 1.0) if abs(lat) < LAT_TIE else math.copysign(1.0, lat))
            why = {}
            for sd in (first, -first):
                if self.mode == 'EXIT' and self.hold is not None and self.side == sd:          # 沿用中的目標：仍通過驗證才沿用
                    g = np.array(self.hold['goal'])
                    line = densify(rp, g)
                    tau = T_DELAY + np.linalg.norm(line - rp, axis=1) / V_EXIT
                    if self._static_ok(line) and not any(traj_hits(line, tau, mm) is not None for mm in M if mm is not threat) and not any(in_band(g, mm)[0] for mm in M):
                        return dict(self.hold), None
                opt, w = self._exit_option(rp, threat, sd, M)
                if opt is not None:
                    return opt, None
                why[sd] = w
            return None, why

        if inside:
            threat = min(inside, key=lambda m_: math.hypot(m_['x'] - rp[0], m_['y'] - rp[1]))
            opt, why = try_exit(threat)
            if opt is not None:
                changed = self.hold is None or self.hold.get('goal') != opt['goal']
                self.hold, self.side = opt, opt['side']
                self._set(t, 'EXIT', changed=changed, **opt)
                return dict(mode='EXIT', hold=self.hold, n_movers=len(M), rev=self.rev, changed=changed)
            self.hold, self.side = None, None
            self._set(t, 'STOP_BLOCKED', why=why, note='兩側都不通過；照既有保護停止，停止不保證行人不撞上來')
            return dict(mode='STOP_BLOCKED', n_movers=len(M), why=why, rev=self.rev)
        i_c, m = min(conf, key=lambda c_: c_[0])
        db = d_brake(sp)

        def stop_ok(j):
            if s[j] < db or any(in_band(path[j], mm)[0] for mm in M):
                return False
            pts = np.vstack([densify(path[k], path[k + 1]) for k in range(0, j)] + [densify(rp, path[0])]) if j > 0 else densify(rp, path[0])
            if not self._static_ok(pts):
                return False
            return not any(traj_hits(path[:j + 1], tau_path[:j + 1], mm) is not None for mm in M)
        cand = None
        if self.mode == 'WAIT' and self.hold is not None and self.hold.get('kind') == 'wait' and self.hold.get('stop') is not None:
            d = np.linalg.norm(path - np.array(self.hold['stop']), axis=1)
            j0 = int(np.argmin(d))
            if d[j0] < 0.06 and j0 < i_c and stop_ok(j0):
                cand = j0
        if cand is None:
            ok = [j for j in range(i_c) if stop_ok(j)]
            cand = ok[-1] if ok else None
        if cand is not None:
            opt = dict(kind='wait', stop=path[cand].round(3).tolist(), i_stop=int(cand), i_conflict=int(i_c), s_stop=round(float(s[cand]), 3), d_brake=round(db, 3),
                       mover=[round(m['x'], 3), round(m['y'], 3), round(m['vx'], 3), round(m['vy'], 3)])
        elif not any(in_band(p_stop, mm)[0] for mm in M) and self._static_ok(densify(rp, p_stop)):
            opt = dict(kind='stop_now', stop=p_stop.round(3).tolist(), i_stop=-1, i_conflict=int(i_c), d_brake=round(db, 3),
                       mover=[round(m['x'], 3), round(m['y'], 3), round(m['vx'], 3), round(m['vy'], 3)], note='前方沒有可用停點；實際停在煞停位置，不是瞬間停住')
        else:
            opt, why = try_exit(m)
            if opt is None:
                self.hold, self.side = None, None
                self._set(t, 'STOP_BLOCKED', why=why, note='無可用停點也無可行離開；照既有保護停止，停止不保證行人不撞上來')
                return dict(mode='STOP_BLOCKED', n_movers=len(M), why=why, rev=self.rev)
            changed = self.hold is None or self.hold.get('goal') != opt['goal']
            self.hold, self.side = opt, opt['side']
            self._set(t, 'EXIT', changed=changed, **opt)
            return dict(mode='EXIT', hold=self.hold, n_movers=len(M), rev=self.rev, changed=changed)
        changed = self.hold is None or self.hold.get('stop') != opt['stop']
        self.hold = opt
        self._set(t, 'WAIT', changed=changed, **opt)
        return dict(mode='WAIT', hold=self.hold, n_movers=len(M), rev=self.rev, changed=changed)
