#!/usr/bin/env python3
"""第八次第一包候選：預測通行衝突層（Codex 20261110_182724 第 2 項；純邏輯、不依賴 ROS，給 m8_nav_obs_node r2 與離線重播共用）。
輸入只有線上量：機器人位姿、剩餘計畫路徑、追蹤器的確認航跡（位置、速度）。不讀行人的預設路線、折返時刻或真值速度。
移動物：確認航跡且估計速度 ≥ V_MOVER。以等速外推 T_H 秒，帶狀掃掠半徑 Rc(τ)＝R_BODY＋R_BASE＋MARGIN＋SIG0＋SIG_V·τ（不確定性隨預測時間放大）。
機器人沿剩餘路徑以 V_NOM 到達各點（τ_i＝弧長／V_NOM）。衝突＝存在路徑點 i 與 |τ − τ_i| ≤ DT_WIN 使該點落在 Rc(τ) 內。
決策：
  CRUISE  無衝突（連續 T_CLEAR 秒才從 WAIT／EXIT 回到 CRUISE）。
  WAIT    有衝突且機器人目前不在任何移動物的前方路線帶（τ∈[0, T_LONG]、半徑 rc(T_H)）內 ⇒ 停點＝第一個衝突點之前、最後一個不在任何前方路線帶內
          且距機器人 ≥ 煞停距離的路徑點；找不到這樣的點但機器人目前位置在掃掠外 ⇒ 原地停（停點＝目前位置）。
  EXIT    機器人已在某移動物的前方路線帶內（原地等也會被掃到）⇒ 垂直該移動物速度、朝機器人已偏向的一側離開，距離＝離開掃掠所需＋0.2 m；
          （橫向偏移 < LAT_TIE 時改往路徑延續的一側）；另一側只在這一側被靜態障礙擋住時才用（呼叫端提供 blocked(p) 檢查）；已在離開中不換邊；兩側都擋 ⇒ STOP_BLOCKED（呼叫端照既有規則受控停止）。
回傳 dict(mode, …)；不產生控制命令，由呼叫端把它轉成給 GMPC 的計畫（截斷路徑或短的離開路徑）。
"""
import math

import numpy as np

V_MOVER, T_H, DT_WIN, T_CLEAR = 0.15, 5.0, 1.5, 1.0
R_BODY, R_BASE, MARGIN, SIG0, SIG_V = 0.30, 0.39, 0.15, 0.05, 0.05   # SIG_V：m8nav_wire 實測行人速度誤差中位 0.005 m/s 的 10 倍
T_LONG = 12.0                                           # 停點與「原地是否安全」看行人前方路線的長度（s）
V_NOM, A_BRAKE = 0.30, 0.5
EXIT_EXTRA = 0.20
LAT_TIE = 0.25                                          # 橫向偏移小於此值視為在路線正中
DTAU = 0.1


def rc(tau):
    return R_BODY + R_BASE + MARGIN + SIG0 + SIG_V * np.asarray(tau, float)


def movers(tracks):
    return [x for x in tracks if x.get('confirmed') and math.hypot(x['vx'], x['vy']) >= V_MOVER]


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def in_full_sweep(P, m):
    """點是否落在移動物「前方路線」（τ∈[0, T_LONG]、半徑 rc(T_H)）內：站在這裡等，行人遲早會掃到（停點與原地安全都用這個，不只看 T_H）。"""
    a = np.array([m['x'], m['y']])
    b = a + T_LONG * np.array([m['vx'], m['vy']])
    return _seg_dist(np.atleast_2d(P), a, b) < float(rc(T_H))


def path_conflict(path, s_robot, m):
    """第一個衝突路徑點索引與時間（無衝突回 None）。path：剩餘路徑（N×2），s_robot：各點到機器人的弧長。"""
    tau_i = s_robot / V_NOM
    taus = np.arange(0.0, T_H + 1e-9, DTAU)
    C = np.array([m['x'], m['y']])[None, :] + taus[:, None] * np.array([m['vx'], m['vy']])[None, :]
    R = rc(taus)
    for i in range(len(path)):
        if tau_i[i] > T_H + DT_WIN:
            break
        k = (taus >= tau_i[i] - DT_WIN) & (taus <= tau_i[i] + DT_WIN)
        if not k.any():
            continue
        if (np.hypot(C[k, 0] - path[i, 0], C[k, 1] - path[i, 1]) < R[k]).any():
            return i, float(tau_i[i])
    return None


class ConflictLayer:
    def __init__(self, blocked=None):
        self.mode, self.t_mode, self.t_noconf = 'CRUISE', None, None
        self.blocked = blocked or (lambda p: False)
        self.hold = None
        self.log = []

    def _set(self, t, mode, **info):
        if mode != self.mode:
            self.log.append(dict(t=round(float(t), 3), mode=mode, **info))
        self.mode, self.t_mode = mode, t

    def update(self, t, pose, path, tracks, v_now=0.0):
        """pose＝(x, y, yaw)；path＝剩餘路徑（自機器人投影點起，N×2）；回傳決策。"""
        rp = np.array(pose[:2], float)
        M = movers(tracks)
        path = np.asarray(path, float)
        if len(path) == 0:
            return dict(mode=self.mode)
        seg = np.r_[np.linalg.norm(path[0] - rp), np.linalg.norm(np.diff(path, axis=0), axis=1)]
        s = np.cumsum(seg)
        conf = []
        for m in M:
            c = path_conflict(path, s, m)
            if c is not None:
                conf.append((c[0], c[1], m))
        inside = [m for m in M if in_full_sweep(rp, m)[0]]
        if not conf and not inside:
            if self.mode != 'CRUISE':
                self.t_noconf = self.t_noconf if self.t_noconf is not None else t
                if t - self.t_noconf >= T_CLEAR:
                    self._set(t, 'CRUISE')
                    self.hold = None
            return dict(mode=self.mode, n_movers=len(M), hold=self.hold)
        self.t_noconf = None
        if inside:
            m = min(inside, key=lambda m_: math.hypot(m_['x'] - rp[0], m_['y'] - rp[1]))
            v = np.array([m['vx'], m['vy']])
            vh = v / max(np.linalg.norm(v), 1e-9)
            nrm = np.array([-vh[1], vh[0]])
            lat = float((rp - np.array([m['x'], m['y']])) @ nrm)
            if self.mode == 'EXIT' and self.hold is not None:          # 已在離開中：不換邊、不重算目標
                return dict(mode='EXIT', hold=self.hold, n_movers=len(M))
            ahead = path[min(len(path) - 1, int(1.0 / 0.05))] - rp      # 路徑前方約 1 m 的方向
            side_path = math.copysign(1.0, float(ahead @ nrm)) if abs(float(ahead @ nrm)) > 1e-6 else 1.0
            first = side_path if abs(lat) < LAT_TIE else math.copysign(1.0, lat)   # 幾乎在路線正中：往路徑延續側；明顯偏一側：往已偏的一側
            sides = [first, -first]
            for sd in sides:
                need = float(rc(T_H)) - sd * lat + EXIT_EXTRA          # 同側＝rc−|lat|，對側＝rc＋|lat|
                goal = rp + sd * nrm * need
                pts = rp[None, :] + np.linspace(0.0, 1.0, 12)[:, None] * (goal - rp)[None, :]
                if not any(self.blocked(p) for p in pts):
                    if self.mode != 'EXIT' or self.hold is None:
                        self.hold = dict(kind='exit', goal=goal.round(3).tolist(), side=sd, lat=round(lat, 3), need=round(need, 3), mover=[round(m['x'], 3), round(m['y'], 3), round(m['vx'], 3), round(m['vy'], 3)])
                    self._set(t, 'EXIT', **self.hold)
                    return dict(mode='EXIT', hold=self.hold, n_movers=len(M))
            self._set(t, 'STOP_BLOCKED', lat=round(lat, 3))
            return dict(mode='STOP_BLOCKED', n_movers=len(M))
        i_c, tau_c, m = min(conf, key=lambda c_: c_[0])
        d_brake = v_now * v_now / (2.0 * A_BRAKE)
        ok = [i for i in range(i_c) if s[i] >= d_brake and not any(in_full_sweep(path[i], mm)[0] for mm in M) and not self.blocked(path[i])]
        if self.mode == 'WAIT' and self.hold is not None:
            return dict(mode='WAIT', hold=self.hold, n_movers=len(M))
        if ok:
            j = ok[-1]
            self.hold = dict(kind='wait', stop=path[j].round(3).tolist(), i_stop=int(j), i_conflict=int(i_c), tau_conflict=round(tau_c, 2), s_stop=round(float(s[j]), 3),
                             mover=[round(m['x'], 3), round(m['y'], 3), round(m['vx'], 3), round(m['vy'], 3)])
        else:
            self.hold = dict(kind='wait', stop=rp.round(3).tolist(), i_stop=-1, i_conflict=int(i_c), tau_conflict=round(tau_c, 2), s_stop=0.0,
                             mover=[round(m['x'], 3), round(m['y'], 3), round(m['vx'], 3), round(m['vy'], 3)], note='停在原地（前方沒有掃掠外的停點）')
        self._set(t, 'WAIT', **self.hold)
        return dict(mode='WAIT', hold=self.hold, n_movers=len(M))
