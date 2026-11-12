#!/usr/bin/env python3
"""理想抓持綁定的判定純函式（執行器與測試共用；不依賴 Isaac）。
  rel_motion      夾爪殼相對門扇的運動：用**同一相對變換**相鄰兩步的平移／轉動差分（不是兩個剛體質心線速度相減——共同剛轉時那個差不為零）。
  classify_points 接觸點分類：到橫桿軸線段的距離 ≤ 半徑＋tol 算橫桿，其餘（門板、立柱）算非橫桿。
  BindGate        逐步餵入；禁止接觸一出現即鎖存（之後永不放行）；全部條件連續成立才放行。"""
import math

import numpy as np


def rel_motion(Trel_prev, Trel_now, dt):
    dp = float(np.linalg.norm(Trel_now[:3, 3] - Trel_prev[:3, 3])) / dt
    c = float(np.clip((np.trace(Trel_prev[:3, :3].T @ Trel_now[:3, :3]) - 1) / 2, -1, 1))
    return dp, math.acos(c) / dt


def classify_points(points, forces, bar_c, bar_u, bar_hl, bar_r, tol=0.003):
    """回 (橫桿合力大小, 非橫桿合力大小, 非橫桿點數)。points (n,3)、forces (n,) 或 (n,3)。"""
    P = np.asarray(points, float).reshape(-1, 3)
    F = np.asarray(forces, float)
    F = np.abs(F).reshape(len(P), -1).sum(axis=1) if len(P) else np.zeros(0)
    if not len(P):
        return 0.0, 0.0, 0
    s = np.clip((P - bar_c) @ bar_u, -bar_hl, bar_hl)
    d = np.linalg.norm(P - (bar_c + s[:, None] * bar_u), axis=1)
    on = d <= bar_r + tol
    return float(F[on].sum()), float(F[~on].sum()), int(((~on) & (F > 0)).sum())


def rel_angle(Ta, Tb):
    """兩個位姿的相對轉角（rad）。用 atan2(‖反對稱部‖, (trace−1)/2)：接近零時不像 acos 那樣把量化誤差放大成 √ε。"""
    Re = Ta[:3, :3].T @ Tb[:3, :3]
    v = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    return math.atan2(float(np.linalg.norm(v)), 0.5 * (float(np.trace(Re)) - 1.0))


def window_motion(Trel_then, Trel_now, window_s):
    """同一相對變換在 window_s 窗的首尾差換算的平均相對速度（線、角）。smoke_2 的逐步差分（10 ms、float32 位姿）會把量化跳成 0.02 rad/s 級。"""
    return float(np.linalg.norm(Trel_now[:3, 3] - Trel_then[:3, 3])) / window_s, rel_angle(Trel_then, Trel_now) / window_s


def pose_from_pq(p, q_wxyz):
    """位置＋四元數（w,x,y,z）→ 4×4（float64）。檢查有限值並**正規化四元數**；不合法回 None。
    smoke_2 的舊算法未正規化：範數差 3e-8 的相同姿態就能算出 0.048 rad/s 的假角速度（Codex 20261112_213728 的反例）。"""
    p, q = np.asarray(p, np.float64).reshape(-1), np.asarray(q_wxyz, np.float64).reshape(-1)
    if p.shape != (3,) or q.shape != (4,) or not (np.isfinite(p).all() and np.isfinite(q).all()):
        return None
    n = float(np.linalg.norm(q))
    if not (0.5 < n < 2.0):
        return None
    w, x, y, z = q / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = p
    return T


def step_motion(Trel_prev, Trel_now, dt):
    """相鄰兩個有效樣本的相對速度（線、角）；轉角用 atan2（rel_angle）。"""
    return float(np.linalg.norm(Trel_now[:3, 3] - Trel_prev[:3, 3])) / dt, rel_angle(Trel_prev, Trel_now) / dt


class BindGate:
    def __init__(self, cond, dt):
        self.c, self.dt = cond, dt
        self.latched, self.n_still = None, 0
        self.hist, self.win = [], None                              # 窗差：只當診斷欄位，不進判定
        self.prev, self.still_s, self.last = None, 0.0, None        # r3：上一個有效樣本 (t, Trel)、連續靜止時間（依樣本時間戳累計）、最近一步速度
        self.n_invalid = 0

    def forbid(self, t, why, **kw):
        if self.latched is None:
            self.latched = dict(t=round(float(t), 3), why=why, **kw)

    def step(self, t, rel_lin, rel_ang, other_contact_n, finger_nonbar_n):
        if other_contact_n > self.c['forbidden_n']:
            self.forbid(t, 'nonfinger_contact', force_n=round(float(other_contact_n), 4))
        if finger_nonbar_n > self.c['forbidden_n']:
            self.forbid(t, 'finger_nonbar_contact', force_n=round(float(finger_nonbar_n), 4))
        still = rel_lin <= self.c['rel_lin_mps'] and rel_ang <= self.c['rel_ang_rps']
        self.n_still = self.n_still + 1 if still else 0

    def step_window(self, t, Trel, other_contact_n, finger_nonbar_n):
        """r3：禁止接觸照舊逐步鎖存；相對靜止改以 rel_still_s 窗的首尾差判定（門檻不變）。"""
        if other_contact_n > self.c['forbidden_n']:
            self.forbid(t, 'nonfinger_contact', force_n=round(float(other_contact_n), 4))
        if finger_nonbar_n > self.c['forbidden_n']:
            self.forbid(t, 'finger_nonbar_contact', force_n=round(float(finger_nonbar_n), 4))
        self.hist.append(np.array(Trel, float))
        n = int(round(self.c['rel_still_s'] / self.dt))
        if len(self.hist) > n + 1:
            self.hist.pop(0)
        if len(self.hist) == n + 1:
            lin, ang = window_motion(self.hist[0], self.hist[-1], n * self.dt)
            self.win = (lin, ang)
            self.n_still = n if (lin <= self.c['rel_lin_mps'] and ang <= self.c['rel_ang_rps']) else 0
        else:
            self.win, self.n_still = None, 0

    def step_seq(self, t, Trel, other_contact_n, finger_nonbar_n):
        """r3（判定語意同原設計）：相對線速度 ≤ rel_lin_mps 且角速度 ≤ rel_ang_rps **連續** rel_still_s。
        速度＝相鄰有效樣本的相對變換差分；連續時間以樣本時間戳累計。第一筆（無前樣本）、非有限／不合法位姿、時間戳不前進或間隔 >2.5·dt ⇒ 連續時間歸零。
        窗首尾差另算，只存診斷（往返振盪首尾回原位會騙過它，不能當判準）。"""
        if other_contact_n > self.c['forbidden_n']:
            self.forbid(t, 'nonfinger_contact', force_n=round(float(other_contact_n), 4))
        if finger_nonbar_n > self.c['forbidden_n']:
            self.forbid(t, 'finger_nonbar_contact', force_n=round(float(finger_nonbar_n), 4))
        ok_T = Trel is not None and np.asarray(Trel).shape == (4, 4) and bool(np.isfinite(Trel).all()) and math.isfinite(float(t))
        if not ok_T:
            self.n_invalid += 1
            self.prev, self.still_s, self.last, self.hist = None, 0.0, None, []
            return
        Trel = np.asarray(Trel, np.float64)
        if self.prev is None:
            self.prev, self.still_s, self.last = (float(t), Trel), 0.0, None
        else:
            dt = float(t) - self.prev[0]
            if not (1e-9 < dt <= 2.5 * self.dt):
                self.n_invalid += 1
                self.prev, self.still_s, self.last, self.hist = (float(t), Trel), 0.0, None, []
            else:
                lin, ang = step_motion(self.prev[1], Trel, dt)
                self.last = (lin, ang)
                self.still_s = self.still_s + dt if (lin <= self.c['rel_lin_mps'] and ang <= self.c['rel_ang_rps']) else 0.0
                self.prev = (float(t), Trel)
        self.hist.append(Trel)
        n = int(round(self.c['rel_still_s'] / self.dt))
        if len(self.hist) > n + 1:
            self.hist.pop(0)
        self.win = window_motion(self.hist[0], self.hist[-1], n * self.dt) if len(self.hist) == n + 1 else None
        self.n_still = int(round(self.still_s / self.dt))             # 讓 decide 沿用同一欄位

    def decide(self, pos_m, rot_deg, finger_q, finger_bar_n, door_pre_deg):
        c = self.c
        ok = dict(no_forbidden_contact=self.latched is None, pos=pos_m <= c['pos_m'], rot=rot_deg <= c['rot_deg'], rel_still=(self.still_s if self.prev is not None or self.still_s > 0 else self.n_still * self.dt) >= c['rel_still_s'] - 1e-9,   # step_seq 路徑直接比連續時間；舊 step 路徑才用拍數
                  finger_q=all(c['finger_q_m'][0] <= q <= c['finger_q_m'][1] for q in finger_q), finger_bar_contact=all(f > c['finger_contact_n'] for f in finger_bar_n), door_pre=door_pre_deg <= c['door_pre_deg'])
        return bool(all(ok.values())), ok
