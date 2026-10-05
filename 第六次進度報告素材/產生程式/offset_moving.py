#!/usr/bin/env python3
"""v2.1：手臂偏移估計在**持續等速**運動中也能取得觀測（純邏輯；求解節點 --offset-moving 時呼叫）。

依 Codex reviews/20261005_125840_reply.md。手臂模型（每物理步）：

    q⁺ = q + α(s − q) + α d ，  s⁺ = s + dt_p · ṡ

等速穩態下 q − s = d − (dt_p/α) · ṡ ⇒ 逐軸觀測

    d_obs = (q − s) + (dt_p/α) · ṡ          （α≈0.095、dt_p 0.01 s ⇒ 係數約 0.105 s）

ṡ 由**同步量測**的設定點與其**實際時間差**計算（不用求解請求代替）。只有窗口內（window_s）各軸 ṡ 與窗口均值之差
都 ≤ accel_tol（持續等速、追蹤暫態已消退）才給觀測；缺資料、時間不前進、非有限值 ⇒ 不給觀測。

事前選定（v2.1 規格）：window_s = 0.30 s ≈ 3 × 0.105 s；accel_tol = 0.01 rad/s（ṡ 偏差造成的補償誤差 ≈ 0.01 × 0.105 ≈ 1 mrad）；
min_samples = 5（20 Hz 下 0.30 s 窗內約 7 個樣本）。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class MovingCfg:
    window_s: float = 0.30
    accel_tol: float = 0.01
    min_samples: int = 5


class MovingOffsetObserver:
    def __init__(self, cfg: MovingCfg, alpha, dt_p: float):
        self.cfg = cfg
        self.alpha = np.asarray(alpha, float).reshape(6)
        self.dt_p = float(dt_p)
        if not (np.all(self.alpha > 0) and np.all(self.alpha < 1) and self.dt_p > 0):
            raise ValueError('alpha 須在 (0, 1)、dt_p 須為正')
        self.k = self.dt_p / self.alpha            # 逐軸補償係數（秒）
        self.hist = deque()                         # (t, s, ṡ 或 None)

    def observe(self, t, s, q):
        """回傳 dict：ok（是否給觀測）、why、sdot、window_ok、d_obs、n_window、span_s。"""
        out = {'ok': False, 'why': None, 'sdot': None, 'window_ok': False, 'd_obs': None,
               'n_window': 0, 'span_s': 0.0}
        try:
            t = float(t)
            s = np.asarray(s, float).reshape(6)
            q = np.asarray(q, float).reshape(6)
        except (TypeError, ValueError):
            out['why'] = 'bad_shape'
            return out
        if not (math.isfinite(t) and np.isfinite(s).all() and np.isfinite(q).all()):
            out['why'] = 'nonfinite'
            return out
        if self.hist and t <= self.hist[-1][0]:
            out['why'] = 'time_not_increasing'
            return out
        sdot = None
        if self.hist:
            t0, s0, _ = self.hist[-1]
            sdot = (s - s0) / (t - t0)
        self.hist.append((t, s.copy(), sdot))
        while self.hist and self.hist[0][0] < t - self.cfg.window_s - 1e-9:
            self.hist.popleft()
        sd = [h[2] for h in self.hist if h[2] is not None]
        out['n_window'] = len(sd)
        out['span_s'] = t - self.hist[0][0]
        if sdot is None:
            out['why'] = 'no_previous_sample'
            return out
        out['sdot'] = [float(x) for x in sdot]
        if len(sd) < self.cfg.min_samples or out['span_s'] < self.cfg.window_s - 0.06:
            out['why'] = 'window_not_full'
            return out
        S = np.array(sd)
        dev = np.abs(S - S.mean(axis=0)).max()
        if dev > self.cfg.accel_tol:
            out['why'] = f'not_constant_velocity（ṡ 偏差 {dev:.4f} > {self.cfg.accel_tol}）'
            return out
        out['window_ok'] = True
        out['d_obs'] = [float(x) for x in (q - s) + self.k * sdot]
        out['ok'] = True
        return out
