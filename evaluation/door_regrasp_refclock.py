#!/usr/bin/env python3
"""換抓 mode 5 參考時鐘：預抓取節點閘控（Codex r62 §3；任務端 policy5 與離線閉迴路 wgmpc_v2 共用）。
mode 5＝A（接近預抓取）＋I（沿工具 +z 插入）。參考時間推進到預抓取節點 τ_pre（A 段最後一點）即**停住**，
直到實測 TCP 對預抓取位姿 ≤ pre_tol_m（既有條件：5 mm，同監看目標切換）才續推插入段；停住期間目標＝預抓取節點。
期限按**模式進入時間**計：進入後 τ_pre＋pre_wait_s（10 s，同 ENGAGE）仍未到達 ⇒ 'pregrasp_timeout'（呼叫端拒絕／中止）；
量測缺或非有限 ⇒ 視為未到達（不放行），仍受同一期限約束。到達閂鎖、不撤回。
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 'door_regrasp_refclock/r1'


def pregrasp_index(stages):
    """mode 5 段的每點階段（'A'／'I'）→ A 段最後一點索引。"""
    ia = [i for i, s in enumerate(stages) if s == 'A']
    if not ia or not any(s == 'I' for s in stages) or max(ia) + 1 >= len(stages) or stages[max(ia) + 1] != 'I':
        raise ValueError('mode 5 段階段標記不合（須 A…A I…I）')
    return max(ia)


class PregraspClock:
    def __init__(self, i_pre, dt, pre_tol_m=0.005, pre_wait_s=10.0):
        self.i_pre, self.dt = int(i_pre), float(dt)
        self.tau_pre = self.i_pre * self.dt
        self.tol, self.wait = float(pre_tol_m), float(pre_wait_s)
        self.t0 = None
        self.t_reached = None
        self.events = []

    def start(self, t):
        self.t0 = float(t)

    def step(self, t, p_tcp, p_pre):
        """回傳 (tau_ref, status)；status ∈ {'approach', 'hold_pre', 'insert', 'pregrasp_timeout'}。"""
        el = float(t) - self.t0
        if self.t_reached is not None:
            return self.tau_pre + (float(t) - self.t_reached), 'insert'
        if el < self.tau_pre - 1e-9:
            return el, 'approach'
        ok = False
        try:
            p, q = np.asarray(p_tcp, float).reshape(3), np.asarray(p_pre, float).reshape(3)
            ok = bool(np.all(np.isfinite(p))) and float(np.linalg.norm(p - q)) <= self.tol
        except (TypeError, ValueError):
            ok = False
        if ok:
            self.t_reached = float(t)
            self.events.append({'t': float(t), 'elapsed': el, 'held_s': el - self.tau_pre, 'what': 'pregrasp_reached'})
            return self.tau_pre, 'insert'
        if el > self.tau_pre + self.wait + 1e-9 or not math.isfinite(el):
            self.events.append({'t': float(t), 'elapsed': el, 'what': 'pregrasp_timeout'})
            return self.tau_pre, 'pregrasp_timeout'
        return self.tau_pre, 'hold_pre'
