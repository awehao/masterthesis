#!/usr/bin/env python3
"""行人（動態障礙物）運動模型：Isaac 執行器與替身共用同一份（場景障礙檔 walkers：p0、p1、speed_mps、dwell_s、phase_s、radius、height）。
來回直線（p0→p1→p0…），端點停留 dwell_s；phase_s＝起始相位（秒）。不看機器人、不避讓（相遇由時間自然發生；Howard 場景設計原則）。"""
import math

import numpy as np


def walker_pos(w, t):
    p0, p1 = np.asarray(w['p0'], float), np.asarray(w['p1'], float)
    L = float(np.linalg.norm(p1 - p0)); v = float(w['speed_mps']); dw = float(w.get('dwell_s', 0.0))
    if L < 1e-9 or v <= 0:
        return p0.copy()
    T_leg = L / v
    T = 2 * (T_leg + dw)
    s = (float(t) + float(w.get('phase_s', 0.0))) % T
    if s < T_leg:
        return p0 + (p1 - p0) * (s / T_leg)
    s -= T_leg
    if s < dw:
        return p1.copy()
    s -= dw
    if s < T_leg:
        return p1 + (p0 - p1) * (s / T_leg)
    return p0.copy()


YIELD = __import__('os').environ.get('D2_WALKER_YIELD', '0') == '1'   # Howard 場景＋第三十趟 Isaac（行人撞上近乎靜止的機器人）：行人讓路——前方 YIELD_GAP 內有機器人且正朝其走 ⇒ 暫停
YIELD_GAP, ROBOT_R = 0.30, 0.30
_TAU, _T_LAST, _PAUSED = {}, [None], {}


def crowd_step(t, robot_xy, walkers):
    """每個物理步呼叫一次：各行人自己的進度時間 τ；讓路時 τ 不前進（同一份給 Isaac 與替身；決定性）。未開 YIELD ⇒ τ＝t（與原模型逐值相同）。"""
    dt = 0.0 if _T_LAST[0] is None else max(0.0, float(t) - _T_LAST[0])
    _T_LAST[0] = float(t)
    for w in walkers:
        nm = w['name']
        tau = _TAU.get(nm, float(t) - dt)
        if not YIELD:
            _TAU[nm] = float(t); continue
        p = walker_pos(w, tau); v = walker_vel(w, tau)
        d = np.asarray(robot_xy, float)[:2] - p
        block = float(np.linalg.norm(d)) < float(w['radius']) + ROBOT_R + YIELD_GAP and float(v @ d) > 0.0
        _PAUSED[nm] = bool(block)
        _TAU[nm] = tau if block else tau + dt


def crowd_pos(w, t=None):
    tau = _TAU.get(w['name'])
    return walker_pos(w, t if tau is None else tau)


def walker_vel(w, t, dt=0.05):
    return (walker_pos(w, t + dt) - walker_pos(w, t - dt)) / (2 * dt)


def cast_circles(org, angles, ranges, circles):
    """2D 射線對圓（行人）：org(2)、angles(N)、ranges(N) 原距離；回新距離（取近者）。"""
    r = np.asarray(ranges, float).copy()
    for c, rad in circles:
        oc = np.asarray(org, float) - np.asarray(c, float)
        for i, th in enumerate(angles):
            d = np.array([math.cos(th), math.sin(th)])
            b = float(oc @ d); cc = float(oc @ oc) - rad * rad
            disc = b * b - cc
            if disc < 0:
                continue
            tt = -b - math.sqrt(disc)
            if 0 < tt < r[i]:
                r[i] = tt
    return r
