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
