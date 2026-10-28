#!/usr/bin/env python3
"""mission_room 行人（運動學）位置：沿 p0↔p1 直線來回，端點停 dwell_s；t＝模擬時間＋phase_s。預覽與執行端共用（純函式）。"""
import math


def position(w, t):
    (x0, y0), (x1, y1) = w['p0'], w['p1']
    L = math.hypot(x1 - x0, y1 - y0)
    T = L / w['speed_mps']
    period = 2.0 * (T + w['dwell_s'])
    s = (t + w['phase_s']) % period
    if s < T:
        f = s / T
    elif s < T + w['dwell_s']:
        f = 1.0
    elif s < 2.0 * T + w['dwell_s']:
        f = 1.0 - (s - T - w['dwell_s']) / T
    else:
        f = 0.0
    return x0 + f * (x1 - x0), y0 + f * (y1 - y0)
