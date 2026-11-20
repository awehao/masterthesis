#!/usr/bin/env python3
"""邊走邊併線（Howard 11/21「茶几跟把手要真的移動中抓」）：取代經過線的原地對齊（ALIGN＝停車轉向）。

自目前底盤位置 p0、行進方向 d0（速度太小時改朝併線點）出發，以三次 Hermite 曲線（端點切向＝d0 與經過線方向 e）
併入經過線 b0＋s·e 的 s_m 點，之後沿線。偏航自目前值以最小急動在弧長 σ_rot 內轉到線上偏航。
監督（送封包前核淨空）與控制器（參考軌跡）用同一份幾何。"""
import math

import numpy as np

S_MERGE_MIN = 0.25          # 併線段沿線最短推進（m）
K_LAT = 1.5                 # 併線段沿線推進 ≥ K_LAT × 側向偏移
WZ_REF = 0.6                # 偏航參考峰值角速度上限（rad/s；最小急動峰值 1.875×平均）


def merge_geom(p0, d0, b0, e, s_cap):
    """回 dict(P0,T0,P1,T1,s_m,Lc,lat,s_p, tab=(u 取樣, 累積弧長))；s_cap＝併線點沿線上限（要在伸手前併完）。"""
    p0 = np.asarray(p0, float)[:2]; b0 = np.asarray(b0, float); e = np.asarray(e, float)[:2]
    n = np.array([-e[1], e[0]])
    d = p0 - b0[:2]; s_p = float(d @ e); lat = float(d @ n)
    s_m = min(s_p + max(S_MERGE_MIN, K_LAT * abs(lat)), float(s_cap))
    s_m = max(s_m, s_p + 0.10)
    P1 = b0[:2] + s_m * e
    L = float(np.linalg.norm(P1 - p0))
    d0 = np.asarray(d0, float)[:2]
    d0 = d0 / float(np.linalg.norm(d0)) if float(np.linalg.norm(d0)) > 1e-6 else (P1 - p0) / max(L, 1e-9)
    T0, T1 = d0 * L, e * L
    u = np.linspace(0.0, 1.0, 61)
    P = _herm(p0, T0, P1, T1, u)
    cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    return dict(P0=p0, T0=T0, P1=P1, T1=T1, s_m=float(s_m), Lc=float(cum[-1]), lat=lat, s_p=s_p, u=u, cum=cum, b0=b0, e=e)


def _herm(P0, T0, P1, T1, u):
    u = np.asarray(u, float)[:, None]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u; h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
    return h00 * P0 + h10 * T0 + h01 * P1 + h11 * T1


def at(G, sig):
    """複合路徑（併線曲線＋沿線）弧長 sig 的位置。"""
    sig = float(sig)
    if sig <= G['Lc']:
        u = float(np.interp(max(sig, 0.0), G['cum'], G['u']))
        return _herm(G['P0'], G['T0'], G['P1'], G['T1'], [u])[0]
    return G['b0'][:2] + (G['s_m'] + sig - G['Lc']) * G['e']


def s_line(G, sig):
    """弧長 → 經過線座標 s（曲線段回負的近似值，只供記錄）。"""
    return G['s_m'] + float(sig) - G['Lc']


def sig_of_s(G, s):
    return G['Lc'] + float(s) - G['s_m']


def yaw_at(G, sig, y0, th_l, sig_rot):
    from math import pi
    dy = (float(th_l) - float(y0) + pi) % (2 * pi) - pi
    u = min(max(float(sig) / max(float(sig_rot), 1e-6), 0.0), 1.0)
    mj = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
    return float(y0) + dy * mj


def sig_rot_for(G, y0, th_l, s_r, v):
    """偏航在線上 s_r 前轉完，且峰值角速度 ≤ WZ_REF（以沿路徑速度 v 換算）。"""
    dy = abs((float(th_l) - float(y0) + math.pi) % (2 * math.pi) - math.pi)
    sr = G['Lc'] + max(0.0, float(s_r) - G['s_m'])
    return max(sr, 1.875 * dy * float(v) / WZ_REF, 0.05)


def time_law(G, v0, v_g, s_slow, v_hi_cap=0.2, a=0.15, ds=0.005):
    """弧長—時間表：併線段以 v_hi＝clip(v0, v_g, v_hi_cap) 前進，於線上 s_slow（伸手起點）前以減速度 a 降到 v_g，之後等速 v_g；
    起始速度 v0 以同一加減速度接到 v_hi。回 (σ 格點, t 格點)；超出末端以 v_g 外插。"""
    v_g = float(v_g); v_hi = min(max(float(v0), v_g, 0.15), float(v_hi_cap))
    sig_s = max(0.0, sig_of_s(G, s_slow))
    L = sig_s + 2.0
    sg = np.arange(0.0, L + 1e-9, ds)
    v_dec = np.sqrt(np.maximum(v_g ** 2 + 2 * a * np.maximum(sig_s - sg, 0.0), 0.0))   # 到 sig_s 時恰降到 v_g
    v_acc = np.sqrt(np.maximum(float(v0) ** 2 + 2 * a * sg, 0.0)) if float(v0) < v_hi else np.full_like(sg, np.inf)
    v_dn0 = np.sqrt(np.maximum(float(v0) ** 2 - 2 * a * sg, 0.0)) if float(v0) > v_hi else np.full_like(sg, np.inf)
    v = np.minimum.reduce([np.full_like(sg, v_hi), v_dec, v_acc, np.maximum(v_dn0, v_hi)])
    v = np.where(sg >= sig_s, v_g, v)
    v = np.maximum(v, 0.02)
    t = np.r_[0.0, np.cumsum(ds / (0.5 * (v[1:] + v[:-1])))]
    return sg, t


def sig_at_t(law, tau, v_g):
    sg, t = law
    if tau <= t[-1]:
        return float(np.interp(max(tau, 0.0), t, sg))
    return float(sg[-1] + (tau - t[-1]) * float(v_g))


def t_at_sig(law, sig, v_g):
    sg, t = law
    if sig <= sg[-1]:
        return float(np.interp(max(sig, 0.0), sg, t))
    return float(t[-1] + (sig - sg[-1]) / float(v_g))


def poly(G, s_end, ds=0.05):
    """核淨空用折線：曲線段＋沿線到 s_end。"""
    sig_end = sig_of_s(G, s_end)
    return np.array([at(G, x) for x in np.arange(0.0, sig_end + 1e-9, ds)])
