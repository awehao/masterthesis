#!/usr/bin/env python3
"""連續交接的接近段幾何（Howard／Codex 20261103_133750；純函式，假世界與實跑共用）。
第一瓶放好後，真規劃器從停車位規劃到 G（第二瓶西側的轉向點），再接一段相切 Bézier 到既有直線段入口 (x_lane, 0)，切線 +x。整條是一個 PlannedPath：σ＝x_lane − 剩餘弧長，到直線段後 σ＝x，所以抓取段完全沿用既有流程。
為什麼有 G：移動抓取沿桌邊朝 +x 走，底盤要繞到瓶子西側才能轉向；這是抓取方向的需要，不是回原點。轉向是半徑 |y_G|／2 的圓弧（第一版的 Bézier 彎半徑只有 5 cm 又往西鼓，假世界 Shield 介入）。G 由幾何決定：圓弧最西點離障礙方塊角點（−1.8, −0.2）留足靜態屏障 0.52 m 以上，且能留足手腕視角序列的弧長。沒有宣稱這是全域最短。
SIG_HANDOVER：換目標的路徑點，比手腕視角序列起點 X_U0 再早 0.10 m（序列沿用既有時長與觸發值，只是改以 σ 判斷）。
"""
import math

import numpy as np

import bottle_n2_path as NP

G_TRANSIT = (-0.95, -0.45)
X_LANE = -0.9
X_U0 = -2.654
SIG_HANDOVER = X_U0 - 0.10
L_RAMP = 1.4                                            # 偏航轉回 0 的弧長（m）：換目標時底盤朝向要符合抓取段
SMOOTH_ITERS = 100


def _smooth(P, iters):
    P = np.array(P, float)
    for _ in range(int(iters)):
        P[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
    return P


def hook(p0, x_lane, ds=NP.DS):
    """從 p0（朝西）順時針繞半圓到 (p0x, 0)，再沿 y＝0 朝東到 x_lane。半徑＝|p0y|／2。"""
    x0, y0 = float(p0[0]), float(p0[1])
    r = abs(y0) / 2.0
    th = np.linspace(-0.5 * math.pi, -1.5 * math.pi, max(int(math.pi * r / ds), 16))
    arc = np.column_stack([x0 + r * np.cos(th), 0.5 * y0 - r * np.sin(th) * (-1.0 if y0 < 0 else 1.0)])
    arc[0], arc[-1] = (x0, y0), (x0, 0.0)
    if float(x_lane) > x0 + 1e-6:
        return np.vstack([arc, [[float(x_lane), 0.0]]])
    return arc


def build_transit(raw, x_lane=X_LANE, smooth_iters=SMOOTH_ITERS, smooth=None):
    """規劃器原始路徑 → 平滑 → 接 Bézier 到 (x_lane, 0)。回傳 N×2（等距重取樣）。smooth 可傳入與控制器相同的平滑函式。"""
    P0 = NP.resample(raw)
    P = smooth(P0, smooth_iters) if smooth is not None else _smooth(P0, smooth_iters)
    B = hook(P[-1], float(x_lane))
    J = NP.resample(np.vstack([P[:-1], B]))
    return J


def handover_arc(L, x_lane=X_LANE, sig_h=SIG_HANDOVER):
    """換目標點的弧長（從路徑起點量）。路徑太短回 None（助跑不足，不硬換）。"""
    s = float(L) - (float(x_lane) - float(sig_h))
    return s if s > 0.0 else None


def heading_ramp(rp, s_h, L_ramp=L_RAMP):
    """把 ReturnPath 的切線角在 [s_h−L_ramp, s_h] 平滑帶到最近的 2πk（世界偏航 0），s_h 之後維持。回傳 (目標角, 起點角)。"""
    sm, h = rp.h_s, rp.h.copy()
    s0 = max(float(s_h) - float(L_ramp), float(sm[0]))
    h0 = float(np.interp(s0, sm, h))
    tgt = 2.0 * math.pi * round(h0 / (2.0 * math.pi))
    b = np.clip((sm - s0) / max(float(s_h) - s0, 1e-6), 0.0, 1.0)
    b = b * b * b * (10.0 - 15.0 * b + 6.0 * b * b)
    rp.h = (1.0 - b) * h + b * tgt
    rp.dh = np.gradient(rp.h, sm)
    return tgt, h0


def curvature(P):
    """逐點曲率（1/m），等距點。"""
    P = np.asarray(P, float)
    T = np.diff(P, axis=0)
    a = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
    ds = np.linalg.norm(T, axis=1)
    return np.abs(np.diff(a)) / np.maximum(0.5 * (ds[1:] + ds[:-1]), 1e-9)
