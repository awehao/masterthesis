#!/usr/bin/env python3
"""門扇開度深度偵測（Howard 11/19「要實際用到視覺」＋回程關門：放手後門會漂，回程時門角不能假設）。幾何法、不讀真值。
門局部座標（door_local_v3）：鉸鏈軸在原點、+z；門角 φ≥0 時門扇沿 u＝(cos(−φ), sin(−φ)) 自鉸鏈伸出（往臥室 y<0 側開）；
門扇長 0.95、厚 0.04（往 −n 側，即厚度在門扇方向的左／右一側，對方向估計無偏：本法用點群主方向，不用點的極角）。
步驟：每 STRIDE 像素反投影（世界＝門局部；呼叫端給 T_cam 於同座標）→ 留 鉸鏈半徑 R_RANGE、高度 Z_RANGE、極角 [−φ_max−5°, −φ_min+5°] 的點；
  極角直方圖（1°）取最多點的角 θ0 ⇒ θ0±4° 內的點做 xy 主成分（過中心的直線）⇒ 方向角 ⇒ φ＝−方向角（取與 θ0 同側）；
  另核：點沿直線範圍 ≥ MIN_SPAN（看到夠長的門扇）、直線到鉸鏈距離 ≤ 0.05 m（門扇過鉸鏈附近）。
回 (φ 或 None, info)。不宣稱：牆、家具在同極角帶會干擾（範圍限在門扇長度內並核直線過鉸鏈）。"""
import math

import numpy as np

STRIDE, MIN_PTS = 3, 60
R_RANGE, Z_RANGE = (0.20, 0.92), (0.30, 1.80)
MIN_SPAN, HINGE_TOL = 0.30, 0.05


def detect(dep, K, T_cam, phi_min=math.radians(10.0), phi_max=math.radians(108.0), stride=STRIDE):
    d = np.asarray(dep, float)
    H, W = d.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = d[0:H:stride, 0:W:stride]
    ok = np.isfinite(z) & (z > 0.05) & (z < 4.0)
    if not ok.any():
        return None, dict(why='no_depth')
    u, v, z = u[ok].astype(float), v[ok].astype(float), z[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    P = (np.asarray(T_cam, float) @ np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)]))[:3].T
    r = np.hypot(P[:, 0], P[:, 1]); th = np.arctan2(P[:, 1], P[:, 0])
    lo, hi = -phi_max - math.radians(5.0), -phi_min + math.radians(5.0)
    m = (r >= R_RANGE[0]) & (r <= R_RANGE[1]) & (P[:, 2] >= Z_RANGE[0]) & (P[:, 2] <= Z_RANGE[1]) & (th >= lo) & (th <= hi)
    Q = P[m]
    if len(Q) < MIN_PTS:
        return None, dict(why='few_points', n=int(len(Q)))
    tq = np.arctan2(Q[:, 1], Q[:, 0])
    hist, edges = np.histogram(tq, bins=np.arange(lo, hi + math.radians(1.0), math.radians(1.0)))
    k = int(np.argmax(hist)); th0 = 0.5 * (edges[k] + edges[k + 1])
    S = Q[np.abs(tq - th0) <= math.radians(4.0)]
    if len(S) < MIN_PTS:
        return None, dict(why='few_leaf_points', n=int(len(S)))
    c = S[:, :2].mean(0)
    w, V = np.linalg.eigh(np.cov((S[:, :2] - c).T))
    a = V[:, 1]
    if a @ np.array([math.cos(th0), math.sin(th0)]) < 0:
        a = -a
    sp = (S[:, :2] - c) @ a
    span = float(sp.max() - sp.min())
    d_h = abs(float(c[0] * a[1] - c[1] * a[0]))                      # 直線到鉸鏈（原點）距離
    if span < MIN_SPAN or d_h > HINGE_TOL:
        return None, dict(why='leaf_shape', span=round(span, 3), hinge_dist=round(d_h, 3), n=int(len(S)))
    phi = -math.atan2(a[1], a[0])
    return phi, dict(n=int(len(S)), span=round(span, 3), hinge_dist=round(d_h, 4), theta0_deg=round(math.degrees(th0), 2))
