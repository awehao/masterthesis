#!/usr/bin/env python3
"""抽屜把手（橫桿）深度偵測（Howard 11/19「不要把東西寫死 要實際用到視覺」；幾何法、不讀真值、不學習）。
輸入：手腕相機深度（到影像平面距離）、內參 K、相機光學座標世界位姿 T_cam、先驗抽屜位姿 prior＝(x, y, yaw_rad)（場景登錄，只用來劃搜尋範圍）、資產 spec（形狀模型）。
座標：先驗抽屜本地（同資產：+x 沿橫桿、+y 指向櫃內、抽屜沿 −y 拉出、原點櫃底中心）。
步驟：
  1. 每 STRIDE 像素反投影 → 留本地搜尋範圍（|x|≤X_HALF、y∈Y_RANGE、z∈Z_RANGE）的點。
  2. 面板：面板高度帶（spec 面板 z 範圍內縮 2 cm）的點，y 直方圖（5 mm）取最多點的面 ⇒ 該面 ±8 mm 內點做 y＝a·x＋b 擬合 ⇒ 面板前緣 y_p（x＝0）、偏航修正 dyaw＝atan(a)。
  3. 橫桿：在面板前方 [BAR_GAP_MIN, BAR_GAP_MAX]、高度 z∈ 桿心 ±BAR_DZ 的點（先以 dyaw 轉正）；點數 ≥ MIN_PTS；
     沿 x 的範圍＝可見長度（須 ≥ BAR_MIN_FRAC×資產桿長，避免只看到一段）⇒ 桿心 x＝範圍中點；桿心 z＝點 z 中位數；桿心 y＝前表面（y 第 5 百分位）＋桿半徑。
  4. 回 抽屜原點修正（本地 dx＝桿心 x − 資產桿心 x、dy＝桿心 y − 資產桿心 y）、dyaw、桿心世界座標、info。
不宣稱：單一形狀模型（已知家具型號）；遮擋只靠可見長度門檻排除；誤差以替身合成觀測與 Isaac 真值事後核對。"""
import math

import numpy as np

STRIDE, MIN_PTS = 2, 25
X_HALF, Y_RANGE, Z_RANGE = 0.45, (-0.40, -0.02), (0.30, 0.75)
BAR_GAP_MIN, BAR_GAP_MAX, BAR_DZ = 0.012, 0.080, 0.035
BAR_MIN_FRAC = 0.6
PANEL_BIN = 0.005


def detect(dep, K, T_cam, prior, spec, stride=STRIDE):
    px, py, yaw = (float(v) for v in prior)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    bar = spec['drawer']['handle']['bar']
    bc = np.asarray(bar['center'], float); br = float(bar['radius']); bl = float(bar['length'])
    pan = [b for b in spec['drawer']['body'] if b[0] == 'front_panel'][0]
    pz0, pz1 = float(pan[1][2]) - float(pan[2][2]) / 2 + 0.02, float(pan[1][2]) + float(pan[2][2]) / 2 - 0.02
    d = np.asarray(dep, float)
    H, W = d.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = d[0:H:stride, 0:W:stride]
    ok = np.isfinite(z) & (z > 0.05) & (z < 3.5)
    if not ok.any():
        return None, dict(why='no_depth')
    u, v, z = u[ok].astype(float), v[ok].astype(float), z[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Pw = (np.asarray(T_cam, float) @ np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)]))[:3].T
    L = (Pw - np.array([px, py, 0.0])) @ Rz                          # 先驗本地
    m = (np.abs(L[:, 0]) <= X_HALF) & (L[:, 1] >= Y_RANGE[0]) & (L[:, 1] <= Y_RANGE[1]) & (L[:, 2] >= Z_RANGE[0]) & (L[:, 2] <= Z_RANGE[1])
    L = L[m]
    if len(L) < MIN_PTS:
        return None, dict(why='few_points', n=int(len(L)))
    P = L[(L[:, 2] >= pz0) & (L[:, 2] <= pz1)]                         # 面板高度帶（含橫桿遮擋處，取最多點的面）
    if len(P) < 4 * MIN_PTS:
        return None, dict(why='few_panel_points', n=int(len(P)))
    hist, edges = np.histogram(P[:, 1], bins=np.arange(Y_RANGE[0], Y_RANGE[1] + PANEL_BIN, PANEL_BIN))
    k = int(np.argmax(hist))
    y_mode = 0.5 * (edges[k] + edges[k + 1])
    face = P[np.abs(P[:, 1] - y_mode) <= 0.008]
    if len(face) < 4 * MIN_PTS or float(np.ptp(face[:, 0])) < 0.15:
        return None, dict(why='panel_not_seen', n=int(len(face)))
    A = np.column_stack([face[:, 0], np.ones(len(face))])
    a, b = np.linalg.lstsq(A, face[:, 1], rcond=None)[0]
    dyaw = math.atan(float(a))
    cr, sr = math.cos(-dyaw), math.sin(-dyaw)                         # 轉正（本地 xy 繞 z 轉 −dyaw）
    Q = L.copy()
    Q[:, 0], Q[:, 1] = cr * L[:, 0] - sr * L[:, 1], sr * L[:, 0] + cr * L[:, 1]
    y_p = float(b) * math.cos(dyaw)                                   # 轉正後面板前緣（近似）
    gap = y_p - Q[:, 1]
    bm = (gap >= BAR_GAP_MIN) & (gap <= BAR_GAP_MAX) & (np.abs(Q[:, 2] - bc[2]) <= BAR_DZ)
    B = Q[bm]
    if len(B) < MIN_PTS:
        return None, dict(why='bar_few_points', n=int(len(B)), y_panel=round(y_p, 4))
    xs = np.sort(B[:, 0])
    x0, x1 = float(np.percentile(xs, 1)), float(np.percentile(xs, 99))
    if x1 - x0 < BAR_MIN_FRAC * bl or x1 - x0 > bl + 0.06:
        return None, dict(why='bar_length', seen=round(x1 - x0, 4), n=int(len(B)))
    xb, zb = 0.5 * (x0 + x1), float(np.median(B[:, 2]))
    yb = float(np.percentile(B[:, 1], 5)) + br
    dx, dy = xb - float(bc[0]), yb - float(bc[1])
    c2, s2 = math.cos(yaw + dyaw), math.sin(yaw + dyaw)
    origin = np.array([px, py]) + np.array([[c2, -s2], [s2, c2]]) @ np.array([dx, dy])
    bar_w = np.array([px, py, 0.0]) + Rz @ np.array([cr * 0 + xb * math.cos(dyaw) - yb * math.sin(dyaw), xb * math.sin(dyaw) + yb * math.cos(dyaw), zb])
    return dict(origin=origin, yaw=yaw + dyaw, dx=dx, dy=dy, dyaw=dyaw, bar_world=bar_w, z_bar=zb), dict(n=int(len(L)), n_face=int(len(face)), n_bar=int(len(B)), seen_len=round(x1 - x0, 4), y_panel=round(y_p, 4))
