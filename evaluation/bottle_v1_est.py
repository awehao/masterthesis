#!/usr/bin/env python3
"""視覺 V1 瓶子估計器（旁路，不進控制；設計：第七次進度報告素材/規格/瓶子第二桌放置_最小設計.md「V1 視覺旁路設計」）。
輸入：深度圖（光軸 Z，公尺，無效＝0 或非有限）、內參 K、擷取時刻相機光學座標的世界位姿 T_wc（z 前、x 右、y 下）。不吃任何真值。
步驟：反投影 → 取已知桌子（地圖先驗：桌面高度與 xy 範圍；不是瓶子真值）範圍內、桌面上方 0.02–0.25 m 的點，以 1 cm 格分群 → 挑符合已知瓶型的群
（xy 範圍 ≤ 0.06 m、高 ≥ 0.06 m、點數 ≥ 15）→ 固定半徑 15 mm 圓擬合得瓶軸 xy → PCA 估軸方向。
draft-1 原用 RANSAC 找桌面；底盤相機離地約 0.36 m，低於桌面 0.6 m，看不到桌面，合成核對時把桌板正面、方塊側面當成平面（z 誤差 30–195 mm），所以改用地圖先驗。
輸出瓶中心位置與軸方向；繞軸朝向不估（圓柱對稱）。瓶底 z 兩種推法都標明：z_from_support＝桌面高度（支撐假設）、z_from_top＝觀測瓶頂−瓶高（瓶型假設）。
"""
import math

import numpy as np

R_BOTTLE, H_BOTTLE = 0.015, 0.20
BAND = (0.40, 1.00)
ABOVE = (0.02, 0.25)
MIN_PLANE, MIN_CLUSTER = 200, 15


def backproject(depth, K):
    """光軸 Z 深度 → 相機座標點（N×3）與其像素。"""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(float)
    ok = np.isfinite(z) & (z > 0)
    x = (u[ok] - K[0, 2]) * z[ok] / K[0, 0]
    y = (v[ok] - K[1, 2]) * z[ok] / K[1, 1]
    return np.column_stack([x, y, z[ok]]), np.column_stack([u[ok], v[ok]])


def circle_fixed_r(P2, r, c0):
    c = np.asarray(c0, float).copy()
    for _ in range(30):
        d = P2 - c
        n = np.linalg.norm(d, axis=1)
        n = np.maximum(n, 1e-9)
        res = n - r
        J = -d / n[:, None]
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        c += step
        if np.linalg.norm(step) < 1e-7:
            break
    return c, float(np.sqrt(np.mean((np.linalg.norm(P2 - c, axis=1) - r) ** 2)))


def estimate(depth, K, T_wc, table):
    """table：地圖先驗 dict(top_z, lo=(x, y), hi=(x, y))。"""
    out = dict(valid=False, why=None)
    Pc, _ = backproject(depth, K)
    if len(Pc) == 0:
        return dict(out, why='no_depth')
    Pw = (T_wc[:3, :3] @ Pc.T).T + T_wc[:3, 3]
    zt = float(table['top_z'])
    lo, hi = np.asarray(table['lo'], float), np.asarray(table['hi'], float)
    cand = Pw[(Pw[:, 2] >= zt + ABOVE[0]) & (Pw[:, 2] <= zt + ABOVE[1]) & (Pw[:, 0] >= lo[0]) & (Pw[:, 0] <= hi[0]) & (Pw[:, 1] >= lo[1]) & (Pw[:, 1] <= hi[1])]
    out.update(plane_z=zt, plane_source='map_prior', n_cand=int(len(cand)))
    if len(cand) < MIN_CLUSTER:
        return dict(out, why='no_object_points')
    cells = np.floor(cand[:, :2] / 0.01).astype(int)
    keys = {tuple(c_): i for i, c_ in enumerate(np.unique(cells, axis=0))}
    lab = np.array([keys[tuple(c_)] for c_ in cells])
    parent = list(range(len(keys)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for (cx, cy), i in keys.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                j = keys.get((cx + dx, cy + dy))
                if j is not None:
                    parent[find(i)] = find(j)
    root = np.array([find(i) for i in lab])
    best = None
    for rt in np.unique(root):
        C = cand[root == rt]
        ext = C[:, :2].max(axis=0) - C[:, :2].min(axis=0)
        hgt = float(C[:, 2].max() - C[:, 2].min())
        if len(C) >= MIN_CLUSTER and ext.max() <= 0.06 and hgt >= 0.06:
            if best is None or len(C) > len(best):
                best = C
    if best is None:
        return dict(out, why='no_bottle_cluster')
    cam = T_wc[:2, 3]
    m = best[:, :2].mean(axis=0)
    away = (m - cam) / max(np.linalg.norm(m - cam), 1e-9)
    c, rms = circle_fixed_r(best[:, :2], R_BOTTLE, m + away * R_BOTTLE)
    X = best - best.mean(axis=0)
    w, V = np.linalg.eigh(X.T @ X)
    ax = V[:, -1] * (1.0 if V[2, -1] >= 0 else -1.0)
    return dict(out, valid=True, why='ok', center_xy=c.tolist(), z_from_support=zt, z_from_top=float(best[:, 2].max()) - H_BOTTLE, axis=ax.tolist(), axis_estimated=True,
                axis_eig_ratio=float(w[-1] / max(w[-2], 1e-12)), n_pts=int(len(best)), circle_rms=rms, top_z=float(best[:, 2].max()))
