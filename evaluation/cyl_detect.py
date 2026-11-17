#!/usr/bin/env python3
"""茶几圓柱體深度偵測（幾何法；不用學習模型、不讀真值）。
輸入：手腕相機深度（到影像平面距離，m）、內參 K、相機光學座標的世界位姿 T_cam（z 前、x 右、y 下）、茶几幾何（靜態地圖登錄的已知家具）。
步驟：每 STRIDE 像素反投影到世界 → 留桌面投影內（外擴 2 cm）、高度在桌面上方 [Z_LO, Z_HI] 的點 → 點數 ≥ MIN_PTS → 水平質心 →
  可見半圓面質心偏向相機約 (2/π)·r：沿相機到質心的水平方向補 0.64·r 回軸心；高度＝桌面＋已知高度一半（已知類別：瓶子代理物 Ø30×200 mm）。
回 (p_world(3) 或 None, info)。"""
import math

import numpy as np

STRIDE, MIN_PTS = 3, 30
Z_LO, Z_HI = 0.015, 0.25
CYL_R, CYL_H = 0.015, 0.20


def detect(dep, K, T_cam, table, stride=STRIDE):
    """table＝dict(c=[x,y,z], s=[sx,sy,sz])（FixedCuboid：中心與尺寸；桌面頂＝sz）。"""
    d = np.asarray(dep, float)
    H, W = d.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = d[0:H:stride, 0:W:stride]
    ok = np.isfinite(z) & (z > 0.05) & (z < 4.0)
    if not ok.any():
        return None, dict(why='no_depth')
    u, v, z = u[ok].astype(float), v[ok].astype(float), z[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Pc = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)])
    Pw = (np.asarray(T_cam, float) @ Pc)[:3].T
    tc, ts = np.asarray(table['c'], float), np.asarray(table['s'], float)
    top = float(ts[2])
    m = (np.abs(Pw[:, 0] - tc[0]) <= ts[0] / 2 + 0.02) & (np.abs(Pw[:, 1] - tc[1]) <= ts[1] / 2 + 0.02) & (Pw[:, 2] >= top + Z_LO) & (Pw[:, 2] <= top + Z_HI)
    n = int(m.sum())
    if n < MIN_PTS:
        return None, dict(why='few_points', n=n)
    P = Pw[m]
    c = P[:, :2].mean(0)
    if float(np.linalg.norm(P[:, :2] - c, axis=1).max()) > 0.05:      # 點群水平擴散過大 ⇒ 不是單一細圓柱（或多物件）
        return None, dict(why='spread', n=n, spread_m=round(float(np.linalg.norm(P[:, :2] - c, axis=1).max()), 4))
    vdir = c - np.asarray(T_cam, float)[:2, 3]
    vdir = vdir / max(float(np.linalg.norm(vdir)), 1e-9)
    xy = c + (2.0 / math.pi) * CYL_R * vdir                          # 初值
    Q = P[:, :2]
    for _ in range(8):                                                # 已知半徑圓擬合（Gauss–Newton）：min Σ(|q−c|−r)²
        dq = Q - xy; rr = np.linalg.norm(dq, axis=1); rr = np.maximum(rr, 1e-9)
        res = rr - CYL_R; J = -dq / rr[:, None]
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        xy = xy + step
        if float(np.linalg.norm(step)) < 1e-6:
            break
    rms = float(np.sqrt(np.mean((np.linalg.norm(Q - xy, axis=1) - CYL_R) ** 2)))
    return np.array([xy[0], xy[1], top + CYL_H / 2]), dict(n=n, z_top=round(float(P[:, 2].max()), 4), z_low=round(float(P[:, 2].min()), 4), fit_rms_mm=round(rms * 1e3, 3))


if __name__ == '__main__':                                            # 合成深度圖自測：相機從桌邊斜看一根圓柱
    K = np.array([[465.6, 0, 320], [0, 465.6, 240], [0, 0, 1.0]])
    table = dict(c=[1.35, 2.3, 0.225], s=[0.6, 1.1, 0.45])
    cyl = np.array([1.30, 2.10])
    cam = np.array([0.6, 2.0, 0.64])
    zf = cyl - cam[:2]; zf = np.r_[zf / np.linalg.norm(zf) * math.cos(0.2), -math.sin(0.2)]
    x = np.cross(zf, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(zf, x)
    T = np.eye(4); T[:3, :3] = np.column_stack([x, y, zf]); T[:3, 3] = cam
    H, W = 480, 640
    dep = np.full((H, W), np.inf)
    vv, uu = np.mgrid[0:H, 0:W]
    rays = np.stack([(uu - 320) / 465.6, (vv - 240) / 465.6, np.ones_like(uu, float)], -1) @ T[:3, :3].T
    o = cam
    # 桌面
    t_tab = (0.45 - o[2]) / rays[..., 2]
    hit = (t_tab > 0) & (np.abs(o[0] + t_tab * rays[..., 0] - 1.35) <= 0.3) & (np.abs(o[1] + t_tab * rays[..., 1] - 2.3) <= 0.55)
    dist = np.where(hit, t_tab, np.inf)
    # 圓柱側面
    dx, dy = rays[..., 0], rays[..., 1]
    ox, oy = o[0] - cyl[0], o[1] - cyl[1]
    a = dx * dx + dy * dy; b = 2 * (ox * dx + oy * dy); c_ = ox * ox + oy * oy - 0.015 ** 2
    disc = b * b - 4 * a * c_
    tc = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    zc = o[2] + tc * rays[..., 2]
    hitc = (tc > 0) & (zc >= 0.45) & (zc <= 0.65)
    dist = np.where(hitc & (tc < dist), tc, dist)
    dep = dist * (rays @ T[:3, 2])                                   # 到影像平面距離
    p, info = detect(dep, K, T, table)
    print('est', None if p is None else np.round(p, 4), 'truth', np.r_[cyl, 0.55], 'err_mm', None if p is None else round(float(np.linalg.norm(p[:2] - cyl)) * 1e3, 2), info)
