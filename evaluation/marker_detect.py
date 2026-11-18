#!/usr/bin/env python3
"""抽屜內粗馬克筆代理（平躺六角柱）深度偵測（幾何法；不用學習模型、不讀真值、不讀登錄的筆位置）。
輸入：手腕相機深度（到影像平面距離）、內參 K、相機光學座標世界位姿 T_cam、抽屜內部範圍 box＝(中心, R, 半尺寸)（世界；由已知家具登錄＋控制器的開度估計得到：drawer_scene.interior_box）。
步驟：每 STRIDE 像素反投影 → 留內部範圍（橫向、深度各內縮 INSET）且高於底板頂 Z_LO 的點 → 點數 ≥ MIN_PTS →
  水平主成分＝筆軸；沿軸範圍＝長度（須在 [L_MIN, L_MAX]）、垂直軸散布＝寬度（≤ W_MAX）；
  中心：沿軸取範圍中點、垂直軸取頂面點（最高 6 mm 內）的平均（頂面對稱於軸）；高度＝底板頂＋已知半對邊 0.013。
回 (p_world(3) 或 None, 筆軸偏航（rad，模 π）或 None, info)。"""
import math

import numpy as np

STRIDE, MIN_PTS = 2, 25
Z_LO = 0.004
INSET = 0.008
L_MIN, L_MAX, W_MAX = 0.06, 0.20, 0.05
HALF_FLATS = 0.013


def detect(dep, K, T_cam, box, stride=STRIDE):
    c, R, hs = (np.asarray(x, float) for x in box)
    d = np.asarray(dep, float)
    H, W = d.shape
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = d[0:H:stride, 0:W:stride]
    ok = np.isfinite(z) & (z > 0.05) & (z < 3.0)
    if not ok.any():
        return None, None, dict(why='no_depth')
    u, v, z = u[ok].astype(float), v[ok].astype(float), z[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Pc = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)])
    Pw = (np.asarray(T_cam, float) @ Pc)[:3].T
    L = (Pw - c) @ R                                                   # 抽屜內部本地座標
    z_bot = -hs[2]
    m = (np.abs(L[:, 0]) <= hs[0] - INSET) & (np.abs(L[:, 1]) <= hs[1] - INSET) & (L[:, 2] >= z_bot + Z_LO) & (L[:, 2] <= hs[2])
    n = int(m.sum())
    if n < MIN_PTS:
        return None, None, dict(why='few_points', n=n)
    P = Pw[m]
    q = P[:, :2] - P[:, :2].mean(0)
    w_, V = np.linalg.eigh(q.T @ q / len(q))
    a = V[:, 1]                                                        # 主軸
    s_a, s_p = q @ a, q @ np.array([-a[1], a[0]])
    length, width = float(s_a.max() - s_a.min()), float(s_p.max() - s_p.min())
    if not (L_MIN <= length <= L_MAX) or width > W_MAX:
        return None, None, dict(why='shape', n=n, length_m=round(length, 4), width_m=round(width, 4))
    top = P[:, 2] >= P[:, 2].max() - 0.006
    nrm = np.array([-a[1], a[0]])
    c_xy = P[:, :2].mean(0) + a * (s_a.max() + s_a.min()) / 2.0
    c_xy = c_xy + nrm * float(((P[top, :2] - c_xy) @ nrm).mean())
    z_b = float(c[2] + (R @ np.array([0.0, 0.0, z_bot]))[2])
    yaw = (math.atan2(a[1], a[0]) + math.pi / 2) % math.pi - math.pi / 2
    return np.array([c_xy[0], c_xy[1], z_b + HALF_FLATS]), yaw, dict(n=n, n_top=int(top.sum()), length_m=round(length, 4), width_m=round(width, 4), z_top=round(float(P[:, 2].max()), 4))


if __name__ == '__main__':                                             # 合成深度自測：相機從斜上方看抽屜底板上的平躺六角柱（以 30×26 mm 方盒近似）
    K = np.array([[465.6, 0, 320], [0, 465.6, 240], [0, 0, 1.0]])
    zb = 0.447
    box = (np.array([-4.6, 10.3, zb + 0.03]), np.eye(3), np.array([0.12, 0.25, 0.03]))
    mk, yaw0 = np.array([-4.57, 10.32]), math.radians(80.0)
    a = np.array([math.cos(yaw0), math.sin(yaw0)]); nrm = np.array([-a[1], a[0]])
    cam = np.array([-4.50, 10.25, 0.72])
    zf = np.r_[mk - cam[:2], zb - cam[2]]; zf /= np.linalg.norm(zf)
    x = np.cross(zf, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(zf, x)
    T = np.eye(4); T[:3, :3] = np.column_stack([x, y, zf]); T[:3, 3] = cam
    H, W = 480, 640
    vv, uu = np.mgrid[0:H, 0:W]
    rays = np.stack([(uu - 320) / 465.6, (vv - 240) / 465.6, np.ones_like(uu, float)], -1) @ T[:3, :3].T
    t_b = (zb - cam[2]) / rays[..., 2]
    dist = np.where(t_b > 0, t_b, np.inf)
    # 方盒（筆）：與 3 個面求交（頂面＋兩側面）
    lo = np.array([-0.07, -0.015, zb]); hi = np.array([0.07, 0.015, zb + 0.026])
    o_l = np.array([(cam[:2] - mk) @ a, (cam[:2] - mk) @ nrm, cam[2]])
    r_l = np.stack([rays[..., :2] @ a, rays[..., :2] @ nrm, rays[..., 2]], -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t1 = (lo - o_l) / r_l; t2 = (hi - o_l) / r_l
    tn = np.nanmax(np.minimum(t1, t2), -1); tx = np.nanmin(np.maximum(t1, t2), -1)
    hitb = (tn <= tx) & (tn > 0)
    dist = np.where(hitb & (tn < dist), tn, dist)
    dep = dist * (rays @ T[:3, 2])
    p, yw, info = detect(dep, K, T, box)
    print('est', None if p is None else np.round(p, 4), 'yaw', None if yw is None else round(math.degrees(yw), 2), 'truth', mk, 80.0,
          'err_mm', None if p is None else round(float(np.linalg.norm(p[:2] - mk)) * 1e3, 2), info)
