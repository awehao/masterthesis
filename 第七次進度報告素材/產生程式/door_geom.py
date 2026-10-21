#!/usr/bin/env python3
"""局部通行門的純幾何（有向方盒、橫桿、門扇、牆）；供離線核對與單元測試共用。不依賴模擬器。

局部座標（door_local_v*.yaml）：鉸鏈軸在原點、牆沿 x 佔 y ∈ [0, wall.thickness]、機器人在 −y 側；
門角 q（rad）＝門扇繞 +z 轉 −q。門扇方向 u(q) = (cos q, −sin q, 0)，牆內側方向 v(q) = (sin q, cos q, 0)，
機器人側法向 n(q) = −v(q)。把手 prim 約定與 DL1 相同：x＝沿橫桿（= u）、y＝指向物體內側（= v）、z＝x × y（= +z）。
"""
from __future__ import annotations

import math

import numpy as np


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def leaf_frame(q):
    """回傳 R（欄＝u, v, z）。"""
    return Rz(-q)


def handle_pose(D, q, h):
    """把手 prim 位姿 T_WH(q)（4×4）：原點＝桿心。"""
    H = D['handle']
    R = leaf_frame(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = R[:, 0] * H['from_hinge'] - R[:, 1] * H['standoff'] + np.array([0.0, 0.0, h])
    return T


def shapes(D, q, h):
    """有向方盒清單 (名稱, 中心, R, 半長)。"""
    L, t = D['leaf']['length'], D['leaf']['thickness']
    z0, z1 = D['leaf']['z']
    R = leaf_frame(q)
    u, v = R[:, 0], R[:, 1]
    out = [('leaf', u * L / 2 + v * t / 2 + np.array([0, 0, (z0 + z1) / 2]), R, np.array([L / 2, t / 2, (z1 - z0) / 2]))]
    H = D['handle']
    su, sn, sz = H['posts']['size']
    for k, du in enumerate(H['posts']['along_u']):
        c = u * (H['from_hinge'] + du) - v * (sn / 2) + np.array([0, 0, h])
        out.append((f'post{k}', c, R, np.array([su / 2, sn / 2, sz / 2])))
    W, T, Ht = D['frame_clear_width'], D['wall']['thickness'], D['wall']['height']
    x0, x1 = D['wall']['extent_x']
    I3 = np.eye(3)
    out.append(('wall_hinge_side', np.array([x0 / 2, T / 2, Ht / 2]), I3, np.array([-x0 / 2, T / 2, Ht / 2])))
    out.append(('wall_latch_side', np.array([(W + x1) / 2, T / 2, Ht / 2]), I3, np.array([(x1 - W) / 2, T / 2, Ht / 2])))
    return out


def box_dist(P, c, R, hs):
    """點到有向方盒的有號距離（外正內負）。P：(N,3)。"""
    L = (np.atleast_2d(P) - c) @ R
    d = np.abs(L) - hs
    return np.linalg.norm(np.maximum(d, 0.0), axis=1) + np.minimum(np.max(d, axis=1), 0.0)


def bar_dist(D, P, q, h):
    """點到橫桿（有限圓柱、端面平切的近似：到線段距離 − 半徑）的距離。"""
    H = D['handle']
    T = handle_pose(D, q, h)
    u, c = T[:3, 0], T[:3, 3]
    P = np.atleast_2d(P)
    s = np.clip((P - c) @ u, -H['bar']['length'] / 2, H['bar']['length'] / 2)
    return np.linalg.norm(P - (c + s[:, None] * u), axis=1) - H['bar']['radius']


def disc_clear(x, y, z, r, c, R, hs):
    """水平圓盤（中心 (x, y)、高度 z、半徑 r）對方盒的間距：圓心到方盒距離 − r（圓心在方盒外時精確，內部為負）。"""
    return float(box_dist(np.array([[x, y, z]]), c, R, hs)[0]) - r


def pass_angle(D, envelope_r, clear, step_deg=0.5):
    """門扇（含厚度）在門洞 x 方向的投影 ≤ 淨寬 − 2(包絡 + 間距) 的最小門角（度）；只是投影寬度估算，非通行充分條件。"""
    L, t, W = D['leaf']['length'], D['leaf']['thickness'], D['frame_clear_width']
    need = W - 2 * (envelope_r + clear)
    for qd in np.arange(0, D['leaf']['limit_q_deg'][1] + 1e-9, step_deg):
        q = math.radians(qd)
        if L * math.cos(q) + t * math.sin(q) <= need:
            return float(qd), float(need)
    return None, float(need)
