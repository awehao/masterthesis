#!/usr/bin/env python3
"""觀察視角的線上生成與可見性核對（R1b；Codex 20261111_013508 第 4 項「真正機身右側觀察」）。
生成：在看到物件之前，只用「任務區域觀看點」（區域知識）與「預測底盤位姿」決定手腕相機往哪看：
  以觀察庫的高位構型（V_HIGH）的腕部形狀為樣板，只解 q1，使觀看點落在影像水平中央；q1 先在關節安全範圍內粗掃（取觀看點在相機前且水平誤差最小者），再牛頓修正。
  左側時結果≈V_HIGH＋既有視向偏移；右側時 q1 轉到機身右前方。不讀真值瓶位，不存「右側固定姿態」。
可見性（候選可就緒的必要幾何條件）：同 bottle_w1_gaze_ctrl 的 T_MOUNT、FX；640×480。
  瓶底心與瓶頂心投影都在畫面內且離邊 ≥ EDGE_PX、在相機前 ≥ Z_MIN；相機對瓶底心（支撐點）的俯角 ≥ ELEV_MIN；
  手臂連桿點雲（含夾爪手指）沒有落在瓶 16 點投影外框內且比瓶近（自遮擋）。
  ELEV_MIN 由名目趟 bottle_motm_nom_1 校準（candidates.jsonl 每列時刻、w1_frames 1 Hz 相機位姿線性內插）：俯角 4.2° 以下各列 support_observed 全為 0，
  5.1° 起全為 1；取已觀測到的最低通過值 5.1°（4.2–5.1° 之間無資料）。此後 K_MIN＝3 列一致才就緒（名目趟：支撐首次量到 11.14 s → 就緒 11.60 s）。
  底盤車體與光達座不在點雲內（未核車體遮擋），分割模型在右側視角的實際表現未核（照實揭露）。
"""
import math

import numpy as np

import bottle_w1_gaze_ctrl as GZ

W_PX, H_PX = 640, 480
FX = GZ.FX
EDGE_PX = 5.0
ELEV_MIN = 5.1                                                  # 度；名目趟校準（見檔頭）
Z_MIN = GZ.Z_MIN


def cam(K, qb, qa):
    return K.fk(np.r_[np.asarray(qb, float), np.asarray(qa, float)], 'link_eef') @ GZ.T_MOUNT


def project(Tc, P):
    Pc = (Tc[:3, :3].T @ (np.atleast_2d(P) - Tc[:3, 3]).T).T
    z = Pc[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = FX * Pc[:, 0] / z + W_PX / 2
        v = FX * Pc[:, 1] / z + H_PX / 2
    return u, v, z


def view_q1(K, qb, q_tmpl, look, lo, hi, n_grid=145):
    """回 (q 視角構型, 資訊)。q_tmpl 只提供 q2–q6 樣板；q1 由觀看點與預測底盤位姿決定。"""
    q = np.asarray(q_tmpl, float).copy()
    best = None
    for q1 in np.linspace(lo, hi, n_grid):
        q[0] = q1
        u, v, z = project(cam(K, qb, q), look)
        if z[0] < Z_MIN:
            continue
        e = abs(u[0] - W_PX / 2)
        if best is None or e < best[0]:
            best = (e, q1)
    if best is None:
        return None, dict(why='look_behind_for_all_q1')
    q[0] = best[1]
    for _ in range(8):
        u, v, z = project(cam(K, qb, q), look)
        e = (u[0] - W_PX / 2) / FX
        q2 = q.copy()
        q2[0] += 1e-4
        u2, _, _ = project(cam(K, qb, q2), look)
        de = ((u2[0] - W_PX / 2) / FX - e) / 1e-4
        if abs(de) < 1e-6:
            break
        q[0] = float(np.clip(q[0] - e / de, lo, hi))
    u, v, z = project(cam(K, qb, q), look)
    return q, dict(q1=round(float(q[0]), 4), u_px=round(float(u[0]), 1), v_px=round(float(v[0]), 1), z=round(float(z[0]), 3))


def bottle_pts(xb, yb, zs, r=0.015, h=0.20, n=8):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.array([[xb + r * math.cos(t), yb + r * math.sin(t), zs + hz] for hz in (0.0, h) for t in a])


def visible(K, qb, qa, P, clouds=None, zs=None):
    """回 dict(ok, edge_px, z_min, elev_deg, n_occ)。P＝瓶 16 點（前 8 點為瓶底緣）；clouds＝{link: 點雲（連桿座標）}，給則核自遮擋。"""
    Tc = cam(K, qb, qa)
    c_bot, c_top = P[:len(P) // 2].mean(axis=0), P[len(P) // 2:].mean(axis=0)
    u, v, z = project(Tc, np.vstack([c_bot, c_top]))
    zmin = float(z.min())
    d = c_bot - Tc[:3, 3]
    elev = math.degrees(math.atan2(-d[2], math.hypot(d[0], d[1])))
    if zmin < Z_MIN:
        return dict(ok=False, edge_px=None, z_min=round(zmin, 3), elev_deg=round(elev, 1), n_occ=None)
    edge = float(min(u.min(), W_PX - u.max(), v.min(), H_PX - v.max()))
    n_occ = 0
    if clouds is not None and edge >= 0:
        ua, va, za = project(Tc, P)
        u0, u1, v0, v1 = ua.min(), ua.max(), va.min(), va.max()
        qf = np.r_[np.asarray(qb, float), np.asarray(qa, float)]
        for n_, C in clouds.items():
            T = K.fk(qf, n_)
            W = (T[:3, :3] @ C.T).T + T[:3, 3]
            uu, vv, zz = project(Tc, W)
            m = (zz > 0.02) & (zz < za.min()) & (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1)
            n_occ += int(m.sum())
    ok = edge >= EDGE_PX and elev >= ELEV_MIN and n_occ == 0
    return dict(ok=bool(ok), edge_px=round(edge, 1), z_min=round(zmin, 3), elev_deg=round(elev, 1), n_occ=n_occ)
