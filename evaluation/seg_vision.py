#!/usr/bin/env python3
"""Mask R-CNN 遮罩＋深度的物體三維估計（Howard 11/23「視覺幫我用 maskrcnn」）。純 numpy，可在 Isaac 環境呼叫。
搜尋範圍來自遮罩，不再用家具外框、抽屜內部範圍等登錄先驗；仍用的先驗只有物體本身的已知形狀（圓柱半徑／高、筆對邊、把手橫桿尺寸、門鉸鏈位置）。
各函式輸出格式與原幾何偵測器相同（cyl_detect／marker_detect／handle_detect／door_leaf_detect），呼叫端可直接替換。
SegClient：啟動 house_seg_server.py（.venv-dl0）子程序並同步請求。"""
import io
import math
import os
import struct
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
CYL_R, CYL_H = 0.015, 0.20
PEN_HALF_FLAT = 0.013


class SegClient:
    def __init__(self, weights, py=os.path.join(WS, '.venv-dl0', 'bin', 'python')):
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD')}
        weights = weights if os.path.isabs(weights) else os.path.join(WS, weights)
        self.p = subprocess.Popen([py, os.path.join(HERE, 'house_seg_server.py'), weights], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        ln = self.p.stdout.readline()
        if not ln.startswith(b'READY'):
            raise RuntimeError(f'seg server 未就緒：{ln!r}')

    def predict(self, rgb):
        bio = io.BytesIO(); np.savez(bio, rgb=np.ascontiguousarray(np.asarray(rgb)[:, :, :3], dtype=np.uint8)); b = bio.getvalue()
        self.p.stdin.write(struct.pack('<Q', len(b)) + b); self.p.stdin.flush()
        n = struct.unpack('<Q', self.p.stdout.read(8))[0]
        z = np.load(io.BytesIO(self.p.stdout.read(n)))
        H, W = (int(x) for x in z['shape'])
        out = dict(_ms=float(z['ms']))
        for k in z.files:
            if k.startswith('m_'):
                c = k[2:]
                out[c] = dict(mask=np.unpackbits(z[k])[:H * W].reshape(H, W).astype(bool), score=float(z['s_' + c]), n=int(z['n_' + c]))
        return out

    def close(self):
        try:
            self.p.stdin.close(); self.p.wait(timeout=5)
        except Exception:                                             # noqa: BLE001
            self.p.kill()


def _erode(m, k=2):
    from numpy.lib.stride_tricks import sliding_window_view as sw
    if k <= 0:
        return m
    p = np.pad(m, k, constant_values=False)
    return sw(p, (2 * k + 1, 2 * k + 1)).all(axis=(2, 3))


def mask_points(dep, K, T_cam, mask, erode=2, max_dev=0.15):
    """遮罩（內縮 erode 像素去邊緣混合深度）內的有效深度點 → 世界座標；去掉與遮罩深度中位數差 > max_dev 的背景漏入點。"""
    d = np.asarray(dep, float)
    m = _erode(np.asarray(mask, bool), erode) & np.isfinite(d) & (d > 0.05) & (d < 4.0)
    if m.sum() < 5:
        return np.zeros((0, 3))
    v, u = np.nonzero(m); z = d[v, u]
    keep = np.abs(z - np.median(z)) <= max_dev
    u, v, z = u[keep].astype(float), v[keep].astype(float), z[keep]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return (np.asarray(T_cam, float) @ np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)]))[:3].T


def cyl(dep, K, T_cam, mask, min_pts=30):
    """茶几圓柱：遮罩點 → 已知半徑水平圓擬合 → 軸心；高度＝可見頂（98 百分位）− 半高。回 (p_world, info)。"""
    P = mask_points(dep, K, T_cam, mask)
    if len(P) < min_pts:
        return None, dict(why='few_points', n=int(len(P)), src='seg')
    c = P[:, :2].mean(0)
    vdir = c - np.asarray(T_cam, float)[:2, 3]; vdir = vdir / max(float(np.linalg.norm(vdir)), 1e-9)
    xy = c + (2.0 / math.pi) * CYL_R * vdir
    Q = P[:, :2]
    for _ in range(8):
        dq = Q - xy; rr = np.maximum(np.linalg.norm(dq, axis=1), 1e-9)
        J = -dq / rr[:, None]
        xy = xy + np.linalg.lstsq(J, -(rr - CYL_R), rcond=None)[0]
    z = float(np.percentile(P[:, 2], 98)) - CYL_H / 2
    return np.array([xy[0], xy[1], z]), dict(n=int(len(P)), src='seg')


def pen(dep, K, T_cam, mask, min_pts=25):
    """筆（平躺六角柱）：遮罩點水平主成分＝筆軸；中心沿軸取範圍中點；高度＝頂面點平均 − 半對邊。回 (p_world, yaw(mod π), info)。"""
    P = mask_points(dep, K, T_cam, mask)
    if len(P) < min_pts:
        return None, None, dict(why='few_points', n=int(len(P)), src='seg')
    c = P[:, :2].mean(0)
    w, V = np.linalg.eigh(np.cov((P[:, :2] - c).T)); a = V[:, 1]
    s = (P[:, :2] - c) @ a
    s0, s1 = float(np.percentile(s, 2)), float(np.percentile(s, 98))
    ctr = c + a * 0.5 * (s0 + s1)
    top = P[P[:, 2] >= P[:, 2].max() - 0.006]
    z = float(top[:, 2].mean()) - PEN_HALF_FLAT
    yaw = math.atan2(a[1], a[0]) % math.pi
    return np.array([ctr[0], ctr[1], z]), yaw, dict(n=int(len(P)), length=round(s1 - s0, 4), src='seg')


def drawer_handle(dep, K, T_cam, bar_mask, front_mask, prior, spec, min_pts=20):
    """抽屜把手：面板遮罩點擬合 y＝a·x＋b（先驗本地座標，只當座標系，不劃範圍）⇒ 偏航修正；橫桿遮罩點 ⇒ 桿心。輸出同 handle_detect.detect。"""
    px, py, yaw = (float(v) for v in prior)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    bar = spec['drawer']['handle']['bar']
    bc = np.asarray(bar['center'], float); br = float(bar['radius']); bl = float(bar['length'])
    B = mask_points(dep, K, T_cam, bar_mask, erode=1)
    if len(B) < min_pts:
        return None, dict(why='bar_few_points', n=int(len(B)), src='seg')
    Bl = (B - np.array([px, py, 0.0])) @ Rz
    F = mask_points(dep, K, T_cam, front_mask) if front_mask is not None else np.zeros((0, 3))
    if len(F) >= 4 * min_pts and float(np.ptp(((F - np.array([px, py, 0.0])) @ Rz)[:, 0])) >= 0.15:
        Fl = (F - np.array([px, py, 0.0])) @ Rz
        a, b = np.linalg.lstsq(np.column_stack([Fl[:, 0], np.ones(len(Fl))]), Fl[:, 1], rcond=None)[0]
        dyaw = math.atan(float(a)); yaw_src = 'panel'
    else:                                                             # 面板看不到 ⇒ 橫桿主方向
        cb = Bl[:, :2].mean(0); w, V = np.linalg.eigh(np.cov((Bl[:, :2] - cb).T)); ax = V[:, 1]
        if ax[0] < 0:
            ax = -ax
        dyaw = math.atan2(ax[1], ax[0]); yaw_src = 'bar'
    cr, sr = math.cos(-dyaw), math.sin(-dyaw)
    Q = Bl.copy(); Q[:, 0], Q[:, 1] = cr * Bl[:, 0] - sr * Bl[:, 1], sr * Bl[:, 0] + cr * Bl[:, 1]
    x0, x1 = float(np.percentile(Q[:, 0], 1)), float(np.percentile(Q[:, 0], 99))
    if x1 - x0 < 0.5 * bl or x1 - x0 > bl + 0.06:
        return None, dict(why='bar_length', seen=round(x1 - x0, 4), n=int(len(Q)), src='seg')
    xb, zb = 0.5 * (x0 + x1), float(np.median(Q[:, 2]))
    yb = float(np.percentile(Q[:, 1], 5)) + br
    dx, dy = xb - float(bc[0]), yb - float(bc[1])
    c2, s2 = math.cos(yaw + dyaw), math.sin(yaw + dyaw)
    origin = np.array([px, py]) + np.array([[c2, -s2], [s2, c2]]) @ np.array([dx, dy])
    bar_w = np.array([px, py, 0.0]) + Rz @ np.array([xb * math.cos(dyaw) - yb * math.sin(dyaw), xb * math.sin(dyaw) + yb * math.cos(dyaw), zb])
    return dict(origin=origin, yaw=yaw + dyaw, dx=dx, dy=dy, dyaw=dyaw, bar_world=bar_w, z_bar=zb), dict(n_bar=int(len(Q)), n_face=int(len(F)), seen_len=round(x1 - x0, 4), yaw_src=yaw_src, src='seg')


def door_leaf(dep, K, T_cam, mask, min_pts=60, min_span=0.30, hinge_tol=0.05):
    """門扇：遮罩點 xy 主方向（門局部＝世界，鉸鏈在原點）⇒ φ＝−方向角；核點沿直線範圍 ≥ min_span、直線過鉸鏈附近。輸出同 door_leaf_detect.detect。"""
    P = mask_points(dep, K, T_cam, mask)
    if len(P) < min_pts:
        return None, dict(why='few_points', n=int(len(P)), src='seg')
    c = P[:, :2].mean(0)
    w, V = np.linalg.eigh(np.cov((P[:, :2] - c).T)); a = V[:, 1]
    if a @ c < 0:                                                    # 方向朝離開鉸鏈
        a = -a
    sp = (P[:, :2] - c) @ a
    span = float(sp.max() - sp.min()); d_h = abs(float(c[0] * a[1] - c[1] * a[0]))
    if span < min_span or d_h > hinge_tol:
        return None, dict(why='leaf_shape', span=round(span, 3), hinge_dist=round(d_h, 3), n=int(len(P)), src='seg')
    return -math.atan2(a[1], a[0]), dict(n=int(len(P)), span=round(span, 3), hinge_dist=round(d_h, 4), src='seg')


def masked_depth(dep, mask, grow=4):
    """門把：既有幾何偵測（d1_handle_detect_v2）只吃遮罩（外擴 grow 像素）內的深度，其他設 NaN。"""
    m = np.asarray(mask, bool)
    if grow > 0:
        m = ~_erode(~m, grow)
    d = np.asarray(dep, float).copy()
    d[~m] = np.nan
    return d
