#!/usr/bin/env python3
"""XH3 幾何後端（評估用；規格 results/vision/XH3_review_train_spec.md draft-2）。全部用 PC2 像素中心 +0.5 契約（d1_handle_detect_v2）。

* sized(d, L)：暫時把 d1_handle_detect_v2 的尺寸相依常數換成資產值（離開即還原）——
  R_BAR = d/2、LEN_BAR = L、P['cover_max'] = 1.15·L、P['width_min'] = 0.008·(d/0.026)、P['width_max'] = 0.041·(d/0.026)；其他門檻不變。
  演算法本身不改（凍結邏輯照跑），bar26（26 mm／200 mm）時與原值完全相同。
* g0k_detect：G0-K＝「已知尺寸的改編幾何基線」＝ sized 下的 d1_handle_detect_v2.detect（含前板／寬度篩選；不是原凍結 D1）。
* k_detect：G1／L1 共用的 K 後端＝ dl0_g1_check.g1_detect（sha256 3055e064ea85caed2630865ee3f72ed1518582c6ec41fe328a92e58d13628709，凍結）的邏輯，改用 +0.5 反投影並在 sized 下執行
  （半徑、長度、覆蓋上限、端外探針半徑皆為資產值；略過前板／寬度篩選，與 DL0 相同）。
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d1_handle_detect_v2 as D                      # noqa: E402

KEYS = ('cover_max', 'width_min', 'width_max')


@contextlib.contextmanager
def sized(d_m, L_m):
    saved = (D.R_BAR, D.LEN_BAR, {k: D.P[k] for k in KEYS})
    D.R_BAR, D.LEN_BAR = d_m / 2.0, L_m
    D.P['cover_max'] = 1.15 * L_m
    D.P['width_min'] = 0.008 * (d_m / 0.026)
    D.P['width_max'] = 0.041 * (d_m / 0.026)
    try:
        yield
    finally:
        D.R_BAR, D.LEN_BAR = saved[0], saved[1]
        for k, v in saved[2].items():
            D.P[k] = v


def g0k_detect(dep, K, T, rng, d_m, L_m):
    with sized(d_m, L_m):
        return D.detect(dep, K, T, rng)


def _g1_detect_pc(dep, K, T, mask, rng):
    P = D.P
    out = {'L0': None, 'L1': None, 'L2': None, 'reject': None, 'reject_L2': None}
    v, u = np.nonzero(mask)
    keep = (v % P['stride'] == 0) & (u % P['stride'] == 0)
    v, u = v[keep], u[keep]
    z = dep[v, u]
    ok = np.isfinite(z) & (z >= D.ZMIN) & (z <= D.ZMAX)
    u, v, z = u[ok], v[ok], z[ok]
    if len(z) < P['cluster_min']:
        out['reject'] = 'mask_too_small'
        return out
    pc = np.stack([(u + D.PIXEL_CENTER - K[0, 2]) / K[0, 0] * z, (v + D.PIXEL_CENTER - K[1, 2]) / K[1, 1] * z, z], 1)
    X = pc @ T[:3, :3].T + T[:3, 3]
    _, k = np.unique(np.floor(X / P['voxel']).astype(np.int64), axis=0, return_index=True)
    X = X[np.sort(k)]
    f = D.fit_cylinder(X, rng) if len(X) >= P['cluster_min'] else None
    if f is None:
        out['reject'] = 'fit_failed'
        return out
    c3, ax, inl, rms = f
    tt = (X[inl] - c3) @ ax
    cover = float(tt.max() - tt.min())
    if not (inl.sum() >= P['inl_min'] and P['cover_min'] <= cover <= P['cover_max']
            and abs(ax[2]) <= P['axz_max'] and rms <= P['rms_max']):
        out['reject'] = 'candidate_filter'
        return out
    p0 = c3 + ax * (tt.min() + tt.max()) / 2
    c = {'p0': p0, 'ax': ax, 'tmin': float(tt.min() - (tt.min() + tt.max()) / 2),
         'tmax': float(tt.max() - (tt.min() + tt.max()) / 2), 'pts': X[inl]}
    out['L0'] = {'n': int(inl.sum()), 'pts_sample': c['pts'][::max(1, len(c['pts']) // 200)].tolist()}
    out['L1'] = {'p0': p0.tolist(), 'axis': ax.tolist(), 'cover_m': cover, 'rms_m': float(rms)}
    # ---- 以下為 D1 的 L2 端點規則原樣（常數取自 D1）----
    cam = T[:3, 3]
    toward = cam - c['p0']
    toward -= (toward @ c['ax']) * c['ax']
    toward /= max(np.linalg.norm(toward), 1e-9)
    ends, why = [], []
    for t_end, sgn in ((c['tmin'], -1.0), (c['tmax'], 1.0)):
        pe = c['p0'] + c['ax'] * t_end
        uv, _ = D.project(T, K, pe)
        if uv is None or not (P['border_px'] < uv[0] < dep.shape[1] - P['border_px']
                              and P['border_px'] < uv[1] < dep.shape[0] - P['border_px']):
            why.append('end_near_border')
            continue
        probe = c['p0'] + c['ax'] * (t_end + sgn * P['end_probe']) + toward * D.R_BAR
        uvp, zp = D.project(T, K, probe)
        if uvp is None or not (0 <= uvp[0] < dep.shape[1] and 0 <= uvp[1] < dep.shape[0]):
            why.append('probe_outside')
            continue
        zo = float(dep[D.to_index(uvp[1]), D.to_index(uvp[0])])
        if not np.isfinite(zo) or zo < D.ZMIN:
            why.append('probe_no_depth')
            continue
        if zo < zp + P['end_bg']:
            why.append('probe_not_background')
            continue
        ends.append(pe)
    if len(ends) == 2:
        L = float(np.linalg.norm(ends[1] - ends[0]))
        if abs(L - D.LEN_BAR) <= P['len_tol'] * D.LEN_BAR:
            out['L2'] = {'center': ((ends[0] + ends[1]) / 2).tolist(), 'length_m': L}
        else:
            why.append(f'length_inconsistent_{L:.3f}')
    if out['L2'] is None:
        out['reject_L2'] = 'center_unobservable:' + ','.join(why)
    return out


def k_detect(dep, K, T, mask, rng, d_m, L_m):
    with sized(d_m, L_m):
        return _g1_detect_pc(dep, K, T, mask, rng)
