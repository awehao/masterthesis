#!/usr/bin/env python3
"""D1-shadow S2：腕部 RGB-D 的把手自動辨識（已知類型與尺寸的深度幾何辨識；離線，逐影格）。

規格：evaluation/results/vision/D1_shadow_spec.yaml（draft-2）。**偵測端不讀真值**：`detect()` 只用
深度、讀回內參與擷取時刻相機位姿；真值（meta 的 handle_center_world_at_capture 與 truth.json）只在 `evaluate()`。

偵測流程（每格）：
  1 深度 → 世界點（光軸深度有效 [0.1, 3.0] m；每 2 px 取一點）
  2 RANSAC 依序找至多 4 個平面；前板候選＝法向近水平（|n_z| ≤ 0.2）的平面（由觀測決定，不指定真值前板）
  3 每個前板候選：取平面朝相機側 20–60 mm 且高於地面 0.1 m 的點 → 半徑 1.5 cm 連通分群
  4 每群：PCA 起始軸 → 在垂直軸的平面上以 RANSAC 擬合固定半徑 13 mm 的圓（內點容差 3 mm）→ 以內點重估軸，迭代
     （柱子等不符圓柱的點自然成為離群點）→ 篩選：內點 ≥ 40、軸向覆蓋 ≥ 40 mm、|軸·z| ≤ 0.2、殘差 RMS ≤ 3 mm
  5 合格候選 0 ⇒ no_candidate；≥ 2 ⇒ ambiguous；1 ⇒ 輸出 L0（內點）＋L1（軸線）
  6 L2：兩端都「可辨識」才輸出中點——(i) 端外 10 mm 處的軸線表面點投影到影像，該像素的觀測深度比橫桿表面遠 ≥ 15 mm
     （看得到背景，不是缺深度或遮擋）；(ii) 端點投影距影像邊界 > 5 px；(iii) 兩端距離與 200 mm 相差 ≤ 15%。
     不以已知長度補端點。否則 center_unobservable

    python3 evaluation/d1_handle_detect.py runs/<CAPTURE_RUN> [--overlay]
輸出 runs/<RUN>/wrist_v0/d1_detect.json（逐影格偵測＋評估）、d1_overlay/（疊圖）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wrist_v0_capture import wxyz_to_R      # noqa: E402

R_BAR, LEN_BAR = 0.013, 0.200
ZMIN, ZMAX = 0.1, 3.0
BAND = (0.020, 0.060)
# 開發集調整紀錄（d1_dev_f02P）：max_planes 4→6（遠距時前板不在前 4 大平面）；voxel 4 mm（近距每格 0.5–5 s → 降耗時）；
# 寬度 [8, 41] mm＝圓柱直徑 26 mm ±（退化線／加 15 mm）；長度上限 230 mm＝先驗 200 mm ×1.15（櫃背板條帶寬 0.12–0.26 m、長 0.34–0.56 m）
P = dict(stride=2, voxel=0.004, plane_tol=0.006, plane_min=400, max_planes=6, plane_iter=300, nz_max=0.2,
         floor_clear=0.10, cluster_r=0.015, cluster_min=30, circ_tol=0.003, circ_iter=200,
         inl_min=40, cover_min=0.040, axz_max=0.2, rms_max=0.003, end_probe=0.010, end_bg=0.015,
         border_px=5, len_tol=0.15, width_min=0.008, width_max=0.041, cover_max=0.230,
         plane_sub=5000)   # 加速（開發集）：平面假設只在 5000 點子樣本上評分，內點再對全部點計算


# ------------------------------------------------------------------ 幾何
def cam_T(f):
    T = np.eye(4)
    T[:3, :3] = wxyz_to_R(f['cam_quat_wxyz_world'])
    T[:3, 3] = f['cam_pos_world']
    return T


def backproject(dep, K, T, stride):
    v, u = np.mgrid[0:dep.shape[0]:stride, 0:dep.shape[1]:stride]
    z = dep[v, u]
    ok = np.isfinite(z) & (z >= ZMIN) & (z <= ZMAX)
    u, v, z = u[ok], v[ok], z[ok]
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    return (pc @ T[:3, :3].T) + T[:3, 3]


def project(T, K, p):
    pc = np.linalg.inv(T) @ np.r_[p, 1.0]
    if pc[2] <= 1e-6:
        return None, float(pc[2])
    return (K @ pc[:3])[:2] / pc[2], float(pc[2])


def ransac_plane(X, rng):
    """向量化 RANSAC：在子樣本上評分全部假設，最佳者再對全部點算內點。"""
    n = len(X)
    S = X[rng.choice(n, min(n, P['plane_sub']), replace=False)]
    idx = rng.integers(0, n, size=(P['plane_iter'], 3))
    a, b, c = X[idx[:, 0]], X[idx[:, 1]], X[idx[:, 2]]
    nv = np.cross(b - a, c - a)
    nn = np.linalg.norm(nv, axis=1)
    good = nn > 1e-9
    if not good.any():
        return None, None
    nv = nv[good] / nn[good][:, None]
    d = -np.einsum('ij,ij->i', nv, a[good])
    score = (np.abs(S @ nv.T + d) < P['plane_tol']).sum(0)
    k = int(np.argmax(score))
    inl = np.abs(X @ nv[k] + d[k]) < P['plane_tol']
    return (nv[k], float(d[k])), inl


def circle_fit_fixed_r(Y, rng):
    """2D 點擬合半徑 R_BAR 的圓（向量化 RANSAC）：每對點給兩個候選圓心，取內點最多者。"""
    n = len(Y)
    if n < 3:
        return None, None
    idx = rng.integers(0, n, size=(P['circ_iter'], 2))
    a, b = Y[idx[:, 0]], Y[idx[:, 1]]
    m = (a + b) / 2
    h = np.linalg.norm(b - a, axis=1) / 2
    ok = (h > 1e-4) & (h < R_BAR)
    if not ok.any():
        return None, None
    a, b, m, h = a[ok], b[ok], m[ok], h[ok]
    perp = np.stack([-(b - a)[:, 1], (b - a)[:, 0]], 1) / (2 * h)[:, None]
    off = np.sqrt(R_BAR ** 2 - h ** 2)[:, None]
    C = np.concatenate([m + perp * off, m - perp * off])
    D = np.abs(np.linalg.norm(Y[None, :, :] - C[:, None, :], axis=2) - R_BAR) < P['circ_tol']
    k = int(np.argmax(D.sum(1)))
    return C[k], D[k]


def fit_cylinder(C, rng):
    """回傳 (軸點, 軸向, 內點遮罩, rms) 或 None。"""
    axis = np.linalg.svd(C - C.mean(0), full_matrices=False)[2][0]
    inl = np.ones(len(C), bool)
    c3 = None
    for _ in range(3):
        e1 = np.cross(axis, [0, 0, 1.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(axis, [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        Y = np.stack([C @ e1, C @ e2], 1)
        c2, inl2 = circle_fit_fixed_r(Y, rng)
        if c2 is None or inl2.sum() < P['inl_min']:
            return None
        inl = inl2
        c3 = c2[0] * e1 + c2[1] * e2
        Q = C[inl] - c3
        ax2 = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2][0]
        axis = ax2 / np.linalg.norm(ax2)
    resid = np.abs(np.linalg.norm(np.cross(C[inl] - c3, axis), axis=1) - R_BAR)
    return c3, axis, inl, float(np.sqrt(np.mean(resid ** 2)))


# ------------------------------------------------------------------ 偵測（不讀真值）
def detect(dep, K, T, rng):
    t0 = time.monotonic()
    out = {'L0': None, 'L1': None, 'L2': None, 'reject': None, 'reject_L2': None}
    X = backproject(dep, K, T, P['stride'])
    if len(X):
        _, keep = np.unique(np.floor(X / P['voxel']).astype(np.int64), axis=0, return_index=True)
        X = X[np.sort(keep)]
    if len(X) < P['plane_min']:
        out['reject'] = 'depth_insufficient'
        return out, (time.monotonic() - t0) * 1e3
    rest = np.arange(len(X))
    planes = []
    for _ in range(P['max_planes']):
        if len(rest) < P['plane_min']:
            break
        pl, inl = ransac_plane(X[rest], rng)
        if pl is None or inl.sum() < P['plane_min']:
            break
        planes.append(pl)
        rest = rest[~inl]
    fronts = [pl for pl in planes if abs(pl[0][2]) <= P['nz_max']]
    if not fronts:
        out['reject'] = 'no_plane'
        return out, (time.monotonic() - t0) * 1e3
    cam = T[:3, 3]
    floor_z = np.percentile(X[:, 2], 1)          # 觀測到的最低處（地面）——不讀真值
    cands = []
    for nv, d in fronts:
        s = 1.0 if (nv @ cam + d) > 0 else -1.0    # 平面朝相機側為正
        dist = s * (X @ nv + d)
        sel = (dist >= BAND[0]) & (dist <= BAND[1]) & (X[:, 2] > floor_z + P['floor_clear'])
        Y = X[sel]
        if len(Y) < P['cluster_min']:
            continue
        tree = cKDTree(Y)
        pairs = tree.query_pairs(P['cluster_r'], output_type='ndarray')
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(Y), len(Y))) \
            if len(pairs) else coo_matrix((len(Y), len(Y)))
        _, roots = connected_components(g, directed=False)
        for r in np.unique(roots):
            Cl = Y[roots == r]
            if len(Cl) < P['cluster_min']:
                continue
            f = fit_cylinder(Cl, rng)
            if f is None:
                continue
            c3, ax, inl, rms = f
            tt = (Cl[inl] - c3) @ ax
            cover = float(tt.max() - tt.min())
            perp = np.cross(ax, nv)
            perp /= max(np.linalg.norm(perp), 1e-9)
            ww = Cl[inl] @ perp
            width = float(np.percentile(ww, 98) - np.percentile(ww, 2))
            ok = (inl.sum() >= P['inl_min'] and P['cover_min'] <= cover <= P['cover_max']
                  and abs(ax[2]) <= P['axz_max'] and rms <= P['rms_max']
                  and P['width_min'] <= width <= P['width_max'])
            if ok:
                p0 = c3 + ax * (tt.min() + tt.max()) / 2
                cands.append({'pts': Cl[inl], 'p0': p0, 'ax': ax, 'tmin': float(tt.min() - (tt.min() + tt.max()) / 2),
                              'tmax': float(tt.max() - (tt.min() + tt.max()) / 2), 'rms': rms,
                              'n_inl': int(inl.sum()), 'cover': cover, 'plane_n': nv.tolist(),
                              'width': width})
    if not cands:
        out['reject'] = 'no_candidate'
        return out, (time.monotonic() - t0) * 1e3
    if len(cands) >= 2:
        out['reject'] = 'ambiguous'
        out['n_candidates'] = len(cands)
        out['candidates'] = [{'p0': c['p0'].tolist(), 'axis': c['ax'].tolist(), 'n_inl': c['n_inl'],
                              'cover': c['cover'], 'rms': c['rms'], 'plane_n': c['plane_n'],
                              'width_m': c['width']} for c in cands]
        return out, (time.monotonic() - t0) * 1e3
    c = cands[0]
    out['L0'] = {'n': c['n_inl'], 'pts_sample': c['pts'][::max(1, len(c['pts']) // 200)].tolist()}
    out['L1'] = {'p0': c['p0'].tolist(), 'axis': c['ax'].tolist(), 'cover_m': c['cover'], 'rms_m': c['rms']}
    # ---- L2 端點可辨識性 ----
    toward = cam - c['p0']
    toward -= (toward @ c['ax']) * c['ax']
    toward /= max(np.linalg.norm(toward), 1e-9)
    ends, why = [], []
    for t_end, sgn in ((c['tmin'], -1.0), (c['tmax'], 1.0)):
        pe = c['p0'] + c['ax'] * t_end
        uv, _ = project(T, K, pe)
        if uv is None or not (P['border_px'] < uv[0] < dep.shape[1] - P['border_px']
                              and P['border_px'] < uv[1] < dep.shape[0] - P['border_px']):
            why.append('end_near_border')
            continue
        probe = c['p0'] + c['ax'] * (t_end + sgn * P['end_probe']) + toward * R_BAR
        uvp, zp = project(T, K, probe)
        if uvp is None or not (0 <= uvp[0] < dep.shape[1] and 0 <= uvp[1] < dep.shape[0]):
            why.append('probe_outside')
            continue
        zo = float(dep[int(uvp[1]), int(uvp[0])])
        if not np.isfinite(zo) or zo < ZMIN:
            why.append('probe_no_depth')          # 缺深度 ≠ 看到背景
            continue
        if zo < zp + P['end_bg']:
            why.append('probe_not_background')    # 端外仍是近物（遮擋或橫桿延續）
            continue
        ends.append(pe)
    if len(ends) == 2:
        L = float(np.linalg.norm(ends[1] - ends[0]))
        if abs(L - LEN_BAR) <= P['len_tol'] * LEN_BAR:
            out['L2'] = {'center': ((ends[0] + ends[1]) / 2).tolist(), 'length_m': L}
        else:
            why.append(f'length_inconsistent_{L:.3f}')
    if out['L2'] is None:
        out['reject_L2'] = 'center_unobservable:' + ','.join(why)
    return out, (time.monotonic() - t0) * 1e3


# ------------------------------------------------------------------ 評估（獨立端，讀真值）
def ray_cyl(o, d, c, ax, r, half):
    """射線與有限圓柱側面的第一交點（世界）；無則 None。"""
    w = o - c
    a_ = d - (d @ ax) * ax
    b_ = w - (w @ ax) * ax
    A, B, C = a_ @ a_, 2 * a_ @ b_, b_ @ b_ - r * r
    disc = B * B - 4 * A * C
    if A < 1e-12 or disc < 0:
        return None
    for s in sorted([(-B - np.sqrt(disc)) / (2 * A), (-B + np.sqrt(disc)) / (2 * A)]):
        if s > 0:
            p = o + s * d
            if abs((p - c) @ ax) <= half:
                return p
    return None


def evaluate(det, f, T, K, true_axis):
    c_true = np.array(f['handle_center_world_at_capture'], float)
    cam = T[:3, 3]
    ev = {'dist_m': float(np.linalg.norm(c_true - cam))}
    uvc, _ = project(T, K, c_true)
    e1, _ = project(T, K, c_true - true_axis * LEN_BAR / 2)
    e2, _ = project(T, K, c_true + true_axis * LEN_BAR / 2)
    def inside(uv):
        return uv is not None and 0 <= uv[0] < 640 and 0 <= uv[1] < 480
    ev['truth_center_in_frame'] = inside(uvc)
    ev['truth_ends_in_frame'] = inside(e1) and inside(e2)
    if det['L0']:
        errs, miss = [], 0
        for p in det['L0']['pts_sample']:
            p = np.array(p)
            d = (p - cam) / np.linalg.norm(p - cam)
            h = ray_cyl(cam, d, c_true, true_axis, R_BAR, LEN_BAR / 2)
            if h is None:
                miss += 1
            else:
                errs.append(float(np.linalg.norm(p - h)))
        ev['L0_misid'] = miss > 0.5 * len(det['L0']['pts_sample'])
        ev['L0_err_mm_median'] = float(np.median(errs) * 1e3) if errs else None
        ev['L0_frac_hit'] = 1 - miss / len(det['L0']['pts_sample'])
    if det['L1']:
        ax = np.array(det['L1']['axis'])
        ang = float(np.degrees(np.arccos(min(1.0, abs(ax @ true_axis)))))
        p0 = np.array(det['L1']['p0'])
        dax = float(np.linalg.norm(np.cross(p0 - c_true, true_axis)))
        ev['L1_angle_deg'] = ang
        ev['L1_point_to_axis_mm'] = dax * 1e3
        ev['L1_misid'] = (dax > 0.020) or (ang > 15.0)
    if det['L2']:
        e = float(np.linalg.norm(np.array(det['L2']['center']) - c_true))
        ev['L2_err_mm'] = e * 1e3
        ev['L2_wrong'] = e > 0.030
    # 獨立評估要求：中心在畫面外或端點被裁切 ⇒ L2 必須拒絕
    ev['L2_violation'] = bool(det['L2'] and not (ev['truth_center_in_frame'] and ev['truth_ends_in_frame']))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--overlay', action='store_true')
    a = ap.parse_args()
    V = os.path.join(a.run, 'wrist_v0')
    m = json.load(open(os.path.join(V, 'meta.json')))
    K = np.array(m['intrinsics_readback']['K'])
    true_axis = np.array([1.0, 0.0, 0.0])        # 評估端：資產軸 x（櫃體未旋轉）
    rng = np.random.default_rng(0)
    rows = []
    for f in m['frames']:
        T = cam_T(f)
        dep = np.load(os.path.join(V, 'frames', f'f{f["n"]:02d}_depth_m.npy'))
        dep = dep[:, :, 0] if dep.ndim == 3 else dep
        det, ms = detect(dep, K, T, rng)
        ev = evaluate(det, f, T, K, true_axis)
        rows.append({'n': f['n'], 'src_t': f['src_record_t'], 'detect_ms': round(ms, 2),
                     'reject': det['reject'], 'reject_L2': det['reject_L2'],
                     'L1': det['L1'], 'L2': det['L2'],
                     'L0_n': det['L0']['n'] if det['L0'] else None, 'eval': ev})
        if a.overlay:
            overlay(V, f, det, ev, T, K)
    json.dump({'spec': 'D1_shadow_spec.yaml draft-2', 'params': P, 'rows': rows},
              open(os.path.join(V, 'd1_detect.json'), 'w'), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    for r in rows:
        e = r['eval']
        print(f"f{r['n']:02d} t={r['src_t']:6.2f} d={e['dist_m']:.2f} rej={r['reject']} L2={'OK %.1fmm' % e['L2_err_mm'] if r['L2'] else r['reject_L2']} "
              f"L1={'%.1f°/%.1fmm' % (e['L1_angle_deg'], e['L1_point_to_axis_mm']) if r['L1'] else '-'} {r['detect_ms']:.0f}ms")
    return 0


def overlay(V, f, det, ev, T, K):
    from PIL import Image, ImageDraw
    od = os.path.join(V, 'd1_overlay')
    os.makedirs(od, exist_ok=True)
    im = Image.open(os.path.join(V, 'frames', f'f{f["n"]:02d}_rgb.png')).convert('RGB')
    dr = ImageDraw.Draw(im)
    if det['L0']:
        for p in det['L0']['pts_sample']:
            uv, _ = project(T, K, np.array(p))
            if uv is not None:
                dr.point((float(uv[0]), float(uv[1])), fill=(255, 140, 0))
    if det['L1']:
        p0, ax = np.array(det['L1']['p0']), np.array(det['L1']['axis'])
        a_, _ = project(T, K, p0 - ax * 0.12)
        b_, _ = project(T, K, p0 + ax * 0.12)
        if a_ is not None and b_ is not None:
            dr.line([tuple(a_), tuple(b_)], fill=(0, 200, 255), width=1)
    if det['L2']:
        uv, _ = project(T, K, np.array(det['L2']['center']))
        if uv is not None:
            dr.ellipse([uv[0] - 5, uv[1] - 5, uv[0] + 5, uv[1] + 5], outline=(0, 255, 0), width=2)
    st = ('L2 %.1f mm' % ev['L2_err_mm']) if det['L2'] else (det['reject'] or det['reject_L2'])
    dr.rectangle([0, 0, 640, 18], fill=(0, 0, 0))
    dr.text((4, 3), f"t={f['src_record_t']:.2f}  d={ev['dist_m']:.2f} m  {st}", fill=(255, 255, 255))
    im.save(os.path.join(od, f'f{f["n"]:02d}.png'))


if __name__ == '__main__':
    sys.exit(main())
