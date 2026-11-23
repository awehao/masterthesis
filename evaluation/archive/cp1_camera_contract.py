#!/usr/bin/env python3
"""CP1：相機投影／深度契約核對（既有資料、純離線；規格 results/vision/CP1_camera_contract_spec.md draft-2）。

前板平面（桿心後 40 mm 的前板外面；排除橫桿／支柱附近與邊緣 10 mm）上，固定影格、原始深度值與像素集合，只比較兩種射線約定：
  整數：K⁻¹[u, v, 1]ᵀ      半像素：K⁻¹[u + 0.5, v + 0.5, 1]ᵀ
同一條候選射線上：量測點＝光學 z 深度 × 射線（z 分量 1），參考點＝射線與真值平面交點；報沿射線有號差。
另報：相機座標下前板量測點擬合法向對真值法向的角差（兩約定）、依影像徑向與方向分層、已知點投影的座標一致性檢查。
不改偵測器、不補償、不調半像素量。

    python3 evaluation/cp1_camera_contract.py     # 輸出 results/vision/CP1_camera_contract.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d1_handle_detect as D1                         # noqa: E402

GROUPS = {'traj_wg4b_f02_P': ('runs/d1_dev_f02P', 'd1_detect_rev2.json'),
          'traj_mt_b1_02_P': ('runs/d1_hold_mt02P', 'd1_detect_hold.json')}
STRIDE = 4
N_MIN = 200
OFFS = {'integer': 0.0, 'half_pixel': 0.5}


def frame_stats(dep, K, T, c):
    H, W = dep.shape
    cam, Rc = T[:3, 3], T[:3, :3]
    Kinv = np.linalg.inv(K)
    yp = c[1] + 0.040
    v, u = np.mgrid[0:H:STRIDE, 0:W:STRIDE]
    u, v = u.ravel().astype(float), v.ravel().astype(float)
    z = dep[v.astype(int), u.astype(int)].astype(float)
    ok = np.isfinite(z) & (z >= D1.ZMIN) & (z <= D1.ZMAX)
    res, keep = {}, ok.copy()
    rays = {}
    for name, o in OFFS.items():
        dc = (Kinv @ np.stack([u + o, v + o, np.ones_like(u)])).T          # 相機座標射線（z 分量 1）
        dw = dc @ Rc.T
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (yp - cam[1]) / dw[:, 1]
        p = cam + dw * t[:, None]
        x_, z_ = p[:, 0] - c[0], p[:, 2] - c[2]
        inside = (t > 0) & (np.abs(x_) <= 0.2875 - 0.010) & (z_ >= -0.130 + 0.010) & (z_ <= 0.130 - 0.010) \
            & ~((np.abs(x_) <= 0.11) & (np.abs(z_) <= 0.03))
        keep &= inside
        rays[name] = (dc, dw, t)
    n = int(keep.sum())
    if n < N_MIN:
        return {'n': n, 'why': f'too_few_panel_pixels<{N_MIN}'}
    nrm_true_cam = Rc.T @ np.array([0.0, -1.0, 0.0])                       # 前板外法向（朝相機側）轉相機座標
    r_img = np.hypot(u[keep] - K[0, 2], v[keep] - K[1, 2])
    for name in OFFS:
        dc, dw, t = rays[name]
        nr = np.linalg.norm(dw[keep], axis=1)
        diff = nr * (z[keep] - t[keep])                                     # 同一射線上：量測點 − 真值交點（沿射線）
        Pc = dc[keep] * z[keep][:, None]                                    # 相機座標量測點
        # 法向診斷只用沿射線差與該格中位相差 ≤ 2 mm 的點（排除斜視時溢出排除區的橫桿／支柱遮擋像素）
        sel = np.abs(diff - np.median(diff)) <= 0.002
        Pc = Pc[sel]
        cen = Pc.mean(0)
        nfit = np.linalg.svd(Pc - cen, full_matrices=False)[2][2]
        if nfit @ nrm_true_cam < 0:
            nfit = -nfit
        ang = math.degrees(math.acos(min(1.0, float(nfit @ nrm_true_cam))))
        dv = nfit - nrm_true_cam
        rb = {}
        for lo, hi in ((0, 100), (100, 200), (200, 400)):
            m = (r_img >= lo) & (r_img < hi)
            rb[f'r{lo}-{hi}px'] = {'n': int(m.sum()), 'median_mm': float(np.median(diff[m])) * 1e3 if m.sum() >= 20 else None}
        db = {}
        for key, m in (('u_left', u[keep] < K[0, 2]), ('u_right', u[keep] >= K[0, 2]),
                       ('v_top', v[keep] < K[1, 2]), ('v_bottom', v[keep] >= K[1, 2])):
            db[key] = {'n': int(m.sum()), 'median_mm': float(np.median(diff[m])) * 1e3 if m.sum() >= 20 else None}
        res[name] = {'n_normal_fit': int(sel.sum()), 'n_occluded_like': int((diff < np.median(diff) - 0.002).sum()),
                     'median_mm': float(np.median(diff)) * 1e3, 'p10_mm': float(np.percentile(diff, 10)) * 1e3,
                     'p90_mm': float(np.percentile(diff, 90)) * 1e3, 'normal_angle_deg': ang,
                     'normal_diff_cam_xyz': dv.tolist(), 'radial_bins': rb, 'direction_bins': db}
    res['n'] = n
    return res


def fit(x, y):
    if len(x) < 3:
        return None
    b, a = np.polyfit(x, y, 1)
    r = np.asarray(y) - (a + b * np.asarray(x))
    return {'n': len(x), 'intercept_mm': float(a), 'slope_mm_per_m': float(b), 'resid_sd_mm': float(np.std(r)),
            'median_mm': float(np.median(y))}


def main():
    out_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'vision', 'CP1_camera_contract.json')
    if os.path.exists(out_p):
        raise SystemExit(f'{out_p} 已存在，不覆寫')
    res = {'schema': 'cp1_camera_contract/1', 'note': '既有資料；前板平面；兩種射線約定比較同一批像素與深度值；評估端用真值', 'groups': {}}
    allrows = []
    for grp, (run, fname) in GROUPS.items():
        V = os.path.join(HERE, run, 'wrist_v0')
        m = json.load(open(os.path.join(V, 'meta.json')))
        K = np.array(m['intrinsics_readback']['K'])
        det = {r['n']: r for r in json.load(open(os.path.join(V, fname)))['rows']}
        rows = []
        for f in m['frames']:
            T = D1.cam_T(f)
            dep = np.load(os.path.join(V, 'frames', f'f{f["n"]:02d}_depth_m.npy'))
            dep = dep[:, :, 0] if dep.ndim == 3 else dep
            c = np.array(f['handle_center_world_at_capture'], float)
            rec = {'n': f['n'], 'dist': float(np.linalg.norm(c - T[:3, 3]))}
            rec.update(frame_stats(dep, K, T, c))
            # 座標一致性：真值把手中心與偵測中心的投影像素（兩約定下像素座標相差 0.5，僅報整數約定）
            if det.get(f['n'], {}).get('L2'):
                uv_t, _ = D1.project(T, K, c)
                uv_d, _ = D1.project(T, K, np.array(det[f['n']]['L2']['center']))
                if uv_t is not None and uv_d is not None:
                    rec['center_projection_px'] = {'truth': uv_t.tolist(), 'detected': uv_d.tolist(),
                                                   'diff': (uv_d - uv_t).tolist()}
            rows.append(rec)
        res['groups'][grp] = {'rows': rows}
        allrows += [(grp, r) for r in rows]
    summ = {}
    for scope, sel in (('traj_wg4b_f02_P', lambda g: g == 'traj_wg4b_f02_P'), ('traj_mt_b1_02_P', lambda g: g == 'traj_mt_b1_02_P'),
                       ('combined', lambda g: True)):
        rr = [r for g, r in allrows if sel(g) and 'integer' in r]
        summ[scope] = {name: fit([r['dist'] for r in rr], [r[name]['median_mm'] for r in rr]) for name in OFFS}
        summ[scope]['normal_angle_deg_median'] = {name: float(np.median([r[name]['normal_angle_deg'] for r in rr])) for name in OFFS} if rr else None
        summ[scope]['n_frames_excluded'] = sum(1 for g, r in allrows if sel(g) and 'integer' not in r)
    res['summary'] = summ
    json.dump(res, open(out_p, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
