#!/usr/bin/env python3
"""BL1：沿接近方向的估計偏差定位（G0；純離線；規格 results/vision/BL1_bias_localization_spec.md draft-2）。

以凍結 d1_handle_detect.detect() 依原順序、原種子逐格重算（與原紀錄逐格一致才納入）；分析端包裝 fit_cylinder()
記錄完整候選群集與呼叫前 RNG 狀態（不改凍結檔、不額外消耗原亂數）。真值只在評估端。只定位、不補償、不調參。

    python3 evaluation/bl1_bias_localization.py      # 輸出 results/vision/BL1_bias_localization.json（已存在則拒絕覆寫）
"""
import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import d1_handle_detect as D1                         # noqa: E402  凍結版
import dl2_eval as EV                                 # noqa: E402
import dl2_obs_to_target as E2                        # noqa: E402
import object_target_geometry as OT                   # noqa: E402

GROUPS = {'traj_wg4b_f02_P': ('runs/d1_dev_f02P', 'd1_detect_rev2.json'),
          'traj_mt_b1_02_P': ('runs/d1_hold_mt02P', 'd1_detect_hold.json')}
A_STAR = np.array([0.0, 1.0, 0.0])                    # 真值接近方向（櫃體 yaw 0）
X_STAR = np.array([1.0, 0.0, 0.0])                    # 真值軸
R = D1.R_BAR
HALF = D1.LEN_BAR / 2
VIS_DEG = 60.0
_orig_fit = D1.fit_cylinder


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, dict):
        return all(same(a.get(k), b.get(k)) for k in set(a) | set(b))
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=1e-12)


def surf_nearest(Pts, c):
    """點到真值圓柱面（無限長）的最近點。"""
    v = Pts - c
    t = v @ X_STAR
    rad = v - np.outer(t, X_STAR)
    n = np.linalg.norm(rad, axis=1, keepdims=True)
    return c + np.outer(t, X_STAR) + R * rad / np.maximum(n, 1e-12)


def ray_hits(T, K, shape, c, stride=2):
    """真值有限圓柱的射線命中（像素、命中點、射線方向、可見面夾角）。"""
    H, W = shape
    cam, Rc = T[:3, 3], T[:3, :3]
    corners = [c + X_STAR * s * HALF + np.array([0, dy, dz]) for s in (-1, 1) for dy in (-R, R) for dz in (-R, R)]
    uv = [D1.project(T, K, p)[0] for p in corners]
    if any(u is None for u in uv):
        return []
    uv = np.array(uv)
    u0, v0 = np.clip(np.floor(uv.min(0)).astype(int) - 2, 0, [W - 1, H - 1])
    u1, v1 = np.clip(np.ceil(uv.max(0)).astype(int) + 2, 0, [W - 1, H - 1])
    out = []
    Kinv = np.linalg.inv(K)
    for v in range(v0, v1 + 1, stride):
        for u in range(u0, u1 + 1, stride):
            dc = Kinv @ np.array([u, v, 1.0])
            d = Rc @ dc
            dn = d / np.linalg.norm(d)
            o = cam - c
            a_ = dn - (dn @ X_STAR) * X_STAR
            b_ = o - (o @ X_STAR) * X_STAR
            A, B, C = a_ @ a_, 2 * a_ @ b_, b_ @ b_ - R * R
            disc = B * B - 4 * A * C
            if A < 1e-12 or disc < 0:
                continue
            t = (-B - math.sqrt(disc)) / (2 * A)
            if t <= 0:
                continue
            p = cam + t * dn
            if abs((p - c) @ X_STAR) > HALF:
                continue
            nrm = (p - c) - ((p - c) @ X_STAR) * X_STAR
            nrm /= np.linalg.norm(nrm)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(-dn @ nrm)))))
            out.append((u, v, p, dn, dc, ang, t))
    return out


def panel_depth(T, K, dep, c, stride=4):
    """D_panel：同一深度語意在抽屜前板（平面 y = c_y + 0.040，即桿心後 40 mm 的前板外面）的有號差；
    排除橫桿與支柱附近（|x| ≤ 0.11 且 |z − c_z| ≤ 0.03）與前板邊緣 10 mm。範圍由資產尺寸推得（drawer_unit_bar26.yaml）。"""
    H, W = dep.shape
    cam, Rc = T[:3, 3], T[:3, :3]
    Kinv = np.linalg.inv(K)
    yp = c[1] + 0.040
    out = []
    for v in range(0, H, stride):
        for u in range(0, W, stride):
            dc = Kinv @ np.array([u, v, 1.0])
            d = Rc @ dc
            if abs(d[1]) < 1e-9:
                continue
            t = (yp - cam[1]) / d[1]
            if t <= 0:
                continue
            p = cam + t * d
            x, z = p[0] - c[0], p[2] - c[2]
            if not (abs(x) <= 0.2875 - 0.010 and -0.130 + 0.010 <= z <= 0.130 - 0.010):
                continue
            if abs(x) <= 0.11 and abs(z) <= 0.03:
                continue
            zm = float(dep[v, u])
            if not np.isfinite(zm) or zm < D1.ZMIN:
                continue
            pm = cam + d * zm
            dn = d / np.linalg.norm(d)
            out.append(float(dn @ (pm - p)))
    return out


def main():
    out_p = os.path.join(HERE, 'results', 'vision', 'BL1_bias_localization.json')
    if len(sys.argv) > 1:
        out_p = sys.argv[1]
    if os.path.exists(out_p):
        raise SystemExit(f'{out_p} 已存在，不覆寫')
    Rg = EV.R_grasp()
    T_HG, a_H = E2.baseline_grasp(Rg)
    dl2 = json.load(open(os.path.join(HERE, 'results', 'vision', 'DL2_dev_eval_r3.json')))
    res = {'schema': 'bl1_bias_localization/1', 'path': 'G0', 'detector_sha256': __import__('hashlib').sha256(
        open(D1.__file__, 'rb').read()).hexdigest(), 'groups': {}}
    for grp, (run, fname) in GROUPS.items():
        V = os.path.join(HERE, run, 'wrist_v0')
        m = json.load(open(os.path.join(V, 'meta.json')))
        K = np.array(m['intrinsics_readback']['K'])
        orig = {r['n']: r for r in json.load(open(os.path.join(V, fname)))['rows']}
        d2 = {r['n']: r for r in dl2['groups'][grp]['rows']}
        rng = np.random.default_rng(0)
        rows = []
        for f in m['frames']:
            T = D1.cam_T(f)
            dep = np.load(os.path.join(V, 'frames', f'f{f["n"]:02d}_depth_m.npy'))
            dep = dep[:, :, 0] if dep.ndim == 3 else dep
            calls = []

            def wrap(C, rng_):
                st = copy.deepcopy(rng_.bit_generator.state)
                r_ = _orig_fit(C, rng_)
                calls.append({'C': C.copy(), 'state': st, 'out': r_})
                return r_
            D1.fit_cylinder = wrap
            try:
                det, _ = D1.detect(dep, K, T, rng)
            finally:
                D1.fit_cylinder = _orig_fit
            o = orig.get(f['n'])
            rec = {'n': f['n'], 'dist': d2.get(f['n'], {}).get('dist')}
            repro = (o is not None and det['reject'] == o['reject'] and det['reject_L2'] == o['reject_L2']
                     and same(det['L1'], o['L1']) and same(det['L2'], o['L2']))
            rec['reproduced'] = bool(repro)
            if not repro:
                rec['why'] = 'not_reproduced'
                rows.append(rec)
                continue
            if not det['L2']:
                rec['why'] = 'no_L2:' + str(det['reject'] or det['reject_L2'])
                rows.append(rec)
                continue
            c_true = np.array(f['handle_center_world_at_capture'], float)
            p0, ax = np.array(det['L1']['p0']), np.array(det['L1']['axis'])
            ax = ax / np.linalg.norm(ax)
            c_hat = np.array(det['L2']['center'])
            tt = X_STAR @ (c_true - p0) / (X_STAR @ ax)
            r_ref = p0 + ax * tt
            eF, eC = float(A_STAR @ (r_ref - c_true)), float(A_STAR @ (c_hat - r_ref))
            cam = T[:3, 3]
            w_hat = (c_true - cam) / np.linalg.norm(c_true - cam)          # 視線方向（相機 → 把手）
            rec.update({'e_F_mm': eF * 1e3, 'e_C_mm': eC * 1e3, 'axis_angle_deg': math.degrees(math.acos(min(1.0, abs(ax @ X_STAR)))),
                        'view_vs_approach_deg': math.degrees(math.acos(float(np.clip(w_hat @ A_STAR, -1, 1)))),
                        'e_F_along_view_mm': float(w_hat @ (r_ref - c_true)) * 1e3})
            x2 = d2.get(f['n'], {}).get('routes', {}).get('G0/N-obs', {})
            if x2.get('ok'):
                Ht = np.eye(4)
                Ht[:3, 3] = c_true
                Tt = OT.object_target(Ht, T_HG, a_H, 0.0)['T_WE']
                e_goal = float(A_STAR @ (np.array(x2['T_WE_s0'])[:3, 3] - Tt[:3, 3]))
                e_ctr = float(A_STAR @ (np.array(x2['T_WH'])[:3, 3] - c_true))
                rec.update({'e_goal_mm': e_goal * 1e3, 'e_T_mm': (e_goal - e_ctr) * 1e3,
                            'closure_mm': (e_goal - (eF + eC + (e_goal - e_ctr))) * 1e3,
                            'dl2_center_matches_L2': bool(np.abs(np.array(x2['T_WH'])[:3, 3] - c_hat).max() < 1e-12)})
            # P：內點（pts_sample）到真值表面最近點的有號向量
            pts = np.array(det['L0']['pts_sample'])
            q = surf_nearest(pts, c_true)
            dv = pts - q
            radial = np.linalg.norm(np.cross(pts - c_true, X_STAR), axis=1) - R
            rec['P'] = {'n': len(pts), 'radial_median_mm': float(np.median(radial)) * 1e3,
                        'along_view_median_mm': float(np.median(dv @ w_hat)) * 1e3,
                        'along_approach_median_mm': float(np.median(dv @ A_STAR)) * 1e3}
            # 固定候選點集下的真值表面替換診斷：找出產生被接受候選的那次呼叫
            hit = None
            for cl in calls:
                if cl['out'] is None:
                    continue
                c3, axc, inl, _ = cl['out']
                axc = axc / np.linalg.norm(axc)
                if abs(abs(axc @ ax) - 1.0) < 1e-12 and np.linalg.norm(np.cross(p0 - c3, axc)) < 1e-9:
                    hit = cl
                    break
            if hit is not None:
                Crep = surf_nearest(hit['C'], c_true)
                rr = np.random.default_rng(0)
                rr.bit_generator.state = copy.deepcopy(hit['state'])
                f2 = _orig_fit(Crep, rr)
                if f2 is not None:
                    c3r, axr, _, _ = f2
                    axr = axr / np.linalg.norm(axr)
                    tr = X_STAR @ (c_true - c3r) / (X_STAR @ axr)
                    rec['replace_diag'] = {'e_F_mm': float(A_STAR @ (c3r + axr * tr - c_true)) * 1e3, 'n_cluster': len(Crep)}
                else:
                    rec['replace_diag'] = {'why': 'fit_none'}
            else:
                rec['replace_diag'] = {'why': 'accepted_call_not_matched'}
            # D：深度／反投影（可見面中央 ±60° 的真值射線命中像素）
            hits = ray_hits(T, K, dep.shape, c_true)
            dd, dda, n_all = [], [], len(hits)
            for u, v, p, dn, dc, ang, t in hits:
                if ang > VIS_DEG:
                    continue
                z = float(dep[v, u])
                if not np.isfinite(z) or z < D1.ZMIN:
                    continue
                pm = cam + (T[:3, :3] @ dc) * z                  # 光學 z 深度反投影到同一射線上的三維點
                dd.append(float(dn @ (pm - p)))
                dda.append(float(A_STAR @ (pm - p)))
            rec['D'] = {'n_hits': n_all, 'n_eval': len(dd), 'along_ray_median_mm': float(np.median(dd)) * 1e3 if dd else None,
                        'along_ray_p10_p90_mm': [float(np.percentile(dd, 10)) * 1e3, float(np.percentile(dd, 90)) * 1e3] if dd else None,
                        'along_approach_median_mm': float(np.median(dda)) * 1e3 if dda else None}
            pdd = panel_depth(T, K, dep, c_true)
            rec['D_panel'] = {'n_eval': len(pdd), 'along_ray_median_mm': float(np.median(pdd)) * 1e3 if pdd else None}
            rows.append(rec)
        res['groups'][grp] = {'n_frames': len(rows), 'n_not_reproduced': sum(1 for r in rows if not r['reproduced']),
                              'rows': rows}
        ok = [r for r in rows if 'e_F_mm' in r]
        print(grp, 'frames', len(rows), 'not_reproduced', res['groups'][grp]['n_not_reproduced'], 'analysed', len(ok))
        if ok:
            med = lambda k: round(float(np.median([r[k] for r in ok if r.get(k) is not None])), 3)
            print('  median e_F', med('e_F_mm'), 'e_C', med('e_C_mm'), 'e_T', med('e_T_mm'), 'e_goal', med('e_goal_mm'),
                  'closure max', round(max(abs(r['closure_mm']) for r in ok if 'closure_mm' in r), 6))
            print('  P radial', round(float(np.median([r['P']['radial_median_mm'] for r in ok])), 3),
                  'D ray', round(float(np.median([r['D']['along_ray_median_mm'] for r in ok if r['D']['along_ray_median_mm'] is not None])), 3),
                  'replace e_F', round(float(np.median([r['replace_diag']['e_F_mm'] for r in ok if 'e_F_mm' in r.get('replace_diag', {})])), 3),
                  'D panel', round(float(np.median([r['D_panel']['along_ray_median_mm'] for r in ok if r['D_panel']['along_ray_median_mm'] is not None])), 3))
    json.dump(res, open(out_p, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print('寫出', out_p)


if __name__ == '__main__':
    main()
