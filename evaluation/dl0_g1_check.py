#!/usr/bin/env python3
"""DL0：G1（真值輔助遮罩參考＋同一深度幾何）vs G0（凍結 D1 幾何基線）在既有開發資料上的核對。只讀既有擷取，不跑模擬。

G1 不是「完美遮罩上限」：遮罩由 dl0_autolabel（真值圓柱 × 實測深度）產生，端蓋未納入、受深度容差與近裁切影響。
幾何端與 D1 相同：遮罩像素 → 世界點（同 D1 的 stride 與 voxel 降採樣）→ D1.fit_cylinder（固定半徑 13 mm）→
同一組候選篩選（內點數、軸向覆蓋、|軸·z|、殘差；**寬度篩選需前板平面，G1 沒有平面 ⇒ 略過**）→
**D1 的 L2 端點規則原樣**（端點距邊界、端外探針看到背景、兩端距離與先驗長度一致）。
G0 讀 D1 既有逐格結果（d1_detect_rev2.json／d1_detect_hold.json），不重跑。

    python3 evaluation/dl0_g1_check.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d1_handle_detect as D1          # noqa: E402  凍結版：只呼叫，不改
import dl0_autolabel as AL            # noqa: E402

DEV = [('d1_dev_f02P/wrist_v0', 'd1_detect_rev2.json', 'traj_wg4b_f02_P'),
       ('d1_hold_mt02P/wrist_v0', 'd1_detect_hold.json', 'traj_mt_b1_02_P')]
BINS = [(0.1, 0.4), (0.4, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 99.0)]


def g1_detect(dep, K, T, mask, rng):
    P = D1.P
    out = {'L0': None, 'L1': None, 'L2': None, 'reject': None, 'reject_L2': None}
    v, u = np.nonzero(mask)
    keep = (v % P['stride'] == 0) & (u % P['stride'] == 0)
    v, u = v[keep], u[keep]
    z = dep[v, u]
    ok = np.isfinite(z) & (z >= D1.ZMIN) & (z <= D1.ZMAX)
    u, v, z = u[ok], v[ok], z[ok]
    if len(z) < P['cluster_min']:
        out['reject'] = 'mask_too_small'
        return out
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    X = pc @ T[:3, :3].T + T[:3, 3]
    _, k = np.unique(np.floor(X / P['voxel']).astype(np.int64), axis=0, return_index=True)
    X = X[np.sort(k)]
    f = D1.fit_cylinder(X, rng) if len(X) >= P['cluster_min'] else None
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
        uv, _ = D1.project(T, K, pe)
        if uv is None or not (P['border_px'] < uv[0] < dep.shape[1] - P['border_px']
                              and P['border_px'] < uv[1] < dep.shape[0] - P['border_px']):
            why.append('end_near_border')
            continue
        probe = c['p0'] + c['ax'] * (t_end + sgn * P['end_probe']) + toward * D1.R_BAR
        uvp, zp = D1.project(T, K, probe)
        if uvp is None or not (0 <= uvp[0] < dep.shape[1] and 0 <= uvp[1] < dep.shape[0]):
            why.append('probe_outside')
            continue
        zo = float(dep[int(uvp[1]), int(uvp[0])])
        if not np.isfinite(zo) or zo < D1.ZMIN:
            why.append('probe_no_depth')
            continue
        if zo < zp + P['end_bg']:
            why.append('probe_not_background')
            continue
        ends.append(pe)
    if len(ends) == 2:
        L = float(np.linalg.norm(ends[1] - ends[0]))
        if abs(L - D1.LEN_BAR) <= P['len_tol'] * D1.LEN_BAR:
            out['L2'] = {'center': ((ends[0] + ends[1]) / 2).tolist(), 'length_m': L}
        else:
            why.append(f'length_inconsistent_{L:.3f}')
    if out['L2'] is None:
        out['reject_L2'] = 'center_unobservable:' + ','.join(why)
    return out


def summarize(rows, key):
    res = {}
    for lo, hi in BINS:
        r = [x for x in rows if lo <= x['dist'] < hi]
        if not r:
            continue
        l2 = [x for x in r if x[key]['L2']]
        e = [x[key + '_err'] for x in l2 if x[key + '_err'] is not None]
        rej = {}
        for x in r:
            k = x[key]['reject'] or ((x[key]['reject_L2'] or '').split(':')[0] or 'L2_ok')
            rej[k] = rej.get(k, 0) + 1
        res[f'{lo}-{hi}'] = {'n': len(r), 'L2': len(l2), 'L2_rate': round(len(l2) / len(r), 3),
                             'err_mm': (None if not e else {'median': round(float(np.median(e)), 2),
                                                             'p95': round(float(np.percentile(e, 95)), 2),
                                                             'max': round(float(max(e)), 2)}),
                             'L2_violation': sum(1 for x in r if x[key + '_viol']),
                             'L2_wrong_gt30mm': sum(1 for x in l2 if (x[key + '_err'] or 0) > 30), 'reasons': rej}
    return res


def main():
    out = {'schema': 'dl0_g1_check/1', 'G1_定義': 'G1＝真值輔助遮罩參考（非完美遮罩上限）＋同一深度幾何；寬度篩選略過（無前板平面）',
           'groups': {}}
    for cap, g0file, grp in DEV:
        D = os.path.join(HERE, 'runs', cap)
        g0 = {r['n']: r for r in json.load(open(os.path.join(D, g0file)))['rows']}
        rng = np.random.default_rng(0)
        rows = []
        for fr in AL.frames_of(cap):
            n = fr['n']
            if n not in g0:
                continue
            dep = AL.load_depth(fr['depth_path'])
            lab = AL.label(fr, depth=dep)
            det = g1_detect(dep, fr['K'], fr['T'], lab['mask'], rng)
            ev = D1.evaluate(det, {'handle_center_world_at_capture': fr['c']}, fr['T'], fr['K'], AL.AXIS_W)
            r0 = g0[n]
            rows.append({'n': n, 'dist': ev['dist_m'], 'flags': lab['flags'],
                         'G0': {'L2': r0['L2'], 'reject': r0['reject'], 'reject_L2': r0['reject_L2']},
                         'G0_err': r0['eval'].get('L2_err_mm'), 'G0_viol': r0['eval']['L2_violation'],
                         'G1': {'L2': det['L2'], 'reject': det['reject'], 'reject_L2': det['reject_L2']},
                         'G1_err': ev.get('L2_err_mm'), 'G1_viol': ev['L2_violation']})
        out['groups'][grp] = {'capture': cap, 'n_frames': len(rows), 'G0': summarize(rows, 'G0'),
                              'G1': summarize(rows, 'G1')}
        print(grp, json.dumps({k: {b: (v['L2'], v['n'], v['err_mm'] and v['err_mm']['median']) for b, v in out['groups'][grp][k].items()}
                               for k in ('G0', 'G1')}, ensure_ascii=False))
    p = os.path.join(HERE, 'results', 'vision', 'DL0_g1_check.json')
    json.dump(out, open(p, 'w'), ensure_ascii=False, indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print('寫出', p)


if __name__ == '__main__':
    main()
