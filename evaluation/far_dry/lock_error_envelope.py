#!/usr/bin/env python3
"""A3（Codex 20261116）：鎖定參考的「已觀測誤差包絡」——用既有 Isaac 影格、同一個三格鎖定算法（最近 3 格有效、彼此 ≤5 mm、各軸中位數）。
事後以把手真值算鎖定位置誤差；只作明示的工程預算，不是可靠上界。分距離段列出。
    python3 evaluation/far_dry/lock_error_envelope.py OUT.json RUN_ID [RUN_ID ...]"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def T3(p):
    c, s = np.cos(p[2]), np.sin(p[2]); T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; T[:2, 3] = p[:2]
    return T


rows = []
for run in sys.argv[2:]:
    F = json.load(open(os.path.join(HERE, 'runs', run, 'frames_meta.json')))['frames']
    Th = np.array(json.load(open(os.path.join(HERE, 'runs', run, 'truth.json')))['T_handle_start'], float)
    hist = []
    for f in F:
        if not f['valid']:
            hist = []                                                # 中斷後重新累積
            continue
        T = T3(f['odom_base_at_capture']) @ np.array(f['T_base_handle'], float)
        if hist and f['t_obs'] - hist[-1][0] > 0.3:
            hist = []
        hist = (hist + [(f['t_obs'], T)])[-3:]
        if len(hist) < 3:
            continue
        P = np.array([h[1][:3, 3] for h in hist])
        spread = max(np.linalg.norm(P[i] - P[j]) for i in range(3) for j in range(3))
        if spread > 0.005:
            continue
        p = np.median(P, axis=0)
        dR = Th[:3, :3].T @ hist[-1][1][:3, :3]
        ang = float(np.degrees(np.arccos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))))
        rows.append(dict(run=run, seq=f['seq'], range_m=f['eval_only']['range_m'], err_mm=float(np.linalg.norm(p - Th[:3, 3]) * 1e3), err_xyz_mm=[float(x) * 1e3 for x in (p - Th[:3, 3])], spread_mm=float(spread * 1e3), ang_deg=ang))
out = dict(note='三格鎖定算法在既有影格上的事後誤差（對把手真值）；理想里程（擷取同刻位姿＝物理位姿）下的值，含影像估計與擷取同步，不含里程漂移；工程預算用，不是可靠上界', n=len(rows))
for nm, sel in (('near_le_0p5m', lambda r: r['range_m'] <= 0.5), ('far_gt_0p5m', lambda r: r['range_m'] > 0.5)):
    g = [r for r in rows if sel(r)]
    if g:
        e = np.array([r['err_mm'] for r in g]); a = np.array([r['ang_deg'] for r in g])
        out[nm] = dict(n=len(g), runs=sorted({r['run'] for r in g}), err_mm=dict(median=round(float(np.median(e)), 2), p95=round(float(np.percentile(e, 95)), 2), max=round(float(e.max()), 2)), ang_deg=dict(median=round(float(np.median(a)), 3), max=round(float(a.max()), 3)),
                       spread_mm_max=round(max(r['spread_mm'] for r in g), 2), err_xyz_abs_max_mm=[round(float(x), 2) for x in np.abs(np.array([r['err_xyz_mm'] for r in g])).max(axis=0)])
        print(nm, json.dumps(out[nm]))
out['rows'] = rows
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
