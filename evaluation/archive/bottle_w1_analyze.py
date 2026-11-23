#!/usr/bin/env python3
"""W1 短程採集結果整理（Codex 20261101_072908 要求：有效估計率、誤差、圓擬合殘差、可見弧段）。
逐格（線上估計，w1_log.npz）：任務時間、相位、真值距離、有效與否、失敗原因、xy 有號誤差、點數、擬合殘差、z_top 可用、處理耗時、資料年齡。
可見弧段（存下的影格 w1_frames.npz，離線）：同一估計函式重估，先核對與線上一致；取估計中心 xy 半徑 30 mm 內、桌面上方 0.02–0.25 m 的點，
算繞瓶軸的角度覆蓋（360° − 最大角間隙）。真值只用來評估。
    python3 evaluation/bottle_w1_analyze.py bottle_w1c_2
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_v1_est as EST                             # noqa: E402
import bottle_v1_reest as RE                            # noqa: E402

WHY = ('ok', 'no_depth', 'no_object_points', 'no_bottle_cluster', 'bad_message')
PH = ['CRUISE', 'PREGRASP', 'INSERT', 'CLOSE', 'LIFT', 'CARRY']


def arc_deg(depth, K, T_wc, TB, c):
    Pc, _ = EST.backproject(depth, K)
    Pw = (T_wc[:3, :3] @ Pc.T).T + T_wc[:3, 3]
    m = (Pw[:, 2] >= TB['top_z'] + EST.ABOVE[0]) & (Pw[:, 2] <= TB['top_z'] + EST.ABOVE[1]) & (np.hypot(Pw[:, 0] - c[0], Pw[:, 1] - c[1]) <= 0.03)
    if m.sum() < 10:
        return None, int(m.sum()), None
    a = np.sort(np.degrees(np.arctan2(Pw[m, 1] - c[1], Pw[m, 0] - c[0])) % 360)
    gaps = np.diff(np.r_[a, a[0] + 360])
    return float(360 - gaps.max()), int(m.sum()), [float(Pw[m, 2].min()), float(Pw[m, 2].max())]


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'bottle_w1c_2'
    D = os.path.join(HERE, 'runs', run)
    Z = np.load(os.path.join(D, 'w1_log.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    rep = json.load(open(os.path.join(D, 's1_report.json')))
    ev = {e['what'] if e['what'] != 'phase' else e['to']: e['t'] for e in rep['events']}
    rows = []
    for i in range(len(c['stamp'])):
        v = c['valid'][i] == 1
        rows.append(dict(t=round(float(c['task_t'][i]), 3), phase=PH[int(c['phase'][i])] if c['phase'][i] >= 0 else '?', dist=round(float(c['dist'][i]), 3), expected=bool(c['expected'][i] > 0),
                         received=bool(c['received'][i] > 0), valid=bool(v), why=WHY[int(c['why'][i])] if np.isfinite(c['why'][i]) and 0 <= c['why'][i] < len(WHY) else None,
                         ex=round(float((c['cx'][i] - c['tb_x'][i]) * 1e3), 3) if v else None, ey=round(float((c['cy'][i] - c['tb_y'][i]) * 1e3), 3) if v else None,
                         n_pts=int(c['n_pts'][i]) if v else None, rms_mm=round(float(c['rms'][i] * 1e3), 3) if v else None, z_top_valid=bool(c['z_top_valid'][i] == 1) if v else None,
                         proc_ms=round(float(c['proc_ms'][i]), 1) if np.isfinite(c['proc_ms'][i]) else None, age_s=round(float(c['age'][i]), 3) if np.isfinite(c['age'][i]) else None,
                         n_valid_px=int(c['n_valid_px'][i])))
    # 可見弧段：存下的影格離線重估
    F = np.load(os.path.join(D, 'w1_frames.npz'))
    TB = RE.table1()
    arcs, same = {}, []
    for k, st in enumerate(F['stamp']):
        e = EST.estimate(F['depth'][k].astype(float), F['K'], F['T_wc'][k], TB)
        i = int(np.argmin(np.abs(c['stamp'] - st)))
        if abs(c['stamp'][i] - st) > 1e-6:
            continue
        if e['valid'] and c['valid'][i] == 1:
            same.append(max(abs(e['center_xy'][0] - c['cx'][i]), abs(e['center_xy'][1] - c['cy'][i])) * 1e3)
        if e['valid']:
            arcs[round(float(c['task_t'][i]), 3)] = arc_deg(F['depth'][k].astype(float), F['K'], F['T_wc'][k], TB, e['center_xy'])
    for r in rows:
        a = arcs.get(r['t'])
        if a is not None:
            r['arc_deg'], r['arc_n'], r['arc_z'] = round(a[0], 1) if a[0] is not None else None, a[1], None if a[2] is None else [round(v, 3) for v in a[2]]
    t_pre, t_ins = ev.get('PREGRASP'), ev.get('INSERT')
    win = [r for r in rows if t_pre is not None and r['t'] >= t_pre - 2.0]
    near = [r for r in rows if r['dist'] <= 0.5]

    def stat(R):
        V = [r for r in R if r['valid']]
        e = [math.hypot(r['ex'], r['ey']) for r in V]
        return dict(n=len(R), valid=len(V), rate=round(len(V) / max(len(R), 1), 3), why={w: sum(1 for r in R if not r['valid'] and r['why'] == w) for w in WHY[1:]},
                    err_mm=None if not e else [round(float(np.percentile(e, q)), 3) for q in (50, 95, 100)],
                    ex_mean=None if not V else round(float(np.mean([r['ex'] for r in V])), 3), ey_mean=None if not V else round(float(np.mean([r['ey'] for r in V])), 3),
                    rms_mm=None if not V else [round(float(np.percentile([r['rms_mm'] for r in V], q)), 3) for q in (50, 95, 100)],
                    arc_deg=None if not any(r.get('arc_deg') for r in V) else [round(min(r['arc_deg'] for r in V if r.get('arc_deg')), 1), round(max(r['arc_deg'] for r in V if r.get('arc_deg')), 1)],
                    t_valid=None if not V else [V[0]['t'], V[-1]['t']], dist=None if not R else [min(r['dist'] for r in R), max(r['dist'] for r in R)])
    out = dict(run=run, events=ev, online_vs_offline_max_mm=max(same) if same else None, n_compared=len(same),
               window_pre2s_to_stop=stat(win), near_0p5m=stat(near), by_phase={p: stat([r for r in rows if r['phase'] == p]) for p in ('CRUISE', 'PREGRASP', 'INSERT')},
               rows=rows, note='線上估計；可見弧段為存下影格離線重估（同函式）；真值只用來評估；深度有效範圍為渲染裁切 0.05–10 m，非實體 D435i 測距能力')
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', f'w1_summary_{run}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != 'rows'}, ensure_ascii=False, indent=1))
    for r in win:
        print(f"{r['t']:6.2f} {r['phase']:8s} 距 {r['dist']:.3f} 有效 {int(r['valid'])} {r['why']:18s} 誤差 {r['ex']},{r['ey']} 點 {r['n_pts']} 殘差 {r['rms_mm']} 頂 {r['z_top_valid']} 弧 {r.get('arc_deg')} 耗時 {r['proc_ms']} 年齡 {r['age_s']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
