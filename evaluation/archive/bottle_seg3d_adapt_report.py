#!/usr/bin/env python3
"""有界自適應觀察兩趟的結果（只讀紀錄；Codex 20261101_230248／232233）：每個視點的觀察窗（到達＋0.4 s 起到決策時刻）內，手腕旁路輸出的
完整列數、各旗標、與真值比的中心／瓶頂／支撐／高度誤差（只算該旗標成立的列）、決策時最新列年齡；整趟時序（proc、age、pair_age）與丟幀。真值只用於評估。
    python3 evaluation/bottle_seg3d_adapt_report.py seg3d_ad_z045_p1b seg3d_ad_z075_p2b
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return None if not len(x) else dict(n=int(len(x)), median=round(float(np.median(x)), 2), max=round(float(x.max()), 2))


def one(run):
    D = os.path.join(HERE, 'runs', run)
    rep = json.load(open(os.path.join(D, 'collect_report.json')))
    dec = [json.loads(l) for l in open(os.path.join(D, 'decisions.jsonl')) if 'final' not in json.loads(l)]
    Z = np.load(os.path.join(D, 'seg3d_w1.npz'))
    lb = str(Z['label'])
    fs = ['recv_t'] + lb.split('fields=')[1].split(',')
    L = np.load(os.path.join(D, 'w1_log.npz'))
    lc = {str(k): L['rows'][:, i] for i, k in enumerate(L['cols'])}
    off = float(np.median(lc['stamp'] - lc['task_t']))
    tru = {round(float(s) - off, 3): (float(lc['tb_x'][k]), float(lc['tb_y'][k]), float(lc['tb_z'][k])) for k, s in enumerate(lc['stamp'])}
    R = []
    for r in Z['rows']:
        d = dict(zip(fs, r))
        d['t'] = d['stamp'] - off
        d['truth'] = tru.get(round(d['t'], 3))
        R.append(d)
    views = []
    for x in dec:
        W = [r for r in R if x['t_arrive'] + 0.4 <= r['t'] <= x['t_decide'] and r['truth'] is not None]
        V = [r for r in W if r['valid'] > 0.5]
        e = lambda r, a_, b_: math.hypot(r[a_] - r['truth'][0], r[b_] - r['truth'][1]) * 1e3   # noqa: E731
        views.append(dict(view=x['view'], action=x['action'], arg=x['arg'], deficits=x['deficits'], n=len(W), n_complete=x['n_complete'],
                          xy_err=stats([e(r, 'cx_known_r', 'cy_known_r') for r in V if r['xy_valid'] > 0.5]), sil_err=stats([e(r, 'cx_sil', 'cy_sil') for r in V if r['xy_valid_silhouette'] > 0.5]),
                          top_err=stats([abs(r['z_top'] - r['truth'][2] - 0.2) * 1e3 for r in V if r['top_observed'] > 0.5]),
                          support_err=stats([abs(r['z_support'] - r['truth'][2]) * 1e3 for r in V if r['support_observed'] > 0.5]),
                          height_err=stats([abs(r['height'] - 0.2) * 1e3 for r in V if r['height_valid'] > 0.5]),
                          latest_age_at_decision_ms=round((x['t_decide'] - max(r['t'] for r in W)) * 1e3, 1) if W else None))
    last = dict(zip(fs, Z['rows'][-1])) if len(Z['rows']) else {}
    return dict(run=run, layout=rep['layout'], final=rep['final'], views_tried=rep['views_tried'], abort=rep['abort'], min_clear_m=rep['min_clear_m'], bottle_move_mm=rep.get('bottle_move_m', float('nan')) * 1e3,
                save_errors=rep['save_errors'], views=views, timing=dict(proc_ms=stats([d['proc_ms'] for d in R]), age_ms=stats([d['age_ms'] for d in R]), pair_age_ms=stats([d['pair_age_ms'] for d in R]),
                drops={k: last.get(k) for k in ('n_drop_pair', 'n_drop_latest', 'n_old_complete', 'n_stale')}, monotonic=all(b['stamp'] > a_['stamp'] for a_, b in zip(R, R[1:]))))


def main():
    res = {r: one(r) for r in sys.argv[1:]}
    json.dump(res, open(os.path.join(HERE, 'results', 'vision', 'seg3d_adapt_summary.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    for r, o in res.items():
        print(f"== {r} 桌高 {o['layout']['table_z']} 瓶 {o['layout']['bottle']}：{o['final']}；視點 {o['views_tried']}；中止 {o['abort']}；淨空 {o['min_clear_m']:.3f}；瓶位移 {o['bottle_move_mm']:.1f} mm")
        for v in o['views']:
            print(f"   {v['view']}：{v['action']} {v['arg'] or ''} 缺項 {v['deficits']} 窗 {v['n']} 列、完整 {v['n_complete']}；xy {v['xy_err']} 輪廓 {v['sil_err']} 瓶頂 {v['top_err']} 支撐 {v['support_err']} 高度 {v['height_err']}；決策時最新列 {v['latest_age_at_decision_ms']} ms")
        print(f"   時序 {o['timing']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
