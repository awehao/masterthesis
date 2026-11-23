#!/usr/bin/env python3
"""F1 兩瓶視覺採集趟的報告（Codex 20261102_225852）：只讀紀錄，真值只供事後比對。
1 兩瓶各自首次被偵測、xy／支撐有效、選出與連續候選就緒的時刻。2 歧義、合併嫌疑、漏檢、錯選、鎖定切換。3 規劃請求、完成、承接時刻與來源影格年齡。4 實際最晚處置點與承接餘裕；未承接就報未承接。
水平投影重疊不等於真實遮擋：這裡是實際的模型與深度結果。逐實例歸屬＝離兩瓶真值較近且在 MATCH_R 內；是事後評估，不進控制。
    python3 evaluation/bottle_f1_capture_report.py RUN_ID
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MATCH_R = 0.08


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    T0 = float(np.load(os.path.join(D, 'scan_log.npz'))['T0'])
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    b1 = np.array([c['bottle_base_x'][0], c['bottle_base_y'][0]])
    b2 = np.array(r['bottle2']['start'][:2])
    truth = dict(target=b1, other=b2)
    bx = lambda t: float(np.interp(t, c['t'], c['base_x']))   # noqa: E731
    F = [json.loads(l) for l in open(os.path.join(D, 'instances.jsonl'))]
    for f in F:
        f['t'] = f['stamp'] - T0
    first, count = {}, dict(frames=len(F), why={}, amb={}, n_inst={}, size_rejected=0, wrong_selected=0, wrong_lock=None, unassigned_inst=0, missed=dict(target=0, other=0), det=dict(target=0, other=0))

    def mark(key, f):
        if key not in first:
            first[key] = dict(t=round(f['t'], 3), x=round(bx(f['t']), 3))
    errs = dict(target=[], other=[])
    for f in F:
        count['why'][f['why']] = count['why'].get(f['why'], 0) + 1
        for a_ in f.get('amb') or []:
            count['amb'][a_] = count['amb'].get(a_, 0) + 1
        n = len(f['inst'])
        count['n_inst'][str(n)] = count['n_inst'].get(str(n), 0) + 1
        seen = set()
        for x in f['inst']:
            if x['valid'] and not x['qual']:
                count['size_rejected'] += 1
            if x['xy'] is None:
                continue
            d = {k: math.hypot(x['xy'][0] - v[0], x['xy'][1] - v[1]) for k, v in truth.items()}
            k = min(d, key=d.get)
            x['who'] = k if d[k] <= MATCH_R else None
            if x['who'] is None:
                count['unassigned_inst'] += 1
                continue
            seen.add(k)
            mark(f'{k}_detected', f)
            if x['qual']:
                mark(f'{k}_qualified', f)
                if x['xy_valid']:
                    mark(f'{k}_xy_valid', f)
                    errs[k].append(d[k])
                    if x['support']:
                        mark(f'{k}_xy_support', f)
        for k in truth:
            if k in seen:
                count['det'][k] += 1
            else:
                count['missed'][k] += 1
        if f.get('just_locked'):
            sel = f['inst'][f['sel']]
            mark('locked', f)
            first['locked'].update(who=sel.get('who'), center=f['lock'])
        if f.get('sel') is not None:
            sel = f['inst'][f['sel']]
            mark('first_passed', f)
            if sel.get('who') != 'target':
                count['wrong_selected'] += 1
    if F:
        locks = {tuple(f['lock']) for f in F if f.get('lock') is not None}
        count['lock_changes'] = max(len(locks) - 1, 0)
    # 候選
    cand = dict(k_ready=None, snap=None)
    for l in open(os.path.join(D, 'candidates.jsonl')):
        try:
            d = json.loads(l)
        except Exception:
            continue
        o = d.get('out') or d.get('output') or d
        if isinstance(o, dict) and float(o.get('k_ready', 0) or 0) > 0.5 and cand['k_ready'] is None:
            t = float(o['t_eval']) - T0
            cand['k_ready'] = dict(t=round(t, 3), x=round(bx(t), 3), cx=o.get('k_cx'), cy=o.get('k_cy'), z_support=o.get('k_z_support'))
        if isinstance(o, dict) and float(o.get('k_snap_ik', 0) or 0) == 2.0 and float(o.get('k_snap_local', 0) or 0) == 2.0 and cand['snap'] is None:
            t = float(o['t_eval']) - T0
            cand['snap'] = dict(t=round(t, 3), x=round(bx(t), 3), seq=o.get('k_snap_seq'))
    on = r.get('online') or {}
    plans = []
    for g in on.get('grasp_plans') or []:
        plans.append({k: (round(g[k], 4) if isinstance(g.get(k), float) else g.get(k)) for k in ('state', 'seq', 't_req', 'x_req', 't_res', 't_acc', 'x_acc', 'x_latest', 'obs_age_at_accept', 'snap_age_at_accept', 'ok', 'error', 'dx')})
        if g.get('t_acc') is not None:
            plans[-1]['accept_margin_m'] = round(float(g['x_latest']) - float(g['x_acc']), 4)
            plans[-1]['target_err_mm'] = round(math.hypot(g['xy'][0] - b1[0], g['xy'][1] - b1[1]) * 1e3, 2)
            plans[-1]['xy'] = g['xy']
            plans[-1]['zs'] = g['zs']
    accepted = [p for p in plans if p.get('t_acc') is not None]
    ev = {e['what']: e['t'] for e in r['events'] if e['what'] in ('relwin_shift', 'online_obs_first', 'unfold_start', 'blend_start', 'f1_capture_boundary')}
    forb = r.get('forbidden_pulses')
    out = dict(run=run, stop=r.get('stop'), cap=(r.get('f1_capture') or {}).get('cap'), base_x_end=round(float(c['base_x'][-1]), 4), x_blend_final=(r.get('f1_capture') or {}).get('cap', {}).get('x_blend') if (r.get('f1_capture') or {}).get('cap') else None, truth=dict(target=b1.tolist(), other=b2.tolist()),
               bottle2=r.get('bottle2'), target_moved_mm=round(float(np.hypot(c['bottle_base_x'][-1] - b1[0], c['bottle_base_y'][-1] - b1[1])) * 1e3, 3), first=first, counts=count, xy_err_mm={k: (dict(n=len(v), p50=round(float(np.median(v)) * 1e3, 2), max=round(float(np.max(v)) * 1e3, 2)) if v else None) for k, v in errs.items()},
               candidates=cand, plans=plans, accepted=bool(accepted), accept_margin_m=accepted[0]['accept_margin_m'] if accepted else None, latest_x=plans[-1].get('x_latest') if plans else None, events=ev, forbidden_pulses=forb, solve_ms=r.get('solve_ms'), n_over_50ms=r.get('n_over_50ms'),
               note='逐實例歸屬是事後比對真值；未承接就報未承接；n＝1')
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', f'f1_capture_{run}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in out.items() if k not in ('bottle2',)}, ensure_ascii=False, indent=0, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
