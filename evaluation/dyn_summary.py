#!/usr/bin/env python3
"""導航動態能力批次彙整：dyn_summary.py <批次名> [--png out.png] [--json out.json]
逐趟：analyze.py 的 CSV（到達、時間、路長、含靜態的最小淨空與碰撞）＋ dyn_encounter（只看行人：最小表面距離、主動／被動碰撞、
0.5／1.0 m 內時間、遭遇速度比、TTC）。需在 BIGARENA=1 SCENE=bigarena_ghost 環境下跑（形狀與地圖同跑法）。
彙整：方法 × 速度 × 型態的到達率、與行人碰撞的趟數比例、平均最小距離、到達趟平均時間；四方法都到達的案例另列成對時間差。"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault('BIGARENA', '1')
os.environ.setdefault('SCENE', 'bigarena_ghost')
import analyze as A                                                   # noqa: E402
import dyn_encounter as E                                             # noqa: E402

BAGTAG = dict(gmpc_scan=('gmpc_cbf', 'scan'), mppi=('mppi', 'mppi'), dwb=('dwb', 'dwb'), rpp=('rpp', 'rpp'))
NAME = dict(gmpc_scan='本研究 GMPC', mppi='MPPI', dwb='DWB', rpp='RPP')


def goals():
    return {int(r['seed']): (float(r['goal_x']), float(r['goal_y'])) for r in csv.DictReader(open(os.path.join(HERE, 'results', 'bigarena_poses.csv')))}


def collect(batch):
    R = os.path.join(HERE, 'results', batch)
    D = os.path.join(HERE, 'experiments', 'dyn_scenarios_' + batch)
    G = goals()
    A.load_static_clearance()
    rows = []
    for f in sorted(glob.glob(os.path.join(R, '*_r*_*.csv'))):
        m = re.match(r'(\w+?)_(\w+?)_r(\d+)_(gmpc_scan|mppi|dwb|rpp)\.csv$', os.path.basename(f))
        if not m:
            continue
        sp, pt, rt, meth = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        rec = list(csv.DictReader(open(f)))
        if not rec:
            continue
        r = rec[-1]
        am, tg = BAGTAG[meth]
        bags = sorted(glob.glob(os.path.join(HERE, 'bags', f'{am}__{tg}_{batch}_{sp}_{pt}_r{rt}_t*_seed{rt}')))
        row = dict(method=meth, speed=sp, pattern=pt, route=rt, success=r.get('success') == 'True',
                   time_s=float(r['arrival_time_s']) if r.get('arrival_time_s') not in ('', 'nan', None) else None,
                   path_m=float(r['path_length_m']) if r.get('path_length_m') else None,
                   min_clr_all=float(r['min_clearance_m']) if r.get('min_clearance_m') not in ('', 'nan') else None,
                   coll_all=int(float(r['collision_count'])) if r.get('collision_count') not in ('', 'nan') else None,
                   tries=len(bags))
        if bags:
            traj = os.path.join(D, f'bigarena_route_{sp}_{pt}_r{rt}_k1.yaml')
            import yaml
            radii = {d['name']: float(d['radius']) for d in yaml.safe_load(open(traj))['dynamic_obstacles']}
            try:
                row.update({k: v for k, v in E.one(bags[-1], radii, G[rt]).items() if k != 'bag'})
            except Exception as e:                                  # noqa: BLE001
                row['enc_err'] = repr(e)[:80]
        rows.append(row)
    return rows


def summarize(rows):
    S = {}
    keys = sorted({(r['speed'], r['pattern']) for r in rows})
    for sp, pt in keys + [('all', 'all')]:
        sub = [r for r in rows if (sp == 'all' or (r['speed'] == sp and r['pattern'] == pt))]
        for meth in BAGTAG:
            x = [r for r in sub if r['method'] == meth]
            if not x:
                continue
            ok = [r for r in x if r['success']]
            dc = [r for r in x if (r.get('n_contact_dyn') or 0) > 0]
            S[f'{sp}_{pt}_{meth}'] = dict(n=len(x), success=len(ok), dyn_contact_runs=len(dc),
                                          dyn_contact_active=sum(r.get('n_contact_active') or 0 for r in x),
                                          dyn_contact_passive=sum(r.get('n_contact_passive') or 0 for r in x),
                                          dmin_dyn_mean=round(float(np.mean([r['dmin_dyn_m'] for r in x if 'dmin_dyn_m' in r])), 3) if any('dmin_dyn_m' in r for r in x) else None,
                                          t_d05_mean=round(float(np.mean([r['t_d05_s'] for r in x if 't_d05_s' in r])), 1) if any('t_d05_s' in r for r in x) else None,
                                          time_ok_mean=round(float(np.mean([r['time_s'] for r in ok if r['time_s']])), 1) if ok else None)
    # 四方法都到達的案例：與 GMPC 的成對時間差
    cases = sorted({(r['speed'], r['pattern'], r['route']) for r in rows})
    for meth in ('mppi', 'dwb', 'rpp'):
        d = []
        for c in cases:
            g = [r for r in rows if (r['speed'], r['pattern'], r['route']) == c and r['method'] == 'gmpc_scan' and r['success']]
            o = [r for r in rows if (r['speed'], r['pattern'], r['route']) == c and r['method'] == meth and r['success']]
            if g and o:
                d.append(o[0]['time_s'] - g[0]['time_s'])
        if d:
            S[f'paired_time_{meth}_minus_gmpc'] = dict(n=len(d), mean=round(float(np.mean(d)), 1), median=round(float(np.median(d)), 1))
    return S


def main():
    av = sys.argv
    rows = collect(av[1])
    for r in rows:
        print('%-9s %-4s %-6s r%-2d %-5s t=%s dmin=%s 碰 %s(主%s/被%s) d<.5 %s tries %d' % (
            r['method'], r['speed'], r['pattern'], r['route'], r['success'], r['time_s'] and round(r['time_s'], 1), r.get('dmin_dyn_m'),
            r.get('n_contact_dyn'), r.get('n_contact_active'), r.get('n_contact_passive'), r.get('t_d05_s'), r['tries']))
    S = summarize(rows)
    print(json.dumps(S, ensure_ascii=False, indent=1))
    if '--json' in av:
        json.dump(dict(summary=S, runs=rows), open(av[av.index('--json') + 1], 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
