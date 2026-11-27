#!/usr/bin/env python3
"""N1 導航外部基線彙整：n1_summary.py <批次名> [--csv-only] [--json out.json] [--png out.png] [--runs-csv out.csv]

逐趟（results/<批次>/<布局>_<條件>_k<種子>_<方法>.csv 的最後一列＝analyze.py 結果）：
  到達、到達時間、路長、繞路比（路長／名目 A* 路長）、對所有障礙（牆、家具、未知靜態、行人真值）的最小淨空與碰撞次數。
不加 --csv-only 時另讀 bag（同 tag 最後一次嘗試）：
  行人遭遇（dyn_encounter：對行人最小表面距離、主動／被動碰撞、0.5 m 內時間、遭遇中速度比）、
  Shield 介入（/shield/diag 第 1 欄＝該拍有改命令；介入比例＝改命令的拍數／總拍數，第 4 欄＝改動量）。
彙整：方法 × 布局的到達率（Wilson 95% 區間）、乾淨到達率（到達且 0 碰撞）、平均最小淨空、到達時間、繞路比、Shield 介入比例；
與 GMPC 的成對比較：時間（兩者都到達，Wilcoxon）、到達與乾淨到達（McNemar 精確）。"""
import csv
import glob
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
METHODS = ['gmpc_scan', 'mppi', 'dwb', 'rpp']
NAME = dict(gmpc_scan='本研究 GMPC（住宅任務版）', mppi='MPPI', dwb='DWB', rpp='RPP')
SHORT = dict(gmpc_scan='GMPC', mppi='MPPI', dwb='DWB', rpp='RPP')
BAGTAG = dict(gmpc_scan=('gmpc_cbf', 'scan'), mppi=('mppi', 'mppi'), dwb=('dwb', 'dwb'), rpp=('rpp', 'rpp'))
LAYN = dict(n1_bigarena='多房間', n1_office='辦公室', n1_warehouse='倉儲')


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def fnum(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def shield_stats(bag):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import Float32MultiArray
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    r.set_filter(rosbag2_py.StorageFilter(topics=['/shield/diag']))
    act, dv = [], []
    while r.has_next():
        _, d, _ = r.read_next()
        m = deserialize_message(d, Float32MultiArray).data
        if len(m) > 4:
            act.append(m[1] > 0.5); dv.append(float(m[4]))
    if not act:
        return {}
    a = np.array(act); v = np.array(dv)
    return dict(shield_ratio=round(float(a.mean()), 4), shield_dv_mean=round(float(v[a].mean()), 4) if a.any() else 0.0, shield_n=int(a.size))


def _sources(batch):
    """(結果檔, 來源批次, 情境資料夾) 清單。
    n1final＝N1 最終（11/28 Howard）：results/plan1127/n1_r2_routes.txt 的 15 條用 n1r2（降難度情境 n1_scenarios），其餘 45 條用 n1main；
    n1main 單獨分析時讀第一輪情境 n1_scenarios_r1（那 15 條的情境 11/28 已換成第二版）。"""
    E = os.path.join(HERE, 'experiments')
    r2f = os.path.join(HERE, 'results', 'plan1127', 'n1_r2_routes.txt')
    r2 = set(open(r2f).read().split()) if os.path.exists(r2f) else set()
    def key(f):
        m = re.match(r'(n1_\w+?)_(?:static|crossing)_k(\d+)_', os.path.basename(f))
        return f'{m.group(1)}:{int(m.group(2))}' if m else ''
    def files(b):
        return sorted(glob.glob(os.path.join(HERE, 'results', b, 'n1_*_k*_*.csv')))
    if batch == 'n1final':
        return ([(f, 'n1main', os.path.join(E, 'n1_scenarios')) for f in files('n1main') if key(f) not in r2]
                + [(f, 'n1r2', os.path.join(E, 'n1_scenarios')) for f in files('n1r2') if key(f) in r2])
    d = os.path.join(E, 'n1_scenarios_r1') if batch == 'n1main' and os.path.isdir(os.path.join(E, 'n1_scenarios_r1')) else os.path.join(E, 'n1_scenarios')
    return [(f, batch, d) for f in files(batch)]


def collect(batch, csv_only=False):
    rows = []
    for f, batch, D in _sources(batch):
        m = re.match(r'(n1_\w+?)_(static|crossing)_k(\d+)_(gmpc_scan|mppi|dwb|rpp)\.csv$', os.path.basename(f))
        if not m:
            continue
        lay, cond, k, meth = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        rec = list(csv.DictReader(open(f)))
        if not rec:
            continue
        r = rec[-1]
        meta_f = os.path.join(D, f'{lay}_{cond}_k{k}_meta.json')
        meta = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
        succ = r.get('success') == 'True'
        path = fnum(r.get('path_length_m'))
        row = dict(layout=lay, cond=cond, seed=k, method=meth, success=succ, time_s=fnum(r.get('arrival_time_s')) if succ else None,
                   path_m=path, path_ratio=round(path / meta['path_len_m'], 3) if path and meta.get('path_len_m') else None,
                   min_clr_m=fnum(r.get('min_clearance_m')), n_coll=int(fnum(r.get('collision_count')) or 0))
        row['clean'] = bool(succ and row['n_coll'] == 0)
        if not csv_only:
            am, tg = BAGTAG[meth]
            bags = sorted(glob.glob(os.path.join(HERE, 'bags', f'{am}__{tg}_{batch}_{lay}_{cond}_k{k}_t*_seed{k}')))
            if bags:
                b = bags[-1]
                os.environ['SCENE'] = lay; os.environ['BIGARENA'] = '1'
                os.environ['AMMR_TRAJ_FILE'] = os.path.join(D, f'{lay}_{cond}_k{k}.yaml')
                try:
                    import importlib
                    import analyze as A
                    import dyn_encounter as E
                    importlib.reload(A); importlib.reload(E)
                    A.load_static_clearance()
                    import yaml
                    radii = {d['name']: float(d['radius']) for d in yaml.safe_load(open(os.environ['AMMR_TRAJ_FILE']))['dynamic_obstacles']}
                    row.update({kk: v for kk, v in E.one(b, radii, meta.get('goal', [0, 0])).items() if kk != 'bag'})
                except Exception as e:                                # noqa: BLE001
                    row['enc_err'] = repr(e)[:80]
                try:
                    row.update(shield_stats(b))
                except Exception as e:                                # noqa: BLE001
                    row['shield_err'] = repr(e)[:80]
        rows.append(row)
    return rows


def summarize(rows):
    S = {}
    lays = sorted({r['layout'] for r in rows})
    for lay in lays + ['all']:
        for meth in METHODS:
            x = [r for r in rows if r['method'] == meth and (lay == 'all' or r['layout'] == lay)]
            if not x:
                continue
            ok = [r for r in x if r['success']]; cl = [r for r in x if r['clean']]
            mean = lambda k, xs: round(float(np.mean([r[k] for r in xs if r.get(k) is not None])), 3) if any(r.get(k) is not None for r in xs) else None   # noqa: E731
            S[f'{lay}|{meth}'] = dict(n=len(x), success=len(ok), success_ci=wilson(len(ok), len(x)), clean=len(cl), clean_ci=wilson(len(cl), len(x)),
                                      min_clr_mean=mean('min_clr_m', x), time_mean=mean('time_s', ok),
                                      path_ratio_median=round(float(np.median([r['path_ratio'] for r in ok if r.get('path_ratio')])), 3) if any(r.get('path_ratio') for r in ok) else None,
                                      shield_ratio_mean=mean('shield_ratio', x), dmin_dyn_mean=mean('dmin_dyn_m', x), t_d05_mean=mean('t_d05_s', x))
    from scipy.stats import binomtest, wilcoxon
    key = lambda r: (r['layout'], r['cond'], r['seed'])                # noqa: E731
    G = {key(r): r for r in rows if r['method'] == 'gmpc_scan'}
    for meth in METHODS[1:]:
        O = {key(r): r for r in rows if r['method'] == meth}
        common = sorted(set(G) & set(O))
        d = [O[c]['time_s'] - G[c]['time_s'] for c in common if G[c]['success'] and O[c]['success']]
        out = dict(n_pairs=len(common), time_n=len(d))
        if d:
            out.update(time_mean=round(float(np.mean(d)), 1), time_median=round(float(np.median(d)), 1), other_faster=sum(1 for v in d if v < 0))
            if len(d) >= 6 and any(abs(v) > 1e-9 for v in d):
                out['time_wilcoxon_p'] = round(float(wilcoxon(d).pvalue), 4)
        for k2 in ('success', 'clean'):
            b = sum(1 for c in common if G[c][k2] and not O[c][k2]); c_ = sum(1 for c in common if O[c][k2] and not G[c][k2])
            out[f'{k2}_gmpc_only'] = b; out[f'{k2}_other_only'] = c_
            out[f'{k2}_mcnemar_p'] = round(float(binomtest(b, b + c_, 0.5).pvalue), 4) if b + c_ else None
        S[f'paired|{meth}_vs_gmpc'] = out
    return S


def plot(rows, S, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    COL = dict(gmpc_scan='#2b6cb0', mppi='#c05621', dwb='#2f855a', rpp='#8a8f98')
    lays = sorted({r['layout'] for r in rows})
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.0), gridspec_kw=dict(width_ratios=[1.5, 1, 1, 1]))
    w = 0.2
    for j, m in enumerate(METHODS):
        xs = np.arange(len(lays) + 1) + (j - 1.5) * w
        ys = [100.0 * S[f'{l}|{m}']['clean'] / S[f'{l}|{m}']['n'] if f'{l}|{m}' in S else np.nan for l in lays + ['all']]
        ys2 = [100.0 * S[f'{l}|{m}']['success'] / S[f'{l}|{m}']['n'] if f'{l}|{m}' in S else np.nan for l in lays + ['all']]
        ax[0].bar(xs, ys2, w, color=COL[m], alpha=0.35)
        ax[0].bar(xs, ys, w, color=COL[m], label=SHORT[m])
        for xx, (l) in zip(xs, lays + ['all']):
            s_ = S.get(f'{l}|{m}')
            if s_:
                ax[0].text(xx, 101, f"{s_['clean']}/{s_['n']}", ha='center', va='bottom', fontsize=6.5, rotation=90)
    ax[0].set_xticks(range(len(lays) + 1)); ax[0].set_xticklabels([LAYN.get(l, l) for l in lays] + ['全部'])
    ax[0].set_ylabel('百分比（%）'); ax[0].set_ylim(0, 125)
    ax[0].set_title('到達率（淡色）與乾淨到達率（深色＝到達且 0 碰撞；柱上＝乾淨／總趟）', fontsize=10)
    ax[0].legend(frameon=False, fontsize=8, ncol=4, loc='upper left')
    for k, (key, lab, ttl) in enumerate((('min_clr_m', '對所有障礙最小淨空（m，<0＝碰撞）', '最小淨空'),
                                         ('shield_ratio', 'Shield 介入比例（改命令的拍數比例）', '安全層介入'))):
        data = [[r[key] for r in rows if r['method'] == m and r.get(key) is not None] for m in METHODS]
        if not any(data):
            continue
        bp = ax[1 + k].boxplot([d if d else [np.nan] for d in data], widths=0.55, patch_artist=True, showfliers=False)
        for b, m in zip(bp['boxes'], METHODS):
            b.set_facecolor(COL[m]); b.set_alpha(0.35); b.set_edgecolor(COL[m])
        for i, (d, m) in enumerate(zip(data, METHODS)):
            ax[1 + k].scatter(np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-0.18, 0.18, len(d)), d, s=9, color=COL[m], alpha=0.8, zorder=3)
        ax[1 + k].set_xticks(range(1, 5)); ax[1 + k].set_xticklabels([SHORT[m] for m in METHODS])
        ax[1 + k].set_ylabel(lab, fontsize=9); ax[1 + k].set_title(ttl, fontsize=10)
        if key == 'min_clr_m':
            ax[1 + k].axhline(0, color='#1a1a1a', lw=0.8, ls='--')
    for i, m in enumerate(METHODS[1:]):
        P = S.get(f'paired|{m}_vs_gmpc', {})
        key = lambda r: (r['layout'], r['cond'], r['seed'])             # noqa: E731
        G = {key(r): r for r in rows if r['method'] == 'gmpc_scan' and r['success']}
        d = [r['time_s'] - G[key(r)]['time_s'] for r in rows if r['method'] == m and r['success'] and key(r) in G]
        ax[3].scatter(np.full(len(d), i) + np.random.default_rng(10 + i).uniform(-0.15, 0.15, len(d)), d, s=11, color=COL[m])
        if d:
            ax[3].plot([i - 0.25, i + 0.25], [np.median(d)] * 2, color='#1a1a1a', lw=1.6)
            ax[3].text(i, max(d) + 2, '中位 %+.1f s\nn=%d' % (np.median(d), len(d)) + (f"\np={P['time_wilcoxon_p']}" if 'time_wilcoxon_p' in P else ''),
                       ha='center', va='bottom', fontsize=7.5)
    ax[3].axhline(0, color='#1a1a1a', lw=0.8, ls='--')
    lo_, hi_ = ax[3].get_ylim(); ax[3].set_ylim(lo_, hi_ + 0.3 * (hi_ - lo_))
    ax[3].set_xticks(range(3)); ax[3].set_xticklabels(['MPPI', 'DWB', 'RPP']); ax[3].set_ylabel('到達時間 − GMPC（s，同一路線）')
    ax[3].set_title('到達時間（兩者都到達的路線）', fontsize=10)
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=130)


def main():
    av = sys.argv
    rows = collect(av[1], csv_only='--csv-only' in av)
    S = summarize(rows)
    for k, v in S.items():
        print(k, v)
    if '--json' in av:
        json.dump(dict(summary=S, runs=rows), open(av[av.index('--json') + 1], 'w'), ensure_ascii=False, indent=1)
    if '--runs-csv' in av:
        keys = sorted({k for r in rows for k in r})
        with open(av[av.index('--runs-csv') + 1], 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]
    if '--png' in av:
        plot(rows, S, av[av.index('--png') + 1])


if __name__ == '__main__':
    main()
