#!/usr/bin/env python3
"""N2 導航消融彙整：n2_summary.py [批次=n2main] [--csv-only] [--json out.json] [--runs-csv out.csv] [--png out.png]

A＝完整方法：直接取 N1 最終結果（n1_summary 的 n1final：15 條降難度路線用 n1r2，其餘 n1main）同布局同種子的 GMPC，不重跑。
B 不用障礙速度（NAV_ZERO_OBS_VEL=1）、C 移除 CBF（CBF_ENABLE=0）、D 移除參考調整（D2_YIELD_SIDE=0）、E 僅速度調整（YS_PRED_SPEED_ONLY=1）：
results/<批次>/<布局>_crossing_k<種子>_<配置>.csv（run_n2_batch.sh）。
逐趟指標與 n1_summary 相同（到達、乾淨到達、最小淨空、時間、繞路比；不加 --csv-only 時另讀 bag：行人遭遇、Shield 介入）；
成對比較（同一路線）：各配置 vs A 的到達／乾淨到達（McNemar 精確）、時間（Wilcoxon）。"""
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import n1_summary as NS                                               # noqa: E402

CFG = ['B', 'C', 'D', 'E']
NAME = dict(A='A 完整方法', B='B 不用障礙速度', C='C 移除 CBF', D='D 移除參考調整', E='E 僅速度調整')


def row_n2(f, batch, csv_only, D=None, cfgs=None):
    m = re.match(r'(n1_\w+?)_(crossing)_k(\d+)_([A-Z]+)\.csv$', os.path.basename(f))
    if not m or m.group(4) not in (cfgs or CFG):
        return None
    lay, cond, k, cf = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    rec = list(csv.DictReader(open(f)))
    if not rec:
        return None
    r = rec[-1]
    D = D or os.path.join(HERE, 'experiments', 'n1_scenarios')
    meta_f = os.path.join(D, f'{lay}_{cond}_k{k}_meta.json')
    meta = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
    succ = r.get('success') == 'True'
    path = NS.fnum(r.get('path_length_m'))
    row = dict(layout=lay, cond=cond, seed=k, method=cf, success=succ, time_s=NS.fnum(r.get('arrival_time_s')) if succ else None,
               path_m=path, path_ratio=round(path / meta['path_len_m'], 3) if path and meta.get('path_len_m') else None,
               min_clr_m=NS.fnum(r.get('min_clearance_m')), n_coll=int(NS.fnum(r.get('collision_count')) or 0))
    row['clean'] = bool(succ and row['n_coll'] == 0)
    if not csv_only:
        bags = sorted(glob.glob(os.path.join(HERE, 'bags', f'gmpc_cbf__scan_{batch}_{cf}_{lay}_{cond}_k{k}_t*_seed{k}')))
        if bags:
            b = bags[-1]
            os.environ['SCENE'] = lay; os.environ['BIGARENA'] = '1'
            os.environ['AMMR_TRAJ_FILE'] = os.path.join(D, f'{lay}_{cond}_k{k}.yaml')
            try:
                import importlib
                import analyze as A
                import dyn_encounter as E
                import yaml
                importlib.reload(A); importlib.reload(E)
                A.load_static_clearance()
                radii = {d['name']: float(d['radius']) for d in yaml.safe_load(open(os.environ['AMMR_TRAJ_FILE']))['dynamic_obstacles']}
                row.update({kk: v for kk, v in E.one(b, radii, meta.get('goal', [0, 0])).items() if kk != 'bag'})
            except Exception as e:                                    # noqa: BLE001
                row['enc_err'] = repr(e)[:80]
            try:
                row.update(NS.shield_stats(b))
            except Exception as e:                                    # noqa: BLE001
                row['shield_err'] = repr(e)[:80]
    return row


def collect(batch='n2main', csv_only=False):
    rows = [x for x in (row_n2(f, batch, csv_only) for f in sorted(glob.glob(os.path.join(HERE, 'results', batch, 'n1_*_crossing_k*_*.csv')))) if x]
    cases = {(r['layout'], r['seed']) for r in rows}
    A = [r for r in NS.collect('n1final', csv_only=csv_only) if r['method'] == 'gmpc_scan' and (r['layout'], r['seed']) in cases]
    return A + rows                                                   # A 列保留 method='gmpc_scan'（NS.summarize 以它為成對基準）


def collect_dev(batch, scen_dir, cfgs=('B',), ref='H', csv_only=False):
    """開發比較（同一批次內含參考配置 ref，例：nfast 的 H＝A 完整方法）：ref 列改名 gmpc_scan 當成對基準。"""
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, 'results', batch, 'n1_*_crossing_k*_*.csv'))):
        r = row_n2(f, batch, csv_only, D=scen_dir, cfgs=list(cfgs) + [ref])
        if r:
            if r['method'] == ref:
                r['method'] = 'gmpc_scan'
            rows.append(r)
    return rows


def summarize(rows, cfgs=None):
    NS.METHODS = ['gmpc_scan'] + list(cfgs or CFG)
    S = NS.summarize(rows)
    for lay in sorted({r['layout'] for r in rows}) + ['all']:
        for meth in NS.METHODS:
            x = [r for r in rows if r['method'] == meth and (lay == 'all' or r['layout'] == lay)]
            if x and f'{lay}|{meth}' in S:
                S[f'{lay}|{meth}']['coll_runs'] = sum(1 for r in x if r['n_coll'] > 0)
    return {k.replace('gmpc_scan', 'A').replace('_vs_gmpc', '_vs_A'): v for k, v in S.items()}


def plot(rows, S, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    M = ['gmpc_scan'] + CFG; LB = dict(gmpc_scan='A 完整', B='B 無障礙速度', C='C 無 CBF', D='D 無參考調整', E='E 僅速度')
    COL = dict(gmpc_scan='#2b6cb0', B='#b7791f', C='#c53030', D='#6b46c1', E='#2f855a')
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))
    for j, m in enumerate(M):
        v = S.get(f"all|{m.replace('gmpc_scan', 'A')}")
        if not v:
            continue
        ax[0].bar(j, 100.0 * v['success'] / v['n'], 0.6, color=COL[m], alpha=0.35)
        ax[0].bar(j, 100.0 * v['clean'] / v['n'], 0.6, color=COL[m])
        ax[0].text(j, 100.0 * v['success'] / v['n'] + 1.5, f"{v['clean']}/{v['n']}", ha='center', fontsize=8)
    ax[0].set_xticks(range(len(M))); ax[0].set_xticklabels([LB[m] for m in M], fontsize=8)
    ax[0].set_ylim(0, 112); ax[0].set_ylabel('百分比（%）'); ax[0].set_title('到達（淡）與乾淨到達（深；柱上＝乾淨／總趟）', fontsize=10)
    for k, (key, lab, ttl) in enumerate((('min_clr_m', '對所有障礙最小淨空（m）', '最小淨空'),
                                         ('dmin_dyn_m', '對行人最小表面距離（m）', '離行人最近距離'),
                                         ('shield_ratio', 'Shield 介入比例', '安全層介入'))):
        data = [[r[key] for r in rows if r['method'] == m and r.get(key) is not None] for m in M]
        bp = ax[1 + k].boxplot([d if d else [np.nan] for d in data], widths=0.55, patch_artist=True, showfliers=False)
        for b, m in zip(bp['boxes'], M):
            b.set_facecolor(COL[m]); b.set_alpha(0.35); b.set_edgecolor(COL[m])
        for i, (d, m) in enumerate(zip(data, M)):
            ax[1 + k].scatter(np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-0.18, 0.18, len(d)), d, s=9, color=COL[m], alpha=0.8, zorder=3)
        ax[1 + k].set_xticks(range(1, len(M) + 1)); ax[1 + k].set_xticklabels([LB[m] for m in M], fontsize=8)
        ax[1 + k].set_ylabel(lab, fontsize=9); ax[1 + k].set_title(ttl, fontsize=10)
        if key != 'shield_ratio':
            ax[1 + k].axhline(0, color='#1a1a1a', lw=0.8, ls='--')
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=130)


def main():
    av = sys.argv
    batch = av[1] if len(av) > 1 and not av[1].startswith('--') else 'n2main'
    if '--dev' in av:                                                 # n2_summary.py nfast --dev <情境資料夾> [--cfgs B]：同批次 H 當 A
        cf = av[av.index('--cfgs') + 1].split(',') if '--cfgs' in av else ['B']
        rows = collect_dev(batch, av[av.index('--dev') + 1], cfgs=cf, csv_only='--csv-only' in av)
        S = summarize(rows, cfgs=cf)
    else:
        rows = collect(batch, csv_only='--csv-only' in av)
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
