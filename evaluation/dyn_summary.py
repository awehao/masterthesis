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
            meta = json.load(open(traj[:-5] + '_meta.json'))
            if row['path_m'] and meta.get('path_len_m'):
                row['path_ratio'] = round(row['path_m'] / meta['path_len_m'], 3)       # 實際路長／名目 A* 路長（繞路程度）
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
                                          time_ok_mean=round(float(np.mean([r['time_s'] for r in ok if r['time_s']])), 1) if ok else None,
                                          path_ratio_ok_median=round(float(np.median([r['path_ratio'] for r in ok if r.get('path_ratio')])), 3) if any(r.get('path_ratio') for r in ok) else None)
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
            from scipy.stats import wilcoxon
            p = float(wilcoxon(d).pvalue) if len(d) >= 6 and any(abs(x) > 1e-9 for x in d) else None
            S[f'paired_time_{meth}_minus_gmpc'] = dict(n=len(d), mean=round(float(np.mean(d)), 1), median=round(float(np.median(d)), 1),
                                                       n_faster=sum(1 for x in d if x < 0), wilcoxon_p=round(p, 4) if p is not None else None)
        # 成對二元（同案例）：碰到行人與否、到達與否 ⇒ McNemar 精確檢定（只看不一致的案例）
        from scipy.stats import binomtest
        for key, fn in (('contact', lambda r: (r.get('n_contact_dyn') or 0) > 0), ('success', lambda r: bool(r['success']))):
            b = c_ = 0
            for c in cases:
                g = [r for r in rows if (r['speed'], r['pattern'], r['route']) == c and r['method'] == 'gmpc_scan']
                o = [r for r in rows if (r['speed'], r['pattern'], r['route']) == c and r['method'] == meth]
                if g and o:
                    gv, ov = fn(g[0]), fn(o[0])
                    b += int(gv and not ov); c_ += int(ov and not gv)
            S[f'paired_{key}_{meth}_vs_gmpc'] = dict(gmpc_only=b, other_only=c_,
                                                     mcnemar_p=round(float(binomtest(b, b + c_, 0.5).pvalue), 4) if b + c_ else None)
    return S


COL = dict(gmpc_scan='#2b6cb0', mppi='#c05621', dwb='#2f855a', rpp='#8a8f98')
COND = [('walk', 'const', '步行・等速'), ('walk', 'stopgo', '步行・停走'), ('fast', 'const', '快速・等速'), ('fast', 'stopgo', '快速・停走')]


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    M = list(BAGTAG)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8), gridspec_kw=dict(width_ratios=[1.4, 1, 1, 1]))
    w = 0.2
    conds = [c for c in COND if any(r['speed'] == c[0] and r['pattern'] == c[1] for r in rows)]
    for j, m in enumerate(M):
        ys, txt = [], []
        for sp, pt, _ in conds:
            x = [r for r in rows if r['method'] == m and r['speed'] == sp and r['pattern'] == pt]
            ys.append(100.0 * sum(1 for r in x if (r.get('n_contact_dyn') or 0) > 0) / len(x) if x else np.nan)
            txt.append('%d/%d' % (sum(1 for r in x if r['success']), len(x)))
        xs = np.arange(len(conds)) + (j - 1.5) * w
        ax[0].bar(xs, ys, w, color=COL[m], label=NAME[m])
        for xx, yy, tt in zip(xs, ys, txt):
            if np.isfinite(yy):
                ax[0].text(xx, yy + 1.5, tt, ha='center', va='bottom', fontsize=6.5, rotation=90)
    ax[0].set_xticks(range(len(conds))); ax[0].set_xticklabels([c[2] for c in conds])
    ax[0].set_ylabel('有碰到行人的趟數比例（%）'); ax[0].set_title('與行人碰撞（柱上＝到達／總趟數）', fontsize=10)
    ax[0].legend(frameon=False, fontsize=8, ncol=4, loc='upper left'); ax[0].set_ylim(0, 115)
    for k, (key, lab, ttl) in enumerate((('dmin_dyn_m', '對行人最小表面距離（m，<0＝碰撞）', '最近距離（全部情境）'),
                                         ('t_d05_s', '行人 0.5 m 內時間（s）', '貼近行人的時間'))):
        data = [[r[key] for r in rows if r['method'] == m and key in r] for m in M]
        bp = ax[1 + k].boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for b, m in zip(bp['boxes'], M):
            b.set_facecolor(COL[m]); b.set_alpha(0.35); b.set_edgecolor(COL[m])
        for i, (d, m) in enumerate(zip(data, M)):
            ax[1 + k].scatter(np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-0.18, 0.18, len(d)), d, s=10, color=COL[m], alpha=0.8, zorder=3)
        ax[1 + k].set_xticks(range(1, len(M) + 1)); ax[1 + k].set_xticklabels([NAME[m] for m in M], fontsize=8)
        ax[1 + k].set_ylabel(lab); ax[1 + k].set_title(ttl, fontsize=10)
        if key == 'dmin_dyn_m':
            ax[1 + k].axhline(0, color='#1a1a1a', lw=0.8, ls='--')
    cases = sorted({(r['speed'], r['pattern'], r['route']) for r in rows})
    for i, m in enumerate(('mppi', 'dwb', 'rpp')):
        d = []
        for c in cases:
            g = [r for r in rows if (r['speed'], r['pattern'], r['route']) == c and r['method'] == 'gmpc_scan' and r['success']]
            o = [r for r in rows if (r['speed'], r['pattern'], r['route']) == c and r['method'] == m and r['success']]
            if g and o:
                d.append(o[0]['time_s'] - g[0]['time_s'])
        ax[3].scatter(np.full(len(d), i) + np.random.default_rng(10 + i).uniform(-0.15, 0.15, len(d)), d, s=12, color=COL[m])
        if d:
            ax[3].plot([i - 0.25, i + 0.25], [np.median(d)] * 2, color='#1a1a1a', lw=1.6)
            ax[3].text(i, max(d) + 2, '中位 %+.1f s\nn=%d' % (np.median(d), len(d)), ha='center', va='bottom', fontsize=7.5)
    ax[3].axhline(0, color='#1a1a1a', lw=0.8, ls='--')
    lo_, hi_ = ax[3].get_ylim(); ax[3].set_ylim(lo_, hi_ + 0.25 * (hi_ - lo_))   # 中位數標註留空間，不壓標題
    ax[3].set_xticks(range(3)); ax[3].set_xticklabels(['MPPI', 'DWB', 'RPP']); ax[3].set_ylabel('到達時間 − GMPC（s，同案例）')
    ax[3].set_title('到達時間（兩者都到達的案例）', fontsize=10)
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(out, dpi=130)


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
    if '--png' in av:
        plot(rows, av[av.index('--png') + 1])


if __name__ == '__main__':
    main()
