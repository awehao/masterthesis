#!/usr/bin/env python3
"""有行人 vs 無行人種子批次：python3 seed_batch_walkers.py --walk 169-189 --nowalk 142-168 [--json out.json] [--png out.png]
逐趟指標沿用 seed_batch_analyze.analyze（含行人中心距離、會車層動作、側向速度佔比）。
圖：(1) 有行人各種子成對整趟時間（虛線＝同種子同組無行人平均）；(2) 回程關門段起點時刻 vs 該段時間（行人路線與時刻固定 ⇒ 抵達門口時刻決定是否遇行人）；
(3) 各段「有行人 − 無行人」平均增量（同種子同組配對）。只用成功趟。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seed_batch_analyze as A                                        # noqa: E402

SEGS = ['門前', '開門穿門到茶几放盒', '抽屜取筆放盒', '關抽屜', '回程關門', '回起點']
THR_S = 253                                                           # 圖示用：行人 l2 經過回程門口的時刻（觀察值，非參數）
SEG_LBL = ['門前', '開門→茶几', '抽屜取筆', '關抽屜', '回程關門', '回起點']


def _runs(spec):
    out = []
    for a in spec.split(','):
        if '-' in a:
            lo, hi = a.split('-'); out += list(range(int(lo), int(hi) + 1))
        elif a:
            out.append(int(a))
    return out


def _mean(res, seed, mode, key, seg=None):
    v = [(r['seg_s'].get(seg) if seg else r.get(key)) for r in res if r.get('verdict') == 'OK' and r['seed'] == seed and r['mode'] == mode]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else np.nan


def summarize(walk, nowalk):
    okw = [r for r in walk if r.get('verdict') == 'OK' and r.get('mission_s')]
    seeds = sorted({r['seed'] for r in okw if any(x['seed'] == r['seed'] and x['mode'] == 'M' for x in okw) and any(x['seed'] == r['seed'] and x['mode'] == 'P' for x in okw)})
    d = [_mean(walk, s, 'M', 'mission_s') - _mean(walk, s, 'P', 'mission_s') for s in seeds]
    S = dict(paired_seeds=seeds, diff_M_minus_P=[round(x, 1) for x in d])
    if d:
        S.update(diff_mean=round(float(np.mean(d)), 1), diff_sd=round(float(np.std(d, ddof=1)), 1) if len(d) > 1 else None)
        try:
            from scipy.stats import wilcoxon
            S['wilcoxon_p_two_sided'] = round(float(wilcoxon(d).pvalue), 4) if len(d) >= 2 else None
        except Exception:                                            # noqa: BLE001
            S['wilcoxon_p_two_sided'] = None
    for m in 'MP':
        tw = [_mean(walk, s, m, 'mission_s') for s in seeds]; tn = [_mean(nowalk, s, m, 'mission_s') for s in seeds]
        S['walk_cost_%s' % m] = [round(a - b, 1) if np.isfinite(a) and np.isfinite(b) else None for a, b in zip(tw, tn)]
        S['seg_cost_%s' % m] = {g: round(float(np.nanmean([_mean(walk, s, m, None, g) - _mean(nowalk, s, m, None, g) for s in seeds])), 1) if seeds else None for g in SEGS}
    for m in 'MP':
        n_all = [r for r in walk if r['mode'] == m]
        S['success_%s' % m] = '%d/%d' % (sum(1 for r in n_all if r.get('verdict') == 'OK'), len(n_all))
    for k in ('walk_dmin', 'walk_near_s', 'walk_close_s', 'ys_n', 'ys_lat', 'ys_behind', 'ys_slow', 'near_lat_frac', 'all_lat_frac'):
        for m in 'MP':
            v = [r.get(k) for r in okw if r['mode'] == m and r.get(k) is not None]
            S['%s_%s' % (k, m)] = (round(float(np.mean(v)), 3), round(float(np.min(v)), 3)) if v else None
    v = [r.get('all_lat_frac') for r in nowalk if r.get('verdict') == 'OK' and r.get('all_lat_frac') is not None]
    S['all_lat_frac_nowalk'] = round(float(np.mean(v)), 3) if v else None
    return S


def plot(walk, nowalk, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    cM, cP = '#2b6cb0', '#c05621'
    S = summarize(walk, nowalk); seeds = S['paired_seeds']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(seeds)); w = 0.38
    for dx, m, c, lb in ((-w / 2, 'M', cM, '移動中操作'), (w / 2, 'P', cP, '停車操作')):
        yw = [_mean(walk, s, m, 'mission_s') for s in seeds]; yn = [_mean(nowalk, s, m, 'mission_s') for s in seeds]
        ax[0].bar(x + dx, yw, w, color=c, label=lb + '（有行人）')
        for i, (a, b) in enumerate(zip(yw, yn)):
            if np.isfinite(b):
                ax[0].plot([i + dx - w / 2 + 0.03, i + dx + w / 2 - 0.03], [b, b], color='#1a1a1a', lw=1.6, ls='--')
            if np.isfinite(a):
                ax[0].text(i + dx, a + 4, '%.0f' % a, ha='center', va='bottom', fontsize=8)
    ax[0].plot([], [], color='#1a1a1a', lw=1.6, ls='--', label='同種子同組無行人')
    hi = np.nanmax([_mean(walk, s, m, 'mission_s') for s in seeds for m in 'MP']) if seeds else 1
    ax[0].set_ylim(0, hi * 1.38); ax[0].set_xticks(x); ax[0].set_xticklabels(['種子 %d' % s for s in seeds])
    ax[0].set_ylabel('整趟時間（s，模擬）'); ax[0].set_title('有行人：成功趟整趟時間（成對種子）'); ax[0].legend(frameon=False, loc='upper left', fontsize=8, ncol=2)
    for res, mk, tag in ((nowalk, 'o', '無行人'), (walk, '^', '有行人')):
        for m, c in (('M', cM), ('P', cP)):
            pts = [(r['marks']['dr_closed'], r['seg_s']['回程關門']) for r in res if r.get('verdict') == 'OK' and r['mode'] == m and r.get('marks', {}).get('dr_closed') and r['seg_s'].get('回程關門')]
            if pts:
                p_ = np.array(pts)
                ax[1].scatter(p_[:, 0], p_[:, 1], s=36 if mk == '^' else 22, marker=mk, facecolor=c if mk == '^' else 'none', edgecolor=c, lw=1.2,
                              label='%s・%s' % ('移動中' if m == 'M' else '停車', tag))
    ax[1].axvline(THR_S, color='#6b6b6b', lw=1.0, ls=':'); ax[1].text(THR_S - 0.5, ax[1].get_ylim()[1] - 1.0, '約 %d s' % THR_S, ha='right', va='top', fontsize=8, color='#4a4a4a')
    ax[1].set_xlabel('回程關門段起點（關抽屜完成時刻，s）'); ax[1].set_ylabel('回程關門段時間（s）')
    ax[1].set_title('抵達門口時刻 vs 回程關門段（行人路線、時刻固定）'); ax[1].legend(frameon=False, fontsize=8, loc='upper left')
    xs = np.arange(len(SEGS))
    ax[2].bar(xs - w / 2, [S['seg_cost_M'][g] for g in SEGS], w, color=cM, label='移動中操作')
    ax[2].bar(xs + w / 2, [S['seg_cost_P'][g] for g in SEGS], w, color=cP, label='停車操作')
    ax[2].axhline(0, color='#1a1a1a', lw=0.8)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(SEG_LBL, rotation=20); ax[2].set_ylabel('有行人 − 無行人（s，同種子平均）')
    ax[2].set_title('各段增加的時間'); ax[2].legend(frameon=False)
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=140)
    return S


def _main():
    av = sys.argv
    walk_r = _runs(av[av.index('--walk') + 1]); now_r = _runs(av[av.index('--nowalk') + 1])
    def run_all(rs):
        out = []
        for n in rs:
            try:
                out.append(A.analyze(n))
            except Exception as e:                                   # noqa: BLE001
                out.append(dict(run=n, verdict='FAIL analyze %r' % e))
        return out
    walk, nowalk = run_all(walk_r), run_all(now_r)
    for r in walk:
        print('%d seed %s %s %-30s mission %s dmin %s near %s ys %s/%s/%s/%s 側向 %s | %s' % (
            r['run'], r.get('seed'), r.get('mode'), r.get('verdict', '')[:30], r.get('mission_s'), r.get('walk_dmin'), r.get('walk_near_s'),
            r.get('ys_n'), r.get('ys_lat'), r.get('ys_behind'), r.get('ys_slow'), r.get('near_lat_frac'), r.get('seg_s')))
    S = plot(walk, nowalk, av[av.index('--png') + 1]) if '--png' in av else summarize(walk, nowalk)
    print(json.dumps(S, ensure_ascii=False, indent=1))
    if '--json' in av:
        json.dump(dict(summary=S, walk=walk, nowalk=nowalk), open(av[av.index('--json') + 1], 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    _main()
