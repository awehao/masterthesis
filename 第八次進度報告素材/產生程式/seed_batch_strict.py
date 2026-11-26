#!/usr/bin/env python3
"""移動中操作（M）vs 嚴格停車操作（S）種子批次，舊停車操作（P，只關移動中抓取）列參考。
用法：python3 seed_batch_strict.py --runs 142-185,202-220 [--json out.json] [--png out.png]
逐趟指標沿用 seed_batch_analyze.analyze（模式、行人旗標取自登錄）。只用成功趟算時間；成功率另列。
圖：(1)(2) 無行人／有行人各種子整趟時間（M、S 成對；P 空心點）；(3) 各段平均時間（無行人）；(4) 底盤靜止中動手臂 vs 邊走邊動手臂（無行人平均）。"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seed_batch_analyze as A                                        # noqa: E402

SEGS = ['門前', '開門穿門到茶几放盒', '抽屜取筆放盒', '關抽屜', '回程關門', '回起點']
SEG_LBL = ['門前', '開門→茶几', '抽屜取筆', '關抽屜', '回程關門', '回起點']
MODES = ('M', 'S', 'P')
NAME = dict(M='移動中操作', S='嚴格停車操作', P='舊停車操作（參考）')
COL = dict(M='#2b6cb0', S='#c05621', P='#8a8f98')


def _runs(spec):
    out = []
    for a in spec.split(','):
        if '-' in a:
            lo, hi = a.split('-'); out += list(range(int(lo), int(hi) + 1))
        elif a:
            out.append(int(a))
    return out


def _ok(res, mode, walk):
    return [r for r in res if r.get('verdict') == 'OK' and r.get('mission_s') and r['mode'] == mode and r.get('walk') == walk]


def _seed_mean(res, seed, mode, walk, key, seg=None):
    v = [(r['seg_s'].get(seg) if seg else r.get(key)) for r in _ok(res, mode, walk) if r['seed'] == seed]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else np.nan


def _paired(res, a, b, walk):
    seeds = sorted({r['seed'] for r in _ok(res, a, walk)} & {r['seed'] for r in _ok(res, b, walk)})
    d = [_seed_mean(res, s, a, walk, 'mission_s') - _seed_mean(res, s, b, walk, 'mission_s') for s in seeds]
    o = dict(seeds=seeds, diff=[round(x, 1) for x in d])
    if d:
        o.update(mean=round(float(np.mean(d)), 1), sd=round(float(np.std(d, ddof=1)), 1) if len(d) > 1 else None,
                 rel_pct=round(100.0 * float(np.mean(d)) / float(np.mean([_seed_mean(res, s, b, walk, 'mission_s') for s in seeds])), 1))
        try:
            from scipy.stats import wilcoxon
            o['wilcoxon_p'] = round(float(wilcoxon(d).pvalue), 4) if len(d) >= 2 else None
        except Exception:                                            # noqa: BLE001
            o['wilcoxon_p'] = None
    return o


def summarize(res):
    S = {}
    for w in (0, 1):
        for m in MODES:
            n_all = [r for r in res if r['mode'] == m and r.get('walk') == w]
            ok = _ok(res, m, w)
            S['success_%s_w%d' % (m, w)] = '%d/%d' % (len(ok), len(n_all))
            S['fails_%s_w%d' % (m, w)] = [(r['run'], r.get('verdict', '')[5:40]) for r in n_all if r.get('verdict') != 'OK']
            if ok:
                S['mission_%s_w%d' % (m, w)] = round(float(np.mean([r['mission_s'] for r in ok])), 1)
                S['seg_%s_w%d' % (m, w)] = {g: round(float(np.mean([r['seg_s'][g] for r in ok if g in r['seg_s']])), 1) for g in SEGS}
                for k in ('still_s', 'still_arm_s', 'moving_arm_s', 'jerk_p95', 'rot_rad', 'reverse_s', 'walk_dmin', 'walk_near_s', 'ys_n'):
                    v = [r[k] for r in ok if r.get(k) is not None]
                    if v:
                        S['%s_%s_w%d' % (k, m, w)] = round(float(np.mean(v)), 2)
        S['M_minus_S_w%d' % w] = _paired(res, 'M', 'S', w)
        S['S_minus_P_w%d' % w] = _paired(res, 'S', 'P', w)
    return S


def plot(res, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK TC', 'Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    S = summarize(res)
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.8), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.1, 0.8]))
    w_ = 0.38
    hi = max([r['mission_s'] for r in res if r.get('verdict') == 'OK' and r.get('mission_s')] or [1.0])
    for k, walk in enumerate((0, 1)):
        a = ax[k]
        seeds = sorted({r['seed'] for m in 'MS' for r in _ok(res, m, walk)})
        x = np.arange(len(seeds))
        for dx, m in ((-w_ / 2, 'M'), (w_ / 2, 'S')):
            y = [_seed_mean(res, s, m, walk, 'mission_s') for s in seeds]
            a.bar(x + dx, y, w_, color=COL[m], label=NAME[m])
            for i, v in enumerate(y):
                if np.isfinite(v):
                    a.text(i + dx, v + 4, '%.0f' % v, ha='center', va='bottom', fontsize=7.5)
                else:
                    a.text(i + dx, 8, '失敗', ha='center', va='bottom', fontsize=7, rotation=90, color='#4a4a4a')
        yp = [_seed_mean(res, s, 'P', walk, 'mission_s') for s in seeds]
        a.scatter(x, yp, s=34, facecolor='none', edgecolor='#1a1a1a', lw=1.2, zorder=3, label=NAME['P'])
        p_ = S['M_minus_S_w%d' % walk]
        ttl = '%s：整趟時間（成功趟）' % ('無行人' if walk == 0 else '有行人')
        if p_.get('mean') is not None:
            ttl += '\nM−S 平均 %+.1f s（%+.1f%%，n=%d，p=%s）' % (p_['mean'], p_['rel_pct'], len(p_['seeds']), p_.get('wilcoxon_p'))
        a.set_title(ttl, fontsize=10)
        a.set_ylim(0, hi * 1.3); a.set_xticks(x); a.set_xticklabels(['種子 %d' % s for s in seeds], fontsize=8)
        a.set_ylabel('整趟時間（s，模擬）')
        a.legend(frameon=False, loc='upper left', fontsize=8, ncol=3)
    xs = np.arange(len(SEGS)); w3 = 0.27
    for j, m in enumerate(MODES):
        sg = S.get('seg_%s_w0' % m)
        if sg:
            ax[2].bar(xs + (j - 1) * w3, [sg[g] for g in SEGS], w3, color=COL[m], label=NAME[m])
    ax[2].set_xticks(xs); ax[2].set_xticklabels(SEG_LBL, rotation=20, fontsize=8); ax[2].set_ylabel('平均時間（s）')
    ax[2].set_title('各段平均時間（無行人）', fontsize=10); ax[2].legend(frameon=False, fontsize=8)
    xm = np.arange(len(MODES))
    a1 = [S.get('still_arm_s_%s_w0' % m, np.nan) for m in MODES]; a2 = [S.get('moving_arm_s_%s_w0' % m, np.nan) for m in MODES]
    ax[3].bar(xm - 0.2, a1, 0.38, color='#9db8d9', label='底盤靜止中動手臂')
    ax[3].bar(xm + 0.2, a2, 0.38, color='#2a4365', label='邊走邊動手臂')
    for i in range(len(MODES)):
        for dx, v in ((-0.2, a1[i]), (0.2, a2[i])):
            if np.isfinite(v):
                ax[3].text(i + dx, v + 1, '%.0f' % v, ha='center', va='bottom', fontsize=8)
    ax[3].set_xticks(xm); ax[3].set_xticklabels(['移動中', '嚴格停車', '舊停車'], fontsize=9); ax[3].set_ylabel('每趟時間（s）')
    ax[3].set_title('手臂動作發生在哪（無行人平均）', fontsize=10); ax[3].legend(frameon=False, fontsize=8)
    for a in ax:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(out_png, dpi=140)
    return S


def _main():
    av = sys.argv
    res = []
    for n in _runs(av[av.index('--runs') + 1]):
        try:
            res.append(A.analyze(n))
        except Exception as e:                                       # noqa: BLE001
            res.append(dict(run=n, mode='?', walk=None, verdict='FAIL analyze %r' % e))
    for r in res:
        print('%d %s w%s seed %s %-28s mission %s still %s 停車動臂 %s 邊走動臂 %s | %s' % (
            r['run'], r.get('mode'), r.get('walk'), r.get('seed'), r.get('verdict', '')[:28], r.get('mission_s'), r.get('still_s'),
            r.get('still_arm_s'), r.get('moving_arm_s'), r.get('seg_s')))
    S = plot(res, av[av.index('--png') + 1]) if '--png' in av else summarize(res)
    print(json.dumps(S, ensure_ascii=False, indent=1))
    if '--json' in av:
        json.dump(dict(summary=S, runs=res), open(av[av.index('--json') + 1], 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    _main()
