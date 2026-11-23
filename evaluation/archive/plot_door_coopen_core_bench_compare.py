#!/usr/bin/env python3
"""圖 29：核心段接觸模擬（指定台架初態）持握漂移四趟比較。
(a) 共同門角區間 0–43.72° 每 5° 平均漂移：bench_3（偏移估計切 free 即凍結）／bench_4、5、6（開門段恢復偏移更新；bench_6 另含 gc.freeze＋暫停 5 N）；
    右側另畫 bench_5、6 的新增段 43.72–47.7°（bench_3、4 未開到）。
(b) 逐物理步漂移對實測門角：bench_3 與 bench_6（OPEN 窗口），5 mm 為持握漂移中止門檻。
資料：runs/door_coopen_core_bench_6/analysis/door_coopen_core_bench_6_compare.json（a）；各趟 room_run／evidence／task（b，同 compare 腳本的 load）。

    python3 evaluation/plot_door_coopen_core_bench_compare.py   # 寫 第七次進度報告素材/圖表/29_coopen_核心段接觸模擬_漂移四趟比較.png（不覆寫）
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib import font_manager                          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_core_bench_6_compare as CMP              # noqa: E402

OUT = os.path.join(WS, '第七次進度報告素材', '圖表', '29_coopen_核心段接觸模擬_漂移四趟比較.png')
CJ = os.path.join(HERE, 'runs', 'door_coopen_core_bench_6', 'analysis', 'door_coopen_core_bench_6_compare.json')
COL = {'bench_3': '#8a8f98', 'bench_4': '#2f6db5', 'bench_5': '#3f9a7a', 'bench_6': '#c0612b'}
LAB = {'bench_3': 'bench_3（切 free 即凍結）', 'bench_4': 'bench_4（開門段更新）', 'bench_5': 'bench_5（同上＋診斷）', 'bench_6': 'bench_6（聯合配置）'}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    fp = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(fp)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=fp).get_name()
    cj = json.load(open(CJ))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={'width_ratios': [1.15, 1]})
    a = ax[0]
    for k in ('bench_3', 'bench_4', 'bench_5', 'bench_6'):
        xs = [(b['q_bin_deg'][0] + b['q_bin_deg'][1]) / 2 for b in cj['common_bins'] if b.get(k)]
        ys = [b[k]['drift_mean_mm'] for b in cj['common_bins'] if b.get(k)]
        a.plot(xs, ys, '-o', ms=4, lw=2, color=COL[k], label=LAB[k])
        ns = (cj.get('new_segment_beyond_common') or {}).get(k)
        if ns:
            xm = (ns['q_range_deg'][0] + ns['q_range_deg'][1]) / 2 + {'bench_4': 0.0, 'bench_5': -0.45, 'bench_6': 0.45}.get(k, 0.0)
            a.errorbar([xm], [ns['drift_mean_mm']], yerr=[[0], [ns['drift_max_mm'] - ns['drift_mean_mm']]], fmt='s', ms=6, color=COL[k], capsize=3)
    a.axvline(cj['q_common_max_deg'], color='#bbbbbb', lw=1, ls=':')
    a.text(cj['q_common_max_deg'] + 0.4, 0.35, '新增段\n■ 平均\n上鬚＝最大', fontsize=8, color='#555555')
    a.axhline(5.0, color='#b23b3b', lw=1, ls='--')
    a.text(25, 5.1, '中止門檻 5 mm', fontsize=8, color='#b23b3b')
    a.set_xlabel('實測門角（°，每 5° 分箱中點）')
    a.set_ylabel('持握漂移平均（mm）')
    a.set_title('(a) 共同門角區間 0–43.72° 的平均漂移', fontsize=10)
    a.set_ylim(0, 5.6)
    a.grid(alpha=0.25)
    a.set_xlim(0, 48.5)
    a.legend(fontsize=8, loc='lower left')
    b = ax[1]
    for k in ('bench_3', 'bench_6'):
        rows, _, _ = CMP.load(CMP.RUNS[k])
        pts = [(x['q'], x['d']) for x in rows if x['d'] is not None]
        b.plot([p[0] for p in pts], [p[1] for p in pts], '.', ms=1.2, color=COL[k], alpha=0.5, label=LAB[k])
    b.axhline(5.0, color='#b23b3b', lw=1, ls='--')
    b.set_xlabel('實測門角（°）')
    b.set_ylabel('逐物理步持握漂移（mm）')
    b.set_title('(b) 漂移對門角（OPEN 窗口）', fontsize=10)
    b.set_ylim(0, 5.6)
    b.grid(alpha=0.25)
    b.text(1, 5.1, '中止門檻 5 mm', fontsize=8, color='#b23b3b')
    b.legend(fontsize=8, loc='lower right', markerscale=8)
    fig.suptitle('指定台架初態的接觸模擬：開門段恢復偏移更新後，共同區間漂移三趟相近；末段 43.7–47.7° 最大 4.8 mm（名目差額 0.2 mm）', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT, dpi=150)
    print('寫', OUT)


if __name__ == '__main__':
    main()
