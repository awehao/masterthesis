#!/usr/bin/env python3
"""圖 20：嚴格停車（PARK_HOLD v2.1）vs MotM 成果表——版式同圖 09（平均 ± 標準差，n = 3）。只讀結果檔，數字不手抄。

來源：results/motm_speed/park_hold_formal_results.json（正式結果）、
      results/motm_speed/park_formal_motm_metrics/<rid>.json（motm_metrics.py 對六趟的唯讀輸出，與圖 09 同一套指標）。

    python3 evaluation/park_result_table.py
"""
from __future__ import annotations

import json
import os
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from matplotlib import font_manager                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MS = os.path.join(HERE, 'results', 'motm_speed')
OUT = os.path.join(HERE, '..', '第六次進度報告素材', '圖表', '20_嚴格停車vsMotM_成果表.png')
# 與圖 09 同一組色（dataviz 參考調色盤，淺色）
SURF, INK, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#898781', '#e1e0d9'


def _tc_font():
    """同 build_sixth_progress_motm.py：由 NotoSansCJK .ttc 取出繁中字面。"""
    from fontTools.ttLib import TTCollection
    out = []
    for w in ('Regular', 'Bold'):
        dst = os.path.join(tempfile.gettempdir(), f'NotoSansCJKtc-{w}.otf')
        if not os.path.exists(dst):
            col = TTCollection(f'/usr/share/fonts/opentype/noto/NotoSansCJK-{w}.ttc')
            for f in col.fonts:
                if f['name'].getDebugName(1).startswith('Noto Sans CJK TC'):
                    f.save(dst)
                    break
        font_manager.fontManager.addfont(dst)
        out.append(dst)
    return font_manager.FontProperties(fname=out[0]).get_name()


plt.rcParams.update({'font.family': _tc_font(), 'font.size': 11, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'axes.titlecolor': INK, 'axes.titleweight': 'bold',
                     'axes.titlesize': 12, 'savefig.dpi': 200, 'savefig.facecolor': SURF,
                     'axes.unicode_minus': False})


def main():
    res = json.load(open(os.path.join(MS, 'park_hold_formal_results.json')))
    H = [r for r in res['runs'] if r['method'] == 'PARK_HOLD']
    M = [r for r in res['runs'] if r['method'] == 'MOTM']
    met = {r['rid']: json.load(open(os.path.join(MS, 'park_formal_motm_metrics', r['rid'] + '.json')))
           for r in res['runs']}
    for r in res['runs']:
        assert met[r['rid']]['final_phase'] == 'DONE', r['rid']

    def ms(L, f, fmt='{:.2f}', metric=False):
        v = np.array([f(met[r['rid']] if metric else r) for r in L], float)
        return f'{fmt.format(v.mean())} ± {fmt.format(v.std(ddof=1))}'

    def cnt(L, f):
        return f'{sum(1 for r in L if f(r))}/{len(L)}'
    ok = lambda r: r['task_ok']                                              # noqa: E731
    uc = {met[r['rid']]['undesignated_contact'].split('（')[0] for r in res['runs']}
    v_low = {met[r['rid']]['v_low_mps'] for r in res['runs']}
    assert len(v_low) == 1
    v_low = v_low.pop()
    rows = [
        ('完成整套流程（S1–S6＋重播）', cnt(H, ok), cnt(M, ok)),
        ('全窗停車判準（保持期偏離 ≤ 1 mm、逐步 ≤ 1 mm/s）', cnt(H, lambda r: r['mode_ok']) + ' 合規', '不適用'),
        ('GO → 回程終端完成 (s)', ms(H, lambda r: r['T_complete_s'], '{:.1f}'), ms(M, lambda r: r['T_complete_s'], '{:.1f}')),
        ('GO → 夾持成立 (s)', ms(H, lambda r: r['segments']['grasp_attached'], '{:.1f}'),
         ms(M, lambda r: r['segments']['grasp_attached'], '{:.1f}')),
        ('夾持建立期間底盤位移 (mm)', ms(H, lambda d: d['grasp_establish']['base_path_mm'], metric=True),
         ms(M, lambda d: d['grasp_establish']['base_path_mm'], metric=True)),
        ('首次接觸 → 開始拉開 (s)', ms(H, lambda d: d['contact_to_pull_s'], metric=True),
         ms(M, lambda d: d['contact_to_pull_s'], metric=True)),
        ('拉開時手臂分擔 (mm)', ms(H, lambda d: abs(d['share_OPEN']['arm_only_dy_mm']), '{:.1f}', True),
         ms(M, lambda d: abs(d['share_OPEN']['arm_only_dy_mm']), '{:.1f}', True)),
        ('拉開時底盤位移 (mm)', ms(H, lambda d: float(np.hypot(*d['share_OPEN']['base_dxy_mm'])), '{:.1f}', True),
         ms(M, lambda d: float(np.hypot(*d['share_OPEN']['base_dxy_mm'])), '{:.1f}', True)),
        ('拉開 200 mm 耗時 (s)', ms(H, lambda d: d['share_OPEN']['duration_s'], metric=True),
         ms(M, lambda d: d['share_OPEN']['duration_s'], metric=True)),
        ('關回耗時 (s)', ms(H, lambda d: d['share_CLOSE']['duration_s'], metric=True),
         ms(M, lambda d: d['share_CLOSE']['duration_s'], metric=True)),
        (f'全身控制期間底盤低速 (< {v_low * 1e3:.0f} mm/s) 最長連續 (s)',
         ms(H, lambda d: d['low_speed_dwell']['longest_incl_open_hold_s'], '{:.1f}', True),
         ms(M, lambda d: d['low_speed_dwell']['longest_incl_open_hold_s'], metric=True)),
        ('保持期底盤相對錨點偏離最大 (mm)', ms(H, lambda r: r['park']['max']['dev_pos_mm'], '{:.3f}'), '—'),
        ('夾持漂移最大 S4，逐物理步 (mm)', ms(H, lambda r: r['S4_drift_max_mm'], '{:.3f}'),
         ms(M, lambda r: r['S4_drift_max_mm'], '{:.3f}')),
        ('啟動擺盪峰對峰 (mm)', ms(H, lambda d: d['startup_4s']['tcp_z_ptp_mm'], '{:.1f}', True),
         ms(M, lambda d: d['startup_4s']['tcp_z_ptp_mm'], '{:.1f}', True)),
        ('非預期接觸', '／'.join(sorted(uc)) if len(uc) == 1 else '見各趟', '／'.join(sorted(uc)) if len(uc) == 1 else '見各趟'),
    ]
    cols = ['指標（平均 ± 標準差，n = 3）', '嚴格停車（PARK_HOLD v2.1）', '移動中操作 MotM']
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.axis('off')
    tb = ax.table(cellText=[list(r) for r in rows], colLabels=cols, loc='center', cellLoc='center',
                  colLoc='center', colWidths=[0.46, 0.27, 0.27])
    tb.auto_set_font_size(False)
    tb.set_fontsize(10)
    tb.scale(1, 1.7)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor('#f2f1ec')
            cell.set_text_props(color=INK, weight='bold')
        else:
            cell.set_text_props(color=INK)
    ax.set_title('抽屜開關：嚴格停車 vs 移動中操作（正式三對，同版本、同環境、交替執行）', loc='left')
    pair = '、'.join(f"{p['M']}／{p['H']}" for _, p in sorted(res['pairs'].items()))
    fig.text(0.01, 0.035, f'趟次 {pair}（results/motm_speed/park_hold_formal_results.json；其餘列由 motm_metrics.py 唯讀重算）。'
             '計時起點＝共同 GO（第一筆實際發布導航計畫）。', color=MUTED, fontsize=8.5)
    fig.text(0.01, 0.005, '嚴格停車的長時間低速是設計（全身控制期間底盤停住），不是不允許的停頓。兩個完整策略的比較，'
             '差異不單獨歸因於底盤是否移動；n = 3，不做顯著性。把手 Ø26、手指拆塊碰撞，未經實物驗證。',
             color=MUTED, fontsize=8.5)
    fig.savefig(OUT, bbox_inches='tight')
    print('寫出', os.path.abspath(OUT))
    for r in rows:
        print(' | '.join(r))


if __name__ == '__main__':
    main()
