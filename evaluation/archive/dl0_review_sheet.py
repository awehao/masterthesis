#!/usr/bin/env python3
"""DL0 自動標籤抽查表：依「群組 × 層」分層抽樣（只含訓練／開發群組；**封存測試不列入**），輸出疊圖頁與核對表。

層：clean_far（> 2 m）、clean_mid（0.4–2 m）、clean_near（< 0.4 m）、truncated、near_clipped、no_handle_in_view。
每群組每層抽至多 PER 格（種子 0）；全體某層總數 ≤ RARE 時全查。疊圖：橘＝可見遮罩、藍＝ignore、綠點＝兩端投影（在畫面內者）。
核對表欄位 review／fix／note 由抽查者填寫；原自動標籤不改（修標另存）。

    python3 evaluation/dl0_review_sheet.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dl0_autolabel as AL  # noqa: E402

PER, RARE = 4, 10
OUT = os.path.join(HERE, 'results', 'vision', 'dl0_review')
TEST_GROUPS = ('dl0_test_lateral', 'dl0_test_view')


def stratum(L):
    f = L['flags']
    if 'no_handle_in_view' in f:
        return 'no_handle_in_view'
    if 'near_clipped' in f:
        return 'near_clipped'
    if 'truncated' in f:
        return 'truncated'
    return 'clean_far' if L['dist_m'] > 2.0 else 'clean_mid' if L['dist_m'] >= 0.4 else 'clean_near'


def main():
    os.makedirs(OUT, exist_ok=True)
    pool = []
    for cap, grp, _ in AL.SOURCES:
        assert not any(t in cap for t in TEST_GROUPS), '封存測試不得列入抽查'
        for fr in AL.frames_of(cap):
            if fr['c'] is None:
                continue
            L = AL.label(fr)
            pool.append({'cap': cap, 'grp': grp, 'fr': fr, 'st': stratum(L), 'dist': L['dist_m'],
                         'flags': '+'.join(L['flags']), 'vis': L['vis_frac'], 'ends_in': L['ends_in_frame']})
    tot = {}
    for p in pool:
        tot[p['st']] = tot.get(p['st'], 0) + 1
    rng = np.random.default_rng(0)
    pick = []
    for grp in sorted({p['grp'] for p in pool}):
        for st in sorted(tot):
            cand = [p for p in pool if p['grp'] == grp and p['st'] == st]
            if not cand:
                continue
            k = len(cand) if tot[st] <= RARE else min(PER, len(cand))
            idx = sorted(rng.choice(len(cand), size=k, replace=False))
            pick += [cand[i] for i in idx]
    # 疊圖頁（每頁 12 格）
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.family': ['Noto Sans CJK JP']})
    rows = []
    for pg in range(0, len(pick), 12):
        fig, axs = plt.subplots(3, 4, figsize=(20, 11.5))
        for ax, (j, p) in zip(axs.ravel(), enumerate(pick[pg:pg + 12], start=pg)):
            fr = p['fr']
            L = AL.label(fr)
            img = np.asarray(Image.open(fr['rgb']).convert('RGB')).astype(float) / 255
            img[L['mask']] = 0.35 * img[L['mask']] + 0.65 * np.array([1.0, 0.45, 0.1])
            img[L['ignore']] = 0.4 * img[L['ignore']] + 0.6 * np.array([0.2, 0.6, 1.0])
            ax.imshow(img)
            T, K, c = fr['T'], fr['K'], np.asarray(fr['c'])
            for sgn in (-1, 1):
                pc = T[:3, :3].T @ (c + sgn * AL.AXIS_W * AL.LEN_BAR / 2 - T[:3, 3])
                if pc[2] > 1e-6:
                    uv = (K @ pc)[:2] / pc[2]
                    if 0 <= uv[0] < AL.W and 0 <= uv[1] < AL.H:
                        ax.plot(uv[0], uv[1], 'o', ms=6, mfc='none', mec='lime', mew=2)
            rid = f'R{j:03d}'
            ax.set_title(f"{rid}｜{p['grp']} f{fr['n']}｜{p['st']}\n{p['dist']:.2f} m　{p['flags']}　可見 {p['vis']}", fontsize=8)
            ax.axis('off')
            rows.append({'id': rid, 'group': p['grp'], 'capture': p['cap'], 'n': fr['n'], 'stratum': p['st'],
                         'dist_m': round(p['dist'], 3), 'flags': p['flags'], 'vis_frac': p['vis'],
                         'sheet': f'sheet_{pg // 12:02d}.png', 'review': '', 'fix': '', 'note': ''})
        for ax in axs.ravel()[len(pick[pg:pg + 12]):]:
            ax.axis('off')
        fig.suptitle('DL0 自動標籤抽查：橘＝可見遮罩、藍＝ignore、綠圈＝端點投影｜review：ok／fix／unsure', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f'sheet_{pg // 12:02d}.png'), dpi=80)
        plt.close(fig)
    with open(os.path.join(OUT, 'review.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('層總數', tot)
    print('抽樣', len(rows), '格；頁數', (len(rows) + 11) // 12, '→', OUT)


if __name__ == '__main__':
    main()
