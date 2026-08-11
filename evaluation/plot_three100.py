"""Three-method results, on the 83 routes every method completed.

Paired rather than pooled: each arm lost a different handful of trials to
bring-up failures, and comparing 96 against 92 against 91 different route sets
would let route difficulty leak into the comparison.

Three panels, in the order the numbers have to be read:

  left    arrival and contact rate together. A controller that abandons hard
          routes buys a low contact count with them, so neither number means
          anything alone.
  middle  the whole clearance distribution, not just its median -- the medians
          are only 0.11 m apart while the lower tails differ by half a metre.
  right   worst penetration, which is what separates a graze from a collision.

    python3 evaluation/plot_three100.py [out.png]
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'],
                     'axes.unicode_minus': False})

ROOT = '/home/howardchen/masterthesis'
ARMS = [('gmpc100', 'GMPC + CBF\n+ shield', '#2f75b5'),
        ('mppi100', 'nav2 MPPI', '#ed7d31'),
        ('rpp100', 'nav2 RPP\n($v_y = 0$)', '#a5a5a5')]


def rows(key):
    p = f'{ROOT}/evaluation/results/{key}/batch.csv'
    if not os.path.exists(p):
        return {}
    return {int(x['run'].split('seed')[-1]): x
            for x in csv.DictReader(open(p)) if x.get('min_clearance_m')}


data = {k: rows(k) for k, _, _ in ARMS}
common = sorted(set.intersection(*[set(v) for v in data.values()]))
if not common:
    print('no common routes'); sys.exit(1)
print(f'  paired on {len(common)} routes')

fig, axs = plt.subplots(1, 3, figsize=(16.0, 5.6))
labels = [l for _, l, _ in ARMS]
cols = [c for _, _, c in ARMS]
x = np.arange(len(ARMS))

# A single NaN removes an entire series from matplotlib's boxplot -- MPPI has
# one trial whose clearance could not be computed, and its box vanished. Drop
# non-finite values per series and say how many.
clr_raw = [[float(data[a][k]['min_clearance_m']) for k in common]
           for a, _, _ in ARMS]
clr = [[v for v in c if np.isfinite(v)] for c in clr_raw]
dropped = [len(a) - len(b) for a, b in zip(clr_raw, clr)]
if any(dropped):
    print('  dropped non-finite:', dict(zip([l for _, l, _ in ARMS], dropped)))

# ---------------------------------------------------------------- panel 1
arr = [100.0 * sum(1 for k in common if data[a][k]['success'] == 'True')
       / len(common) for a, _, _ in ARMS]
con = [100.0 * sum(1 for v in c if v < 0) / max(len(c), 1) for c in clr]
w = 0.36
b1 = axs[0].bar(x - w/2, arr, w, color=cols, label='到達率')
b2 = axs[0].bar(x + w/2, con, w, color=cols, alpha=0.42, hatch='//',
                label='碰撞率')
for xx, v in zip(x - w/2, arr):
    axs[0].text(xx, v + 1.5, f'{v:.0f}%', ha='center', fontsize=12,
                fontweight='bold')
for xx, v in zip(x + w/2, con):
    axs[0].text(xx, v + 1.5, f'{v:.1f}%', ha='center', fontsize=12)
axs[0].set_xticks(x); axs[0].set_xticklabels(labels, fontsize=11)
axs[0].set_ylabel('比例 [%]', fontsize=12)
axs[0].set_ylim(0, 115)
axs[0].set_title('到達率與碰撞率', fontsize=13, fontweight='bold')
axs[0].legend(loc='upper right', fontsize=10.5)
axs[0].grid(axis='y', alpha=0.25)
axs[0].spines[['top', 'right']].set_visible(False)

# ---------------------------------------------------------------- panel 2
for xx, nd, c in zip(x, dropped, clr):
    if nd:
        axs[1].text(xx, min(c) - 0.03, f'（{nd} 趟無法計算）', ha='center',
                    fontsize=9, color='#888888')
bp = axs[1].boxplot(clr, positions=x, widths=0.5, patch_artist=True,
                    medianprops=dict(color='#333333', lw=2),
                    flierprops=dict(marker='.', ms=5, mfc='#888888',
                                    mec='none', alpha=0.7))
for patch, c in zip(bp['boxes'], cols):
    patch.set_facecolor(c); patch.set_alpha(0.55)
axs[1].axhline(0, color='#c00000', lw=1.6, ls='--')
axs[1].text(2.45, 0.012, '接觸', color='#c00000', fontsize=11, ha='right')
axs[1].set_xticks(x); axs[1].set_xticklabels(labels, fontsize=11)
axs[1].set_ylabel('每趟最小間距 [m]', fontsize=12)
axs[1].set_title('最小間距分佈（每趟一點）', fontsize=13, fontweight='bold')
axs[1].grid(axis='y', alpha=0.25)
axs[1].spines[['top', 'right']].set_visible(False)

# ---------------------------------------------------------------- panel 3
worst = [-min(c) for c in clr]          # penetration depth, positive = worse
bars = axs[2].bar(x, worst, 0.5, color=cols, alpha=0.85)
for xx, v in zip(x, worst):
    axs[2].text(xx, v + 0.012, f'{v*100:.1f} cm', ha='center', fontsize=12,
                fontweight='bold')
axs[2].axhline(0.30, color='#c00000', lw=1.4, ls=':')
axs[2].text(2.45, 0.313, '車體半徑 0.30 m', color='#c00000', fontsize=10.5,
            ha='right')
axs[2].set_xticks(x); axs[2].set_xticklabels(labels, fontsize=11)
axs[2].set_ylabel('最深穿透 [m]', fontsize=12)
axs[2].set_ylim(0, 0.68)
axs[2].set_title('最深穿透', fontsize=13, fontweight='bold')
axs[2].grid(axis='y', alpha=0.25)
axs[2].spines[['top', 'right']].set_visible(False)

fig.suptitle(f'三方對照：統一硬體上限，配對 {len(common)} 條隨機路線',
             fontsize=15, fontweight='bold', y=1.00)
fig.text(0.5, -0.03,
         '所有方法使用相同運動上限（0.2775 m/s 每軸、6.25 m/s²、1.1327 rad/s）。'
         'RPP 的 $v_y$ 維持 0：純追蹤法不產生側移指令，屬演算法特性。',
         ha='center', fontsize=10.5, color='#666666')

out = sys.argv[1] if len(sys.argv) > 1 else \
    f'{ROOT}/evaluation/results/figs/three100.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout()
plt.savefig(out, dpi=125, bbox_inches='tight')
print(f'saved -> {out}')
for (a, l, _), ar, co, wo in zip(ARMS, arr, con, worst):
    print(f'  {l.replace(chr(10)," "):22} 到達 {ar:5.1f}%  碰撞 {co:5.1f}%  '
          f'最深 {wo*100:.1f} cm')
