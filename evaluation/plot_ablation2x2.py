"""The 2x2: what each safety layer buys, on the 28 routes all four arms ran.

Three panels, because the interesting result is not in the contact count alone.

  left    contacts. A without either layer is the reference; B and C each cut it
          to two or three, and only both together reach zero.
  middle  the clearance distribution. This is where the two layers stop looking
          interchangeable: the CBF lifts the whole distribution because it
          avoids early, while the shield leaves it low and merely keeps the tail
          out of contact.
  right   the cost, in arrival time.

C was expected to be safe but clumsy -- braking late, detouring badly, getting
stuck. It is not: C arrives on every route, faster than B and by a shorter path.
So the case for keeping both layers cannot rest on C being bad, and rests
instead on the tails: B and C each still contact 2-3 times with a worst
penetration of 0.30 m, and only D reaches zero with a positive worst clearance.
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
ARMS = [('ablA', 'A\nGMPC', '#a5a5a5'),
        ('ablB', 'B\n+ CBF', '#2f75b5'),
        ('ablC', 'C\n+ shield', '#ed7d31'),
        ('gmpc100', 'D\n+ CBF + shield', '#548235')]


def rows(key):
    p = f'{ROOT}/evaluation/results/{key}/batch.csv'
    if not os.path.exists(p):
        return {}
    return {int(x['run'].split('seed')[-1]): x
            for x in csv.DictReader(open(p)) if x.get('min_clearance_m')}


data = {k: rows(k) for k, _, _ in ARMS}
common = sorted(set.intersection(*[set(v) for v in data.values()]))
print(f'  paired on {len(common)} routes')

labels = [l for _, l, _ in ARMS]
cols = [c for _, _, c in ARMS]
x = np.arange(len(ARMS))
clr = [[v for v in (float(data[a][k]['min_clearance_m']) for k in common)
        if np.isfinite(v)] for a, _, _ in ARMS]
con = [sum(1 for v in c if v < 0) for c in clr]
worst = [min(c) for c in clr]
tarr = [[float(data[a][k]['arrival_time_s']) for k in common
         if data[a][k]['success'] == 'True' and data[a][k]['arrival_time_s']]
        for a, _, _ in ARMS]

fig, axs = plt.subplots(1, 3, figsize=(16.0, 5.6))

# ------------------------------------------------------------- panel 1
axs[0].bar(x, con, 0.55, color=cols, alpha=0.9)
for xx, v, c in zip(x, con, clr):
    axs[0].text(xx, v + 0.5, f'{v}', ha='center', fontsize=14,
                fontweight='bold')
    axs[0].text(xx, v + 2.0, f'{100*v/len(c):.0f}%', ha='center', fontsize=11,
                color='#666666')
axs[0].set_xticks(x); axs[0].set_xticklabels(labels, fontsize=11)
axs[0].set_ylabel(f'碰撞趟數（共 {len(common)} 條）', fontsize=12)
axs[0].set_ylim(0, max(con) * 1.30 + 1)
axs[0].set_title('碰撞', fontsize=13, fontweight='bold')
axs[0].grid(axis='y', alpha=0.25)
axs[0].spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------- panel 2
bp = axs[1].boxplot(clr, positions=x, widths=0.5, patch_artist=True,
                    medianprops=dict(color='#222222', lw=2),
                    flierprops=dict(marker='.', ms=5, mfc='#888888',
                                    mec='none', alpha=0.7))
for patch, c in zip(bp['boxes'], cols):
    patch.set_facecolor(c); patch.set_alpha(0.55)
axs[1].axhline(0, color='#c00000', lw=1.6, ls='--')
axs[1].text(3.45, 0.015, '接觸', color='#c00000', fontsize=11, ha='right')
for xx, w in zip(x, worst):
    axs[1].annotate(f'最差 {w:+.3f}', (xx, w), textcoords='offset points',
                    xytext=(0, -16), ha='center', fontsize=9, color='#444444')
axs[1].set_xticks(x); axs[1].set_xticklabels(labels, fontsize=11)
axs[1].set_ylabel('每趟最小間距 [m]', fontsize=12)
axs[1].set_title('最小間距分佈 —— CBF 抬高整體，shield 只守住尾端',
                 fontsize=12.5, fontweight='bold')
axs[1].grid(axis='y', alpha=0.25)
axs[1].spines[['top', 'right']].set_visible(False)

# ------------------------------------------------------------- panel 3
med = [float(np.median(t)) if t else float('nan') for t in tarr]
axs[2].bar(x, med, 0.55, color=cols, alpha=0.9)
for xx, v in zip(x, med):
    axs[2].text(xx, v + 1.5, f'{v:.0f} s', ha='center', fontsize=13,
                fontweight='bold')
axs[2].set_xticks(x); axs[2].set_xticklabels(labels, fontsize=11)
axs[2].set_ylabel('到達時間中位 [s]', fontsize=12)
axs[2].set_ylim(0, max(med) * 1.22)
axs[2].set_title('代價：到達時間（四組皆 28/28 到達）',
                 fontsize=12.5, fontweight='bold')
axs[2].grid(axis='y', alpha=0.25)
axs[2].spines[['top', 'right']].set_visible(False)

fig.suptitle(f'CBF × Shield 消融：配對 {len(common)} 條隨機路線',
             fontsize=15, fontweight='bold', y=1.00)
fig.text(0.5, -0.04,
         'C（僅 shield）並不笨拙 —— 它每趟都到達，且比 B 更快、路徑更短。'
         '兩層都保留的理由在尾端：B 與 C 各自仍有 2–3 次碰撞、最深 0.30 m，'
         '只有 D 達到零碰撞且最差間距為正。',
         ha='center', fontsize=10.5, color='#666666')

out = sys.argv[1] if len(sys.argv) > 1 else \
    f'{ROOT}/evaluation/results/figs/ablation2x2.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout()
plt.savefig(out, dpi=125, bbox_inches='tight')
print(f'saved -> {out}')
for (a, l, _), cn, w, m in zip(ARMS, con, worst, med):
    print(f'  {l.replace(chr(10)," "):18} 碰撞 {cn:2d}  最差 {w:+.3f}  到達 {m:.0f} s')
