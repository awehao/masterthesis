"""Who does the work: the CBF, or the raw-scan shield?

Two panels.

Left, the attribution. The obvious figure for a safety layer is "contacts went
to zero", but that invites the wrong reading -- that the barrier stopped
mattering. Counting cycles instead shows the barrier carrying almost all of the
load, the shield touching under 2% of commands, and a minority of those landing
where the barrier had nothing to say. That minority is the part no CBF tuning
can reach, because the obstacle was never in the QP.

Right, the shape of the shield's constraint, to make clear it is not a stop
button: the permitted approach speed falls with distance and only turns
negative -- demanding retreat -- inside d_stop. Tangential motion and retreat
are never restricted at all.

    python3 evaluation/plot_shield_attrib.py [out.png]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

fm.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc')
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'],
                     'axes.unicode_minus': False})

# Measured over the gmpc100 batch, 96 runs.
# Every number below comes from shield_attribution.json, produced by
# summarise_shield_attribution.py straight off the bags. NOTHING is hand-copied
# here any more: the constants that used to sit at this spot had drifted so far
# that they no longer reproduced (they said 4195 / 2896 / 1134 and a 28.1%
# split; recomputing over the same 96 runs gives 3695 / 1683 / 1777 and 51.4%),
# and the figure captioned the split with the wrong denominator on top of that.
import json
_J = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  'results', 'shield_attribution.json')
with open(_J) as _f:
    A = json.load(_f)

TOTAL         = A['total_cycles']
CBF_ROWS      = A['cbf_rows_cycles']
CBF_DANGER    = A['cbf_danger_cycles']
CBF_HNEG      = A['cbf_hneg_cycles']
SH_ACTIVE     = A['shield_active']
SH_BOTH       = A['shield_with_h_negative']
SH_ALONE      = A['shield_with_h_nonnegative']
SH_CLASSIFIED = A['shield_cbf_classifiable']
SH_UNCLASS    = A['shield_unclassified']
SH_D_MED      = A['shield_distance']['median_m']
SH_DV_MED     = A['shield_dv_median_mps']
N_RUNS        = A['n_runs']
DANGER_H      = A['danger_threshold_h']
# Shield parameters, as configured.
ALPHA, D0, TAU, ABRAKE, EPS = 2.0, 0.05, 0.15, 6.25, 0.05

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.0, 6.2),
                               gridspec_kw={'width_ratios': [1.15, 1.0]})

# ---------------------------------------------------------------- left panel
labels = ['CBF 建立約束', f'CBF 進入 danger 區\n($h < {DANGER_H}$)',
          'CBF $h < 0$', 'shield 介入']
vals = [CBF_ROWS, CBF_DANGER, CBF_HNEG, SH_ACTIVE]
cols = ['#2f75b5', '#5b9bd5', '#9dc3e6', '#ed7d31']
y = np.arange(len(vals))[::-1]
axL.barh(y, vals, color=cols, height=0.62)
for yy, v in zip(y, vals):
    axL.text(v + TOTAL * 0.012, yy, f'{v:,}　({v/TOTAL:.1%})',
             va='center', fontsize=12)
axL.set_yticks(y)
axL.set_yticklabels(labels, fontsize=12)
axL.set_xlim(0, TOTAL * 1.32)
axL.set_ylim(y[-1] - 1.15, y[0] + 0.55)
axL.set_xlabel(f'控制週期數（總計 {TOTAL:,}）', fontsize=12)
axL.set_title(f'誰在工作：{N_RUNS} 趟、{TOTAL:,} 個控制週期',
              fontsize=14, fontweight='bold')
axL.grid(axis='x', alpha=0.25)
axL.spines[['top', 'right']].set_visible(False)

# breakdown of the shield's own interventions
# below the bars, not beside them: at the earlier position it covered the
# 116,968 label on the danger-zone row
bx, by, bw, bh = TOTAL * 0.30, y[-1] - 0.92, TOTAL * 0.68, 0.95
axL.add_patch(Rectangle((bx, by), bw, bh, fc='#fdf0e6', ec='#ed7d31',
                        lw=1.6, zorder=3))
frac = SH_ALONE / SH_CLASSIFIED
axL.text(bx + bw * 0.03, by + bh * 0.74,
         f'shield 介入 {SH_ACTIVE:,} 次，其中 {SH_CLASSIFIED:,} 次可對齊 CBF 診斷',
         fontsize=11.5, fontweight='bold', color='#7f3f10', zorder=4)
axL.text(bx + bw * 0.03, by + bh * 0.44,
         f'CBF 也在救（$h<0$）　{SH_BOTH:,}　{1-frac:.1%}',
         fontsize=11, color='#7f3f10', zorder=4)
axL.text(bx + bw * 0.03, by + bh * 0.14,
         f'CBF 未判定為危險　　{SH_ALONE:,}　{frac:.1%}　← 兩層判定不重合',
         fontsize=11, fontweight='bold', color='#c00000', zorder=4)

# --------------------------------------------------------------- right panel
d = np.linspace(0, 1.2, 400)
for v, c, ls in ((0.2775, '#c00000', '-'), (0.15, '#ed7d31', '--'),
                 (0.05, '#2f75b5', ':')):
    d_stop = D0 + v * TAU + v * v / (2 * ABRAKE) + EPS
    axR.plot(d, ALPHA * (d - d_stop), ls, color=c, lw=2.4,
             label=f'接近速度 {v:.2f} m/s　($d_{{stop}}$ = {d_stop:.2f} m)')
axR.axhline(0, color='#666666', lw=1.2)
axR.fill_between(d, -1, 0, color='#fce4d6', alpha=0.45, zorder=0)
axR.text(0.02, -0.62, '要求退離', fontsize=12, color='#c00000',
         fontweight='bold')
axR.text(0.72, 0.62, '容許接近', fontsize=12, color='#2f75b5',
         fontweight='bold')
axR.axvline(SH_D_MED, color='#7f7f7f', lw=1.6, ls='-.')
axR.text(SH_D_MED + 0.02, -0.88, f'實測介入時\n間距中位 {SH_D_MED:.3f} m',
         fontsize=10.5, color='#444444')
axR.set_xlim(0, 1.2)
axR.set_ylim(-1.0, 1.6)
axR.set_xlabel('車體表面到回波的距離  $d_i$  [m]', fontsize=12)
axR.set_ylabel('容許的接近速度  $\\alpha\\,(d_i - d_{stop})$  [m/s]', fontsize=12)
axR.set_title('屏障形狀：不是急停，是隨距離收緊',
              fontsize=14, fontweight='bold')
axR.grid(alpha=0.25)
axR.legend(loc='upper left', fontsize=10.5, framealpha=0.95)
axR.spines[['top', 'right']].set_visible(False)
axR.text(0.99, 0.02,
         f'切向與遠離不受限制\n介入時修正量中位 {SH_DV_MED:.3f} m/s\n'
         f'整體速度代價 0.1%',
         transform=axR.transAxes, ha='right', va='bottom', fontsize=10.5,
         bbox=dict(fc='#eef5fb', ec='#9dc3e6', boxstyle='round,pad=0.45'))

out = sys.argv[1] if len(sys.argv) > 1 else \
    'evaluation/results/figs/shield_attribution.png'
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.tight_layout()
plt.savefig(out, dpi=125, bbox_inches='tight')
print(f'saved -> {out}')
