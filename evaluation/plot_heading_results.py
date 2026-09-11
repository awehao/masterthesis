"""朝向 OFF／ON 的結果圖。**保留每一次的數值，不只畫平均。**

輸出：
  fig1_5seed.png       五個 seed 的總表（每個指標一格，OFF→ON 連線）
  fig2_repeats.png     seed 1／8 每一次配對各一條線（三次全部畫出）
  fig3_encounter.png   遭遇事件的前／中／後（每個事件一條細線 ＋ 中位數粗線）
"""
import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

for f in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc',
          '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'):
    if os.path.exists(f):
        fm.fontManager.addfont(f); break
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'],
                     'axes.unicode_minus': False, 'figure.dpi': 130})
OFFC, ONC = '#B45309', '#1D4ED8'

# ---------------------------------------------------------------- 圖 1
# seed: 直線m, (OFF, ON) ×（夾角中位, <15%, 路徑, 任務時間, 轉角, 淨距）
S5 = {
 1:  (16.90, (126.81, 8.25), (0.0, 71.1), (17.733, 17.744), (78.550, 80.360), (12.39, 354.95), (0.1195, 0.0524)),
 7:  (19.09, (55.24, 11.52), (4.7, 63.4), (22.298, 22.291), (98.620, 100.210), (29.60, 539.13), (0.1495, 0.1438)),
 8:  (14.95, (88.17, 13.12), (5.4, 54.6), (19.374, 19.802), (86.820, 90.000), (17.73, 627.33), (0.0863, 0.4204)),
 13: (12.13, (45.40, 10.45), (14.2, 71.7), (13.953, 13.840), (179.880, 179.920), (13.02, 359.25), (0.0534, 0.0464)),
 24: (22.34, (123.66, 12.60), (0.0, 57.0), (24.678, 25.483), (109.070, 116.110), (32.00, 725.18), (0.0829, 0.1026)),
}
TITLES = ['車頭−行進夾角 中位（°）', '夾角 <15° 的時間占比（%）', '實際路徑長度（m）',
          '任務時間（s）', '累計絕對轉角（°）', '動態最近距離（m）']
LOG = [False, False, False, False, True, False]
seeds = sorted(S5)
fig, axes = plt.subplots(2, 3, figsize=(13, 7.2))
for k, ax in enumerate(axes.ravel()):
    for i, sd in enumerate(seeds):
        o, n = S5[sd][k + 1]
        ax.plot([i - .16, i + .16], [o, n], '-', color='#9CA3AF', lw=1.4, zorder=1)
        ax.plot(i - .16, o, 'o', color=OFFC, ms=7, zorder=2)
        ax.plot(i + .16, n, 'o', color=ONC, ms=7, zorder=2)
    if LOG[k]:
        ax.set_yscale('log')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'seed {s}\n{S5[s][0]:.1f} m' for s in seeds], fontsize=8)
    ax.set_title(TITLES[k], fontsize=10)
    ax.grid(alpha=.25, axis='y')
    ax.margins(x=.12)
# 任務時間那格（axes[1,0]）才是 seed 13 需要註解的地方
axes[1, 0].annotate('seed 13 兩趟皆逾時\n容差參考點不一致，非避障失敗',
                    xy=(3, 179.9), xytext=(0.55, 152), fontsize=7.5,
                    ha='center',
                    arrowprops=dict(arrowstyle='->', lw=.8, color='#666'))
fig.suptitle('五組起終點的 OFF → ON 配對（每組一次，全部數值保留）', fontsize=12)
fig.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=OFFC, label='OFF (gmpc_scan)'),
                    plt.Line2D([], [], marker='o', ls='', color=ONC, label='ON (gmpc_scan_heading)')],
           loc='lower center', ncol=2, frameon=False, fontsize=9)
fig.tight_layout(rect=[0, .045, 1, .955])
fig.savefig('evaluation/results/fig1_5seed.png'); plt.close(fig)

# ---------------------------------------------------------------- 圖 2
REP = {
 1: [((126.81, 8.25), (17.733, 17.744), (78.550, 80.360), (12.39, 354.95), (0.1195, 0.0524)),
     ((127.94, 8.25), (17.700, 17.763), (78.760, 80.160), (10.04, 351.75), (0.1121, 0.0536)),
     ((126.29, 8.19), (17.751, 17.753), (78.820, 80.290), (10.84, 354.76), (0.0978, 0.0496))],
 8: [((88.17, 13.12), (19.374, 19.802), (86.820, 90.000), (17.73, 627.33), (0.0863, 0.4204)),
     ((88.14, 11.71), (19.402, 19.513), (87.100, 89.170), (16.53, 613.58), (0.0895, 0.3759)),
     ((87.41, 12.35), (19.293, 19.554), (87.060, 89.200), (16.66, 593.94), (0.0680, 0.3569))],
}
T2 = ['車頭−行進夾角 中位（°）', '實際路徑長度（m）', '任務時間（s）',
      '累計絕對轉角（°）', '動態最近距離（m）']
LOG2 = [False, False, False, True, False]
MK = ['o', 's', '^']
fig, axes = plt.subplots(2, 5, figsize=(15, 6.4))
for row, sd in enumerate((1, 8)):
    for k in range(5):
        ax = axes[row, k]
        for j, rep in enumerate(REP[sd]):
            o, n = rep[k]
            ax.plot([0, 1], [o, n], '-', color='#9CA3AF', lw=1.2, zorder=1)
            ax.plot(0, o, MK[j], color=OFFC, ms=6.5, zorder=2)
            ax.plot(1, n, MK[j], color=ONC, ms=6.5, zorder=2)
        if LOG2[k]:
            ax.set_yscale('log')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['OFF', 'ON'], fontsize=9)
        ax.set_xlim(-.35, 1.35); ax.grid(alpha=.25, axis='y')
        if row == 0:
            ax.set_title(T2[k], fontsize=9.5)
        if k == 0:
            ax.set_ylabel(f'seed {sd}', fontsize=11)
fig.suptitle('seed 1 與 seed 8：各三次配對，每一次單獨畫出（○ 第1次  □ 第2次  △ 第3次）',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, .945])
fig.savefig('evaluation/results/fig2_repeats.png'); plt.close(fig)

# ---------------------------------------------------------------- 圖 3
rows = list(csv.DictReader(open(
    '/tmp/claude-1000/-home-howardchen-masterthesis/'
    '757c950c-2cbf-43e3-a66b-44e0527a4873/scratchpad/encounters.csv')))
PH = ['前', '中', '後']
fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
for row, side in enumerate(['OFF', 'ON']):
    R = [r for r in rows if r['side'] == side]
    col = OFFC if side == 'OFF' else ONC
    for k, (key, lab) in enumerate([('vyb', '|v_y,body|（m/s）'),
                                    ('ang', '車頭−行進夾角（°）'),
                                    ('wz', '|ω_z|（rad/s）')]):
        ax = axes[row, k]
        M = []
        for r in R:
            y = [float(r[f'{p}_{key}']) for p in PH]
            if any(np.isnan(y)):
                continue
            ax.plot(range(3), y, '-', color=col, alpha=.28, lw=1)
            M.append(y)
        M = np.array(M)
        ax.plot(range(3), np.median(M, axis=0), '-o', color=col, lw=2.6, ms=7,
                zorder=3, label=f'中位（n={len(M)}）')
        ax.set_xticks(range(3)); ax.set_xticklabels(PH, fontsize=11)
        ax.grid(alpha=.25, axis='y'); ax.legend(fontsize=8, frameon=False)
        if row == 0:
            ax.set_title(lab, fontsize=10)
        if k == 0:
            ax.set_ylabel(side, fontsize=12)
fig.suptitle('遭遇事件的前／中／後（判準事前固定：進入 1.00 m 為事件、'
             '最近需 ≤ 0.50 m；每條細線 = 一個事件）', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, .94])
fig.savefig('evaluation/results/fig3_encounter.png'); plt.close(fig)
print('fig1_5seed.png / fig2_repeats.png / fig3_encounter.png 已輸出')
