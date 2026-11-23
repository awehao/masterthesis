"""朝向 OFF／ON 的結果圖。**保留每一次的數值，不只畫平均。**

時間軸一律使用 `arrival_time_s`（`isaac_result.arrival_fields`）：
自**首個相符 /plan** 起算到停止觸發。這與 `motion_start_sim_t`
（首次非零 /cmd_vel）是不同的時刻，程式中明訂兩者不可互相替代，
故未到達的趟次此欄為 None，**不畫成到達時間**。

輸出：
  fig1_5seed.png       五個 seed 的總表（OFF→ON 連線；逾時另以叉號標示）
  fig2_repeats.png     seed 1／8 每一次配對各一條線（三次全部畫出）
  fig3_encounter.png   遭遇事件的前／中／後（同欄共用縱軸）
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
OFFC, ONC, MISS = '#B45309', '#1D4ED8', '#6B7280'
LEG = [plt.Line2D([], [], marker='o', ls='', color=OFFC, label='OFF（gmpc_scan）'),
       plt.Line2D([], [], marker='o', ls='', color=ONC, label='ON（gmpc_scan_heading）')]

# ══════════════════════════════════════════════════════ 圖 1
# seed: 直線m,（OFF, ON）×（夾角中位, <15%, 路徑, 到達時間, 轉角, 最小取樣間距）
# 到達時間 None = 未到達（task_timeout）
S5 = {
 1:  (16.90, (126.81, 8.25), (0.0, 71.1), (17.733, 17.744), (78.630, 80.430), (12.39, 354.95), (0.1195, 0.0524)),
 7:  (19.09, (55.24, 11.52), (4.7, 63.4), (22.298, 22.291), (98.620, 100.300), (29.60, 539.13), (0.1495, 0.1438)),
 8:  (14.95, (88.17, 13.12), (5.4, 54.6), (19.374, 19.802), (86.820, 90.080), (17.73, 627.33), (0.0863, 0.4204)),
 13: (12.13, (45.40, 10.45), (14.2, 71.7), (13.953, 13.840), (None, None), (13.02, 359.25), (0.0534, 0.0464)),
 24: (22.34, (123.66, 12.60), (0.0, 57.0), (24.678, 25.483), (109.120, 116.190), (32.00, 725.18), (0.0829, 0.1026)),
}
TITLES = ['車頭−行進方向夾角 中位（°）', '夾角 <15° 的時間占比（%）',
          '實際路徑長度（m）',
          '到達時間，自首個相符 /plan 起算（模擬秒）',
          '累計絕對轉角（°，對數軸）',
          '底盤近似表面至動態障礙的\n最小取樣間距（m）']
LOG = [False, False, False, False, True, False]
seeds = sorted(S5)
fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.2))
for k, ax in enumerate(axes.ravel()):
    for i, sd in enumerate(seeds):
        o, n = S5[sd][k + 1]
        if o is None or n is None:            # 逾時：不畫成到達時間，也不連線
            ax.plot(i, 180.0, 'x', color=MISS, ms=10, mew=2.2, zorder=3)
            ax.annotate('逾時；未到達\n（180 s 模擬時間上限，\n自發布目標起算）',
                        xy=(i, 180.0), xytext=(i - 0.05, 158), fontsize=6.6,
                        ha='center', color=MISS, linespacing=1.5)
            continue
        ax.plot([i - .16, i + .16], [o, n], '-', color='#9CA3AF', lw=1.4, zorder=1)
        ax.plot(i - .16, o, 'o', color=OFFC, ms=7, zorder=2)
        ax.plot(i + .16, n, 'o', color=ONC, ms=7, zorder=2)
    if LOG[k]:
        ax.set_yscale('log')
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'seed {s}\n{S5[s][0]:.1f} m' for s in seeds], fontsize=8)
    ax.set_title(TITLES[k], fontsize=9.5)
    ax.grid(alpha=.25, axis='y'); ax.margins(x=.14)
axes[1, 0].set_ylim(60, 195)
fig.suptitle('五組起終點的 OFF → ON 配對（每組一次，全部數值保留）', fontsize=12.5)
fig.legend(handles=LEG + [plt.Line2D([], [], marker='x', ls='', color=MISS,
                                     mew=2, label='逾時；未到達')],
           loc='lower center', ncol=3, frameon=False, fontsize=9,
           bbox_to_anchor=(.5, .085))
fig.text(.5, .012,
         'seed 下方的公尺數為「起終點直線距離」，非路徑長。'
         '最小取樣間距 = ‖p_robot − p_obs‖ − 障礙半徑（0.15–0.82 m，逐顆不同）'
         ' − 機器人半徑 0.30 m（整車以圓盤近似）；取樣率即真值位姿發布率。\n'
         '到達時間採 arrival_time_s（自首個相符 /plan 至停止觸發）；'
         'seed 13 兩趟皆 task_timeout，該欄為未定義，故以叉號另標而不連成配對線。',
         ha='center', fontsize=7.4, color='#4B5563', linespacing=1.7)
fig.tight_layout(rect=[0, .115, 1, .955])
fig.savefig('evaluation/results/fig1_5seed.png'); plt.close(fig)

# ══════════════════════════════════════════════════════ 圖 2
REP = {
 1: [((126.81, 8.25), (17.733, 17.744), (78.630, 80.430), (12.39, 354.95), (0.1195, 0.0524)),
     ((127.94, 8.25), (17.700, 17.763), (78.790, 80.260), (10.04, 351.75), (0.1121, 0.0536)),
     ((126.29, 8.19), (17.751, 17.753), (78.870, 80.340), (10.84, 354.76), (0.0978, 0.0496))],
 8: [((88.17, 13.12), (19.374, 19.802), (86.820, 90.080), (17.73, 627.33), (0.0863, 0.4204)),
     ((88.14, 11.71), (19.402, 19.513), (87.170, 89.220), (16.53, 613.58), (0.0895, 0.3759)),
     ((87.41, 12.35), (19.293, 19.554), (87.130, 89.290), (16.66, 593.94), (0.0680, 0.3569))],
}
T2 = ['車頭−行進方向夾角 中位（°）', '實際路徑長度（m）',
      '到達時間，自首個相符 /plan\n起算（模擬秒）',
      '累計絕對轉角（°，對數軸）',
      '底盤近似表面至動態障礙的\n最小取樣間距（m）']
LOG2 = [False, False, False, True, False]
MK = ['o', 's', '^']
fig, axes = plt.subplots(2, 5, figsize=(15, 7.4))
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
            ax.set_title(T2[k], fontsize=9)
        if k == 0:
            ax.set_ylabel(f'seed {sd}', fontsize=11.5)
fig.suptitle('seed 1 與 seed 8：各三次配對，每一次單獨畫出', fontsize=12.5)
fig.legend(handles=LEG + [plt.Line2D([], [], marker=m, ls='', color='#4B5563',
                                     label=f'第 {i+1} 次')
                          for i, m in enumerate(MK)],
           loc='lower center', ncol=5, frameon=False, fontsize=9,
           bbox_to_anchor=(.5, .075))
fig.text(.5, .016,
         '各面板縱軸範圍不同；路徑與時間採局部放大，請依刻度解讀。'
         '累計轉角為對數軸。連線僅配對同一次試驗的 OFF 與 ON，不表示中間過程。\n'
         'seed 1、8 為「五 seed 批次中最小間距變化方向相反」而刻意挑選的診斷案例，'
         '不是隨機抽樣。',
         ha='center', fontsize=7.4, color='#4B5563', linespacing=1.7)
fig.tight_layout(rect=[0, .105, 1, .95])
fig.savefig('evaluation/results/fig2_repeats.png'); plt.close(fig)

# ══════════════════════════════════════════════════════ 圖 3
rows = list(csv.DictReader(open('evaluation/results/encounters_20260911.csv')))
PH = ['前', '中', '後']
COLS = [('vyb', '|v_y,body|（m/s）'), ('ang', '車頭−行進方向夾角（°）'),
        ('wz', '|ω_z|（rad/s）')]
fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.6), sharey='col')
for row, side in enumerate(['OFF', 'ON']):
    R = [r for r in rows if r['side'] == side]
    col = OFFC if side == 'OFF' else ONC
    for k, (key, lab) in enumerate(COLS):
        ax = axes[row, k]
        M = []
        for r in R:
            y = [float(r[f'{p}_{key}']) for p in PH]
            if any(np.isnan(y)):
                continue
            ax.plot(range(3), y, '-', color=col, alpha=.30, lw=1)
            M.append(y)
        M = np.array(M)
        ax.plot(range(3), np.median(M, axis=0), '-o', color=col, lw=2.6, ms=7,
                zorder=3, label=f'事件間中位（n={len(M)}）')
        ax.set_xticks(range(3)); ax.set_xticklabels(PH, fontsize=11)
        ax.grid(alpha=.25, axis='y'); ax.legend(fontsize=8, frameon=False)
        if row == 0:
            ax.set_title(lab, fontsize=10)
        if k == 0:
            ax.set_ylabel(side, fontsize=12.5)
fig.suptitle('遭遇事件摘要（探索性）：同一欄的 OFF／ON 共用縱軸', fontsize=12.5)
fig.text(.5, .018,
         '每模式 17 個事件、來自 9 趟；事件非獨立樣本（seed 1 每趟 3 個 × 3 趟、'
         'seed 8 每趟 1 個 × 3 趟，共 9 類 seed／障礙物組合）。\n'
         '階段切分（分析前固定，只依距離）：淨距 ≤ 1.00 m 為一事件、其內最近需 ≤ 0.50 m；'
         '前 = 進入 1.00 m 至首次跌破 0.50 m，中 = ≤ 0.50 m，後 = 末次升回 0.50 m 至離開 1.00 m。\n'
         '細線 = 一個事件；粗線 = 先取每事件各階段的中位，再取事件間的中位。'
         '折線只連接三個階段摘要，不是連續時間軌跡。',
         ha='center', fontsize=7.4, color='#4B5563', linespacing=1.75)
fig.tight_layout(rect=[0, .105, 1, .95])
fig.savefig('evaluation/results/fig3_encounter.png'); plt.close(fig)
print('三張圖已更新')
