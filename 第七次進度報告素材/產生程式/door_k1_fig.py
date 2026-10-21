#!/usr/bin/env python3
"""K1 曲線：五項短測的門角 q(t)、門角速度、門扇對兩側牆接觸力（只讀 door_k1.json）。

    python3 evaluation/door_k1_fig.py evaluation/runs/door_k1_1     # 輸出 <run>/analysis/door_k1_curves.png 與 第七次素材圖表副本
"""
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                        # noqa: E402

run = sys.argv[1]
d = json.load(open(os.path.join(run, 'door_k1.json')))
chk = json.load(open(os.path.join(run, 'analysis', 'door_k1_check.json')))
for f in ('Noto Sans CJK TC', 'Noto Sans CJK JP'):
    if any(f in x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams['font.family'] = f
        break
names = list(d['tests'])
fig, axs = plt.subplots(3, len(names), figsize=(16, 7.5), dpi=150, sharex='col')
for j, n in enumerate(names):
    R = d['tests'][n]['rows']
    t = [r['t_post'] for r in R]
    P = d['tests'][n]['plan']
    axs[0, j].plot(t, [r['q_post_deg'] for r in R], color='#2a7ab0', lw=1.2)
    axs[0, j].axhline(110, color='#8c959f', lw=0.6, ls='--')
    axs[0, j].axhline(0, color='#8c959f', lw=0.6, ls='--')
    if P['force_s']:
        for a in axs[:, j]:
            a.axvspan(0, min(P['force_s'], max(t)) if d['tests'][n].get('force_stopped_at_s') is None else d['tests'][n]['force_stopped_at_s'],
                      color='#f2cc8f', alpha=0.35, lw=0)
    axs[0, j].set_title(f"{n}：q0 {P['q0_deg']:.0f}°、力 {P['force_n']:+.0f} N×{P['force_s']} s", fontsize=9)
    axs[1, j].plot(t, [r['qdot_post'] for r in R], color='#2e8b57', lw=1.0)
    axs[2, j].plot(t, [r['contact_walls_n'][0] for r in R], color='#d1495b', lw=1.0, label='鉸鏈側牆')
    axs[2, j].plot(t, [r['contact_walls_n'][1] for r in R], color='#6e7781', lw=1.0, label='門鎖側牆')
    axs[2, j].set_xlabel('t (s)', fontsize=8)
    for a in axs[:, j]:
        a.tick_params(labelsize=7)
        a.grid(color='#e6e8eb', lw=0.5)
axs[0, 0].set_ylabel('門角 q (°)', fontsize=8)
axs[1, 0].set_ylabel('q̇ (rad/s)', fontsize=8)
axs[2, 0].set_ylabel('門扇對牆接觸力 (N)', fontsize=8)
axs[2, 0].legend(fontsize=7)
v = {k: chk[k] for k in ('direction', 'lower_limit', 'upper_limit', 'passivity')}
fig.suptitle(f"K1 被動門資產的腳本外力診斷（door_k1_1；非機器人開門）　判定：方向 {v['direction']}、下限 {v['lower_limit']}、上限 {v['upper_limit']}、"
             f"被動 {v['passivity']}；讀回全 verified；阻尼不可解讀（{', '.join(chk['damping_report']['flags'])}）　黃底＝施力時窗",
             fontsize=9, x=0.01, ha='left')
fig.tight_layout()
out = os.path.join(run, 'analysis', 'door_k1_curves.png')
fig.savefig(out)
ws = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shutil.copy(out, os.path.join(ws, '第七次進度報告素材', '圖表', 'fig_door_k1_1_curves.png'))
print('寫出', out)
