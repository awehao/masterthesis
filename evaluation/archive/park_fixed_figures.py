#!/usr/bin/env python3
"""PARK_FIXED v1 功能確認（runs/pf_func_F）：底盤相對錨點偏離與實測速率 vs 時間，標靜止閘門、轉移、違規、手臂關節速率。

    python3 evaluation/park_fixed_figures.py → results/motm_speed/figures/park_fixed_v1_drift.png
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'], 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False})


def main():
    R = os.path.join(HERE, 'runs', 'pf_func_F')
    r = json.load(open(os.path.join(R, 'room_run.json')))
    g = json.load(open(os.path.join(R, 'park_gate.json')))
    c = r['steps_cols'].index
    S = [s for s in r['steps'] if 44.5 <= s[1] <= 54.5]
    t = np.array([s[1] for s in S])
    P = np.array([s[c('base_xyth')] for s in S])
    Q = np.array([s[c('q_arm_meas')] for s in S])
    an = np.array(g['gate']['anchor'])
    dev = np.linalg.norm(P[:, :2] - an[:2], axis=1) * 1e3
    v = np.r_[np.nan, np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) / np.diff(t)] * 1e3
    qd = np.r_[np.nan, np.abs(np.diff(Q, axis=0) / np.diff(t)[:, None]).max(1)]
    w0, w1 = g['gate']['window']
    t_tr = g['hold']['start'][1]
    t_v = g['violation']['t']
    fig, axs = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)
    for ax in axs:
        ax.axvspan(w0, w1, color='#86efac', alpha=0.35, lw=0)
        ax.axvline(t_tr, color='#1d4ed8', lw=1.0, ls='--')
        ax.axvline(t_v, color='#b91c1c', lw=1.0, ls='--')
        ax.grid(axis='y', color='#e5e7eb', lw=0.6)
    axs[0].plot(t, dev, color='#374151', lw=1.4)
    axs[0].axhline(1.0, color='#b91c1c', lw=0.8, ls=':')
    axs[0].set_ylabel('相對錨點偏離 (mm)')
    axs[0].text(t_tr + 0.1, 3.5, '轉給全身（展開開始）', color='#1d4ed8', fontsize=8)
    axs[0].text(t_v + 0.1, 2.5, '違規閂鎖 47.26 s', color='#b91c1c', fontsize=8)
    axs[0].text(w0, 0.3, '靜止閘門窗', color='#166534', fontsize=8)
    axs[1].plot(t, v, color='#374151', lw=1.0)
    axs[1].axhline(1.0, color='#b91c1c', lw=0.8, ls=':')
    axs[1].set_ylabel('實測底盤速率 (mm/s)')
    axs[1].set_ylim(0, 2.0)
    axs[2].plot(t, qd, color='#6b7280', lw=1.0)
    axs[2].set_ylabel('手臂關節速率最大 (rad/s)')
    axs[2].set_xlabel('模擬時間 (s)；底盤套用命令全程 0')
    fig.suptitle('PARK_FIXED v1 功能確認：零底盤命令下，手臂展開後底盤持續漂移（門檻 1 mm/s、1 mm 以紅點線標示）', fontsize=10)
    fig.tight_layout()
    od = os.path.join(HERE, 'results', 'motm_speed', 'figures')
    os.makedirs(od, exist_ok=True)
    fig.savefig(os.path.join(od, 'park_fixed_v1_drift.png'), dpi=160)
    print('ok')


if __name__ == '__main__':
    main()
