#!/usr/bin/env python3
"""WG4-B 正式結果圖（只讀實錄與 wg4b_formal_results.json；開度一律取 room_run.json 逐物理步）。

  wg4b_opening.png  左：全任務視圖（模擬時間；標開啟帶、各趟 OPEN_HOLD 起點、P 的 DONE、B1 的 240 s 時限截止；
                    P 完成後不補畫到 240 s）。右：以 OPEN_HOLD 起點對齊的局部視圖
  wg4b_paired.png   共同區段（ALIGN、ENGAGE_WAIT、OPEN）三對 B1 − P 配對差：段長、位置 p95、姿態 p95、
                    手臂命令增量 p95、核心求解 p50（零線、三對、n = 3）

    python3 evaluation/wg4b_figures.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OD = os.path.join(HERE, 'results', 'wg4b', 'figures')
C = {'B1': '#c2410c', 'P': '#1d4ed8'}
LAB = {'B1': 'B1（單步 QP 改編基線，kp 1.0）', 'P': 'P（W-GMPC H5）'}
plt.rcParams.update({'font.family': ['Noto Sans CJK JP'], 'axes.unicode_minus': False,
                     'axes.spines.top': False, 'axes.spines.right': False})


def physical(rid):
    room = json.load(open(os.path.join(HERE, 'runs', rid, 'room_run.json')))
    ci = room['steps_cols'].index
    t = [s[1] for s in room['steps']]
    op = [s[ci('opening_m')] * 1e3 for s in room['steps']]
    task = json.load(open(os.path.join(HERE, 'runs', rid, 'task.json')))
    ph = {}
    for e in task['events']:
        if 'phase' in e and e['phase'] not in ph:
            ph[e['phase']] = float(e['sim_t'])
    return t, op, ph


def main():
    d = json.load(open(os.path.join(HERE, 'results', 'wg4b', 'wg4b_formal_results.json')))
    os.makedirs(OD, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={'width_ratios': [1.6, 1]})
    seen = set()
    for r in d['runs']:
        m = r['method']
        t, op, ph = physical(r['rid'])
        lab = LAB[m] if m not in seen else None
        seen.add(m)
        a1.plot(t, op, color=C[m], lw=1.3, alpha=0.85, label=lab)
        if 'OPEN_HOLD' in ph:
            a1.plot(ph['OPEN_HOLD'], 196, marker='v', ms=5, color=C[m])
            th = ph['OPEN_HOLD']
            sel = [(x - th, y) for x, y in zip(t, op) if th - 12 <= x <= th + 25]
            a2.plot([x for x, _ in sel], [y for _, y in sel], color=C[m], lw=1.4, alpha=0.85)
        if 'DONE' in ph:
            a1.plot(ph['DONE'], 0, marker='o', ms=6, color=C[m], mfc='white')
    for ax in (a1, a2):
        ax.axhspan(195, 205, color='#9ca3af', alpha=0.28, lw=0)
        ax.axhline(220, color='#6b7280', lw=0.8, ls=':')
        ax.grid(axis='y', color='#e5e7eb', lw=0.6)
        ax.set_ylim(-5, 232)
    a1.axvline(240, color='#374151', lw=1.0, ls='--')
    a1.text(238, 120, '模擬時限 240 s\n（B1 三趟於此截止，\n未到 DONE）', ha='right', fontsize=8, color='#374151')
    a1.text(3, 199, '開啟帶 195–205 mm', fontsize=8, color='#374151', va='center')
    a1.text(3, 223, '行程端 220 mm', fontsize=8, color='#374151')
    a1.plot([], [], 'v', color='#374151', ms=5, label='OPEN_HOLD 起點')
    a1.plot([], [], 'o', color='#374151', mfc='white', ms=6, label='DONE（P）')
    a1.set_xlim(0, 245)
    a1.set_xlabel('模擬時間（s）')
    a1.set_ylabel('抽屜開度（mm，逐物理步）')
    a1.set_title('全任務：P 3/3 完成；B1 0/3（停在 OPEN_HOLD 至時限）', fontsize=10)
    a1.legend(frameon=False, fontsize=8, loc='center left', bbox_to_anchor=(0.0, 0.62))
    a2.axvline(0, color='#6b7280', lw=0.8)
    a2.set_xlim(-12, 25)
    a2.set_xlabel('自 OPEN_HOLD 起點的時間（s）')
    a2.set_title('OPEN_HOLD 對齊局部：P 帶內 2.1–2.2 s 後進 CLOSE；\nB1 0.6–0.9 s 內超帶後外移', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OD, 'wg4b_opening.png'), dpi=160)
    plt.close(fig)

    phases = ['ALIGN', 'ENGAGE_WAIT', 'OPEN']
    mets = [('duration_s', '段長差（s）', 1), ('err_p_mm_p95', '位置誤差 p95 差（mm）', 1),
            ('err_r_rad_p95', '姿態誤差 p95 差（mrad）', 1e3), ('du_arm_p95', '手臂命令增量 p95 差（rad/s）', 1),
            ('core_ms_p50', '核心求解 p50 差（ms）', 1)]
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.6))
    for ax, (k, lab, sc) in zip(axs, mets):
        for j, p in enumerate(phases):
            for i, pr in enumerate(sorted(d['pairs'])):
                v = d['pairs'][pr]['diff_B1_minus_P'][p][k]
                if isinstance(v, (int, float)):
                    ax.plot(j + (i - 1) * 0.14, v * sc, 'o', ms=7, color='#374151',
                            mfc=['#ffffff', '#9ca3af', '#374151'][i],
                            label=(f'配對 {pr}' if j == 0 else None))
        ax.axhline(0, color='#6b7280', lw=0.8)
        ax.set_xticks(range(3), ['ALIGN', 'ENGAGE\nWAIT', 'OPEN'], fontsize=8)
        ax.set_title(lab, fontsize=9)
        ax.grid(axis='y', color='#e5e7eb', lw=0.6)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle('共同區段 B1 − P 配對差（> 0 = B1 較大；n = 3 對，不做顯著性；不含 OPEN_HOLD）', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OD, 'wg4b_paired.png'), dpi=160)
    plt.close(fig)
    print('ok')


if __name__ == '__main__':
    main()
