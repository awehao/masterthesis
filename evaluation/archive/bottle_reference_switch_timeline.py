#!/usr/bin/env python3
"""Continuous measured-state evidence around the task-reference switch.

Reads one existing run. No screenshots, new simulation, planning, or inference.
The figure describes measured behaviour and the lift-off guard, not command
continuity, a tracking-error ablation, or a universal safety guarantee.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'evaluation/runs/bottle_rb1_1'
OUT = ROOT / '第七次進度報告素材/圖表/任務參考切換_連續量測曲線_20261103_v1'
sys.path.insert(0, str(ROOT / 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics


def digest(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def main():
    outputs = [OUT.with_suffix(ext) for ext in ('.png', '.svg', '.json')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Preserve previous figures; use a new version.')
    report = json.loads((RUN / 's1_report.json').read_text())
    switch = float(report['paired_strategy']['switch']['t'])
    compact = next(float(e['t']) for e in report['events'] if e['what'] == 'compact_start')
    with np.load(RUN / 's1_steps.npz', allow_pickle=False) as data:
        all_rows = data['rows']
        cols = {k: i for i, k in enumerate(data['cols'].tolist())}
        start = all_rows[all_rows[:, cols['phase_index']] == 3, cols['t']][0]
        selected = (all_rows[:, cols['t']] >= start - 1e-8) & (all_rows[:, cols['t']] < compact - 1e-8)
        rows = all_rows[selected].copy()
    t = rows[:, cols['t']]
    qi = [cols[k] for k in ['base_x', 'base_y', 'base_yaw'] + [f'q{i}' for i in range(1, 7)]]
    q = rows[:, qi]
    urdf = ROOT / 'evaluation/models/omni_bot_wholebody_expanded.urdf'
    kin = WholeBodyKinematics.from_urdf_file(str(urdf))
    tcp = np.array([kin.fk(state, 'link_tcp')[:3, 3] for state in q])
    c, s = np.cos(q[:, 2]), np.sin(q[:, 2])
    dp = tcp[:, :2] - q[:, :2]
    relative_x = c*dp[:, 0] + s*dp[:, 1]
    series = dict(
        time_s=t.tolist(), base_world_dx_mm=((q[:, 0] - q[0, 0])*1000).tolist(),
        tcp_world_dx_mm=((tcp[:, 0] - tcp[0, 0])*1000).tolist(),
        tcp_base_relative_dx_mm=((relative_x-relative_x[0])*1000).tolist(),
        bottle_bottom_gap_mm=((rows[:, cols['bottle_base_z']] - report['table1_z'])*1000).tolist(),
        bottle_table_force_N=rows[:, cols['f_bottle_table']].tolist(),
    )
    switch_index = int(np.argmin(abs(t-switch)))
    assert abs(t[switch_index]-switch) < 1e-7
    after = t >= switch - 1e-8
    summary = dict(
        switch_time_s=switch, grasp_evidence_time_s=report['t_grasp'],
        final_lift_judgement_time_s=report['t_lift_ok'],
        switch_gap_threshold_mm=report['paired_strategy']['g_switch']*1000,
        switch_gap_judge_mm=report['paired_strategy']['switch']['gap']*1000,
        switch_gap_step_mm=series['bottle_bottom_gap_mm'][switch_index],
        switch_table_force_N=series['bottle_table_force_N'][switch_index],
        post_switch_relative_x_net_mm=float((relative_x[-1]-relative_x[switch_index])*1000),
        post_switch_relative_x_peak_to_peak_mm=float(np.ptp(relative_x[after])*1000),
        entire_window_s=[float(t[0]), float(t[-1])], n_samples=len(t),
    )
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    plt.rcParams.update({
        'font.family': font_manager.FontProperties(fname=font_path).get_name(),
        'svg.fonttype': 'path', 'axes.unicode_minus': False,
    })
    blue, orange, purple, gray = '#087EB0', '#CA5C18', '#852B99', '#657382'
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.4), dpi=180, sharex=True)
    fig.subplots_adjust(left=.14, right=.985, bottom=.105, top=.985, hspace=.15)
    axes[0].plot(t, series['base_world_dx_mm'], color=blue, lw=2.7, label='底盤')
    axes[0].plot(t, series['tcp_world_dx_mm'], color=orange, lw=2.5, ls='--', label='TCP')
    axes[0].set_ylabel('環境 Δx\n（mm）', fontsize=13)
    axes[0].set_ylim(-10, max(series['base_world_dx_mm'] + series['tcp_world_dx_mm']) + 22)
    axes[1].plot(t, series['tcp_base_relative_dx_mm'], color='#009E73', lw=2.6)
    axes[1].set_ylabel('TCP 底盤相對 Δx\n（mm）', fontsize=13)
    axes[2].plot(t, series['bottle_bottom_gap_mm'], color=blue, lw=2.6)
    axes[2].axhline(summary['switch_gap_threshold_mm'], color=gray, lw=1.5,
                   ls=':', label='間隙門檻')
    axes[2].set_ylabel('瓶底間隙\n（mm）', fontsize=13)
    axes[2].set_ylim(-4, 56)
    axes[2].set_yticks([0, 20, 40])
    axes[2].set_xlabel('任務時間（s）', fontsize=13)
    axes[2].legend(loc='upper left', fontsize=11, frameon=False)
    for ax in axes:
        ax.axvline(switch, color=purple, lw=1.7, ls='--', label='參考切換')
        ax.set_xlim(t[0]-.04, t[-1]+.04)
        ax.set_xticks([20, 21, 22, 23])
        ax.grid(axis='y', color='#E2E7ED', linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#BAC4CE')
        ax.tick_params(labelsize=11, colors=gray)
    axes[0].legend(loc='upper left', ncol=3, frameon=False, fontsize=11)
    fig.savefig(outputs[0], dpi=180, facecolor='white')
    fig.savefig(outputs[1], facecolor='white')
    plt.close(fig)
    outputs[2].write_text(json.dumps(dict(
        run=RUN.name, summary=summary, series=series,
        source_sha256={name: digest(RUN/name) for name in ['s1_steps.npz', 's1_report.json']},
        urdf_path=str(urdf.relative_to(ROOT)), urdf_sha256=digest(urdf),
        window_selection='Entire CLOSE start through the sample immediately before compact_start. No favourable endpoint search.',
        baseline='World x and base-relative x are each baseline-subtracted at the same CLOSE-start sample, not re-zeroed at the switch.',
        gap_definition='Recorded world bottle-bottom z minus scene table1_z, used only for evaluation. The guard gap in judge logging and the physics-step readback differ by their sampling convention.',
        processing='Native data plot only; recorded measured joints and URDF forward kinematics; no simulation, inference, or control changes.',
        limitation='One execution, descriptive x-component motion evidence. Not a proof of reference/command continuity, tracking accuracy, causal benefit, or safety for other conditions.',
    ), ensure_ascii=False, indent=2) + '\n')
    for path in outputs:
        print(path)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
