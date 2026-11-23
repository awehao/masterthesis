#!/usr/bin/env python3
"""Three fixed-TCP IK configurations illustrating whole-body coordination.

Uses recorded robot state only as an IK seed. Base displacements are illustrative,
not a controller rollout. No physics, model training, or GMPC solve is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

import bottle_reach_scan as R


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'evaluation/runs/bottle_rb1_1/s1_steps.npz'
URDF = ROOT / 'evaluation/models/omni_bot_wholebody_expanded.urdf'
OUT = ROOT / '第七次進度報告素材/圖表/全身協調_底盤前進與手臂補償_運動學示意_20261103'
NAVY, GRAY, BLUE, ORANGE = '#18334F', '#697C8F', '#137EAA', '#C65D16'


def main():
    outputs = [OUT.with_suffix(s) for s in ('.png', '.svg', '.json')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('Preserve existing outputs; choose another filename.')
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'svg.fonttype': 'path', 'axes.unicode_minus': False})
    data = np.load(SOURCE)
    cols = {str(k): i for i, k in enumerate(data['cols'])}
    row = data['rows'][np.argmin(abs(data['rows'][:, cols['t']] - 19.72))]
    names = ['base_x', 'base_y', 'base_yaw'] + [f'q{i}' for i in range(1, 7)]
    seed = np.array([row[cols[n]] for n in names])
    target = R.K.fk(seed, 'link_tcp')
    configurations, checks = [], []
    previous = seed[3:].copy()
    for dx in (0., .08, .16):
        base = seed[:3].copy()
        base[0] += dx
        ok, _, arm, pe, re, margin = R.ik(base, target, [previous])
        if not ok:
            raise RuntimeError(f'Illustrative configuration unreachable: dx={dx}')
        configurations.append(np.r_[base, arm])
        checks.append(dict(base_dx_m=dx, ik_position_error_m=pe,
                           ik_rotation_error_rad=re, joint_margin_rad=margin))
        previous = arm
    origin = np.r_[seed[:2], 0.]
    target_p = target[:3, 3] - origin

    fig = plt.figure(figsize=(15, 7.5), facecolor='white', dpi=190)
    fig.text(.035, .94, '底盤前進，手臂協調維持同一個 TCP 目標',
             fontsize=24, color=NAVY, weight='bold')
    fig.text(.035, .887, '接觸靜止物體時的協調原理', fontsize=15, color=GRAY)
    ax = fig.add_axes([.005, .13, .69, .73], projection='3d', computed_zorder=False)
    ax.set_proj_type('ortho')
    ax.view_init(elev=23, azim=-63)
    ax.set_box_aspect((1.12, 1.3, .92))
    ax.set_xlim(-.39, .75)
    ax.set_ylim(-.40, .91)
    ax.set_zlim(0., .93)
    ax.set_axis_off()

    def world(points):
        return np.asarray(points, float) - origin

    # Light table geometry provides context; no clearance/safety claim is made.
    table = world([[.2, 1.16, .58], [1., 1.16, .58], [1., 1.66, .58], [.2, 1.66, .58]])
    ax.add_collection3d(Poly3DCollection([table], facecolor='#D8D2C4',
                        edgecolor='#BBB5AA', alpha=.16, linewidths=.8, zorder=0))
    for x, y in [(.27, 1.39), (.93, 1.39), (.27, 1.59), (.93, 1.59)]:
        points = world([[x, y, .02], [x, y, .58]])
        ax.plot(*points.T, color='#D3CEC4', lw=3, alpha=.65, zorder=0)

    theta = np.linspace(0, 2 * np.pi, 60)
    arm_links = ['link_base'] + [f'link{i}' for i in range(1, 7)] + ['link_eef', 'link_tcp']
    pose_colors = ['#D9DEE4', '#8D9AA8', NAVY]
    for i, q in enumerate(configurations):
        center = np.r_[q[:2], 0.] - origin
        bx, by = center[:2]
        style = [':', '--', '-'][i]
        alpha = [.40, .65, 1.][i]
        for z in (.05, .33):
            ax.plot(bx + .30 * np.cos(theta), by + .30 * np.sin(theta),
                    np.full_like(theta, z), color=BLUE, alpha=alpha, lw=1.7,
                    linestyle=style, zorder=2 + i)
        if i == 2:
            top = np.column_stack([bx + .30 * np.cos(theta), by + .30 * np.sin(theta),
                                   np.full_like(theta, .33)])
            ax.add_collection3d(Poly3DCollection([top], facecolor='#65B2D2',
                                edgecolor='none', alpha=.15, zorder=2))
            for angle in (0., np.pi / 2, np.pi, 3 * np.pi / 2):
                ax.plot([bx + .30 * np.cos(angle)] * 2,
                        [by + .30 * np.sin(angle)] * 2, [.05, .33],
                        color=BLUE, lw=1.2, alpha=.6, zorder=3)
        points = world([R.K.fk(q, name)[:3, 3] for name in arm_links])
        ax.plot(*points.T, color=ORANGE, alpha=alpha, linestyle=style,
                lw=4.8 if i == 2 else 2.8, solid_capstyle='round', zorder=5 + i)
        ax.scatter(*points[1:-1].T, s=30 if i == 2 else 13, color='white',
                   edgecolors=ORANGE, alpha=alpha, linewidths=1.2, zorder=8 + i)
        ax.scatter(bx, by, .335, s=25, color=BLUE, alpha=alpha, zorder=8 + i)

    # Cylinder is contextual, from the same recorded state; target is the seed FK.
    bottle_xy = np.array([row[cols['bottle_base_x']], row[cols['bottle_base_y']]]) - origin[:2]
    bottle_z = float(row[cols['bottle_base_z']])
    ring = np.linspace(0, 2 * np.pi, 25)
    xx, zz = np.meshgrid(ring, [bottle_z, bottle_z + .20])
    ax.plot_surface(bottle_xy[0] + .015 * np.cos(xx), bottle_xy[1] + .015 * np.sin(xx), zz,
                    color='#6AC2D3', alpha=.75, linewidth=0, shade=False, zorder=7)
    ax.scatter(*target_p, marker='+', s=240, color='#C63C35', linewidths=2.5, zorder=15)

    # The blue arrow is the assumed base-displacement direction, not a measured velocity.
    ax.quiver(-.18, -.37, .02, .53, 0., 0., color=BLUE, lw=2.4,
              arrow_length_ratio=.10, zorder=15)

    # Right: both predictive models feed one coupled task, then two command streams.
    def box(x, y, w, h, title, color=NAVY, fill='white', fs=13, text_color=None):
        fig.add_artist(FancyBboxPatch((x, y), w, h,
                        boxstyle='round,pad=0.004,rounding_size=0.006',
                        transform=fig.transFigure, facecolor=fill, edgecolor=color, lw=1.2))
        fig.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                 fontsize=fs, color=text_color or color, linespacing=1.4)

    def arrow(a, b, color=GRAY):
        fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure,
                        arrowstyle='-|>', mutation_scale=15, lw=1.5, color=color))

    box(.708, .765, .265, .055, '實測狀態＋在途命令', color=GRAY, fs=12)
    box(.708, .632, .12, .085, '底盤\n預測', color=BLUE, fill='#F0F7FB', fs=13)
    box(.853, .632, .12, .085, '手臂＋設定點\n預測', color=ORANGE, fill='#FEF5EB', fs=12)
    arrow((.84, .761), (.768, .72), BLUE)
    arrow((.84, .761), (.913, .72), ORANGE)
    arrow((.768, .627), (.80, .579), BLUE)
    arrow((.913, .627), (.882, .579), ORANGE)
    box(.708, .442, .265, .132, '耦合 GMPC\n共同 TCP 任務＋限制', fill=NAVY,
        fs=17, text_color='white')
    arrow((.80, .437), (.768, .387), BLUE)
    arrow((.882, .437), (.913, .387), ORANGE)
    box(.708, .292, .12, .09, '底盤命令\n平面速度', color=BLUE, fs=12)
    box(.853, .292, .12, .09, '手臂命令\n設定點速率', color=ORANGE, fs=12)

    # Place callouts in screen coordinates after the 3D projection has settled.
    fig.canvas.draw()

    def projected(point):
        x, y, _ = proj3d.proj_transform(*point, ax.get_proj())
        return fig.transFigure.inverted().transform(ax.transData.transform((x, y)))

    target_screen = projected(target_p)
    fig.text(.385, .825, '同一 TCP 目標', fontsize=16, color='#C63C35', ha='center')
    fig.add_artist(FancyArrowPatch((.385, .806), tuple(target_screen),
                   transform=fig.transFigure, arrowstyle='-', color='#C63C35', lw=1.3))
    base_screen = projected(np.array([.12, -.37, .02]))
    fig.text(.20, .177, '底盤向前', fontsize=16, color=BLUE, ha='center')
    fig.add_artist(FancyArrowPatch((.20, .21), tuple(base_screen),
                   transform=fig.transFigure, arrowstyle='-', color=BLUE, lw=1.1))
    elbow = world([R.K.fk(configurations[-1], 'link3')[:3, 3]])[0]
    fig.text(.075, .66, '手臂構型調整', fontsize=15, color=ORANGE)
    fig.add_artist(FancyArrowPatch((.23, .66), tuple(projected(elbow)),
                   transform=fig.transFigure, arrowstyle='-', color=ORANGE, lw=1.2))

    handles = [Line2D([0], [0], color=ORANGE, alpha=a, lw=3, linestyle=s)
               for a, s in zip([.40, .65, 1.], [':', '--', '-'])]
    fig.legend(handles, ['構型①', '構型②', '構型③'], loc='lower left',
               bbox_to_anchor=(.055, .08), ncol=3, frameon=False, fontsize=12)
    fig.text(.705, .187, '兩組命令由同一個末端任務連結', fontsize=13, color=NAVY)
    fig.text(.035, .035, '運動學示意：固定 TCP 位姿，以底盤位移求 IK；非 GMPC 預測輸出／非實測軌跡。',
             fontsize=10, color=GRAY)
    fig.savefig(outputs[0], dpi=190, facecolor='white')
    fig.savefig(outputs[1], facecolor='white')
    plt.close(fig)
    meta = {
        'diagram_type': 'kinematic illustration, NOT a GMPC prediction or measured trajectory',
        'seed_run': 'bottle_rb1_1', 'seed_task_time_s': float(row[cols['t']]),
        'seed_q': seed.tolist(), 'target_T': target.tolist(),
        'illustrative_base_dx_m': [0., .08, .16],
        'q_configurations': [q.tolist() for q in configurations], 'ik_checks': checks,
        'fixed_target_rule': 'Entire TCP pose fixed at seed FK; base y/yaw unchanged; arm-only continuation IK.',
        'not_checked': 'No time law, actuation dynamics, command limits, continuous collision or stopping verification; not an executable plan.',
        'display_origin': origin.tolist(), 'model': str(URDF.relative_to(ROOT)),
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (SOURCE, URDF, Path(__file__), ROOT / 'evaluation/bottle_reach_scan.py')},
        'processing': 'Native vector scientific diagram; no new simulation, training, GMPC solve or image generation.'
    }
    outputs[2].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    for p in outputs:
        print(p)


if __name__ == '__main__':
    main()
