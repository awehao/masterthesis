#!/usr/bin/env python3
"""Whole-body-control presentation figure from saved R-B1 imagery.

Data-flow arrows are explanatory annotations, not measured vectors. The camera
image and replay scene belong to the same run at different timestamps. No new
simulation, inference, training, or synthetic robot imagery.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / '第七次進度報告素材/圖表/rb1_keyframes/rb1_t19.7.png'
CAMERA = ROOT / '第七次進度報告素材/圖表/acq_chain_frames/rb1_accept_t12.57.png'
OUT = ROOT / '第七次進度報告素材/圖表/視覺目標驅動_全身協調主圖_20261102'


def main():
    paths = [OUT.with_suffix(s) for s in ('.png', '.svg', '.json')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Preserve existing outputs; use another filename.')
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'svg.fonttype': 'path', 'axes.unicode_minus': False})
    scene = np.asarray(Image.open(SCENE).convert('RGB'))
    camera = np.asarray(Image.open(CAMERA).convert('RGB'))
    navy, blue, orange, gray = '#152C46', '#087FAD', '#C75C1C', '#647589'
    fig = plt.figure(figsize=(16, 9), facecolor='white', dpi=180)
    fig.text(.045, .944, '視覺目標驅動的全身協調追蹤',
             fontsize=30, color=navy, weight='bold')
    fig.text(.045, .894, '由量測目標，聯合決定底盤與手臂命令', fontsize=17, color=gray)

    # Hero: true scene viewport. Video title and diagnostic dashboard are excluded.
    ax = fig.add_axes([.045, .205, .615, .645])
    ax.imshow(scene, interpolation='nearest')
    ax.set_xlim(150, 1140)
    ax.set_ylim(735, 150)
    ax.set_axis_off()
    # This ring is an object callout, not an estimated projected TCP or a mask.
    ax.add_patch(Circle((583, 326), 37, fill=False, linewidth=1.8, edgecolor=orange))
    ax.annotate('抓取目標', xy=(565, 294), xytext=(330, 223),
                fontsize=18, color=orange,
                arrowprops={'arrowstyle': '-', 'color': orange, 'lw': 1.4})

    # Right rail: vision input into one joint solver, not two independent controls.
    fig.text(.818, .832, '視覺量測', ha='center', fontsize=19, color=navy)
    cam = fig.add_axes([.707, .641, .222, .165])
    cam.imshow(camera, interpolation='nearest')
    cam.set_xlim(180, 620)
    cam.set_ylim(345, 135)
    cam.set_axis_off()
    fig.text(.818, .605, 'RGB-D → 抓取目標', ha='center', fontsize=15, color=gray)
    fig.add_artist(FancyArrowPatch((.818, .577), (.818, .531),
                   transform=fig.transFigure, arrowstyle='-|>',
                   mutation_scale=18, lw=1.7, color=gray))
    fig.add_artist(FancyBboxPatch((.707, .414), .222, .109,
                   boxstyle='round,pad=0.006,rounding_size=0.008',
                   facecolor=navy, edgecolor='none', transform=fig.transFigure))
    fig.text(.818, .476, '全身 GMPC', ha='center', va='center',
             fontsize=24, color='white', weight='bold')
    fig.text(.818, .438, '同一個耦合求解', ha='center', va='center',
             fontsize=12, color='#C6D4E3')

    # Two output branches are connected to a single solver, with distinct colors.
    for x, label, color in [(.746, '底盤命令', blue), (.89, '手臂命令', orange)]:
        fig.add_artist(FancyArrowPatch((.818, .408), (x, .345),
                       transform=fig.transFigure, arrowstyle='-|>',
                       mutation_scale=18, lw=1.8, color=color,
                       connectionstyle='angle,angleA=-90,angleB=0,rad=7'))
        fig.text(x, .308, label, ha='center', fontsize=16, color=color)

    # Minimal image callouts identify the two coordinated parts, not velocity data.
    ax.annotate('手臂追蹤', xy=(725, 460), xytext=(925, 345),
                fontsize=18, color=orange,
                arrowprops={'arrowstyle': '-', 'color': orange, 'lw': 1.5})
    ax.annotate('底盤移動', xy=(740, 661), xytext=(935, 650),
                fontsize=18, color=blue,
                arrowprops={'arrowstyle': '-', 'color': blue, 'lw': 1.5})

    fig.add_artist(plt.Line2D([.045, .955], [.162, .162],
                   transform=fig.transFigure, color='#DCE4EC', lw=1))
    fig.text(.5, .107, '車子前進，夾爪仍對準目標。', ha='center',
             fontsize=24, weight='bold', color=navy)
    fig.text(.045, .037, 'R-B1｜相機擷取影格＋當趟姿態重播｜連線為控制關係示意',
             fontsize=10, color=gray)
    fig.text(.955, .037, 'Isaac 物理模擬', ha='right', fontsize=10, color=gray)
    fig.savefig(paths[0], dpi=180, facecolor='white')
    fig.savefig(paths[1], facecolor='white')
    plt.close(fig)
    meta = {
        'run': 'bottle_rb1_1',
        'scene_task_time_approx_s': 19.7,
        'camera_task_time_approx_s': 12.57,
        'sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (SCENE, CAMERA)},
        'diagram': 'Explanatory control relationships; output arrows are not measured vectors.',
        'target_ring': 'Illustrative object callout, not a projected estimated grasp pose.',
        'disclosure': 'Same run, different timestamps; scene is recorded-pose replay, camera is capture-time RGB.',
        'implementation': 'GMPC outputs base twist and arm setpoint-rate commands; perception and reference generation are external.',
        'processing': 'Viewport crops and native vector annotation only; no synthetic scene or image generation.'
    }
    paths[2].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    for p in paths:
        print(p)


if __name__ == '__main__':
    main()
