#!/usr/bin/env python3
"""Presentation diagram using saved segmentation and replay imagery.

This is a conceptual pipeline, not a same-run timing comparison. The coordinate
graphic is explicitly schematic; it contains no invented measured coordinates.
No physics, image generation, training, inference, or control changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'evaluation/dl0_ckpt/bottle_seg_v1_data/test/bottle_w1_2/w1/0009_rgb.png'
MASKS = ROOT / 'evaluation/dl0_ckpt/bottle_seg_slide_w1_0009_20261102/masks_0.npz'
PRED = MASKS.parent / 'prediction.json'
SCENE = ROOT / '第七次進度報告素材/圖表/rb1_keyframes/rb1_t19.7.png'
OUT = ROOT / '第七次進度報告素材/圖表/視覺到GMPC移動抓取_流程圖_20261102'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    outputs = [OUT.with_suffix(s) for s in ('.png', '.svg', '.json')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Preserve existing figures; choose a new stem.')
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'svg.fonttype': 'path', 'axes.unicode_minus': False})
    rgb = np.asarray(Image.open(RAW).convert('RGB'))
    scene = np.asarray(Image.open(SCENE).convert('RGB'))
    pred = json.loads(PRED.read_text())
    instances = pred['predictions'][0]['instances']
    selected = int(np.argmax([item['confidence'] for item in instances]))
    mask = np.load(MASKS)['masks'][selected].astype(bool)
    assert mask.shape == rgb.shape[:2]
    orange, navy, gray = '#D55E00', '#18334F', '#627387'
    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor='white')
    fig.text(.045, .92, '從看見瓶子，到移動抓取', fontsize=29, weight='bold', color=navy)
    xs = [.035, .278, .521, .764]
    titles = ['相機畫面', 'CNN 找出瓶子', '深度算出位置', 'GMPC 移動抓取']
    captions = ['RGB 影像', '瓶子遮罩', '三維目標 (x, y, z)', '底盤＋手臂']
    for i, (x, title, caption) in enumerate(zip(xs, titles, captions)):
        fig.add_artist(FancyBboxPatch((x, .225), .201, .575,
                       boxstyle='round,pad=0.008,rounding_size=0.012',
                       transform=fig.transFigure, facecolor='#F6F8FB',
                       edgecolor='#DEE5ED', linewidth=1.2, zorder=0))
        fig.text(x + .1005, .749, title, ha='center', fontsize=18, color=navy)
        fig.text(x + .1005, .265, caption, ha='center', fontsize=16, color=gray)
        if i < 3:
            fig.add_artist(FancyArrowPatch((x + .208, .495), (xs[i + 1] - .014, .495),
                           transform=fig.transFigure, arrowstyle='-|>',
                           mutation_scale=19, linewidth=2.0, color=gray))

    # 1: keep some table context; mark the bottle with a callout, not a new label.
    ax = fig.add_axes([xs[0] + .009, .333, .183, .35])
    ax.imshow(rgb, interpolation='nearest')
    ax.set_xlim(85, 330)
    ax.set_ylim(460, 240)
    ax.annotate('瓶子', xy=(207, 358), xytext=(278, 300), fontsize=14, color=navy,
                ha='center', arrowprops={'arrowstyle': '->', 'color': orange, 'lw': 2})
    ax.set_axis_off()

    # 2: actual predicted pixels alone, so segmentation is visually explicit.
    ax = fig.add_axes([xs[1] + .009, .333, .183, .35])
    rgba = np.zeros((*mask.shape, 4))
    rgba[mask] = [*matplotlib.colors.to_rgb(orange), 1.0]
    ax.imshow(rgba, interpolation='nearest')
    ax.set_xlim(165, 250)
    ax.set_ylim(428, 305)
    ax.set_axis_off()

    # 3: coordinate icon is a schematic, not a measured point-cloud reconstruction.
    ax = fig.add_axes([xs[2] + .009, .333, .183, .35])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    origin = (.20, .20)
    for end, name in [((.88, .20), 'x'), ((.08, .40), 'y'), ((.20, .88), 'z')]:
        ax.annotate('', xy=end, xytext=origin,
                    arrowprops={'arrowstyle': '-|>', 'color': gray, 'lw': 1.7})
        ax.text(end[0] + .025, end[1], name, fontsize=16, color=gray)
    point = (.65, .64)
    ax.plot([point[0], point[0], origin[0]], [point[1], origin[1], origin[1]],
            color='#A8B5C4', linestyle='--', lw=1.4)
    ax.plot([origin[0], point[0]], [point[1], point[1]],
            color='#A8B5C4', linestyle='--', lw=1.4)
    ax.scatter(*point, s=220, marker='+', color=orange, linewidths=3)
    ax.annotate('抓取目標', xy=point, xytext=(.70, .82), ha='center',
                fontsize=15, color=orange)
    ax.set_axis_off()

    # 4: actual replay scene, cropped by viewport to omit the video dashboard.
    ax = fig.add_axes([xs[3] + .009, .333, .183, .35])
    ax.imshow(scene, interpolation='nearest')
    ax.set_xlim(360, 940)
    ax.set_ylim(745, 170)
    ax.set_axis_off()

    fig.text(.5, .14, '視覺提供目標，GMPC 協調底盤與手臂。',
             ha='center', fontsize=23, color=navy)
    fig.text(.5, .055, '流程示意｜影像取自不同模擬紀錄；座標圖為示意',
             ha='center', fontsize=10, color=gray)
    fig.savefig(outputs[0], dpi=160, facecolor='white')
    fig.savefig(outputs[1], facecolor='white')
    plt.close(fig)
    meta = {
        'figure_type': 'conceptual perception-to-control pipeline',
        'segmentation_frame': 'bottle_w1_2/w1/0009',
        'selected_instance': selected,
        'scene_run': 'bottle_rb1_1',
        'scene_approx_task_time_s': 19.7,
        'source_files': {str(p.relative_to(ROOT)): digest(p) for p in [RAW, MASKS, PRED, SCENE]},
        'stage_3': 'schematic axes and target; no measured numeric coordinates claimed',
        'disclosure': 'Separate source runs. Not a same-run or synchronized experiment.',
        'target_assumptions': 'Known bottle radius and grasp band; depth/localization and validity checks are separate from CNN.',
        'omitted_for_readability': 'Online feasibility checks, target freshness checks, planner and safety layers.',
        'processing': 'Code-native scientific illustration with faithful source viewports; no image generation or new simulation.'
    }
    outputs[2].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
