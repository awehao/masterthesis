#!/usr/bin/env python3
"""Build a data-faithful segmentation figure from a saved frame and prediction.

No rendering, physics, training or inference here. Source pixels and masks are
displayed with identical axis limits; the wide image retains a context inset.
Selection: largest held-out wrist label wholly inside the image with >=12 px
margin (chosen before prediction, not by prediction quality).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
KEY = 'bottle_w1_2/w1/0009'
DATA = ROOT / 'evaluation/dl0_ckpt/bottle_seg_v1_data'
PRED = ROOT / 'evaluation/dl0_ckpt/bottle_seg_slide_w1_0009_20261102'
WEIGHTS = ROOT / 'evaluation/dl0_ckpt/bottle_seg_v1_continue3/best.pth'
OUT = ROOT / '第七次進度報告素材/圖表/瓶子分割_同影格局部放大_20261102_v2'


def digest(path):
    with path.open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minimal', action='store_true', help='Only three short panel labels for slides.')
    args = parser.parse_args()
    out = OUT.with_name('瓶子分割_同影格局部放大_20261102_精簡版') if args.minimal else OUT
    paths = [out.with_suffix(s) for s in ('.png', '.svg', '.json')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Figure exists; preserve it and use a new name.')
    rows = json.loads((DATA / 'index.json').read_text())['rows']
    row = next(r for r in rows if r['key'] == KEY)
    rgb_path, label_path = Path(row['rgb']), Path(row['label'])
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
    label = np.asarray(Image.open(label_path))
    gt = label == 1
    prediction = json.loads((PRED / 'prediction.json').read_text())
    assert Path(prediction['weights']).resolve() == WEIGHTS.resolve()
    masks = np.load(PRED / 'masks_0.npz')['masks']
    instances = prediction['predictions'][0]['instances']
    assert rgb.shape[:2] == label.shape == masks.shape[1:]
    assert len(masks) == len(instances) and len(masks) > 0
    chosen = int(np.argmax([r['confidence'] for r in instances]))
    mask, inst = masks[chosen].astype(bool), instances[chosen]
    yy, xx = np.where(gt)
    # Same viewport in all panels; no rotation, enhancement or synthetic pixels.
    x0, x1 = max(int(xx.min()) - 32, 0), min(int(xx.max()) + 33, rgb.shape[1])
    y0, y1 = max(int(yy.min()) - 22, 0), min(int(yy.max()) + 23, rgb.shape[0])
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False, 'svg.fonttype': 'path'})
    fig = plt.figure(figsize=(12, 6.75) if args.minimal else (16, 9),
                     facecolor='white', dpi=160)
    if not args.minimal:
        fig.text(.045, .94, '瓶子實例分割：同一影格的局部放大對照',
                 fontsize=24, fontweight='bold', color='#172b4d')
        fig.text(.045, .894, 'Mask R-CNN · ResNet-50 FPN v2｜既有凍結權重｜手腕相機實際模擬影格',
                 fontsize=13, color='#52616b')
    colors = {'label': '#0072B2', 'pred': '#D55E00'}
    titles = ['① 原始 RGB（局部放大）', '② 幾何／深度自動標註',
              f'③ 模型預測（信心 {inst["confidence"]:.3f}）']
    if args.minimal:
        titles = ['原圖', '自動標註', '模型預測']
    for i, title in enumerate(titles):
        bounds = [.045 + .315 * i, .055, .28, .82] if args.minimal else [.045 + .315 * i, .245, .28, .585]
        ax = fig.add_axes(bounds)
        ax.imshow(rgb, interpolation='nearest')
        if i in (1, 2):
            shown = gt if i == 1 else mask
            color = matplotlib.colors.to_rgb(colors['label' if i == 1 else 'pred'])
            rgba = np.zeros((*label.shape, 4))
            rgba[shown, :3] = color
            rgba[shown, 3] = .52
            ax.imshow(rgba, interpolation='nearest')
            if shown.any():
                ax.contour(shown.astype(float), levels=[.5], colors=[color], linewidths=1.0)
            if i == 1:
                ignore = np.zeros((*label.shape, 4))
                ignore[label == 2] = [.7, .7, .7, .60]
                ax.imshow(ignore, interpolation='nearest')
        ax.set_xlim(x0 - .5, x1 - .5)
        ax.set_ylim(y1 - .5, y0 - .5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=22 if args.minimal else 15, pad=16 if args.minimal else 12,
                     color='#172b4d')
        for spine in ax.spines.values():
            spine.set_color('#bcc8d1')
    if not args.minimal:
        context = fig.add_axes([.045, .035, .19, .17])
        context.imshow(rgb, interpolation='nearest')
        context.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    fill=False, linewidth=1.7, edgecolor='#D55E00'))
        context.set_axis_off()
        context.set_title('原始全景；橘框＝上方放大範圍', fontsize=10, pad=2)
        fig.text(.265, .168, '三欄使用完全相同的裁切範圍；僅顯示放大，不改原始影像或遮罩。',
                 fontsize=13, color='#172b4d')
        fig.text(.265, .123, '藍色＝模擬幾何＋深度自動標註；橘色＝模型預測；灰色＝忽略邊界。',
                 fontsize=12, color='#52616b')
        fig.text(.265, .078, f'來源：{KEY}｜held-out 影格示例，非整體準確率或跨物體驗證。',
                 fontsize=11, color='#52616b')
    fig.savefig(paths[0], dpi=160, facecolor='white')
    fig.savefig(paths[1], facecolor='white')
    plt.close(fig)
    meta = dict(frame_key=KEY, split=row['split'], raw=str(rgb_path.relative_to(ROOT)),
                label=str(label_path.relative_to(ROOT)), prediction=str((PRED / 'prediction.json').relative_to(ROOT)),
                masks=str((PRED / 'masks_0.npz').relative_to(ROOT)),
                crop_xyxy=[x0, y0, x1, y1], source_size=list(rgb.shape[:2][::-1]),
                chosen_instance=chosen, confidence=inst['confidence'],
                weights=str(WEIGHTS.relative_to(ROOT)), weights_sha256=digest(WEIGHTS),
                raw_sha256=digest(rgb_path), label_sha256=digest(label_path),
                selection='largest test wrist label with >=12px image margin; selected before inference',
                processing='scientific figure only; identical viewport; nearest-neighbor display; no image generation',
                label_source='capture-time simulator cylinder geometry + depth; not manually drawn',
                limitation='one same-scene example; crops enlarge pixels and do not add detail',
                minimal_labels=args.minimal)
    paths[2].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
    for path in paths:
        print(path)


if __name__ == '__main__':
    main()
