#!/usr/bin/env python3
"""Make a two-panel slide figure from one saved RGB frame and a frozen model.

This is offline CPU inference and scientific plotting, not an Isaac run or AI
image generation. Both panels use identical source pixels and viewport. No
truth, depth, manually drawn object outline, or enhanced pixels enter inference.
"""
from __future__ import annotations

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
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from bottle_seg_infer import predict
from bottle_seg_train import acquire_lock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'evaluation/runs/bottle_rb1_1/w1_frames.npz'
WEIGHTS = ROOT / 'evaluation/dl0_ckpt/bottle_seg_v1_continue3/best.pth'
OUTPUT = ROOT / '第七次進度報告素材/圖表/瓶子模型遮罩_原圖與分割_20261104_v3'
CACHED = ROOT / '第七次進度報告素材/圖表/瓶子模型遮罩_原圖與分割_20261104'
FRAME_INDEX = 12  # Same capture (task 11.54 s) used in the existing RGB-D slide.
CROP = (338, 124, 578, 364)  # Fixed common square viewport in original 640x480 RGB.


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def draw_pair(rgb: np.ndarray, mask: np.ndarray, instance: dict, output: Path, labels: bool) -> None:
    x0, y0, x1, y1 = CROP
    fig = plt.figure(figsize=(10, 5.30 if labels else 5.0), dpi=160, facecolor='white')
    color = np.array([1.0, 0.43, 0.06])
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask, :3] = color
    overlay[mask, 3] = 0.46
    image_bottom, image_height = (0.018, 0.895) if labels else (0.01, 0.98)
    for i, title in enumerate(('原始影像', '模型分割結果')):
        ax = fig.add_axes((0.008 + i * 0.502, image_bottom, 0.482, image_height))
        ax.imshow(rgb, interpolation='nearest')
        if i == 1:
            ax.imshow(overlay, interpolation='nearest')
            ax.contour(mask.astype(float), levels=[0.5], colors=['#C55B00'], linewidths=1.45)
            bx0, by0, bx1, by1 = instance['box_xyxy']
            ax.add_patch(Rectangle(
                (bx0, by0), bx1 - bx0, by1 - by0,
                fill=False, edgecolor='#159447', linewidth=2.2, zorder=5,
            ))
            if labels:
                ax.text(
                    bx0, by0 - 4, f"bottle {instance['confidence']:.3f}",
                    fontsize=11.5, color='white', ha='left', va='bottom',
                    bbox=dict(facecolor='#159447', edgecolor='none', pad=3), zorder=6,
                )
        ax.set_xlim(x0 - 0.5, x1 - 0.5)
        ax.set_ylim(y1 - 0.5, y0 - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#D7DADD')
            spine.set_linewidth(0.8)
        if labels:
            ax.set_title(title, fontsize=19, color='#252525', pad=11, fontweight='medium')
    fig.savefig(output.with_suffix('.png'), dpi=160, facecolor='white')
    fig.savefig(output.with_suffix('.svg'), facecolor='white')
    plt.close(fig)


def main() -> None:
    outputs = [OUTPUT.with_suffix(ext) for ext in ('.png', '.svg', '.json', '.npz')]
    plain = OUTPUT.with_name(OUTPUT.name + '_無文字')
    outputs.extend(plain.with_suffix(ext) for ext in ('.png', '.svg'))
    if any(path.exists() for path in outputs):
        raise FileExistsError('Preserve existing figure artifacts; use a new version name.')

    # Shared lock prevents competing with an Isaac/replay/training workflow.
    # CPU-only inference does not allocate CUDA or download any model weights.
    lock = acquire_lock()
    try:
        torch.set_num_threads(2)
        source = np.load(SOURCE, allow_pickle=False)
        rgb = source['rgb'][FRAME_INDEX].copy()
        capture_ros = float(source['stamp'][FRAME_INDEX])
        assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        if CACHED.with_suffix('.npz').exists() and CACHED.with_suffix('.json').exists():
            prior = json.loads(CACHED.with_suffix('.json').read_text())
            assert prior['frame_index'] == FRAME_INDEX
            assert prior['source_sha256'] == digest(SOURCE)
            assert prior['weights_sha256'] == digest(WEIGHTS)
            masks = np.load(CACHED.with_suffix('.npz'), allow_pickle=False)['masks']
            instances, elapsed_ms = prior['instances'], prior['inference_ms']
            print('Reusing exact saved prediction; plotting only.', flush=True)
        else:
            model = maskrcnn_resnet50_fpn_v2(
                weights=None, weights_backbone=None, min_size=480, max_size=640,
            )
            model.roi_heads.box_predictor = FastRCNNPredictor(
                model.roi_heads.box_predictor.cls_score.in_features, 2,
            )
            model.roi_heads.mask_predictor = MaskRCNNPredictor(
                model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2,
            )
            model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
            model.eval()
            masks, instances, elapsed_ms = predict(model, Image.fromarray(rgb), torch.device('cpu'))
        if len(instances) != 1:
            raise RuntimeError(f'Expected one bottle prediction for this fixed frame, got {len(instances)}.')
        mask = masks[0].astype(bool)
        if not mask.any():
            raise RuntimeError('Empty model prediction; do not fabricate a mask.')
        print(json.dumps(dict(frame=FRAME_INDEX, stamp_ros_s=capture_ros,
                              instances=instances, inference_ms=elapsed_ms), ensure_ascii=False), flush=True)
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        font_manager.fontManager.addfont(font_path)
        plt.rcParams.update({
            'font.family': font_manager.FontProperties(fname=font_path).get_name(),
            'svg.fonttype': 'path',
            'axes.unicode_minus': False,
        })
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        draw_pair(rgb, mask, instances[0], OUTPUT, labels=True)
        draw_pair(rgb, mask, instances[0], plain, labels=False)
        np.savez_compressed(OUTPUT.with_suffix('.npz'), masks=masks)
        metadata = dict(
            run='bottle_rb1_1', camera='w1', frame_index=FRAME_INDEX,
            capture_stamp_ros_s=capture_ros,
            source=str(SOURCE.relative_to(ROOT)), source_sha256=digest(SOURCE),
            weights=str(WEIGHTS.relative_to(ROOT)), weights_sha256=digest(WEIGHTS),
            model='Mask R-CNN ResNet-50 FPN v2',
            input='Saved RGB only; no depth, truth, labels, or geometry provided to the model.',
            inference='Frozen checkpoint, offline CPU recomputation; not a new control trial.',
            inference_ms=elapsed_ms, instances=instances,
            selected_instance=0, crop_xyxy=list(CROP),
            selection='Fixed frame index 12, same capture already used in the RGB-D slide; no frame search.',
            display='Identical viewport; nearest-neighbor enlargement; orange predicted mask and green original model bounding box. Labeled version includes model class and confidence.',
            limitation='One simulator image illustration, not a segmentation accuracy evaluation.',
            outputs=[str(path.relative_to(ROOT)) for path in outputs],
        )
        OUTPUT.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        for path in outputs:
            print(path, flush=True)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
