#!/usr/bin/env python3
"""RGB-only inference for the bottle segmentation pilot, no truth/depth/map input.

    .venv-dl0/bin/python evaluation/bottle_seg_infer.py --weights .../best.pth \
        --image image.png --out NEW_DIRECTORY

Offline CLI shares the Isaac/replay lock. ROS/control integration is separate.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import dl0_train as D
from bottle_seg_train import acquire_lock, temperatures


def predict(model, image, dev):
    tensor = torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2, 0, 1).float().to(dev) / 255
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        prediction = model([tensor])[0]
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    keep = (prediction['scores'] >= D.SCORE_T) & (prediction['labels'] == 1)
    masks = (prediction['masks'][keep, 0] >= D.BIN_T).cpu().numpy()
    records = [dict(class_name='bottle', confidence=float(s), box_xyxy=[float(x) for x in b])
               for s, b in zip(prediction['scores'][keep].cpu(), prediction['boxes'][keep].cpu())]
    return masks, records, elapsed


def overlay(image, masks, records):
    img = np.array(image.convert('RGB'), copy=True)
    if len(masks):
        fg = masks.any(0)
        img[fg] = (img[fg].astype(float)*.45 + np.array([0, 255, 0])*.55).astype(np.uint8)
    result = Image.fromarray(img)
    draw = ImageDraw.Draw(result)
    for row in records:
        draw.rectangle(row['box_xyxy'], outline='lime', width=2)
        x, y = row['box_xyxy'][:2]
        draw.text((x, max(0.,y-12)), f'bottle {row["confidence"]:.3f}', fill='lime')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--image')
    inputs.add_argument('--demo-index', help='Four fixed held-out examples, selected by camera and label size, not prediction quality')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    lock = acquire_lock()
    temperatures()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # No accidental ImageNet download at inference time.
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=480, max_size=640)
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    model = model.to(dev)
    model.load_state_dict(torch.load(a.weights, map_location=dev, weights_only=True))
    model.eval()
    if a.image:
        examples = [dict(rgb=a.image, key=Path(a.image).name)]
    else:
        rows = json.loads(Path(a.demo_index).read_text())['rows']
        examples = []
        for cam in ('v1', 'w1'):
            for kind in ('positive', 'negative'):
                rr = sorted([r for r in rows if r['split']=='test' and r['kind']==kind and f'/{cam}/' in r['key']], key=lambda r:r['visible'])
                examples.append(rr[len(rr)//2])
    outputs = []
    sheet = Image.new('RGB', (1280, 1000)) if len(examples) == 4 else None
    for i, row in enumerate(examples):
        temperatures()
        img = Image.open(row['rgb']).convert('RGB')
        masks, records, ms = predict(model, img, dev)
        ov = overlay(img, masks, records)
        ov.save(out / f'overlay_{i}.png')
        np.savez_compressed(out / f'masks_{i}.npz', masks=masks)
        outputs.append(dict(key=row['key'], instances=records, model_only_ms=ms))
        if sheet is not None:
            x,y = i%2*640, i//2*500
            sheet.paste(ov, (x,y))
            ImageDraw.Draw(sheet).text((x+5,y+480), row['key']+' RGB-only inference', fill='white')
    if sheet is not None:
        sheet.save(out / 'demo.png')
    result = dict(predictions=outputs, input='RGB only', weights=str(Path(a.weights).resolve()))
    (out / 'prediction.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    lock.close()


if __name__ == '__main__':
    main()
