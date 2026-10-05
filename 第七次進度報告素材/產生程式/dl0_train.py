#!/usr/bin/env python3
"""DL0 第一版學習式把手遮罩：Mask R-CNN R50-FPN v2（COCO 預訓練）微調；規格 results/vision/DL0_train_spec.md。

在 .venv-dl0 執行（torch cu128）：
    .venv-dl0/bin/python evaluation/dl0_train.py --out evaluation/dl0_ckpt/run1 [--max-iters N]（短程試跑）

要點（Codex 審查後的必修）：
* 影像尺寸固定：min_size=480、max_size=640 ⇒ 640×480 影像內部縮放倍率 1（啟動時核對）。
* **ignore 排除於遮罩損失**（不是所有損失）：目標遮罩以三值編碼（0 背景、1 前景、2 ignore），以 nearest 方式通過內部 resize；
  自訂 `maskrcnn_loss_ignore` 取代 torchvision 的遮罩損失：前景與 ignore 各自做 RoI 投影，ignore 投影 ≥ 0.5 的位置權重 0；
  損失＝Σ(w·BCE)／max(Σw, 1)。框與分類損失不變。
* 群組均衡取樣：每格權重＝1／該群組訓練影格數；每 epoch 600 次迭代、batch 2、上限 20 epoch（12,000 次迭代）。
* 選模型：開發群組逐格 IoU（ignore 排除；分數最高單一實例、分數門檻 0.5、遮罩二值化 0.5；漏檢＝0），兩群組各自平均後等權平均；
  負樣本另報誤檢率；不把空對空算 1。
* CPU 熱保護：k10temp ≥ 92 °C 即停並存檔。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import roi_heads as RH
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import roi_align

HERE = os.path.dirname(os.path.abspath(__file__))
IDX = os.path.join(HERE, 'dl0_ckpt', 'labels', 'index.json')
SEED, BATCH, LR, MOM, WD = 0, 2, 0.005, 0.9, 1e-4
EPOCHS, ITERS_PER_EPOCH, LR_DROP_EPOCH = 20, 600, 15
SCORE_T, BIN_T, CPU_LIMIT_C = 0.5, 0.5, 92


# ------------------------------------------------------------------ ignore 遮罩損失
def _project(m, proposals, idxs, M):
    rois = torch.cat([idxs[:, None].to(proposals), proposals], dim=1)
    return roi_align(m[:, None].to(rois), rois, (M, M), 1.0)[:, 0]


def maskrcnn_loss_ignore(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    """gt_masks：每張影像 [N, H, W] 三值（0 背景、1 前景、2 ignore）。ignore 位置不貢獻損失與梯度。"""
    M = mask_logits.shape[-1]
    labels = [gl[i] for gl, i in zip(gt_labels, mask_matched_idxs)]
    fg = [_project((gm == 1).float(), p, i, M) for gm, p, i in zip(gt_masks, proposals, mask_matched_idxs)]
    ig = [_project((gm == 2).float(), p, i, M) for gm, p, i in zip(gt_masks, proposals, mask_matched_idxs)]
    labels = torch.cat(labels, dim=0)
    fg = torch.cat(fg, dim=0)
    ig = torch.cat(ig, dim=0)
    if fg.numel() == 0:
        return mask_logits.sum() * 0
    logits = mask_logits[torch.arange(labels.shape[0], device=labels.device), labels]
    w = (ig < 0.5).float()
    loss = F.binary_cross_entropy_with_logits(logits, fg, weight=w, reduction='sum')
    return loss / w.sum().clamp(min=1.0)


def install_ignore_loss():
    RH.maskrcnn_loss = maskrcnn_loss_ignore            # RoIHeads.forward 以模組層名稱呼叫


# ------------------------------------------------------------------ 資料
def load_index():
    return json.load(open(IDX))['rows']


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = np.asarray(Image.open(os.path.join(HERE, r['rgb'])).convert('RGB')).copy()
        lab = np.asarray(Image.open(os.path.join(HERE, r['label']))).copy()
        if self.train:
            if random.random() < 0.5:                  # 水平翻轉：影像、前景、ignore 一起翻
                img, lab = img[:, ::-1].copy(), lab[:, ::-1].copy()
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.train:
            t = torchvision.transforms.ColorJitter(0.2, 0.2, 0.2)(t)
        fg = lab == 1
        tgt = {'labels': torch.zeros((0,), dtype=torch.int64), 'boxes': torch.zeros((0, 4)),
               'masks': torch.zeros((0,) + lab.shape, dtype=torch.uint8)}
        if fg.any():
            ys, xs = np.nonzero(fg)
            box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            tgt = {'labels': torch.tensor([1]), 'boxes': torch.tensor([box], dtype=torch.float32),
                   'masks': torch.from_numpy(lab.astype(np.uint8))[None]}
        return t, tgt, r


def collate(b):
    return tuple(zip(*b))


# ------------------------------------------------------------------ 模型
def build_model(pretrained=True):
    w = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    m = maskrcnn_resnet50_fpn_v2(weights=w, min_size=480, max_size=640)
    inf = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(inf, 2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2)
    return m


def check_no_resize(model, dev):
    model.eval()
    x = [torch.zeros(3, 480, 640, device=dev)]
    il, _ = model.transform(x)
    assert tuple(il.image_sizes[0]) == (480, 640), il.image_sizes
    return il.image_sizes[0]


# ------------------------------------------------------------------ 評估
@torch.no_grad()
def evaluate(model, rows, dev):
    model.eval()
    per_group, neg_fp, n_neg, excluded = {}, 0, 0, []
    ms = []
    for r in rows:
        img = np.asarray(Image.open(os.path.join(HERE, r['rgb'])).convert('RGB'))
        lab = np.asarray(Image.open(os.path.join(HERE, r['label'])))
        t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().to(dev) / 255.0
        t0 = time.perf_counter()
        o = model([t])[0]
        torch.cuda.synchronize() if dev.type == 'cuda' else None
        ms.append((time.perf_counter() - t0) * 1e3)
        keep = o['scores'] >= SCORE_T
        pred = (o['masks'][keep][0, 0] >= BIN_T).cpu().numpy() if keep.any() else np.zeros(lab.shape, bool)
        if r['kind'] == 'negative':
            n_neg += 1
            neg_fp += int(keep.any())
            continue
        fg, ig = lab == 1, lab == 2
        if not fg.any():
            excluded.append(r['key'])
            continue
        ev = ~ig
        inter = (pred & fg & ev).sum()
        union = ((pred | fg) & ev).sum()
        per_group.setdefault(r['group'], []).append(float(inter / max(union, 1)))
    gm = {g: float(np.mean(v)) for g, v in per_group.items()}
    score = float(np.mean(list(gm.values()))) if gm else 0.0
    return {'score_equal_weight_groups': score, 'per_group_mean_iou': gm, 'n_per_group': {g: len(v) for g, v in per_group.items()},
            'negatives': n_neg, 'negative_false_positive': neg_fp, 'excluded_no_evaluable_fg': excluded,
            'infer_ms_p50': float(np.median(ms)) if ms else None}


def cpu_temp():
    for h in glob.glob('/sys/class/hwmon/hwmon*'):
        try:
            if open(os.path.join(h, 'name')).read().strip() == 'k10temp':
                return int(open(os.path.join(h, 'temp1_input')).read()) / 1000
        except OSError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-iters', type=int, default=None, help='短程試跑：總迭代上限（覆蓋 20×600）')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert dev.type == 'cuda', '需要 CUDA；不改用 CPU（環境不符即停）'
    install_ignore_loss()
    rows = load_index()
    tr = [r for r in rows if r['split'] == 'train']
    dv = [r for r in rows if r['split'] == 'dev']
    gsz = {}
    for r in tr:
        gsz[r['group']] = gsz.get(r['group'], 0) + 1
    wts = torch.tensor([1.0 / gsz[r['group']] for r in tr], dtype=torch.double)
    g = torch.Generator().manual_seed(SEED)
    total = a.max_iters or EPOCHS * ITERS_PER_EPOCH
    sampler = torch.utils.data.WeightedRandomSampler(wts, num_samples=total * BATCH, replacement=True, generator=g)
    dl = torch.utils.data.DataLoader(DS(tr, True), batch_size=BATCH, sampler=sampler, num_workers=2,
                                     collate_fn=collate, worker_init_fn=lambda k: random.seed(SEED + k))
    model = build_model().to(dev)
    sz = check_no_resize(model, dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=MOM, weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda')
    log = open(os.path.join(a.out, 'train_log.jsonl'), 'w')
    meta = {'spec': 'DL0_train_spec.md', 'seed': SEED, 'batch': BATCH, 'lr': LR, 'epochs_cap': EPOCHS,
            'iters_per_epoch': ITERS_PER_EPOCH, 'total_iters': total, 'internal_image_size': list(sz),
            'group_sizes': gsz, 'sample_weight': '1／該群組訓練影格數', 'torch': torch.__version__,
            'torchvision': torchvision.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'weights': 'MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1', 'score_t': SCORE_T, 'bin_t': BIN_T,
            'label_index_sha256': hashlib.sha256(open(IDX, 'rb').read()).hexdigest()}
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), ensure_ascii=False, indent=1)
    model.train()
    it, best, stop_why = 0, None, 'completed'
    epoch_iters = ITERS_PER_EPOCH if not a.max_iters else max(1, min(ITERS_PER_EPOCH, a.max_iters))
    t_start = time.time()
    for imgs, tgts, _ in dl:
        ep = it // epoch_iters
        lr = LR * (min(1.0, (it + 1) / epoch_iters) if ep == 0 else 1.0) * (0.1 if ep >= LR_DROP_EPOCH else 1.0)
        for gp in opt.param_groups:
            gp['lr'] = lr
        imgs = [x.to(dev) for x in imgs]
        tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
        with torch.autocast('cuda', dtype=torch.float16):
            ld = model(imgs, tgts)
        loss = sum(ld.values())
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        it += 1
        if it % 50 == 0:
            c = cpu_temp()
            log.write(json.dumps({'it': it, 'lr': lr, 'loss': float(loss), **{k: float(v) for k, v in ld.items()},
                                  'cpu_c': c, 'gpu_mem_mb': torch.cuda.max_memory_allocated() / 2**20,
                                  'wall_s': round(time.time() - t_start, 1)}) + '\n')
            log.flush()
            if c is not None and c >= CPU_LIMIT_C:
                stop_why = f'thermal_abort cpu {c} °C'
                break
        if it % epoch_iters == 0 or it == total:
            r = evaluate(model, dv, dev)
            r.update({'epoch': ep, 'it': it})
            log.write(json.dumps({'eval': r}, ensure_ascii=False) + '\n')
            log.flush()
            ck = os.path.join(a.out, f'ep{ep:02d}.pth')
            torch.save(model.state_dict(), ck)
            if best is None or r['score_equal_weight_groups'] > best['score']:   # 同分取較早
                best = {'score': r['score_equal_weight_groups'], 'epoch': ep, 'ckpt': ck}
            model.train()
        if it >= total:
            break
    if best:
        best['sha256'] = hashlib.sha256(open(best['ckpt'], 'rb').read()).hexdigest()
    json.dump({'stop_why': stop_why, 'iters': it, 'best': best, 'wall_s': round(time.time() - t_start, 1)},
              open(os.path.join(a.out, 'result.json'), 'w'), ensure_ascii=False, indent=1)
    print('結束', stop_why, it, best)


if __name__ == '__main__':
    main()
