#!/usr/bin/env python3
"""住宅任務 Mask R-CNN（Howard 11/23「視覺幫我用 maskrcnn」）：多類別實例分割的訓練、評估、推論共用模組。
模型：torchvision Mask R-CNN R50-FPN v2（COCO 預訓練，本機權重 ~/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth）微調；
輸入只有手腕相機 RGB（640×480，內部不縮放）；標籤＝Isaac 語意分割（isaac_m8_house_seg_collect.py），只在訓練時用。
類別：0 背景 1 門把 2 門扇 3 圓柱 4 抽屜把手 5 筆 6 抽屜面板。同類多個連通塊（例如門內外兩側把手）各為一個實例。
在 .venv-dl0 執行（torch cu128）：
  訓練：.venv-dl0/bin/python evaluation/house_seg.py train --data <收集資料夾,...> --out evaluation/house_seg_ckpt/run1 [--iters 6000]
  評估：.venv-dl0/bin/python evaluation/house_seg.py eval --data ... --weights .../best.pth --split test
推論（實跑視覺工作程序）：from house_seg import load_model, predict"""
import argparse
import json
import math
import os
import random
import time

import numpy as np

CLASSES = ['background', 'door_handle', 'door_leaf', 'cylinder', 'drawer_handle', 'pen', 'drawer_front']
NC = len(CLASSES)
SCORE_T, BIN_T, CPU_LIMIT_C = 0.5, 0.5, 92
MIN_AREA = 25                                                        # 連通塊最少像素（太小的標註不當實例）
COCO_W = os.path.expanduser('~/.cache/torch/hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth')


def cpu_temp():
    import glob
    for d in glob.glob('/sys/class/hwmon/*'):
        try:
            if open(d + '/name').read().strip() == 'k10temp':
                return int(open(d + '/temp1_input').read()) / 1000.0
        except Exception:                                             # noqa: BLE001
            pass
    return None


def build_model(pretrained=True):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    m = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, min_size=480, max_size=640)
    if pretrained:
        m.load_state_dict(torch.load(COCO_W, map_location='cpu'))
    inf = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(inf, NC)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(m.roi_heads.mask_predictor.conv5_mask.in_channels, 256, NC)
    return m


def load_model(weights, dev='cuda'):
    import torch
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(weights, map_location='cpu'))
    m.to(dev).eval()
    return m


def predict(model, rgb, dev='cuda', score_t=SCORE_T):
    """rgb：H×W×3 uint8。回 {類別名: [dict(mask=bool H×W, score, box)]}（分數由高到低）。"""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(rgb[:, :, :3])).permute(2, 0, 1).float().div(255.0).to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=(dev == 'cuda')):
        o = model([x])[0]
    out = {}
    for k in range(len(o['scores'])):
        s = float(o['scores'][k])
        if s < score_t:
            continue
        c = CLASSES[int(o['labels'][k])]
        out.setdefault(c, []).append(dict(mask=(o['masks'][k, 0] >= BIN_T).cpu().numpy(), score=s, box=[float(v) for v in o['boxes'][k]]))
    return out


# ------------------------------------------------------------------ 資料
def load_rows(dirs):
    rows = []
    for d in dirs.split(','):
        p = os.path.join(d, 'index.jsonl')
        for ln in open(p):
            r = json.loads(ln)
            r['dir'] = d
            rows.append(r)
    return rows


def split_of(run, dev_runs, test_runs):
    return 'test' if run in test_runs else ('dev' if run in dev_runs else 'train')


def instances(lab):
    from scipy import ndimage
    ms, ls = [], []
    for c in range(1, NC):
        cc, n = ndimage.label(lab == c)
        for i in range(1, n + 1):
            m = cc == i
            if m.sum() >= MIN_AREA:
                ms.append(m); ls.append(c)
    return ms, ls


class DS:
    def __init__(self, rows, train):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        r = self.rows[i]
        rgb = np.asarray(Image.open(os.path.join(r['dir'], r['stem'] + '_rgb.png')).convert('RGB'), np.float32) / 255.0
        lab = np.asarray(Image.open(os.path.join(r['dir'], r['stem'] + '_label.png')), np.uint8)
        if self.train:                                                # 色彩擾動（亮度、對比、色偏）＋左右翻轉
            rgb = np.clip((rgb - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.1, 0.1) + np.random.uniform(-0.04, 0.04, 3), 0, 1)
            if random.random() < 0.5:
                rgb, lab = rgb[:, ::-1].copy(), lab[:, ::-1].copy()
        ms, ls = instances(lab)
        if ms:
            M = np.stack(ms)
            boxes = []
            for m in ms:
                ys, xs = np.where(m)
                boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            tgt = dict(boxes=torch.tensor(boxes, dtype=torch.float32), labels=torch.tensor(ls, dtype=torch.int64), masks=torch.from_numpy(M.astype(np.uint8)))
        else:
            tgt = dict(boxes=torch.zeros((0, 4), dtype=torch.float32), labels=torch.zeros((0,), dtype=torch.int64), masks=torch.zeros((0,) + lab.shape, dtype=torch.uint8))
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)), tgt, lab


def collate(b):
    return tuple(zip(*b))


# ------------------------------------------------------------------ 評估
def evaluate(model, rows, dev, max_n=400):
    """逐格逐類：預測（該類所有實例聯集）與標註聯集的 IoU；標註沒有該類而預測有 ⇒ 誤檢。回各類平均 IoU（只算有標註的格）、誤檢率、整體平均。"""
    import torch
    from PIL import Image
    model.eval()
    iou = {c: [] for c in CLASSES[1:]}; fp = {c: [0, 0] for c in CLASSES[1:]}
    rs = rows if len(rows) <= max_n else random.Random(0).sample(rows, max_n)
    for r in rs:
        rgb = np.asarray(Image.open(os.path.join(r['dir'], r['stem'] + '_rgb.png')).convert('RGB'))
        lab = np.asarray(Image.open(os.path.join(r['dir'], r['stem'] + '_label.png')), np.uint8)
        pr = predict(model, rgb, dev)
        for ci, c in enumerate(CLASSES[1:], start=1):
            g = lab == ci
            p = np.zeros_like(g)
            for d in pr.get(c, []):
                p |= d['mask']
            if g.sum() >= MIN_AREA:
                iou[c].append(float((g & p).sum()) / max(float((g | p).sum()), 1.0))
            else:
                fp[c][1] += 1
                fp[c][0] += int(p.sum() >= MIN_AREA)
    model.train()
    per = {c: (round(float(np.mean(v)), 4) if v else None, len(v)) for c, v in iou.items()}
    fpr = {c: (round(v[0] / v[1], 4) if v[1] else None) for c, v in fp.items()}
    vals = [v[0] for v in per.values() if v[0] is not None]
    return dict(iou=per, fp_rate=fpr, mean_iou=round(float(np.mean(vals)), 4) if vals else 0.0)


# ------------------------------------------------------------------ 訓練
def train(a):
    import torch
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    dev = 'cuda'
    rows = load_rows(a.data)
    dv_r = {int(x) for x in a.dev_runs.split(',') if x}; ts_r = {int(x) for x in a.test_runs.split(',') if x}
    for r in rows:
        r['split'] = split_of(int(r['run']), dv_r, ts_r)
    tr = [r for r in rows if r['split'] == 'train']; dv = [r for r in rows if r['split'] == 'dev']
    os.makedirs(a.out, exist_ok=False)
    cls_n = {c: sum(1 for r in tr if c in r['counts']) for c in CLASSES[1:]}
    wts = torch.tensor([1.0 + 2.0 * sum(1.0 / max(cls_n[c], 1) for c in r['counts']) * len(tr) / NC for r in tr], dtype=torch.double)   # 稀有類別（筆、抽屜把手）多抽
    g = torch.Generator().manual_seed(0)
    sampler = torch.utils.data.WeightedRandomSampler(wts, num_samples=a.iters * a.batch, replacement=True, generator=g)
    dl = torch.utils.data.DataLoader(DS(tr, True), batch_size=a.batch, sampler=sampler, num_workers=2, collate_fn=collate)
    model = build_model().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=a.lr, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    meta = dict(n_train=len(tr), n_dev=len(dv), n_test=sum(1 for r in rows if r['split'] == 'test'), class_frames_train=cls_n, dev_runs=sorted(dv_r), test_runs=sorted(ts_r),
                iters=a.iters, batch=a.batch, lr=a.lr, data=a.data, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), classes=CLASSES, score_t=SCORE_T, bin_t=BIN_T)
    json.dump(meta, open(os.path.join(a.out, 'meta.json'), 'w'), ensure_ascii=False, indent=1)
    log = open(os.path.join(a.out, 'train_log.jsonl'), 'w')
    model.train(); it, best, t0 = 0, -1.0, time.time()
    for imgs, tgts, _ in dl:
        lr = a.lr * min(1.0, (it + 1) / 300) * (0.1 if it >= int(0.75 * a.iters) else 1.0)
        for gp in opt.param_groups:
            gp['lr'] = lr
        imgs = [x.to(dev) for x in imgs]; tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
        with torch.autocast('cuda', dtype=torch.float16):
            ld = model(imgs, tgts)
        loss = sum(ld.values())
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        it += 1
        if it % 50 == 0:
            c = cpu_temp()
            log.write(json.dumps(dict(it=it, lr=lr, loss=round(float(loss), 4), **{k: round(float(v), 4) for k, v in ld.items()}, cpu_c=c, wall_s=round(time.time() - t0, 1))) + '\n'); log.flush()
            if c is not None and c >= CPU_LIMIT_C:
                print('熱中止', c, flush=True); break
        if it % a.eval_every == 0 or it == a.iters:
            r = evaluate(model, dv, dev)
            r['it'] = it
            log.write(json.dumps(dict(eval=r)) + '\n'); log.flush()
            print(f'[seg] it {it} dev mean IoU {r["mean_iou"]} {r["iou"]}', flush=True)
            if r['mean_iou'] > best:
                best = r['mean_iou']
                torch.save(model.state_dict(), os.path.join(a.out, 'best.pth'))
                json.dump(r, open(os.path.join(a.out, 'best_dev.json'), 'w'), ensure_ascii=False, indent=1)
        if it >= a.iters:
            break
    print(f'[seg] 完成 it {it} 最佳 dev mean IoU {best}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'eval'])
    ap.add_argument('--data', required=True, help='收集資料夾，逗號分隔')
    ap.add_argument('--out')
    ap.add_argument('--weights')
    ap.add_argument('--split', default='test')
    ap.add_argument('--dev-runs', default='84')
    ap.add_argument('--test-runs', default='85,86')
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--eval-every', type=int, default=1000)
    a = ap.parse_args()
    if a.cmd == 'train':
        train(a)
    else:
        rows = load_rows(a.data)
        dv_r = {int(x) for x in a.dev_runs.split(',') if x}; ts_r = {int(x) for x in a.test_runs.split(',') if x}
        rs = [r for r in rows if split_of(int(r['run']), dv_r, ts_r) == a.split]
        m = load_model(a.weights)
        r = evaluate(m, rs, 'cuda', max_n=10 ** 9)
        print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
