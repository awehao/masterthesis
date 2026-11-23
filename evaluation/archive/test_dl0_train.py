#!/usr/bin/env python3
"""dl0_train.py 的必要測試（Codex 訓練規格審查）。在 .venv-dl0 執行：
    .venv-dl0/bin/python evaluation/test_dl0_train.py
"""
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from torchvision.models.detection import roi_heads as RH  # noqa: E402

ORIG = RH.maskrcnn_loss
import dl0_train as T  # noqa: E402

fails = []


def check(name, cond, detail=''):
    print(('PASS ' if cond else 'FAIL ') + name + ('' if cond else f'  {detail}'))
    if not cond:
        fails.append(name)


torch.manual_seed(0)
M, H, W = 28, 60, 80
props = [torch.tensor([[10., 10., 50., 40.], [5., 5., 70., 55.]])]
idxs = [torch.tensor([0, 0])]
labels = [torch.tensor([1])]
base = torch.zeros(1, H, W, dtype=torch.uint8)
base[0, 20:30, 15:60] = 1


def logits_():
    return torch.randn(2, 2, M, M, requires_grad=True)


# 1 不忽略 ⇒ 與 torchvision 原損失相同
lg = logits_()
l_new = T.maskrcnn_loss_ignore(lg, props, [base], labels, idxs)
l_old = ORIG(lg, props, [base.float()], labels, idxs)
check('no_ignore_equals_original', torch.allclose(l_new, l_old, atol=1e-6), (float(l_new), float(l_old)))
# 2 全忽略 ⇒ 損失 0
allig = torch.full((1, H, W), 2, dtype=torch.uint8)
check('all_ignore_zero', float(T.maskrcnn_loss_ignore(logits_(), props, [allig], labels, idxs)) == 0.0)
# 3 改變被忽略位置的 logits：損失不變、該處梯度為零
gm = base.clone()
gm[0, :, 40:] = 2                                     # 右半部 ignore
lg = logits_()
l1 = T.maskrcnn_loss_ignore(lg, props, [gm], labels, idxs)
l1.backward()
g = lg.grad[torch.arange(2), labels[0][idxs[0]]].clone()
# 找出 ignore 投影 ≥ 0.5 的位置
ig = torch.cat([T._project((gm == 2).float(), props[0], idxs[0], M)])
mask_ig = ig >= 0.5
check('ignore_region_nonempty', bool(mask_ig.any()))
lg2 = lg.detach().clone()
sel = lg2[torch.arange(2), labels[0][idxs[0]]]
sel[mask_ig] += 7.0                                    # 只改被忽略位置
lg2[torch.arange(2), labels[0][idxs[0]]] = sel
l2 = T.maskrcnn_loss_ignore(lg2, props, [gm], labels, idxs)
check('ignored_logits_do_not_change_loss', torch.allclose(l1.detach(), l2, atol=1e-6), (float(l1), float(l2)))
check('ignored_positions_zero_grad', bool((g[mask_ig] == 0).all()) and bool((g[~mask_ig] != 0).any()))

# 4 模型：不縮放、正負樣本 batch 前向／反向
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T.install_ignore_loss()
model = T.build_model(pretrained=False).to(dev)
sz = T.check_no_resize(model, dev)
check('internal_size_480x640', tuple(sz) == (480, 640), sz)
model.eval()
with torch.no_grad():
    out = model([torch.rand(3, 480, 640, device=dev)])[0]
check('output_mask_coords_match_input', out['masks'].shape[-2:] == (480, 640), out['masks'].shape)
model.train()
pos_lab = torch.zeros(480, 640, dtype=torch.uint8)
pos_lab[200:230, 100:400] = 1
pos_lab[199, 100:400] = 2
tg_pos = {'labels': torch.tensor([1], device=dev), 'boxes': torch.tensor([[100., 200., 400., 230.]], device=dev),
          'masks': pos_lab[None].to(dev)}
tg_neg = {'labels': torch.zeros((0,), dtype=torch.int64, device=dev), 'boxes': torch.zeros((0, 4), device=dev),
          'masks': torch.zeros((0, 480, 640), dtype=torch.uint8, device=dev)}
ld = model([torch.rand(3, 480, 640, device=dev), torch.rand(3, 480, 640, device=dev)], [tg_pos, tg_neg])
loss = sum(ld.values())
loss.backward()
gfin = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
check('pos_neg_batch_forward_backward', torch.isfinite(loss) and 'loss_mask' in ld and gfin, {k: float(v) for k, v in ld.items()})

# 5 翻轉一致：強制翻轉，前景與 ignore 跟著翻
rows = [r for r in T.load_index() if r['kind'] == 'positive'][:1]
ds = T.DS(rows, train=True)
orig_random = random.random
random.random = lambda: 0.0
try:
    img_f, tgt_f, _ = ds[0]
finally:
    random.random = orig_random
lab = np.asarray(__import__('PIL.Image', fromlist=['Image']).open(os.path.join(HERE, rows[0]['label'])))
check('flip_mask_consistent', np.array_equal(tgt_f['masks'][0].numpy(), lab[:, ::-1]))
ys, xs = np.nonzero(lab[:, ::-1] == 1)
check('flip_box_from_flipped_mask', tgt_f['boxes'][0].tolist() == [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])

# 6 IoU 定義：漏檢＝0、ignore 排除、負樣本另計、空前景另列排除
class Fake(torch.nn.Module):
    def __init__(self, masks):
        super().__init__()
        self.m = masks
        self.k = 0

    def forward(self, xs):
        m = self.m[self.k]
        self.k += 1
        if m is None:
            return [{'scores': torch.zeros(0), 'masks': torch.zeros(0, 1, 480, 640)}]
        return [{'scores': torch.tensor([0.9]), 'masks': torch.from_numpy(m.astype(np.float32))[None, None]}]


import tempfile  # noqa: E402
from PIL import Image  # noqa: E402
d = tempfile.mkdtemp()
rgb = os.path.join(d, 'x.png')
Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(rgb)


def lab_file(arr, name):
    p = os.path.join(d, name)
    Image.fromarray(arr).save(p)
    return p


g1 = np.zeros((480, 640), np.uint8)
g1[10:20, 10:30] = 1
g1[10:20, 30:40] = 2                                   # ignore
pred_full = np.zeros((480, 640), bool)
pred_full[10:20, 10:40] = True                         # 覆蓋前景＋ignore ⇒ ignore 不計 ⇒ IoU 1
rows6 = [{'rgb': rgb, 'label': lab_file(g1, 'a.png'), 'kind': 'positive', 'group': 'A', 'key': 'a'},
         {'rgb': rgb, 'label': lab_file(g1, 'b.png'), 'kind': 'positive', 'group': 'B', 'key': 'b'},   # 漏檢 ⇒ 0
         {'rgb': rgb, 'label': lab_file(np.zeros((480, 640), np.uint8), 'c.png'), 'kind': 'negative', 'group': 'B', 'key': 'c'},
         {'rgb': rgb, 'label': lab_file(np.full((480, 640), 2, np.uint8), 'e.png'), 'kind': 'excluded', 'group': 'B', 'key': 'e'}]
for r in rows6:
    r['rgb'] = os.path.relpath(r['rgb'], HERE)
    r['label'] = os.path.relpath(r['label'], HERE)
fake = Fake([pred_full, None, pred_full, None])
res = T.evaluate(fake, rows6, torch.device('cpu'))
check('iou_ignore_excluded_and_miss_zero', abs(res['per_group_mean_iou']['A'] - 1.0) < 1e-9
      and abs(res['per_group_mean_iou']['B'] - 0.0) < 1e-9, res)
check('iou_group_equal_weight', abs(res['score_equal_weight_groups'] - 0.5) < 1e-9, res['score_equal_weight_groups'])
check('negatives_separate', res['negatives'] == 1 and res['negative_false_positive'] == 1, res)
check('excluded_listed', res['excluded_no_evaluable_fg'] == ['e'], res['excluded_no_evaluable_fg'])

print(f'{len(fails)} 失敗')
sys.exit(1 if fails else 0)
