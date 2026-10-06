#!/usr/bin/env python3
"""XH3 開發評估（訓練後、凍結前；規格 results/vision/XH3_review_train_spec.md draft-2 §4–5）。只用 dev 資產（R_d28_200、R_d32_260、N_k30）。

三條路徑，同影格、同座標（PC2 +0.5）：
  L1＝學習式遮罩（選出的檢查點、分數門檻 0.5、二值化 0.5）→ K 後端（資產目錄尺寸）
  G1＝真值輔助遮罩（標籤快取的目標遮罩）→ 同一 K 後端（遮罩來源比較；不是性能上限）
  G0-K＝已知尺寸的改編幾何基線（含前板／寬度篩選；完整方法比較）
N 資產主要報 RGB 誤檢（不因目錄已知是 N 而跳過偵測）；幾何不適用欄位填 NA。按資產分報＋資產等權彙總；影格不是獨立樣本。

    .venv-dl0/bin/python evaluation/xh_dev_eval.py --ckpt evaluation/xh_ckpt/run1/epXX.pth
輸出 results/vision/XH3_dev_eval.json（已存在則拒絕覆寫）。
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import dl0_autolabel_v2 as V2                          # noqa: E402
import drawer_asset_v2 as DA                           # noqa: E402
import xh_geom as XG                                   # noqa: E402
import xh_train as TR                                  # noqa: E402

BINS = [(0.1, 0.4), (0.4, 1.0), (1.0, 2.0), (2.0, 3.1)]
DEV = ['R_d28_200', 'R_d32_260', 'N_k30']


def center_eval(det, c_true):
    if not det.get('L2'):
        return {'L2': False, 'reject': det.get('reject'), 'reject_L2': det.get('reject_L2')}
    e = np.array(det['L2']['center']) - np.array(c_true)
    ax = np.array(det['L1']['axis'])
    ang = float(np.degrees(np.arccos(min(1.0, abs(ax @ np.array([1.0, 0, 0])) / np.linalg.norm(ax)))))
    return {'L2': True, 'center_err_mm': float(np.linalg.norm(e)) * 1e3, 'approach_signed_mm': float(e[1]) * 1e3, 'axis_deg': ang}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'vision', 'XH3_dev_eval.json'))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫')
    dev = torch.device('cuda')
    model = TR.build_model(pretrained=False)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    model.to(dev).eval()
    idx = {r['key']: r for r in json.load(open(TR.IDX))['rows']}
    reg = yaml.safe_load(open(os.path.join(HERE, 'results', 'vision', 'XH1_registration.yaml')))
    assets = {x['id']: x for x in reg['assets']}
    res = {'schema': 'xh3_dev_eval/1', 'ckpt': a.ckpt, 'ckpt_sha256': hashlib.sha256(open(a.ckpt, 'rb').read()).hexdigest(),
           'score_t': TR.SCORE_T, 'bin_t': TR.BIN_T, 'rows': [], 'evidence_issues': []}
    for aid in DEV:
        A = assets[aid]
        spec = DA.load(os.path.join(WS, 'src/my_omnibot_description/config/xh', f'drawer_unit_xh_{aid}.yaml'))
        is_R = A['type'] == 'R'
        d_m = A['diameter_mm'] / 1000.0 if is_R else None
        L_m = A['length_mm'] / 1000.0 if is_R else None
        for tmpl in ('A_front', 'B_oblique'):
            seq = f'xh_{aid}_{tmpl}'
            try:
                frs = list(V2.frames_of_xh(os.path.join(HERE, 'runs', seq, 'wrist_v0'), spec))
            except V2.EvidenceError as e:
                res['evidence_issues'].append({'seq': seq, 'why': str(e)})
                continue
            rng_g0, rng_g1, rng_l1 = (np.random.default_rng(0) for _ in range(3))
            for f in frs:
                key = f'{seq}:{f["n"]}'
                r = idx.get(key)
                if r is None:
                    res['evidence_issues'].append({'key': key, 'why': 'missing_label_row'})
                    continue
                lab = np.asarray(Image.open(os.path.join(HERE, r['label'])))
                img = np.asarray(Image.open(f['rgb']).convert('RGB'))
                with torch.no_grad():
                    o = model([torch.from_numpy(img.copy()).permute(2, 0, 1).float().to(dev) / 255.0])[0]
                keep = o['scores'] >= TR.SCORE_T
                pm = (o['masks'][keep][0, 0] >= TR.BIN_T).cpu().numpy() if keep.any() else np.zeros(lab.shape, bool)
                rec = {'key': key, 'asset': aid, 'type': A['type'], 'template': tmpl, 'kind': r['kind'], 'neg_reason': r['neg_reason'],
                       'dist_m': r['dist_m'], 'L1_detected': bool(keep.any())}
                fg, ig = lab == 1, lab == 2
                if r['kind'] == 'positive':
                    ev = ~ig
                    rec['L1_iou'] = float((pm & fg & ev).sum() / max(((pm | fg) & ev).sum(), 1))
                if is_R:
                    dep = V2.load_depth(f['depth_path'])
                    rec['L1'] = center_eval(XG.k_detect(dep, f['K'], f['T'], pm, rng_l1, d_m, L_m), f['c'])
                    rec['G1'] = center_eval(XG.k_detect(dep, f['K'], f['T'], fg, rng_g1, d_m, L_m), f['c'])
                    g0, _ = XG.g0k_detect(dep, f['K'], f['T'], rng_g0, d_m, L_m)
                    rec['G0K'] = center_eval(g0, f['c'])
                else:
                    rec['L1'] = rec['G1'] = rec['G0K'] = 'NA（無目標資產）'
                res['rows'].append(rec)
    rows = res['rows']
    summ = {}
    for aid in DEV:
        rr = [r for r in rows if r['asset'] == aid]
        s = {'n': len(rr), 'n_positive': sum(r['kind'] == 'positive' for r in rr)}
        pos = [r for r in rr if r['kind'] == 'positive']
        if pos:
            s['L1_mask_iou_mean_missed0'] = float(np.mean([r['L1_iou'] for r in pos]))
            s['L1_detect_rate'] = float(np.mean([r['L1_detected'] for r in pos]))
            for path in ('L1', 'G1', 'G0K'):
                bins = {}
                for lo, hi in BINS:
                    b = [r for r in pos if r['dist_m'] is not None and lo <= r['dist_m'] < hi]
                    ok = [r[path] for r in b if r[path]['L2']]
                    bins[f'{lo}-{hi}'] = {'n': len(b), 'L2': len(ok),
                                          'center_err_mm_median': float(np.median([x['center_err_mm'] for x in ok])) if ok else None,
                                          'approach_signed_mm_median': float(np.median([x['approach_signed_mm'] for x in ok])) if ok else None}
                rej = {}
                for r in pos:
                    if not r[path]['L2']:
                        k = str(r[path].get('reject') or r[path].get('reject_L2'))
                        rej[k.split(':')[0]] = rej.get(k.split(':')[0], 0) + 1
                s[path] = {'L2_rate': float(np.mean([r[path]['L2'] for r in pos])), 'bins': bins, 'reject_reasons': rej}
        negs = [r for r in rr if r['kind'] == 'negative']
        s['negatives'] = {k: {'n': sum(1 for r in negs if r['neg_reason'] == k), 'L1_false_positive': sum(1 for r in negs if r['neg_reason'] == k and r['L1_detected'])}
                          for k in sorted({r['neg_reason'] for r in negs})}
        summ[aid] = s
    Rdev = [a_ for a_ in DEV if assets[a_]['type'] == 'R']
    summ['asset_equal_weight_R'] = {
        'L1_mask_iou': float(np.mean([summ[a_]['L1_mask_iou_mean_missed0'] for a_ in Rdev])),
        **{f'{p}_L2_rate': float(np.mean([summ[a_][p]['L2_rate'] for a_ in Rdev])) for p in ('L1', 'G1', 'G0K')}}
    res['summary'] = summ
    res['labels_note'] = {'R_d28_200': '未見直徑值、已見長度', 'R_d32_260': '未見直徑與長度值', 'N_k30': '干擾物（未見尺寸）'}
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps(summ, ensure_ascii=False, indent=1, default=str)[:3000])


if __name__ == '__main__':
    main()
