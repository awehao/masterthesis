#!/usr/bin/env python3
"""DL0 開發評估：L1（學習式遮罩）vs G1（真值輔助遮罩參考）vs G0（凍結 D1），同影格、同座標、同幾何後端與拒絕規則。

在 .venv-dl0 執行：
    .venv-dl0/bin/python evaluation/dl0_l1_eval.py --ckpt evaluation/dl0_ckpt/run1/epNN.pth [--groups dev]

L1／G1 的幾何後端＝dl0_g1_check.g1_detect（遮罩點 → D1 圓柱擬合 → D1 端點規則；**寬度篩選略過**，G0 有，照實註明）。
L1 遮罩：分數最高單一實例、分數門檻 0.5、二值化 0.5（訓練規格固定）。每方法各用 rng(0)、相同影格順序。
報：遮罩 IoU（ignore 排除；漏檢＝0）、軸線（L1 層）有效率與角度、中心（L2）有效率與誤差（分距離箱）、拒絕原因、誤認與違反、推論耗時。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d1_handle_detect as D1          # noqa: E402
import dl0_autolabel as AL            # noqa: E402
import dl0_g1_check as G              # noqa: E402
import dl0_train as TR                # noqa: E402

# 封存測試：**凍結後才可執行**（--split test 會先核對封存與模型凍結清單）；標籤於開封時以 dl0_autolabel 產生；G0 讀開封時跑出的 D1 結果
TEST_SRC = {'test_lateral': ('dl0_test_lateral/wrist_v0', 'd1_detect_test.json'),
            'test_view': ('dl0_test_view/wrist_v0', 'd1_detect_test.json')}
DEV_SRC = {'traj_wg4b_f02_P': ('d1_dev_f02P/wrist_v0', 'd1_detect_rev2.json'),
           'traj_mt_b1_02_P': ('d1_hold_mt02P/wrist_v0', 'd1_detect_hold.json')}
BINS = G.BINS


def iou(pred, lab):
    fg, ev = lab == 1, lab != 2
    if not fg.any():
        return None
    return float((pred & fg & ev).sum() / max(((pred | fg) & ev).sum(), 1))


def summarize(rows, key):
    out = {}
    for lo, hi in BINS:
        r = [x for x in rows if lo <= x['dist'] < hi]
        if not r:
            continue
        l2 = [x for x in r if x[key]['L2']]
        l1 = [x for x in r if x[key]['L1']]
        e2 = [x[key]['err'] for x in l2 if x[key]['err'] is not None]
        a1 = [x[key]['ang'] for x in l1 if x[key]['ang'] is not None]
        rej = {}
        for x in r:
            k = x[key]['reject'] or ((x[key]['reject_L2'] or '').split(':')[0] or 'L2_ok')
            rej[k] = rej.get(k, 0) + 1
        ious = [x[key]['iou'] for x in r if x[key].get('iou') is not None]
        out[f'{lo}-{hi}'] = {
            'n': len(r), 'L1_axis': len(l1), 'L1_angle_deg_median': (round(float(np.median(a1)), 2) if a1 else None),
            'L2_center': len(l2), 'L2_rate': round(len(l2) / len(r), 3),
            'L2_err_mm': (None if not e2 else {'median': round(float(np.median(e2)), 2), 'p95': round(float(np.percentile(e2, 95)), 2),
                                                'max': round(float(max(e2)), 2)}),
            'L2_violation': sum(1 for x in r if x[key]['viol']), 'L2_wrong_gt30mm': sum(1 for x in l2 if (x[key]['err'] or 0) > 30),
            'L1_misid': sum(1 for x in l1 if x[key]['l1_misid']),
            'mask_iou_mean': (round(float(np.mean(ious)), 3) if ious else None), 'reasons': rej}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--split', choices=('dev', 'test'), default='dev')
    ap.add_argument('--model-freeze', default=os.path.join(HERE, 'results', 'vision', 'freeze_dl0_model_v2.sha256'))
    a = ap.parse_args()
    dev = torch.device('cuda')
    model = TR.build_model(pretrained=False)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    model.to(dev).eval()
    a.out = a.out or os.path.join(HERE, 'results', 'vision', f'DL0_{a.split}_eval.json')
    if a.split == 'test':
        import subprocess
        ws = os.path.abspath(os.path.join(HERE, '..'))
        for fz in (os.path.join(HERE, 'results', 'vision', 'seal_dl0_test.sha256'), a.model_freeze):
            r = subprocess.run(['sha256sum', '-c', '--quiet', fz], cwd=ws, capture_output=True, text=True)
            assert r.returncode == 0, f'凍結／封存核對失敗 {fz}：{r.stdout[-500:]}'
        assert __import__('hashlib').sha256(open(a.ckpt, 'rb').read()).hexdigest() in open(a.model_freeze).read(), '檢查點不在模型凍結清單'
    SRC = TEST_SRC if a.split == 'test' else DEV_SRC
    if a.split == 'test':
        tix = os.path.join(HERE, 'dl0_ckpt', 'labels_test', 'index.json')
        assert os.path.exists(tix), '缺測試標籤快取（須由開封程序以系統 python 產生，與開發標籤同一環境）'
        idx = {(r['group'], r['n']): r for r in json.load(open(tix))['rows']}
    else:
        idx = {(r['group'], r['n']): r for r in TR.load_index() if r['split'] == 'dev'}
    res = {'schema': 'dl0_dev_eval/1', 'ckpt': a.ckpt,
           'ckpt_sha256': __import__('hashlib').sha256(open(a.ckpt, 'rb').read()).hexdigest(),
           'note': 'G1／L1 共用 g1_detect（寬度篩選略過）；G0＝凍結 D1 既有逐格結果（含寬度篩選）', 'groups': {}}
    res['evidence_issues'] = []
    for grp, (cap, g0file) in SRC.items():
        Dc = os.path.join(HERE, 'runs', cap)
        meta_ns = [f['n'] for f in json.load(open(os.path.join(Dc, 'meta.json')))['frames']]
        if len(set(meta_ns)) != len(meta_ns):
            res['evidence_issues'].append({'group': grp, 'why': 'duplicate_frame_ids'})
        g0p = os.path.join(Dc, g0file)
        g0 = {r['n']: r for r in json.load(open(g0p))['rows']} if os.path.exists(g0p) else {}
        if not g0:
            res['evidence_issues'].append({'group': grp, 'why': f'missing_G0_file {g0file}'})
        frs = {f['n']: f for f in AL.frames_of(cap)}
        rng_g1, rng_l1 = np.random.default_rng(0), np.random.default_rng(0)
        rows, lat = [], []
        for n in meta_ns:                                  # 以來源完整影格清單為分母，不默默略過
            fr = frs.get(n)
            issue = (None if fr is not None and fr['c'] is not None else 'missing_depth_or_truth')
            if issue is None and n not in g0:
                issue = 'missing_G0_row'
            if issue is None and (grp, n) not in idx:
                issue = 'missing_label_cache'
            if issue:
                res['evidence_issues'].append({'group': grp, 'n': n, 'why': issue})
                continue
            try:
                lab = np.asarray(Image.open(os.path.join(HERE, idx[(grp, n)]['label'])))   # 開發與測試標籤皆由系統 python 快取
                flags = idx[(grp, n)]['flags']
                kind = ('negative' if 'no_handle_in_view' in flags else 'positive' if (lab == 1).any() else 'excluded')
                dep = AL.load_depth(fr['depth_path'])
                img = np.asarray(Image.open(fr['rgb']).convert('RGB'))
                t = torch.from_numpy(img.copy()).permute(2, 0, 1).float().to(dev) / 255.0
                t0 = time.perf_counter()
                with torch.no_grad():
                    o = model([t])[0]
                torch.cuda.synchronize()
                lat.append((time.perf_counter() - t0) * 1e3)
                keep = o['scores'] >= TR.SCORE_T
                pm = (o['masks'][keep][0, 0] >= TR.BIN_T).cpu().numpy() if keep.any() else np.zeros(lab.shape, bool)
                gm = lab == 1
                truth = {'handle_center_world_at_capture': fr['c']}
                rec = {'n': n, 'flags': flags, 'kind': kind,
                       'excluded_reason': ('no_evaluable_foreground' if kind == 'excluded' else None),
                       'L1_top_score': (float(o['scores'][0]) if len(o['scores']) else None),
                       'L1_detected': bool(keep.any()),
                       'negative_false_positive': (bool(keep.any()) if kind == 'negative' else None)}
                for key, mask, rng in (('G1', gm, rng_g1), ('L1', pm, rng_l1)):
                    det = G.g1_detect(dep, fr['K'], fr['T'], mask, rng)
                    ev = D1.evaluate(det, truth, fr['T'], fr['K'], AL.AXIS_W)
                    rec[key] = {'L1': det['L1'], 'L2': det['L2'], 'reject': det['reject'], 'reject_L2': det['reject_L2'],
                                'err': ev.get('L2_err_mm'), 'ang': ev.get('L1_angle_deg'), 'viol': ev['L2_violation'],
                                'l1_misid': bool(ev.get('L1_misid')),
                                'iou': (iou(mask, lab) if (key == 'L1' and kind == 'positive') else None)}
                r0 = g0[n]
                rec['G0'] = {'L1': r0['L1'], 'L2': r0['L2'], 'reject': r0['reject'], 'reject_L2': r0['reject_L2'],
                             'err': r0['eval'].get('L2_err_mm'), 'ang': r0['eval'].get('L1_angle_deg'),
                             'viol': r0['eval']['L2_violation'], 'l1_misid': bool(r0['eval'].get('L1_misid'))}
                rec['dist'] = D1.evaluate({'L0': None, 'L1': None, 'L2': None}, truth, fr['T'], fr['K'], AL.AXIS_W)['dist_m']
                rows.append(rec)
            except Exception as e:                         # noqa: BLE001
                res['evidence_issues'].append({'group': grp, 'n': n, 'why': f'eval_error {e!r}'})
        neg = [r for r in rows if r['kind'] == 'negative']
        res['groups'][grp] = {
            'n_source_frames': len(meta_ns), 'n_evaluated': len(rows),
            'n_positive': sum(r['kind'] == 'positive' for r in rows), 'n_negative': len(neg),
            'n_excluded': sum(r['kind'] == 'excluded' for r in rows),
            'negative_false_positive': sum(bool(r['negative_false_positive']) for r in neg),
            'out_of_bins_lt_0p1m': [r['n'] for r in rows if r['dist'] < BINS[0][0]],
            'L1_infer_ms_p50': (round(float(np.median(lat)), 1) if lat else None),
            'L1_infer_ms_p95': (round(float(np.percentile(lat, 95)), 1) if lat else None),
            **{k: summarize(rows, k) for k in ('G0', 'G1', 'L1')}, 'rows': rows}
        print(grp, 'source', len(meta_ns), 'evaluated', len(rows), json.dumps(
            {k: {b: (v['L2_center'], v['n']) for b, v in res['groups'][grp][k].items()} for k in ('G0', 'G1', 'L1')}, ensure_ascii=False))
    res['complete'] = (not res['evidence_issues']
                       and all(g['n_evaluated'] == g['n_source_frames'] for g in res['groups'].values()))
    if not res['complete']:
        print('**證據不完整**：', res['evidence_issues'][:10])
    if os.path.exists(a.out):
        raise SystemExit(f'輸出已存在，不覆寫：{a.out}')
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print('寫出', a.out)


if __name__ == '__main__':
    main()
