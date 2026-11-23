#!/usr/bin/env python3
"""XH3 開封前一致性核對：用測試評估程式 `xh_test_eval.run` 在**開發資產**上重算，逐格對照已封存的 `XH3_dev_eval.json`。

只讀開發資產與既有影格；不碰測試資產、不需要開封紀錄。對照欄位：L1_detected、L1_iou、三條路徑（L1／G1／G0K）的 L2 有無與
center_err_mm、approach_signed_mm、axis_deg、dist_m。數值差以絕對值記最大差；任一離散欄位不符或缺列 ⇒ 結束碼 1。

    .venv-dl0/bin/python evaluation/xh_test_eval_devcheck.py --ckpt evaluation/xh_ckpt/run1/ep11.pth
輸出 results/vision/XH3_test_eval_devcheck.json（已存在則拒絕覆寫）。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import xh_test_eval as TE                              # noqa: E402
import xh_train as TR                                  # noqa: E402

DEV = ['R_d28_200', 'R_d32_260', 'N_k30']
TOL = 1e-6
NUM = ('center_err_mm', 'approach_signed_mm', 'axis_deg')


def verdict(new, mism, over, n_old, n_new):
    """通過條件：新評估完整（complete、閉合、列數＝已評估數、零證據問題）＋列數相同＋零離散不符＋數值差在容差內。"""
    why = []
    if new.get('complete') is not True:
        why.append('new_not_complete')
    cl = new.get('closure') or {}
    if not (cl.get('closed') and cl.get('rows_match_evaluated')):
        why.append('closure_not_closed')
    if new.get('evidence_issues'):
        why.append(f"evidence_issues:{len(new['evidence_issues'])}")
    if n_old != n_new:
        why.append(f'row_count:{n_old}!={n_new}')
    if mism:
        why.append(f'discrete_mismatch:{len(mism)}')
    if over:
        why.append('over_tol:' + ','.join(sorted(over)))
    return not why, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'vision', 'XH3_test_eval_devcheck.json'))
    ap.add_argument('--ref', default=os.path.join(HERE, 'results', 'vision', 'XH3_dev_eval.json'),
                    help='對照基準（預設＝封存開發評估；也可指向同環境重跑的 xh_dev_eval 輸出）')
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫')
    dev = torch.device('cuda')
    model = TR.build_model(pretrained=False)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    model.to(dev).eval()

    def infer(img):
        with torch.no_grad():
            o = model([torch.from_numpy(img.copy()).permute(2, 0, 1).float().to(dev) / 255.0])[0]
        keep = o['scores'] >= TR.SCORE_T
        pm = (o['masks'][keep][0, 0] >= TR.BIN_T).cpu().numpy() if keep.any() else np.zeros(img.shape[:2], bool)
        return bool(keep.any()), pm
    reg = yaml.safe_load(open(os.path.join(HERE, 'results', 'vision', 'XH1_registration.yaml')))
    assets = {x['id']: x for x in reg['assets']}
    new = TE.run(infer, assets, DEV, os.path.join(HERE, 'runs'), os.path.join(WS, 'src/my_omnibot_description/config/xh'))
    old = {r['key']: r for r in json.load(open(a.ref))['rows']}
    nk = {r['key']: r for r in new['rows']}
    mism, maxd = [], {}
    for k in sorted(set(old) | set(nk)):
        if k not in old or k not in nk:
            mism.append({'key': k, 'why': 'missing_in_' + ('old' if k not in old else 'new')})
            continue
        o, n = old[k], nk[k]
        for f in ('L1_detected', 'kind', 'neg_reason'):
            if o.get(f) != n.get(f):
                mism.append({'key': k, 'field': f, 'old': o.get(f), 'new': n.get(f)})
        for f in ('L1_iou', 'dist_m'):
            if (o.get(f) is None) != (n.get(f) is None):
                mism.append({'key': k, 'field': f, 'old': o.get(f), 'new': n.get(f)})
            elif o.get(f) is not None:
                maxd[f] = max(maxd.get(f, 0.0), abs(o[f] - n[f]))
        for p in ('L1', 'G1', 'G0K'):
            op, np_ = o[p], n[p]
            if isinstance(op, str) or isinstance(np_, str):
                if not (isinstance(op, str) and isinstance(np_, str)):
                    mism.append({'key': k, 'field': p, 'old': op, 'new': np_})
                continue
            if op['L2'] != np_['L2']:
                mism.append({'key': k, 'field': f'{p}.L2', 'old': op['L2'], 'new': np_['L2']})
                continue
            if op['L2']:
                for f in NUM:
                    maxd[f'{p}.{f}'] = max(maxd.get(f'{p}.{f}', 0.0), abs(op[f] - np_[f]))
    over = {f: d for f, d in maxd.items() if d > TOL}
    ok, why = verdict(new, mism, over, len(old), len(nk))
    out = {'schema': 'xh3_test_eval_devcheck/1', 'ckpt': a.ckpt, 'ref': a.ref, 'n_old': len(old), 'n_new': len(nk),
           'new_complete': new['complete'], 'new_closure': new['closure'], 'new_evidence_issues': new['evidence_issues'],
           'discrete_mismatches': mism, 'max_abs_diff': maxd, 'tol': TOL, 'over_tol': over,
           'env': TE.env_fingerprint(torch), 'pass': ok, 'fail_reasons': why}
    json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ('n_old', 'n_new', 'new_complete', 'max_abs_diff', 'over_tol', 'pass')}, ensure_ascii=False, default=str))
    print('不符', len(mism), mism[:5])
    sys.exit(0 if out['pass'] else 1)


if __name__ == '__main__':
    main()
