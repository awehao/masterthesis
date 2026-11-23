#!/usr/bin/env python3
"""D 補充（事後加做，非預先定義）：xh_rerender_compare.py 依來源時刻配對得 0 對（兩次擷取相位錯開 0.06–0.07 s），
逐格影像差無法計算。本檔只做兩件可計算的事，供審查決定如何解讀：

1. 最近鄰時刻配對（|Δsrc_t| ≤ 0.105 s＝半個 5 Hz 週期）的相機位姿差、真值中心差——說明「最接近的兩張」其實差多少視角。
   這些配對**不是同一觀測**，不據此算像素差。
2. 分布層級：新、舊兩組各自完整評估（每組自己的影格順序、每路徑種子 0，與 xh_test_eval.run 對單一序列相同），
   依距離分箱報輸出數、輸出且 ≤10 mm、中心誤差中位／p95。兩組影格不同，差異同時含「觀測時刻不同」與「渲染機器不同」。

    .venv-dl0/bin/python evaluation/xh_rerender_compare_dist.py     # 輸出 results/vision/XH3_rerender_msi_compare_dist.json（已存在則拒絕覆寫）
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import drawer_asset_v2 as DA                           # noqa: E402
import xh_rerender_compare as RC                       # noqa: E402
import xh_test_eval as TE                              # noqa: E402
import xh_test_eval_v2 as E2                           # noqa: E402
import xh_train as TR                                  # noqa: E402

OUT = os.path.join(HERE, 'results', 'vision', 'XH3_rerender_msi_compare_dist.json')
BINS = TE.BINS


def evaluate_all(D, meta, spec, A, infer):
    rngs = {p: np.random.default_rng(0) for p in ('L1', 'G1', 'G0K')}
    rows, issues = [], []
    for f in meta['frames']:
        try:
            rows.append(TE.evaluate_frame(TE.frame_from_meta(D, meta, spec, f), A, RC.SEQ, RC.TMPL, infer, rngs))
        except Exception as e:                           # noqa: BLE001
            issues.append({'n': f.get('n'), 'why': f'{type(e).__name__}: {e}'})
    return rows, issues


def summarize(rows):
    pos = [r for r in rows if r['kind'] == 'positive']
    s = {'n': len(rows), 'n_positive': len(pos), 'kinds': {k: sum(1 for r in rows if r['kind'] == k) for k in sorted({r['kind'] for r in rows})},
         'L1_detected': sum(1 for r in pos if r['L1_detected']),
         'L1_iou_mean_missed0': float(np.mean([r['L1_iou'] for r in pos])) if pos else 'NA'}
    for p in ('L1', 'G1', 'G0K'):
        out_ = [r for r in pos if r[p]['L2']]
        bins = {}
        for lo, hi in BINS:
            b = [r for r in pos if r['dist_m'] is not None and lo <= r['dist_m'] < hi]
            ok = [r[p] for r in b if r[p]['L2']]
            bins[f'{lo}-{hi}'] = {'n': len(b), 'output': len(ok), 'le_10mm': sum(1 for x in ok if x['center_err_mm'] <= 10),
                                  'center_err_mm_median': float(np.median([x['center_err_mm'] for x in ok])) if ok else 'NA'}
        e = [r[p]['center_err_mm'] for r in out_]
        s[p] = {'output': len(out_), 'output_and_le_10mm': sum(1 for x in e if x <= 10),
                'center_err_mm_median': float(np.median(e)) if e else 'NA', 'center_err_mm_p95': float(np.percentile(e, 95)) if e else 'NA',
                'bins': bins}
    return s


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    mo, mn = json.load(open(os.path.join(RC.OLD, 'meta.json'))), json.load(open(os.path.join(RC.NEW, 'meta.json')))
    spec = DA.load(RC.ASSET)
    res = {'schema': 'xh3_rerender_compare_dist/1', 'status': '事後補充（非預先定義）；預先定義的逐格配對為 0 對，見 XH3_rerender_msi_compare.json',
           'first_render_time_old': mo['rejected_frames'][0]['rendering_time'] if mo.get('rejected_frames') else None,
           'first_render_time_new': mn['rejected_frames'][0]['rendering_time'] if mn.get('rejected_frames') else None}
    so = np.array([f['src_record_t'] for f in mo['frames']])
    nn = []
    for f in mn['frames']:
        i = int(np.argmin(np.abs(so - f['src_record_t'])))
        g = mo['frames'][i]
        dt = float(f['src_record_t'] - g['src_record_t'])
        if abs(dt) <= 0.105:
            nn.append({'n_new': f['n'], 'n_old': g['n'], 'dt_s': dt,
                       'cam_pos_diff_mm': float(np.linalg.norm(np.subtract(f['cam_pos_world'], g['cam_pos_world']))) * 1e3,
                       'cam_rot_diff_deg': RC.quat_angle_deg(f['cam_quat_wxyz_world'], g['cam_quat_wxyz_world']),
                       'handle_center_diff_mm': float(np.linalg.norm(np.subtract(f['handle_center_world_at_capture'],
                                                                                  g['handle_center_world_at_capture']))) * 1e3})
    res['nearest_pairs'] = {'n': len(nn), 'of_new': len(mn['frames']), **{k: RC.q([x[k] for x in nn]) for k in
                            ('dt_s', 'cam_pos_diff_mm', 'cam_rot_diff_deg', 'handle_center_diff_mm')}}
    dev = torch.device('cuda')
    model = TR.build_model(pretrained=False)
    ck = os.path.join(HERE, 'xh_ckpt', 'run1', 'ep11.pth')
    model.load_state_dict(torch.load(ck, map_location='cpu'))
    model.to(dev).eval()
    res['ckpt_sha256'] = hashlib.sha256(open(ck, 'rb').read()).hexdigest()

    def infer(img):
        with torch.no_grad():
            o = model([torch.from_numpy(img.copy()).permute(2, 0, 1).float().to(dev) / 255.0])[0]
        keep = o['scores'] >= TR.SCORE_T
        pm = (o['masks'][keep][0, 0] >= TR.BIN_T).cpu().numpy() if keep.any() else np.zeros(img.shape[:2], bool)
        return bool(keep.any()), pm
    A = {x['id']: x for x in yaml.safe_load(open(os.path.join(HERE, 'results', 'vision', 'XH1_registration.yaml')))['assets']}[RC.AID]
    ro, io_ = evaluate_all(RC.OLD, mo, spec, A, infer)
    rn, in_ = evaluate_all(RC.NEW, mn, spec, A, infer)
    res['old'] = {'summary': summarize(ro), 'issues': io_}
    res['new'] = {'summary': summarize(rn), 'issues': in_}
    res['rows_old'], res['rows_new'] = ro, rn
    res['env'] = E2.env_fingerprint_v2(torch)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=str)
    pr = lambda s: {k: s[k] for k in ('n', 'n_positive', 'kinds', 'L1_detected', 'L1_iou_mean_missed0')} | \
        {p: {k: s[p][k] for k in ('output', 'output_and_le_10mm', 'center_err_mm_median', 'center_err_mm_p95')} for p in ('L1', 'G1', 'G0K')}  # noqa: E731
    print(json.dumps({'first_render': [res['first_render_time_old'], res['first_render_time_new']], 'nearest_pairs': res['nearest_pairs'],
                      'old': pr(res['old']['summary']), 'new': pr(res['new']['summary']), 'issues': [io_, in_]}, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
