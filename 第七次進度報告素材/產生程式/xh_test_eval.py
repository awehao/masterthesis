#!/usr/bin/env python3
"""XH3 測試評估（凍結後一次開封；由 xh_open_test.py 呼叫，不單獨執行）。只用 test 資產（R_x25_150、R_x40_300、N_k40）。

與 xh_dev_eval.py 同一評估定義（L1／G1／G0-K、K 後端、門檻 0.5、PC2），差異只在流程完整性：
* 須有狀態 opened 的開封紀錄；輸出已存在即拒絕。
* **逐格唯一去向**：以 meta 的每一格為來源鍵（序列:n）；重複鍵、缺 RGB／深度、讀取或推論／幾何例外都記去向與證據問題，不崩潰、不略過；
  核對 來源鍵 ＝ 已評估鍵 ∪ 問題鍵（互斥）。
* 標籤以凍結標註器 label_xh 現算；無資料的統計填 NA。
* 任何證據問題 ⇒ complete = False、結束碼非 0；未預期例外也先存部分 JSON（complete = False）再以非 0 結束。
* 另報「輸出且中心誤差 ≤ 10 mm」（開發分析新增、測試前固定的診斷門檻，不是抓取容差）與誤差／直徑。
"""
import argparse
import hashlib
import json
import os
import sys
import traceback

import numpy as np
import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import dl0_autolabel_v2 as V2                          # noqa: E402
import drawer_asset_v2 as DA                           # noqa: E402
import xh_geom as XG                                   # noqa: E402

BINS = [(0.1, 0.4), (0.4, 1.0), (1.0, 2.0), (2.0, 3.1)]
TEST = ['R_x25_150', 'R_x40_300', 'N_k40']
TEMPLATES = ('A_front', 'B_oblique')
OPENING = os.path.join(HERE, 'results', 'vision', 'XH_test_opening.json')
NA = 'NA'


def center_eval(det, c_true):
    if not det.get('L2'):
        return {'L2': False, 'reject': det.get('reject'), 'reject_L2': det.get('reject_L2')}
    e = np.array(det['L2']['center']) - np.array(c_true)
    ax = np.array(det['L1']['axis'])
    ang = float(np.degrees(np.arccos(min(1.0, abs(ax @ np.array([1.0, 0, 0])) / np.linalg.norm(ax)))))
    return {'L2': True, 'center_err_mm': float(np.linalg.norm(e)) * 1e3, 'approach_signed_mm': float(e[1]) * 1e3, 'axis_deg': ang}


def frame_from_meta(D, meta, spec, f):
    """與 V2.frames_of_xh 同一欄位規則，但逐格建立（缺檔／缺欄拋 V2.EvidenceError 由呼叫端記去向）。"""
    K = np.array(meta['intrinsics_readback']['K'], float)
    bar = V2.bar_params(spec)
    n = f['n']
    dp = os.path.join(D, 'frames', f'f{n:02d}_depth_m.npy')
    rgb = os.path.join(D, 'frames', f'f{n:02d}_rgb.png')
    if not os.path.exists(dp):
        raise V2.EvidenceError('missing_depth')
    if not os.path.exists(rgb):
        raise V2.EvidenceError('missing_rgb')
    need = ['cam_pos_world', 'cam_quat_wxyz_world'] + (['handle_center_world_at_capture'] if bar else
           ['knob_center_world_at_capture', 'knob_axis_out_world_at_capture', 'observation_ref_center_world_at_capture'])
    miss = [x for x in need if f.get(x) is None]
    if miss:
        raise V2.EvidenceError(f'missing_fields {miss}')
    fr = {'n': n, 'rgb': rgb, 'depth_path': dp, 'T': V2._T(f['cam_pos_world'], f['cam_quat_wxyz_world']), 'K': K,
          'target_kind': 'R' if bar else 'N'}
    if bar:
        fr.update({'c': f['handle_center_world_at_capture'], 'bar': bar})
    else:
        k = spec['drawer']['distractors'][0]
        fr.update({'c': None, 'knob': {'outer_face_center_world': np.array(f['observation_ref_center_world_at_capture'], float),
                                       'axis_world': np.array(f['knob_axis_out_world_at_capture'], float),
                                       'radius': float(k['diameter']) / 2.0, 'length': float(k['protrusion'])}})
    return fr


def evaluate_frame(fr, A, seq, tmpl, infer, rngs):
    dep = V2.load_depth(fr['depth_path'])
    lx = V2.label_xh(fr, dep)
    img = np.asarray(Image.open(fr['rgb']).convert('RGB'))
    detected, pm = infer(img)
    rec = {'key': f'{seq}:{fr["n"]}', 'asset': A['id'], 'type': A['type'], 'template': tmpl, 'kind': lx['kind'],
           'neg_reason': None if lx['kind'] != 'negative' else ('N_asset' if A['type'] == 'N' else 'no_target_in_view'),
           'dist_m': lx.get('dist_m'), 'L1_detected': bool(detected)}
    fg, ig = lx['mask'], lx['ignore'] & ~lx['mask']
    if lx['kind'] == 'positive':
        ev = ~ig
        rec['L1_iou'] = float((pm & fg & ev).sum() / max(((pm | fg) & ev).sum(), 1))
    if A['type'] == 'R':
        d_m, L_m = A['diameter_mm'] / 1000.0, A['length_mm'] / 1000.0
        rec['L1'] = center_eval(XG.k_detect(dep, fr['K'], fr['T'], pm, rngs['L1'], d_m, L_m), fr['c'])
        rec['G1'] = center_eval(XG.k_detect(dep, fr['K'], fr['T'], fg, rngs['G1'], d_m, L_m), fr['c'])
        g0, _ = XG.g0k_detect(dep, fr['K'], fr['T'], rngs['G0K'], d_m, L_m)
        rec['G0K'] = center_eval(g0, fr['c'])
    else:
        rec['L1'] = rec['G1'] = rec['G0K'] = 'NA（無目標資產）'
    return rec


def run(infer, assets, asset_ids, runs_root, asset_dir):
    """回傳 res（含 rows、fates、evidence_issues、closure、complete、summary）。不拋出逐格例外。"""
    res = {'rows': [], 'fates': {}, 'evidence_issues': [], 'source_keys': []}
    for aid in asset_ids:
        A = assets[aid]
        spec = DA.load(os.path.join(asset_dir, f'drawer_unit_xh_{aid}.yaml'))
        for tmpl in TEMPLATES:
            seq = f'xh_{aid}_{tmpl}'
            D = os.path.join(runs_root, seq, 'wrist_v0')
            mp = os.path.join(D, 'meta.json')
            if not os.path.exists(mp):
                res['evidence_issues'].append({'key': f'{seq}:*', 'why': 'missing_capture'})
                res['fates'][f'{seq}:*'] = 'missing_capture'
                continue
            meta = json.load(open(mp))
            truth = json.load(open(os.path.join(D, 'truth.json')))
            is_R = V2.bar_params(spec) is not None
            if is_R != str(truth.get('target_kind', '')).startswith('R_bar'):
                for f in meta['frames']:
                    k = f'{seq}:{f["n"]}'
                    res['source_keys'].append(k)
                    res['fates'][k] = 'evidence:target_kind_mismatch'
                    res['evidence_issues'].append({'key': k, 'why': 'target_kind_mismatch'})
                continue
            rngs = {p: np.random.default_rng(0) for p in ('L1', 'G1', 'G0K')}
            for f in meta['frames']:
                k = f'{seq}:{f.get("n")}'
                if k in res['fates']:
                    res['evidence_issues'].append({'key': k, 'why': 'duplicate_frame_key'})
                    res['fates'][k + '#dup'] = 'duplicate_frame_key'
                    continue
                res['source_keys'].append(k)
                try:
                    fr = frame_from_meta(D, meta, spec, f)
                    res['rows'].append(evaluate_frame(fr, A, seq, tmpl, infer, rngs))
                    res['fates'][k] = 'evaluated'
                except V2.EvidenceError as e:
                    res['fates'][k] = f'evidence:{e}'
                    res['evidence_issues'].append({'key': k, 'why': str(e)})
                except Exception as e:                       # noqa: BLE001  逐格例外：記去向，不崩潰
                    res['fates'][k] = f'exception:{type(e).__name__}'
                    res['evidence_issues'].append({'key': k, 'why': f'{type(e).__name__}: {e}'})
    src = set(res['source_keys'])
    ev = {k for k, v in res['fates'].items() if v == 'evaluated'}
    iss = {k for k, v in res['fates'].items() if v != 'evaluated' and not k.endswith('#dup') and not k.endswith(':*')}
    res['closure'] = {'n_source': len(src), 'n_evaluated': len(ev), 'n_issue': len(iss), 'disjoint': not (ev & iss),
                      'closed': src == (ev | iss) and not (ev & iss), 'rows_match_evaluated': len(res['rows']) == len(ev)}
    res['complete'] = (not res['evidence_issues']) and res['closure']['closed'] and res['closure']['rows_match_evaluated']
    res['summary'] = summarize(res['rows'], assets, asset_ids)
    return res


def summarize(rows, assets, asset_ids):
    summ = {}
    for aid in asset_ids:
        rr = [r for r in rows if r['asset'] == aid]
        pos = [r for r in rr if r['kind'] == 'positive']
        s = {'n': len(rr), 'n_positive': len(pos)}
        s['L1_mask_iou_mean_missed0'] = float(np.mean([r['L1_iou'] for r in pos])) if pos else NA
        s['L1_detect_rate'] = float(np.mean([r['L1_detected'] for r in pos])) if pos else NA
        for path in ('L1', 'G1', 'G0K'):
            if assets[aid]['type'] != 'R' or not pos:
                s[path] = NA
                continue
            bins = {}
            for lo, hi in BINS:
                b = [r for r in pos if r['dist_m'] is not None and lo <= r['dist_m'] < hi]
                ok = [r[path] for r in b if r[path]['L2']]
                bins[f'{lo}-{hi}'] = {'n': len(b), 'output': len(ok),
                                      'center_err_mm_median': float(np.median([x['center_err_mm'] for x in ok])) if ok else NA}
            rej = {}
            for r in pos:
                if not r[path]['L2']:
                    k = str(r[path].get('reject') or r[path].get('reject_L2')).split(':')[0]
                    rej[k] = rej.get(k, 0) + 1
            out_ = [r for r in pos if r[path]['L2']]
            s[path] = {'output_rate': len(out_) / len(pos), 'output_and_err_le_10mm': f"{sum(1 for r in out_ if r[path]['center_err_mm'] <= 10)}/{len(pos)}",
                       'err_over_diameter_max': (max(r[path]['center_err_mm'] / assets[aid]['diameter_mm'] for r in out_) if out_ else NA),
                       'bins': bins, 'reject_reasons': rej}
        negs = [r for r in rr if r['kind'] == 'negative']
        s['negatives'] = {k: {'n': sum(1 for r in negs if r['neg_reason'] == k),
                              'L1_false_positive': sum(1 for r in negs if r['neg_reason'] == k and r['L1_detected'])}
                          for k in sorted({r['neg_reason'] for r in negs})}
        summ[aid] = s
    Rs = [a for a in asset_ids if assets[a]['type'] == 'R' and summ[a]['n_positive']]
    summ['asset_equal_weight_R'] = ({'L1_mask_iou': float(np.mean([summ[a]['L1_mask_iou_mean_missed0'] for a in Rs])),
                                     **{f'{p}_output_rate': float(np.mean([summ[a][p]['output_rate'] for a in Rs])) for p in ('L1', 'G1', 'G0K')}}
                                    if Rs else NA)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'results', 'vision', 'XH3_test_eval.json'))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f'{a.out} 已存在，不覆寫（測試只開封一次）')
    if not os.path.exists(OPENING) or json.load(open(OPENING)).get('state') != 'opened':
        raise SystemExit('沒有狀態 opened 的開封紀錄 ⇒ 拒絕（必須由 xh_open_test.py 執行）')
    res = {'schema': 'xh3_test_eval/2', 'ckpt': a.ckpt, 'complete': False}
    try:
        import torch
        import xh_train as TR
        dev = torch.device('cuda')
        model = TR.build_model(pretrained=False)
        model.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
        model.to(dev).eval()
        res.update({'ckpt_sha256': hashlib.sha256(open(a.ckpt, 'rb').read()).hexdigest(), 'score_t': TR.SCORE_T, 'bin_t': TR.BIN_T})

        def infer(img):
            with torch.no_grad():
                o = model([torch.from_numpy(img.copy()).permute(2, 0, 1).float().to(dev) / 255.0])[0]
            keep = o['scores'] >= TR.SCORE_T
            pm = (o['masks'][keep][0, 0] >= TR.BIN_T).cpu().numpy() if keep.any() else np.zeros(img.shape[:2], bool)
            return bool(keep.any()), pm
        reg = yaml.safe_load(open(os.path.join(HERE, 'results', 'vision', 'XH1_registration.yaml')))
        assets = {x['id']: x for x in reg['assets']}
        res.update(run(infer, assets, TEST, os.path.join(HERE, 'runs'), os.path.join(WS, 'src/my_omnibot_description/config/xh')))
        res['labels_note'] = {'R_x25_150': '名目開口篩選內；直徑未見、長度外插', 'R_x40_300': '名目開口篩選外；直徑與長度皆外插', 'N_k40': '干擾物（未見尺寸）'}
    except Exception as e:                                   # noqa: BLE001
        res['complete'] = False
        res['fatal'] = f'{type(e).__name__}: {e}'
        res['traceback'] = traceback.format_exc()
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({'complete': res.get('complete'), 'closure': res.get('closure'), 'n_issues': len(res.get('evidence_issues', [])),
                      'fatal': res.get('fatal')}, ensure_ascii=False, default=str))
    sys.exit(0 if res.get('complete') else 1)


if __name__ == '__main__':
    main()
