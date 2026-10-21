#!/usr/bin/env python3
"""D：跨機器／跨次渲染敏感性對照（Codex 20261021 裁定；不碰測試資產）。

同一條開發序列 R_d28_200_A_front：舊＝ASUS 原錄 runs/xh_R_d28_200_A_front；新＝MSI 重渲染 runs/xh_rerender_msi/xh_R_d28_200_A_front。
定位：跨機器＋跨次渲染的**合併**敏感性，不是硬體單因素實驗；不定義「可忽略」門檻、不據此調模型。

1. 先核對：資產檔雜湊、內參讀回、重播路徑與時間窗；影格以來源時刻 src_record_t 配對（不假設同編號＝同觀測），
   配對後再核對 rendering_time、相機位姿、真值把手中心。報完整配對／未配對分母。
2. 影像差（配對影格）：RGB 每像素最大通道絕對差（0–255）與深度絕對差，各報全畫面與把手區域（新舊真值輔助遮罩聯集），
   含有效深度比例（有限且 > 0）。
3. 推論：舊、新影像都在**本次同一推論環境**跑固定檢查點與三路徑（L1／G1／G0-K），只用配對影格、同順序、每路徑亂數種子 0
   （與 xh_test_eval.run 對單一序列的做法相同）。分開報輸出數、輸出且 ≤10 mm 計數，以及兩邊都有輸出的共同集合上的中心誤差。

    .venv-dl0/bin/python evaluation/xh_rerender_compare.py      # 輸出 results/vision/XH3_rerender_msi_compare.json（已存在則拒絕覆寫）
"""
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
import xh_test_eval as TE                              # noqa: E402
import xh_test_eval_v2 as E2                           # noqa: E402
import xh_train as TR                                  # noqa: E402

SEQ, AID, TMPL = 'xh_R_d28_200_A_front', 'R_d28_200', 'A_front'
OLD = os.path.join(HERE, 'runs', SEQ, 'wrist_v0')
NEW = os.path.join(HERE, 'runs', 'xh_rerender_msi', SEQ, 'wrist_v0')
OUT = os.path.join(HERE, 'results', 'vision', 'XH3_rerender_msi_compare.json')
ENV_OUT = os.path.join(HERE, 'results', 'vision', 'XH3_dev_rerender_msi_env.json')
ASSET = os.path.join(WS, 'src/my_omnibot_description/config/xh', f'drawer_unit_xh_{AID}.yaml')


def q(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return 'NA'
    return {'n': int(v.size), 'mean': float(v.mean()), 'median': float(np.median(v)), 'p99': float(np.percentile(v, 99)),
            'max': float(v.max())}


def quat_angle_deg(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(2 * np.arccos(min(1.0, abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))))))


def main():
    for p in (OUT, ENV_OUT):
        if os.path.exists(p):
            raise SystemExit(f'{p} 已存在，不覆寫')
    mo, mn = json.load(open(os.path.join(OLD, 'meta.json'))), json.load(open(os.path.join(NEW, 'meta.json')))
    spec = DA.load(ASSET)
    res = {'schema': 'xh3_rerender_compare/1', 'seq': SEQ, 'old': os.path.relpath(OLD, WS), 'new': os.path.relpath(NEW, WS),
           'positioning': '跨機器（ASUS → MSI）＋跨次渲染的合併敏感性對照；不是硬體單因素實驗；未定義可忽略門檻'}
    # 1 前置核對
    sim_old = open(os.path.join(OLD, '..', 'sim.log'), errors='replace').read()
    sim_new = open(os.path.join(NEW, '..', 'sim.log'), errors='replace').read()
    rel = os.path.relpath(ASSET, WS)
    res['precheck'] = {
        'asset_sha256_now': hashlib.sha256(open(ASSET, 'rb').read()).hexdigest(),
        'asset_in_old_sim_log': rel in sim_old, 'asset_in_new_sim_log': rel in sim_new,
        'K_old': mo['intrinsics_readback']['K'], 'K_new': mn['intrinsics_readback']['K'],
        'K_equal': np.allclose(mo['intrinsics_readback']['K'], mn['intrinsics_readback']['K'], atol=0, rtol=0),
        'replay_old': mo.get('replay'), 'replay_new': mn.get('replay'), 'replay_equal': mo.get('replay') == mn.get('replay'),
        'n_frames_old': len(mo['frames']), 'n_frames_new': len(mn['frames']),
        'n_rejected_old': len(mo.get('rejected_frames') or []), 'n_rejected_new': len(mn.get('rejected_frames') or [])}
    # 配對：來源時刻
    key = lambda f: round(float(f['src_record_t']), 6)   # noqa: E731
    io, inew = {key(f): f for f in mo['frames']}, {key(f): f for f in mn['frames']}
    common = sorted(set(io) & set(inew))
    res['pairing'] = {'by': 'src_record_t（round 1e-6 s）', 'n_old': len(io), 'n_new': len(inew), 'n_paired': len(common),
                      'old_only_src_t': sorted(set(io) - set(inew)), 'new_only_src_t': sorted(set(inew) - set(io)),
                      'dup_old': len(mo['frames']) - len(io), 'dup_new': len(mn['frames']) - len(inew)}
    pc = []
    for t in common:
        a, b = io[t], inew[t]
        pc.append({'src_t': t, 'n_old': a['n'], 'n_new': b['n'],
                   'rendering_time_diff_s': abs(a['rendering_time'] - b['rendering_time']),
                   'cam_pos_diff_mm': float(np.linalg.norm(np.subtract(a['cam_pos_world'], b['cam_pos_world']))) * 1e3,
                   'cam_rot_diff_deg': quat_angle_deg(a['cam_quat_wxyz_world'], b['cam_quat_wxyz_world']),
                   'handle_center_diff_mm': float(np.linalg.norm(np.subtract(a['handle_center_world_at_capture'],
                                                                              b['handle_center_world_at_capture']))) * 1e3})
    res['pair_consistency'] = {k: q([x[k] for x in pc]) for k in ('rendering_time_diff_s', 'cam_pos_diff_mm', 'cam_rot_diff_deg',
                                                                  'handle_center_diff_mm')}
    res['pair_consistency']['n_same_frame_number'] = sum(1 for x in pc if x['n_old'] == x['n_new'])
    # 2 影像差＋3 推論
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
    A = {x['id']: x for x in yaml.safe_load(open(os.path.join(HERE, 'results', 'vision', 'XH1_registration.yaml')))['assets']}[AID]
    rngs = {side: {p: np.random.default_rng(0) for p in ('L1', 'G1', 'G0K')} for side in ('old', 'new')}
    img_rows, ev_rows, issues = [], [], []
    for t in common:
        fo = TE.frame_from_meta(OLD, mo, spec, io[t])
        fn = TE.frame_from_meta(NEW, mn, spec, inew[t])
        ro_, rn_ = np.asarray(Image.open(fo['rgb']).convert('RGB'), int), np.asarray(Image.open(fn['rgb']).convert('RGB'), int)
        do_, dn_ = V2.load_depth(fo['depth_path']).astype(float), V2.load_depth(fn['depth_path']).astype(float)
        lo, ln = V2.label_xh(fo, do_), V2.label_xh(fn, dn_)
        reg = lo['mask'] | ln['mask']
        drgb = np.abs(ro_ - rn_).max(axis=2)
        vo, vn = np.isfinite(do_) & (do_ > 0), np.isfinite(dn_) & (dn_ > 0)
        both = vo & vn
        dd = np.abs(do_ - dn_)
        row = {'src_t': t, 'kind_old': lo['kind'], 'kind_new': ln['kind'], 'mask_px_old': int(lo['mask'].sum()),
               'mask_px_new': int(ln['mask'].sum()), 'region_px': int(reg.sum()),
               'rgb_full_mean': float(drgb.mean()), 'rgb_full_max': int(drgb.max()), 'rgb_full_frac_changed': float((drgb > 0).mean()),
               'depth_valid_old': float(vo.mean()), 'depth_valid_new': float(vn.mean()),
               'depth_full_mean_mm': float(dd[both].mean()) * 1e3 if both.any() else 'NA',
               'depth_full_max_mm': float(dd[both].max()) * 1e3 if both.any() else 'NA',
               'depth_full_frac_changed': float((dd[both] > 0).mean()) if both.any() else 'NA'}
        if reg.any():
            row.update({'rgb_region_mean': float(drgb[reg].mean()), 'rgb_region_max': int(drgb[reg].max()),
                        'rgb_region_frac_changed': float((drgb[reg] > 0).mean()),
                        'depth_valid_region_old': float(vo[reg].mean()), 'depth_valid_region_new': float(vn[reg].mean())})
            br = both & reg
            row['depth_region_mean_mm'] = float(dd[br].mean()) * 1e3 if br.any() else 'NA'
            row['depth_region_max_mm'] = float(dd[br].max()) * 1e3 if br.any() else 'NA'
        img_rows.append(row)
        pair = {'src_t': t}
        for side, fr in (('old', fo), ('new', fn)):
            try:
                pair[side] = TE.evaluate_frame(fr, A, SEQ, TMPL, infer, rngs[side])
            except Exception as e:                       # noqa: BLE001
                pair[side] = None
                issues.append({'src_t': t, 'side': side, 'why': f'{type(e).__name__}: {e}'})
        ev_rows.append(pair)
    res['image_rows'] = img_rows
    num = lambda k: [r[k] for r in img_rows if isinstance(r.get(k), (int, float))]   # noqa: E731
    res['image_summary'] = {k: q(num(k)) for k in ('rgb_full_mean', 'rgb_full_max', 'rgb_full_frac_changed', 'rgb_region_mean',
                                                    'rgb_region_max', 'rgb_region_frac_changed', 'depth_valid_old', 'depth_valid_new',
                                                    'depth_valid_region_old', 'depth_valid_region_new', 'depth_full_mean_mm',
                                                    'depth_full_max_mm', 'depth_full_frac_changed', 'depth_region_mean_mm',
                                                    'depth_region_max_mm')}
    res['image_summary']['n_kind_differs'] = sum(1 for r in img_rows if r['kind_old'] != r['kind_new'])
    res['image_summary']['n_identical_rgb'] = sum(1 for r in img_rows if r['rgb_full_max'] == 0)
    res['eval_rows'] = ev_rows
    res['eval_issues'] = issues
    ok_rows = [r for r in ev_rows if r['old'] and r['new']]
    summ = {'n_pairs_evaluated': len(ok_rows)}
    for side in ('old', 'new'):
        pos = [r[side] for r in ok_rows if r[side]['kind'] == 'positive']
        summ[f'{side}_n_positive'] = len(pos)
        summ[f'{side}_L1_detected'] = sum(1 for r in pos if r['L1_detected'])
        summ[f'{side}_L1_iou_mean_missed0'] = float(np.mean([r['L1_iou'] for r in pos])) if pos else 'NA'
        for p in ('L1', 'G1', 'G0K'):
            out_ = [r for r in pos if r[p]['L2']]
            summ[f'{side}_{p}'] = {'output': len(out_), 'output_and_le_10mm': sum(1 for r in out_ if r[p]['center_err_mm'] <= 10)}
    both_pos = [r for r in ok_rows if r['old']['kind'] == 'positive' and r['new']['kind'] == 'positive']
    summ['n_both_positive'] = len(both_pos)
    for p in ('L1', 'G1', 'G0K'):
        cm = [r for r in both_pos if r['old'][p]['L2'] and r['new'][p]['L2']]
        summ[f'common_{p}'] = {
            'n': len(cm), 'old_only_output': sum(1 for r in both_pos if r['old'][p]['L2'] and not r['new'][p]['L2']),
            'new_only_output': sum(1 for r in both_pos if r['new'][p]['L2'] and not r['old'][p]['L2']),
            'old_center_err_mm': q([r['old'][p]['center_err_mm'] for r in cm]),
            'new_center_err_mm': q([r['new'][p]['center_err_mm'] for r in cm]),
            'abs_diff_center_err_mm': q([abs(r['new'][p]['center_err_mm'] - r['old'][p]['center_err_mm']) for r in cm]),
            'abs_diff_approach_signed_mm': q([abs(r['new'][p]['approach_signed_mm'] - r['old'][p]['approach_signed_mm']) for r in cm])}
    res['eval_summary'] = summ
    res['env'] = E2.env_fingerprint_v2(torch)
    json.dump({'for': os.path.relpath(OUT, WS), 'note': 'D 對照（MSI 重渲染＋推論）同一程序內記錄；供 xh_test_eval_v2 的 env_vs_dev_ref',
               'env': res['env']}, open(ENV_OUT, 'w'), ensure_ascii=False, indent=1)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: res[k] for k in ('precheck', 'pairing', 'pair_consistency')}, ensure_ascii=False, default=str)[:2500])
    print(json.dumps({'image_summary': res['image_summary'], 'eval_summary': summ}, ensure_ascii=False, default=str)[:5000])


if __name__ == '__main__':
    main()
