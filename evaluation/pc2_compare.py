#!/usr/bin/env python3
"""PC2：G0→DL2 像素座標契約新版 vs 凍結舊版的離線對照（兩開發群組 186 格；規格 results/vision/PC2_pixel_contract_v2_spec.md）。

新版：d1_handle_detect_v2（像素中心反投影）依原順序與種子重算；dl2_obs_to_target_v2（N-obs，新版反投影）。
舊版：凍結紀錄 d1_detect_rev2.json／d1_detect_hold.json（G0）與 DL2_dev_eval_r3.json（G0/N-obs）。
評估端用真值：接近軸有號與絕對誤差、共同接受集合配對（G0 中心與 DL2 工具目標分開）、GC1 固定真值物體的靜態模型相容性。
不線上替換、不重新鎖定、不接續接近或閉爪。

    python3 evaluation/pc2_compare.py       # 輸出 results/vision/PC2_compare.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import d1_handle_detect_v2 as D1v2                    # noqa: E402
import dl2_eval as EV                                 # noqa: E402
import dl2_obs_to_target_v2 as E2v2                   # noqa: E402
import geometry_contract as GC                        # noqa: E402
import object_target_geometry as OT                   # noqa: E402

assert E2v2.D1 is D1v2, 'DL2 新版必須使用新版反投影'
GROUPS = {'traj_wg4b_f02_P': ('runs/d1_dev_f02P', 'd1_detect_rev2.json'),
          'traj_mt_b1_02_P': ('runs/d1_hold_mt02P', 'd1_detect_hold.json')}
A = np.array([0.0, 1.0, 0.0])
BINS = [(0.1, 0.4), (0.4, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 99.0)]


def gc_margin(C1, c_true, T_WE0, T_HG, a_H):
    """固定真值物體：以等效 T_HG 使 s = 0 的目標恰為估計工具目標，回報根部／淺側餘裕（靜態模型相容性）。"""
    H = np.eye(4)
    H[:3, 3] = c_true
    T_WO = H @ np.linalg.inv(OT.trans(C1['bar_c']))
    g = GC.check(C1, T_WO, 0.0, np.linalg.inv(H) @ np.asarray(T_WE0, float), a_H, 0.0)
    gc = g.get('grasp_compat') or {}
    out = {'ok': g['ok'], 'why': g.get('why'), 'root_gap_mm': gc.get('root_gap_along_approach_mm')}
    if gc.get('bar_center_z_finger_mm') is not None:
        out['shallow_margin_mm'] = gc['blade_z_range_mm'][1] - gc['bar_center_z_finger_mm']
    return out


def main():
    out_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'vision', 'PC2_compare.json')
    if os.path.exists(out_p):
        raise SystemExit(f'{out_p} 已存在，不覆寫')
    Rg = EV.R_grasp()
    T_HG, a_H = E2v2.baseline_grasp(Rg)
    C1 = GC.load_contract()
    assert C1['ok'], C1.get('why')
    dl2 = json.load(open(os.path.join(HERE, 'results', 'vision', 'DL2_dev_eval_r3.json')))
    res = {'schema': 'pc2_compare/1', 'detector_v2_sha256': __import__('hashlib').sha256(open(D1v2.__file__, 'rb').read()).hexdigest(),
           'estimator_v2_sha256': __import__('hashlib').sha256(open(E2v2.__file__, 'rb').read()).hexdigest(), 'rows': []}
    for grp, (run, fname) in GROUPS.items():
        V = os.path.join(HERE, run, 'wrist_v0')
        m = json.load(open(os.path.join(V, 'meta.json')))
        K = np.array(m['intrinsics_readback']['K'])
        old = {r['n']: r for r in json.load(open(os.path.join(V, fname)))['rows']}
        d2 = {r['n']: r for r in dl2['groups'][grp]['rows']}
        rng = np.random.default_rng(0)
        for f in m['frames']:
            T = D1v2.cam_T(f)
            dep = np.load(os.path.join(V, 'frames', f'f{f["n"]:02d}_depth_m.npy'))
            dep = dep[:, :, 0] if dep.ndim == 3 else dep
            det, _ = D1v2.detect(dep, K, T, rng)
            c = np.array(f['handle_center_world_at_capture'], float)
            H = np.eye(4)
            H[:3, 3] = c
            Tt = OT.object_target(H, T_HG, a_H, 0.0)['T_WE']
            rec = {'key': f'{grp}:{f["n"]}', 'group': grp, 'n': f['n'], 'dist': float(np.linalg.norm(c - T[:3, 3]))}
            o = old.get(f['n'])
            rec['old_G0'] = ({'L2': bool(o['L2']), 'reject': o['reject'], 'reject_L2': o['reject_L2']} if o is not None
                             else {'evidence': 'missing_old_row'})
            rec['new_G0'] = {'L2': bool(det['L2']), 'reject': det['reject'], 'reject_L2': det['reject_L2']}
            for tag, L2 in (('old_G0', o['L2'] if o else None), ('new_G0', det['L2'])):
                if L2:
                    e = np.array(L2['center']) - c
                    rec[tag].update({'center_signed_mm': float(A @ e) * 1e3, 'center_abs_mm': abs(float(A @ e)) * 1e3,
                                     'center_3d_mm': float(np.linalg.norm(e)) * 1e3})
            # DL2：舊＝凍結 r3；新＝新版估計端在新版偵測輸出上
            x_old = d2.get(f['n'], {}).get('routes', {}).get('G0/N-obs')
            rec['old_DL2'] = ({'ok': bool(x_old['ok']), 'why': x_old.get('why')} if x_old is not None else {'evidence': 'missing_old_row'})
            obs = {'path': 'G0', 'L1': det['L1'], 'L2': det['L2'], 'reject': det['reject'], 'reject_L2': det['reject_L2'],
                   'depth': dep, 'K': K, 'T_cam': T, 't_obs': f['rendering_time']}
            r_new = E2v2.estimate(obs, 'N-obs', Rg, T_HG, a_H, query_t=f['rendering_time'])
            rec['new_DL2'] = {'ok': r_new['ok'], 'why': r_new.get('why')}
            for tag, ok, T0, R_WH in (('old_DL2', bool(x_old and x_old['ok']), x_old and x_old.get('T_WE_s0'), x_old and x_old.get('T_WH')),
                                      ('new_DL2', r_new['ok'], r_new.get('T_WE_s0'), r_new.get('T_WH'))):
                if ok:
                    T0 = np.asarray(T0, float)
                    e = T0[:3, 3] - Tt[:3, 3]
                    Rw = np.asarray(R_WH, float)[:3, :3]
                    rec[tag].update({'tool_signed_mm': float(A @ e) * 1e3, 'tool_abs_mm': abs(float(A @ e)) * 1e3,
                                     'tool_3d_mm': float(np.linalg.norm(e)) * 1e3,
                                     'axial_deg': math.degrees(math.acos(min(1.0, abs(float(Rw[:, 0] @ np.array([1.0, 0, 0])))))),
                                     'gc1_fixed_truth_object': gc_margin(C1, c, T0, T_HG, a_H)})
            res['rows'].append(rec)
    rows = res['rows']

    def summ(sel_rows, kind):
        o_key, n_key = f'old_{kind}', f'new_{kind}'
        acc = 'L2' if kind == 'G0' else 'ok'
        sig = 'center_signed_mm' if kind == 'G0' else 'tool_signed_mm'
        ab = 'center_abs_mm' if kind == 'G0' else 'tool_abs_mm'
        both = [r for r in sel_rows if r[o_key].get(acc) and r[n_key].get(acc)]
        out = {'n_frames': len(sel_rows), 'n_old_accept': sum(1 for r in sel_rows if r[o_key].get(acc)),
               'n_new_accept': sum(1 for r in sel_rows if r[n_key].get(acc)), 'n_common': len(both),
               'old_only': [r['key'] for r in sel_rows if r[o_key].get(acc) and not r[n_key].get(acc)],
               'new_only': [r['key'] for r in sel_rows if r[n_key].get(acc) and not r[o_key].get(acc)],
               'missing_evidence': [r['key'] for r in sel_rows if r[o_key].get('evidence')]}
        if both:
            q = lambda v: {'median': float(np.median(v)), 'p5': float(np.percentile(v, 5)), 'p95': float(np.percentile(v, 95)),
                           'min': float(np.min(v)), 'max': float(np.max(v))}
            out['common_old_signed'] = q([r[o_key][sig] for r in both])
            out['common_new_signed'] = q([r[n_key][sig] for r in both])
            out['common_old_abs'] = q([r[o_key][ab] for r in both])
            out['common_new_abs'] = q([r[n_key][ab] for r in both])
            out['paired_abs_change_new_minus_old'] = q([r[n_key][ab] - r[o_key][ab] for r in both])
            out['n_abs_worse'] = sum(1 for r in both if r[n_key][ab] > r[o_key][ab] + 1e-9)
            out['n_sign_flip'] = sum(1 for r in both if np.sign(r[n_key][sig]) != np.sign(r[o_key][sig]))
            if kind == 'DL2':
                for t in ('old', 'new'):
                    g = [r[f'{t}_DL2']['gc1_fixed_truth_object'] for r in both]
                    out[f'{t}_gc1_root_gap_mm'] = q([x['root_gap_mm'] for x in g if x['root_gap_mm'] is not None])
                    out[f'{t}_gc1_shallow_mm'] = q([x['shallow_margin_mm'] for x in g if x.get('shallow_margin_mm') is not None])
                    out[f'{t}_gc1_n_ok'] = sum(1 for x in g if x['ok'])
        return out
    res['summary'] = {}
    for scope in ('traj_wg4b_f02_P', 'traj_mt_b1_02_P', 'combined'):
        sel = [r for r in rows if scope == 'combined' or r['group'] == scope]
        res['summary'][scope] = {'G0': summ(sel, 'G0'), 'DL2': summ(sel, 'DL2'),
                                 'by_bin_DL2': {f'{lo}-{hi}': summ([r for r in sel if lo <= r['dist'] < hi], 'DL2') for lo, hi in BINS}}
    json.dump(res, open(out_p, 'w'), ensure_ascii=False, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    S = res['summary']['combined']
    for k in ('G0', 'DL2'):
        s_ = S[k]
        print(k, 'old', s_['n_old_accept'], 'new', s_['n_new_accept'], 'common', s_['n_common'], 'old_only', len(s_['old_only']), 'new_only', len(s_['new_only']))
        if s_['n_common']:
            print('  signed old', round(s_['common_old_signed']['median'], 3), 'new', round(s_['common_new_signed']['median'], 3),
                  '| abs old', round(s_['common_old_abs']['median'], 3), 'new', round(s_['common_new_abs']['median'], 3),
                  '| worse', s_['n_abs_worse'], 'flip', s_['n_sign_flip'])
    print('寫出', out_p)


if __name__ == '__main__':
    main()
