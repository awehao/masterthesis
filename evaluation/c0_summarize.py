#!/usr/bin/env python3
"""C0 批次彙整：逐趟跑已凍結的判定工具，再算配對差（H1 − H5）。

不新增指標定義 —— 只呼叫 horizon_replay_check／motm_physical_check／horizon_config_check／
horizon_metrics／motm_metrics，並把它們的輸出排成表。分析規則見 experiment_spec_C0.yaml：
樣本層級是趟次；n = 3 對只報三對各自的差、平均與範圍，不做顯著性宣稱。

    python3 evaluation/c0_summarize.py --batch c0b1 [--out results/horizon_ablation/c0b1_summary.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')


def sh(args):
    p = subprocess.run([sys.executable] + args, cwd=HERE, capture_output=True, text=True)
    return p.returncode, p.stdout


def run_tool(args, out_path, ok_rcs):
    """執行一支判定工具。先刪舊輸出檔，避免工具失敗時讀到舊結果冒充本次。
    回傳 (結果 dict 或 None, rc, 錯誤說明或 None)。"""
    if out_path and os.path.exists(out_path):
        os.remove(out_path)
    rc, txt = sh(args)
    if rc not in ok_rcs:
        return None, rc, f'{args[0]} 離開碼 {rc} 不在預期 {sorted(ok_rcs)}'
    if out_path:
        if not os.path.exists(out_path):
            return None, rc, f'{args[0]} 沒有寫出 {os.path.basename(out_path)}'
        try:
            return json.load(open(out_path)), rc, None
        except ValueError as e:
            return None, rc, f'{args[0]} 輸出無法解析：{e}'
    try:
        return json.loads(txt[txt.index('{'):]), rc, None
    except ValueError as e:
        return None, rc, f'{args[0]} 輸出無法解析：{e}'


def dig(d, *keys):
    """安全取值：任何一層缺漏或型別不對都回 None。"""
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def analyse(rid, expect_open_m=None):
    """逐趟跑判定工具並取指標。缺欄位或工具失敗逐趟記為證據問題（evidence_issues），
    相關指標填 None，不拋例外、不讀舊分析檔。"""
    D = os.path.join(RUNS, rid)
    A = os.path.join(D, 'analysis')
    os.makedirs(A, exist_ok=True)
    out = {'run': rid, 'evidence_issues': [], 'tool_errors': []}
    if not os.path.exists(os.path.join(D, 'align_solver.json')):
        out['tool_errors'].append('缺 align_solver.json')
        return out
    rp, out['replay_rc'], e1 = run_tool(
        ['horizon_replay_check.py', D, '--out', os.path.join(A, 'replay_check.json')],
        os.path.join(A, 'replay_check.json'), {0, 1, 2, 3})
    pargs = ['motm_physical_check.py', D, '--out', os.path.join(A, 'physical_check.json')]
    if expect_open_m is not None:
        pargs += ['--expect-open-m', repr(float(expect_open_m))]
    pc, out['physical_rc'], e2 = run_tool(pargs, os.path.join(A, 'physical_check.json'), {0, 1, 2})
    hm, out['metrics_rc'], e3 = run_tool(
        ['horizon_metrics.py', D, '--out', os.path.join(A, 'horizon_metrics.json')],
        os.path.join(A, 'horizon_metrics.json'), {0})
    mm, out['motm_rc'], e4 = run_tool(['motm_metrics.py', D], None, {0})
    if mm is not None:
        json.dump(mm, open(os.path.join(A, 'motm_metrics.json'), 'w'), ensure_ascii=False, indent=1)
    out['tool_errors'] += [e for e in (e1, e2, e3, e4) if e]
    task = json.load(open(os.path.join(D, 'task.json'))) if os.path.exists(os.path.join(D, 'task.json')) else {}
    ph = {e['phase']: float(e['sim_t']) for e in task.get('events', []) if 'phase' in e}
    W = dig(hm, 'by_phase', 'WINDOW') or {}

    def win(*k):
        return dig(W, *k)
    s4 = dig(pc, 'S4_drift')
    s4 = s4 if isinstance(s4, dict) else {}
    s5 = dig(pc, 'S5_home')
    s5 = s5 if isinstance(s5, dict) else {}
    s1 = dig(pc, 'S1_open_hold')
    s1 = s1 if isinstance(s1, dict) else {}
    out.update({
        'N': dig(hm, 'N') if hm else dig(rp, 'N_args'),
        'flow_final_phase': dig(mm, 'final_phase'), 'abort': dig(mm, 'abort'),
        'replay': dig(rp, 'verdict'), 'physical': dig(pc, 'verdict'),
        'physical_failed': dig(pc, 'failed'), 'physical_insufficient': dig(pc, 'insufficient'),
        'm': {
            'open_hold_s': s1.get('longest_s'),
            'grasp_longest_s': dig(pc, 'S3_grasp', 'longest_both_fingers_s'),
            'grip_fraction_in_window': s4.get('grip_fraction_in_window'),
            'drift_max_mm': s4.get('max_mm'),
            'home_dist_m': s5.get('dist_m'),
            'window_s': (None if not hm else round(hm['window']['to_sim_t'] - hm['window']['from_sim_t'], 3)),
            'align_to_done_s': (round(ph['DONE'] - ph['ALIGN'], 3)
                                if 'DONE' in ph and 'ALIGN' in ph else None),
            'disallowed_stops': dig(mm, 'whole_run', 'n_disallowed_stops'),
            'longest_dwell_excl_open_hold_s': dig(mm, 'low_speed_dwell', 'longest_excl_open_hold_s'),
            'err_p_p95_m': win('tracking', 'err_p_m', 'p95'),
            'err_p_max_m': win('tracking', 'err_p_m', 'max'),
            'err_r_p95_rad': win('tracking', 'err_r_rad', 'p95'),
            'sp_margin_min_rad': win('limits', 'setpoint', 'min_rad'),
            'meas_margin_min_rad': win('limits', 'measured', 'min_rad'),
            'arm_rev_adjacent_per_s': win('smoothness', 'arm_reversals', 'adjacent_rate_per_s'),
            'arm_rev_adjacent': win('smoothness', 'arm_reversals', 'adjacent'),
            'arm_rev_valid_pairs': win('smoothness', 'arm_reversals', 'adjacent_valid_pairs'),
            'du_arm_p95': win('smoothness', 'du_arm', 'p95'),
            'du_base_p95': win('smoothness', 'du_base', 'p95'),
            'tcp_speed_10ms_max_mps': win('smoothness', 'tcp_speed_10ms_mps', 'max'),
            'core_ms_p50': win('compute', 'core_ms', 'p50'),
            'core_ms_p99': win('compute', 'core_ms', 'p99'),
            'cycle_wall_ms_p95': win('compute', 'cycle_wall_ms', 'p95'),
            'cycle_wall_ms_max': win('compute', 'cycle_wall_ms', 'max'),
            'n_missed_slot': dig(hm, 'solver', 'n_missed_slot'),
            'n_failed_solve': dig(hm, 'solver', 'n_failed'),
            'n_shaped': dig(hm, 'solver', 'n_shaped'),
            'n_cycles': dig(hm, 'solver', 'n_cycles'),
            'arm_share_open_mm': dig(mm, 'share_OPEN', 'arm_only_dy_mm'),
            'arm_share_close_mm': dig(mm, 'share_CLOSE', 'arm_only_dy_mm'),
        }})
    if out['physical_insufficient']:
        out['evidence_issues'].append('物理判定證據不足：' + '、'.join(out['physical_insufficient']))
    missing = [k for k, v in out['m'].items() if v is None]
    if missing:
        out['evidence_issues'].append('指標缺值：' + '、'.join(missing))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True)
    ap.add_argument('--pair-ids', default='p1,p2,p3',
                    help='配對名稱（逗號分隔），例如 p1,p2,p3r（整對補跑見 c0b1_revision_*.yaml）')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    pids = [x.strip() for x in a.pair_ids.split(',') if x.strip()]
    res = {}
    for p in pids:
        for M in ('H5', 'H1'):
            rid = f'{a.batch}_{p}_{M}'
            res[rid] = analyse(rid)
    cfg = {}
    for p in pids:
        h5, h1 = (os.path.join(RUNS, f'{a.batch}_{p}_{M}') for M in ('H5', 'H1'))
        rc, txt = sh(['horizon_config_check.py', h5, h1, '--allow', 'solver.N'])
        cfg[p] = {'rc': rc, 'text': txt.strip()}
    keys = list(next(v for v in res.values() if 'm' in v)['m'].keys())
    paired = {}
    for k in keys:
        d = []
        for p in pids:
            r5, r1 = res[f'{a.batch}_{p}_H5'], res[f'{a.batch}_{p}_H1']
            v5 = (r5.get('m') or {}).get(k)
            v1 = (r1.get('m') or {}).get(k)
            d.append(None if v5 is None or v1 is None else round(v1 - v5, 6))
        h5v = [(res[f'{a.batch}_{p}_H5'].get('m') or {}).get(k) for p in pids]
        h1v = [(res[f'{a.batch}_{p}_H1'].get('m') or {}).get(k) for p in pids]
        dd = [x for x in d if x is not None]
        paired[k] = {'H5': h5v, 'H1': h1v, 'diff_H1_minus_H5': d,
                     'diff_mean': round(float(np.mean(dd)), 6) if dd else None,
                     'diff_range': [min(dd), max(dd)] if dd else None,
                     'same_sign_all_pairs': (len(dd) == len(pids) and
                                             (all(x > 0 for x in dd) or all(x < 0 for x in dd)))}
    out = {'batch': a.batch, 'pair_ids': pids, 'runs': res, 'config_check': cfg, 'paired': paired,
           'note': 'n = 3 對；只報各對差、平均與範圍，不做顯著性宣稱；樣本層級是趟次'}
    print(json.dumps({'runs': {k: {kk: v.get(kk) for kk in ('N', 'flow_final_phase', 'replay',
                                                               'physical', 'physical_failed',
                                                               'physical_insufficient')}
                               for k, v in res.items()},
                      'config_rc': {k: v['rc'] for k, v in cfg.items()}},
                     ensure_ascii=False, indent=1))
    if a.out:
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
