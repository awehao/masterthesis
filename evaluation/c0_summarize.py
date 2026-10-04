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


def analyse(rid):
    D = os.path.join(RUNS, rid)
    A = os.path.join(D, 'analysis')
    os.makedirs(A, exist_ok=True)
    out = {'run': rid}
    if not os.path.exists(os.path.join(D, 'align_solver.json')):
        out['status'] = '缺 align_solver.json（啟動前中止？）'
        return out
    rc, _ = sh(['horizon_replay_check.py', D, '--out', os.path.join(A, 'replay_check.json')])
    out['replay_rc'] = rc
    rc, _ = sh(['motm_physical_check.py', D, '--out', os.path.join(A, 'physical_check.json')])
    out['physical_rc'] = rc
    rc, _ = sh(['horizon_metrics.py', D, '--out', os.path.join(A, 'horizon_metrics.json')])
    out['metrics_rc'] = rc
    rc, txt = sh(['motm_metrics.py', D])
    open(os.path.join(A, 'motm_metrics.out'), 'w').write(txt)
    out['motm_rc'] = rc
    rp = json.load(open(os.path.join(A, 'replay_check.json')))
    pc = json.load(open(os.path.join(A, 'physical_check.json')))
    hm = json.load(open(os.path.join(A, 'horizon_metrics.json')))
    mm = json.loads(txt[txt.index('{'):]) if '{' in txt else {}
    task = json.load(open(os.path.join(D, 'task.json')))
    ph = {e['phase']: float(e['sim_t']) for e in task['events'] if 'phase' in e}
    W = hm['by_phase']['WINDOW']
    sm = W['smoothness']
    ar = sm['arm_reversals']
    out.update({
        'N': hm['N'],
        'flow_final_phase': mm.get('final_phase'), 'abort': mm.get('abort'),
        'replay': rp['verdict'], 'physical': pc['verdict'],
        'physical_failed': pc['failed'], 'physical_insufficient': pc['insufficient'],
        'm': {
            'open_hold_s': pc['S1_open_hold']['longest_s'],
            'grasp_longest_s': pc['S3_grasp']['longest_both_fingers_s'],
            'grip_fraction_in_window': (pc['S4_drift'] or {}).get('grip_fraction_in_window')
            if isinstance(pc['S4_drift'], dict) else None,
            'drift_max_mm': pc['S4_drift'].get('max_mm') if isinstance(pc['S4_drift'], dict) else None,
            'home_dist_m': pc['S5_home'].get('dist_m') if isinstance(pc['S5_home'], dict) else None,
            'window_s': round(hm['window']['to_sim_t'] - hm['window']['from_sim_t'], 3),
            'align_to_done_s': (round(ph['DONE'] - ph['ALIGN'], 3)
                                if 'DONE' in ph and 'ALIGN' in ph else None),
            'disallowed_stops': (mm.get('whole_run') or {}).get('n_disallowed_stops'),
            'longest_dwell_excl_open_hold_s': (mm.get('low_speed_dwell') or {}).get('longest_excl_open_hold_s'),
            'err_p_p95_m': W['tracking']['err_p_m']['p95'],
            'err_p_max_m': W['tracking']['err_p_m']['max'],
            'err_r_p95_rad': W['tracking']['err_r_rad']['p95'],
            'sp_margin_min_rad': W['limits']['setpoint']['min_rad'],
            'meas_margin_min_rad': W['limits']['measured']['min_rad'],
            'arm_rev_adjacent_per_s': ar['adjacent_rate_per_s'],
            'arm_rev_adjacent': ar['adjacent'],
            'arm_rev_valid_pairs': ar['adjacent_valid_pairs'],
            'du_arm_p95': sm['du_arm']['p95'], 'du_base_p95': sm['du_base']['p95'],
            'tcp_speed_10ms_max_mps': sm['tcp_speed_10ms_mps']['max'],
            'core_ms_p50': W['compute']['core_ms']['p50'],
            'core_ms_p99': W['compute']['core_ms']['p99'],
            'cycle_wall_ms_p95': W['compute']['cycle_wall_ms']['p95'],
            'cycle_wall_ms_max': W['compute']['cycle_wall_ms']['max'],
            'n_missed_slot': hm['solver']['n_missed_slot'],
            'n_failed_solve': hm['solver']['n_failed'],
            'n_shaped': hm['solver']['n_shaped'],
            'n_cycles': hm['solver']['n_cycles'],
            'arm_share_open_mm': (mm.get('share_OPEN') or {}).get('arm_only_dy_mm'),
            'arm_share_close_mm': (mm.get('share_CLOSE') or {}).get('arm_only_dy_mm'),
        }})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True)
    ap.add_argument('--pairs', type=int, default=3)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    res = {}
    for p in range(1, a.pairs + 1):
        for M in ('H5', 'H1'):
            rid = f'{a.batch}_p{p}_{M}'
            res[rid] = analyse(rid)
    cfg = {}
    for p in range(1, a.pairs + 1):
        h5, h1 = (os.path.join(RUNS, f'{a.batch}_p{p}_{M}') for M in ('H5', 'H1'))
        rc, txt = sh(['horizon_config_check.py', h5, h1, '--allow', 'solver.N'])
        cfg[f'p{p}'] = {'rc': rc, 'text': txt.strip()}
    keys = list(next(v for v in res.values() if 'm' in v)['m'].keys())
    paired = {}
    for k in keys:
        d = []
        for p in range(1, a.pairs + 1):
            r5, r1 = res[f'{a.batch}_p{p}_H5'], res[f'{a.batch}_p{p}_H1']
            v5 = (r5.get('m') or {}).get(k)
            v1 = (r1.get('m') or {}).get(k)
            d.append(None if v5 is None or v1 is None else round(v1 - v5, 6))
        h5v = [(res[f'{a.batch}_p{p}_H5'].get('m') or {}).get(k) for p in range(1, a.pairs + 1)]
        h1v = [(res[f'{a.batch}_p{p}_H1'].get('m') or {}).get(k) for p in range(1, a.pairs + 1)]
        dd = [x for x in d if x is not None]
        paired[k] = {'H5': h5v, 'H1': h1v, 'diff_H1_minus_H5': d,
                     'diff_mean': round(float(np.mean(dd)), 6) if dd else None,
                     'diff_range': [min(dd), max(dd)] if dd else None,
                     'same_sign_all_pairs': (len(dd) == a.pairs and
                                             (all(x > 0 for x in dd) or all(x < 0 for x in dd)))}
    out = {'batch': a.batch, 'runs': res, 'config_check': cfg, 'paired': paired,
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
