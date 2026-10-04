#!/usr/bin/env python3
"""H5 紀錄完整性與重播驗收（H1／H5 比較的前置關卡）。

只用趟次自己的紀錄：原本的 N、原紀錄的暖啟動、當輪的目標／協同／偏差。
缺欄位的輪次列為 missing，**不猜值**。開迴路逐輪重解，不推進物理。

驗收門檻（事前固定，見 results/horizon_ablation/experiment_spec_C0.yaml）：
  A1  首筆命令差（整形前，|Δu0| / vmax，逐軸取最大）p95 ≤ 1e-3、max ≤ 1e-2
  A2  SQP 停止理由一致率 ≥ 95%
  A3  原解與重播解的原單位限制殘差都 ≤ r_tol
另報：完整序列差、接受／拒絕（ok）一致、缺欄輪次、計算時效（記錄值，非重播）。

    python3 evaluation/horizon_replay_check.py runs/<RUN> [--out x.json] [--max-cycles K]

離開碼：0 = 通過；1 = 不通過；2 = 紀錄不完整到無法驗收。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import (             # noqa: E402
    WholeBodyKinematics)
import horizon_replay as HR                                      # noqa: E402

THRESH = {'A1_p95': 1e-3, 'A1_max': 1e-2, 'A2_stop_agree': 0.95}
FIELDS = ('q_pred', 's_pred', 'u_prev', 'T_cyc', 'U_warm', 'U_sol',
          'arm_bias', 'solver_N')


def pct(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return None
    return {'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95)),
            'max': float(v.max()), 'n': int(len(v))}


def check(run_dir, max_cycles=None):
    sol, ident = HR.load_run(run_dir)
    args = sol['args']
    K = WholeBodyKinematics.from_urdf_file(args['urdf'])
    cfg0 = HR.base_cfg(args, ident)
    vmax = cfg0.vmax()
    r_tol = cfg0.r_tol
    need_coord = bool(args.get('coord_topic'))
    recs = [L for L in sol['log'] if 'solve_in' in L]
    if max_cycles:
        recs = recs[:max_cycles]
    missing = collections.Counter()
    bad_N = 0
    usable = []
    for L in recs:
        si = L['solve_in']
        miss = [f for f in FIELDS if f not in si]
        if need_coord and 'coord' not in L:
            miss.append('coord')
        for f in miss:
            missing[f] += 1
        if miss:
            continue
        if int(si['solver_N']) != int(args['N']):
            bad_N += 1
            continue
        usable.append(L)
    out = {'run': os.path.basename(run_dir.rstrip('/')), 'N_args': args['N'],
           'n_solve_cycles': len(recs), 'n_usable': len(usable),
           'missing_fields': dict(missing), 'n_solver_N_mismatch': bad_N,
           'thresholds': THRESH, 'r_tol': r_tol}
    if not usable:
        out['verdict'] = 'INCOMPLETE'
        return out, 2
    du0, dU, stop_same, ok_same = [], [], [], []
    res_orig, res_rep = [], []
    stop_pairs = collections.Counter()
    for L in usable:
        si = L['solve_in']
        rr, info = HR.replay_cycle(K, args, ident, L)
        ok_o = bool(L.get('ok'))
        ok_same.append(rr.ok == ok_o)
        stop_same.append(rr.sqp_stop_reason == L.get('sqp_stop'))
        stop_pairs[(L.get('sqp_stop'), rr.sqp_stop_reason)] += 1
        if L.get('residual') is not None:
            res_orig.append(float(L['residual']))
        if rr.max_residual is not None:
            res_rep.append(float(rr.max_residual))
        if ok_o and rr.ok and si['U_sol'] is not None:
            Uo = np.asarray(si['U_sol'], float)
            du0.append(float(np.max(np.abs(rr.u0 - Uo[0]) / vmax)))
            if rr.U.shape == Uo.shape:
                dU.append(float(np.max(np.abs(rr.U - Uo) / np.tile(vmax, (len(Uo), 1)))))
    a1 = pct(du0)
    a2 = float(np.mean(stop_same)) if stop_same else None
    a3_o = (max(res_orig) if res_orig else None)
    a3_r = (max(res_rep) if res_rep else None)
    passed = {
        'A1': (a1 is not None and a1['p95'] <= THRESH['A1_p95']
               and a1['max'] <= THRESH['A1_max']),
        'A2': (a2 is not None and a2 >= THRESH['A2_stop_agree']),
        'A3': (a3_o is not None and a3_r is not None
               and a3_o <= r_tol and a3_r <= r_tol),
    }
    tm = [L.get('timing_ms', {}).get('total') for L in recs]
    cw = [L.get('cycle_wall_ms') for L in recs]
    out.update({
        'A1_du0_over_vmax': a1, 'A2_stop_agree': a2,
        'A3_residual_max': {'original': a3_o, 'replay': a3_r},
        'full_sequence_diff_over_vmax': pct(dU),
        'ok_agree': float(np.mean(ok_same)) if ok_same else None,
        'stop_reason_pairs': {f'{a}->{b}': n for (a, b), n in stop_pairs.items()},
        'timing_recorded_ms': {'core_total': pct([x for x in tm if x is not None]),
                               'cycle_wall': pct([x for x in cw if x is not None]),
                               'note': '紀錄值；紀錄開銷對即時性的影響要對照未加紀錄的趟次'},
        'passed': passed,
        'verdict': 'PASS' if all(passed.values()) and not missing and not bad_N
                   else 'FAIL',
    })
    return out, 0 if out['verdict'] == 'PASS' else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--max-cycles', type=int, default=None)
    a = ap.parse_args()
    out, rc = check(a.run_dir, a.max_cycles)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if a.out:
        json.dump(out, open(a.out, 'w'), ensure_ascii=False, indent=1,
                  default=str)
    return rc


if __name__ == '__main__':
    sys.exit(main())
