#!/usr/bin/env python3
"""偏移估計準入的**開迴路重放比較**（只讀；不代表閉迴路行為）。

以實錄求解週期的同步輸入（snap_sim_t_exact、s_meas、q_meas[3:]）餵 offset_admission.OffsetEstimator：
  legacy：須逐週期重現實錄 d̂（驗證重放正確）；consistent：未准入即保持。
手臂實際軌跡不會因估計不同而改變（開迴路），因此只能比較「同一組輸入下估計量的差」，**不得稱閉迴路夾持改善**。

    python3 evaluation/door_k3a_offset_replay.py <run_dir>   # 寫 <run_dir>/analysis/door_k3a_offset_replay.json（已存在則拒絕覆寫）
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offset_admission as OA                         # noqa: E402


def main():
    run = sys.argv[1]
    out = os.path.join(run, 'analysis', 'door_k3a_offset_replay.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    a = json.load(open(os.path.join(run, 'align_solver.json')))
    t = json.load(open(os.path.join(run, 'task.json')))
    ident = json.load(open(a['args'].get('arm_ident') or os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
    A = a['args']
    t_open = next(e['t'] for e in t['policy_events'] if e.get('to') == 'OPEN')
    mk = lambda adm: OA.OffsetEstimator(ident['alpha'], ident['phys_dt_measured_s'], 1.0 / A['rate'], A['offset_tau_s'], A['offset_gate_rad'],
                                        A['offset_max_rad'], bool(A.get('offset_init_static')), bool(A.get('offset_moving')), adm)
    E = {'legacy': mk('legacy'), 'consistent': mk('consistent')}
    rows, worst_legacy = [], 0.0
    for x in a['log']:
        si = x.get('solve_in')
        if not si:
            continue
        rec = {k: e.update(si['snap_sim_t_exact'], si['s_meas'], si['q_meas'][3:]) for k, e in E.items()}
        worst_legacy = max(worst_legacy, float(np.abs(np.array(rec['legacy']['d_hat_after']) - np.array(si['offset_d_hat'])).max()))
        rows.append({'t': x['sim_t'], 'legacy_used': rec['legacy']['used'], 'consistent_used': rec['consistent']['used'],
                     'consistent_why': rec['consistent']['why'], 'd_legacy': rec['legacy']['d_hat_after'], 'd_consistent': rec['consistent']['d_hat_after']})
    diff = [float(np.abs(np.array(r['d_legacy']) - np.array(r['d_consistent'])).max()) for r in rows]
    after = [r for r in rows if r['t'] >= t_open]
    res = {'run': os.path.basename(os.path.normpath(run)), 'open_loop_only': True,
           'legacy_reproduces_log_max_abs': worst_legacy, 'n_cycles': len(rows),
           'n_static_bypass_after_open_legacy': sum(1 for r in after if r['legacy_used'] == 'static' and r['consistent_used'] == 'held'),
           'counts_after_open': {k: {u: sum(1 for r in after if r[f'{k}_used'] == u) for u in sorted({r[f'{k}_used'] for r in after})}
                                 for k in ('legacy', 'consistent')},
           'max_abs_d_diff_rad': max(diff), 'max_abs_d_diff_t': rows[int(np.argmax(diff))]['t'],
           'd_end': {'legacy': rows[-1]['d_legacy'], 'consistent': rows[-1]['d_consistent']},
           'note': '開迴路重放：輸入取自實錄，手臂軌跡不隨估計改變；只比較估計量，不得稱閉迴路夾持改善。',
           'rows': rows}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
