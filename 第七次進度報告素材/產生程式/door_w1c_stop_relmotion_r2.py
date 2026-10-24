#!/usr/bin/env python3
"""door_w1c_stop_relmotion_analysis.py 的 r2 補算（只讀；Codex 20261025_023608 更正）：
r1 每 5 步取樣，表格終點實為 61.97 s；本檔以**全部物理步**補：
  (1) 終點分解改用中止前最後一個物理步（同法、兩種順序、閉合殘差）；
  (2) 手臂實測關節變化的全程峰值（軸、時刻）與終點值分開報；設定點與套用命令全程核對。
輸出：results/door/w1c_30_1_stop_relmotion_r2.json（r1 保留）。

    python3 evaluation/door_w1c_stop_relmotion_r2.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_w1c_stop_relmotion_analysis as A           # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'w1c_30_1_stop_relmotion_r2.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    r = json.load(open(os.path.join(A.RUN, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    s = json.load(open(os.path.join(A.RUN, 'align_solver.json')))
    t = json.load(open(os.path.join(A.RUN, 'task.json')))
    t0 = s['stats']['k3m_pause']['supervisor_outcome']['transitions']['STOPPED']
    ta = next(e['t'] for e in t['policy_events'] if e.get('abort'))
    ev = {}
    for line in open(os.path.join(A.RUN, 'evidence.jsonl')):
        d = json.loads(line)
        if t0 - 0.05 <= d['t'] <= ta + 0.05:
            ev[d['step']] = d
    st = [x for x in r['steps'] if t0 - 1e-9 <= x[I['sim_t']] <= ta + 1e-9]
    missing = [int(x[I['step']]) for x in st if int(x[I['step']]) not in ev]
    x0, x1 = st[0], st[-1]
    e0, e1 = ev[int(x0[I['step']])], ev[int(x1[I['step']])]
    H0, H1 = (np.array(e['T_WH'], float).reshape(4, 4) for e in (e0, e1))
    tot, Ao, Bo, split = A.decompose(H0, x0[I['base_xyth']], x0[I['q_arm_meas']], H1, x1[I['base_xyth']], x1[I['q_arm_meas']])
    q0 = np.array(x0[I['q_arm_meas']])
    dq = np.array([np.array(x[I['q_arm_meas']]) - q0 for x in st]) * 1e3
    k, j = np.unravel_index(np.abs(dq).argmax(), dq.shape)
    sp0 = np.array(e0['arm_sp'])
    sp_max = max(float(np.abs(np.array(ev[int(x[I['step']])]['arm_sp']) - sp0).max()) for x in st if int(x[I['step']]) in ev) * 1e3
    res = {'schema': 'w1c_stop_relmotion/2', 't_start': float(x0[I['sim_t']]), 't_end_last_step': float(x1[I['sim_t']]), 't_abort': ta,
           'n_steps': len(st), 'n_missing_evidence': len(missing),
           'endpoint_total_mm': (tot * 1e3).tolist(), 'endpoint_order_A_mm': {k_: (v * 1e3).tolist() for k_, v in Ao.items()},
           'endpoint_order_B_mm': {k_: (v * 1e3).tolist() for k_, v in Bo.items()},
           'endpoint_dH_split_mm': {k_: (v * 1e3).tolist() for k_, v in split.items()},
           'closure_mm': float(max(np.abs(tot - sum(Ao.values())).max(), np.abs(tot - sum(Bo.values())).max()) * 1e3),
           'arm_meas_change_peak_mrad': {'value': float(dq[k, j]), 'joint': f'j{j + 1}', 't': float(st[k][I['sim_t']])},
           'arm_meas_change_endpoint_mrad': dq[-1].tolist(),
           'arm_sp_change_max_mrad_all_steps': sp_max,
           'applied_arm_absmax_all_steps': max(float(np.abs(x[I['applied_arm']]).max()) for x in st),
           'sampling_note': 'r1 表格為每 5 步取樣、終點 61.97 s；本檔終點＝中止前最後物理步'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
