#!/usr/bin/env python3
"""模型重播的附帶核對（未封存、描述性）：HOLD＋50 步起以同一模型重播至 OPEN 的殘差；OPEN 主窗內求解輸入 d̂ 的首末值。

    python3 evaluation/door_k3a_model_replay_check.py <run_dir> [...]   # 只印出，不寫檔
"""
import json, sys
import numpy as np
sys.path.insert(0, 'evaluation')
import door_k3a_model_replay as MR, door_k2_check as KC
from door_k3a_diag2 import rotvec
alpha = json.load(open(MR.IDENT))['alpha']
for run in sys.argv[1:]:
    t = json.load(open(f'{run}/task.json')); r = json.load(open(f'{run}/room_run.json')); a = json.load(open(f'{run}/align_solver.json'))
    I = {c: i for i, c in enumerate(r['steps_cols'])}; room = {int(s[I['step']]): s for s in r['steps']}
    ev = {int(x['step']): x for x in map(json.loads, open(f'{run}/evidence.jsonl'))}
    tr = t['trace']; s_hold = int(next(e['step'] for e in tr if e['phase'] == 'HOLD')); s_open = int(next(e['step'] for e in tr if e['phase'] == 'OPEN'))
    k0 = s_hold + 50
    d0, _ = MR.d0_at(a['log'], room[k0][I['sim_t']])
    qh, stop = MR.replay(room[k0][I['q_arm_meas']], lambda k: (ev[k]['arm_sp'], None), d0, alpha, k0, s_open)
    G = np.array(t['G_T_H']); s = room[s_open]; H = KC._H_of(s[I['door_q_rad']])
    Rq = (np.linalg.inv(H) @ KC._tcp(s[I['base_xyth']], list(qh[s_open])) @ G)[:3, :3]
    Rm = (np.linalg.inv(H) @ KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']]) @ G)[:3, :3]
    jr = (np.array(s[I['q_arm_meas']]) - qh[s_open]) * 1e3
    print(run.split('/')[-1], f'HOLD 重播 {s_open-k0} 步（{k0}→{s_open}）：D_MQ', np.round(np.degrees(rotvec(Rm @ Rq.T)), 3), 'joint_res mrad', np.round(jr, 3))
    # d̂ 在 OPEN 主窗內的變化
    s_low = next(k for k in sorted(room) if k >= s_open and min(((room[k][I['leaf_contacts']] or {}).get(f, {}) or {}).get('bar', 0.0) for f in KC.FINGERS) < 0.5)
    t_open, t_low = room[s_open][I['sim_t']], room[s_low][I['sim_t']]
    D = [(x['solve_in']['snap_sim_t_exact'], x['solve_in'].get('offset_d_hat')) for x in a['log'] if x.get('solve_in')]
    D = [(ts, d) for ts, d in D if d is not None and t_open - 1e-9 <= ts < t_low]
    print('   d̂ 首→末（OPEN 主窗內求解輸入，mrad）', np.round(np.array(D[0][1]) * 1e3, 3), '→', np.round(np.array(D[-1][1]) * 1e3, 3), f'n={len(D)}')
