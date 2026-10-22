#!/usr/bin/env python3
"""K3-A 單趟診斷 v3（只讀；依 Codex 20261022 診斷 v2 複核的口徑修正；v2 輸出保留不改）。

與 v2 的差異：
1. 設定點差分報：同時刻追蹤差 s_meas − q_meas[3:]；延遲預測增量 s_pred − s_meas（v2 的 s_pred − q_meas 混用時刻，作廢）。
2. 目標分欄：task trace 目標（≤ 週期時刻的最後一筆已發布）與求解器實際採用的 solve_in.T_cyc，各自對 FK（把手座標），並報兩者差。
3. 事件時刻分來源：物理步（room_run，0.01 s）與求解週期（20 Hz）各自報；先後排序以**物理步**為準，求解週期量另列其取樣時刻。
4. 中止後窗口用物理紀錄補報（求解紀錄止於停止，停止後**沒有新的偏移估計**，不延填）。
門檻只用於排序，不是判準；相關或先後不等於因果。

    python3 evaluation/door_k3a_diag3.py <run_dir>   # 寫 <run_dir>/analysis/door_k3a_diag3.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
from door_k3a_diag2 import rotvec                     # noqa: E402

FINGERS = KC.FINGERS
THR_PHYS = {'f_split_n': 2.0, 'f_low_n': 0.5, 'rot_x_deg': 0.1, 'rot_y_deg': 0.1, 'rot_z_deg': 0.1, 'off_x_mm': 0.5, 'off_z_mm': 0.3,
            'wrist_rise_n': 2.0, 'door_move_deg': 0.01}
THR_SOLV = {'track_gap_change_rad': 0.002, 'pred_incr_rad': 0.002, 'd_hat_change_rad': 0.001, 'lead_deg': 0.2, 'trace_tgt_y_mm': 1.0,
            'tcyc_tgt_y_mm': 1.0}


def phys_rows(st, I, G, lo, hi):
    out = []
    for s in st:
        tt = s[I['sim_t']]
        if tt < lo or tt > hi:
            continue
        q = s[I['door_q_rad']]
        H = KC._H_of(q)
        E = np.linalg.inv(H) @ KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']]) @ G
        lc = s[I['leaf_contacts']] or {}
        out.append({'t': tt, 'q_deg': math.degrees(q), 'qdot': s[I['door_qdot_rad_s']], 'f_bar': [lc.get(f, {}).get('bar', 0.0) for f in FINGERS],
                    'q_finger_mm': [x * 1e3 for x in s[I['q_finger']]], 'wrist_n': s[I['wrist_force_n']],
                    'grasp_off_mm': (E[:3, 3] * 1e3).tolist(), 'grasp_rot_deg': np.degrees(rotvec(E[:3, :3])).tolist()})
    return out


def main():
    run = sys.argv[1]
    out_j = os.path.join(run, 'analysis', 'door_k3a_diag3.json')
    if os.path.exists(out_j):
        raise SystemExit(f'{out_j} 已存在，不覆寫')
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    a = json.load(open(os.path.join(run, 'align_solver.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = [e for e in t['policy_events'] if 'to' in e]
    t_hold = next(e['t'] for e in ev if e['to'] == 'HOLD')
    t_open = next(e['t'] for e in ev if e['to'] == 'OPEN')
    ab = next((e for e in t['policy_events'] if 'abort' in e), None)
    t_ab = ab['t'] if ab else math.inf
    G = np.array(t['G_T_H'])
    st = r['steps']
    t_end_phys = st[-1][I['sim_t']]
    P = phys_rows(st, I, G, t_hold, t_end_phys)
    # ---- 物理步事件（OPEN 起，以 OPEN 第一物理步為基準）
    p0 = next(x for x in P if x['t'] >= t_open)
    first_p = {}
    for x in P:
        if x['t'] < t_open:
            continue
        tests = {'f_split_n': abs(x['f_bar'][0] - x['f_bar'][1]), 'f_low_n': -min(x['f_bar']),
                 'rot_x_deg': abs(x['grasp_rot_deg'][0] - p0['grasp_rot_deg'][0]), 'rot_y_deg': abs(x['grasp_rot_deg'][1] - p0['grasp_rot_deg'][1]),
                 'rot_z_deg': abs(x['grasp_rot_deg'][2] - p0['grasp_rot_deg'][2]), 'off_x_mm': abs(x['grasp_off_mm'][0] - p0['grasp_off_mm'][0]),
                 'off_z_mm': abs(x['grasp_off_mm'][2] - p0['grasp_off_mm'][2]), 'wrist_rise_n': x['wrist_n'] - p0['wrist_n'],
                 'door_move_deg': x['q_deg'] - p0['q_deg']}
        for k, v in tests.items():
            thr = -THR_PHYS[k] if k == 'f_low_n' else THR_PHYS[k]
            if k not in first_p and v > thr:
                first_p[k] = {'t': x['t'], 'q_deg': x['q_deg'], 'value': (-v if k == 'f_low_n' else v)}
    # ---- 求解週期量
    tr = sorted([e for e in t['trace'] if e.get('target') is not None], key=lambda e: e['t'])
    tr_t = np.array([e['t'] for e in tr])
    st_t = np.array([s[I['sim_t']] for s in st])
    S = []
    for L in a['log']:
        si = L.get('solve_in') or {}
        tt = L['sim_t']
        if tt < t_hold - 0.5 or not si:
            continue
        ts = si.get('snap_sim_t_exact', tt)
        k = int(np.searchsorted(st_t, ts + 1e-9, side='right') - 1)
        s_ = st[k]
        q = s_[I['door_q_rad']]
        H = KC._H_of(q)
        T = KC._tcp(s_[I['base_xyth']], s_[I['q_arm_meas']])
        j = int(np.clip(np.searchsorted(tr_t, tt, side='right') - 1, 0, len(tr) - 1))
        Ttr = np.array(tr[j]['target']).reshape(4, 4)
        Tcy = np.array(si['T_cyc']).reshape(4, 4) if si.get('T_cyc') is not None else None
        sm, sp, qm = si.get('s_meas'), si.get('s_pred'), si.get('q_meas')
        om = L.get('offset_moving') or {}
        S.append({'t': tt, 't_snap': ts, 'q_deg': math.degrees(q),
                  'lead_deg': None if tr[j].get('q_ref') is None else math.degrees(tr[j]['q_ref'] - q),
                  'trace_tgt_minus_fk_mm': (H[:3, :3].T @ (Ttr[:3, 3] - T[:3, 3]) * 1e3).tolist(),
                  'tcyc_minus_fk_mm': None if Tcy is None else (H[:3, :3].T @ (Tcy[:3, 3] - T[:3, 3]) * 1e3).tolist(),
                  'tcyc_minus_trace_mm': None if Tcy is None else float(np.linalg.norm(Tcy[:3, 3] - Ttr[:3, 3]) * 1e3),
                  'track_gap_rad': None if (sm is None or qm is None) else (np.array(sm) - np.array(qm[3:9])).tolist(),
                  'pred_incr_rad': None if (sm is None or sp is None) else (np.array(sp) - np.array(sm)).tolist(),
                  'd_hat': om.get('d_hat_after', si.get('offset_d_hat')), 'offset_used': om.get('used'), 'offset_ok': om.get('ok'),
                  'offset_why': om.get('why')})
    s0 = next(x for x in S if x['t'] >= t_open)
    first_s = {}
    for x in S:
        if x['t'] < t_open:
            continue
        tests = {'track_gap_change_rad': float(np.abs(np.array(x['track_gap_rad']) - np.array(s0['track_gap_rad'])).max()),
                 'pred_incr_rad': float(np.abs(x['pred_incr_rad']).max()),
                 'd_hat_change_rad': float(np.abs(np.array(x['d_hat']) - np.array(s0['d_hat'])).max()),
                 'lead_deg': abs(x['lead_deg'] or 0.0), 'trace_tgt_y_mm': abs(x['trace_tgt_minus_fk_mm'][1]),
                 'tcyc_tgt_y_mm': abs(x['tcyc_minus_fk_mm'][1]) if x['tcyc_minus_fk_mm'] else 0.0}
        for k, v in tests.items():
            if k not in first_s and v > THR_SOLV[k]:
                first_s[k] = {'t': x['t'], 'q_deg': x['q_deg'], 'value': v}
    bypass = [{'t': x['t'], 'why': x['offset_why']} for x in S if x['t'] >= t_open and x['offset_used'] == 'static' and x['offset_ok'] is False]
    wins = [('HOLD', t_hold, t_open), ('OPEN→split', t_open, first_p.get('f_split_n', {}).get('t')),
            ('split→first_low', first_p.get('f_split_n', {}).get('t'), first_p.get('f_low_n', {}).get('t')),
            ('first_low→abort', first_p.get('f_low_n', {}).get('t'), t_ab), ('post_abort', t_ab, t_end_phys + 1e-9)]

    def ex(vals):
        vals = [v for v in vals if v is not None]
        return None if not vals else {'start': vals[0], 'end': vals[-1], 'min': min(vals), 'max': max(vals)}

    def win(lo, hi):
        if lo is None or hi is None:
            return None
        wp = [x for x in P if lo <= x['t'] < hi]
        ws = [x for x in S if lo <= x['t'] < hi]
        o = {'phys': None, 'solver': None}
        if wp:
            o['phys'] = {'n': len(wp), 't0': wp[0]['t'], 't1': wp[-1]['t'], 'q_deg': ex([x['q_deg'] for x in wp]),
                         'f1': ex([x['f_bar'][0] for x in wp]), 'f2': ex([x['f_bar'][1] for x in wp]), 'wrist_n': ex([x['wrist_n'] for x in wp]),
                         'finger1_mm': ex([x['q_finger_mm'][0] for x in wp]), 'finger2_mm': ex([x['q_finger_mm'][1] for x in wp])}
            for i, n in enumerate('xyz'):
                o['phys'][f'grasp_off_{n}_mm'] = ex([x['grasp_off_mm'][i] for x in wp])
                o['phys'][f'grasp_rot_{n}_deg'] = ex([x['grasp_rot_deg'][i] for x in wp])
        if ws:
            o['solver'] = {'n': len(ws), 't0': ws[0]['t'], 't1': ws[-1]['t'], 'lead_deg': ex([x['lead_deg'] for x in ws]),
                           'track_gap_start_rad': ws[0]['track_gap_rad'], 'track_gap_end_rad': ws[-1]['track_gap_rad'],
                           'pred_incr_absmax_rad': max(float(np.abs(x['pred_incr_rad']).max()) for x in ws),
                           'tcyc_minus_trace_max_mm': max((x['tcyc_minus_trace_mm'] or 0.0) for x in ws),
                           'd_hat_start': ws[0]['d_hat'], 'd_hat_end': ws[-1]['d_hat'],
                           'offset_used_counts': {}}
            for i, n in enumerate('xyz'):
                o['solver'][f'trace_tgt_minus_fk_{n}_mm'] = ex([x['trace_tgt_minus_fk_mm'][i] for x in ws])
                o['solver'][f'tcyc_minus_fk_{n}_mm'] = ex([x['tcyc_minus_fk_mm'][i] for x in ws if x['tcyc_minus_fk_mm']])
            for x in ws:
                kk = f"{x['offset_ok']}|{x['offset_used']}"
                o['solver']['offset_used_counts'][kk] = o['solver']['offset_used_counts'].get(kk, 0) + 1
        else:
            o['solver'] = {'note': '本窗無求解紀錄（求解器已停止）；停止後沒有新的偏移估計，不延填。'}
        return o
    res = {'run': os.path.basename(os.path.normpath(run)), 'version': 3, 't_hold': t_hold, 't_open': t_open, 'abort': ab,
           't_end_phys': t_end_phys, 'n_phys_post_abort': sum(1 for x in P if x['t'] >= t_ab),
           'first_crossings_phys': dict(sorted(first_p.items(), key=lambda kv: kv[1]['t'])),
           'first_crossings_solver_samples': dict(sorted(first_s.items(), key=lambda kv: kv[1]['t'])),
           'thresholds_for_ordering_only': {'phys': THR_PHYS, 'solver': THR_SOLV},
           'static_bypass_after_open': bypass,
           'windows': {n: win(lo, hi) for n, lo, hi in wins},
           'note': 'track_gap＝s_meas − q_meas（同時刻）；pred_incr＝s_pred − s_meas（延遲預測增量）。trace 目標與 T_cyc 分欄。'
                   '物理步事件與求解週期取樣分列。先後不等於因果。'}
    os.makedirs(os.path.dirname(out_j), exist_ok=True)
    json.dump(res, open(out_j, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ('first_crossings_phys', 'first_crossings_solver_samples', 'static_bypass_after_open', 'n_phys_post_abort',
                                          't_end_phys')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
