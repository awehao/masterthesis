#!/usr/bin/env python3
"""door_coopen_core_1 中止診斷（只讀；不改任何紀錄）：APPROACH 中 finger2 觸橫桿（finger_bar_outside_allowed_phase）。
1. 逐步：實測 TCP 相對抓取位姿（把手座標）沿接近軸 a_H 的剩餘距離、垂直分量；兩指對橫桿距離（door_k3b_pair_geom 同一算法）；兩指接觸力。
2. 目標 vs 實測：任務 trace 的 APPROACH 目標相對抓取位姿的垂直分量（應為 0）與實測差；沿開合方向（T_HG y 軸）的分量。
3. 對照：正面夾法實跑 door_wide_w1f_60_3、door_wide_w2c_30_1 的同一指標。
4. 各關節設定點（取證 arm_sp）− 實測（q_arm_meas）在 APPROACH 期間的最大值與末值。
**界線**：只描述；關節 2 落差與 TCP 側向偏移的對應是推論（重力下位置驅動穩態下垂、設定點模型偏差在正面類構型辨識），未證明為唯一原因。

    python3 evaluation/door_coopen_core_1_diag.py   # 寫 runs/door_coopen_core_1/analysis/door_coopen_core_1_diag.json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402
import door_k3b_pair_geom as PG                        # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_1')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_1_diag.json')
GB = os.path.join(HERE, 'results', 'door', 'coopen_core_grasp_basis_r4.json')


def approach_stats(run, T_HG, a_H):
    r = json.load(open(os.path.join(run, 'room_run.json')))
    t = json.load(open(os.path.join(run, 'task.json')))
    cols = {c: i for i, c in enumerate(r['steps_cols'])}
    byst = {s[cols['step']]: s for s in r['steps']}
    out = []
    for row in t['trace']:
        if row.get('phase') != 'APPROACH' or row.get('target') is None:
            continue
        s = byst.get(row.get('state_step'))
        if s is None:
            continue
        H = KC._H_of(s[cols['door_q_rad']])
        Tm = KC._tcp(list(s[cols['base_xyth']]), list(s[cols['q_arm_meas']]))
        Tt = np.array(row['target']).reshape(4, 4)
        e = (np.linalg.inv(H) @ Tm)[:3, 3] - (np.linalg.inv(H) @ Tt)[:3, 3]
        et = (np.linalg.inv(H) @ Tt)[:3, 3] - T_HG[:3, 3]
        perp = e - (e @ a_H) * a_H
        out.append({'t': row['t'], 'target_s_m': float(et @ a_H), 'target_perp_mm': float(np.linalg.norm(et - (et @ a_H) * a_H) * 1e3),
                    'lag_along_mm': float(e @ a_H * 1e3), 'perp_mm': float(np.linalg.norm(perp) * 1e3), 'perp_close_dir_mm': float(perp @ T_HG[:3, 1] * 1e3),
                    'perp_handle_mm': (perp * 1e3).tolist()})
    ev = [e for e in t['policy_events'] if e.get('to') == 'APPROACH']
    t0 = ev[0]['t']
    t1 = next((e['t'] for e in t['policy_events'] if e.get('from') == 'APPROACH' or 'abort' in e), t0 + 6)
    d = []
    for ln in open(os.path.join(run, 'evidence.jsonl')):
        e = json.loads(ln)
        if t0 <= e.get('t', -1) <= t1 and isinstance(e.get('arm_sp'), list) and e['step'] in byst:
            d.append(np.array(e['arm_sp']) - np.array(byst[e['step']][cols['q_arm_meas']]))
    d = np.array(d)
    return {'n': len(out), 'lag_along_max_mm': max(o['lag_along_mm'] for o in out), 'perp_max_mm': max(o['perp_mm'] for o in out),
            'perp_close_dir_absmax_mm': max(abs(o['perp_close_dir_mm']) for o in out), 'perp_end_mm': out[-1]['perp_mm'],
            'target_perp_max_mm': max(o['target_perp_mm'] for o in out),
            'joint_sp_minus_meas_end_rad': d[-1].tolist(), 'joint_sp_minus_meas_absmax_rad': np.abs(d).max(axis=0).tolist(), 'rows': out}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    gb = json.load(open(GB))['args']
    T_HG, a_H = np.array(gb['T_HG_full']), np.array(gb['a_H'])
    res = {'schema': 'door_coopen_core_1_diag/1', 'core_1': approach_stats(RUN, T_HG, a_H), 'front_runs': {}}
    for rn in ('door_wide_w1f_60_3', 'door_wide_w2c_30_1'):
        run = os.path.join(HERE, 'runs', rn)
        t = json.load(open(os.path.join(run, 'task.json')))
        res['front_runs'][rn] = approach_stats(run, np.array(t['grasp_basis']['T_HG']), np.array([0.0, -1.0, 0.0]))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    cols = {c: i for i, c in enumerate(r['steps_cols'])}
    pair = PG.default_pair_backend()
    contact = []
    for s in r['steps']:
        if 14.5 <= s[cols['sim_t']] <= 15.05:
            qf = s[cols['q_finger']]
            dd = pair(list(s[cols['base_xyth']]) + list(s[cols['q_arm_meas']]), s[cols['door_q_rad']], qf if isinstance(qf, list) else [qf, qf])
            contact.append({'t': s[cols['sim_t']], 'pair_mm': [x * 1e3 for x in dd], 'finger_contact_n': s[cols['finger_contact_n']]})
    res['contact_window'] = contact
    res['interpretation'] = ('任務端接近目標在接近軸上（垂直分量 0）；實測 TCP 側向偏移隨接近增大（終點 8.8 mm，沿開合方向 8.2 mm），正面夾法兩趟則收斂到 1.4 mm；'
                             '關節 2 設定點−實測末值 +0.035 rad（正面 −0.020 rad）。推論：本構型（j2 −1.67、手臂後仰）重力下位置驅動穩態下垂，'
                             '設定點模型偏差在正面類構型辨識；未證明為唯一原因。')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    c = res['core_1']
    print(f"core_1：側向 max {c['perp_max_mm']:.1f}、終點 {c['perp_end_mm']:.1f} mm、目標垂直 max {c['target_perp_max_mm']:.3f} mm、j2 落差末值 {c['joint_sp_minus_meas_end_rad'][1]:+.4f} rad")
    for k, v in res['front_runs'].items():
        print(f"{k}：側向 max {v['perp_max_mm']:.1f}、終點 {v['perp_end_mm']:.1f} mm、j2 落差末值 {v['joint_sp_minus_meas_end_rad'][1]:+.4f} rad")


if __name__ == '__main__':
    main()
