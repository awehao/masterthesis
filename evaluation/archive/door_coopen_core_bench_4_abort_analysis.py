#!/usr/bin/env python3
"""door_coopen_core_bench_4 中止與鏈理由分類（只讀；Codex r28：中止 ⇒ 保留失敗，不補跑、不追加參數）。
1. 中止：task policy_events 的中止事件（monitor_stale_or_missing：state_age／contacts_age 以**牆鐘**計，> stale_s 0.2 s）。
2. 執行端停頓證據（兩趟同法）：task.json trace 的 state_age_wall > 0.09 s 列；task_trace.jsonl 相鄰輪次看到的 sim_t 停住的輪數與牆鐘跨度；
   room_run 物理步 sim_t 連續性（有無模擬時間缺口）；room_run evidence 寫入最大耗時。
3. 鏈理由分類（判定器 r12 結果；以時間判斷是否在中止前）：
   - 漂移：analysis2 同法 E＝H_meas⁻¹·(T_tcp·G_T_H)，中止前最大、中止後首次 > 5 mm；
   - geometry_evidence_missing：OPEN 列 state_age_wall＋compute_s > 0.2 s 的時刻；
   - state_step 重複：task_trace 相鄰列同 state_step 的時刻與相位；
   - 手指—橫桿在非允許相位：中止後（ABORTED）首筆時刻；
   - door_angle_pre_open_over_1deg：OPEN 前實測最大門角（判定器窗口含中止後觀察期，與 bench_3 同類）。
限制：停頓原因（例如執行端行程的垃圾回收或其他一次性工作）**未定位**；本腳本只報時序證據，不判定原因。

    python3 evaluation/door_coopen_core_bench_4_abort_analysis.py   # 寫 runs/door_coopen_core_bench_4/analysis/door_coopen_core_bench_4_abort_analysis.json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                                   # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_4')
REF = os.path.join(HERE, 'runs', 'door_coopen_core_bench_3')
OUT = os.path.join(RUN, 'analysis', 'door_coopen_core_bench_4_abort_analysis.json')


def stall(run):
    t = json.load(open(os.path.join(run, 'task.json')))
    r = json.load(open(os.path.join(run, 'room_run.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    ts = [s[c['sim_t']] for s in r['steps']]
    gaps = [a for a, b in zip(ts, ts[1:]) if b - a > 0.0105]
    big = [(e['t'], e['state_age_wall']) for e in t['trace'] if isinstance(e.get('state_age_wall'), (int, float)) and e['state_age_wall'] > 0.09]
    rows = [json.loads(ln) for ln in open(os.path.join(run, 'task_trace.jsonl'))]
    runs_, i = [], 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and rows[j + 1]['row'].get('t') == rows[i]['row'].get('t'):
            j += 1
        if j > i and rows[i]['row'].get('t', 0) > 1.0:
            runs_.append({'sim_t': rows[i]['row']['t'], 'n_loops_same_t': j - i + 1, 'wall_span_s': rows[j]['enq_wall_mono'] - rows[i]['enq_wall_mono'],
                          'wall_since_trace_start_s': rows[i]['enq_wall_mono'] - rows[0]['enq_wall_mono']})
        i = j + 1
    return {'state_age_wall_gt_0p09': big, 'task_loops_seeing_same_sim_t': runs_, 'room_sim_t_gaps': gaps,
            'evidence_write_s_max': (r.get('evidence') or {}).get('write_s_max'), 'stale_s': (t.get('config') or {}).get('stale_s')}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t = json.load(open(os.path.join(RUN, 'task.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    j = json.load(open(os.path.join(RUN, 'analysis', 'door_coopen_check_core12.json')))
    c = {k: i for i, k in enumerate(r['steps_cols'])}
    ab = next(e for e in t['policy_events'] if e.get('abort'))
    ta = float(ab['t'])
    t_open = next(e['t'] for e in t['policy_events'] if e.get('to') == 'OPEN')
    GTH = np.asarray(t['G_T_H'], float)
    evd = {}
    for ln in open(os.path.join(RUN, 'evidence.jsonl')):
        e = json.loads(ln)
        if isinstance(e.get('step'), int) and isinstance(e.get('T_WH'), list) and len(e['T_WH']) == 16:
            evd[e['step']] = np.asarray(e['T_WH'], float).reshape(4, 4)
    pre, post = [], []
    for s in r['steps']:
        if s[c['step']] not in evd or s[c['sim_t']] < t_open:
            continue
        E = np.linalg.inv(evd[s[c['step']]]) @ (KC._tcp(list(s[c['base_xyth']]), list(s[c['q_arm_meas']])) @ GTH)
        (pre if s[c['sim_t']] < ta else post).append((s[c['sim_t']], float(np.linalg.norm(E[:3, 3])) * 1e3))
    geom = [(e['t'], e['state_age_wall'], (e.get('k3m_geom') or {}).get('compute_s')) for e in t['trace'] if e.get('phase') == 'OPEN'
            and isinstance(e.get('state_age_wall'), (int, float)) and isinstance((e.get('k3m_geom') or {}).get('compute_s'), (int, float))
            and e['state_age_wall'] + e['k3m_geom']['compute_s'] > 0.2]
    tr = [json.loads(ln)['row'] for ln in open(os.path.join(RUN, 'task_trace.jsonl'))]
    rep = [(tr[k]['t'], tr[k].get('phase'), tr[k].get('state_step')) for k in range(1, len(tr))
           if tr[k].get('state_step') is not None and tr[k].get('state_step') == tr[k - 1].get('state_step')]
    fb = next(((s[c['sim_t']], max(s[c['finger_contact_n']])) for s in r['steps'] if isinstance(s[c['finger_contact_n']], list)
               and s[c['sim_t']] >= ta and max(s[c['finger_contact_n']]) > 0.1), None)
    q_pre = max(math.degrees(s[c['door_q_rad']]) for s in r['steps'] if s[c['sim_t']] < t_open)
    out = {'schema': 'door_coopen_core_bench_4_abort_analysis/1', 'abort': ab, 'chain_why': j['why'],
           'stall': {'bench_4': stall(RUN), 'bench_3': stall(REF)},
           'classify': {
               'grasp_drift_over_limit': {'pre_abort_max_mm': max(pre, key=lambda x: x[1]), 'post_abort_first_gt5mm': next((x for x in post if x[1] > 5.0), None),
                                          'post_abort_max_mm': max(post, key=lambda x: x[1]) if post else None, 'class': 'post_abort'},
               'geometry_evidence_missing': {'rows': geom, 'class': 'stall（與中止同一停頓）'},
               'state_step_repeat': {'rows': rep, 'class': 'frozen_period（BENCH_START）＋stall（96.36 s）'},
               'finger_bar_outside_allowed_phase': {'first_after_abort': fb, 'class': 'post_abort'},
               'door_angle_pre_open_over_1deg': {'q_pre_open_max_deg': q_pre, 'class': 'post_abort_phase_classification（OPEN 前實測未超過 1°）'},
               'termination': {'class': 'consequence（aborted:monitor_stale_or_missing、termination:ABORTED、end_hold／settle／pause／reference／trace 等收尾項）'}},
           'note': '停頓原因未定位；只報時序證據'}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(out['stall'], ensure_ascii=False, default=float)[:1500])
    print(json.dumps(out['classify'], ensure_ascii=False, default=float)[:1800])


if __name__ == '__main__':
    main()
