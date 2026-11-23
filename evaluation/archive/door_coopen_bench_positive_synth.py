#!/usr/bin/env python3
"""判定器 r11 最外層 judge() 的**合成完整正例**生成器（Codex r22：明標合成、不 mock 子判定器、真正最外層 judge() 須 why＝[]）。
**合成資料**：以真實通過鏈判定的 door_wide_w1f_60_3（door_k3m_check_w1f2 原判通過）為底，轉換為台架流程的形式；不是台架實錄，不得當成果引用。
轉換（全部寫入 --out 目錄，原實錄不改）：
  1. 台架起點 k_s＝第一筆帶求解端來源（applied_src seq ≥ 0）的 room 步；room／evidence 只保留 k_s 起的列（凍結期無物理列，同 ev9r2）。
  2. 合成登錄：w1f_params_60_r2 的鏈常數原值（θ、走廊、包絡、門速…）＋ grasp（合成抓取基準：T_HG_full＝該趟 T_HG、a_H＝(0,−1,0)）、
     task_param_extension.r_max_dps＝該趟 config.r_max_dps、approach_gate（r6 值）、grasp_init（qa＝k_s 實測）、
     bench（q_nominal＝k_s 實測、s_init＝k_s 證據 arm_sp − dt·applied_arm、d_init＝q_nominal − s_init、不確定度沿 r9）。
  3. 任務：policy_events 中 ENGAGE 之前的相位換成 BENCH_START（t_s）、INIT_CHECK（原 APPROACH 起點），加 init_check enter／pass（win_t0＝t_E − 0.55）；
     trace 相位同步改名、確認窗列補 init；task.json 補 coopen／coopen_gate／grasp_init／bench 紀錄。
  4. room_run bench（名目、寫入、種子、錨點、釋放、first_intake、first_step 以 k_s 列實值）；求解端 bench（宣告、d0、預熱、錨點）；
     park_gate.json 的 gate.anchor 設 None（台架流程無閘門錨點；r11 以交叉核對的 bench_anchor 提供鏈視圖）。

    python3 evaluation/door_coopen_bench_positive_synth.py --out <dir>   # 寫合成資料；印出 r11 judge() 的 why
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
SRC = os.path.join(HERE, 'runs', 'door_wide_w1f_60_3')
R9 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r9.json')
R6 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r6.json')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def build(out):
    os.makedirs(out, exist_ok=True)
    ld = (lambda f: json.load(open(os.path.join(SRC, f))))
    task, room, gate, solver, obs = ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json')
    c = {k: i for i, k in enumerate(room['steps_cols'])}
    dt = float(room['config']['physics_dt'])
    ks_row = next(s for s in room['steps'] if isinstance(s[c['applied_src']], list) and s[c['applied_src']][2] >= 0)
    ks, ts, seq = int(ks_row[c['step']]), float(ks_row[c['sim_t']]), int(ks_row[c['applied_src']][2])
    ev_lines = [json.loads(ln) for ln in open(os.path.join(SRC, 'evidence.jsonl'))]
    ev_ks = next(e for e in ev_lines if e.get('step') == ks)
    u = [float(x) for x in ks_row[c['base_cmd_body']]] + [float(x) for x in ks_row[c['applied_arm']]]
    q_nom = [float(x) for x in ks_row[c['q_arm_meas']]]
    sp_ev = [float(x) for x in ev_ks['arm_sp']]
    s_init = [a - dt * r for a, r in zip(sp_ev, u[3:])]
    d_init = [a - b for a, b in zip(q_nom, s_init)]
    anchor = [float(v) for v in gate['gate']['anchor']]
    # ---- 合成抓取基準與登錄 ----
    T_HG = task['grasp_basis']['T_HG']
    gbp = os.path.join(out, 'grasp_basis_synth.json')
    json.dump({'args': {'T_HG_full': T_HG, 'a_H': [0.0, -1.0, 0.0]}, 'note': '合成（w1f_60_3 的 T_HG）'}, open(gbp, 'w'))
    prm = json.load(open(os.path.join(WS, task['w1b']['params_path'])))
    r6, r9 = json.load(open(R6)), json.load(open(R9))
    prm['grasp'] = {'basis': gbp, 'basis_sha256': sha(gbp)}
    prm.setdefault('task_param_extension', {})['r_max_dps'] = float(task['config']['r_max_dps'])
    prm['approach_gate'] = r6['approach_gate']
    prm['grasp_init'] = dict(r9['grasp_init'], qa_grasp=q_nom, qa_grasp_cli=','.join(f'{v:.6f}' for v in q_nom))
    prm['bench'] = dict(r9['bench'], q_nominal=q_nom, s_init=s_init, d_init=d_init, q_nominal_cli=','.join(f'{v:.9f}' for v in q_nom),
                        s_init_cli=','.join(f'{v:.9f}' for v in s_init))
    prm['synthetic'] = '合成登錄（判定器完整正例用；不是實驗登錄）'
    pp = os.path.join(out, 'params_synth.json')
    json.dump(prm, open(pp, 'w'), ensure_ascii=False)
    # ---- 任務 ----
    t2 = copy.deepcopy(task)
    evs = t2['policy_events']
    t_appr = next(e['t'] for e in evs if e.get('to') == 'APPROACH')
    t_eng = next(e['t'] for e in evs if e.get('to') == 'ENGAGE')
    i_eng = next(i for i, e in enumerate(evs) if e.get('to') == 'ENGAGE')
    pre = [e for e in evs[:i_eng] if 'to' not in e]
    w0 = t_eng - 0.55
    t2['policy_events'] = ([{'t': ts, 'from': None, 'to': 'BENCH_START', 'bench': True}] + pre
                           + [{'t': t_appr, 'from': 'BENCH_START', 'to': 'INIT_CHECK', 'bench_released': True},
                              {'t': t_appr, 'init_check': 'enter', 'err_m': 0.001, 'from_bench': True},
                              {'t': t_eng, 'init_check': 'pass', 'err_m': 0.0005, 'rot_rad': 0.001, 'win_t0': w0}] + evs[i_eng:])
    t2['coopen'] = {'params_path': pp, 'params_sha256': sha(pp), 'grasp_basis_has_full': True, 'T_HG': T_HG, 'a_H': [0.0, -1.0, 0.0],
                    'r_max_dps': float(task['config']['r_max_dps'])}
    t2['w1b']['params_path'] = pp
    t2['w1b']['params_sha256'] = sha(pp)
    t2['coopen_gate'] = {'config': dict(r6['approach_gate'])}
    t2['grasp_init'] = {'config': prm['grasp_init']}
    t2['bench'] = {'config': prm['bench'], 'stats': {'monitor_skipped_cycles': 20, 'monitored_cycles': 1}}
    ren = {'WAIT_HANDOVER': 'BENCH_START', 'UNFOLD': 'BENCH_START', 'ALIGN_PRE': 'BENCH_START', 'ALIGN_HOLD': 'BENCH_START', 'APPROACH': 'INIT_CHECK'}

    def conv(r):
        r['phase'] = ren.get(r.get('phase'), r.get('phase'))
        r['init'] = ({'valid': True, 'why': None, 'err_p_m': 0.0005, 'err_r_rad': 0.001, 'good': True, 'win_t0': w0}
                     if w0 - 1e-9 <= float(r['t']) <= t_eng + 1e-9 else None)
        return r
    tp = os.path.join(out, 'task_trace.jsonl')
    tr_in = [json.loads(ln) for ln in open(os.path.join(WS, task['trace_log']['path']) if not os.path.isabs(task['trace_log']['path'])
                                           else task['trace_log']['path'])]
    with open(tp, 'w') as f:
        for d in tr_in:
            d['row'] = conv(d['row'])
            f.write(json.dumps(d) + '\n')
    t2['trace'] = [conv(r) for r in (t2.get('trace') or [])]
    t2['trace_log'] = dict(t2['trace_log'], path=tp)
    json.dump(t2, open(os.path.join(out, 'task.json'), 'w'), ensure_ascii=False, default=float)
    # ---- room／evidence ----
    r2 = copy.deepcopy(room)
    r2['steps'] = [s for s in room['steps'] if s[c['step']] >= ks]
    r2['config']['stow_q'] = q_nom
    r2['config']['start_pose'] = ','.join(str(v) for v in anchor)
    fs_sp = [a + dt * r for a, r in zip(s_init, u[3:])]
    r2['bench'] = {'q_nominal': q_nom, 's_init_written': s_init, 'd_init': d_init, 'q_arm_meas_t0': q_nom, 'anchor': anchor, 't0': ts,
                   'chain_seed': {'arm_setpoint': s_init, 'source': 'bench_start'},
                   'release': {'state': 'released', 'announced': {'source_seq': seq, 'sim_t': ts}, 'released': {'source_seq': seq}},
                   'first_intake': {'ok': True, 'why': None, 'source_seq': seq},
                   'first_step': {'ok': True, 'why': None, 'step': ks, 'sim_t': ts, 'owner': 'wholebody', 'kind': 'wholebody_applied',
                                  'applied_source_seq': seq, 'mode_code': 0, 'u_applied': u, 'arm_setpoint': fs_sp, 'setpoint_dev': 0.0},
                   'synthetic': True}
    r2.setdefault('events', []).append({'sim_t': ts, 'step': ks, 'park': 'hold_start', 'bench': True, 'anchor': anchor})
    json.dump(r2, open(os.path.join(out, 'room_run.json'), 'w'), default=float)
    with open(os.path.join(out, 'evidence.jsonl'), 'w') as f:
        for e in ev_lines:
            if isinstance(e.get('step'), int) and e['step'] >= ks:
                f.write(json.dumps(e) + '\n')
    s2 = copy.deepcopy(solver)
    s2['bench'] = {'announced': {'run_id': 'synthetic', 'source_seq': seq, 'sim_t': ts, 'kind': 'solution'}, 'd0_set': d_init, 'n_upd_at_init': 0,
                   'warmup': {'published': False}, 'anchor': anchor}
    json.dump(s2, open(os.path.join(out, 'align_solver.json'), 'w'), default=float)
    g2 = copy.deepcopy(gate)
    g2['gate']['anchor'] = None
    json.dump(g2, open(os.path.join(out, 'park_gate.json'), 'w'), default=float)
    json.dump(obs, open(os.path.join(out, 'lw_observer.json'), 'w'), default=float)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = build(a.out)
    import door_coopen_check_core11 as J11
    ld = (lambda f: json.load(open(os.path.join(out, f))))
    res = J11.judge(ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                    os.path.join(out, 'evidence.jsonl'), os.path.join(out, 'task_trace.jsonl'))
    print(json.dumps({'why': res['why'], 'chain_exempt': res['chain_exempt'], 'iso_replaced': res['iso_replaced'],
                      'anchor': res['bench'].get('anchor_crosscheck', {}).get('consistent')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
