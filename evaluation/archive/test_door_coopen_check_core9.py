#!/usr/bin/env python3
"""判定器 r9 的 J7（台架啟動）測試（**合成資料**；登錄取實檔 r9）。
  Q1 正常 ⇒ J7 無理由。
  Q2 每個理由各一反例：登錄 r8（無 bench）、task 設定不符、首相位非 BENCH_START、出現 UNFOLD、缺 INIT_CHECK、寫入設定點≠s_init、種子 source 錯、
     stow_q≠q_nominal、t0 實測偏 1e-4、未釋放、釋放 seq≠宣告、求解端宣告 sim_t≠t0、kind 非 solution、d0≠d_init、n_upd_at_init≠0、預熱已發布、
     錨點缺、無 bench hold_start 事件、room 缺 bench。
  Q3 窗口別名：bench_window() 期間 phase_windows 有 UNFOLD＝BENCH_START；結束後還原為原函式（無 UNFOLD）。
  Q4 豁免清單只有 park_anchor_missing；iso 替換清單只有 iso_phase_sequence、iso_setpoint_mismatch。

    python3 evaluation/test_door_coopen_check_core9.py
"""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core9 as J9                   # noqa: E402
import door_k2_check as KC2                            # noqa: E402

PRM9 = os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r9.json')
REG = json.load(open(PRM9))['bench']


def mk():
    task = {'coopen': {'params_path': os.path.relpath(PRM9, WS)}, 'bench': {'config': copy.deepcopy(REG), 'stats': {'monitor_skipped_cycles': 30}},
            'policy_events': [{'t': 0.0, 'to': 'BENCH_START'}, {'t': 0.3, 'to': 'INIT_CHECK'}, {'t': 1.0, 'to': 'ENGAGE'}, {'t': 3.0, 'to': 'HOLD'}]}
    room = {'config': {'stow_q': [round(v, 9) for v in REG['q_nominal']]},
            'bench': {'q_nominal': REG['q_nominal'], 's_init_written': REG['s_init'], 'd_init': REG['d_init'], 'q_arm_meas_t0': REG['q_nominal'],
                      'anchor': [1.02, -0.5, 3.03], 't0': 0.0, 'chain_seed': {'arm_setpoint': REG['s_init'], 'source': 'bench_start'},
                      'release': {'state': 'released', 'announced': {'source_seq': 3, 'sim_t': 0.0}, 'released': {'source_seq': 3}}},
            'events': [{'park': 'hold_start', 'bench': True}]}
    solver = {'bench': {'announced': {'source_seq': 3, 'sim_t': 0.0, 'kind': 'solution'}, 'd0_set': REG['d_init'], 'n_upd_at_init': 0,
                        'warmup': {'published': False}}}
    return task, room, solver


def run(t, r, s):
    why = []
    J9.check_bench(t, r, s, why)
    return why


class Q(unittest.TestCase):
    def test_q1_nominal(self):
        self.assertEqual(run(*mk()), [])

    def test_q2_reasons(self):
        cases = []

        def mod(f, want):
            t, r, s = mk()
            f(t, r, s)
            cases.append((want, t, r, s))
        mod(lambda t, r, s: t['coopen'].update(params_path=os.path.relpath(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r8.json'), WS)), 'bench_reg_invalid')
        mod(lambda t, r, s: t['bench']['config'].update(release_timeout_wall_s=99.0), 'bench_reg_invalid')
        mod(lambda t, r, s: t['policy_events'].insert(0, {'t': -1.0, 'to': 'WAIT_HANDOVER'}), 'bench_phase_sequence')
        mod(lambda t, r, s: t['policy_events'].insert(1, {'t': 0.1, 'to': 'UNFOLD'}), 'bench_phase_sequence')
        mod(lambda t, r, s: t.__setitem__('policy_events', [e for e in t['policy_events'] if e['to'] != 'INIT_CHECK']), 'bench_phase_sequence')
        mod(lambda t, r, s: r['bench'].__setitem__('s_init_written', REG['q_nominal']), 'bench_setpoint_written')
        mod(lambda t, r, s: r['bench']['chain_seed'].update(source='nav'), 'bench_setpoint_written')
        mod(lambda t, r, s: r['config'].__setitem__('stow_q', [v + 1e-4 for v in REG['q_nominal']]), 'bench_initial_state')
        mod(lambda t, r, s: r['bench'].__setitem__('q_arm_meas_t0', [REG['q_nominal'][0] + 1e-4] + REG['q_nominal'][1:]), 'bench_initial_state')
        mod(lambda t, r, s: r['bench']['release'].update(state='aborted'), 'bench_release')
        mod(lambda t, r, s: r['bench']['release']['released'].update(source_seq=4), 'bench_release')
        mod(lambda t, r, s: s['bench']['announced'].update(sim_t=0.05), 'bench_release')
        mod(lambda t, r, s: s['bench']['announced'].update(kind='handshake'), 'bench_release')
        mod(lambda t, r, s: s['bench'].__setitem__('d0_set', [0.0] * 6), 'bench_offset_init')
        mod(lambda t, r, s: s['bench'].update(n_upd_at_init=1), 'bench_offset_init')
        mod(lambda t, r, s: s['bench']['warmup'].update(published=True), 'bench_offset_init')
        mod(lambda t, r, s: r['bench'].update(anchor=None), 'bench_anchor')
        mod(lambda t, r, s: r.__setitem__('events', []), 'bench_anchor')
        mod(lambda t, r, s: r.pop('bench'), 'bench_evidence_missing')
        for want, t, r, s in cases:
            self.assertIn(want, run(t, r, s), want)

    def test_q3_window_alias(self):
        task = {'policy_events': [{'t': 0.0, 'to': 'BENCH_START'}, {'t': 0.3, 'to': 'INIT_CHECK'}], 'trace': [{'t': 1.0}]}
        orig = KC2.phase_windows
        with J9.bench_window():
            w = KC2.phase_windows(task)
            self.assertEqual(w['UNFOLD'], w['BENCH_START'])
        self.assertIs(KC2.phase_windows, orig)
        self.assertNotIn('UNFOLD', KC2.phase_windows(task))

    def test_q4_lists(self):
        self.assertEqual(J9.CHAIN_EXEMPT, ('park_anchor_missing',))
        self.assertEqual(J9.ISO_REPLACED, ('iso_phase_sequence', 'iso_setpoint_mismatch'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
