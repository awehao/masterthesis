#!/usr/bin/env python3
"""door_k3m_check_p.check_pause 反例（合成紀錄；K3-M 部分沿用 door_k3m_check 既有測試）：
  完整一致 ⇒ 無理由；零速後舊序號或非零套用、STOPPED 後仍發布、命令停止 5 步不足、停止後移動（底盤漂移）、觀察覆蓋不足、
  缺步（資料無效）、凍結目標非最後已發布、兩 ack 不一致、求解端結果非 DONE、獨立停止時刻與求解端差 > 0.1 s ⇒ 各自理由。

    python3 evaluation/test_door_k3m_check_p.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_p as C                          # noqa: E402

COLS = ['step', 'sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad', 'base_cmd_body', 'applied_arm', 'applied_src']
XYZ = [0.8, -0.05, 0.85]


def make(t_cmd_step=1015, n_end=2700, drift=None, gap=None):
    T = np.eye(4)
    T[:3, 3] = XYZ
    task = {'schema': 'door_k3m_task_p/1', 'final_phase': 'DONE',
            'trace': [{'pub_seq': 400, 'target': T.tolist()}],
            'k3m_pause': {'request': {'run_id': 'r', 'seq': 2, 'mode': 'pause', 'req_t': 10.0, 'frozen_target_seq': 400, 'frozen_target_xyz': XYZ},
                          't_exec_recv': 10.05, 't_solver_recv': 10.1, 'wall_req': 1.0, 'wall_exec_recv': 1.05, 'wall_solver_recv': 1.1,
                          'result': 'confirmed', 'outcome': {'state': 'DONE'}}}
    ea = {'run_id': 'r', 'seq': 2, 'mode': 'pause', 'switch_step': 1001, 'switch_t': 10.01}
    sa = dict(ea, first_pause_slot=5, first_pause_sim_t=10.02, publish_sim_t=10.03, published=True, frozen_target_xyz=XYZ)
    steps = []
    for k in range(900, n_end):
        if gap and gap[0] <= k < gap[1]:
            continue
        t = round(k * 0.01, 4)
        x = 0.6 + (drift * max(0.0, t - 15.0) if drift else 0.0)
        zero = k >= t_cmd_step - 4
        src = 50 if zero else 40
        steps.append([k, t, [x, -0.53, 1.57], [0.0] * 6, 0.14, [0.0, 0.0, 0.0] if zero else [0.001, 0.0, 0.0],
                      [0.0] * 6 if zero else [0.01] + [0.0] * 5, [k, t, src, 0]])
    room = {'steps_cols': COLS, 'steps': steps, 'k3m': {'pause_switch': ea}}
    log = [{'published': True, 'sim_t': 10.05, 'pause_zero': False, 'pause_source_seq': 40},
           {'published': True, 'sim_t': 10.10, 'pause_zero': True, 'pause_source_seq': 50},
           {'published': True, 'sim_t': 10.15, 'pause_zero': True, 'pause_source_seq': 51}]
    o = {'state': 'DONE', 'command_stop_confirmed': True, 'physical_stop_confirmed': True,
         'transitions': {'ZERO_PUB': 10.10, 'STOPPED': t_cmd_step * 0.01}}
    solver = {'stats': {'k3m_pause': {'record': {'ack': sa}, 'supervisor_outcome': o, 'infeasible_base_hold_cycles': 0}}, 'log': log}
    return task, room, solver


def reasons(task, room, solver):
    why, st = [], {}
    C.check_pause(task, room, solver, why, st)
    return why


class T(unittest.TestCase):
    def test_consistent(self):
        self.assertEqual(reasons(*make()), [])

    def test_failures(self):
        t, r, s = make()
        r['steps'][-50][7][2] = 40                                         # 停止後舊序號
        self.assertTrue(any(w.startswith('pause_old_or_nonzero_after_stop') for w in reasons(t, r, s)))
        t, r, s = make()
        r['steps'][-30][6] = [0.002] + [0.0] * 5                           # 停止後非零套用
        self.assertTrue(any(w.startswith('pause_old_or_nonzero_after_stop') for w in reasons(t, r, s)))
        t, r, s = make()
        s['log'].append({'published': True, 'sim_t': 12.0, 'pause_zero': True, 'pause_source_seq': 60})
        self.assertTrue(any(w.startswith('pause_published_after_stop') for w in reasons(t, r, s)))
        t, r, s = make(drift=0.003)                                         # 15 s 起底盤漂移 3 mm/s
        self.assertTrue(any(w.startswith('pause_moved') for w in reasons(t, r, s)))
        t, r, s = make(n_end=2000)                                          # 紀錄只到停止後約 9.8 s
        self.assertIn('pause_observation_incomplete', reasons(t, r, s))
        t, r, s = make(gap=(1500, 1503))
        self.assertTrue(any(w.startswith('pause_physical_data_invalid') for w in reasons(t, r, s)))
        t, r, s = make()
        t['trace'][0]['pub_seq'] = 399
        self.assertIn('pause_frozen_target_not_last_published', reasons(t, r, s))
        t, r, s = make()
        r['k3m']['pause_switch'] = dict(r['k3m']['pause_switch'], switch_step=1002)
        self.assertIn('pause_ack_mismatch', reasons(t, r, s))
        t, r, s = make()
        s['stats']['k3m_pause']['supervisor_outcome']['state'] = 'FAIL'
        self.assertIn('pause_solver_outcome_not_done', reasons(t, r, s))
        t, r, s = make()
        s['stats']['k3m_pause']['supervisor_outcome']['transitions']['STOPPED'] = 10.5
        self.assertIn('pause_stop_time_inconsistent', reasons(t, r, s))
        t, r, s = make()
        for row in r['steps']:
            if 1011 <= row[0] <= 1030 and row[0] % 2 == 0:                  # 零速段每隔一步序號舊 ⇒ 5 連續步延到 1031 起
                row[7][2] = 40
        w = reasons(t, r, s)
        self.assertIn('pause_stop_time_inconsistent', w)                    # 獨立停止 10.35 s vs 求解端 10.15 s
        for row in r['steps']:
            if row[0] >= 1011 and row[0] % 4 == 0:                           # 全程每 4 步就有一步舊序號 ⇒ 永遠湊不滿 5 步
                row[7][2] = 40
        self.assertIn('pause_command_stop_not_confirmed', reasons(t, r, s))
        t, r, s = make()
        r['cmd_env'] = {'n_plain_after_env': 2}
        self.assertIn('plain_cmd_after_envelope', reasons(t, r, s))
        t, r, s = make()
        del r['steps_cols'][7]
        self.assertIn('room_applied_src_missing', reasons(t, r, s))

    def test_mapping_pause_to_end_hold(self):
        task = {'schema': 'door_k3m_task_p/1', 'policy_events': [{'from': 'SETTLE', 'to': 'PAUSE'}, {'abort': 'x', 'phase': 'PAUSE'}],
                'trace': [{'phase': 'PAUSE'}]}
        t2 = C.to_k3m_task(task)
        self.assertEqual(t2['schema'], 'door_k3m_task/1')
        self.assertEqual(t2['policy_events'][0]['to'], 'END_HOLD')
        self.assertEqual(t2['policy_events'][1]['phase'], 'END_HOLD')
        self.assertEqual(t2['trace'][0]['phase'], 'END_HOLD')
        self.assertEqual(task['trace'][0]['phase'], 'PAUSE')                  # 不改原紀錄


if __name__ == '__main__':
    unittest.main(verbosity=2)
