#!/usr/bin/env python3
"""door_k3m_check_w1d.check_reanchor 反例（夾具＝ADM10 ROS 假世界 R1／R0 的實際求解紀錄，results/door/fixtures/；
room／gate 由測試合成：來源步 base_xyth＝夾具 new_anchor（6 位小數）、gate.anchor＝假世界 /park/gate 的 [0,0,0]）。
  D1 R1 ⇒ 通過；D2 R0（未啟用）／缺紀錄 ⇒ reanchor_missing_or_not_applied；D3 status＝rejected 或 n_calls＝2 ⇒ 同上；
  D4 old≠hold 錨點 ⇒ reanchor_old_not_hold_anchor；D5 多一筆 reanchor 事件 ⇒ reanchor_event_count_2；
  D6 room 來源步位姿不符／缺來源步 ⇒ reanchor_new_not_measured_pose；D7 age 超過 max ⇒ reanchor_source_age_invalid；
  D8 邊緣列的 pause_anchor_used 改成 new（本輪已生效的假象）⇒ reanchor_edge_row_invalid；
  D9 effective.slot＝邊緣 slot ⇒ reanchor_effective_not_after_edge；D10 生效後某列仍用 old ⇒ reanchor_anchor_used_inconsistent；
  D11 first_published 序號改掉 ⇒ reanchor_first_published_invalid；D12 某列 park_anchor 移動 ⇒ reanchor_park_anchor_moved；
  D13 某列發布不可行 ⇒ reanchor_published_infeasible_1；D14 凍結目標缺 ⇒ reanchor_frozen_target_missing；
  D15 judge_with 登錄未含 pause_hold_reanchor ⇒ w1d_solver_change_not_registered；求解端 max_age 與登錄不符 ⇒ w1d_solver_args_mismatch。

    python3 evaluation/test_door_k3m_check_w1d.py
"""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_w1d as D                         # noqa: E402

FX = os.path.join(HERE, 'results', 'door', 'fixtures')


def fx(name):
    return json.load(open(os.path.join(FX, name)))


def room_for(sol, pose=None, dt=0.0):
    r = sol['stats']['pause_reanchor']['record']
    p = [round(v, 6) for v in (pose or r['new_anchor'])]
    return {'steps_cols': ['step', 'sim_t', 'base_xyth'],
            'steps': [[1, round(r['source_sim_t'] - 0.01, 6), [0.0, 0.0, 0.0]], [2, r['source_sim_t'] + dt, p]]}


GATE = {'gate': {'anchor': [0.0, 0.0, 0.0]}}


def run(sol, room=None, gate=GATE):
    w, st = [], {}
    ok = D.check_reanchor(room if room is not None else room_for(sol), gate, sol, w, st)
    return ok, w


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r1, cls.r0 = fx('w1d_adm10_fakeworld_r1.json'), fx('w1d_adm10_fakeworld_r0.json')

    def rec(self, s):
        return s['stats']['pause_reanchor']['record']

    def edge_slot(self, s):
        return next(x['slot'] for x in s['log'] if isinstance(x.get('pause_reanchor_edge'), dict))

    def test_d1_pass(self):
        ok, w = run(self.r1)
        self.assertTrue(ok, w)

    def test_d2_missing(self):
        self.assertIn('reanchor_missing_or_not_applied', run(self.r0, room={'steps_cols': [], 'steps': []})[1])
        s = copy.deepcopy(self.r1)
        s['stats']['pause_reanchor'] = None
        self.assertIn('reanchor_missing_or_not_applied', run(s, room=room_for(self.r1))[1])

    def test_d3_not_applied_or_twice(self):
        s = copy.deepcopy(self.r1)
        self.rec(s)['status'] = 'rejected'
        self.assertIn('reanchor_missing_or_not_applied', run(s, room=room_for(self.r1))[1])
        s = copy.deepcopy(self.r1)
        s['stats']['pause_reanchor']['n_calls'] = 2
        self.assertIn('reanchor_missing_or_not_applied', run(s)[1])

    def test_d4_old(self):
        s = copy.deepcopy(self.r1)
        self.rec(s)['old_anchor'] = [0.001, 0.0, 0.0]
        self.assertIn('reanchor_old_not_hold_anchor', run(s)[1])

    def test_d5_two_events(self):
        s = copy.deepcopy(self.r1)
        ev = s['stats']['k3m_pause']['events']
        ev.append(copy.deepcopy(next(e for e in ev if e.get('stage') == 'reanchor')))
        self.assertIn('reanchor_event_count_2', run(s)[1])

    def test_d6_measured(self):
        self.assertIn('reanchor_new_not_measured_pose', run(self.r1, room=room_for(self.r1, pose=[0.0031, 0.0, 0.0]))[1])
        self.assertIn('reanchor_new_not_measured_pose', run(self.r1, room=room_for(self.r1, dt=0.01))[1])

    def test_d7_age(self):
        s = copy.deepcopy(self.r1)
        self.rec(s)['edge_sim_t'] = self.rec(s)['source_sim_t'] + 0.5
        self.assertIn('reanchor_source_age_invalid', run(s)[1])

    def test_d8_edge_row(self):
        s = copy.deepcopy(self.r1)
        e = next(x for x in s['log'] if isinstance(x.get('pause_reanchor_edge'), dict))
        e['pause_anchor_used'] = list(self.rec(s)['new_anchor'])
        self.assertIn('reanchor_edge_row_invalid', run(s)[1])

    def test_d9_effective(self):
        s = copy.deepcopy(self.r1)
        self.rec(s)['effective']['slot'] = self.edge_slot(s)
        self.assertIn('reanchor_effective_not_after_edge', run(s)[1])

    def test_d10_used(self):
        s = copy.deepcopy(self.r1)
        ef = self.rec(s)['effective']['slot']
        x = next(x for x in s['log'] if x.get('slot', -1) > ef + 5 and x.get('pause_anchor_used') is not None)
        x['pause_anchor_used'] = list(self.rec(s)['old_anchor'])
        self.assertIn('reanchor_anchor_used_inconsistent', run(s)[1])

    def test_d11_first_published(self):
        s = copy.deepcopy(self.r1)
        self.rec(s)['first_published']['source_seq'] += 1
        self.assertIn('reanchor_first_published_invalid', run(s)[1])

    def test_d12_park_anchor(self):
        s = copy.deepcopy(self.r1)
        x = [x for x in s['log'] if 'pause_anchor_used' in x][-3]
        x['park_anchor'] = [0.001, 0.0, 0.0]
        self.assertIn('reanchor_park_anchor_moved', run(s)[1])

    def test_d13_infeasible(self):
        s = copy.deepcopy(self.r1)
        x = [x for x in s['log'] if 'pause_published_feasible' in x][-2]
        x['pause_published_feasible'] = {'ok': False, 'why': ['a_lin']}
        self.assertIn('reanchor_published_infeasible_1', run(s)[1])

    def test_d14_frozen_target(self):
        s = copy.deepcopy(self.r1)
        s['stats']['k3m_pause']['record'].pop('frozen_target_T', None)
        self.assertIn('reanchor_frozen_target_missing', run(s)[1])

    def test_d15_registration(self):
        import door_k3m_check_w1c as W1C
        orig = W1C.judge_with
        W1C.judge_with = lambda *a, **k: {'pass': True, 'why': [], 'stats': {}}
        try:
            res = D.judge_with({}, room_for(self.r1), GATE, self.r1, None, {'solver_change': {'offset_hold_on_free': True}})
            self.assertIn('w1d_solver_change_not_registered', res['why'])
            res = D.judge_with({}, room_for(self.r1), GATE, self.r1, None,
                               {'solver_change': {'offset_hold_on_free': True, 'pause_hold_reanchor': True, 'pause_reanchor_max_age_s': 0.2}})
            self.assertTrue(res['pass'], res['why'])
            s = copy.deepcopy(self.r1)
            s['args']['pause_reanchor_max_age_s'] = 0.5
            res = D.judge_with({}, room_for(s), GATE, s, None,
                               {'solver_change': {'offset_hold_on_free': True, 'pause_hold_reanchor': True, 'pause_reanchor_max_age_s': 0.2}})
            self.assertIn('w1d_solver_args_mismatch', res['why'])
        finally:
            W1C.judge_with = orig


if __name__ == '__main__':
    unittest.main(verbosity=1)
