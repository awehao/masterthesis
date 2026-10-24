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
  r2（Codex 20261025_012250）：D16 生效後求解列刪 pause_anchor_used／park_anchor／pause_state ⇒ pause_rows_missing；D17 刪全部 pause_published_feasible
  ⇒ infeasible；D18 source_step 改成不存在（含 stats 與邊緣列同改）⇒ source_not_edge_snapshot；D19 effective.sim_t −99 ⇒ not_next_solve、
  goal_u NaN／被改／生效列位姿不符 ⇒ goal_mismatch；D20 old＝new 合法零位移重設 ⇒ 通過；D21 邊緣列紀錄／事件 slot／邊緣時間越出週期；
  D22 減速段錨點非空 ⇒ inconsistent；D7 加 age 重算超過登錄上限、stats max_age≠登錄；D13 ok 非明確 True。

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


def room_for(sol, pose=None, dt=0.0, eff_pose=None):
    """來源步（source_sim_t）與生效列（effective.sim_t）各一步；位姿 6 位小數（同 room_run）。"""
    r = sol['stats']['pause_reanchor']['record']
    p = [round(v, 6) for v in (pose or r['new_anchor'])]
    pe = [round(v, 6) for v in (eff_pose or r['new_anchor'])]
    steps = [[1, round(r['source_sim_t'] - 0.01, 6), [0.0, 0.0, 0.0]], [2, r['source_sim_t'] + dt, p]]
    if W.get('eff_t') is not None:
        steps.append([3, W['eff_t'], pe])
    return {'steps_cols': ['step', 'sim_t', 'base_xyth'], 'steps': steps}


W = {'eff_t': None}


GATE = {'gate': {'anchor': [0.0, 0.0, 0.0]}}


def run(sol, room=None, gate=GATE, max_age=0.2):
    w, st = [], {}
    ok = D.check_reanchor(room if room is not None else room_for(sol), gate, sol, w, st, max_age)
    return ok, w


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r1, cls.r0 = fx('w1d_adm10_fakeworld_r1.json'), fx('w1d_adm10_fakeworld_r0.json')
        W['eff_t'] = cls.r1['stats']['pause_reanchor']['record']['effective']['sim_t']

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
        self.rec(s)['age_s'] = 0.5
        self.assertIn('reanchor_source_age_invalid', run(s)[1])
        self.assertIn('reanchor_source_age_invalid', run(self.r1, max_age=0.01)[1])     # 重算 age 0.03 > 登錄上限
        self.assertIn('reanchor_source_age_invalid', run(self.r1, max_age=0.5)[1])      # stats max_age≠登錄

    def test_d8_edge_row(self):
        s = copy.deepcopy(self.r1)
        e = next(x for x in s['log'] if isinstance(x.get('pause_reanchor_edge'), dict))
        e['pause_anchor_used'] = list(self.rec(s)['new_anchor'])
        self.assertIn('reanchor_edge_row_invalid', run(s)[1])

    def test_d9_effective(self):
        s = copy.deepcopy(self.r1)
        self.rec(s)['effective']['slot'] = self.edge_slot(s)
        self.assertIn('reanchor_effective_not_next_solve', run(s)[1])

    def test_d10_used(self):
        s = copy.deepcopy(self.r1)
        ef = self.rec(s)['effective']['slot']
        x = next(x for x in s['log'] if x.get('slot', -1) > ef + 5 and x.get('pause_anchor_used') is not None)
        x['pause_anchor_used'] = list(self.rec(s)['old_anchor'])
        self.assertTrue(any(w.startswith('reanchor_anchor_used_inconsistent') for w in run(s)[1]))

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
        s = copy.deepcopy(self.r1)
        x = [x for x in s['log'] if 'pause_published_feasible' in x][-2]
        x['pause_published_feasible'] = {'ok': 1}                                       # 必須明確為 True
        self.assertIn('reanchor_published_infeasible_1', run(s)[1])

    # r2（Codex 20261025_012250 記憶體內反例）
    def test_d16_missing_fields_after_effective(self):
        for k in ('pause_anchor_used', 'park_anchor', 'pause_state'):
            s = copy.deepcopy(self.r1)
            ef = self.rec(s)['effective']['slot']
            x = next(x for x in s['log'] if x.get('slot', -1) > ef + 3 and isinstance(x.get('solve_in'), dict))
            del x[k]
            self.assertIn('reanchor_pause_rows_missing_1', run(s)[1], k)

    def test_d17_all_feasible_removed(self):
        s = copy.deepcopy(self.r1)
        for x in s['log']:
            x.pop('pause_published_feasible', None)
        self.assertTrue(any(w.startswith('reanchor_published_infeasible_') for w in run(s)[1]))

    def test_d18_source_step(self):
        s = copy.deepcopy(self.r1)
        self.rec(s)['source_step'] = {'slot': 99999, 'snap_sim_t': 999.0, 'sp_step_id': 1}
        w = run(s)[1]
        self.assertIn('reanchor_source_not_edge_snapshot', w)
        self.assertIn('reanchor_edge_record_mismatch', w)
        s = copy.deepcopy(self.r1)
        for d in (self.rec(s), next(x for x in s['log'] if 'pause_reanchor_edge' in x)['pause_reanchor_edge']):
            d['source_step'] = dict(d['source_step'], slot=99999)                       # stats 與邊緣列同改：仍須對上邊緣列
        self.assertIn('reanchor_source_not_edge_snapshot', run(s)[1])

    def test_d19_effective_time_goal(self):
        s = copy.deepcopy(self.r1)
        self.rec(s)['effective']['sim_t'] = -99.0
        self.assertIn('reanchor_effective_not_next_solve', run(s)[1])
        s = copy.deepcopy(self.r1)
        self.rec(s)['effective']['goal_u'] = [float('nan'), 0.0, 0.0]
        self.assertIn('reanchor_effective_goal_mismatch', run(s)[1])
        s = copy.deepcopy(self.r1)
        self.rec(s)['effective']['goal_u'] = [0.001, 0.0, 0.0]
        self.assertIn('reanchor_effective_goal_mismatch', run(s)[1])
        # 生效列實測位姿與新錨點差 1 mm ⇒ 重算目標 −2 mm/s ≠ 紀錄 0
        self.assertIn('reanchor_effective_goal_mismatch', run(self.r1, room=room_for(self.r1, eff_pose=[0.004, 0.0, 0.0]))[1])

    def test_d20_zero_shift_positive(self):
        """old＝new 的合法零位移重設：依 slot 判定生效，不靠數值不同。"""
        s = copy.deepcopy(self.r1)
        old = list(self.rec(s)['old_anchor'])
        ed = next(x for x in s['log'] if 'pause_reanchor_edge' in x)
        for d in (self.rec(s), ed['pause_reanchor_edge']):
            d['new_anchor'] = list(old)
            d['anchor_shift'] = [0.0, 0.0, 0.0]
        next(e for e in s['stats']['k3m_pause']['events'] if e.get('stage') == 'reanchor')['anchor'] = list(old)
        for x in s['log']:
            if x.get('pause_anchor_used') is not None:
                x['pause_anchor_used'] = list(old)
        ok, w = run(s, room=room_for(s, pose=old, eff_pose=old))
        self.assertTrue(ok, w)

    def test_d21_edge_record_and_event(self):
        s = copy.deepcopy(self.r1)
        next(x for x in s['log'] if 'pause_reanchor_edge' in x)['pause_reanchor_edge']['edge_sim_t'] += 0.01
        self.assertIn('reanchor_edge_record_mismatch', run(s)[1])
        s = copy.deepcopy(self.r1)
        next(e for e in s['stats']['k3m_pause']['events'] if e.get('stage') == 'reanchor')['slot'] += 1
        self.assertIn('reanchor_event_not_at_edge', run(s)[1])
        s = copy.deepcopy(self.r1)
        self.rec(s)['edge_sim_t'] = self.rec(s)['edge_sim_t'] + 0.2
        next(x for x in s['log'] if 'pause_reanchor_edge' in x)['pause_reanchor_edge']['edge_sim_t'] = self.rec(s)['edge_sim_t']
        self.assertIn('reanchor_edge_time_out_of_cycle', run(s)[1])

    def test_d22_decel_anchor_must_be_empty(self):
        s = copy.deepcopy(self.r1)
        x = next(x for x in s['log'] if 'pause_anchor_used' in x and x['pause_anchor_used'] is None)
        x['pause_anchor_used'] = [0.0, 0.0, 0.0]
        self.assertTrue(any(w.startswith('reanchor_anchor_used_inconsistent') for w in run(s)[1]))

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
