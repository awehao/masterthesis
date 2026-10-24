#!/usr/bin/env python3
"""door_k3m_check_w1c.check_offset_hold 反例（夾具＝ADM9 ROS 假世界 H1／H0 的實際求解紀錄，results/door/fixtures/）：
  K1 H1（啟用保持）⇒ 通過；K2 缺 offset_hold／H0（未啟用）⇒ offset_hold_missing_or_invalid；
  K3 凍結後插入 used＝updated ⇒ not_held；K4 凍結後 d_after 改變 ⇒ not_held；K5 凍結時刻偏離首次 free ⇒ boundary_mismatch；
  K6 first_published 缺或早於凍結 ⇒ first_published_invalid；K7 報告更新數≠凍結前更新數 ⇒ update_count_mismatch；
  K8 d_frozen≠凍結前最後一筆 ⇒ frozen_value_not_last_available；K9 登錄未含 solver_change ⇒ w1c_solver_change_not_registered（judge_with）；
  r2（Codex 20261025_003515 三個誤收反例）：K10 刪掉一筆凍結後已發布解的 offset_interval ⇒ not_held；
  K11 改 solve_in.offset_d_hat 或 arm_bias ⇒ solve_input_mismatch；K12 first_published 的 slot／發布時間改成不存在的值 ⇒ first_published_invalid；
  K13 暫停觀察等無 solve_in 的診斷列另列計數、不要求求解證據。

    python3 evaluation/test_door_k3m_check_w1c.py
"""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_check_w1c as C                         # noqa: E402

FX = os.path.join(HERE, 'results', 'door', 'fixtures')


def fx(name):
    return json.load(open(os.path.join(FX, name)))


def why_of(sol):
    w, st = [], {}
    ok = C.check_offset_hold(sol, w, st)
    return ok, w


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h1, cls.h0 = fx('w1c_adm9_fakeworld_h1.json'), fx('w1c_adm9_fakeworld_h0.json')

    def post_rows(self, sol):
        th = sol['stats']['offset_hold']['sim_t']
        return [x for x in sol['log'] if isinstance(x.get('offset_interval'), dict) and x['sim_t'] >= th]

    def test_k1_consistent(self):
        ok, w = why_of(self.h1)
        self.assertTrue(ok, w)
        self.assertEqual(w, [])

    def test_k2_missing(self):
        s = copy.deepcopy(self.h1)
        del s['stats']['offset_hold']
        self.assertIn('offset_hold_missing_or_invalid', why_of(s)[1])
        self.assertIn('offset_hold_missing_or_invalid', why_of(self.h0)[1])

    def test_k3_k4_not_held(self):
        s = copy.deepcopy(self.h1)
        self.post_rows(s)[5]['offset_interval']['used'] = 'updated'
        self.assertTrue(any(x.startswith('offset_hold_not_held') for x in why_of(s)[1]))
        s = copy.deepcopy(self.h1)
        r = self.post_rows(s)[7]['offset_interval']
        r['d_after'] = [v + 1e-6 for v in r['d_after']]
        self.assertTrue(any(x.startswith('offset_hold_not_held') for x in why_of(s)[1]))

    def test_k5_boundary(self):
        s = copy.deepcopy(self.h1)
        s['stats']['offset_hold']['sim_t'] += 0.2
        self.assertIn('offset_hold_boundary_mismatch', why_of(s)[1])

    def test_k6_first_published(self):
        s = copy.deepcopy(self.h1)
        s['stats']['offset_hold']['first_published'] = None
        self.assertIn('offset_hold_first_published_invalid', why_of(s)[1])
        s = copy.deepcopy(self.h1)
        s['stats']['offset_hold']['first_published']['publish_sim_t_exact'] = s['stats']['offset_hold']['sim_t'] - 0.1
        self.assertIn('offset_hold_first_published_invalid', why_of(s)[1])

    def test_k7_count(self):
        s = copy.deepcopy(self.h1)
        s['offset_free']['n_updates'] += 1
        self.assertIn('offset_hold_update_count_mismatch', why_of(s)[1])

    def test_k8_frozen_value(self):
        s = copy.deepcopy(self.h1)
        s['stats']['offset_hold']['d_frozen'] = [0.001] * 6
        self.assertIn('offset_hold_frozen_value_not_last_available', why_of(s)[1])

    def test_k9_unregistered(self):
        orig = C.W1B.judge_with
        try:
            C.W1B.judge_with = lambda *a, **k: {'why': [], 'stats': {}, 'pass': True}
            r = C.judge_with({}, {}, None, self.h1, None, {'solver_change': {}})
            self.assertIn('w1c_solver_change_not_registered', r['why'])
            self.assertFalse(r['pass'] or r['offset_hold_ok'])
            r = C.judge_with({}, {}, None, self.h1, None, {'solver_change': {'offset_hold_on_free': True}})
            self.assertEqual(r['why'], [])
            self.assertTrue(r['pass'] and r['offset_hold_ok'])
        finally:
            C.W1B.judge_with = orig

    def pub_after(self, sol):
        th = sol['stats']['offset_hold']['sim_t']
        return [x for x in sol['log'] if x.get('sim_t', 0) >= th and x.get('published')]

    def test_k10_missing_offset_interval_on_published(self):
        s = copy.deepcopy(self.h1)
        del self.pub_after(s)[10]['offset_interval']
        self.assertTrue(any(x.startswith('offset_hold_not_held') for x in why_of(s)[1]), why_of(s)[1])

    def test_k11_solve_input(self):
        for key in ('offset_d_hat', 'arm_bias'):
            s = copy.deepcopy(self.h1)
            si = self.pub_after(s)[20]['solve_in']
            si[key] = [v + 1e-6 for v in si[key]]
            self.assertTrue(any(x.startswith('offset_hold_solve_input_mismatch') for x in why_of(s)[1]), key)

    def test_k12_first_published_mismatch(self):
        for key, val in (('slot', 99999), ('publish_sim_t_exact', 123.456)):
            s = copy.deepcopy(self.h1)
            s['stats']['offset_hold']['first_published'][key] = val
            self.assertIn('offset_hold_first_published_invalid', why_of(s)[1], key)

    def test_k13_diag_rows_counted(self):
        w, st = [], {}
        C.check_offset_hold(self.h1, w, st)
        c = st['offset_hold_check']
        self.assertGreater(c['n_diag_rows_after'], 0)
        self.assertGreater(c['n_solve_rows_after'], 0)
        self.assertEqual((c['n_bad'], c['n_bad_solve_input']), (0, 0))

if __name__ == '__main__':
    unittest.main(verbosity=2)
