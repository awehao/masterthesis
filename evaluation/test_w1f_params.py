#!/usr/bin/env python3
"""W1f 60° 登錄與跑法的離線測試（Codex 20261025_030420：差異規格＋離線核對＋最小實作／測試）。
  P1 θ 相關項逐字＝door_w1b_params.py 60 的推得結果（w1b_params_60_r2.json），且 derived_from sha256 對上；
  P2 solver_change／executor_change 與 W1e 登錄逐字相同（OPEN 5 N、暫停期 20 N、ADM9 偏移保持）；
  P3 規則值：θ 60、任務上界 61、觀察器 64、OPEN 逾時＝斜坡＋20.3 s（82.0）、包絡餘裕核對 ok、內帶 20 mm／0.8° 不變；
  P4 走廊表角度涵蓋 ≥ 觀察器上限 64°，表 sha256＝登錄；
  P5 執行腳本與 run_door_wide_w1e_r3.sh 除註解外只差 FRZ、θ 預設、PRM、拒跑字樣；
  P6 判定器 w1e2 以本登錄的 executor_change 判 w1e_30_2 的手指上限證據 ⇒ 通過（判定器由登錄驅動、與 θ 無關的部分可直接沿用）。

    python3 evaluation/test_w1f_params.py
"""
import hashlib
import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
D = os.path.join(HERE, 'results', 'door')


def ld(n):
    return json.load(open(os.path.join(D, n)))


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f, cls.b, cls.e = ld('w1f_params_60.json'), ld('w1b_params_60_r2.json'), ld('w1e_params_30.json')

    def test_p1_theta_items_from_rule(self):
        for k in ('theta_goal_deg', 'task_param_extension', 'protection_scope_change', 'control_cost_change', 'pause', 'reference_table_note'):
            self.assertEqual(self.f[k], self.b[k], k)
        df = self.f['derived_from']
        self.assertEqual(df['w1b_params_60_r2'], hashlib.sha256(open(os.path.join(D, 'w1b_params_60_r2.json'), 'rb').read()).hexdigest())
        self.assertEqual(df['w1e_params_30'], hashlib.sha256(open(os.path.join(D, 'w1e_params_30.json'), 'rb').read()).hexdigest())

    def test_p2_changes_same_as_w1e(self):
        self.assertEqual(self.f['solver_change'], self.e['solver_change'])
        self.assertEqual(self.f['executor_change'], self.e['executor_change'])
        self.assertEqual(self.f['executor_change']['pause_finger_max_effort'], 20.0)
        self.assertEqual(self.f['executor_change']['original_effort'], 5.0)
        self.assertIs(self.f['solver_change']['offset_hold_on_free'], True)

    def test_p3_rule_values(self):
        tp, pc = self.f['task_param_extension'], self.f['protection_scope_change']
        self.assertEqual(tp['theta_goal_deg'], 60.0)
        self.assertEqual(tp['policy_upper_deg'], 61.0)
        self.assertEqual(pc['observer_door_max_deg'], 64.0)
        self.assertAlmostEqual(tp['open_timeout_s'], round(tp['ramp_s'] + 20.3, 1))
        self.assertEqual(tp['open_timeout_s'], 82.0)
        self.assertIs(pc['envelope_margin_check']['ok'], True)
        self.assertTrue(all(v >= 0 for v in pc['envelope_margin_check']['min_margin_m_or_deg'].values()))
        self.assertEqual((pc['corridor_inner_mm'], pc['corridor_inner_deg']), (20.0, 0.8))

    def test_p4_corridor_coverage(self):
        pc = self.f['protection_scope_change']
        p = os.path.join(WS, pc['corridor_table'])
        self.assertEqual(hashlib.sha256(open(p, 'rb').read()).hexdigest(), pc['corridor_table_sha256'])
        q = json.load(open(p))['table']['q_door_deg']
        self.assertGreaterEqual(max(q), pc['observer_door_max_deg'])

    def test_p5_runner_diff(self):
        out = subprocess.run(['diff', os.path.join(HERE, 'run_door_wide_w1e_r3.sh'), os.path.join(HERE, 'run_door_wide_w1f.sh')],
                             capture_output=True, text=True).stdout
        ch = [l for l in out.splitlines() if l[:2] in ('< ', '> ') and not l[2:].startswith('#')]
        self.assertEqual(len(ch), 8, ch)
        for l in ch:
            self.assertTrue(any(k in l for k in ('FRZ=', 'W1B_THETA=', 'PRM=', '開發凍結不符')), l)

    def test_p6_judge_uses_registration(self):
        import door_k3m_check_w1e as E
        import door_k3m_check_w1e2 as E2
        room = json.load(open(os.path.join(HERE, 'runs', 'door_wide_w1e_30_2', 'room_run.json')))
        orig = E._at_room
        E._at_room = E2._at_room_r2
        try:
            w, st = [], {}
            self.assertTrue(E.check_finger_effort(room, self.f['executor_change'], w, st), w)
        finally:
            E._at_room = orig


if __name__ == '__main__':
    unittest.main(verbosity=1)
