#!/usr/bin/env python3
"""door_k1_check.judge 的離線反例（合成紀錄；不啟動模擬器）。

    python3 evaluation/test_door_k1_check.py
"""
import copy
import math
import os
import sys
import unittest

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k1_check as KC                             # noqa: E402

SPEC = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), 'src/my_omnibot_description/config/door_local_v1.yaml')))
DT, LO, HI = 0.01, 0.0, 110.0
I_HINGE, C, R_H = 2.4, 2.0, 0.80      # 合成用：門扇對鉸鏈慣量、角阻尼、把手半徑


def sim(q0_deg, force_n, force_s, observe_s, stop_at=None):
    """簡化門動力學：I q̈ = r·F − c·I·q̇，限位硬夾；回傳 rows 與 q_init。"""
    q, w, rows, stopped = math.radians(q0_deg), 0.0, [], None
    n_f, n = int(round(force_s / DT)), int(round((force_s + observe_s) / DT))
    for k in range(n):
        applying = k < n_f and force_n != 0.0 and stopped is None
        if applying and stop_at is not None and math.degrees(q) >= stop_at:
            stopped, applying = k * DT, False
        tau = R_H * force_n if applying else 0.0
        rows.append({'t': k * DT, 'q_rad': q, 'q_deg': math.degrees(q), 'qdot_rad_s': w, 'F': [0, 0, 0], 'tau': [0, 0, 0],
                     'open_torque_nm': tau, 'applying': applying, 'contact_walls_n': [0.0, 0.0]})
        w += (tau / I_HINGE - C * w) * DT
        q += w * DT
        if math.degrees(q) < LO:
            q, w = math.radians(LO), 0.0
        if math.degrees(q) > HI:
            q, w = math.radians(HI), 0.0
    return {'plan': {}, 'q_init_deg': q0_deg, 'q_final_deg': math.degrees(q), 'force_stopped_at_s': stopped, 'rows': rows}


def good():
    return {'complete': True, 'sleep_threshold_zero': True, 'readback_physx': {'masses': [8.0]},
            'tests': {'S1': sim(0, 0, 0, 10), 'S2': sim(0, 10, 0.5, 5), 'S3': sim(0, -10, 1.0, 1.0),
                      'S4': sim(0, 20, 6.0, 2.0, stop_at=109.5), 'S5': sim(45, 0, 0, 10)}}


class Judge(unittest.TestCase):
    def test_good_all_verified(self):
        r = KC.judge(good(), SPEC)
        self.assertEqual([r[k] for k in ('direction', 'lower_limit', 'upper_limit', 'passivity')], ['verified'] * 4, r)
        self.assertTrue(r['all_verified'])

    def test_wrong_direction(self):
        rep = good()
        rep['tests']['S2'] = sim(0, -10, 0.5, 5)          # 方向反了：門往負向、開門力矩為負
        self.assertEqual(KC.judge(rep, SPEC)['direction'], 'fail')

    def test_positive_motion_but_negative_torque_fails(self):
        rep = good()
        for r in rep['tests']['S2']['rows']:
            if r['applying']:
                r['open_torque_nm'] = -abs(r['open_torque_nm'])
        self.assertEqual(KC.judge(rep, SPEC)['direction'], 'fail')

    def test_hidden_hold_returns_to_zero(self):
        rep = good()
        for k, r in enumerate(rep['tests']['S5']['rows']):        # 隱藏保持：45° 慢慢被拉回 0
            r['q_deg'] = 45.0 * math.exp(-0.3 * r['t'])
        v = KC.judge(rep, SPEC)
        self.assertEqual(v['passivity'], 'fail')
        self.assertEqual(v['passivity_detail']['S5_nonzero_hold'], 'fail')

    def test_rest_at_zero_alone_is_not_enough(self):
        rep = good()
        del rep['tests']['S5']
        v = KC.judge(rep, SPEC)
        self.assertNotEqual(v['passivity'], 'verified')
        self.assertFalse(v['all_verified'])

    def test_lower_limit_with_wall_contact_not_separable(self):
        rep = good()
        for r in rep['tests']['S3']['rows']:
            r['contact_walls_n'] = [3.0, 0.0]
        self.assertEqual(KC.judge(rep, SPEC)['lower_limit'], 'not_separable')

    def test_lower_limit_penetration_fails(self):
        rep = good()
        rep['tests']['S3']['rows'][50]['q_deg'] = -2.5
        self.assertEqual(KC.judge(rep, SPEC)['lower_limit'], 'fail')

    def test_upper_not_reached_unverified(self):
        rep = good()
        rep['tests']['S4'] = sim(0, 0.5, 6.0, 2.0, stop_at=109.5)  # 力太小、時間內到不了 ⇒ 不自動加力
        self.assertEqual(KC.judge(rep, SPEC)['upper_limit'], 'unverified')

    def test_upper_overshoot_fails(self):
        rep = good()
        rep['tests']['S4']['rows'][-1]['q_deg'] = 113.0
        self.assertEqual(KC.judge(rep, SPEC)['upper_limit'], 'fail')

    def test_incomplete_blocks_all_verified(self):
        rep = good()
        rep['wall_timeouts'] = ['S4']
        v = KC.judge(rep, SPEC)
        self.assertFalse(v['complete'])
        self.assertFalse(v['all_verified'])

    def test_mass_readback_mismatch(self):
        rep = good()
        rep['readback_physx'] = {'masses': [8.5]}
        self.assertFalse(KC.judge(rep, SPEC)['all_verified'])

    def test_damping_flags_sleep(self):
        rep = good()
        rep['sleep_threshold_zero'] = 'failed: X'
        d = KC.judge(rep, SPEC)['damping_report']
        self.assertFalse(d['interpretable'])
        self.assertIn('sleep_threshold_not_zero', d['flags'])

    def test_damping_fit_matches_synthetic(self):
        d = KC.judge(good(), SPEC)['damping_report']
        self.assertTrue(d['interpretable'])
        self.assertAlmostEqual(d['fit_decay_per_s'], C, delta=0.05)


if __name__ == '__main__':
    unittest.main(verbosity=2)
