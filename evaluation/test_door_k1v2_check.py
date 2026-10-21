#!/usr/bin/env python3
"""【K1 v2】door_k1v2_check 的離線反例（複本自 test_door_k1_check.py，加門框接觸判準反例）；door_k1_force（合成紀錄；不啟動模擬器）。USD 讀回用 K0 r2 的真實讀回值。

    python3 evaluation/test_door_k1v2_check.py
"""
import copy
import hashlib
import json
import math
import os
import sys
import unittest

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_asset_v2 as DA                                # noqa: E402
import door_k1v2_check as KC                             # noqa: E402
import door_k1_force as KF                             # noqa: E402
import door_k1_plan_v2 as KP                              # noqa: E402

SP = os.path.join(WS, KP.SPEC_REL)
SPEC = yaml.safe_load(open(SP))
SHA = hashlib.sha256(open(SP, 'rb').read()).hexdigest()
USD_RB = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k0v2_readback.json')))['readback']
HINGE = np.zeros(3)
I_HINGE, C = 2.4, 2.0
LO, HI = SPEC['leaf']['limit_q_deg']


def p_handle(q):
    H = SPEC['handle']
    c, s = math.cos(q), math.sin(q)
    R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])       # Rz(−q)
    return R @ np.array([H['from_hinge'], -H['standoff'], H['height']])


def sim(P):
    """簡化門動力學（I q̈ = τ − c I q̇，限位硬夾），紀錄格式同 door_k1_sim。"""
    q, w, rows, stopped = math.radians(P['q0_deg']), 0.0, [], None
    n_force, n_total = KP.n_steps(P)
    for k in range(n_total):
        applying = k < n_force and P['force_n'] != 0.0 and stopped is None
        if applying and P['stop_force_at_deg'] is not None and math.degrees(q) >= P['stop_force_at_deg']:
            stopped, applying = k * KP.DT, False
        pH = p_handle(q)
        cmd = KF.handle_force(pH, HINGE, KP.A_OPEN, P['force_n']) if applying else \
            {'p_apply': pH, 'F': np.zeros(3), 'tau_extra': np.zeros(3), 'tau_open': 0.0}
        q_pre, w_pre = q, w
        w += (cmd['tau_open'] / I_HINGE - C * w) * KP.DT
        q += w * KP.DT
        if math.degrees(q) < LO:
            q, w = math.radians(LO), 0.0
        if math.degrees(q) > HI:
            q, w = math.radians(HI), 0.0
        rows.append({'k': k, 't_pre': round(k * KP.DT, 6), 'q_pre_deg': math.degrees(q_pre), 'qdot_pre': w_pre, 'applying': applying,
                     'p_apply': cmd['p_apply'].tolist(), 'F': cmd['F'].tolist(), 'tau_extra': cmd['tau_extra'].tolist(),
                     'tau_open_nm': cmd['tau_open'], 't_post': round((k + 1) * KP.DT, 6), 'q_post_deg': math.degrees(q),
                     'qdot_post': w, 'contact_walls_n': [0.0, 0.0]})
    return {'plan': P, 'status': 'done', 'sleep_threshold_zero': True, 'q0_actual_deg': P['q0_deg'], 'qdot0_actual': 0.0,
            'force_stopped_at_s': stopped, 'rows': rows}


def good():
    m, com, I = DA.leaf_inertia(SPEC)
    plan = KP.plan(SPEC)
    return {'schema': 'door_k1/3', 'spec_sha256': SHA, 'plan': plan, 'complete': True, 'readback_usd': copy.deepcopy(USD_RB),
            'readback_physx': {'masses': [m], 'coms': [[list(com)], [[1.0, 0, 0, 0]]], 'inertias': [np.diag(I).ravel().tolist()]},
            'tests': {n: sim(P) for n, P in plan.items()}}


def J(rep):
    return KC.judge(rep, SPEC, SHA)


class Judge(unittest.TestCase):
    def test_good_all_verified(self):
        r = J(good())
        self.assertEqual(r['integrity_issues'], [])
        self.assertEqual([r[k] for k in ('direction', 'lower_limit', 'upper_limit', 'passivity')], ['verified'] * 4, r)
        self.assertTrue(r['all_verified'], r['readback'])

    # ---- Codex 的六個反例
    def test_s3_contacts_none(self):
        rep = good()
        for r in rep['tests']['S3']['rows']:
            r['contact_walls_n'] = None
        v = J(rep)
        self.assertFalse(v['integrity'])
        self.assertNotEqual(v['lower_limit'], 'verified')
        self.assertFalse(v['all_verified'])

    def test_s3_no_force_rows(self):
        rep = good()
        for r in rep['tests']['S3']['rows']:
            r['applying'] = False
            r['F'] = [0, 0, 0]
            r['tau_open_nm'] = 0.0
        v = J(rep)
        self.assertNotEqual(v['lower_limit'], 'verified')
        self.assertFalse(v['all_verified'])

    def test_s1_s5_truncated(self):
        rep = good()
        rep['tests']['S1']['rows'] = rep['tests']['S1']['rows'][:1]
        rep['tests']['S5']['rows'] = rep['tests']['S5']['rows'][:1]
        v = J(rep)
        self.assertFalse(v['integrity'])
        self.assertNotEqual(v['passivity'], 'verified')
        self.assertFalse(v['all_verified'])

    def test_s3_final_post_angle_negative(self):
        rep = good()
        rep['tests']['S3']['rows'][-1]['q_post_deg'] = -5.0
        self.assertEqual(J(rep)['lower_limit'], 'fail')

    def test_usd_nonzero_drive(self):
        rep = good()
        rep['readback_usd']['drive_stiffness'] = 100.0
        v = J(rep)
        self.assertEqual(v['readback']['usd'], 'fail')
        self.assertFalse(v['all_verified'])

    def test_s3_contact_nan(self):
        rep = good()
        rep['tests']['S3']['rows'][10]['contact_walls_n'] = [float('nan'), 0.0]
        v = J(rep)
        self.assertFalse(v['integrity'])
        self.assertNotEqual(v['lower_limit'], 'verified')

    # ---- 其他
    def test_wrong_direction(self):
        rep = good()
        P = dict(rep['plan']['S2'], force_n=-10.0)
        rep['tests']['S2'] = sim(P)
        self.assertNotEqual(J(rep)['direction'], 'verified')

    def test_tau_recompute_mismatch(self):
        rep = good()
        rep['tests']['S2']['rows'][3]['tau_open_nm'] *= 2.0             # 重複計入力臂
        self.assertIn('S2:tau_recompute_mismatch@3', J(rep)['integrity_issues'])

    def test_hidden_hold_returns_to_zero(self):
        rep = good()
        for r in rep['tests']['S5']['rows']:
            r['q_pre_deg'] = 45.0 * math.exp(-0.3 * r['t_pre'])
            r['q_post_deg'] = 45.0 * math.exp(-0.3 * r['t_post'])
        self.assertEqual(J(rep)['passivity'], 'fail')

    def test_s5_sleep_failed_unverified(self):
        rep = good()
        rep['tests']['S5']['sleep_threshold_zero'] = 'failed: X'
        self.assertEqual(J(rep)['passivity_detail']['S5_nonzero_hold'], 'unverified')

    def test_frame_contact_good_verified(self):
        self.assertEqual(J(good())['frame_contact_excluded'], 'verified')

    def test_frame_contact_in_static_fails(self):
        rep = good()
        rep['tests']['S1']['rows'][500]['contact_walls_n'] = [0.0, 0.5]
        v = J(rep)
        self.assertEqual(v['frame_contact_excluded'], 'fail')
        self.assertFalse(v['all_verified'])

    def test_frame_contact_in_release_segment_fails(self):
        rep = good()
        rep['tests']['S2']['rows'][-1]['contact_walls_n'] = [0.2, 0.0]   # 撤力段最後一步
        self.assertEqual(J(rep)['frame_contact_excluded'], 'fail')

    def test_frame_contact_nonzero_hold_fails(self):
        rep = good()
        rep['tests']['S5']['rows'][0]['contact_walls_n'] = [0.16, 0.0]   # v1 實錄 S5 的大小
        self.assertEqual(J(rep)['frame_contact_excluded'], 'fail')

    def test_wall_contact_not_separable(self):
        rep = good()
        for r in rep['tests']['S3']['rows']:
            r['contact_walls_n'] = [3.0, 0.0]
        self.assertEqual(J(rep)['lower_limit'], 'not_separable')

    def test_upper_not_reached(self):
        rep = good()
        rep['tests']['S4'] = sim(dict(rep['plan']['S4'], force_n=0.5))
        rep['plan']['S4']['force_n'] = 0.5
        v = J(rep)
        self.assertIn('plan_mismatch', v['integrity_issues'])         # 改計畫被抓到
        self.assertFalse(v['all_verified'])

    def test_upper_overshoot(self):
        rep = good()
        rep['tests']['S4']['rows'][-1]['q_post_deg'] = 113.0
        self.assertEqual(J(rep)['upper_limit'], 'fail')

    def test_spec_hash_mismatch(self):
        rep = good()
        rep['spec_sha256'] = '0' * 64
        self.assertIn('spec_sha_mismatch', J(rep)['integrity_issues'])

    def test_q0_mismatch(self):
        rep = good()
        rep['tests']['S5']['q0_actual_deg'] = 40.0
        self.assertIn('S5:q0_mismatch', J(rep)['integrity_issues'])

    def test_time_gap(self):
        rep = good()
        del rep['tests']['S2']['rows'][100]
        self.assertFalse(J(rep)['integrity'])

    def test_interrupted_run(self):
        rep = good()
        rep['interrupted'] = 'SIGTERM'
        rep['tests']['S4']['status'] = 'interrupted:SIGTERM'
        self.assertFalse(J(rep)['all_verified'])

    def test_physx_readback_unparseable_unverified(self):
        rep = good()
        rep['readback_physx'] = {'masses': {'error': 'x'}, 'coms': {'error': 'x'}, 'inertias': {'error': 'x'}}
        v = J(rep)
        self.assertEqual([v['readback'][k] for k in ('physx_mass', 'physx_com', 'physx_inertia')], ['unverified'] * 3)
        self.assertFalse(v['all_verified'])

    def test_damping_fit(self):
        d = J(good())['damping_report']
        self.assertTrue(d['interpretable'])
        self.assertAlmostEqual(d['fit_decay_per_s'], C, delta=0.05)


class ForceSemantics(unittest.TestCase):
    def test_hinge_moment_is_r_cross_F(self):
        q = math.radians(30)
        pH = p_handle(q)
        cmd = KF.handle_force(pH, HINGE, KP.A_OPEN, 10.0)
        self.assertAlmostEqual(cmd['tau_open'], 10.0 * math.hypot(pH[0], pH[1]), places=9)
        self.assertGreater(cmd['tau_open'], 0)
        self.assertTrue(np.allclose(cmd['tau_extra'], 0))

    def test_old_origin_plus_com_torque_is_wrong(self):
        """舊做法：positions=None（力施於剛體原點＝鉸鏈）＋ τ=(p_H−p_COM)×F ⇒ 鉸鏈合力矩 ≠ (p_H−c)×F。"""
        q = math.radians(30)
        pH = p_handle(q)
        F = KF.handle_force(pH, HINGE, KP.A_OPEN, 10.0)['F']
        m, com_local, _ = DA.leaf_inertia(SPEC)
        c, s = math.cos(q), math.sin(q)
        pC = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]]) @ np.array(com_local)
        old = KF.hinge_moment(HINGE, F, np.cross(pH - pC, F), HINGE, KP.A_OPEN)
        new = KF.hinge_moment(pH, F, np.zeros(3), HINGE, KP.A_OPEN)
        self.assertGreater(abs(old - new), 0.5)

    def test_api_called_with_world_handle_point_forces_only(self):
        calls = []

        class Mock:
            def apply_forces_and_torques_at_pos(self, **kw):
                calls.append(kw)
        pH = p_handle(0.2)
        cmd = KF.handle_force(pH, HINGE, KP.A_OPEN, 10.0)
        KF.apply(Mock(), cmd)
        kw = calls[0]
        self.assertIsNone(kw['torques'])
        self.assertTrue(kw['is_global'])
        np.testing.assert_allclose(kw['positions'][0], pH, atol=1e-6)
        np.testing.assert_allclose(kw['forces'][0], cmd['F'], atol=1e-6)

    def test_tangent_on_axis_rejected(self):
        with self.assertRaises(ValueError):
            KF.tangent(np.array([0, 0, 0.85]), HINGE, KP.A_OPEN)


if __name__ == '__main__':
    unittest.main(verbosity=2)
