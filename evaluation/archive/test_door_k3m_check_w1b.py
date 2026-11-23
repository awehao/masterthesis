#!/usr/bin/env python3
"""door_k3m_check_w1b 反例（以 door_k3m_pause_p4b_1 實錄＋合成登錄檔）：
  J1 以 8° 登錄（同 p4b_1 實際參數）重判 ⇒ 除引導類理由外與 p4b2 相同（pass）；
  J2 判完（含例外）door_k3a_check／door_k3m_check 的模組常數一定還原；
  J3 登錄不符（θ、包絡、觀察器上限、走廊表雜湊）⇒ 各自理由；
  J4 引導相位：PAUSE 中 w＞0、OPEN 未切 free 時 w＞0 ⇒ guide_phase_violation；OPEN 已切 free 卻從未開啟 ⇒ guide_never_on；
  J5 以 30° 登錄判 8° 資料 ⇒ 出現未達目標類理由（證明 30° 常數確實生效、未暗用 8°）；
  J6（r2，Codex 20261024_234630 第 3 項）缺證據不誤收：刪 task θ、內帶或包絡元素 NaN、觀察器上限缺、走廊表雜湊非法 ⇒ *_missing_or_invalid；
     PAUSE 列引導權重 NaN、權重非登錄值、開啟列 v_body 非有限 ⇒ guide_phase_violation。

    python3 evaluation/test_door_k3m_check_w1b.py
"""
import copy
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3a_check as KC3                           # noqa: E402
import door_k3m_check as KM                            # noqa: E402
import door_k3m_check_p4b2 as C4B2                     # noqa: E402
import door_k3m_check_w1b as C                         # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_k3m_pause_p4b_1')


def load():
    ld = lambda f: json.load(open(os.path.join(RUN, f)))
    return ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json')


def params(theta=8.0, envelope=(-0.15, 0.02, -0.05, 0.10, 6.0), obs=12.0, sha=None, room=None):
    kc = room['k3m']['config']
    return {'task_param_extension': {'theta_goal_deg': theta, 'over_goal_deg': 1.0, 'open_timeout_s': 30.0},
            'protection_scope_change': {'corridor_table_sha256': sha or kc['corridor_table_sha256'], 'corridor_inner_mm': 20.0,
                                        'corridor_inner_deg': 0.8, 'observer_door_max_deg': obs,
                                        'envelope': dict(zip(('fwd_lo', 'fwd_hi', 'lat_lo', 'lat_hi', 'yaw_abs_deg'), envelope))},
            'control_cost_change': {'guidance': {'w_vref': 1.0}}}


def with_guide(task, rule):
    t = copy.deepcopy(task)
    for x in t['trace']:
        x['guide'] = {'w': rule(x), 'v_body': [0.0, 0.0, 0.0] if rule(x) else [float('nan')] * 3, 'err': None}
    return t


GOOD = (lambda x: 1.0 if x.get('phase') in ('OPEN', 'SETTLE') and x.get('k3m_mode') == 'free' else 0.0)


@unittest.skipUnless(os.path.exists(os.path.join(RUN, 'task.json')), 'no run record')
class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task, cls.room, cls.gate, cls.solver, cls.obs = load()

    def test_j1_regression_8deg(self):
        t = with_guide(self.task, GOOD)
        r = C.judge_with(t, self.room, self.gate, self.solver, self.obs, params(room=self.room))
        ref = C4B2.judge(self.task, self.room, self.gate, self.solver)
        self.assertTrue(ref['pass'])
        self.assertEqual(r['why'], [], r['why'])
        self.assertTrue(r['pass'] and r['w1b_registration_ok'] and r['w1b_guidance_ok'])

    def test_j2_constants_restored(self):
        before = (KC3.CFG, KC3.TCFG, KC3.THETA, KM.ENVELOPE)
        C.judge_with(with_guide(self.task, GOOD), self.room, self.gate, self.solver, self.obs, params(theta=30.0, room=self.room))
        self.assertEqual(before, (KC3.CFG, KC3.TCFG, KC3.THETA, KM.ENVELOPE))
        with self.assertRaises(RuntimeError):
            with C.target_constants(30.0, 1.0, (0, 0, 0, 0, 0)):
                self.assertNotEqual(KC3.THETA, before[2])
                raise RuntimeError('x')
        self.assertEqual(before, (KC3.CFG, KC3.TCFG, KC3.THETA, KM.ENVELOPE))

    def test_j3_registration_mismatch(self):
        t = with_guide(self.task, GOOD)
        for p, w in ((params(theta=9.0, room=self.room), 'w1b_theta_mismatch'),
                     (params(envelope=(-0.2, 0.02, -0.05, 0.10, 6.0), room=self.room), 'w1b_envelope_mismatch'),
                     (params(obs=34.0, room=self.room), 'w1b_observer_limit_mismatch'),
                     (params(sha='0' * 64, room=self.room), 'w1b_corridor_table_mismatch')):
            self.assertIn(w, C.judge_with(t, self.room, self.gate, self.solver, self.obs, p)['why'])

    def test_j4_guidance_phase(self):
        p = params(room=self.room)
        bad = with_guide(self.task, lambda x: 1.0 if x.get('phase') in ('OPEN', 'SETTLE', 'PAUSE') else 0.0)
        why = C.judge_with(bad, self.room, self.gate, self.solver, self.obs, p)['why']
        self.assertTrue(any(w.startswith('guide_phase_violation') for w in why), why)
        never = with_guide(self.task, lambda x: 0.0)
        self.assertIn('guide_never_on', C.judge_with(never, self.room, self.gate, self.solver, self.obs, p)['why'])

    def test_j5_theta_effective(self):
        t = with_guide(self.task, GOOD)
        why = C.judge_with(t, self.room, self.gate, self.solver, self.obs, params(theta=30.0, room=self.room))['why']
        self.assertIn('w1b_theta_mismatch', why)
        self.assertTrue(any(w in why for w in ('not_reached_goal', 'settle_p2_condition_not_held_1s', 'reference_not_at_goal')) or
                        any('goal' in w for w in why), why)


    def test_j6_missing_or_invalid(self):
        p = params(room=self.room)
        t = with_guide(self.task, GOOD)
        t1 = copy.deepcopy(t)
        del t1['config']['theta_goal_deg']
        self.assertIn('w1b_theta_missing_or_invalid', C.judge_with(t1, self.room, self.gate, self.solver, self.obs, p)['why'])
        for key, val, w in (('inner_mm', float('nan'), 'w1b_inner_mm_missing_or_invalid'),
                            ('corridor_table_sha256', 'xyz', 'w1b_corridor_table_missing_or_invalid')):
            r = copy.deepcopy(self.room)
            r['k3m']['config'][key] = val
            self.assertIn(w, C.judge_with(t, r, self.gate, self.solver, self.obs, p)['why'])
        r = copy.deepcopy(self.room)
        r['k3m']['config']['envelope'][2] = float('nan')
        self.assertIn('w1b_envelope_missing_or_invalid', C.judge_with(t, r, self.gate, self.solver, self.obs, p)['why'])
        self.assertIn('w1b_observer_limit_missing_or_invalid', C.judge_with(t, self.room, self.gate, self.solver, {'config': {}}, p)['why'])
        for rule in ((lambda x: float('nan') if x.get('phase') == 'PAUSE' else GOOD(x)),
                     (lambda x: 0.5 if GOOD(x) else 0.0)):
            why = C.judge_with(with_guide(self.task, rule), self.room, self.gate, self.solver, self.obs, p)['why']
            self.assertTrue(any(w_.startswith('guide_phase_violation') for w_ in why), why)
        t2 = with_guide(self.task, GOOD)
        for x in t2['trace']:
            if x['guide']['w']:
                x['guide']['v_body'] = [0.0, float('nan'), 0.0]
                break
        why = C.judge_with(t2, self.room, self.gate, self.solver, self.obs, p)['why']
        self.assertTrue(any(w_.startswith('guide_phase_violation') for w_ in why), why)

if __name__ == '__main__':
    unittest.main(verbosity=2)
