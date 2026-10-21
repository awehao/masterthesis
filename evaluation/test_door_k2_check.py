#!/usr/bin/env python3
"""door_k2_check 的離線反例（合成 task.json／room_run.json／park_gate.json）。

    python3 evaluation/test_door_k2_check.py
"""
import copy
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402

COLS = ['step', 'sim_t', 'owner', 'kind', 'base_cmd_body', 'meas_vb_body', 'applied_arm', 'base_xyth', 'q_arm_meas', 'door_q_rad',
        'door_qdot_rad_s', 'q_finger', 'finger_contact_n', 'contact_held_s', 'leaf_contacts', 'wall_contacts', 'wrist_force_n', 'park_mode']
PH = [('UNFOLD', 1.0), ('ALIGN_PRE', 8.0), ('ALIGN_HOLD', 10.0), ('APPROACH', 11.0), ('ENGAGE', 16.0), ('HOLD', 19.5), ('RELEASE', 22.6),
      ('RETREAT', 23.5), ('DONE', 28.6)]
ANCHOR = [0.6, -0.5332, 1.570796]


def good():
    ev = [{'t': t, 'from': 'x', 'to': p} for p, t in PH]
    task = {'schema': 'door_k2_task/1', 'policy_events': ev, 'abort': None, 'trace': [{'t': 29.0}]}
    steps = []
    for k in range(0, 3000):
        t = k * 0.01
        ph = None
        for (p, t0), nxt in zip(PH, PH[1:] + [('END', 1e9)]):
            if t0 <= t < nxt[1]:
                ph = p
        fb = 2.0 if ph in ('HOLD',) or (ph == 'ENGAGE' and t > 17.0) else 0.0
        lc = {'uflite_finger1': {'bar': fb, 'panel': 0.0}, 'uflite_finger2': {'bar': fb, 'panel': 0.0}} if fb else {}
        steps.append([k, t, 1, 'wholebody_applied', [0, 0, 0], [0, 0, 0], [0] * 6, list(ANCHOR), [0] * 6, 0.0005, 0.0,
                      [0.0, 0.0], [fb, fb], 0.0, lc, {}, 3.0, 'hold'])
    room = {'steps_cols': COLS, 'steps': steps}
    gate = {'gate': {'anchor': ANCHOR}}
    return task, room, gate


class Judge(unittest.TestCase):
    def J(self, t, r, g):
        return KC.judge(t, r, g)

    def test_good_passes(self):
        res = self.J(*good())
        self.assertTrue(res['pass'], res['why'])

    def mutate_steps(self, f, why):
        t, r, g = good()
        for s in r['steps']:
            f(s)
        res = self.J(t, r, g)
        self.assertFalse(res['pass'])
        self.assertIn(why, res['why'])

    def test_door_over_1deg(self):
        def f(s):
            if 20.0 < s[1] < 20.5:
                s[9] = math.radians(1.5)
        self.mutate_steps(f, 'door_angle_over_1deg')

    def test_finger_touches_panel(self):
        def f(s):
            if 12.0 < s[1] < 12.1:
                s[14] = {'uflite_finger1': {'bar': 0.0, 'panel': 0.3}}
        self.mutate_steps(f, 'unexpected_contact')

    def test_hold_one_finger_low(self):
        def f(s):
            if 20.0 < s[1] < 20.3:
                s[14] = {'uflite_finger1': {'bar': 2.0, 'panel': 0.0}, 'uflite_finger2': {'bar': 0.2, 'panel': 0.0}}
        self.mutate_steps(f, 'hold_finger_contact_below_min')

    def test_finger_bar_after_release(self):
        def f(s):
            if 24.0 < s[1] < 24.1:
                s[14] = {'uflite_finger2': {'bar': 0.3, 'panel': 0.0}}
        self.mutate_steps(f, 'finger_bar_outside_allowed_phase')

    def test_wall_contact(self):
        def f(s):
            if 5.0 < s[1] < 5.1:
                s[15] = {'link4': 0.4}
        self.mutate_steps(f, 'wall_contact')

    def test_wrist(self):
        def f(s):
            if 20.0 < s[1] < 20.02:
                s[16] = 31.0
        self.mutate_steps(f, 'wrist_force')

    def test_base_drift(self):
        def f(s):
            if 15.0 < s[1] < 15.1:
                s[7] = [0.6 + 0.025, -0.5332, 1.570796]
        self.mutate_steps(f, 'base_drift')

    def test_missing_contacts_fails(self):
        def f(s):
            if 13.0 < s[1] < 13.02:
                s[14] = None
        self.mutate_steps(f, 'missing_or_nonfinite')

    def test_nonfinite_wrist_fails(self):
        def f(s):
            if 13.0 < s[1] < 13.02:
                s[16] = float('nan')
        self.mutate_steps(f, 'missing_or_nonfinite')

    def test_step_gap(self):
        t, r, g = good()
        del r['steps'][1500]
        self.assertIn('step_discontinuity', self.J(t, r, g)['why'])

    def test_abort_fails(self):
        t, r, g = good()
        t['abort'] = 'door_moving'
        self.assertFalse(self.J(t, r, g)['pass'])

    def test_sequence_missing_release(self):
        t, r, g = good()
        t['policy_events'] = [e for e in t['policy_events'] if e['to'] != 'RELEASE']
        self.assertFalse(self.J(t, r, g)['pass'])

    def test_hold_too_short(self):
        t, r, g = good()
        for e in t['policy_events']:
            if e['to'] == 'RELEASE':
                e['t'] = 21.0
        self.assertIn('hold_too_short_or_missing', self.J(t, r, g)['why'])

    def test_no_anchor(self):
        t, r, g = good()
        self.assertIn('park_anchor_missing', self.J(t, r, None)['why'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
