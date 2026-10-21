#!/usr/bin/env python3
"""door_k2_check 的離線反例（合成但物理一致：HOLD／退離的手臂姿態由 IK 求，抓取關係由 FK 算）。

    python3 evaluation/test_door_k2_check.py
"""
import copy
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402
import door_k2_policy as KP                            # noqa: E402
import door_k2_offline_v2 as V2                        # noqa: E402

COLS = ['step', 'sim_t', 'owner', 'kind', 'base_cmd_body', 'meas_vb_body', 'applied_arm', 'base_xyth', 'q_arm_meas', 'door_q_rad',
        'door_qdot_rad_s', 'q_finger', 'finger_contact_n', 'contact_held_s', 'leaf_contacts', 'wall_contacts', 'wrist_force_n', 'park_mode']
PH = [('UNFOLD', 1.0), ('ALIGN_PRE', 8.0), ('ALIGN_HOLD', 10.0), ('APPROACH', 11.0), ('ENGAGE', 16.0), ('HOLD', 19.5), ('RELEASE', 22.6),
      ('RETREAT', 23.5), ('DONE', 28.6)]
BASE = list(V2.base_of(0, 0, 0))
GB = json.load(open(os.path.join(HERE, 'runs', 'phf_01_M', 'task.json')))['args']
T_HG = np.eye(4)
T_HG[:3, :3] = np.asarray(GB['R_grasp'], float)
T_HG[:3, 3] = [0.0, float(GB['grasp_depth_m']), 0.0]
H0 = KC._H_of(0.0)
POL = KP.K2Policy(KP.K2Config(), T_HG)
Q_HOLD = V2.C.ik(np.array(BASE), POL.target(H0, 0.0), None)[V2.C.IDX]
Q_RET = V2.C.ik(np.array(BASE), POL.target(H0, 0.10), Q_HOLD, only_prev=True)[V2.C.IDX]
G_T_H = np.linalg.inv(KC._tcp(BASE, Q_HOLD)) @ H0


def ph_of(t):
    cur = None
    for p, t0 in PH:
        if t >= t0:
            cur = p
    return cur


def good():
    ev = [{'t': t, 'from': 'x', 'to': p} for p, t in PH]
    task = {'schema': 'door_k2_task/1', 'policy_events': ev, 'abort': None, 'final_phase': 'DONE', 'received_any_sync': True,
            'trace': [{'t': 28.7}], 'G_T_H': G_T_H.tolist(), 'H_frozen': H0.tolist(), 'grasp_basis': {'T_HG': T_HG.tolist()}}
    steps, held = [], 0.0
    for k in range(0, 2880):
        t = round(k * 0.01, 4)
        ph = ph_of(t)
        closed = ph in ('HOLD',) or (ph == 'ENGAGE' and t >= 16.5) or (ph == 'RELEASE' and t < 22.8)
        fb = 2.0 if closed else 0.0
        held = held + 0.01 if closed else 0.0
        qa = Q_RET if ph in (None, 'UNFOLD', 'ALIGN_PRE', 'ALIGN_HOLD') else (Q_RET if (ph in ('RETREAT', 'DONE') and t >= 27.0) else Q_HOLD)
        if ph == 'RETREAT' and t < 27.0:
            a = (t - 23.5) / 3.5
            qa = (1 - a) * Q_HOLD + a * Q_RET
        qf = [0.0, 0.0] if closed else [0.0089, 0.0089]
        lc = {'uflite_finger1': {'bar': fb, 'ambiguous': 0.0, 'panel': 0.0}, 'uflite_finger2': {'bar': fb, 'ambiguous': 0.0, 'panel': 0.0}} if fb else {}
        steps.append([k, t, 1, 'wholebody_applied', [0, 0, 0], [0, 0, 0], [0] * 6, list(BASE), list(map(float, qa)), 0.0, 0.0,
                      qf, [fb, fb], held, lc, {}, 3.0, 'hold'])
    room = {'steps_cols': COLS, 'steps': steps}
    gate = {'gate': {'anchor': BASE}}
    return task, room, gate


class Judge(unittest.TestCase):
    def J(self, *a):
        return KC.judge(*a)

    def test_good_passes(self):
        res = self.J(*good())
        self.assertTrue(res['pass'], res['why'])

    def fails_with(self, f, why):
        t, r, g = good()
        f(t, r, g)
        res = self.J(t, r, g)
        self.assertFalse(res['pass'])
        self.assertIn(why, ' '.join(res['why']), res['why'])

    # ---- Codex 的五個誤收反例
    def test_record_truncated_before_hold(self):
        def f(t, r, g):
            r['steps'] = [s for s in r['steps'] if s[1] < 19.0]
        self.fails_with(f, 'record_does_not_cover_task_window')

    def test_wall_nan(self):
        def f(t, r, g):
            for s in r['steps']:
                s[15] = {'link4': float('nan')}
        self.fails_with(f, 'missing_or_nonfinite')

    def test_no_grasp_evidence_no_finger_pos(self):
        def f(t, r, g):
            for s in r['steps']:
                s[13] = 0.0
                s[11] = None
        res_t = good()
        f(*res_t)
        res = self.J(*res_t)
        self.assertFalse(res['pass'])
        self.assertTrue({'grasp_evidence_missing', 'missing_or_nonfinite'} & set(res['why']), res['why'])

    def test_aborted_and_wall_timeout(self):
        def f(t, r, g):
            t['final_phase'] = 'ABORTED'
            t['wall_timeout'] = True
        self.fails_with(f, 'termination')

    def test_door_rate_high(self):
        def f(t, r, g):
            for s in r['steps']:
                s[10] = 0.08
        self.fails_with(f, 'door_rate_over_0.05')

    # ---- 其他
    def test_release_fingers_not_open(self):
        def f(t, r, g):
            for s in r['steps']:
                if 23.0 <= s[1] < 23.5:
                    s[11] = [0.0089, 0.002]
        self.fails_with(f, 'release_not_confirmed')

    def test_hold_drift(self):
        def f(t, r, g):
            for s in r['steps']:
                if 20.0 <= s[1] < 21.0:
                    s[9] = math.radians(0.6)          # 門角 0.6°（< 1°）但把手移 ~8 mm，TCP 不動 ⇒ 漂移
        self.fails_with(f, 'hold_drift_over_5mm')

    def test_retreat_not_reached(self):
        def f(t, r, g):
            for s in r['steps']:
                if s[1] >= 27.0:
                    s[8] = list(map(float, Q_HOLD))
        self.fails_with(f, 'retreat_not_reached')

    def test_missing_G_T_H(self):
        def f(t, r, g):
            t['G_T_H'] = None
        self.fails_with(f, 'hold_drift_unverifiable')

    def test_ambiguous_counts_unexpected(self):
        def f(t, r, g):
            for s in r['steps']:
                if 20.0 <= s[1] < 20.05:
                    s[14]['uflite_finger1']['ambiguous'] = 0.3
        self.fails_with(f, 'unexpected_contact')

    def test_finger_panel(self):
        def f(t, r, g):
            for s in r['steps']:
                if 12.0 <= s[1] < 12.05:
                    s[14] = {'uflite_finger1': {'bar': 0.0, 'ambiguous': 0.0, 'panel': 0.3}}
        self.fails_with(f, 'unexpected_contact')

    def test_wrist(self):
        def f(t, r, g):
            r['steps'][2000][16] = 31.0
        self.fails_with(f, 'wrist_force')

    def test_base_drift(self):
        def f(t, r, g):
            r['steps'][1500][7] = [BASE[0] + 0.025, BASE[1], BASE[2]]
        self.fails_with(f, 'base_drift')

    def test_step_gap(self):
        def f(t, r, g):
            del r['steps'][1500]
        self.fails_with(f, 'step_discontinuity')

    def test_no_anchor(self):
        t, r, g = good()
        self.assertIn('park_anchor_missing', self.J(t, r, None)['why'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
