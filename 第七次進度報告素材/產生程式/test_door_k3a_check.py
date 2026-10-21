#!/usr/bin/env python3
"""door_k3a_check 的離線反例（合成但物理一致：門角剖面由 DoorTargetK3 理想跟隨產生；手臂姿態由 IK 隨門角求解後內插；
抓取關係由 FK 算；trace 的參考角與目標由實際 DoorTargetK3 逐步產生）。

    python3 evaluation/test_door_k3a_check.py
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
import door_k3a_check as KC3                           # noqa: E402
import door_k2_check as KC                             # noqa: E402
import door_k3a_policy as KP                           # noqa: E402
import door_target_k3 as T3                            # noqa: E402
import door_k2_offline_v2 as V2                        # noqa: E402

COLS = ['step', 'sim_t', 'owner', 'kind', 'base_cmd_body', 'meas_vb_body', 'applied_arm', 'base_xyth', 'q_arm_meas', 'door_q_rad',
        'door_qdot_rad_s', 'q_finger', 'finger_contact_n', 'contact_held_s', 'leaf_contacts', 'wall_contacts', 'wrist_force_n', 'park_mode']
T_OPEN = 20.5
BASE = list(V2.base_of(0, 0, 0))
GB = json.load(open(os.path.join(HERE, 'runs', 'phf_01_M', 'task.json')))['args']
T_HG = np.eye(4)
T_HG[:3, :3] = np.asarray(GB['R_grasp'], float)
T_HG[:3, 3] = [0.0, float(GB['grasp_depth_m']), 0.0]
POL = KP.K3APolicy(KP.K3AConfig(), T_HG)
H0 = KC._H_of(0.0)
T_HOLD = POL.target(H0, 0.0)

# 參考角剖面（20 Hz、理想跟隨）
_cfg = KP.K3AConfig()
_dt = T3.DoorTargetK3(T_HOLD, H0, _cfg.target_cfg())
REF = []                                   # (t, q_ref, target)
t = T_OPEN
while True:
    q_meas = REF[-1][1] if REF else 0.0
    Tg, info = _dt.step(KC._H_of(q_meas), q_meas, 0.05)
    REF.append((round(t, 4), _dt.q_ref, Tg, q_meas))
    if _dt.reached:
        break
    t += 0.05
T_SETTLE = round(REF[-1][0] + 0.05, 4)
T_END = round(T_SETTLE + 1.6, 4)
T_DONE = round(T_END + 3.0, 4)
PH = [('UNFOLD', 1.0), ('ALIGN_PRE', 8.0), ('ALIGN_HOLD', 10.0), ('APPROACH', 11.0), ('ENGAGE', 16.0), ('HOLD', 19.5), ('OPEN', T_OPEN),
      ('SETTLE', T_SETTLE), ('END_HOLD', T_END), ('DONE', T_DONE)]

# 手臂 IK（門角每 0.25°）
_Q = {}
_prev = V2.C.ik(np.array(BASE), POL.target(H0, 0.10), None)[V2.C.IDX]
_prev = V2.C.ik(np.array(BASE), POL.target(H0, 0.0), _prev, only_prev=True)[V2.C.IDX]
Q_HOLD = _prev.copy()
for k in range(0, 33):
    qd = math.radians(0.25 * k)
    _prev = V2.C.ik(np.array(BASE), POL.target(KC._H_of(qd), 0.0), _prev, only_prev=True)[V2.C.IDX]
    _Q[k] = _prev.copy()
Q_PRE = V2.C.ik(np.array(BASE), POL.target(H0, 0.10), Q_HOLD, only_prev=True)[V2.C.IDX]
G_T_H = np.linalg.inv(KC._tcp(BASE, Q_HOLD)) @ H0


def q_arm_of(qd):
    x = math.degrees(qd) / 0.25
    k = max(0, min(31, int(math.floor(x))))
    a = x - k
    return (1 - a) * _Q[k] + a * _Q[k + 1]


def door_q(t):
    if t < T_OPEN:
        return 0.0
    if t >= REF[-1][0]:
        return REF[-1][1]
    k = int((t - T_OPEN) / 0.05)
    a = (t - REF[k][0]) / 0.05
    return (1 - a) * REF[k][1] + a * REF[k + 1][1]


def ph_of(t):
    cur = None
    for p, t0 in PH:
        if t >= t0:
            cur = p
    return cur


def good():
    ev = [{'t': t0, 'from': 'x', 'to': p} for p, t0 in PH]
    trace = []
    tt = 19.5
    while tt < T_OPEN - 1e-3 - 1e-9:
        trace.append({'t': round(tt, 4), 'phase': 'HOLD', 'q_door': 0.0, 'q_ref': None, 'target': T_HOLD.reshape(-1).tolist()})
        tt += 0.05
    trace.append({'t': T_OPEN - 1e-3, 'phase': 'OPEN', 'q_door': 0.0, 'q_ref': None, 'target': T_HOLD.reshape(-1).tolist()})
    for t_, qr, Tg, qm in REF:
        trace.append({'t': t_, 'phase': 'OPEN', 'q_door': qm, 'q_ref': qr, 'target': Tg.reshape(-1).tolist()})
    tt = T_SETTLE
    while tt < T_END - 1e-9:
        trace.append({'t': round(tt, 4), 'phase': 'SETTLE', 'q_door': REF[-1][1], 'q_ref': REF[-1][1], 'target': REF[-1][2].reshape(-1).tolist()})
        tt += 0.05
    while tt < T_DONE - 1e-9:
        trace.append({'t': round(tt, 4), 'phase': 'END_HOLD', 'q_door': REF[-1][1], 'q_ref': REF[-1][1], 'target': REF[-1][2].reshape(-1).tolist()})
        tt += 0.05
    trace.append({'t': T_DONE, 'phase': 'DONE', 'q_door': REF[-1][1], 'q_ref': REF[-1][1], 'target': None})
    n = 0
    for e in trace:
        e['grip'] = 0.0
        if e['target'] is not None:
            n += 1
            e['pub_seq'] = n
        else:
            e['pub_seq'] = None
    task = {'schema': 'door_k3a_task/1', 'policy_events': ev, 'abort': None, 'final_phase': 'DONE', 'received_any_sync': True,
            'trace': trace, 'n_target_published': n, 'G_T_H': G_T_H.tolist(), 'H_frozen': H0.tolist(), 'grasp_basis': {'T_HG': T_HG.tolist()}}
    steps, held, qprev = [], 0.0, 0.0
    for k in range(0, int((T_DONE + 0.5) / 0.01)):
        t = round(k * 0.01, 4)
        ph = ph_of(t)
        closed = ph in ('HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE') or (ph == 'ENGAGE' and t >= 16.5)
        fb = 2.0 if closed else 0.0
        held = held + 0.01 if closed else 0.0
        q = door_q(t)
        qa = Q_PRE if ph in (None, 'UNFOLD', 'ALIGN_PRE', 'ALIGN_HOLD') else (Q_HOLD if ph in ('APPROACH', 'ENGAGE', 'HOLD') else q_arm_of(q))
        qf = [0.0, 0.0] if closed else [0.0089, 0.0089]
        lc = {'uflite_finger1': {'bar': fb, 'ambiguous': 0.0, 'panel': 0.0}, 'uflite_finger2': {'bar': fb, 'ambiguous': 0.0, 'panel': 0.0}} if fb else {}
        steps.append([k, t, 1, 'wholebody_applied', [0, 0, 0], [0, 0, 0], [0] * 6, list(BASE), list(map(float, qa)), q, (q - qprev) / 0.01,
                      qf, [fb, fb], held, lc, {}, 3.0, 'hold'])
        qprev = q
    return task, {'steps_cols': COLS, 'steps': steps}, {'gate': {'anchor': BASE}}


_GOOD = good()


def G():
    return copy.deepcopy(_GOOD)


def tr_open(task):
    return [e for e in task['trace'] if e.get('phase') in ('OPEN', 'SETTLE') and e.get('q_ref') is not None]


class Judge(unittest.TestCase):
    def fails_with(self, f, why):
        t, r, g = G()
        f(t, r, g)
        res = KC3.judge(t, r, g)
        self.assertFalse(res['pass'])
        self.assertIn(why, ' '.join(res['why']), res['why'])

    def test_good_passes(self):
        res = KC3.judge(*G())
        self.assertTrue(res['pass'], (res['why'], res['stats']))
        self.assertLess(res['stats']['max_drift_m'], 0.002)

    def test_release_phase_present(self):
        def f(t, r, g):
            t['policy_events'].insert(-1, {'t': T_DONE - 0.5, 'from': 'END_HOLD', 'to': 'RELEASE'})
        self.fails_with(f, 'phase_sequence')

    def test_reference_over_goal(self):
        def f(t, r, g):
            e = tr_open(t)[-1]
            e['q_ref'] = KC3.THETA + 1e-4
        self.fails_with(f, 'reference_over_goal')

    def test_reference_rate(self):
        def f(t, r, g):
            o = tr_open(t)
            o[40]['q_ref'] += math.radians(0.2)
        self.fails_with(f, 'reference_rate_violated')

    def test_reference_lead(self):
        def f(t, r, g):
            o = tr_open(t)
            o[60]['q_door'] = o[60]['q_ref'] - math.radians(0.6)
        self.fails_with(f, 'reference_lead_violated')

    def test_open_first_discontinuous(self):
        def f(t, r, g):
            e = tr_open(t)[0]
            T = np.asarray(e['target']).reshape(4, 4)
            T[0, 3] += 0.001
            e['target'] = T.reshape(-1).tolist()
        self.fails_with(f, 'open_first_target_discontinuous')

    def test_target_not_rotation(self):
        def f(t, r, g):
            e = tr_open(t)[50]
            T = np.asarray(e['target']).reshape(4, 4)
            T[2, 3] += 0.0005
            e['target'] = T.reshape(-1).tolist()
        self.fails_with(f, 'target_not_rigid_rotation_of_first')

    def test_not_reached(self):
        def f(t, r, g):
            for e in t['trace']:
                if e.get('phase') == 'SETTLE':
                    e['q_ref'] = REF[-3][1]
                    e['target'] = REF[-3][2].reshape(-1).tolist()
        self.fails_with(f, 'reference_not_reached')

    def test_door_overshoot(self):
        def f(t, r, g):
            for s in r['steps']:
                if T_END + 1.0 <= s[1] < T_END + 1.02:
                    s[9] = math.radians(9.2)
        self.fails_with(f, 'door_angle_out_of_bounds')

    def test_settle_not_quiet(self):
        def f(t, r, g):
            for s in r['steps']:
                if T_END - 0.3 <= s[1] < T_END - 0.29:
                    s[10] = 0.003
        self.fails_with(f, 'settle_condition_not_held_1s')

    def test_finger_dip_open(self):
        def f(t, r, g):
            for s in r['steps']:
                if 24.0 <= s[1] < 24.01:
                    s[14]['uflite_finger1']['bar'] = 0.3
        self.fails_with(f, 'grasp_finger_contact_below_min')

    def test_drift_open(self):
        def f(t, r, g):
            for s in r['steps']:
                if 25.0 <= s[1] < 25.1:
                    s[9] += math.radians(0.6)                              # 把手移 ~8 mm、手臂不動
        self.fails_with(f, 'grasp_drift_over_limit')

    def test_end_hold_short(self):
        def f(t, r, g):
            t['policy_events'][-1]['t'] = T_END + 2.0
        self.fails_with(f, 'end_hold_too_short_or_missing')

    def test_truncated(self):
        def f(t, r, g):
            r['steps'] = [s for s in r['steps'] if s[1] < T_END]
        self.fails_with(f, 'record_does_not_cover_task_window')

    def test_reference_trace_missing(self):
        def f(t, r, g):
            for e in t['trace']:
                e['q_ref'] = None
        self.fails_with(f, 'reference_trace_missing')

    def test_pre_open_door(self):
        def f(t, r, g):
            for s in r['steps']:
                if 18.0 <= s[1] < 18.01:
                    s[9] = math.radians(1.2)
        self.fails_with(f, 'door_angle_pre_open_over_1deg')

    def test_wrist(self):
        def f(t, r, g):
            r['steps'][2500][16] = 31.0
        self.fails_with(f, 'wrist_force')

    def test_ambiguous(self):
        def f(t, r, g):
            r['steps'][2600][14]['uflite_finger2']['ambiguous'] = 0.2
        self.fails_with(f, 'unexpected_contact')

    def test_aborted(self):
        def f(t, r, g):
            t['abort'] = 'door_target_conflict'
            t['final_phase'] = 'ABORTED'
        self.fails_with(f, 'aborted')

    def test_finger_bar_before_engage(self):
        def f(t, r, g):
            r['steps'][1200][14] = {'uflite_finger1': {'bar': 0.3, 'ambiguous': 0.0, 'panel': 0.0}}
        self.fails_with(f, 'finger_bar_outside_allowed_phase')

    def test_nonfinite(self):
        def f(t, r, g):
            r['steps'][2700][10] = float('nan')
        self.fails_with(f, 'missing_or_nonfinite')

    # ---- Codex 20261021 K3-A 實作審查的三個誤收反例
    def test_fingers_open_after_done(self):
        def f(t, r, g):
            for s_ in r['steps']:
                if s_[1] >= T_DONE + 0.1:
                    s_[11] = [0.0089, 0.0089]
                    s_[12] = [0.0, 0.0]
                    s_[14] = {}
        res = KC3.judge(*self._mut(f))
        self.assertFalse(res['pass'])
        self.assertTrue({'grasp_finger_contact_below_min', 'fingers_opened_after_grasp'} <= set(res['why']), res['why'])

    def test_end_hold_grip_open_and_target_moving(self):
        def f(t, r, g):
            for e in t['trace']:
                if e['phase'] == 'END_HOLD':
                    e['grip'] = 0.0089
                    T = np.asarray(e['target']).reshape(4, 4)
                    T[1, 3] -= 0.01 * (e['t'] - T_END)
                    e['target'] = T.reshape(-1).tolist()
        res = KC3.judge(*self._mut(f))
        self.assertFalse(res['pass'])
        self.assertTrue({'grip_command_not_closed_after_grasp', 'end_hold_target_not_fixed'} <= set(res['why']), res['why'])

    def test_reference_trace_thinned_to_two_rows(self):
        def f(t, r, g):
            o = [e for e in t['trace'] if e['phase'] in ('OPEN', 'SETTLE') and e.get('q_ref') is not None]
            keep = {id(o[0]), id(o[-1])}
            t['trace'] = [e for e in t['trace'] if not (e['phase'] in ('OPEN', 'SETTLE') and e.get('q_ref') is not None) or id(e) in keep]
        res = KC3.judge(*self._mut(f))
        self.assertFalse(res['pass'])
        self.assertTrue({'publish_sequence_mismatch', 'trace_gap'} & set(res['why']), res['why'])

    def test_reference_row_missing_q_ref(self):
        def f(t, r, g):
            o = [e for e in t['trace'] if e['phase'] == 'OPEN' and e.get('q_ref') is not None]
            o[30]['q_ref'] = None
        self.fails_with(f, 'reference_evidence_incomplete')

    def test_publish_count_mismatch(self):
        def f(t, r, g):
            t['n_target_published'] += 3
        self.fails_with(f, 'publish_sequence_mismatch')

    def test_end_hold_rows_missing(self):
        def f(t, r, g):
            t['trace'] = [e for e in t['trace'] if e['phase'] != 'END_HOLD']
        res = KC3.judge(*self._mut(f))
        self.assertFalse(res['pass'])
        self.assertIn('end_hold_evidence_missing', res['why'])

    def test_grip_missing(self):
        def f(t, r, g):
            t['trace'][-5].pop('grip')
        self.fails_with(f, 'grip_command_not_closed_after_grasp')

    def test_drift_after_done(self):
        def f(t, r, g):
            for s_ in r['steps']:
                if s_[1] >= T_DONE + 0.2:
                    s_[9] += math.radians(0.6)
        self.fails_with(f, 'grasp_drift_over_limit')

    def test_policy_trace_format_accepted(self):
        """相位機合成世界跑一趟，依任務節點格式組 trace：判定器的參考角與 trace 對帳應無誤（格式假設一致）。"""
        import test_door_k3a_policy as TP
        sim = TP.Sim()
        sim.run()
        pol = sim.pol
        self.assertEqual(pol.phase, 'DONE')
        tr, n = [], 0
        for t_, ph, a, m in sim.acts:
            tgt = None if (a['target'] is None or ph == 'ABORTED') else np.asarray(a['target']).reshape(-1).tolist()
            seq = None
            if tgt is not None:
                n += 1
                seq = n
            tr.append({'t': t_, 'phase': ph, 'q_door': m.q_door, 'q_ref': None, 'target': tgt,
                       'grip': a['grip'], 'pub_seq': seq})
        # q_ref 依節點：記錄「該輪 step 之後」的 pol.q_ref ⇒ 重跑一次逐輪擷取
        sim2 = TP.Sim()
        qrefs = []
        orig = sim2.pol.step

        def wrap(m, dt):
            a = orig(m, dt)
            qrefs.append(sim2.pol.q_ref)
            return a
        sim2.pol.step = wrap
        sim2.run()
        self.assertEqual(len(qrefs), len(tr))
        for e, q in zip(tr, qrefs):
            e['q_ref'] = q
        task = {'trace': tr, 'policy_events': pol.events, 'n_target_published': n}
        W = KC.phase_windows(task)
        why, st = [], {}
        KC3.check_reference(task, why, st)
        KC3.check_trace(task, W, why, st)
        self.assertEqual(why, [], (why, st))
        self.assertLess(st['open_first_vs_last_hold_m'], 1e-12)

    def _mut(self, f):
        t, r, g = G()
        f(t, r, g)
        return t, r, g

    def test_G_T_H_invalid(self):
        def f(t, r, g):
            t['G_T_H'][0][0] *= 1.1
        self.fails_with(f, 'grasp_matrices_missing_or_invalid')


if __name__ == '__main__':
    unittest.main(verbosity=2)
