#!/usr/bin/env python3
"""door_k3au_check.judge（**完整判定**）的合成實錄與反例（依 Codex 20261022_122517：反例須打在完整 judge 上）。

合成實錄由 test_door_k3a_check 的 K3-A 實錄改出：HOLD 內加入接近軸參考卸載。
  - H_frozen＝FK(Q_HOLD)·T_HG⁻¹ ⇒ T_ref0＝FK(Q_HOLD)（避開 IK 收斂極限 4.5e-5 m）；
  - 非零介入：HOLD 期間手臂停在 IK(T_ref0 − 1.05 mm·a)，δ 由擷取步的 FK 投影而得；零介入：手臂停在 Q_HOLD，δ ≈ 0；
  - 已發布目標依卸載公式；OPEN 參考角由卸載後目標建立的 DoorTargetK3 逐步產生；擷取事件 T_tcp＝該物理步 FK。
正常通過：非零介入（擷取 τ＝0.5 s）、零介入、延後擷取（τ＝1.2 s）。反例：Codex 的四項誤收＋時序與快照的其他竄改。

    python3 evaluation/test_door_k3au_judge.py
"""
import copy
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
import door_k3au_check as UC                          # noqa: E402
import door_k3au_policy as U                          # noqa: E402
import door_target_k3 as T3                           # noqa: E402
import test_door_k3a_check as B                       # noqa: E402

V2 = B.V2
T_HOLD_FK = KC._tcp(B.BASE, B.Q_HOLD)
H_FROZEN = T_HOLD_FK @ np.linalg.inv(B.T_HG)
A = T_HOLD_FK[:3, 2] / np.linalg.norm(T_HOLD_FK[:3, 2])
_T_out = T_HOLD_FK.copy()
_T_out[:3, 3] -= A * 0.00105
Q_OUT = V2.C.ik(np.array(B.BASE), _T_out, B.Q_HOLD, only_prev=True)[V2.C.IDX]
DT_S = 0.05
T_ENG, T_HOLD = 16.0, 19.5


def build(tau_cap=0.5, zero=False):
    q_hold_arm = B.Q_HOLD if zero else Q_OUT
    T0 = T_HOLD_FK
    t_cap = round(T_HOLD + tau_cap, 4)
    k_cap = int(round(t_cap / 0.01))
    T_cap_fk = KC._tcp(B.BASE, q_hold_arm)
    delta = float(A @ (T_cap_fk[:3, 3] - T0[:3, 3]))
    if zero:
        delta = min(delta, 0.0)
    t_ramp_end = round(t_cap + 0.5, 4)
    t_open = round(t_cap + 1.0, 4)
    T_unl = T0.copy()
    T_unl[:3, 3] = T0[:3, 3] + A * delta
    cfg = U.K3AUConfig()
    dtk = T3.DoorTargetK3(T_unl, B.H0, cfg.target_cfg())
    REF, t = [], t_open
    while True:
        q_meas = REF[-1][1] if REF else 0.0
        Tg, _ = dtk.step(KC._H_of(q_meas), q_meas, DT_S)
        REF.append((round(t, 4), dtk.q_ref, Tg, q_meas))
        if dtk.reached:
            break
        t += DT_S
    t_settle = round(REF[-1][0] + DT_S, 4)
    t_end = round(t_settle + 1.6, 4)
    t_done = round(t_end + 3.0, 4)
    PH = [('UNFOLD', 1.0), ('ALIGN_PRE', 8.0), ('ALIGN_HOLD', 10.0), ('APPROACH', 11.0), ('ENGAGE', T_ENG), ('HOLD', T_HOLD),
          ('OPEN', t_open), ('SETTLE', t_settle), ('END_HOLD', t_end), ('DONE', t_done)]

    def ph_of(tt):
        cur = None
        for p_, t0 in PH:
            if tt >= t0:
                cur = p_
        return cur

    def unl(tt):
        if tt < t_cap - 1e-9:
            return T0
        T = T0.copy()
        T[:3, 3] = T0[:3, 3] + A * delta * U.smoothstep((tt - t_cap) / 0.5)
        return T

    def door_q(tt):
        if tt < t_open:
            return 0.0
        if tt >= REF[-1][0]:
            return REF[-1][1]
        k = int((tt - t_open) / DT_S)
        a_ = (tt - REF[k][0]) / DT_S
        return (1 - a_) * REF[k][1] + a_ * REF[k + 1][1]
    trace = []
    tt = T_HOLD
    while tt < t_open - 1e-9:
        trace.append({'t': round(tt, 4), 'phase': 'HOLD', 'q_door': 0.0, 'q_ref': None, 'target': unl(tt).reshape(-1).tolist()})
        tt = round(tt + DT_S, 4)
    trace.append({'t': t_open - 1e-3, 'phase': 'OPEN', 'q_door': 0.0, 'q_ref': None, 'target': T_unl.reshape(-1).tolist()})
    for t_, qr, Tg, qm in REF:
        trace.append({'t': t_, 'phase': 'OPEN', 'q_door': qm, 'q_ref': qr, 'target': Tg.reshape(-1).tolist()})
    tt = t_settle
    while tt < t_end - 1e-9:
        trace.append({'t': round(tt, 4), 'phase': 'SETTLE', 'q_door': REF[-1][1], 'q_ref': REF[-1][1], 'target': REF[-1][2].reshape(-1).tolist()})
        tt = round(tt + DT_S, 4)
    while tt < t_done - 1e-9:
        trace.append({'t': round(tt, 4), 'phase': 'END_HOLD', 'q_door': REF[-1][1], 'q_ref': REF[-1][1], 'target': REF[-1][2].reshape(-1).tolist()})
        tt = round(tt + DT_S, 4)
    trace.append({'t': t_done, 'phase': 'DONE', 'q_door': REF[-1][1], 'q_ref': REF[-1][1], 'target': None})
    n = 0
    for e in trace:
        e['grip'] = 0.0
        if e['target'] is not None:
            n += 1
            e['pub_seq'] = n
        else:
            e['pub_seq'] = None
    ev = [{'t': t0, 'from': 'x', 'to': p_} for p_, t0 in PH]
    cap = {'t': t_cap, 'step': k_cap, 'T_tcp': T_cap_fk.tolist(), 'T_ref0': T0.tolist(), 'a': A.tolist(), 'delta': delta}
    ev += [{'t': t_cap, 'unload_capture': cap}, {'t': t_ramp_end, 'unload_ramp_end': True}, {'t': t_open, 'unload_hold_end': True}]
    ev.sort(key=lambda e: e['t'])
    G_T_H = np.linalg.inv(KC._tcp(B.BASE, B.Q_HOLD)) @ B.H0
    task = {'schema': 'door_k3au_task/1', 'policy_events': ev, 'abort': None, 'final_phase': 'DONE', 'received_any_sync': True,
            'trace': trace, 'n_target_published': n, 'G_T_H': G_T_H.tolist(), 'H_frozen': H_FROZEN.tolist(), 'grasp_basis': {'T_HG': B.T_HG.tolist()}}
    steps, held, qprev = [], 0.0, 0.0
    for k in range(0, int((t_done + 0.5) / 0.01)):
        t = round(k * 0.01, 4)
        ph = ph_of(t)
        closed = ph in ('HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE') or (ph == 'ENGAGE' and t >= 16.5)
        fb = 2.0 if closed else 0.0
        held = held + 0.01 if closed else 0.0
        q = door_q(t)
        if ph in (None, 'UNFOLD', 'ALIGN_PRE', 'ALIGN_HOLD'):
            qa = B.Q_PRE
        elif ph in ('APPROACH', 'ENGAGE'):
            qa = B.Q_HOLD
        elif ph == 'HOLD':
            qa = q_hold_arm
        else:
            qa = B.q_arm_of(q)
        qf = [0.0, 0.0] if closed else [0.0089, 0.0089]
        lc = {'uflite_finger1': {'bar': fb, 'ambiguous': 0.0, 'panel': 0.0}, 'uflite_finger2': {'bar': fb, 'ambiguous': 0.0, 'panel': 0.0}} if fb else {}
        steps.append([k, t, 1, 'wholebody_applied', [0, 0, 0], [0, 0, 0], [0] * 6, list(B.BASE), list(map(float, qa)), q, (q - qprev) / 0.01,
                      qf, [fb, fb], held, lc, {}, 3.0, 'hold'])
        qprev = q
    return task, {'steps_cols': B.COLS, 'steps': steps}, {'gate': {'anchor': B.BASE}}


_CACHE = {}


def G(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _CACHE:
        _CACHE[key] = build(**kw)
    return copy.deepcopy(_CACHE[key])


def cap_of(task):
    return next(e for e in task['policy_events'] if 'unload_capture' in e)


class FullJudge(unittest.TestCase):
    def J(self, t, r, g):
        return UC.judge(t, r, g)

    def fails_with(self, mut, flag, **kw):
        t, r, g = G(**kw)
        mut(t, r, g)
        res = self.J(t, r, g)
        self.assertFalse(res['pass'])
        self.assertTrue(any(w.startswith(flag) for w in res['why']), res['why'])
        return res

    # ---- 正常通過
    def test_pass_nonzero(self):
        res = self.J(*G())
        self.assertTrue(res['pass'], (res['why'], res['stats']))
        self.assertAlmostEqual(res['stats']['delta_m'], -0.00105, delta=1e-4)
        self.assertLess(res['stats']['capture_fk_dev_m'], 1e-9)

    def test_pass_zero(self):
        res = self.J(*G(zero=True))
        self.assertTrue(res['pass'], (res['why'], res['stats']))
        self.assertLessEqual(abs(res['stats']['delta_m']), 1e-12)

    def test_pass_delayed(self):
        res = self.J(*G(tau_cap=1.2))
        self.assertTrue(res['pass'], (res['why'], res['stats']))
        self.assertAlmostEqual(res['stats']['capture_tau_s'], 1.2, places=6)

    # ---- Codex 的四項誤收（完整 judge）
    def test_capture_time_nan(self):
        def m(t, r, g):
            cap_of(t)['unload_capture']['t'] = float('nan')
        self.fails_with(m, 'unload_capture_time_invalid')

    def test_capture_outer_time_nan(self):
        def m(t, r, g):
            cap_of(t)['t'] = float('nan')
        self.fails_with(m, 'unload_capture_time_invalid')

    def test_capture_too_early(self):
        def m(t, r, g):
            c = cap_of(t)
            c['t'] = c['unload_capture']['t'] = T_HOLD + 0.1
        self.fails_with(m, 'unload_capture_outside_window')

    def test_ramp_end_before_capture(self):
        def m(t, r, g):
            e = next(e for e in t['policy_events'] if 'unload_ramp_end' in e)
            e['t'] = cap_of(t)['t'] - 0.05
        res = self.fails_with(m, 'unload_ramp_end_timing')
        self.assertIn('unload_marks_order', res['why'])

    def test_step_swapped_to_other_static_step(self):
        def m(t, r, g):
            cap_of(t)['unload_capture']['step'] += 20                                   # 0.2 s 後（同樣靜止）
        self.fails_with(m, 'unload_capture_time_not_snapshot')

    # ---- 其他時序與快照
    def test_capture_after_deadline(self):
        def m(t, r, g):
            c = cap_of(t)
            c['t'] = c['unload_capture']['t'] = T_HOLD + 1.6
        self.fails_with(m, 'unload_capture_outside_window')

    def test_duplicate_ramp_end(self):
        def m(t, r, g):
            e = next(e for e in t['policy_events'] if 'unload_ramp_end' in e)
            t['policy_events'].append(copy.deepcopy(e))
        self.fails_with(m, 'unload_ramp_end_count')

    def test_hold_end_too_late(self):
        def m(t, r, g):
            e = next(e for e in t['policy_events'] if 'unload_hold_end' in e)
            e['t'] += 0.2
        self.fails_with(m, 'unload_hold_end_timing')

    def test_hold_end_time_nan(self):
        def m(t, r, g):
            e = next(e for e in t['policy_events'] if 'unload_hold_end' in e)
            e['t'] = float('nan')
        self.fails_with(m, 'unload_hold_end_time_invalid')

    def test_capture_qdot_over(self):
        def m(t, r, g):
            k = cap_of(t)['unload_capture']['step']
            r['steps'][k][10] = 0.003
        self.fails_with(m, 'unload_capture_qdot_over')

    def test_capture_rotation_mismatch(self):
        def m(t, r, g):
            T = np.array(cap_of(t)['unload_capture']['T_tcp'])
            c, s_ = math.cos(1e-3), math.sin(1e-3)
            T[:3, :3] = T[:3, :3] @ np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
            cap_of(t)['unload_capture']['T_tcp'] = T.tolist()
        self.fails_with(m, 'unload_capture_not_from_snapshot')

    def test_delta_tamper_full_judge(self):
        def m(t, r, g):
            cap_of(t)['unload_capture']['delta'] -= 1e-5
        self.fails_with(m, 'unload_delta_not_projection')

    def test_open_without_capture(self):
        def m(t, r, g):
            t['policy_events'] = [e for e in t['policy_events'] if 'unload_capture' not in e]
        self.fails_with(m, 'unload_capture_count')


if __name__ == '__main__':
    unittest.main(verbosity=2)
