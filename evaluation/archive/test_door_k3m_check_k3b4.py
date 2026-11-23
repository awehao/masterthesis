#!/usr/bin/env python3
"""door_k3m_check_k3b4 反例（**合成資料，不是實跑結果**；新／替換項逐項單元驗證＋組合接線核對）。
  E  分段證據：正例無理由；E1 暫停切換非 5→20；E2 卸載前讀回 5、間隔 2 s；E3 斜坡紀錄失敗、設定回升；E4 斜坡中讀回≠斜坡值、
     t_settling 後讀回 20、紀錄末缺讀回；E5 存在 5 N 恢復紀錄或 5 N 權威 pending；E6 非手指上限改變；E0 紀錄缺。
  K8b 任務端：正例；eff_verify 缺或讀回 20、核對早於 C_CLEAR、仍有 5 N 請求、unload_gate 非 go、mode 3 於 release_settled 前請求、未 DONE。
  S8 求解端：正例；版本不符、轉移序顛倒、瞬態未收斂、t_settling 缺。
  RT 釋放瞬態：30_1 HELD 姿態合成；回彈 0.004 rad 正例；j2 漂移超 5 mm ⇒ bounds；settled 晚於 1.5 s；settled 後手臂再動。
  X  登錄與時間：以 test_door_k3m_check_k3b2 的合成紀錄補 r4 欄位；正例無 X 理由；缺 --w2-hold-r8 ⇒ X1b；mode 2 請求早於 UNLOADED ⇒ X3b；
     eff_verify 早於 C_CLEAR ⇒ X3b。
  W  組合：check_k3b4 不呼叫被替換的 check_effort2／check_task／check_reg_id_time；新項皆有呼叫；內部例外 ⇒ 回理由。

    python3 evaluation/test_door_k3m_check_k3b4.py
"""
import copy
import inspect
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3b_unload2 as KU                          # noqa: E402
import door_k3m_check_k3b4 as J4                       # noqa: E402

A0 = [-0.8957, 0.3327, 1.9056, -1.5765, -0.557, 0.0063]
B0 = [0.3215, -0.8037, 1.3843]


def eff_room():
    t_sw, t0, ts, t_end = 5.0, 8.0, 9.0, 20.0
    per = []
    for k in range(1, 15):
        t = t_sw + k * 1.0 + 0.01
        v = 20.0 if t < t0 else (KU.f_cmd(KU.REG_UNLOAD, t0, t) if t < ts else 5.0)
        per.append({'t': t, 'readback': [v, v], 'ok': True, 'others_same': True})
    uc = [{'t': t0 + i * 0.1, 'set': KU.f_cmd(KU.REG_UNLOAD, t0, t0 + i * 0.1), 'ok': True} for i in range(11)]
    return {'steps_cols': ['step', 'sim_t'], 'steps': [[int(t_end * 100), t_end]],
            'finger_effort': {'pause': {'ok': True, 'old': [5.0, 5.0], 'readback': [20.0, 20.0], 't': t_sw, 'periodic': per, 'restore': None}},
            'w2': {'unload': {'t0': t0, 't_settling': ts}, 'unload_ctrl': uc, 'effort': {'events': [], 'restore': None}}}


def task4():
    return {'final_phase': 'DONE',
            'k3b': {'monitor': {'events': [{'to': x} for x in ('ARMED', 'NEAR')] + [{'to': 'C_CLEAR', 's_meas': 0.021, 'pair': 0.013}]
                                + [{'to': 'RETREAT_DONE'}, {'to': 'COMPLETE'}], 'why': None,
                                't': {'C_CLEAR': 30.0, 'RETREAT_DONE': 34.0, 'COMPLETE': 37.1}},
                    'rel': {'state': 'done', 'eff_req': None, 'eff_done': None, 'solver_sub_at_req3': 'release_settled',
                            'eff_verify': {'ok': True, 'readback': [5.0, 5.0], 't': 30.05, 'readback_t': 29.7},
                            'unload_gate': {'gate': 'go', 'state_t': 20.0, 'solver_t': 19.99}}}}


def sol4():
    return {'stats': {'k3m_pause': {'supervisor_outcome': {
        'version': 'door_w2_mode8/r8', 't_settling': 19.0,
        'transitions': {'UNLOAD': 18.0, 'UNLOADED': 19.6, 'RELEASE_OPEN': 20.1, 'RELEASE_SETTLED': 21.4, 'HANDOVER': 21.6},
        'transient': {'1': {'state': 'CONVERGED'}, '2': {'state': 'CONVERGED'}}}}}}


class TE(unittest.TestCase):
    def run_e(self, f=None):
        r = eff_room()
        if f:
            f(r)
        why, st = [], {}
        J4.check_effort_segments(r, why, st)
        return why

    def test_e(self):
        self.assertEqual(self.run_e(), [])
        P = lambda r: r['finger_effort']['pause']['periodic']                 # noqa: E731
        cases = [
            (lambda r: r['finger_effort']['pause'].update(readback=[5.0, 5.0]), 'E1'),
            (lambda r: P(r)[0].update(readback=[5.0, 5.0]), 'E2_readback'),
            (lambda r: P(r).pop(1), 'E2_periodic_gap'),
            (lambda r: r['w2']['unload_ctrl'][3].update(ok=False), 'E3_unload_ctrl'),
            (lambda r: r['w2']['unload_ctrl'][5].update(set=19.9), 'E3_ramp_not_nonincreasing'),
            (lambda r: P(r)[2].update(readback=[20.0, 20.0]), 'E4_readback_not_ramp'),
            (lambda r: P(r)[6].update(readback=[20.0, 20.0]), 'E4_readback_not_5'),
            (lambda r: [P(r).pop() for _ in range(3)], 'E4_periodic_coverage'),
            (lambda r: r['finger_effort']['pause'].update(restore={'ok': True}), 'E5'),
            (lambda r: r['w2']['effort']['events'].append({'pending': 1}), 'E5'),
            (lambda r: P(r)[4].update(others_same=False), 'E6'),
            (lambda r: r['w2']['unload'].update(t_settling=None), 'E0'),
        ]
        for f, tag in cases:
            why = self.run_e(f)
            self.assertTrue(any(w.startswith(tag) for w in why), (tag, why))


class TK8b(unittest.TestCase):
    def test_k8b(self):
        why, st = [], {}
        J4.check_task4(task4(), why, st)
        self.assertEqual(why, [])
        R = lambda t: t['k3b']['rel']                                          # noqa: E731
        cases = [(lambda t: R(t).update(eff_verify=None), 'K8b_effort_verify'),
                 (lambda t: R(t)['eff_verify'].update(readback=[20.0, 20.0]), 'K8b_effort_verify'),
                 (lambda t: R(t)['eff_verify'].update(t=29.0), 'K8b_effort_verify'),
                 (lambda t: R(t).update(eff_req={'req_seq': 1}), 'K8b_restore_request_present'),
                 (lambda t: R(t)['unload_gate'].update(gate='wait'), 'K8b_unload_gate'),
                 (lambda t: R(t).update(solver_sub_at_req3='release_open'), 'K8b_mode3_before_release_settled'),
                 (lambda t: t.update(final_phase='ABORTED'), 'K8b_task_not_done')]
        for f, tag in cases:
            t = task4()
            f(t)
            why = []
            J4.check_task4(t, why, {})
            self.assertIn(tag, why)


class TS8(unittest.TestCase):
    def test_s8(self):
        why = []
        J4.check_solver4(sol4(), why, {})
        self.assertEqual(why, [])
        O = lambda s: s['stats']['k3m_pause']['supervisor_outcome']           # noqa: E731
        for f, tag in ((lambda s: O(s).update(version='door_w2_mode7/r7'), 'S8_supervisor_version'),
                       (lambda s: O(s)['transitions'].update(RELEASE_SETTLED=21.7), 'S8_transition_order'),
                       (lambda s: O(s)['transient']['2'].update(state='ACTIVE'), 'S8_transient_not_converged'),
                       (lambda s: O(s).update(t_settling=None), 'S8_t_settling_missing')):
            s = sol4()
            f(s)
            why = []
            J4.check_solver4(s, why, {})
            self.assertIn(tag, why)


def rt_room(j2=lambda t: 0.0, t_end=24.0):
    rows = []
    for k in range(2000, int(t_end * 100)):
        t = k * 0.01
        qa = list(A0)
        qa[1] += j2(t)
        rows.append([k, t, list(B0), qa])
    return {'steps_cols': ['step', 'sim_t', 'base_xyth', 'q_arm_meas'], 'steps': rows}


CS = {2: {'step': 2059, 't': 20.59}, 3: {'step': 2210, 't': 22.10}}


class TRT(unittest.TestCase):
    def rt(self, room, settled=21.95):
        o = {'transitions': {'RELEASE_SETTLED': settled}}
        why, st = [], {}
        J4.check_release_transient(room, o, CS, why, st)
        return why, st

    def test_rt(self):
        spring = lambda t: 0.004 * (1 - math.exp(-max(0.0, t - 20.59) / 0.2))  # noqa: E731
        why, st = self.rt(rt_room(spring))
        self.assertEqual(why, [], st)
        why, _ = self.rt(rt_room(lambda t: 0.012 * max(0.0, t - 20.59)))
        self.assertIn('RT_release_transient_bounds', why)
        why, _ = self.rt(rt_room(spring), settled=22.2)
        self.assertIn('RT_release_settled_late', why)
        why, _ = self.rt(rt_room(lambda t: spring(t) + 0.01 * max(0.0, t - 21.95)))
        self.assertTrue(any(w.startswith('RT_arm_not_quiet_before_retreat') for w in why), why)


class TX(unittest.TestCase):
    def base(self):
        import door_k3b_release4 as KR4
        import test_door_k3m_check_k3b2 as T2
        task, room, solver, I = copy.deepcopy(T2.BASE2)
        room['w2']['release']['reg'] = dict(KR4.REG)
        task['k3b']['monitor']['reg'] = dict(KR4.REG)
        room['w2']['unload'] = {'reg': dict(KU.REG_UNLOAD), 't': {'UNLOADED': 1.6}}
        solver['args'].update(w2_hold_r8=True, w2_unload=True)
        solver['stats'] = {'k3m_pause': {'supervisor_outcome': {'transitions': {'UNLOADED': 1.7}}}}
        task['args'].update(w2_unload=True)
        task['k3b']['rel']['eff_verify'] = {'t': 4.1}
        cs = {1: {'t': 1.02, 'step': 102}, 2: {'t': 2.0, 'step': 200}, 3: {'t': 3.02, 'step': 302}}
        return task, room, solver, cs

    def xs(self, f=None):
        task, room, solver, cs = self.base()
        if f:
            f(task, room, solver)
        why, st = [], {}
        J4.check_reg_id_time4(task, room, solver, cs, why, st)
        return why, st

    def test_x(self):
        why, st = self.xs()
        self.assertEqual(why, [], st['k3b4_cross'])
        why, _ = self.xs(lambda t, r, s: s['args'].pop('w2_hold_r8'))
        self.assertTrue(any(w.startswith('X1b_registration') and 'solver_args' in w for w in why), why)
        why, _ = self.xs(lambda t, r, s: r['w2']['unload']['t'].update(UNLOADED=1.95))
        self.assertTrue(any(w.startswith('X3b_time') and 'mode2_request_before_unloaded' in w for w in why), why)
        why, _ = self.xs(lambda t, r, s: t['k3b']['rel']['eff_verify'].update(t=3.9))
        self.assertTrue(any(w.startswith('X3b_time') and 'effort_verify_times' in w for w in why), why)
        why, _ = self.xs(lambda t, r, s: r['w2']['unload']['reg'].update(ramp_s=0.5))
        self.assertTrue(any('exec_unload' in w for w in why), why)


class TW(unittest.TestCase):
    def test_w(self):
        src = inspect.getsource(J4.check_k3b4)
        for old in ('check_effort2', 'J1.check_task,', 'J2.check_reg_id_time'):
            self.assertNotIn(old, src)
        for new in ('check_effort_segments', 'check_task4', 'check_solver4', 'check_release_transient', 'check_reg_id_time4', 'CU2.check_unload2',
                    'J3.check_end_hold_3d', 'J2.check_general_clearance', 'J2.check_exit_recompute'):
            self.assertIn(new, src)
        why = []
        J4._guard(lambda: 1 / 0, 'Z', why)
        self.assertEqual(why, ['Z_internal_error:ZeroDivisionError'])
        why = J4.check_k3b4({}, {'w2': {}}, None, {}, {})
        self.assertTrue(why and all(isinstance(w, str) for w in why), why)


if __name__ == '__main__':
    unittest.main(verbosity=1)
