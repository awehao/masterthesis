#!/usr/bin/env python3
"""door_k3au_check 的卸載核對反例。結構核對用 K3-A-U 相位機合成世界產生、依任務節點格式組成的 trace；
快照核對用 IK／FK 自洽的合成 room 步。

    python3 evaluation/test_door_k3au_check.py
"""
import copy
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                            # noqa: E402
import door_k3au_check as UC                          # noqa: E402
import test_door_k3au_policy as TU                    # noqa: E402


def task_from(sim):
    pol = sim.pol
    tr = []
    for t, ph, a, m in sim.acts:
        tr.append({'t': t, 'phase': ph, 'q_door': m.q_door, 'q_ref': pol.q_ref if ph in ('OPEN', 'SETTLE', 'END_HOLD') else None,
                   'target': None if (a['target'] is None or ph == 'ABORTED') else np.asarray(a['target']).reshape(-1).tolist(),
                   'grip': a['grip']})
    return {'schema': 'door_k3au_task/1', 'policy_events': copy.deepcopy(pol.events), 'trace': tr,
            'H_frozen': pol.H_frozen.tolist(), 'grasp_basis': {'T_HG': pol.T_HG.tolist()}}


def run(inj=None, off=(0.0, -0.00105, 0.0)):
    s = TU.mk_sim(inj, off)
    s.run()
    return task_from(s)


def chk(task):
    why, st = [], {}
    UC.check_unload_structure(task, why, st)
    return why, st


class Structure(unittest.TestCase):
    def test_nominal_passes(self):
        why, st = chk(run())
        self.assertEqual(why, [], why)
        self.assertAlmostEqual(st['delta_m'], -0.00105, places=12)
        self.assertGreater(st['n_rows_after_capture'], 10)

    def test_tampered_delta_not_projection(self):
        t = run()
        cap = next(e for e in t['policy_events'] if 'unload_capture' in e)
        cap['unload_capture']['delta'] -= 1e-6
        why, _ = chk(t)
        self.assertIn('unload_delta_not_projection', why)

    def test_axis_must_be_tool_z_not_retreat_axis(self):
        t = run()
        cap = next(e for e in t['policy_events'] if 'unload_capture' in e)
        cap['unload_capture']['a'] = [0.0, -1.0, 0.0]                              # 退讓軸 a_H
        why, _ = chk(t)
        self.assertIn('unload_axis_not_tool_z', why)

    def test_target_path_mismatch(self):
        t = run()
        cap = next(e for e in t['policy_events'] if 'unload_capture' in e)['unload_capture']
        e = next(r for r in t['trace'] if r['phase'] == 'HOLD' and r['t'] > cap['t'] + 0.2 and r['target'])
        e['target'][3] += 1e-6
        why, _ = chk(t)
        self.assertIn('unload_target_path_mismatch', why)

    def test_double_capture_rejected(self):
        t = run()
        cap = next(e for e in t['policy_events'] if 'unload_capture' in e)
        t['policy_events'].append(copy.deepcopy(cap))
        why, _ = chk(t)
        self.assertTrue(any(w.startswith('unload_capture_count') for w in why), why)

    def test_open_too_early(self):
        t = run()
        ev = next(e for e in t['policy_events'] if e.get('to') == 'OPEN')
        ev['t'] -= 0.3
        why, _ = chk(t)
        self.assertIn('open_before_unload_complete', why)

    def test_abort_mid_ramp_clean(self):
        def inj(sim, m):
            cap = [e for e in sim.pol.events if 'unload_capture' in e]
            if cap and 0.2 <= m.t - cap[0]['t'] < 0.3:
                m.wrist_max = 31.0
        why, st = chk(run(inj))
        self.assertEqual(why, [], why)                                               # 結構上乾淨（無 OPEN、中止後無目標）

    def test_abort_mid_ramp_but_target_after_abort_flagged(self):
        def inj(sim, m):
            cap = [e for e in sim.pol.events if 'unload_capture' in e]
            if cap and 0.2 <= m.t - cap[0]['t'] < 0.3:
                m.wrist_max = 31.0
        t = run(inj)
        ab = next(e for e in t['policy_events'] if 'abort' in e)
        r = next(r for r in t['trace'] if r['t'] > ab['t'] + 0.1)
        r['target'] = np.eye(4).reshape(-1).tolist()
        why, _ = chk(t)
        self.assertIn('targets_after_abort', why)

    def test_positive_delta_bounds(self):
        why, st = chk(run(off=(0.0, 0.0008, 0.0)))
        self.assertIn('unload_delta_out_of_bounds', why)

    def test_zero_delta_ok(self):
        why, st = chk(run(off=(0.0, 0.0, 0.0)))
        self.assertEqual(why, [], why)
        self.assertEqual(st['delta_m'], 0.0)

    def test_T_ref0_must_be_hold_reference(self):
        t = run()
        cap = next(e for e in t['policy_events'] if 'unload_capture' in e)['unload_capture']
        cap['T_ref0'][0][3] += 1e-6
        why, _ = chk(t)
        self.assertIn('unload_T_ref0_not_hold_reference', why)


class Snapshot(unittest.TestCase):
    def setUp(self):
        import door_k2_offline_v2 as V2
        base = list(V2.base_of(0, 0, 0))
        T = V2.C.target(0.0, 0.0)
        q = V2.C.ik(np.array(base), T, None)[V2.C.IDX]
        self.base, self.q = base, [round(float(x), 6) for x in q]
        self.room = {'steps_cols': ['step', 'base_xyth', 'q_arm_meas'], 'steps': [[100, base, self.q]]}

    def task(self, T_tcp, step=100):
        return {'policy_events': [{'t': 1.0, 'unload_capture': {'t': 1.0, 'step': step, 'T_tcp': T_tcp.tolist()}}]}

    def test_consistent_snapshot(self):
        why, st = [], {}
        UC.check_unload_snapshot(self.task(KC._tcp(self.base, self.q)), self.room, why, st)
        self.assertEqual(why, [])
        self.assertLess(st['capture_fk_dev_m'], 1e-9)

    def test_tampered_snapshot(self):
        T = KC._tcp(self.base, self.q)
        T[1, 3] += 1e-4
        why, st = [], {}
        UC.check_unload_snapshot(self.task(T), self.room, why, st)
        self.assertIn('unload_capture_not_from_snapshot', why)

    def test_missing_step(self):
        why, st = [], {}
        UC.check_unload_snapshot(self.task(KC._tcp(self.base, self.q), step=999), self.room, why, st)
        self.assertIn('unload_capture_step_missing_in_room', why)


if __name__ == '__main__':
    unittest.main(verbosity=2)
