#!/usr/bin/env python3
"""判定器 r13（C2 接線獨立核對）測試。正例以**真走廊表**（登錄 r12 的 corridor_table）＋真 closure_cycle＋真 phase_coord 生成合成 trace
（門角 30→47°、底盤略偏、TCP 與目標差 3 mm、中段插入一次重送步）；只驗判定邏輯，不是實跑證據。
  W1 正例 ⇒ check_closure 無理由；o 最大 > 0、s 首次 > 0 於 36° 附近。
  W2 反例（各自命中）：登錄 cfg 不符／模組 sha 不符 ⇒ closure_registration；o 超 8 mm ⇒ closure_output_over_max；o 跳變超限 ⇒ closure_output_rate；
     o_prev 不接續 ⇒ closure_discontinuity；held 列改值 ⇒ closure_held_changed；updated 列在 HOLD ⇒ closure_update_outside_phase；
     s 改 ⇒ closure_weight_mismatch；步號倒退 ⇒ closure_step_not_advanced；ref_orig 改 1e-6 ⇒ closure_ref_orig_mismatch；
     ref_mod 偏航改 ⇒ closure_ref_mod_mismatch；guide v_body 改 1e-9 ⇒ closure_guide_mismatch；缺 base_pose ⇒ closure_guide_inputs_missing；
     連續 input_invalid 0.25 s 而無中止 ⇒ closure_invalid_abort_mismatch；無紀錄 ⇒ closure_records_missing；why 非法 ⇒ closure_record_invalid。
  W3 回歸：bench_6 實跑紀錄（登錄 r11，未啟用 C2）⇒ r13 與 r12 的 why 完全相同、closure_guidance_registered False。

    python3 evaluation/test_door_coopen_check_core13.py
"""
import copy
import hashlib
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_coopen_check_core12 as J12                 # noqa: E402
import door_coopen_check_core13 as J13                 # noqa: E402
import door_coopen_closure_guidance2 as CG2            # noqa: E402
import door_k3m_corridor as K3MCOR                     # noqa: E402
import door_wide_guidance as WG                        # noqa: E402

PRM = json.load(open(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r12.json')))
CT = PRM['protection_scope_change']['corridor_table']
GCFG = PRM['control_cost_change']['guidance']


def synth():
    ref = WG.CorridorRef(K3MCOR.Corridor.from_json(json.load(open(os.path.join(WS, CT)))['table']))
    gcfg = WG.GuidanceCfg(**GCFG)
    corr = CG2.ClosureCorrector2(CG2.ClosureCfg2(**PRM['control_cost_change']['closure_guidance']['cfg']))
    rows, step = [], 1000
    qs = np.linspace(math.radians(30.0), math.radians(47.0), 400)
    for k, q in enumerate(qs):
        t = 50.0 + 0.05 * k
        step = step + (0 if k == 200 else 5)                                       # k＝200：重送同一步
        pr = ref.pose(q)
        pose = (pr[0] + 0.002, pr[1] - 0.001, pr[2])
        H = np.eye(4)
        H[0, 3], H[1, 3] = pose[0] + 0.5, pose[1] + 0.3
        Ttcp = H.copy()
        Ttcp[0, 3] += 0.003
        phase = 'OPEN'
        tbl, rec, _ = CG2.closure_cycle(corr, ref, phase, 'free', np.eye(4), step, t, 0.05, float(q), Ttcp, H)
        rec.update(compute_s=0.002, base_pose=list(pose), q_rate=0.01, phase=phase)
        coord, _, _ = WG.phase_coord(tbl, gcfg, phase, 'free', pose, float(q), 0.01)
        rows.append({'t': t, 'step': step, 'phase': phase, 'q_door': float(q), 'k3m_mode': 'free', 'guide': {'v_body': coord[2:5]}, 'closure': rec})
    task = {'args': {'corridor_table': CT}, 'w1b': {'guidance_cfg': GCFG}, 'trace': rows, 'policy_events': [],
            'closure': {'enabled': True, 'cfg': PRM['control_cost_change']['closure_guidance']['cfg'],
                        'module_sha256': hashlib.sha256(open(CG2.__file__, 'rb').read()).hexdigest(), 'counts': None}}
    return task


def why_of(task, prm=PRM):
    w, st = [], {}
    J13.check_closure(task, prm, w, st)
    return w, st


class W(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.T = synth()

    def test_w1_positive(self):
        w, st = why_of(copy.deepcopy(self.T))
        self.assertEqual(w, [], w)
        self.assertGreater(st['closure_check']['o_max_mm'], 0.5)
        self.assertAlmostEqual(st['closure_check']['s_first_deg'], 36.0, delta=0.1)

    def test_w2_counterexamples(self):
        T0 = self.T
        i = 300
        cases = []

        def mk(f, tag, prm=PRM):
            t = copy.deepcopy(T0)
            f(t)
            cases.append((t, tag, prm))
        p2 = copy.deepcopy(PRM)
        p2['control_cost_change']['closure_guidance']['cfg']['tau_s'] = 2.0
        cases.append((copy.deepcopy(T0), 'closure_registration', p2))
        mk(lambda t: t['closure'].update(module_sha256='0' * 64), 'closure_registration')
        mk(lambda t: t['trace'][i]['closure'].update(o=[0.009, 0.0]), 'closure_output_over_max')

        def jump(t):
            r = t['trace'][i]['closure']
            r['o'] = [r['o'][0] + 0.001, r['o'][1]]
            t['trace'][i + 1]['closure']['o_prev'] = r['o']
        mk(jump, 'closure_output_rate')
        mk(lambda t: t['trace'][i]['closure'].update(o_prev=[0.0, 0.0]), 'closure_discontinuity')

        def held(t):
            r = t['trace'][200]['closure']
            self.assertEqual(r['why'], 'step_not_advanced')
            r['o'] = [r['o'][0] + 1e-6, r['o'][1]]
        mk(held, 'closure_held_changed')
        mk(lambda t: t['trace'][i].update(phase='HOLD'), 'closure_update_outside_phase')
        mk(lambda t: t['trace'][i]['closure'].update(s=0.5), 'closure_weight_mismatch')
        mk(lambda t: t['trace'][i]['closure'].update(step=10), 'closure_step_not_advanced')
        mk(lambda t: t['trace'][i]['closure']['ref_orig'].__setitem__(0, t['trace'][i]['closure']['ref_orig'][0] + 1e-6), 'closure_ref_orig_mismatch')
        mk(lambda t: t['trace'][i]['closure']['ref_mod'].__setitem__(2, t['trace'][i]['closure']['ref_mod'][2] + 1e-6), 'closure_ref_mod_mismatch')
        mk(lambda t: t['trace'][i]['guide'].update(v_body=[t['trace'][i]['guide']['v_body'][0] + 1e-9] + list(t['trace'][i]['guide']['v_body'][1:])),
           'closure_guide_mismatch')
        mk(lambda t: t['trace'][i]['closure'].pop('base_pose'), 'closure_guide_inputs_missing')

        def invalid(t):
            for k in range(320, 326):
                r = t['trace'][k]['closure']
                r.update(why='input_invalid', o=t['trace'][319]['closure']['o'], o_prev=t['trace'][319]['closure']['o'])
            t['trace'][326]['closure']['o_prev'] = t['trace'][319]['closure']['o']
        mk(invalid, 'closure_invalid_abort_mismatch')
        mk(lambda t: [r.pop('closure') for r in t['trace']], 'closure_records_missing')
        mk(lambda t: t['trace'][i]['closure'].update(why='weird'), 'closure_record_invalid')
        for t, tag, prm in cases:
            with self.subTest(tag):
                w, _ = why_of(t, prm)
                self.assertTrue(any(x.startswith(tag) for x in w), (tag, w))

    def test_w3_regression_bench6(self):
        R = os.path.join(HERE, 'runs', 'door_coopen_core_bench_6')
        ld = (lambda f: json.load(open(os.path.join(R, f))))
        args = (ld('task.json'), ld('room_run.json'), ld('park_gate.json'), ld('align_solver.json'), ld('lw_observer.json'),
                os.path.join(R, 'evidence.jsonl'), os.path.join(R, 'task_trace.jsonl'))
        a = J12.judge(*copy.deepcopy(args))
        b = J13.judge(*copy.deepcopy(args))
        self.assertEqual(a['why'], b['why'])
        self.assertFalse(b['closure_guidance_registered'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
