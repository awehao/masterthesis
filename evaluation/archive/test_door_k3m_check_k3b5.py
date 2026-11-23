#!/usr/bin/env python3
"""door_k3m_check_k3b5 最外層 judge() 反例（Codex 20261026_034842 §1、§3；**測試用合成資料，不是實跑結果**）。
合成：door_k3b5_synth.build（實跑 door_wide_w2c_30_1 為基底＋卸載段＋合成張爪／退離；各監看紀錄以真監看邏輯逐步產生）。
  F0 完整合成通過例：直接呼叫最外層 judge()（不替換任何判定函式）⇒ pass True、why 空；被排除的舊理由恰為登錄的三項。
  F1 Codex：RELEASE_SETTLED 宣告為張爪後 0.01 s ⇒ RT5_settled_declared_before_recomputed_convergence。
  F2 張爪前（卸載前）一指接觸 0.3 N ⇒ 舊鏈理由保留（不被排除）。
  F3 卸載前一筆定期讀回 5 N ⇒ E2_readback_not_20_before_unload（舊 W1e 理由雖屬替換項，分段證據照抓）。
  F4 暫停切換讀回非 20 ⇒ 舊 finger_effort_not_effective 保留＋E1。
  F5 提交前出現 RELEASE 事件（任務端先於執行端提交進入釋放）⇒ 舊鏈 phase_sequence 保留。
  F6 真趟未 DONE（ABORTED）⇒ 視圖不附 DONE ⇒ 舊鏈 termination 保留＋K8b。
  F7 張爪後手臂 6 mm 去回（settled 前）⇒ RT5_release_transient_*。
  F8 settled 後、退離前手臂再動 ⇒ RT5_arm_not_quiet_before_retreat。
  F9 求解端監督器版本 r8 ⇒ S9_supervisor_version；F10 mode 1 提交時刻差 1.5125e-6（Codex 實例）⇒ 仍通過。
  F11 張爪前 room 缺一步 ⇒ 舊鏈 step_discontinuity 保留。
  F12 對照：k3b4.judge 對同一合成趟仍報舊鏈理由（Codex 重現）。

    python3 evaluation/test_door_k3m_check_k3b5.py
"""
import copy
import math
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3b5_synth as SY                           # noqa: E402
import door_k3m_check_k3b4 as J4                       # noqa: E402
import door_k3m_check_k3b5 as J5                       # noqa: E402

TD = tempfile.mkdtemp(prefix='k3b5_test_')
BASE = SY.build(TD)
M = BASE['meta']
FILES = ('task.json', 'room_run.json', 'park_gate.json', 'align_solver.json', 'lw_observer.json')


def run(f=None, judge=J5.judge):
    d = copy.deepcopy(BASE)
    if f:
        f(d)
    return judge(*(d[k] for k in FILES))


def I(d, c):
    return d['room_run.json']['steps_cols'].index(c)


def rows(d, a, b):
    return [s for s in d['room_run.json']['steps'] if a <= s[0] < b]


class TF(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TD, ignore_errors=True)

    def has(self, r, tag):
        self.assertTrue(any(w.startswith(tag) for w in r['why']), (tag, r['why']))
        self.assertFalse(r['pass'])

    def test_f0_full_pass(self):
        r = run()
        self.assertTrue(r['pass'], r['why'])
        self.assertEqual(r['why'], [])
        self.assertEqual(sorted(r['replaced_old_reasons']), sorted(['pause_solver_outcome_not_done', 'pause_task_not_done'] +
                                                                   [w for w in r['replaced_old_reasons'] if w.startswith('finger_effort_periodic_short_')]))
        self.assertEqual(len(r['replaced_old_reasons']), 3)
        self.assertEqual(r['stats']['k3b5_view']['k2'], M['k2'])

    def test_f1_settled_too_early(self):
        r = run(lambda d: d['align_solver.json']['stats']['k3m_pause']['supervisor_outcome']['transitions'].update(RELEASE_SETTLED=M['t2'] + 0.01))
        self.has(r, 'RT5_settled_declared_before_recomputed_convergence')

    def test_f2_f4_pre_release_old_chain(self):
        def f2(d):
            for s in rows(d, M['k1'] - 150, M['k1'] - 140):
                s[I(d, 'leaf_contacts')]['uflite_finger2']['bar'] = 0.3
                if isinstance(s[I(d, 'finger_contact_n')], list):
                    s[I(d, 'finger_contact_n')][1] = 0.3
        r = run(f2)
        self.assertFalse(r['pass'])
        self.assertTrue(r['why'] and not all(w.startswith(('E', 'K', 'U', 'RT5', 'S9', 'X')) for w in r['why']), r['why'])

        def f3(d):
            per = d['room_run.json']['finger_effort']['pause']['periodic']
            p = [x for x in per if x['t'] < M['t1']][-2]
            p['readback'] = [5.0, 5.0]
        self.has(run(f3), 'E2_readback_not_20_before_unload')
        r = run(lambda d: d['room_run.json']['finger_effort']['pause'].update(readback=[5.0, 5.0]))
        self.assertIn('finger_effort_not_effective', r['why'])
        self.has(r, 'E1_pause_switch_not_5_to_20')

    def test_f5_f6_view_events(self):
        def f5(d):
            d['task.json']['policy_events'].insert(-2, {'t': M['t2'] - 0.5, 'from': 'PAUSE', 'to': 'RELEASE'})
        self.has(run(f5), 'phase_sequence')

        def f6(d):
            t = d['task.json']
            t['final_phase'], t['abort'] = 'ABORTED', 'k3b_release:x'
        r = run(f6)
        self.has(r, 'termination:ABORTED')
        self.has(r, 'K8b_task_not_done')

    def test_f7_f8_release_transient(self):
        def f7(d):
            j = I(d, 'q_arm_meas')
            for s in rows(d, M['k2'], M['k2'] + 40):
                x = (s[0] - M['k2']) / 40.0
                s[j] = list(s[j])
                s[j][1] += 0.02 * math.sin(math.pi * x)
        self.has(run(f7), 'RT5_release_transient_')
        k_set = int(round(M['t_settled'] * 100)) - 2

        def f8(d):
            j = I(d, 'q_arm_meas')
            for s in rows(d, k_set + 5, M['k3']):
                s[j] = list(s[j])
                s[j][1] += 0.004 * (s[0] - k_set - 5) / max(1, M['k3'] - k_set - 5)
        self.has(run(f8), 'RT5_arm_not_quiet_before_retreat')

    def test_f9_f11(self):
        self.has(run(lambda d: d['align_solver.json']['stats']['k3m_pause']['supervisor_outcome'].update(version='door_w2_mode8/r8')),
                 'S9_supervisor_version')

        def f10(d):
            for e in d['room_run.json']['w2']['auth']['events']:
                if e.get('kind') == 'commit' and e.get('mode_seq') == 1:
                    e['t'] += 1.5125e-6
        r = run(f10)
        self.assertTrue(r['pass'], r['why'])

        def f11(d):
            st = d['room_run.json']['steps']
            i = next(n for n, s in enumerate(st) if s[0] == M['k1'] - 300)
            del st[i]
        self.has(run(f11), 'step_discontinuity')

    def test_f12_k3b4_contrast(self):
        r = run(judge=J4.judge)
        self.assertFalse(r['pass'])
        for tag in ('phase_sequence', 'door_angle_pre_open_over_1deg', 'grasp_finger_contact_below_min', 'finger_effort_periodic_invalid_',
                    'finger_effort_periodic_short_'):
            self.assertTrue(any(w.startswith(tag) for w in r['why']), (tag, r['why']))


if __name__ == '__main__':
    unittest.main(verbosity=1)
