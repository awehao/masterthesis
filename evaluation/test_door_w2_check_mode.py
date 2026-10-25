#!/usr/bin/env python3
"""door_w2_check_mode（逐步證據 C1–C7）與 cmd_envelope_v3 的離線反例。
  K0 合成正常紀錄 ⇒ 通過。
  K1 提交步套用的 src≠first_source_seq ⇒ C1；K2 切換附近一步逾時 ⇒ C2；K3 mode_seq 倒退 ⇒ C3；K4 k 之前出現 mode 1 ⇒ C3；
  K5 命令 mode_seq≠權威 ⇒ C3；K6 profile 晚一步切換 ⇒ C4；K7 注入舊命令出現在 k 之後 ⇒ C5；K8 n_stale 少於注入數 ⇒ C5；
  K9 求解端未得知提交 ⇒ C6；K10 HELD 期間（觀察通過後、請求前）有逾時步 ⇒ C7；K11 提交兩次 ⇒ commit_count_2。
  V1 v3 編解碼往返、mode_seq 保留；V2 mode_seq 負數／非整數／缺欄、schema 不符 ⇒ 拒絕；V3 v2 的身分規則照樣檢查（derived＝0 但 source_seq≠−1）。

    python3 evaluation/test_door_w2_check_mode.py
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmd_envelope_v3 as E3                           # noqa: E402
import door_w2_check_mode as C                         # noqa: E402
import wgmpc_cmd_envelope as V2                        # noqa: E402

K = 1000


def synth():
    rows = []
    src = 100
    for s in range(800, 1200):
        t = s * 0.01
        if s % 5 == 0:
            src += 1
        ms = 1 if s >= K else 0
        if s == K:
            src = 200                                                             # 首筆新模式 src
        rows.append({'step': s, 't': t, 'src': src, 'mode_seq': ms, 'chain_mode': 0,
                     'profile': 'release_prep_test' if s >= K else 'pause_hold', 'auth_mode_seq': ms})
    auth = {'commit': {'step': K, 't': K * 0.01, 'first_source_seq': 200, 'mode_seq': 1},
            'events': [{'kind': 'pending'}, {'kind': 'commit', 'step': K}], 'n_stale': 0}
    sw = {'committed': 1, 'first_src_by_mode': {'0': 1, '1': 200}}
    return rows, auth, sw


def run(rows, auth, sw, stale=()):
    return C.check(rows, auth, sw, t_req=9.8, t_obs_passed=8.5, stale_injected=stale)


class T(unittest.TestCase):
    def test_k0_ok(self):
        r = run(*synth())
        self.assertTrue(r['ok'], r)

    def mut(self, f, tag, stale=()):
        rows, auth, sw = synth()
        f(rows, auth, sw)
        r = run(rows, auth, sw, stale)
        self.assertFalse(r['ok'], tag)
        self.assertIn(tag, r['why'], (tag, r['why']))

    def test_counterexamples(self):
        at = lambda rows, s: next(r for r in rows if r['step'] == s)   # noqa: E731
        self.mut(lambda R, A, S: at(R, K).__setitem__('src', 199), 'C1_first_applied_mismatch')
        self.mut(lambda R, A, S: at(R, K + 3).update(chain_mode=1, src=-1), 'C2_gap_around_switch')
        self.mut(lambda R, A, S: at(R, K + 30).update(mode_seq=0, auth_mode_seq=0), 'C3_mode_seq_backward')
        self.mut(lambda R, A, S: at(R, K - 2).update(mode_seq=1, auth_mode_seq=1), 'C3_mode_seq_not_switched_at_k')
        self.mut(lambda R, A, S: at(R, K + 10).__setitem__('auth_mode_seq', 0), 'C3_cmd_mode_ne_authority')
        self.mut(lambda R, A, S: at(R, K).__setitem__('profile', 'pause_hold'), 'C4_profile_not_switched_at_k')
        self.mut(lambda R, A, S: (at(R, K + 20).__setitem__('src', 150), A.__setitem__('n_stale', 1)), 'C5_stale_applied', stale=(150,))
        self.mut(lambda R, A, S: None, 'C5_stale_not_counted', stale=(150,))
        self.mut(lambda R, A, S: S.__setitem__('committed', 0), 'C6_solver_not_committed')
        self.mut(lambda R, A, S: at(R, 900).update(chain_mode=1, src=-1), 'C7_held_not_publishing')
        rows, auth, sw = synth()
        auth['events'].append({'kind': 'commit', 'step': K + 50})
        self.assertIn('commit_count_2', run(rows, auth, sw)['why'])

    def test_v1_v3_roundtrip(self):
        d = E3.encode3(12345.0, 7, V2.ST_ADAPTER, 3, 7, True, 1.5, [0.1] * 9, 4)
        e = E3.decode3(d)
        self.assertEqual((e['mode_seq'], e['source_seq'], e['stage_id'], e['schema'], len(d)), (4, 7, V2.ST_ADAPTER, 3.0, 19))

    def test_v2_v3_rejects(self):
        good = E3.encode3(1.0, 7, V2.ST_SOLVER, 7, 7, True, 1.0, [0.0] * 9, 1)
        for bad in (good[:-1] + [-1.0], good[:-1] + [1.5], good[:-1], [2.0] + good[1:]):
            with self.assertRaises(ValueError):
                E3.decode3(bad)
        with self.assertRaises(ValueError):
            E3.encode3(1.0, 7, V2.ST_SOLVER, 7, 7, True, 1.0, [0.0] * 9, -1)

    def test_v3_keeps_v2_identity_rules(self):
        good = E3.encode3(1.0, 7, V2.ST_SOLVER, 7, 7, True, 1.0, [0.0] * 9, 1)
        bad = list(good)
        bad[6] = 0.0                                                                     # derived 0 但 source_seq 7
        with self.assertRaises(ValueError):
            E3.decode3(bad)


if __name__ == '__main__':
    unittest.main(verbosity=1)
