#!/usr/bin/env python3
"""door_w2_check_mode3 反例（以真鏈接線 C1 的正常紀錄為底，逐項竄改）。
  J0 正常紀錄 ⇒ ok。J1 刪去窗口內一列權威確認 ⇒ C8_confirm_not_covering_window。
  J2 執行端紀錄九軸與權威確認不一致 ⇒ C8_records_disagree。J3 某步 confirm_ok False ⇒ C8_confirm_not_ok。
  J4 提交步 applied_u 非保持形 ⇒ C9_commit_applied_not_hold；提交後一步九軸非保持形 ⇒ C9_post_commit_not_hold。
  J5 提交筆 stamp 早於選定 0.3 s ⇒ C10_commit_source_stale；提交時刻晚於請求 1.01 s ⇒ C10_commit_after_deadline。
  J6 有中止或入口違規 ⇒ C11。J7 r2 項目仍生效：截斷到提交步 ⇒ C0_window_not_covered。

    python3 evaluation/test_door_w2_check_mode3.py
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_w2_check_mode3 as CK3                      # noqa: E402
import test_door_w2_exec_chain as TC                   # noqa: E402

W = TC.World()
ROWS = W.run(3.0, req_at=1.0)
AUTH = W.auth.stats()
AR = copy.deepcopy(W.auth.applied_rows)
K = AUTH['commit']['step']
SW = {'committed': 1, 'first_src_by_mode': {'1': AUTH['commit']['first_source_seq']}}


def run(rows=None, auth=None, ar=None):
    return CK3.check(rows if rows is not None else copy.deepcopy(ROWS), auth if auth is not None else copy.deepcopy(AUTH),
                     ar if ar is not None else copy.deepcopy(AR), SW, t_req=W.t_req, t_obs_passed=0.5, window=50)


class TJudge(unittest.TestCase):
    def test_j0_ok(self):
        r = run()
        self.assertTrue(r['ok'], r)

    def test_j1_missing_confirm_row(self):
        ar = [a for a in copy.deepcopy(AR) if a['step'] != K + 7]
        self.assertIn('C8_confirm_not_covering_window', run(ar=ar)['why'])

    def test_j2_records_disagree(self):
        rows = copy.deepcopy(ROWS)
        next(r for r in rows if r['step'] == K + 3)['u'][0] += 1e-4
        self.assertIn('C8_records_disagree', run(rows=rows)['why'])

    def test_j3_confirm_not_ok(self):
        rows = copy.deepcopy(ROWS)
        next(r for r in rows if r['step'] == K - 3)['confirm_ok'] = False
        self.assertIn('C8_confirm_not_ok', run(rows=rows)['why'])

    def test_j4_not_hold(self):
        a = copy.deepcopy(AUTH)
        a['commit']['applied_u'] = [0.009, 0.0, 0.0] + [0.0] * 6
        self.assertIn('C9_commit_applied_not_hold', run(auth=a)['why'])
        rows, ar = copy.deepcopy(ROWS), copy.deepcopy(AR)
        next(r for r in rows if r['step'] == K + 1)['u'][3] = 1e-3
        next(r for r in ar if r['step'] == K + 1)['u'][3] = 1e-3
        self.assertIn('C9_post_commit_not_hold', run(rows=rows, ar=ar)['why'])

    def test_j5_time(self):
        a = copy.deepcopy(AUTH)
        a['commit']['stamp_sim_t'] = a['commit']['t'] - 0.30
        self.assertIn('C10_commit_source_stale', run(auth=a)['why'])
        a = copy.deepcopy(AUTH)
        next(e for e in a['events'] if e['kind'] == 'pending')['t'] = a['commit']['t'] - 1.01
        self.assertIn('C10_commit_after_deadline', run(auth=a)['why'])

    def test_j6_abort_or_violation(self):
        a = copy.deepcopy(AUTH)
        a['aborted'] = {'why': 'x', 'step': K + 60, 't': 0.0}
        self.assertIn('C11_aborted_or_violation', run(auth=a)['why'])
        a = copy.deepcopy(AUTH)
        a['violations'] = [{'why': 'future_stamp'}]
        self.assertIn('C11_aborted_or_violation', run(auth=a)['why'])

    def test_j7_r2_items_still_apply(self):
        rows = [r for r in copy.deepcopy(ROWS) if r['step'] <= K]
        self.assertIn('C0_window_not_covered', run(rows=rows)['why'])


if __name__ == '__main__':
    unittest.main(verbosity=1)
