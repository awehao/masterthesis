#!/usr/bin/env python3
"""xh_test_eval_devcheck.verdict 的離線反例（不載入模型、不讀影格）。

    .venv-dl0/bin/python evaluation/test_xh_devcheck_verdict.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xh_test_eval_devcheck as DC                     # noqa: E402

GOOD = {'complete': True, 'closure': {'closed': True, 'rows_match_evaluated': True}, 'evidence_issues': []}


class Verdict(unittest.TestCase):
    def test_all_good_passes(self):
        self.assertEqual(DC.verdict(GOOD, [], {}, 420, 420), (True, []))

    def test_same_rows_but_incomplete_fails(self):
        ok, why = DC.verdict(dict(GOOD, complete=False), [], {}, 420, 420)
        self.assertFalse(ok)
        self.assertIn('new_not_complete', why)

    def test_open_closure_fails(self):
        ok, why = DC.verdict(dict(GOOD, closure={'closed': False, 'rows_match_evaluated': True}), [], {}, 420, 420)
        self.assertFalse(ok)
        self.assertIn('closure_not_closed', why)

    def test_evidence_issue_fails(self):
        ok, why = DC.verdict(dict(GOOD, evidence_issues=[{'key': 'x', 'why': 'missing_rgb'}]), [], {}, 420, 420)
        self.assertFalse(ok)
        self.assertIn('evidence_issues:1', why)

    def test_missing_closure_fails(self):
        g = dict(GOOD)
        del g['closure']
        self.assertFalse(DC.verdict(g, [], {}, 420, 420)[0])

    def test_row_count_mismatch_and_tol(self):
        ok, why = DC.verdict(GOOD, [{'key': 'k'}], {'L1_iou': 0.03}, 420, 419)
        self.assertFalse(ok)
        self.assertEqual(why, ['row_count:420!=419', 'discrete_mismatch:1', 'over_tol:L1_iou'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
