#!/usr/bin/env python3
"""door_regrasp_contact_window 驗證。
  W1 mode 6 提交步（權威模式 6）即有 bar_in 2 N ⇒ 不違規（r53 §3 提交當輪）；同步在 mode 5 ⇒ 違規；
  W2 mode 1–2 只允許 bar_out：post_out0 0.2 N ⇒ 違規；ambiguous 0.2 N ⇒ 違規；panel ⇒ 違規；
  W3 mode 3：未離開 bar_out 允許；exited 後 bar_out 0.2 N ⇒ 違規（閂鎖由呼叫端帶入）；
  W4 非手指任何類別 > 0.1 ⇒ 違規；第一筆違規閂鎖不被後續覆寫；
  W5 摘要：依模式分組、來源步／時刻、跨模式窗口兩組都在；空窗口 ⇒ None；紀錄錯步／缺口 ⇒ complete False（不當零接觸）。

    python3 evaluation/test_door_regrasp_contact_window.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_regrasp_contact_window as CW               # noqa: E402

F1 = 'uflite_finger1'


def rec(pts, step, t):
    return {'status': 'ok', 'complete': True, 'step': step, 't': t, 'points': [{'robot_link': l, 'class': c, 'f': f} for l, c, f in pts]}


def run(seq):
    w = CW.ContactWindow()
    for k, (mode, pts, ex) in enumerate(seq, start=1):
        t = round(k * 0.01, 6)
        w.step(k, t, mode, rec(pts, k, t), ex)
    return w


class TW(unittest.TestCase):
    def test_w1(self):
        self.assertIsNone(run([(5, [], False), (6, [(F1, 'bar_in', 2.0)], False)]).violation)
        self.assertEqual(run([(5, [(F1, 'bar_in', 2.0)], False)]).violation['cls'], 'bar_in')

    def test_w2(self):
        for c in ('post_out0', 'ambiguous', 'panel'):
            self.assertEqual(run([(2, [(F1, c, 0.2)], False)]).violation['cls'], c)
        self.assertIsNone(run([(1, [(F1, 'bar_out', 3.0)], False), (2, [(F1, 'bar_out', 0.5)], False)]).violation)

    def test_w3(self):
        self.assertIsNone(run([(3, [(F1, 'bar_out', 0.2)], False)]).violation)
        v = run([(3, [], True), (3, [(F1, 'bar_out', 0.2)], True)]).violation
        self.assertTrue(v['exited'] and v['cls'] == 'bar_out')

    def test_w4(self):
        v = run([(6, [('link6', 'bar_in', 0.2)], False), (6, [('link5', 'panel', 5.0)], False)]).violation
        self.assertEqual((v['link'], v['step']), ('link6', 1))

    def test_w5(self):
        w = CW.ContactWindow()
        self.assertIsNone(w.take_summary())
        w.step(1, 0.01, 5, rec([], 1, 0.01), False)
        w.step(2, 0.02, 6, rec([(F1, 'bar_in', 1.5)], 2, 0.02), False)
        s = w.take_summary()
        self.assertEqual((s['src_step_lo'], s['src_step'], s['src_t'], s['complete']), (1, 2, 0.02, True))
        self.assertEqual(set(s['by_mode']), {'5', '6'})
        self.assertEqual(s['by_mode']['6'][F1]['bar_in'], 1.5)
        self.assertIsNone(w.take_summary())
        w.step(3, 0.03, 6, rec([], 2, 0.03), False)                          # 錯步紀錄
        self.assertFalse(w.take_summary()['complete'])
        w.step(5, 0.05, 6, rec([], 5, 0.05), False)                          # 缺步
        self.assertFalse(w.take_summary()['complete'])


if __name__ == '__main__':
    unittest.main(verbosity=1)
