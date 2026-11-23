#!/usr/bin/env python3
"""door_coopen_approach_cost（接近段成本切換純邏輯）測試（Codex r13 §3-1／§2）。
  P1 APPROACH ⇒ 與 off_coord 只差 [12]＝0.01（其他 13 欄逐位相同，含 NaN）。
  P2 其他相位（WAIT_HANDOVER、UNFOLD、ALIGN_HOLD、ENGAGE、OPEN、SETTLE、PAUSE、DONE、ABORTED）⇒ 輸出＝輸入（逐位，含 NaN 與引導開啟值）。
  P3 閂鎖：APPROACH → ENGAGE → APPROACH ⇒ 第二次 APPROACH 仍 NaN；t_exit／exit_phase 記第一次離開。
  P4 中止：APPROACH → ABORTED ⇒ NaN、exited。
  P5 防禦：APPROACH 但輸入引導開啟（w_vref ≠ 0）⇒ 不覆寫、n_conflict 1。
  P6 登錄檢查：缺鍵、多鍵、w_a ≤ 0／NaN、index ≠ 12、latch ≠ True ⇒ ValueError；coord 長度錯 ⇒ ValueError；輸入 list 不被改動。
  P7 登錄檔 r7 實檔可建構、值＝Codex 指定。

    python3 evaluation/test_door_coopen_approach_cost.py
"""
import json
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_approach_cost as AC                 # noqa: E402
import door_wide_guidance as WG                         # noqa: E402

REG = {'w_a': 0.01, 'phases': ['APPROACH'], 'latch_after_exit': True, 'coord_index': 12, 'launch_w_a': 0.05, 'restore_deadline_s': 0.55}


def same(a, b):
    return len(a) == len(b) and all((math.isnan(x) and math.isnan(y)) or x == y for x, y in zip(a, b))


class P(unittest.TestCase):
    def test_p1_approach(self):
        ac = AC.ApproachCost(REG)
        off = WG.off_coord()
        out, r = ac.apply(off, 'APPROACH', 10.6)
        self.assertEqual(out[12], 0.01)
        self.assertTrue(same(out[:12] + out[13:], off[:12] + off[13:]))
        self.assertTrue(r['on'])
        self.assertEqual(ac.t_first_on, 10.6)

    def test_p2_other_phases(self):
        on = [1.0, 1.0, 0.005, 0.0, 0.01, 0.0] + [float('nan')] * 6 + [float('nan'), float('nan')]
        for ph in ('WAIT_HANDOVER', 'UNFOLD', 'ALIGN_HOLD', 'ENGAGE', 'OPEN', 'SETTLE', 'PAUSE', 'DONE', 'ABORTED'):
            for c in (WG.off_coord(), on):
                ac = AC.ApproachCost(REG)
                out, r = ac.apply(c, ph, 1.0)
                self.assertTrue(same(out, c), ph)
                self.assertFalse(r['on'])

    def test_p3_latch(self):
        ac = AC.ApproachCost(REG)
        ac.apply(WG.off_coord(), 'APPROACH', 10.0)
        o2, _ = ac.apply(WG.off_coord(), 'ENGAGE', 12.0)
        self.assertTrue(math.isnan(o2[12]))
        o3, r3 = ac.apply(WG.off_coord(), 'APPROACH', 12.05)
        self.assertTrue(math.isnan(o3[12]))
        self.assertEqual(r3['why'], 'latched')
        self.assertEqual((ac.t_exit, ac.exit_phase), (12.0, 'ENGAGE'))

    def test_p4_abort(self):
        ac = AC.ApproachCost(REG)
        ac.apply(WG.off_coord(), 'APPROACH', 10.0)
        o, _ = ac.apply(WG.off_coord(), 'ABORTED', 25.6)
        self.assertTrue(math.isnan(o[12]))
        self.assertTrue(ac.exited)
        self.assertEqual(ac.exit_phase, 'ABORTED')

    def test_p5_guidance_conflict(self):
        ac = AC.ApproachCost(REG)
        on = [1.0, 1.0, 0.005, 0.0, 0.01, 0.0] + [float('nan')] * 6 + [float('nan'), float('nan')]
        o, r = ac.apply(on, 'APPROACH', 10.0)
        self.assertTrue(same(o, on))
        self.assertEqual(ac.n_conflict, 1)
        self.assertEqual(r['why'], 'guidance_on')

    def test_p6_validation(self):
        bad = [{k: v for k, v in REG.items() if k != 'w_a'}, dict(REG, extra=1), dict(REG, w_a=0.0), dict(REG, w_a=float('nan')),
               dict(REG, coord_index=11), dict(REG, latch_after_exit=False), None]
        for b in bad:
            with self.assertRaises((ValueError, TypeError)):
                AC.ApproachCost(b)
        ac = AC.ApproachCost(REG)
        with self.assertRaises(ValueError):
            ac.apply([1.0] * 13, 'APPROACH', 0.0)
        c = WG.off_coord()
        c0 = list(c)
        ac.apply(c, 'APPROACH', 0.0)
        self.assertTrue(same(c, c0))

    def test_p7_registry_file(self):
        reg = json.load(open(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r7.json')))['approach_cost']
        self.assertEqual(reg, REG)
        AC.ApproachCost(reg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
