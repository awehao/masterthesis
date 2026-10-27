#!/usr/bin/env python3
"""door_coopen_closure_guidance2（C2 r2）測試（Codex r36 §2 A／B 必補反例）。
  Y1 Codex 反例：o＝(8/√2, 8/√2) mm、dt 0.05、目標 (20, 0) mm ⇒ 每輪輸出變化範數 ≤ 0.1 mm、|o| ≤ 8 mm（r1 的 y 變化 0.100868 mm 不再出現）。
  Y2 對角飽和長程：目標 (30, 30) mm 跑 2000 輪 ⇒ 每輪 |Δo| ≤ rate·dt、|o| ≤ 8 mm、收斂到 8 mm 對角。
  Y3 啟用權重變化：門角 30→45° 線性掃過（36–40 smoothstep）⇒ 每輪 |Δo| ≤ rate·dt；36° 以下 o 維持 0。
  Y4 相位邊界：closure_cycle 在 OPEN→SETTLE 持續更新、SETTLE→PAUSE 不再更新且輸出沿用（不撤除、不跳變）；非 free 或 G_T_H 缺 ⇒ not_active。
  Y5 同步步號：重送同一步／倒退 ⇒ 不更新、不刷新（why step_not_advanced）；非整數步 ⇒ 同。
  Y6 無效輸入：非有限 TCP ⇒ input_invalid、沿用；連續 ≥ 0.2 s（模擬時間）⇒ timeout ⇒ closure_cycle 回 closure_input_invalid；有效輸入一到即清除。
  Y7 參考紀錄：ref_mod＝ref_orig＋(o, 0)（偏航不變）；o＝0 時兩者相同；ShiftedTable 的 dpose_dq 與原表相同。
  Y8 r1 與 r2 在 A/B 正常條件（無限速、無截斷、s 從 0 平滑上升）下的差異量級（只報，不作判準）。

    python3 evaluation/test_door_coopen_closure_guidance2.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_closure_guidance2 as C2             # noqa: E402
import door_wide_guidance as G                         # noqa: E402

LIM = 0.002 * 0.05


class Tab:
    def pose(self, q):
        return (1.0 + 0.1 * q, -0.5 - 0.2 * q, 3.0 + 0.05 * q)

    def dpose_dq(self, q):
        return (0.1, -0.2, 0.05)


def T(x, y):
    m = np.eye(4)
    m[0, 3], m[1, 3] = x, y
    return m


class Y(unittest.TestCase):
    def test_y1_codex_counterexample(self):
        c = C2.ClosureCorrector2()
        c.o = np.array([0.008 / math.sqrt(2), 0.008 / math.sqrt(2)])
        o0 = c.o.copy()
        r = c.update(1, 1.0, 0.05, 45.0, [0.0, 0.0], [0.02, 0.0])
        d = np.array(r['o']) - o0
        self.assertLessEqual(float(np.linalg.norm(d)), LIM + 1e-15)
        self.assertLessEqual(abs(d[1]), LIM + 1e-15)
        self.assertLessEqual(float(np.linalg.norm(r['o'])), 0.008 + 1e-15)

    def test_y2_diagonal_saturation(self):
        c = C2.ClosureCorrector2()
        prev = c.o.copy()
        for k in range(2000):
            r = c.update(k + 1, k * 0.05, 0.05, 45.0, [0.0, 0.0], [0.03, 0.03])
            o = np.array(r['o'])
            self.assertLessEqual(float(np.linalg.norm(o - prev)), LIM + 1e-15)
            self.assertLessEqual(float(np.linalg.norm(o)), 0.008 + 1e-15)
            prev = o
        np.testing.assert_allclose(prev, [0.008 / math.sqrt(2)] * 2, atol=1e-9)

    def test_y3_enable_weight_change(self):
        c = C2.ClosureCorrector2()
        prev = c.o.copy()
        for k, q in enumerate(np.linspace(30.0, 45.0, 600)):
            r = c.update(k + 1, k * 0.05, 0.05, float(q), [0.0, 0.0], [0.006, -0.004])
            o = np.array(r['o'])
            self.assertLessEqual(float(np.linalg.norm(o - prev)), LIM + 1e-15)
            if q <= 36.0:
                self.assertEqual(float(np.linalg.norm(o)), 0.0)
            prev = o

    def test_y4_phase_boundaries(self):
        c, tab, GTH = C2.ClosureCorrector2(), Tab(), np.eye(4)
        q = math.radians(45.0)
        outs = []
        for k, ph in enumerate(['OPEN'] * 40 + ['SETTLE'] * 20 + ['PAUSE'] * 20):
            _, rec, ab = C2.closure_cycle(c, tab, ph, 'free', GTH, k + 1, k * 0.05, 0.05, q, T(0.0, 0.0), T(0.005, 0.0))
            outs.append((ph, rec['why'], np.array(rec['o'])))
            self.assertIsNone(ab)
        self.assertTrue(all(w is None for p, w, _ in outs if p in ('OPEN', 'SETTLE')))
        self.assertTrue(all(w == 'not_active' for p, w, _ in outs if p == 'PAUSE'))
        last_settle = [o for p, _, o in outs if p == 'SETTLE'][-1]
        for p, _, o in outs:
            if p == 'PAUSE':
                np.testing.assert_array_equal(o, last_settle)
        for ms, gth in (('hold', GTH), ('free', None)):
            _, rec, _ = C2.closure_cycle(C2.ClosureCorrector2(), tab, 'OPEN', ms, gth, 1, 0.0, 0.05, q, T(0, 0), T(0.005, 0))
            self.assertEqual(rec['why'], 'not_active')

    def test_y5_step(self):
        c = C2.ClosureCorrector2()
        c.update(10, 1.0, 0.05, 45.0, [0, 0], [0.005, 0])
        o = list(c.o)
        for st in (10, 9, 10.0, True, None):
            r = c.update(st, 1.05, 0.05, 45.0, [0, 0], [0.005, 0])
            self.assertEqual(r['why'], 'step_not_advanced')
            self.assertEqual(list(c.o), o)
        self.assertEqual(c.last_step, 10)

    def test_y6_invalid_timeout(self):
        c, tab = C2.ClosureCorrector2(), Tab()
        q = math.radians(45.0)
        C2.closure_cycle(c, tab, 'OPEN', 'free', np.eye(4), 1, 1.0, 0.05, q, T(0, 0), T(0.005, 0))
        o = list(c.o)
        bad = T(0, 0)
        bad[0, 3] = float('nan')
        res = []
        for k in range(6):
            _, rec, ab = C2.closure_cycle(c, tab, 'OPEN', 'free', np.eye(4), 2 + k, 1.05 + 0.05 * k, 0.05, q, bad, T(0.005, 0))
            res.append((rec['why'], ab))
            self.assertEqual(list(c.o), o)
        self.assertEqual(res[0], ('input_invalid', None))
        self.assertEqual(res[3], ('input_invalid', None))                # 0.15 s
        self.assertEqual(res[4], ('input_invalid', 'closure_input_invalid'))   # 0.20 s
        _, rec, ab = C2.closure_cycle(c, tab, 'OPEN', 'free', np.eye(4), 20, 2.0, 0.05, q, T(0, 0), T(0.005, 0))
        self.assertIsNone(rec['why'])
        self.assertIsNone(c.invalid_since)

    def test_y7_ref_record(self):
        c, tab = C2.ClosureCorrector2(), Tab()
        q = math.radians(45.0)
        for k in range(100):
            tbl, rec, _ = C2.closure_cycle(c, tab, 'OPEN', 'free', np.eye(4), k + 1, k * 0.05, 0.05, q, T(0, 0), T(0.004, -0.003))
        np.testing.assert_allclose(np.array(rec['ref_mod'][:2]) - np.array(rec['ref_orig'][:2]), rec['o'], atol=1e-15)
        self.assertEqual(rec['ref_mod'][2], rec['ref_orig'][2])
        self.assertEqual(tbl.dpose_dq(q), tab.dpose_dq(q))
        cfg = G.GuidanceCfg(k_lin=1.0, k_yaw=1.0, v_cap=0.035, w_cap=0.10)
        _, rec0, _ = C2.closure_cycle(C2.ClosureCorrector2(), tab, 'OPEN', 'free', np.eye(4), 1, 0.0, 0.05, math.radians(30.0), T(0, 0), T(0.004, 0))
        self.assertEqual(rec0['ref_mod'], rec0['ref_orig'])
        a, _ = G.guidance(tab, cfg, tab.pose(q), q, 0.01)
        b, _ = G.guidance(C2.ShiftedTable(tab, (0, 0)), cfg, tab.pose(q), q, 0.01)
        np.testing.assert_array_equal(np.array(a, float), np.array(b, float))

    def test_y8_r1_vs_r2_normal(self):
        import door_coopen_closure_guidance as C1
        c1, c2 = C1.ClosureCorrector(), C2.ClosureCorrector2()
        diff = 0.0
        for k, q in enumerate(np.linspace(30.0, 48.0, 1200)):
            e = [0.0035 * C2.smoothstep(q, 40, 47), 0.0]
            c1.update(0.05, float(q), [0, 0], e)
            c2.update(k + 1, k * 0.05, 0.05, float(q), [0, 0], e)
            diff = max(diff, float(np.linalg.norm(c1.ref_offset() - c2.o)))
        print(f'  r1 vs r2 正常條件下輸出最大差 {diff * 1e3:.3f} mm（只報）')
        self.assertLess(diff, 0.002)


if __name__ == '__main__':
    unittest.main(verbosity=2)
