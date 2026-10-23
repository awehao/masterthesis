#!/usr/bin/env python3
"""door_k3ab_balance 的反例：調整律數值、條件積分、缺步／時間／無效即停、重送不重積分、批次＝逐筆、未來資料不提前使用。

    python3 evaluation/test_door_k3ab_balance.py
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3ab_balance as B                          # noqa: E402


def entries(s0, n, f=lambda k: (2.0, 2.0), t0=None):
    out = []
    for k in range(n):
        s = s0 + k
        f1, f2 = f(k)
        out.append((s, round((s if t0 is None else t0 + k) * 0.01, 4), f1, f2, True))
    return out


def raw(es):
    return {'pairs': [list(e) for e in es]}


class Law(unittest.TestCase):
    def test_balanced_stays_zero(self):
        L = B.BalanceLaw()
        L.start(100, 1.0)
        L.update(entries(101, 50))
        self.assertEqual(L.delta, 0.0)
        self.assertIsNone(L.stopped)

    def test_numeric_law_and_filter_init(self):
        c = B.BalanceConfig()
        L = B.BalanceLaw(c)
        L.start(0, 0.0)
        L.update(entries(1, 3, f=lambda k: (1.0, 3.0 + k)))          # e＝2, 3, 4
        ef = 2.0
        d = 0.0
        for k, e in enumerate((2.0, 3.0, 4.0)):
            if k:
                ef = ef + c.beta * (e - ef)
            u = math.copysign(max(abs(ef) - 0.5, 0.0), ef)
            d = d + max(-1e-3, min(1e-3, 0.25e-3 * u)) * 0.01
            self.assertAlmostEqual(L.records[k]['ef'], ef, places=15)
            self.assertAlmostEqual(L.records[k]['delta'], d, places=18)
        self.assertAlmostEqual(L.records[0]['ef'], 2.0, places=15)          # 登錄初值＝首步 e

    def test_deadband(self):
        L = B.BalanceLaw()
        L.start(0, 0.0)
        L.update(entries(1, 100, f=lambda k: (2.0, 2.45)))
        self.assertEqual(L.delta, 0.0)

    def test_rate_and_bound_and_reverse(self):
        L = B.BalanceLaw()
        L.start(0, 0.0)
        L.update(entries(1, 200, f=lambda k: (0.5, 9.5)))              # e＝9 ⇒ k_i·u＝2.1 mm/s ⇒ 截在 1 mm/s
        r = L.records
        self.assertTrue(all(abs(b['delta'] - a['delta']) <= 1e-5 + 1e-18 for a, b in zip(r, r[1:])))
        self.assertAlmostEqual(L.delta, 1e-3, places=15)
        self.assertTrue(r[-1]['sat'])
        k_sat = next(i for i, x in enumerate(r) if x['sat'])
        self.assertGreaterEqual(k_sat, 99)                                 # 0→1 mm 至少 1 s（100 步）
        L.update(entries(201, 30, f=lambda k: (9.5, 0.5), t0=201))         # 反向：立即離開上限
        self.assertLess(L.delta, 1e-3)
        self.assertLess(L.records[-1]['delta'], L.records[-2]['delta'])

    def test_gap_stops_without_resume(self):
        L = B.BalanceLaw()
        L.start(0, 0.0)
        L.update(entries(1, 3, f=lambda k: (1.0, 4.0)))
        d = L.delta
        L.update(entries(5, 3, f=lambda k: (1.0, 4.0)))                    # 缺第 4 步
        self.assertEqual(L.stopped['why'], 'gap')
        L.update(entries(4, 3, f=lambda k: (1.0, 4.0)))                    # 補來也不恢復
        self.assertEqual(L.delta, d)
        self.assertEqual(len(L.records), 3)

    def test_time_step_and_invalid(self):
        L = B.BalanceLaw()
        L.start(0, 0.0)
        L.update([(1, 0.01, 1.0, 2.0, True), (2, 0.0305, 1.0, 2.0, True)])
        self.assertEqual(L.stopped['why'], 'time_step')
        for bad in [(1, 0.01, 1.0, 2.0, False), (1, 0.01, None, 2.0, True), (1, 0.01, -0.1, 2.0, True)]:
            L = B.BalanceLaw()
            L.start(0, 0.0)
            L.update([bad])
            self.assertEqual(L.stopped['why'], 'invalid')
            self.assertEqual(L.delta, 0.0)

    def test_batch_equals_one_by_one(self):
        es = entries(1, 120, f=lambda k: (2.0 - 0.01 * k, 2.0 + 0.01 * k))
        A = B.BalanceLaw()
        A.start(0, 0.0)
        A.update(es)
        C = B.BalanceLaw()
        C.start(0, 0.0)
        for e in es:
            C.update([e])
        self.assertEqual(A.records, C.records)


class Buffer(unittest.TestCase):
    def test_take_leaves_future(self):
        b = B.PairBuffer()
        b.push(raw(entries(1, 10)), 0.0)
        got = b.take(6)
        self.assertEqual([e[0] for e in got], [1, 2, 3, 4, 5, 6])
        self.assertEqual(sorted(b.pending), [7, 8, 9, 10])
        self.assertEqual([e[0] for e in b.take(10)], [7, 8, 9, 10])

    def test_resend_not_reintegrated(self):
        b = B.PairBuffer()
        L = B.BalanceLaw()
        L.start(0, 0.0)
        es = entries(1, 5, f=lambda k: (1.0, 4.0))
        b.push(raw(es), 0.0)
        L.update(b.take(5))
        d = L.delta
        b.push(raw(es), 0.1)                                               # 同值重送
        L.update(b.take(10))
        self.assertEqual(L.delta, d)
        self.assertEqual(b.n_dup, 5)
        self.assertIsNone(b.fault)

    def test_conflicting_resend_faults(self):
        b = B.PairBuffer()
        b.push(raw(entries(1, 3)), 0.0)
        b.push({'pairs': [[2, 0.02, 2.0, 2.5, True]]}, 0.1)
        self.assertEqual(b.fault, 'conflict')

    def test_malformed(self):
        for msg in (None, {'pairs': 3}, {'pairs': [[1, 0.01, 2.0, 2.0]]}, {'pairs': [[1.5, 0.01, 2.0, 2.0, True]]},
                    {'pairs': [[1, float('nan'), 2.0, 2.0, True]]}, {'pairs': [[1, 0.01, 'x', 2.0, True]]},
                    {'pairs': [[1, 0.01, 2.0, 2.0, 1]]}, {'pairs': [[True, 0.01, 2.0, 2.0, True]]},
                    {'pairs': [[1, 0.01, float('inf'), 2.0, True]]}):
            b = B.PairBuffer()
            b.push(msg, 0.0)
            self.assertEqual(b.fault, 'malformed', msg)

    def test_discard_and_late_arrival(self):
        b = B.PairBuffer()
        b.push(raw(entries(1, 5)), 0.0)
        b.discard_upto(5)
        self.assertEqual(b.pending, {})
        b.push(raw([(3, 0.03, 2.0, 2.0, True)]), 0.1)                       # 重送：同值 ⇒ 忽略
        b.push(raw([(0, 0.0, 2.0, 2.0, True)]), 0.1)                        # 丟棄區的遲到筆 ⇒ 只計數
        self.assertEqual(b.n_late_discarded, 1)
        self.assertEqual(b.take(5), [])

    def test_age(self):
        b = B.PairBuffer()
        self.assertEqual(b.age(1.0), math.inf)
        b.push(raw(entries(1, 1)), 2.0)
        self.assertAlmostEqual(b.age(2.3), 0.3)
        b.push({'pairs': []}, 5.0)                                          # 空訊息不更新年齡
        self.assertAlmostEqual(b.age(2.3), 0.3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
