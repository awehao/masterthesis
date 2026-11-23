#!/usr/bin/env python3
"""door_k3a_model_replay 的反例：索引契約（s_k 作用於 [k, k+1)）、缺值即停不重設、d₀ 只取 OPEN 時已可取得者。

    python3 evaluation/test_door_k3a_model_replay.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3a_model_replay as MR                    # noqa: E402

A6 = [0.1] * 6


class Replay(unittest.TestCase):
    def test_index_contract_uses_s_k(self):
        """s_k 只影響 q̂_{k+1}：第 0 步設定點 1、第 1 步設定點 5 ⇒ q̂_1＝0.1、q̂_2＝0.9·0.1＋0.1·5。"""
        sp = {0: [1.0] * 6, 1: [5.0] * 6}
        qh, stop = MR.replay(np.zeros(6), lambda k: (sp[k], None), np.zeros(6), A6, 0, 2)
        self.assertIsNone(stop)
        np.testing.assert_allclose(qh[1], 0.1)
        np.testing.assert_allclose(qh[2], 0.9 * 0.1 + 0.5)

    def test_offset_and_convergence(self):
        qh, _ = MR.replay(np.zeros(6), lambda k: ([1.0] * 6, None), [0.2] * 6, A6, 0, 400)
        np.testing.assert_allclose(qh[400], 1.2, atol=1e-12)

    def test_per_axis_alpha_not_rounded(self):
        al = [0.09500983461563241, 0.095, 0.09, 0.1, 0.2, 0.05]
        qh, _ = MR.replay(np.zeros(6), lambda k: ([1.0] * 6, None), np.zeros(6), al, 0, 1)
        np.testing.assert_allclose(qh[1], al, atol=0)

    def test_gap_stops_without_reset(self):
        def sp(k):
            return (None, 'evidence 列缺') if k == 3 else ([1.0] * 6, None)
        qh, stop = MR.replay(np.zeros(6), sp, np.zeros(6), A6, 0, 10)
        self.assertEqual(stop, {'step': 3, 'why': 'evidence 列缺'})
        self.assertEqual(sorted(qh), [0, 1, 2, 3])                                     # 第 3 步之後不再有值

    def test_d0_only_available_before_open(self):
        log = [{'solve_in': {'snap_sim_t_exact': 19.90, 'offset_d_hat': [1] * 6}},
               {'solve_in': {'snap_sim_t_exact': 19.95, 'offset_d_hat': [2] * 6}},
               {'solve_in': {'snap_sim_t_exact': 20.05, 'offset_d_hat': [3] * 6}},
               {'other': 1}]
        d0, ts = MR.d0_at(log, 20.0)
        self.assertEqual(d0, [2] * 6)
        self.assertAlmostEqual(ts, 19.95)
        self.assertEqual(MR.d0_at(log, 19.0), (None, None))

    def test_missing_dhat_is_none(self):
        d0, ts = MR.d0_at([{'solve_in': {'snap_sim_t_exact': 1.0}}], 2.0)
        self.assertIsNone(d0)
        self.assertEqual(ts, 1.0)

    def test_d0_published_conservative(self):
        """快照在 OPEN 前、但發布在 OPEN 後 ⇒ 不可用；取 OPEN 前已發布者。"""
        log = [{'published': True, 'publish_sim_t_exact': 19.97, 'solve_in': {'snap_sim_t_exact': 19.95, 'solve_start_sim_t_exact': 19.96, 'offset_d_hat': [1] * 6}},
               {'published': True, 'publish_sim_t_exact': 20.03, 'solve_in': {'snap_sim_t_exact': 20.00, 'solve_start_sim_t_exact': 20.02, 'offset_d_hat': [2] * 6}},
               {'published': False, 'publish_sim_t_exact': 19.99, 'solve_in': {'snap_sim_t_exact': 19.98, 'offset_d_hat': [3] * 6}}]
        D = MR.d0_published(log, 20.0)
        self.assertEqual(D['d_hat'], [1] * 6)
        self.assertEqual((D['snap_t'], D['solve_start_t'], D['publish_t']), (19.95, 19.96, 19.97))
        self.assertIsNone(MR.d0_published(log, 19.0)['d_hat'])

    def test_windows_separate_trigger(self):
        w = [MR.window_of(k, 10, 15) for k in (9, 10, 11, 14, 15)]
        self.assertEqual(w, ['main', 'first_low', 'intermittent', 'intermittent', 'trigger'])
        self.assertEqual(MR.window_of(20, None, None), 'main')


if __name__ == '__main__':
    unittest.main(verbosity=2)
