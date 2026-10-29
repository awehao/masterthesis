#!/usr/bin/env python3
"""door_k3m_mode2 最小核對：approach → hold（seq 1、錨點、base_slow）→ free（seq 2，r3 語義）三端一致；序號錯置、重送冪等、中止優先。"""
import types
import unittest

import door_k3m_mode2 as M


class T(unittest.TestCase):
    def test_full_sequence(self):
        ex, sh, sf = M.ExecutorModeAuthority2('R'), M.SolverHoldSwitch('R'), M.SolverFreeSwitchSeq('R')
        w1, w2 = M.TaskModeWaiter2('R', 1), M.TaskFreeWaiterSeq('R')
        rq = w1.make_request(100, 1.0, 10.0)
        k, n = ex.on_request(rq, 101, 1.01, False, True, pose=(0.6, -0.53, 1.57), base_slow=False)
        self.assertEqual((k, n['why']), ('reject', 'base_not_slow'))
        k, a1 = ex.on_request(rq, 102, 1.02, False, True, pose=(0.6, -0.53, 1.57), base_slow=True)
        self.assertEqual(k, 'switch')
        self.assertEqual(a1['anchor'], [0.6, -0.53, 1.57])
        self.assertEqual(ex.on_request(rq, 103, 1.03, False, True, pose=(0, 0, 0), base_slow=True)[0], 'repeat')
        self.assertEqual(sf.on_executor_ack(a1, False, False), 'not_mine')
        self.assertIsNone(sh.on_executor_ack(a1, False))
        self.assertEqual(sh.apply(snap_t=1.03), (0.6, -0.53, 1.57))
        s1 = sh.note_first_solution(7, 1.05, True, 1.06)
        w1.on_executor(a1, 1.03, 10.1)
        w1.on_solver(s1, 1.07, 10.2)
        self.assertEqual(w1.status(1.07, 10.2), 'confirmed')
        rq2 = w2.make_request(200, 3.0, 20.0)
        self.assertEqual(rq2['seq'], 2)
        k, a2 = ex.on_request(rq2, 201, 3.01, False, True)
        self.assertEqual((k, a2['mode']), ('switch', 'free'))
        self.assertEqual(sh.on_executor_ack(a2, False), 'not_mine')
        self.assertIsNone(sf.on_executor_ack(a2, False, True))
        self.assertIsNone(sf.on_executor_ack(a2, False, True))                       # 重送冪等
        cfg = types.SimpleNamespace(base_hold=(1, 2, 3), base_fixed=False)
        self.assertTrue(sf.apply(cfg, shaping_enabled=False, snap_t=3.02))
        self.assertIsNone(cfg.base_hold)
        s2 = sf.note_first_solution(9, 3.05, [0.0] * 9, [0.0] * 9, True, 3.06)
        self.assertEqual(s2['seq'], 2)
        w2.on_executor(a1, 3.02, 20.05)                                              # 前一序號 ack ⇒ 忽略
        w2.on_executor(a2, 3.02, 20.1)
        w2.on_executor(a2, 3.03, 20.15)
        w2.on_solver(s1, 3.06, 20.16)
        w2.on_solver(s2, 3.07, 20.2)
        self.assertEqual(w2.status(3.07, 20.2), 'confirmed')

    def test_out_of_order_and_abort(self):
        ex = M.ExecutorModeAuthority2('R')
        rq2 = {'run_id': 'R', 'seq': 2, 'mode': 'free', 'req_step': 5, 'req_t': 0.05}
        self.assertEqual(ex.on_request(rq2, 6, 0.06, False, True)[1]['why'], 'seq_out_of_order')
        rq1 = {'run_id': 'R', 'seq': 1, 'mode': 'free', 'req_step': 5, 'req_t': 0.05}
        self.assertEqual(ex.on_request(rq1, 6, 0.06, False, True)[1]['why'], 'mode_invalid')
        rq1['mode'] = 'hold'
        self.assertEqual(ex.on_request(rq1, 6, 0.06, True, True, pose=(0, 0, 0), base_slow=True)[1]['why'], 'after_abort')
        self.assertEqual(ex.on_request(rq1, 6, 0.06, False, False, pose=(0, 0, 0), base_slow=True)[1]['why'], 'not_wholebody_owner')


if __name__ == '__main__':
    unittest.main()
