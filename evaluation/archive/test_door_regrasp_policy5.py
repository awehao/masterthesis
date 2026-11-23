#!/usr/bin/env python3
"""door_regrasp_refclock／door_regrasp_policy5 驗證（Codex r62 §3：mode 5 預抓取節點閘控）。
參考包＝door_regrasp_ref_v3 加 segments['5']['stage']（由來源時間化 v5 依 src_index 取；只供本測試，不寫檔）。
  Z1 時鐘單元：τ_pre 前照推；到節點停住（hold_pre）；到達後自 τ_pre 續推；NaN 量測不放行；進入後 τ_pre＋10 s 未到 ⇒ pregrasp_timeout；
  Z2 理想追蹤（假世界 TCP＝目標）⇒ DONE，時鐘事件 pregrasp_reached、停住時間 ≤ 一個週期；
  Z3 mode 5 實測 TCP 恆偏 10 mm（未達 5 mm）⇒ rg5_pregrasp_timeout，中止時刻＝進入＋τ_pre＋10 s（±0.1 s），且未請求 mode 6；
  Z4 參考包缺 stage ⇒ 建構 ValueError。

    python3 evaluation/test_door_regrasp_policy5.py
"""
import copy
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_policy_p2 as PP2                       # noqa: E402
import door_regrasp_policy5 as RP5                     # noqa: E402
import door_regrasp_refclock as RC                     # noqa: E402
import test_door_regrasp_policy1 as T1                 # noqa: E402
import test_door_regrasp_policy3 as T3                 # noqa: E402
import test_door_regrasp_policy4 as T4                 # noqa: E402

V3 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v3.json')))
_R = json.load(open(os.path.join(HERE, V3['source'])))['reference_20hz']['rows']
V3S = copy.deepcopy(V3)
for _k, _sg in V3S['segments'].items():
    _sg['stage'] = [_R[i][4] for i in _sg['src_index']]


def mk5():
    p = T4.mk4()
    p.__class__ = RP5.RegraspPolicy5
    p.i_pre = RC.pregrasp_index(V3S['segments']['5']['stage'])
    p.clock5 = None
    p.ref.p = json.loads(json.dumps(V3S))
    return p


class TZ(unittest.TestCase):
    def setUp(self):
        self._orig = PP2.K3MP2Policy.step
        self._pkg = T1.PKG
        PP2.K3MP2Policy.step = T1._fake_parent_step
        T1.PKG = V3S

    def tearDown(self):
        PP2.K3MP2Policy.step = self._orig
        T1.PKG = self._pkg

    def test_z1_clock(self):
        c = RC.PregraspClock(10, 0.05, pre_tol_m=0.005, pre_wait_s=10.0)
        c.start(100.0)
        far, pre = np.r_[0.1, 0, 0], np.zeros(3)
        tau, s = c.step(100.2, far, pre)
        self.assertAlmostEqual(tau, 0.2)
        self.assertEqual(s, 'approach')
        self.assertEqual(c.step(101.0, far, pre), (0.5, 'hold_pre'))
        self.assertEqual(c.step(103.0, np.r_[np.nan, 0, 0], pre), (0.5, 'hold_pre'))
        self.assertEqual(c.step(104.0, np.r_[0.004, 0, 0], pre), (0.5, 'insert'))
        tau, s = c.step(104.3, far, pre)
        self.assertAlmostEqual(tau, 0.8)
        self.assertEqual(s, 'insert')
        c2 = RC.PregraspClock(10, 0.05)
        c2.start(0.0)
        self.assertEqual(c2.step(10.5, far, pre)[1], 'hold_pre')
        self.assertEqual(c2.step(10.6, far, pre)[1], 'pregrasp_timeout')

    def test_z2_ideal_done(self):
        p = mk5()
        w = T3.Fake3(p).run(220.0)
        self.assertEqual(p.phase, 'DONE', p.abort)
        self.assertIn(6, w.t_commit)
        ev = [e for e in p.clock5.events if e['what'] == 'pregrasp_reached']
        self.assertEqual(len(ev), 1)
        self.assertLessEqual(ev[0]['held_s'], T1.DT + 1e-6)

    def test_z3_timeout(self):
        p = mk5()
        orig = p.step

        def step(m, dt):
            if p.rg.get('k_committed') == 5 and getattr(m, 'T_tcp', None) is not None:
                T = np.asarray(m.T_tcp, float).copy()
                T[2, 3] += 0.010
                m.T_tcp = T
            return orig(m, dt)
        p.step = step
        w = T3.Fake3(p).run(220.0)
        self.assertEqual(p.abort, 'rg5_pregrasp_timeout')
        t_ab = [e for e in p.events if e.get('abort') == 'rg5_pregrasp_timeout'] or [{'t': p.t_abort}]
        self.assertAlmostEqual(float(t_ab[0]['t']), p.rg['t_enter5'] + p.clock5.tau_pre + 10.0, delta=0.1)
        self.assertNotIn(6, w.req_t)

    def test_z4_missing_stage(self):
        with self.assertRaises(ValueError):
            RP5.RegraspPolicy5(ref_pkg=V3)


if __name__ == '__main__':
    unittest.main(verbosity=1)
