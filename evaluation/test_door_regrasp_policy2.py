#!/usr/bin/env python3
"""door_regrasp_policy2 驗證（沿用 test_door_regrasp_policy1 的假世界）。
  Q1 未接幾何後端 ⇒ 不請求 mode 4（事件記 pair_geometry_missing），參考走完後 10 s ⇒ rg3_gate_timeout；
  Q2 接真 v3 後端＋既有 GeometryMonitor（登錄 v_floor／ds_err）、理想追蹤 ⇒ mode 3 終點配對預測距離 ≥ 6 mm，提交 mode 4；事件含 d／d_pred／need。

    python3 evaluation/test_door_regrasp_policy2.py
"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_k3m_policy_p2 as PP2                       # noqa: E402
import door_regrasp_policy2 as RP2                     # noqa: E402
import test_door_regrasp_policy1 as T1                 # noqa: E402


def mk2(with_geom):
    p = T1.mk()
    p.__class__ = RP2.RegraspPolicy2
    if with_geom:
        g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
        fn = GM3.backend_v3()
        p.geom_v3_fn = fn
        p.geom = GM.GeometryMonitor(GM.GeomMonitorConfig(v_floor=dict(g['v_floor_registered']), ds_err=dict(g['downsampling']['ds_err_registered'])), fn)
        p.geom_last, p.geom_t = None, None
    return p


class TQ(unittest.TestCase):
    def setUp(self):
        self._orig = PP2.K3MP2Policy.step
        PP2.K3MP2Policy.step = T1._fake_parent_step

    def tearDown(self):
        PP2.K3MP2Policy.step = self._orig

    def test_q1_no_geom(self):
        p = mk2(False)
        T1.Fake(p).run(60.0)
        self.assertEqual(p.abort, 'rg3_gate_timeout')
        self.assertTrue(any(e.get('why') == 'pair_geometry_missing' for e in p.rg['events']))

    def test_q2_geom(self):
        p = mk2(True)
        w = T1.Fake(p).run(40.0)
        self.assertIn(4, w.t_commit, p.abort)
        ev = [e for e in p.rg['events'] if 'pair_check_before_mode4' in e]
        self.assertTrue(ev and ev[-1]['pair_check_before_mode4'] is True, ev[-1:])
        d = ev[-1]['detail']
        self.assertGreaterEqual(d['d_pred'], d['need'])
        print('mode 3 終點配對：', {k: round(v * 1e3, 2) if k != 'T_pred' else round(v, 3) for k, v in d.items()})


if __name__ == '__main__':
    unittest.main(verbosity=1)
