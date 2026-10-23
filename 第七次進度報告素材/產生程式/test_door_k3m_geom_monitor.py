#!/usr/bin/env python3
"""door_k3m_geom_monitor 的反例：與離線 parts_clearance 同值（三類距離）、手指只排除橫桿、輸入無效／過期／計算失敗／結果非有限 ⇒ stop，
以實測接近率與 v_floor 預測停止前距離、warn 只記錄、降採樣誤差界計入。

    python3 evaluation/test_door_k3m_geom_monitor.py
"""
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_geom_monitor as GM                     # noqa: E402

CFG = GM.GeomMonitorConfig()


def fake(dist):
    """dist: dict class -> d 或 callable(t)。"""
    st = {'t': 0.0}

    def fn(qf, q_door, q_finger):
        return {k: ((v(st['t']) if callable(v) else v), 'x') for k, v in dist.items()}
    return fn, st


class Decision(unittest.TestCase):
    def mon(self, dist, **kw):
        fn, st = fake(dist)
        c = GM.GeomMonitorConfig(**kw)
        return GM.GeometryMonitor(c, fn), st

    def test_ok_and_floor(self):
        M, st = self.mon({'nonfinger': 0.0072, 'finger': 0.035, 'base': 0.2})
        d, det = M.step(1.0, np.zeros(9), 0.01, 0.00166, 0.03, 0.006)
        self.assertEqual(d, 'ok')
        T = 0.05 + 0.03 + 0.006 + 0.02
        self.assertAlmostEqual(det['classes']['nonfinger']['d_pred'], 0.0072 - 0.002 * T)

    def test_closing_rate_triggers_stop_before_threshold(self):
        M, st = self.mon({'nonfinger': lambda t: 0.012 - 0.05 * (t - 1.0), 'finger': 0.035, 'base': 0.2})
        st['t'] = 1.0
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02, 0.005)[0], 'ok')
        st['t'] = 1.05                                                    # d＝9.5 mm（仍 > 5 mm），但以 50 mm/s 接近
        d, det = M.step(1.05, np.zeros(9), 0.0, 0.0, 0.02, 0.005)
        self.assertEqual(d, 'stop')
        self.assertEqual(det['why'], 'geometry_clearance_nonfinger')
        self.assertGreater(det['classes']['nonfinger']['d'], 0.005)

    def test_warn_only(self):
        M, st = self.mon({'nonfinger': 0.0061, 'finger': 0.035, 'base': 0.2})
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02, 0.005)[0], 'warn')

    def test_ds_err_counted(self):
        M, st = self.mon({'nonfinger': 0.0072, 'finger': 0.035, 'base': 0.2}, ds_err={'nonfinger': 0.0025, 'finger': 0.0, 'base': 0.0})
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02, 0.005)[0], 'stop')

    def test_invalid_paths_stop(self):
        M, st = self.mon({'nonfinger': 0.01, 'finger': 0.03, 'base': 0.2})
        self.assertEqual(M.step(1.0, None, 0.0, 0.0, 0.02)[1]['why'], 'geometry_input_invalid')
        self.assertEqual(M.step(1.0, np.full(9, np.nan), 0.0, 0.0, 0.02)[1]['why'], 'geometry_input_invalid')
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.25)[1]['why'], 'geometry_state_stale')
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, -0.01)[1]['why'], 'geometry_state_stale')
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02, float('nan'))[1]['why'], 'geometry_compute_time_invalid')
        M2, _ = self.mon({'nonfinger': float('nan'), 'finger': 0.03, 'base': 0.2})
        self.assertEqual(M2.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_result_invalid')
        M3, _ = self.mon({'finger': 0.03, 'base': 0.2})
        self.assertEqual(M3.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_result_invalid')

        def boom(*a):
            raise RuntimeError('x')
        M4 = GM.GeometryMonitor(CFG, boom)
        self.assertEqual(M4.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_compute_failed')
        self.assertEqual(GM.GeometryMonitor(CFG, None).step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_backend_missing')


class Backend(unittest.TestCase):
    def test_equals_offline_parts_clearance(self):
        import door_k3au_offline_geom as GA
        fn = GM.default_backend()
        rp = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay_r2.json')))
        rows = next(c for c in rp['configs'] if c['name'] == 'free')['rows']
        for r in rows[::60]:
            q, qd = np.array(r['q']), math.radians(r['q_door_deg'])
            d = fn(q, qd, 0.00166)
            parts, bc = GA.parts_clearance(q, qd, 0.00166)
            nf = min(min(v['min_to_door_parts_m'], v['min_to_bar_m']) for n, v in parts.items() if n not in GA.FINGERS)
            fg = min(parts[f]['min_to_door_parts_m'] for f in GA.FINGERS)
            self.assertAlmostEqual(d['nonfinger'][0], nf, places=12)
            self.assertAlmostEqual(d['finger'][0], fg, places=12)
            self.assertAlmostEqual(d['base'][0], bc, places=12)

    def test_fingers_exclude_only_bar(self):
        import door_k2_offline_v2 as V2
        import door_geom_v2 as G2
        K1 = V2.K1
        sh = G2.shapes(K1.D2, 0.0, K1.H)
        names = [s[0] for s in sh]
        self.assertTrue({'leaf', 'post0', 'post1', 'wall_hinge_side', 'wall_latch_side'} <= set(names))
        # 手指點雲平移進門扇 ⇒ finger 類距離 < 0（未被排除）
        W = {'uflite_finger1': np.array([sh[names.index('leaf')][1]])}
        shapes = [(n, (lambda P, c=c, R=R, hs=hs: G2.box_dist(P, c, R, hs))) for n, c, R, hs in sh]
        out = GM.clearance(None, 0.0, 0.0, lambda qf, links, qq: W, shapes, lambda P: np.array([1.0]), lambda qf: (1.0, 'x'), ['uflite_finger1'])
        self.assertLess(out['finger'][0], 0.0)
        self.assertIn('leaf', out['finger'][1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
