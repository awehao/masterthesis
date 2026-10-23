#!/usr/bin/env python3
"""door_k3m_geom_monitor 的反例（r2）：與離線 parts_clearance／self_clear 同值（四類）、手指只排除橫桿、自碰快取＝全量，
輸入無效（含 Codex 反例：錯誤長度 qf、文字輸入）／過期／時間不遞增／計算失敗／結果結構錯／結果年齡過期 ⇒ stop，
耗時一律內部實測（不補零）、以實測接近率與 v_floor 預測停止前距離、warn 只記錄、ds_err 計入。

    python3 evaluation/test_door_k3m_geom_monitor.py
"""
import json
import math
import os
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_geom_monitor as GM                     # noqa: E402

CFG = GM.GeomMonitorConfig()
NOM = {'nonfinger': 0.0072, 'finger': 0.035, 'base': 0.2, 'self': 0.0098}


def fake(dist, delay=0.0):
    st = {'t': 0.0}

    def fn(qf, q_door, q_finger):
        if delay:
            time.sleep(delay)
        return {k: ((v(st['t']) if callable(v) else v), 'x') for k, v in dist.items()}
    return fn, st


class Decision(unittest.TestCase):
    def mon(self, dist, delay=0.0, **kw):
        fn, st = fake(dist, delay)
        return GM.GeometryMonitor(GM.GeomMonitorConfig(**kw), fn), st

    def test_ok_and_internal_timing(self):
        M, st = self.mon(NOM, delay=0.004)
        d, det = M.step(1.0, np.zeros(9), 0.01, 0.00166, 0.03)
        self.assertEqual(d, 'ok')
        self.assertGreaterEqual(det['compute_s'], 0.004)                  # 實測耗時，不補零
        self.assertAlmostEqual(det['T_pred_s'], 0.05 + 0.03 + det['compute_s'] + 0.02)

    def test_closing_rate_triggers_stop_before_threshold(self):
        M, st = self.mon(dict(NOM, nonfinger=lambda t: 0.012 - 0.05 * (t - 1.0)))
        st['t'] = 1.0
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[0], 'ok')
        st['t'] = 1.05
        d, det = M.step(1.05, np.zeros(9), 0.0, 0.0, 0.02)
        self.assertEqual((d, det['why']), ('stop', 'geometry_clearance_nonfinger'))
        self.assertGreater(det['classes']['nonfinger']['d'], 0.005)

    def test_self_class_stops(self):
        M, st = self.mon(dict(NOM, self=0.0049))
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_clearance_self')

    def test_warn_only(self):
        M, st = self.mon(dict(NOM, nonfinger=0.0061))
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[0], 'warn')

    def test_ds_err_counted(self):
        M, st = self.mon(NOM, ds_err={'nonfinger': 0.0025, 'finger': 0.0, 'base': 0.0, 'self': 0.0})
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[0], 'stop')

    def test_codex_bad_inputs(self):
        M, st = self.mon(NOM)
        for qf in (np.zeros(8), np.zeros(10), [0.0] * 8 + ['x'], 'abcdefghi', None, np.full(9, np.nan), [True] * 9):
            self.assertEqual(M.step(1.0, qf, 0.0, 0.0, 0.02)[1]['why'], 'geometry_input_invalid', repr(qf))
        for args in (('1.0', 0.0, 0.0, 0.02), (1.0, 'x', 0.0, 0.02), (1.0, 0.0, True, 0.02), (1.0, 0.0, 0.0, None), (float('inf'), 0.0, 0.0, 0.02)):
            self.assertEqual(M.step(args[0], np.zeros(9), args[1], args[2], args[3])[1]['why'], 'geometry_input_invalid', repr(args))

    def test_stale_and_time_order(self):
        M, st = self.mon(NOM)
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.25)[1]['why'], 'geometry_state_stale')
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, -0.01)[1]['why'], 'geometry_state_stale')
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[0], 'ok')
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_time_not_increasing')
        self.assertEqual(M.step(0.9, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_time_not_increasing')

    def test_result_stale_after_compute(self):
        M, st = self.mon(NOM, delay=0.06)
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.15)[1]['why'], 'geometry_result_stale')

    def test_result_structure(self):
        for bad in ({k: v for k, v in NOM.items() if k != 'self'}, dict(NOM, base=float('nan')), dict(NOM, base=True)):
            M, st = self.mon(bad)
            self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_result_invalid')
        M = GM.GeometryMonitor(CFG, lambda q, a, b: [1, 2, 3])
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_result_invalid')
        M = GM.GeometryMonitor(CFG, lambda q, a, b: {k: (0.1,) for k in GM.CLASSES})
        self.assertEqual(M.step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_result_invalid')

    def test_compute_failure_and_missing_backend(self):
        def boom(*a):
            raise RuntimeError('x')
        self.assertEqual(GM.GeometryMonitor(CFG, boom).step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_compute_failed')
        self.assertEqual(GM.GeometryMonitor(CFG, None).step(1.0, np.zeros(9), 0.0, 0.0, 0.02)[1]['why'], 'geometry_backend_missing')


class Backend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rp = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay_r2.json')))
        cls.rows = next(c for c in rp['configs'] if c['name'] == 'free')['rows']

    def test_equals_offline_parts_clearance_and_self_clear(self):
        import door_k3au_offline_geom as GA
        fn = GM.default_backend(self_cache=False)
        for r in self.rows[::60]:
            q, qd = np.array(r['q']), math.radians(r['q_door_deg'])
            d = fn(q, qd, 0.00166)
            parts, bc = GA.parts_clearance(q, qd, 0.00166)
            nf = min(min(v['min_to_door_parts_m'], v['min_to_bar_m']) for n, v in parts.items() if n not in GA.FINGERS)
            fg = min(parts[f]['min_to_door_parts_m'] for f in GA.FINGERS)
            sc, _ = GA.C.self_clear(q, 0.00166)
            self.assertAlmostEqual(d['nonfinger'][0], nf, places=12)
            self.assertAlmostEqual(d['finger'][0], fg, places=12)
            self.assertAlmostEqual(d['base'][0], bc, places=12)
            self.assertAlmostEqual(d['self'][0], sc, places=12)

    def test_self_cache_equals_full_along_candidate(self):
        fc, ff = GM.default_backend(), GM.default_backend(self_cache=False)
        for r in self.rows[::3]:
            q, qd = np.array(r['q']), math.radians(r['q_door_deg'])
            self.assertEqual(fc(q, qd, 0.00166)['self'][0], ff(q, qd, 0.00166)['self'][0])

    def test_fingers_exclude_only_bar(self):
        import door_k2_offline_v2 as V2
        import door_geom_v2 as G2
        K1 = V2.K1
        sh = G2.shapes(K1.D2, 0.0, K1.H)
        names = [s[0] for s in sh]
        self.assertTrue({'leaf', 'post0', 'post1', 'wall_hinge_side', 'wall_latch_side'} <= set(names))
        W = {'uflite_finger1': np.array([sh[names.index('leaf')][1]])}
        shapes = [(n, (lambda P, c=c, R=R, hs=hs: G2.box_dist(P, c, R, hs))) for n, c, R, hs in sh]
        out = GM.clearance(None, 0.0, 0.0, lambda qf, links, qq: W, shapes, lambda P: np.array([1.0]), lambda qf: (1.0, 'x'), ['uflite_finger1'])
        self.assertLess(out['finger'][0], 0.0)
        self.assertIn('leaf', out['finger'][1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
