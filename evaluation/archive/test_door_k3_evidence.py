#!/usr/bin/env python3
"""door_k3_evidence 的離線反例（合成接觸串流；不啟動 Isaac）。

    python3 evaluation/test_door_k3_evidence.py
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3_evidence as EV                         # noqa: E402

NAMES = ['link5', 'link6', 'uflite_finger1', 'uflite_finger2', 'link_base']


def stream():
    # pair 順序同 NAMES：link5 0 點、link6 1 點、finger1 2 點、finger2 1 點、link_base 1 點（不在 keep）
    cnt = np.array([[0, 1, 2, 1, 1]])
    start = np.array([[0, 0, 1, 3, 4]])
    pts = np.arange(15, dtype=float).reshape(5, 3)
    nrm = np.tile([0.0, 0.0, 1.0], (5, 1))
    fn = np.array([[0.5], [1.0], [2.0], [3.0], [9.0]])
    sep = np.array([-1e-4, 0.0, -2e-4, 1e-4, 0.0])
    return fn, pts, nrm, sep, cnt, start


class Evidence(unittest.TestCase):
    def test_extract_contacts_slices_and_keep(self):
        c = EV.extract_contacts(*stream(), NAMES)
        self.assertEqual(set(c), {'link6', 'uflite_finger1', 'uflite_finger2'})       # link_base 不在 keep、link5 無接觸
        self.assertEqual(len(c['uflite_finger1']), 2)
        self.assertEqual(c['uflite_finger1'][0]['fn'], 1.0)
        self.assertEqual(c['uflite_finger1'][1]['p'], [6.0, 7.0, 8.0])
        self.assertEqual(c['uflite_finger2'][0]['fn'], 3.0)
        self.assertEqual(c['link6'][0]['fn'], 0.5)

    def test_pair_count_mismatch_raises_and_safe_wraps(self):
        fn, pts, nrm, sep, cnt, start = stream()
        with self.assertRaises(ValueError):
            EV.extract_contacts(fn, pts, nrm, sep, cnt, start, NAMES[:-1])
        v, why = EV.safe(EV.extract_contacts, fn, pts, nrm, sep, cnt, start, NAMES[:-1])
        self.assertIsNone(v)
        self.assertIn('ValueError', why)

    def test_nonfinite_values_recorded_as_none(self):
        fn, pts, nrm, sep, cnt, start = stream()
        fn = fn.copy()
        fn[1, 0] = float('nan')
        c = EV.extract_contacts(fn, pts, nrm, sep, cnt, start, NAMES)
        self.assertIsNone(c['uflite_finger1'][0]['fn'])                                # 不寫成 0

    def test_friction(self):
        ff = np.array([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
        fp = np.ones((3, 3))
        cnt = np.array([[0, 0, 2, 1, 0]])
        start = np.array([[0, 0, 0, 2, 3]])
        f = EV.extract_friction(ff, fp, cnt, start, NAMES)
        self.assertEqual(f['uflite_finger1'][1]['ft'], [0.0, 0.2, 0.0])
        self.assertEqual(f['uflite_finger2'][0]['ft'], [0.0, 0.0, 0.3])

    def test_wrench_and_efforts(self):
        F = np.arange(60, dtype=float).reshape(10, 6)
        self.assertEqual(EV.wrench_row(F, 7), [42.0, 43.0, 44.0, 45.0, 46.0, 47.0])
        self.assertEqual(EV.wrench_row(F[None], 7), [42.0, 43.0, 44.0, 45.0, 46.0, 47.0])     # (1, J, 6)
        with self.assertRaises(ValueError):
            EV.wrench_row(np.zeros((3, 3)), 1)
        names = ['base', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'finger_joint1', 'finger_joint2']
        e = EV.efforts_by_name(np.arange(9.0), names, EV.ARM_DOF + EV.FINGER_DOF)
        self.assertEqual(e['joint1'], 1.0)
        self.assertEqual(e['finger_joint2'], 8.0)
        with self.assertRaises(ValueError):
            EV.efforts_by_name(np.arange(8.0), names, EV.ARM_DOF)
        with self.assertRaises(ValueError):
            EV.efforts_by_name(np.arange(9.0), names, ('joint7',))

    def test_summarize_in_handle_frame(self):
        T = np.eye(4)
        T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]                                 # 把手 x＝世界 y
        T[:3, 3] = [1.0, 2.0, 3.0]
        con = {'uflite_finger1': [{'p': [1.0, 2.5, 3.0], 'n': [0.0, 1.0, 0.0], 'fn': 2.0, 'sep': 0.0},
                                  {'p': [1.0, 2.5, 3.2], 'n': [0.0, 1.0, 0.0], 'fn': 2.0, 'sep': 0.0}]}
        s = EV.summarize_contacts(con, T)['uflite_finger1']
        np.testing.assert_allclose(s['p_handle_absw'], [0.5, 0.0, 0.1], atol=1e-12)
        np.testing.assert_allclose(s['n_handle_absw'], [1.0, 0.0, 0.0], atol=1e-12)
        self.assertEqual(s['fn_abs_sum'], 4.0)
        self.assertEqual(s['weight'], 'abs(fn)')

    # ---- Codex 20261022_115319 必修 2：pair 索引
    def test_pair_index_validation(self):
        fn, pts, nrm, sep, cnt, start = stream()
        bad = []
        c = cnt.astype(float).copy(); s_ = start.astype(float).copy(); s_[0, 2] = -1.0; bad.append(('start=-1', c, s_))
        c = cnt.astype(float).copy(); c[0, 2] = -1.0; bad.append(('count=-1', c, start))
        c = cnt.astype(float).copy(); c[0, 2] = 1.5; bad.append(('count=1.5', c, start))
        c = cnt.astype(float).copy(); c[0, 3] = 5.0; bad.append(('slice beyond buffer', c, start))
        c = cnt.astype(float).copy(); c[0, 1] = float('nan'); bad.append(('count=nan', c, start))
        for name, c, s_ in bad:
            v, why = EV.safe(EV.extract_contacts, fn, pts, nrm, sep, c, s_, NAMES)
            self.assertIsNone(v, name)                                                 # 不當成零接觸
            self.assertIn('ValueError', why, name)
        # 不在 keep 的 pair 越界也要報錯（索引整體不可信）
        c = cnt.astype(float).copy(); c[0, 4] = 9.0
        v, why = EV.safe(EV.extract_contacts, fn, pts, nrm, sep, c, start, NAMES)
        self.assertIsNone(v)
        ff, fp = np.zeros((2, 3)), np.zeros((2, 3))
        v, why = EV.safe(EV.extract_friction, ff, fp, np.array([[0, 0, 3, 0, 0]]), np.array([[0, 0, 0, 0, 0]]), NAMES)
        self.assertIsNone(v)

    # ---- 必修 3：幅值權重
    def test_abs_weighting_and_undefined(self):
        T = np.eye(4)
        con = {'uflite_finger1': [{'p': [0.0, 0.0, 0.0], 'n': [0, 0, 1.0], 'fn': -1.0, 'sep': 0.0},
                                  {'p': [1.0, 0.0, 0.0], 'n': [0, 0, 1.0], 'fn': -3.0, 'sep': 0.0}]}
        s = EV.summarize_contacts(con, T)['uflite_finger1']
        self.assertAlmostEqual(s['p_handle_absw'][0], 0.75)                            # Codex 反例：不是 0.5
        self.assertEqual(s['fn_signed_sum'], -4.0)
        self.assertEqual(s['fn_abs_sum'], 4.0)
        self.assertAlmostEqual(s['p_handle_unweighted'][0], 0.5)
        con0 = {'uflite_finger2': [{'p': [0.0, 0.0, 0.0], 'n': [0, 0, 1.0], 'fn': 0.0, 'sep': 0.0}]}
        s0 = EV.summarize_contacts(con0, T)['uflite_finger2']
        self.assertIsNone(s0['p_handle_absw'])                                          # 全零 ⇒ 未定
        self.assertIsNone(s0['n_handle_absw'])

    # ---- 必修 1：開檔／寫出／flush／關檔故障隔離
    def test_writer_open_failure(self):
        import tempfile
        d = tempfile.mkdtemp()
        w = EV.EvidenceWriter(d)                                                         # 目錄 ⇒ 開檔失敗
        self.assertFalse(w.active)
        self.assertFalse(w.write({'a': 1}))
        st = w.close()
        self.assertFalse(st['complete'])
        self.assertIn('open', st['n_err'])
        self.assertTrue(st['enabled'])

    def test_writer_write_flush_close_failures(self):
        class F:
            def __init__(self, fail):
                self.fail = fail
                self.buf = []

            def write(self, x):
                if 'write' in self.fail:
                    raise OSError('disk full')
                self.buf.append(x)

            def flush(self):
                if 'flush' in self.fail:
                    raise OSError('flush')

            def close(self):
                if 'close' in self.fail:
                    raise OSError('close')
        for fail in ({'write'}, {'flush'}, {'close'}):
            w = EV.EvidenceWriter('x.jsonl', flush_every=1, opener=lambda p, m, f=fail: F(f))
            w.write({'a': 1})
            w.write({'b': float('nan')})                                                 # allow_nan=False ⇒ 記 write 錯誤，不拋
            st = w.close()
            self.assertFalse(st['complete'], fail)
            self.assertTrue(st['n_err'], fail)
        ok = EV.EvidenceWriter('x.jsonl', opener=lambda p, m: F(set()))
        ok.write({'a': 1})
        st = ok.close()
        self.assertTrue(st['complete'])
        self.assertEqual(st['n'], 1)

    def test_writer_disabled_and_errors_count_incomplete(self):
        w = EV.EvidenceWriter('')
        self.assertFalse(w.active)
        self.assertFalse(w.close()['complete'])
        class F:
            def write(self, x): pass
            def flush(self): pass
            def close(self): pass
        w = EV.EvidenceWriter('x', opener=lambda p, m: F())
        w.write({'a': 1})
        w.count_err('friction_err')                                                      # 欄位讀取錯誤 ⇒ 不完整
        self.assertFalse(w.close()['complete'])

    def test_torch_like_input(self):
        class T:
            def __init__(self, a):
                self.a = a

            def cpu(self):
                return self

            def numpy(self):
                return self.a
        fn, pts, nrm, sep, cnt, start = stream()
        c = EV.extract_contacts(T(fn), T(pts), T(nrm), T(sep), cnt, start, NAMES)
        self.assertIn('uflite_finger1', c)


if __name__ == '__main__':
    unittest.main(verbosity=2)
