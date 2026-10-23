#!/usr/bin/env python3
"""door_root_evidence 反例：立即複製（之後改來源不影響已存值）、例外與非有限記原因不拋、bool 不當數值、
缺欄清單、快照三分類（None／例外／非有限 ⇒ unknown，不猜預設值）、模組來源與 sha256、寫入沿用 EvidenceWriter（allow_nan=False 不會被 NaN 擊穿）。

    python3 evaluation/test_door_root_evidence.py
"""
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_root_evidence as RE                         # noqa: E402
import door_k3_evidence as KEV                          # noqa: E402


class Grab(unittest.TestCase):
    def test_copy_not_view(self):
        buf = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        v, why = RE.grab(lambda: buf)
        buf[0, 0] = 99.0                                   # 來源之後被更新
        self.assertEqual(v, [1.0, 2.0, 3.0])
        self.assertIsNone(why)

    def test_errors_and_nonfinite(self):
        v, why = RE.grab(lambda: 1 / 0)
        self.assertIsNone(v)
        self.assertIn('ZeroDivisionError', why)
        v, why = RE.grab(lambda: [1.0, float('nan')])
        self.assertEqual((v, why), (None, 'nonfinite'))
        v, why = RE.grab(lambda: [True, 1.0])
        self.assertIsNone(v)
        v, why = RE.grab(lambda: 'abc')
        self.assertIsNone(v)

    def test_tuple_pose(self):
        p, o = np.array([0.6, -0.5, 0.0]), np.array([1.0, 0, 0, 0])
        v, _ = RE.grab(lambda: (p, o))
        self.assertEqual(len(v), 7)


class RowT(unittest.TestCase):
    def test_row_missing_and_errors(self):
        r = RE.Row(10, 0.1)
        r.put('A_root_pre', lambda: [1, 2, 3])
        r.put('B_root_post_write', lambda: (_ for _ in ()).throw(RuntimeError('x')))
        r.put('handle_leaf', lambda: True, kind='bool')
        r.set('api_args', [0.0, float('inf'), 0.0])
        d = r.finish(RE.POINTS)
        self.assertIn('B_root_post_write', d['errors'])
        self.assertEqual(d['errors']['api_args'], 'nonfinite')
        self.assertIn('C_leaf_pre_step', d['missing'])
        self.assertIn('api_args', d['missing'])
        self.assertTrue(d['handle_leaf'])
        json.dumps(d, allow_nan=False)                     # 可嚴格序列化

    def test_clean_row_has_no_errors_key(self):
        r = RE.Row(1, 0.01)
        for k in RE.POINTS:
            r.put(k, lambda: [0.0])
        d = r.finish(RE.POINTS)
        self.assertNotIn('errors', d)
        self.assertEqual(d['missing'], [])


class Snapshot(unittest.TestCase):
    def test_three_categories(self):
        s = RE.physics_snapshot({'solverType': lambda: 'TGS', 'iters': lambda: None, 'boom': lambda: 1 / 0},
                                {'dt': lambda: 0.01, 'nan': lambda: float('nan'), 'arr': lambda: np.array([1, 2])})
        self.assertEqual(s['usd_declared'], {'solverType': 'TGS'})
        self.assertEqual(s['runtime_readback'], {'dt': 0.01, 'arr': [1, 2]})
        self.assertEqual(set(s['unknown']), {'usd_declared:iters', 'usd_declared:boom', 'runtime_readback:nan'})
        json.dumps(s, allow_nan=False)


class Modules(unittest.TestCase):
    def test_module_info(self):
        m = RE.module_info({'grab': RE.grab, 'np_array': np.array})
        self.assertEqual(len(m['objects']['grab']['sha256']), 64)
        self.assertIn('error', m['objects']['np_array'])          # 內建函式沒有原始檔 ⇒ 記原因，不拋
        self.assertIn('isaacsim', m['packages'])


class WriterNaN(unittest.TestCase):
    def test_writer_strict(self):
        d = tempfile.mkdtemp()
        w = KEV.EvidenceWriter(os.path.join(d, 'x.jsonl'))
        r = RE.Row(1, 0.01)
        r.put('A_root_pre', lambda: [float('nan')])
        self.assertTrue(w.write(r.finish(RE.POINTS)))             # NaN 已轉 None ⇒ 寫得進
        self.assertFalse(w.write({'x': math.nan}))                # 未經 Row 的 NaN ⇒ 寫入失敗被記錄、不拋
        st = w.close()
        self.assertEqual(st['n_err'].get('write'), 1)
        self.assertFalse(st['complete'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
