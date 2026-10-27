#!/usr/bin/env python3
"""exec_timing_diag 與 EV9r3 插入的測試（Codex r29：診斷只觀測、故障不影響控制）。
  X1 區段：假時鐘逐標記遞增 ⇒ 各區段＝相鄰標記差，最後一段＝下一輪起點 − 本輪 after_world_step；summary 最慢列與分位數正確。
  X2 有界：cap 3、begin 5 次 ⇒ 只記 3 列、n_overflow 2、陣列不擴充；溢位後 mark 不寫入。
  X3 GC：真 gc.collect()（由測試觸發）⇒ 記到成對 start／stop、generation 2、列號＝當下列；stop() 後回呼卸除、gc.callbacks 還原。
  X4 故障吞掉：時鐘丟例外 ⇒ begin／mark／GC 回呼都不拋、n_err 增加；summary 仍回傳。
  X5 不改 GC 行為：模組原始碼不含 gc.collect／disable／enable／freeze／set_threshold；start_gc／stop 前後 gc.get_threshold、gc.isenabled 不變。
  X6 存檔：save 寫出 npz 可讀回（T、meta、G、marks、sections）；寫入失敗（目錄不存在）⇒ 回 False、n_err＋1、不拋。
  X7 靜態：ev9r3 對 ev9r2 沒有任何刪改行；新增非註解行全部含 _xt 或 XTD（只做插入）；8 個標記名各出現一次、begin 一次、stop 在封存段。

    python3 evaluation/test_exec_timing_diag.py
"""
import difflib
import gc
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exec_timing_diag as X                            # noqa: E402


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 0.001
        return self.t


class T(unittest.TestCase):
    def test_x1_sections(self):
        c = Clock()
        st = X.StepTimer(cap=10, clock=c)
        for k in range(3):
            st.begin(k + 1, 0.01 * (k + 1))
            for m in X.MARKS[1:]:
                if k == 1 and m == 'after_world_step':
                    c.t += 0.25                                   # 第 2 輪 world.step 慢 0.25 s
                st.mark(m)
        S = st.sections()
        self.assertEqual(S.shape, (3, len(X.SECTIONS)))
        np.testing.assert_allclose(S[0, :-1], 0.001, atol=1e-12)
        np.testing.assert_allclose(S[0, -1], 0.001, atol=1e-12)
        self.assertAlmostEqual(S[1, X.SECTIONS.index('world_step')], 0.251, places=9)
        self.assertTrue(np.isnan(S[2, -1]))
        sm = st.summary()
        self.assertEqual(sm['slowest_rows'][0]['step'], 2.0)
        self.assertAlmostEqual(sm['section_quantiles_s']['world_step']['max'], 0.251, places=9)
        self.assertEqual(sm['section_quantiles_s']['world_step']['argmax_step'], 2.0)

    def test_x2_bounded(self):
        st = X.StepTimer(cap=3, clock=Clock())
        for k in range(5):
            st.begin(k, k * 0.01)
            st.mark('after_callbacks')
        self.assertEqual((st.row, st.n_overflow, st.T.shape[0]), (2, 2, 3))
        self.assertEqual(st.meta[2, 0], 2.0)

    def test_x3_gc(self):
        before = list(gc.callbacks)
        st = X.StepTimer(cap=5)
        st.start_gc()
        st.begin(7, 0.07)
        gc.collect()
        st.stop()
        ev = st.gc_events()
        self.assertTrue(any(e[2] == 2 and e[4] == 0 for e in ev), ev)
        self.assertTrue(all(e[1] >= 0 for e in ev))
        self.assertEqual(gc.callbacks, before)
        n = st.n_gc
        gc.collect()
        self.assertEqual(st.n_gc, n)

    def test_x4_errors_swallowed(self):
        def bad():
            raise RuntimeError('clock')
        st = X.StepTimer(cap=5, clock=bad)
        st.start_gc()
        st.begin(1, 0.01)
        st.mark('after_callbacks')
        gc.collect()
        st.stop()
        self.assertGreaterEqual(st.n_err, 3)
        self.assertIn('n_err', st.summary())

    def test_x5_gc_behavior_untouched(self):
        src = open(os.path.join(HERE, 'exec_timing_diag.py')).read()
        code = '\n'.join(ln for ln in src.split('"""', 2)[2].splitlines() if not ln.lstrip().startswith('#'))   # 去掉模組說明與註解
        for f in ('gc.collect', 'gc.disable', 'gc.enable', 'gc.freeze', 'gc.set_threshold'):
            self.assertNotIn(f, code, f)
        th, en = gc.get_threshold(), gc.isenabled()
        st = X.StepTimer(cap=2)
        st.start_gc()
        st.stop()
        self.assertEqual((gc.get_threshold(), gc.isenabled()), (th, en))

    def test_x6_save(self):
        st = X.StepTimer(cap=4, clock=Clock())
        st.begin(1, 0.01)
        for m in X.MARKS[1:]:
            st.mark(m)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'x.npz')
            self.assertTrue(st.save(p))
            z = np.load(p)
            self.assertEqual(z['T'].shape, (1, len(X.MARKS)))
            self.assertEqual(list(z['sections']), list(X.SECTIONS))
        n = st.n_err
        self.assertFalse(st.save('/nonexistent_dir_xt/x.npz'))
        self.assertEqual(st.n_err, n + 1)

    def test_x7_static_insert_only(self):
        A = open(os.path.join(HERE, 'isaac_door_k2_sim_ev9r2.py')).read().splitlines()
        B = open(os.path.join(HERE, 'isaac_door_k2_sim_ev9r3.py')).read().splitlines()
        d = [ln for ln in difflib.unified_diff(A, B, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        self.assertEqual([ln for ln in d if ln.startswith('-')], [])
        add = [ln[1:] for ln in d if ln.startswith('+') and not ln[1:].lstrip().startswith('#')]
        self.assertTrue(all(('_xt' in ln or 'XTD' in ln) for ln in add), add)
        src = '\n'.join(B)
        for m in X.MARKS[1:]:
            self.assertEqual(src.count(f"_xt.mark('{m}')"), 1, m)
        self.assertEqual(src.count('_xt.begin('), 1)
        self.assertLess(src.index('# ---- 封存 ----'), src.index('_xt.stop()'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
