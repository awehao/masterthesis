#!/usr/bin/env python3
"""跑法 bench8 的版本化差異靜態測試（Codex r30：bench7＋一次性 gc.freeze 候選；test_door_coopen_core_bench7_static.py 的版本化複本）。
  B1 bench8 對 bench7：被刪改的非註解行只限 FRZ、開場字樣、執行端那一行；執行端那一行除 ev9r3→ev9r4 與加 --gc-freeze-before-release 外逐字相同。
  B2 ev9r4 可編譯（py_compile；不啟動 Isaac）。

    python3 evaluation/test_door_coopen_core_bench8_static.py
"""
import difflib
import os
import py_compile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class B(unittest.TestCase):
    def test_b1_runner_diff(self):
        A = open(os.path.join(HERE, 'run_door_coopen_core_bench7.sh')).read().splitlines()
        Bv = open(os.path.join(HERE, 'run_door_coopen_core_bench8.sh')).read().splitlines()
        d = [ln for ln in difflib.unified_diff(A, Bv, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        rm = [ln[1:] for ln in d if ln.startswith('-') and not ln[1:].lstrip().startswith('#')]
        add = [ln[1:] for ln in d if ln.startswith('+') and not ln[1:].lstrip().startswith('#')]
        self.assertEqual(len(rm), 3, rm)
        self.assertEqual(len(add), 3, add)
        self.assertTrue(rm[0].startswith('FRZ=') and add[0].startswith('FRZ=evaluation/results/door/freeze_door_coopen_core_bench_dev_r8.sha256'))
        self.assertTrue(rm[1].startswith('say "=== 核心段 bench7') and add[1].startswith('say "=== 核心段 bench8'))
        self.assertEqual(rm[2].replace('isaac_door_k2_sim_ev9r3.py --bench-start', 'isaac_door_k2_sim_ev9r4.py --gc-freeze-before-release --bench-start'), add[2])

    def test_b2_compile(self):
        py_compile.compile(os.path.join(HERE, 'isaac_door_k2_sim_ev9r4.py'), doraise=True,
                           cfile=os.path.join('/tmp/claude-1000', 'ev9r4_compile_check.pyc'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
