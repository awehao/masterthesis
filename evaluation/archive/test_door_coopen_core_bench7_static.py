#!/usr/bin/env python3
"""跑法 bench7 的版本化差異靜態測試（Codex r29：沿用 bench_4 控制配置，只加執行端停頓診斷）。
  B1 bench7 對 bench6：被刪改的非註解行只限 FRZ、開場字樣、執行端那一行；執行端那一行除 ev9r2→ev9r3 外逐字相同。
  B2 ev9r3 可編譯（py_compile；不啟動 Isaac）。

    python3 evaluation/test_door_coopen_core_bench7_static.py
"""
import difflib
import os
import py_compile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class B(unittest.TestCase):
    def test_b1_runner_diff(self):
        A = open(os.path.join(HERE, 'run_door_coopen_core_bench6.sh')).read().splitlines()
        Bv = open(os.path.join(HERE, 'run_door_coopen_core_bench7.sh')).read().splitlines()
        d = [ln for ln in difflib.unified_diff(A, Bv, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        rm = [ln[1:] for ln in d if ln.startswith('-') and not ln[1:].lstrip().startswith('#')]
        add = [ln[1:] for ln in d if ln.startswith('+') and not ln[1:].lstrip().startswith('#')]
        self.assertEqual(len(rm), 3, rm)
        self.assertEqual(len(add), 3, add)
        self.assertTrue(rm[0].startswith('FRZ=') and add[0].startswith('FRZ=evaluation/results/door/freeze_door_coopen_core_bench_dev_r7.sha256'))
        self.assertTrue(rm[1].startswith('say "=== 核心段 bench6') and add[1].startswith('say "=== 核心段 bench7'))
        self.assertEqual(rm[2].replace('isaac_door_k2_sim_ev9r2.py', 'isaac_door_k2_sim_ev9r3.py'), add[2])

    def test_b2_compile(self):
        py_compile.compile(os.path.join(HERE, 'isaac_door_k2_sim_ev9r3.py'), doraise=True,
                           cfile=os.path.join('/tmp/claude-1000', 'ev9r3_compile_check.pyc'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
