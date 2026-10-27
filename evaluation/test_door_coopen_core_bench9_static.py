#!/usr/bin/env python3
"""跑法 bench9 的版本化差異靜態測試（Codex r31：聯合配置驗證 door_coopen_core_bench_6）。
  N1 bench9 對 bench8：被刪改的非註解行只限 FRZ、PRM（r10→r11）、開場字樣、執行端行（ev9r4→ev9r5，其餘逐字相同）。
  N2 跑法讀登錄 r11 的 executor_change 得到暫停上限 5.0、原值 5.0（同一個 python3 -c 讀法）；登錄旗標檢查（offset_update_in_open）仍成立。
  N3 ev9r5 可編譯。

    python3 evaluation/test_door_coopen_core_bench9_static.py
"""
import difflib
import json
import os
import py_compile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class N(unittest.TestCase):
    def test_n1_runner_diff(self):
        A = open(os.path.join(HERE, 'run_door_coopen_core_bench8.sh')).read().splitlines()
        B = open(os.path.join(HERE, 'run_door_coopen_core_bench9.sh')).read().splitlines()
        d = [ln for ln in difflib.unified_diff(A, B, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        rm = [ln[1:] for ln in d if ln.startswith('-') and not ln[1:].lstrip().startswith('#')]
        add = [ln[1:] for ln in d if ln.startswith('+') and not ln[1:].lstrip().startswith('#')]
        self.assertEqual((len(rm), len(add)), (4, 4), (rm, add))
        self.assertTrue(add[0].startswith('FRZ=evaluation/results/door/freeze_door_coopen_core_bench_dev_r9.sha256'))
        self.assertTrue(add[1].startswith('PRM=evaluation/results/door/coopen_core_params_48_r11.json'))
        self.assertTrue(rm[2].startswith('say "=== 核心段 bench8') and add[2].startswith('say "=== 核心段 bench9'))
        self.assertEqual(rm[3].replace('isaac_door_k2_sim_ev9r4.py', 'isaac_door_k2_sim_ev9r5.py'), add[3])

    def test_n2_registry_values(self):
        p = json.load(open(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r11.json')))
        self.assertEqual((p['executor_change']['pause_finger_max_effort'], p['executor_change']['original_effort']), (5.0, 5.0))
        self.assertIs(p['solver_change'].get('offset_update_in_open'), True)
        src = open(os.path.join(HERE, 'run_door_coopen_core_bench9.sh')).read()
        self.assertIn("['executor_change']['pause_finger_max_effort']", src)
        self.assertIn("--pause-finger-max-effort \"$W1E_FEFF\"", src)

    def test_n3_compile(self):
        py_compile.compile(os.path.join(HERE, 'isaac_door_k2_sim_ev9r5.py'), doraise=True, cfile='/tmp/claude-1000/ev9r5_compile_check.pyc')


if __name__ == '__main__':
    unittest.main(verbosity=2)
