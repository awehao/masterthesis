#!/usr/bin/env python3
"""跑法 bench10 的版本化差異靜態測試（Codex r36：C2 接線；test_door_coopen_core_bench9_static.py 的版本化複本）。
  N1 bench10 對 bench9：被刪改的非註解行只限 FRZ、PRM（r11→r12）、開場字樣、任務節點行（v8→v9，其餘逐字相同）、結束字樣；新增一行登錄旗標拒跑檢查。
  N2 登錄 r12：closure_guidance.enabled True、暫停上限仍 5.0、offset_update_in_open 仍 True；跑法含 closure_guidance 拒跑檢查。
  N3 任務節點 v9 可編譯。

    python3 evaluation/test_door_coopen_core_bench10_static.py
"""
import difflib
import json
import os
import py_compile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


class N(unittest.TestCase):
    def test_n1_runner_diff(self):
        A = open(os.path.join(HERE, 'run_door_coopen_core_bench9.sh')).read().splitlines()
        B = open(os.path.join(HERE, 'run_door_coopen_core_bench10.sh')).read().splitlines()
        d = [ln for ln in difflib.unified_diff(A, B, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
        rm = [ln[1:] for ln in d if ln.startswith('-') and not ln[1:].lstrip().startswith('#')]
        add = [ln[1:] for ln in d if ln.startswith('+') and not ln[1:].lstrip().startswith('#')]
        self.assertEqual((len(rm), len(add)), (5, 6), (rm, add))
        self.assertTrue(add[0].startswith('FRZ=evaluation/results/door/freeze_door_coopen_core_bench_dev_r10.sha256'))
        self.assertTrue(add[1].startswith('PRM=evaluation/results/door/coopen_core_params_48_r12.json'))
        self.assertTrue(add[2].startswith('[ "$(python3') and 'closure_guidance' in add[2] and add[2].rstrip().endswith('exit 97; }'))
        self.assertTrue(rm[2].startswith('say "=== 核心段 bench9') and add[3].startswith('say "=== 核心段 bench10'))
        self.assertEqual(rm[3].replace('door_coopen_core_task_node8.py', 'door_coopen_core_task_node9.py'), add[4])
        self.assertTrue(rm[4].startswith('say "=== 結束 rc=') and 'door_coopen_check_core13.py' in add[5])

    def test_n2_registry_values(self):
        p = json.load(open(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r12.json')))
        self.assertIs(p['control_cost_change']['closure_guidance']['enabled'], True)
        self.assertEqual(p['executor_change']['pause_finger_max_effort'], 5.0)
        self.assertIs(p['solver_change'].get('offset_update_in_open'), True)
        p11 = json.load(open(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r11.json')))
        self.assertEqual(set(p) - set(p11), {'derived_from_r11'})
        for k in p11:
            if k not in ('control_cost_change', 'schema'):
                self.assertEqual(p11[k], p[k], k)
        self.assertEqual({k: v for k, v in p['control_cost_change'].items() if k != 'closure_guidance'}, p11['control_cost_change'])

    def test_n3_compile(self):
        py_compile.compile(os.path.join(HERE, 'door_coopen_core_task_node9.py'), doraise=True, cfile='/tmp/claude-1000/task_node9_compile_check.pyc')


if __name__ == '__main__':
    unittest.main(verbosity=2)
