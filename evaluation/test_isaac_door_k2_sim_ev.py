#!/usr/bin/env python3
"""isaac_door_k2_sim_ev（取證執行端）的結構測試（不啟動 Isaac）。

1. 相對 isaac_door_k2_sim.py **只有插入**（無刪改）⇒ 原有控制、監看、判定逐字保留。
2. --evidence-out 預設空字串。
3. 新 API（get_friction_data、get_measured_joint_efforts、dof_names、door_k3_evidence）只出現在取證區塊：
   _ev_write 內，或由 `_ev_f is not None` 守住；迴圈內呼叫與收尾都在守衛下。
4. 取證寫出與逐步呼叫都包在 try 內（取證失敗不得中斷模擬）。
5. 取證 runner 相對 INT runner 只換執行端並加 --evidence-out。

    python3 evaluation/test_isaac_door_k2_sim_ev.py
"""
import ast
import difflib
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(HERE, 'isaac_door_k2_sim.py')
EV = os.path.join(HERE, 'isaac_door_k2_sim_ev.py')


class Structure(unittest.TestCase):
    def setUp(self):
        self.a = open(ORIG).read().splitlines()
        self.b = open(EV).read().splitlines()
        self.ops = [o for o in difflib.SequenceMatcher(None, self.a, self.b, autojunk=False).get_opcodes() if o[0] != 'equal']
        self.ins = [ln for o in self.ops for ln in self.b[o[3]:o[4]]]

    def test_insert_only(self):
        self.assertTrue(self.ops)
        self.assertTrue(all(o[0] == 'insert' for o in self.ops), [o for o in self.ops if o[0] != 'insert'])

    def test_flag_default_empty(self):
        src = '\n'.join(self.b)
        m = re.search(r"ap\.add_argument\('--evidence-out', default='([^']*)'", src)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), '')

    def test_new_api_only_in_inserted_lines(self):
        for tok in ('get_friction_data', 'get_measured_joint_efforts', 'dof_names', 'KEV.'):
            n_a = sum(ln.count(tok) for ln in self.a)
            n_b = sum(ln.count(tok) for ln in self.b)
            n_ins = sum(ln.count(tok) for ln in self.ins)
            self.assertEqual(n_b - n_a, n_ins, tok)               # 多出來的出現次數全在插入行
        for tok in ('get_friction_data', 'get_measured_joint_efforts', 'KEV.'):
            self.assertEqual(sum(ln.count(tok) for ln in self.a), 0, f'原版已含 {tok}')

    def test_guards(self):
        src = '\n'.join(self.b)
        tree = ast.parse(src)
        guarded_calls = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and 'is not None' in ast.unparse(node.test) and '_ev_f' in ast.unparse(node.test):
                body = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
                if '_ev_write(' in body:
                    guarded_calls += 1
                    self.assertIn('try:', body)
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and ast.unparse(node.test) == '_ev_w is not None':
                if '_ev_w.close()' in ast.unparse(ast.Module(body=node.body, type_ignores=[])):
                    guarded_calls += 1
        self.assertEqual(guarded_calls, 2)                       # 迴圈內呼叫（_ev_f 守衛）＋收尾（_ev_w 守衛）
        calls = [ln for ln in self.b if '_ev_write(' in ln and 'def _ev_write' not in ln]
        self.assertEqual(len(calls), 1)
        self.assertIn("KEV.safe(lambda: list(robot.dof_names)) if _ev_f is not None", src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_ev_write')
        body = ast.unparse(fn)
        for tok in ('get_contact_force_data', 'get_friction_data', 'get_measured_joint_forces', 'get_measured_joint_efforts'):
            self.assertIn(f'KEV.safe(lambda: ', body.split(tok)[0][-120:], tok)       # 每個 API 讀取都經 KEV.safe
        self.assertIn('_ev_w.write(rec_ev)', body)                 # 寫出經 EvidenceWriter（其內部隔離故障）
        self.assertNotIn('open(', body)

    def test_no_undefined_names(self):
        import subprocess, sys
        r = subprocess.run([sys.executable, '-m', 'pyflakes', EV], capture_output=True, text=True)
        und = [ln for ln in (r.stdout + r.stderr).splitlines() if 'undefined name' in ln]
        self.assertEqual(und, [])
        self.assertIn('import time as _ev_time', '\n'.join(self.b))
        self.assertNotIn(' time.perf_counter', '\n'.join(self.ins))

    def test_writer_used_and_summary_always_sealed(self):
        src = '\n'.join(self.b)
        self.assertIn("_ev_w = KEV.EvidenceWriter(a.evidence_out) if a.evidence_out else None", src)
        self.assertIn("if _ev_w is not None:                                  # EV 版：取證檔收尾與摘要", src)
        self.assertNotIn('open(a.evidence_out', src)                                     # 不直接開檔

    def test_runner_diff(self):
        a = open(os.path.join(HERE, 'run_door_k3a_int.sh')).read().splitlines()
        b = open(os.path.join(HERE, 'run_door_k3a_int_ev.sh')).read().splitlines()
        changed = [ln for o in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != 'equal' for ln in b[o[3]:o[4]]]
        code = [ln for ln in changed if not ln.lstrip().startswith('#')]
        self.assertTrue(any('isaac_door_k2_sim_ev.py --evidence-out "$DIR/evidence.jsonl"' in ln for ln in code))
        self.assertTrue(any('freeze_door_k3a_ev_dev_r2.sha256' in ln for ln in code))
        others = [ln for ln in code if 'isaac_door_k2_sim_ev.py' not in ln and 'freeze_door_k3a_ev_dev_r2' not in ln and not ln.lstrip().startswith(('sha256sum', 'say'))]
        self.assertEqual(others, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
