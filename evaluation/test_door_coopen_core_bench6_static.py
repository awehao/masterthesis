#!/usr/bin/env python3
"""跑法 bench6 的版本化差異靜態測試（Codex r27）。
  S1 bench6 對 bench5：被刪改的非註解行只限 FRZ、PRM、開場字樣、求解端首行（節點 r3→r4、--offset-hold-on-free 後加 --offset-update-in-open，其餘逐字相同）、結束字樣；
     新增行只多一行登錄旗標拒跑檢查。
  S2 bench6 的求解端參數交給 ADM17r4 的 argparse 解析成功（不啟動 ROS）：offset_update_in_open、offset_hold_on_free、bench_start 皆為 True。
  S3 登錄 r10 對 r9：只多 solver_change 的新鍵、schema 與 derived_from_r9；其餘鍵值逐一相同。

    python3 evaluation/test_door_coopen_core_bench6_static.py
"""
import difflib
import json
import os
import re
import shlex
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def diff(a, b):
    A, B = open(os.path.join(HERE, a)).read().splitlines(), open(os.path.join(HERE, b)).read().splitlines()
    d = [ln for ln in difflib.unified_diff(A, B, lineterm='', n=0) if ln[:1] in '+-' and not ln.startswith(('+++', '---'))]
    rm = [ln[1:] for ln in d if ln.startswith('-') and not ln[1:].lstrip().startswith('#')]
    add = [ln[1:] for ln in d if ln.startswith('+') and not ln[1:].lstrip().startswith('#')]
    return rm, add


class S(unittest.TestCase):
    def test_s1_runner_diff(self):
        rm, add = diff('run_door_coopen_core_bench5.sh', 'run_door_coopen_core_bench6.sh')
        self.assertEqual(len(rm), 5, rm)
        self.assertTrue(rm[0].startswith('FRZ=') and rm[1].startswith('PRM=') and rm[2].startswith('say "=== 核心段 bench（'), rm[:3])
        self.assertTrue(rm[3].startswith('spawn solver python3 -u evaluation/dl3_wgmpc_wg2_adm17r3_node.py'), rm[3])
        self.assertTrue(rm[4].startswith('say "=== 結束 rc='), rm[4])
        self.assertEqual(len(add), 6, add)
        self.assertTrue(add[0].startswith('FRZ=') and add[1].startswith('PRM='), add[:2])
        self.assertTrue(add[2].startswith('[ "$(python3') and 'offset_update_in_open' in add[2] and add[2].rstrip().endswith('exit 97; }'), add[2])
        self.assertEqual(rm[3].replace('adm17r3', 'adm17r4').replace('--offset-hold-on-free', '--offset-hold-on-free --offset-update-in-open'), add[4])
        self.assertTrue(add[5].startswith('say "=== 結束 rc=') and 'door_coopen_check_core12.py' in add[5], add[5])

    def test_s2_solver_args_parse(self):
        src = open(os.path.join(HERE, 'run_door_coopen_core_bench6.sh')).read()
        m = re.search(r'spawn solver python3 -u (evaluation/dl3_wgmpc_wg2_adm17r4_node\.py .*?--out "\$DIR/align_solver.json")', src, re.S)
        self.assertIsNotNone(m)
        cmd = m.group(1).replace('\\\n', ' ')
        env = {'PRM': 'evaluation/results/door/coopen_core_params_48_r10.json', 'RUN_ID': 'x', 'W1B_COORD': '/door/coord', 'ARM_IDENT':
               'evaluation/results/wgmpc_arm_sp_ident_free4.json', 'DIR': '/tmp/claude-1000/bench6_static'}
        for k, v in env.items():
            cmd = cmd.replace(f'"${k}"', v).replace(f'${k}', v)
        argv = shlex.split(cmd)[1:]
        code = ('import importlib.util,sys,os;sys.path.insert(0,os.path.dirname(sys.argv[1]));sp=importlib.util.spec_from_file_location("m",sys.argv[1]);m=importlib.util.module_from_spec(sp);sys.modules["m"]=m;'
                'sp.loader.exec_module(m);import argparse;P=argparse.ArgumentParser.parse_args;'
                'argparse.ArgumentParser.parse_args=lambda self,*a,**k:(lambda r:(print("ARGS",r.offset_update_in_open,r.offset_hold_on_free,r.bench_start),sys.exit(0)))(P(self,*a,**k));'
                'sys.argv=[sys.argv[1]]+sys.argv[2:];m.main()')
        p = subprocess.run([sys.executable, '-c', code, os.path.join(HERE, 'dl3_wgmpc_wg2_adm17r4_node.py')] + argv,
                           cwd=os.path.dirname(HERE), capture_output=True, text=True, timeout=120)
        self.assertIn('ARGS True True True', p.stdout, p.stderr[-400:])

    def test_s3_registry_diff(self):
        r9 = json.load(open(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r9.json')))
        r10 = json.load(open(os.path.join(HERE, 'results', 'door', 'coopen_core_params_48_r10.json')))
        self.assertEqual(set(r10) - set(r9), {'derived_from_r9'})
        for k in r9:
            if k not in ('solver_change', 'schema'):
                self.assertEqual(r9[k], r10[k], k)
        sc9, sc10 = r9['solver_change'], r10['solver_change']
        self.assertEqual(set(sc10) - set(sc9), {'offset_update_in_open', 'rule_r10', 'rule_r9_replaced', 'basis_r10'})
        self.assertEqual({k: sc10[k] for k in sc9 if k != 'node'}, {k: sc9[k] for k in sc9 if k != 'node'})
        self.assertEqual(sc10['node'], 'dl3_wgmpc_wg2_adm17r4_node.py')
        self.assertIs(sc10['offset_update_in_open'], True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
