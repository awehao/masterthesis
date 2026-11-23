#!/usr/bin/env python3
"""W2c r2 的靜態核對。
  U1 mk_ev6r3_patch.py 重跑輸出與現檔逐位元相同，且由 ev5 insert-only（腳本內 assert）。
  U2 ev6r3 與 ev6r2 去掉註解列後，差異只有 room_run.json w2.config 紀錄（控制不變）。
  U3 run_door_wide_w2c_r2.sh 與 r1 去掉註解列後，差異只有凍結名稱、執行端檔名（ev6r2→ev6r3）、結尾判定器名稱。
  U4 判定器 w2c2 的登錄值、容差為固定常數（不讀待判資料）。

    python3 evaluation/test_w2c2_wiring_static.py
"""
import difflib
import hashlib
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(p):
    with open(os.path.join(HERE, p), encoding='utf-8') as fh:
        return fh.read()


def code(p):
    return [x for x in rd(p).splitlines() if not x.startswith('#')]


class TU(unittest.TestCase):
    def test_u1_regen(self):
        h0 = hashlib.sha256(rd('isaac_door_k2_sim_ev6r3.py').encode()).hexdigest()
        r = subprocess.run([sys.executable, 'results/door/mk_ev6r3_patch.py'], cwd=HERE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('insert-only ok', r.stdout)
        self.assertEqual(hashlib.sha256(rd('isaac_door_k2_sim_ev6r3.py').encode()).hexdigest(), h0)

    def test_u2_ev6r3_record_only(self):
        d = [x for x in difflib.unified_diff(code('isaac_door_k2_sim_ev6r2.py'), code('isaac_door_k2_sim_ev6r3.py'), lineterm='', n=0)
             if x[:1] in '+-' and not x.startswith(('+++', '---'))]
        rm = [x for x in d if x.startswith('-')]
        add = [x for x in d if x.startswith('+')]
        self.assertEqual(len(rm), 1)
        self.assertIn("rec['w2'] = {'auth': w2_auth.stats()", rm[0])
        self.assertTrue(add[0].startswith("+        rec['w2'] = {'config': {"))
        self.assertTrue(add[-1].lstrip('+').strip().startswith("'auth': w2_auth.stats()"))
        self.assertEqual(rm[0].split("'auth': ", 1)[1], add[-1].split("'auth': ", 1)[1])

    def test_u3_runner_diff(self):
        d = [x for x in difflib.unified_diff(code('run_door_wide_w2c.sh'), code('run_door_wide_w2c_r2.sh'), lineterm='', n=0)
             if x[:1] in '+-' and not x.startswith(('+++', '---'))]
        self.assertEqual(len(d), 6)
        pairs = list(zip([x for x in d if x.startswith('-')], [x for x in d if x.startswith('+')]))
        subs = [('freeze_door_wide_w2c_dev_r1', 'freeze_door_wide_w2c_dev_r2'), ('isaac_door_k2_sim_ev6r2.py', 'isaac_door_k2_sim_ev6r3.py'),
                ('door_k3m_check_w2c.py', 'door_k3m_check_w2c2.py')]
        for (a, b), (x, y) in zip(pairs, subs):
            self.assertEqual(a[1:].replace(x, y), b[1:])

    def test_u4_constants(self):
        sys.path.insert(0, HERE)
        import door_k3m_check_w2c2 as J
        self.assertEqual((J.REG['test_hold_s'], J.REG['commit_timeout_s'], J.REG['req_delay_s'], J.REG['state_stale_s']), (3.0, 2.0, 1.0, 1.5))
        self.assertEqual((J.REG['cand_timeout_s'], J.REG['post_commit_timeout_s'], J.REG['applied_timeout_s'], J.REG['held_max_s']),
                         (1.0, 0.25, 0.25, 60.0))
        self.assertEqual((J.CLOCK_TOL, J.DELIVER_MAX), (0.05, 0.5))
        src = rd('door_k3m_check_w2c2.py')
        self.assertIn("test_hold_s=REG['test_hold_s'], commit_timeout_s=REG['commit_timeout_s']", src)


if __name__ == '__main__':
    unittest.main(verbosity=1)
