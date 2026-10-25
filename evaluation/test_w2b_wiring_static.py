#!/usr/bin/env python3
"""ev6／任務節點 w2 的靜態接線核對（無 Isaac；Isaac 主迴圈無法離線執行，故以原始碼次序與 insert-only 核對）。
  S1 兩個生成腳本重跑後輸出與現檔逐位元相同，且原檔每一行依序保留（insert-only，腳本內 assert）。
  S2 ev6 主迴圈次序：v3 佇列交 on_env3 → 權威違規走 _ev4_violation → 請求 on_request → pending 才轉送 → pre_step（進鏈）
     → ex.step → post_step（鏈讀出）→ 權威中止走 _ev4_violation → 發布權威狀態；全部在同一個 while 迴圈內。
  S3 ev6 的 v2 回呼在 W2 開啟時先 return（不解碼、不進鏈）；W2 需 --cmd-env 與 --run-id（exit 41）。
  S4 任務節點 w2：W2 策略只在 --w2-mode-run-id 時建立；outbox 只發 /door/w2_mode_req_task；權威狀態只訂 /door/w2_mode_state。
  S5 ev6 不訂、不發 /door/w2_mode_req_task 以外的請求來源；/door/w2_mode_req 只在 pending 分支發布。

    python3 evaluation/test_w2b_wiring_static.py
"""
import hashlib
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    return hashlib.sha256(open(os.path.join(HERE, p), 'rb').read()).hexdigest()


class TStatic(unittest.TestCase):
    def test_s1_regenerate(self):
        for mk, outp in (('results/door/mk_ev6_patch.py', 'isaac_door_k2_sim_ev6.py'),
                         ('results/door/mk_task_w2_patch.py', 'door_k3m_task_node_w2.py')):
            before = sha(outp)
            r = subprocess.run([sys.executable, mk], cwd=HERE, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn('insert-only ok', r.stdout)
            self.assertEqual(sha(outp), before, outp)

    def test_s2_loop_order(self):
        s = open(os.path.join(HERE, 'isaac_door_k2_sim_ev6.py')).read()
        keys = ['w2x.on_env3(_e6, t, _wall6)', "_ev4_violation(f\"w2_entry:{_v6['why']}\")", 'w2_auth.on_request(_r6, step_id, t)',
                "if _k6 == 'pending':", 'node.w2_req_pub.publish(', '_w2sel = w2x.pre_step(step_id, t, chain)',
                'res = ex.step(step_id, t, a.physics_dt', 'w2x.post_step(step_id, t, chain, res.owner == AUTH_WHOLEBODY,',
                "_ev4_violation(f\"w2_mode:{w2_auth.aborted['why']}\")", 'node.w2_state_pub.publish(']
        pos = []
        for k in keys:
            self.assertEqual(s.count(k), 1, k)
            pos.append(s.index(k))
        self.assertEqual(pos, sorted(pos))
        loop0 = s.index('        def _ev4_violation(_why4):')
        self.assertLess(loop0, pos[0])
        self.assertLess(pos[-1], s.index('        step_id += 1'))

    def test_s3_v2_ignored_and_args(self):
        s = open(os.path.join(HERE, 'isaac_door_k2_sim_ev6.py')).read()
        i = s.index('    def _wb_env(self, m: Float64MultiArray):')
        body = s[i:s.index('    def _wb_env3(', i)]
        self.assertLess(body.index('return'), body.index('ENV.decode'))
        self.assertIn("if not a.cmd_env or not a.run_id:", s)
        self.assertIn('sys.exit(41)', s)

    def test_s4_task_node(self):
        s = open(os.path.join(HERE, 'door_k3m_task_node_w2.py')).read()
        self.assertEqual(s.count("'/door/w2_mode_req_task'"), 1)
        self.assertEqual(s.count("'/door/w2_mode_state'"), 1)
        self.assertNotIn("'/door/w2_mode_req'", s)
        i = s.index('    if a.w2_mode_run_id:                                                   # W2：同參數另建 W2 策略')
        self.assertIn('PW2.K3MW2Policy(', s[i:i + 400])

    def test_s5_exec_request_topics(self):
        s = open(os.path.join(HERE, 'isaac_door_k2_sim_ev6.py')).read()
        self.assertEqual(s.count("'/door/w2_mode_req'"), 1)
        self.assertEqual(s.count("'/door/w2_mode_req_task'"), 1)
        self.assertEqual(s.count('node.w2_req_pub.publish('), 1)


if __name__ == '__main__':
    unittest.main(verbosity=1)
