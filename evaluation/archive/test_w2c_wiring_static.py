#!/usr/bin/env python3
"""ev6r2／W2c 跑法的靜態核對（Isaac 主迴圈與整趟無法離線執行）。
  S1 mk_ev6r2_patch.py 重跑輸出與現檔逐位元相同，且由 ev5 insert-only（腳本內 assert）。
  S2 ev6r2 每物理步次序：step_begin（已知中止 → 請求 → select）→ 轉送 → 入口違規 → ex.step → post_step → sync_after → 權威中止處置
     → 權威狀態發布 → 迴圈末 sync_after（本步稍後的手指讀回、走廊等中止記在本步）→ step_id += 1。
  S3 _w2_known_abort 彙整 K2 閂鎖、PARK 違規、K3-M 違規；ev6r2 使用 ModeAuthority4 與 W2Exec2；外部中止不重記入口違規。
  S4 跑法 run_door_wide_w2c.sh：θ 固定 30°、登錄 w1e_params_30.json、凍結 freeze_door_wide_w2c_dev_r1；
     ev6r2＋--w2-mode-run-id、ADM11＋--w2-held＋--w2-mode-run-id、adapter_w2＋--cmd-env-v3（不帶 --cmd-env）、task_node_w2＋--w2-mode-run-id；
     其餘求解器參數與 W1f2 跑法逐字相同（只多 W2 旗標）。
  S5 舊判定器取代項只有兩個名稱（door_k3m_check_w2c.W2_REPLACED）。

    python3 evaluation/test_w2c_wiring_static.py
"""
import hashlib
import os
import re
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(p):
    with open(os.path.join(HERE, p), encoding='utf-8') as fh:
        return fh.read()


class TS(unittest.TestCase):
    def test_s1_regen(self):
        h0 = hashlib.sha256(rd('isaac_door_k2_sim_ev6r2.py').encode()).hexdigest()
        r = subprocess.run([sys.executable, 'results/door/mk_ev6r2_patch.py'], cwd=HERE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('insert-only ok', r.stdout)
        self.assertEqual(hashlib.sha256(rd('isaac_door_k2_sim_ev6r2.py').encode()).hexdigest(), h0)

    def test_s2_order(self):
        s = rd('isaac_door_k2_sim_ev6r2.py')
        keys = ['_w2sel, _relay6, _nv6 = w2x.step_begin(step_id, t, chain, known_abort=_w2_known_abort(), inbox=_ib6, requests=_rq6)',
                'node.w2_req_pub.publish(String(data=json.dumps(_r6, default=float)))', '_ev4_violation(f"w2_entry:{_nv6[0][\'why\']}")',
                'res = ex.step(step_id, t, a.physics_dt', 'w2x.post_step(step_id, t, chain, res.owner == AUTH_WHOLEBODY,',
                'w2x.sync_after(step_id, t, known_abort=_w2_known_abort())                # 本步 select 之後',
                '_ev4_violation(f"w2_mode:{w2_auth.aborted[\'why\']}")', 'node.w2_state_pub.publish(',
                '            w2x.sync_after(step_id, t, known_abort=_w2_known_abort())\n        step_id += 1']
        pos = []
        for k in keys:
            self.assertEqual(s.count(k), 1, k)
            pos.append(s.index(k))
        self.assertEqual(pos, sorted(pos))
        self.assertLess(s.index('        def _ev4_violation(_why4):'), pos[0])

    def test_s3_known_abort_and_classes(self):
        s = rd('isaac_door_k2_sim_ev6r2.py')
        i = s.index('    def _w2_known_abort():')
        body = s[i:i + 600]
        for k in ('k2_latch.latched', "park_state.get('violation')", "k3m.get('violation')"):
            self.assertIn(k, body)
        self.assertIn('W2M4.ModeAuthority4(', s)
        self.assertIn('W2X2.W2Exec2(', s)
        self.assertNotIn('W2M3.', s)
        self.assertIn("if not str(w2_auth.aborted['why']).startswith('external:'):", s)

    def test_s4_runner(self):
        s = rd('run_door_wide_w2c.sh')
        f = rd('run_door_wide_w1f2.sh')
        self.assertIn('W1B_THETA=30 ', s)
        self.assertIn('PRM=evaluation/results/door/w1e_params_30.json', s)
        self.assertIn('FRZ=evaluation/results/door/freeze_door_wide_w2c_dev_r1.sha256', s)
        self.assertIn('evaluation/isaac_door_k2_sim_ev6r2.py --w2-mode-run-id "$RUN_ID"', s)
        self.assertIn('evaluation/door_k3m_task_node_w2.py --w2-mode-run-id "$RUN_ID"', s)
        ad = re.search(r'^spawn adapter .*$', s, re.M).group(0)
        self.assertIn('arm_vel_adapter_w2.py', ad)
        self.assertIn('--cmd-env-v3', ad)
        self.assertNotRegex(ad, r'--cmd-env(\s|$)')

        def solver_block(t):
            i = t.index('spawn solver ')
            j = t.index('--out "$DIR/align_solver.json"', i)
            return t[i:j]
        so, sf = solver_block(s), solver_block(f)
        self.assertEqual(so.replace('dl3_wgmpc_wg2_adm11_node.py --w2-held --w2-held-max-s 60 --w2-mode-run-id "$RUN_ID"',
                                    'dl3_wgmpc_wg2_adm9_node.py'), sf)

    def test_s5_replaced(self):
        sys.path.insert(0, HERE)
        import door_k3m_check_w2c as W2C
        self.assertEqual(W2C.W2_REPLACED, ('pause_solver_outcome_not_done', 'pause_task_not_done'))


if __name__ == '__main__':
    unittest.main(verbosity=1)
