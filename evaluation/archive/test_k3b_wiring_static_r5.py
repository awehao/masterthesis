#!/usr/bin/env python3
"""K3-B r5 接線靜態核對：
  W1 mk_adm16 重跑逐位元相同且對 ADM12 insert-only；ADM16 只有一處 HeldSupervisorP4b9、旗標 --w2-hold-r9、無 r8 監督器／旗標殘留。
  W2 ADM16 掛鉤次序同 ADM15（本趟過濾 → exec_unload_of → validated_commit → on_commit）。
  W3 跑法 r5：只換求解器 ADM16（--w2-hold-r9）、任務節點 k3b5（策略 r5／核心 release5）、判定器 k3b5、凍結 r5；執行端 ev8 不變。
  W5 mk_task_k3b5 重跑逐位元相同且 insert-only（對 k3b4 與 w1b2）；只在 --w2-unload 區塊內建 K3MReleasePolicy5。
  W4 判定器 k3b5：最外層 judge 以 prerelease_view 呼叫舊鏈；排除規則只有 K3B_REPLACED 與兩個 W1e 前綴。

    python3 evaluation/test_k3b_wiring_static_r5.py
"""
import hashlib
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def rd(p):
    with open(os.path.join(HERE, p), encoding='utf-8') as fh:
        return fh.read()


class TW(unittest.TestCase):
    def test_w1_regen(self):
        h0 = hashlib.sha256(rd('dl3_wgmpc_wg2_adm16_node.py').encode()).hexdigest()
        r = subprocess.run([sys.executable, 'results/door/mk_adm16_patch.py'], cwd=HERE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('insert-only ok', r.stdout)
        self.assertEqual(hashlib.sha256(rd('dl3_wgmpc_wg2_adm16_node.py').encode()).hexdigest(), h0)
        s = rd('dl3_wgmpc_wg2_adm16_node.py')
        self.assertEqual(s.count('W2M9.HeldSupervisorP4b9('), 1)
        for bad in ('HeldSupervisorP4b8(', 'w2_hold_r8', 'w2-hold-r8', 'W2M8'):
            self.assertNotIn(bad, s)
        self.assertIn("ap.add_argument('--w2-hold-r9', action='store_true'", s)

    def test_w2_hooks(self):
        s = rd('dl3_wgmpc_wg2_adm16_node.py')
        k = s.index('    def _w2r7_state(self, st):')
        blk = s[k:k + 2200]
        idx = [blk.index(x) for x in ("st.get('run_id') != self._w2src.run_id", 'W2M9.exec_unload_of(st)', 'W2M7.validated_commit(',
                                      'self._pause_sup.on_commit(')]
        self.assertEqual(idx, sorted(idx))

    def test_w3_run_script(self):
        s, s4 = rd('run_door_wide_k3b_r5.sh'), rd('run_door_wide_k3b_r4.sh')
        for k in ('evaluation/dl3_wgmpc_wg2_adm16_node.py --w2-retreat --w2-hold-r9 --w2-unload ', 'FRZ=evaluation/results/door/freeze_door_wide_k3b_dev_r5.sha256',
                  '通過與否由 door_k3m_check_k3b5.py 判定', 'evaluation/isaac_door_k2_sim_ev8.py --w2-release --w2-unload ',
                  'evaluation/door_k3m_task_node_k3b5.py --w2-release --w2-unload '):
            self.assertEqual(s.count(k), 1, k)
        body = '\n'.join(ln for ln in s.splitlines() if not ln.startswith('#'))
        self.assertNotIn('adm15_node.py', body)                               # 歷史註解可提及，執行行不得
        d = [ln for ln in set(s.splitlines()) ^ set(s4.splitlines()) if not ln.startswith('#')]
        self.assertEqual(len(d), 10, d)                                      # 求解器、任務節點、凍結、結束說明、開頭說明 各一對

    def test_w5_task_node(self):
        h0 = hashlib.sha256(rd('door_k3m_task_node_k3b5.py').encode()).hexdigest()
        r = subprocess.run([sys.executable, 'results/door/mk_task_k3b5_patch.py'], cwd=HERE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.count('insert-only ok'), 2)
        self.assertEqual(hashlib.sha256(rd('door_k3m_task_node_k3b5.py').encode()).hexdigest(), h0)
        s = rd('door_k3m_task_node_k3b5.py')
        j = s.index('        if a.w2_unload:                                                    # K3-B r4')
        self.assertIn('PK3B5.K3MReleasePolicy5(', s[j:j + 1200])
        self.assertEqual(s.count('PK3B5.K3MReleasePolicy5('), 1)

    def test_w4_judge(self):
        import door_k3m_check_k3b5 as J5
        self.assertEqual(J5.REPLACED_OLD_PREFIX, ('finger_effort_periodic_invalid_', 'finger_effort_periodic_short_'))
        self.assertTrue(J5._replaced('pause_task_not_done') and J5._replaced('finger_effort_periodic_short_3'))
        for w in ('finger_effort_not_effective', 'phase_sequence:[]', 'grasp_finger_contact_below_min', 'termination:ABORTED'):
            self.assertFalse(J5._replaced(w))
        import inspect
        src = inspect.getsource(J5.judge)
        self.assertIn('prerelease_view(', src)
        self.assertIn('F2.judge(view[0], view[1], gate, view[2], observer)', src)


if __name__ == '__main__':
    unittest.main(verbosity=1)
