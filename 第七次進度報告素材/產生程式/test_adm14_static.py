#!/usr/bin/env python3
"""ADM14 靜態核對（Codex 20261026_023226）：
  S1 mk_adm14_patch 重跑輸出與現檔逐位元相同且 insert-only（對 ADM12，不以 ADM13 為基底）。
  S2 --w2-hold-r7 預設關、需 --w2-retreat；--w2-unload 需 --w2-hold-r7；只有開啟時才換 HeldSupervisorP4b7。
  S3 狀態掛鉤：在 on_state 之後；先濾 run_id 再設 pending；提交時刻只經 validated_commit；受理掛鉤在 on_request 之後。
  S4 ADM12 既有 HELD 交接條件未改；ADM14 不含 ADM13 的 release_open／commit_t_of 掛鉤。

    python3 evaluation/test_adm14_static.py
"""
import hashlib
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def rd(p):
    with open(os.path.join(HERE, p), encoding='utf-8') as fh:
        return fh.read()


class TS(unittest.TestCase):
    def test_s1_regen(self):
        h0 = hashlib.sha256(rd('dl3_wgmpc_wg2_adm14_node.py').encode()).hexdigest()
        r = subprocess.run([sys.executable, 'results/door/mk_adm14_patch.py'], cwd=HERE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('insert-only ok', r.stdout)
        self.assertEqual(hashlib.sha256(rd('dl3_wgmpc_wg2_adm14_node.py').encode()).hexdigest(), h0)
        self.assertIn("open('dl3_wgmpc_wg2_adm12_node.py')", rd('results/door/mk_adm14_patch.py'))

    def test_s2_flags(self):
        s = rd('dl3_wgmpc_wg2_adm14_node.py')
        self.assertIn("ap.add_argument('--w2-hold-r7', action='store_true'", s)
        self.assertIn("ap.error('--w2-hold-r7 需要 --w2-retreat')", s)
        self.assertIn("ap.error('--w2-unload 需要 --w2-hold-r7')", s)
        i = s.index("        if getattr(a, 'w2_hold_r7', False) and self._w2r is not None:")
        self.assertIn('W2M7.HeldSupervisorP4b7(', s[i:i + 300])
        self.assertEqual(s.count('W2M7.HeldSupervisorP4b7('), 1)

    def test_s3_hooks(self):
        s = rd('dl3_wgmpc_wg2_adm14_node.py')
        a = s.index("            self._w2src.on_state(json.loads(m.data), self.sim_now())\n")
        self.assertEqual(s[a:a + 300].count('self._w2r7_state(json.loads(m.data))'), 1)
        b = s.index("        self._w2src.on_request(msg, held=(self._pause_sup is not None and self._pause_sup.state == 'HELD'), t=self.sim_now())\n")
        self.assertIn('self._w2r7_pending()', s[b:b + 300])
        k = s.index('    def _w2r7_state(self, st):')
        blk = s[k:k + 1500]
        self.assertLess(blk.index("st.get('run_id') != self._w2src.run_id"), blk.index('self._w2r7_pending()'))
        self.assertLess(blk.index('W2M7.validated_commit('), blk.index('self._pause_sup.on_commit('))
        self.assertEqual(s.count('self._pause_sup.on_commit('), 1)

    def test_s4(self):
        s, s12 = rd('dl3_wgmpc_wg2_adm14_node.py'), rd('dl3_wgmpc_wg2_adm12_node.py')
        for k in ("and self._pause_sup.state == 'HELD' and self._pause_sup.handover(self.sim_now()):",
                  "if self._w2r['handover'] is None and self._pause_sup is not None and self._pause_sup.state == 'HELD' \\"):
            self.assertEqual((s.count(k), s12.count(k)), (1, 1))
        self.assertNotIn('release_open(', s)
        self.assertNotIn('commit_t_of', s)


if __name__ == '__main__':
    unittest.main(verbosity=1)
