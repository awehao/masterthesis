#!/usr/bin/env python3
"""ADM13 靜態核對（Codex 20261026_021025 §2）：
  S1 mk_adm13_patch 重跑輸出與現檔逐位元相同且 insert-only（對 ADM12）。
  S2 --w2-release-hold 預設關、需 --w2-retreat；只有開啟時才換 HeldSupervisorP4b6。
  S3 得知 mode 2：在 on_state 之後；受理請求：在 on_request 之後；提交時刻取權威 commits 列（無時刻 ⇒ 不切換）。
  S4 ADM12 既有的 HELD 交接條件（state == 'HELD'）未改：release_open 為 HELD 內部子狀態。

    python3 evaluation/test_adm13_static.py
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
        h0 = hashlib.sha256(rd('dl3_wgmpc_wg2_adm13_node.py').encode()).hexdigest()
        r = subprocess.run([sys.executable, 'results/door/mk_adm13_patch.py'], cwd=HERE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('insert-only ok', r.stdout)
        self.assertEqual(hashlib.sha256(rd('dl3_wgmpc_wg2_adm13_node.py').encode()).hexdigest(), h0)

    def test_s2_flag(self):
        s = rd('dl3_wgmpc_wg2_adm13_node.py')
        self.assertIn("ap.add_argument('--w2-release-hold', action='store_true'", s)
        self.assertIn("ap.error('--w2-release-hold 需要 --w2-retreat')", s)
        i = s.index("        if getattr(a, 'w2_release_hold', False) and self._w2r is not None:")
        self.assertIn('W2M6.HeldSupervisorP4b6(', s[i:i + 300])
        self.assertEqual(s.count('W2M6.HeldSupervisorP4b6('), 1)

    def test_s3_hooks(self):
        s = rd('dl3_wgmpc_wg2_adm13_node.py')
        a = s.index("            self._w2src.on_state(json.loads(m.data), self.sim_now())\n")
        self.assertEqual(s[a:a + 300].count('self._w2rh_state(json.loads(m.data))'), 1)
        b = s.index("        self._w2src.on_request(msg, held=(self._pause_sup is not None and self._pause_sup.state == 'HELD'), t=self.sim_now())\n")
        self.assertIn('self._w2rh_pending()', s[b:b + 300])
        k = s.index('    def _w2rh_state(self, st):')
        blk = s[k:k + 1200]
        self.assertIn('W2M6.commit_t_of(st, 2)', blk)
        self.assertLess(blk.index('if _tc13 is None:'), blk.index('self._pause_sup.release_open('))

    def test_s4_handover_unchanged(self):
        s, s12 = rd('dl3_wgmpc_wg2_adm13_node.py'), rd('dl3_wgmpc_wg2_adm12_node.py')
        for k in ("and self._pause_sup.state == 'HELD' and self._pause_sup.handover(self.sim_now()):",
                  "if self._w2r['handover'] is None and self._pause_sup is not None and self._pause_sup.state == 'HELD' \\"):
            self.assertEqual(s.count(k), s12.count(k))
            self.assertEqual(s.count(k), 1)


if __name__ == '__main__':
    unittest.main(verbosity=1)
