#!/usr/bin/env python3
"""door_regrasp_wgmpc_v2 首筆承接約束啟用邊界（Codex r62 §2）與來源標記傳遞（§3）。
參考包＝ref v3＋segments stage（記憶體；同 test_door_regrasp_policy5）。只跑前 12 s（保持形 4 s → mode 3 → mode 4 起）。
  W1 solve_sp_fs 每次呼叫記 cfg.first_step 與 U_warm：first_step 非 None 的呼叫恰為候選首解（mode 3、mode 4 各至少一次），且全為 U_warm None；
     其餘一般輪 first_step 全為 None；
  W2 候選成立的提交事件含 t_req，提交時刻 − 請求時刻 ≤ 2.0 s；
  W3 套用命令來源標記：候選標記（candidate3）在發布 3＋生效 5 物理步後才出現在套用紀錄（標記隨佇列傳遞，不是當輪標記）。

    python3 evaluation/test_door_regrasp_wgmpc_v2.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_regrasp_wgmpc_v2 as W2                     # noqa: E402
import test_door_regrasp_policy5 as T5                 # noqa: E402

CALLS = []
_orig = W2.SFS.solve_sp_fs
APPLIED = {}


def _spy(K, z, u_prev, T, cfg, U_warm=None):
    CALLS.append({'fs': None if cfg.first_step is None else tuple(np.asarray(cfg.first_step[0], float)), 'warm_none': U_warm is None})
    return _orig(K, z, u_prev, T, cfg, U_warm=U_warm)


class TW(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        W2.SFS.solve_sp_fs = _spy
        try:
            cls.res = W2.run(T5.V3S, 'test', t_max=12.0)
        finally:
            W2.SFS.solve_sp_fs = _orig

    def test_w1_first_step_only_candidates(self):
        fs = [c for c in CALLS if c['fs'] is not None]
        ev = [e for e in self.res['events'] if e['to'] in (3, 4)]
        self.assertGreaterEqual(len(ev), 2, self.res['events'])
        n_cand = sum(self.res['candidates']['ok'].values()) + sum(self.res['candidates']['fail'].values())
        self.assertEqual(len(fs), n_cand)
        self.assertTrue(all(c['warm_none'] for c in fs))
        self.assertGreater(len(CALLS) - len(fs), 50)

    def test_w2_commit_within_timeout(self):
        for e in self.res['events']:
            self.assertLessEqual(e['t'] - e['t_req'], 2.0 + 1e-9)

    def test_w3_tag_through_queue(self):
        ev3 = next(e for e in self.res['events'] if e['to'] == 3)
        self.assertEqual(self.res['applied_first_step_by_tag']['candidate3'] - ev3['step'], W2.N_PUB + W2.N_DEL)
        acc = self.res['accel_ratio_by_transition']
        self.assertTrue(any(k.endswith('->candidate3') for k in acc), acc)
        self.assertTrue(any(k.startswith('candidate3->') for k in acc), acc)


if __name__ == '__main__':
    unittest.main(verbosity=1)
