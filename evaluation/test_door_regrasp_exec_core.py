#!/usr/bin/env python3
"""door_regrasp_exec_core（ev9r8 --regrasp 與假執行端共用的執行端核心）驗證。參考包＝door_regrasp_ref_v4（登錄核驗通過）。
  X1 gate 生命週期：mode 2 提交 ⇒ Gate3 逐步、權威 gate3 回 (False, gate3_A_OPENING)；張爪＋小力滿確認窗 ⇒ (True, None)；
     mode 3 提交 ⇒ Gate3 結束（ended 記 B_CONFIRMED）、gate3 回 inactive、Gate4 啟動；
  X2 Gate3 結束後手指閉合（mode 6 情境）不使任何 gate 失敗；
  X3 mode 4 手指碰 bar_in 0.2 N（> 0.1）⇒ 中止 rg_contact:uflite_finger1/bar_in/0.200N@mode4，且閂鎖（後續步回同一原因）；
  X4 mode 1 起：紀錄不綁定當步 0.15 s ⇒ 不中止；恢復後再缺 0.25 s ⇒ rg_contact_data_invalid（不當作無接觸）；
  X5 HoldConfirm：TCP＝H_in·T_HG、兩指 bar_in 0.6 N ⇒ 2 s 後 hold_confirmed；之後一指降到 0.3 N ⇒ 中止 rg_hold:hold_lost_after_confirm；E0 只設一次；
  X6 bench_release_w2：宣告序號的 mode_seq 0 封裝經 W2Exec 進鏈 ⇒ ok、chain n_recv 1；序號不符／mode_seq≠0／同輪中止 ⇒ 拒絕且不進鏈；
  X7 參考包 v2（無 inner_grasp）⇒ 建構 ValueError；
  X8 Gate3 確認後提交 mode 3、Gate4 啟動後退開量一直 0 ⇒ 15 s 後中止 rg_gate4:retreat_timeout；
  X9（r2，Codex r65 §3）交接：Gate3 無有效證據即提交 mode 3 ⇒ 當步中止 rg_handover_no_valid_evidence；有證據 ⇒ handover.last_old_step＝k−1、
     first_new_step＝k、evidence_step ∈ {k−1, k}。

    python3 evaluation/test_door_regrasp_exec_core.py
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_geom_v3 as G3                              # noqa: E402
import door_regrasp_exec_core as XC                    # noqa: E402
import door_regrasp_mode1 as RG                        # noqa: E402
import test_door_regrasp_mode1_chain as TR             # noqa: E402

V4 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v4.json')))
V2 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v2.json')))
DT = 0.01
F1, F2 = 'uflite_finger1', 'uflite_finger2'
Q = np.radians(V4['q_ref_deg'])


def H(q):
    return G3.handle_pose_inner(PI.D3, q)


def rec(step, pts=()):
    return {'status': 'ok', 'complete': True, 'step': step, 't': round(step * DT, 6), 'points': [{'robot_link': l, 'class': c, 'f': f} for l, c, f in pts]}


def mk():
    auth = RG.RegraspAuthority('r', run_id_num=TR.RUNN, dt=DT)
    return auth, XC.RegraspExecCore(auth, V4, H, dt=DT)


class Run:
    def __init__(self, core, k0=0):
        self.c, self.k = core, k0
        self.T = np.eye(4)
        self.base, self.arm = np.zeros(3), np.zeros(6)
        self.fq = (0.0, 0.0)

    def go(self, n, mode, pts=(), rec_fn=None):
        out = None
        for _ in range(n):
            self.k += 1
            r = rec_fn(self.k) if rec_fn else rec(self.k, pts)
            out = self.c.step(self.k, round(self.k * DT, 6), mode, r, self.T, self.base, self.arm, self.fq, Q)
        return out


class TX(unittest.TestCase):
    def test_x1_x2_lifecycle(self):
        auth, c = mk()
        w = Run(c)
        c.on_commit(1, 0.01, 1, w.T)
        w.k = 1
        c.on_commit(2, 0.02, 2, w.T)
        self.assertEqual(auth.gates[3](2), (False, 'gate3_A_OPENING'))
        w.fq = (c.g[3].r['finger_open_m'],) * 2
        self.assertIsNone(w.go(150, 2))
        self.assertEqual(auth.gates[3](w.k), (True, None))
        c.on_commit(w.k, round(w.k * DT, 6), 3, w.T)
        self.assertEqual(c.ended[3]['state'], 'B_CONFIRMED')
        self.assertEqual(auth.gates[3](w.k), (False, 'gate3_inactive'))
        self.assertIn(4, c.active)
        w.fq = (0.0, 0.0)
        self.assertIsNone(w.go(50, 3))
        self.assertIsNone(c.abort)

    def test_x3_contact(self):
        auth, c = mk()
        w = Run(c)
        c.on_commit(1, 0.01, 1, w.T)
        w.k = 1
        self.assertIsNone(w.go(5, 4))
        why = w.go(1, 4, pts=[(F1, 'bar_in', 0.2)])
        self.assertEqual(why, 'rg_contact:uflite_finger1/bar_in/0.200N@mode4')
        self.assertEqual(w.go(3, 4), why)

    def test_x4_invalid(self):
        auth, c = mk()
        w = Run(c)
        c.on_commit(1, 0.01, 1, w.T)
        w.k = 1
        self.assertIsNone(w.go(15, 1, rec_fn=lambda k: {'status': 'query_failed', 'step': k, 't': k * DT}))
        self.assertIsNone(w.go(5, 1))
        self.assertIsNone(w.go(20, 1, rec_fn=lambda k: rec(k - 1)))
        self.assertEqual(w.go(6, 1, rec_fn=lambda k: None), 'rg_contact_data_invalid')

    def test_x5_hold(self):
        auth, c = mk()
        w = Run(c)
        c.on_commit(1, 0.01, 1, w.T)
        c.on_commit(1, 0.01, 6, w.T)
        w.k = 1
        w.T = H(Q) @ c.T_HG
        self.assertIsNone(w.go(210, 6, pts=[(F1, 'bar_in', 0.6), (F2, 'bar_in', 0.6)]))
        self.assertTrue(c.hold_confirmed(w.k))
        e0 = c.hold.E0.copy()
        why = w.go(1, 6, pts=[(F1, 'bar_in', 0.6), (F2, 'bar_in', 0.3)])
        self.assertEqual(why, 'rg_hold:hold_lost_after_confirm')
        self.assertTrue(np.array_equal(c.hold.E0, e0))
        self.assertEqual(sum(1 for e in c.hold.events if e.get('baseline')), 1)

    def test_x6_bench_release(self):
        w = TR.WorldR()
        e = TR.TC.env(0, 7, TR.U_HOLD, stamp=0.0)
        ok, why, r, _ = XC.bench_release_w2(w.wx, w.chain, e, 7, 0, 0.0, False)
        self.assertTrue(ok, why)
        self.assertEqual(w.chain.n_recv, 1)
        for env, seq, ab, exp in ((TR.TC.env(0, 8, TR.U_HOLD, stamp=0.0), 9, False, 'seq_not_announced'),
                                  (TR.TC.env(1, 9, TR.U_HOLD, stamp=0.0), 9, False, 'mode_seq_not_0'),
                                  (TR.TC.env(0, 9, TR.U_HOLD, stamp=0.0), 9, True, 'aborted_same_iteration')):
            w2 = TR.WorldR()
            ok, why, _, _ = XC.bench_release_w2(w2.wx, w2.chain, env, seq, 0, 0.0, ab)
            self.assertEqual((ok, why), (False, exp))
            self.assertEqual(w2.chain.n_recv, 0)

    def test_x7_registry(self):
        auth = RG.RegraspAuthority('r', run_id_num=TR.RUNN, dt=DT)
        with self.assertRaises(ValueError):
            XC.RegraspExecCore(auth, V2, H, dt=DT)

    def _to_mode3(self):
        auth, c = mk()
        w = Run(c)
        c.on_commit(1, 0.01, 1, w.T)
        w.k = 1
        c.on_commit(2, 0.02, 2, w.T)
        w.fq = (c.g[3].r['finger_open_m'],) * 2
        w.go(150, 2)
        c.on_commit(w.k, round(w.k * DT, 6), 3, w.T)
        return auth, c, w

    def test_x8_gate_timeout(self):
        auth, c, w = self._to_mode3()
        self.assertIsNone(c.abort)
        why = w.go(1600, 3)
        self.assertEqual(why, 'rg_gate4:retreat_timeout')

    def test_x9_handover(self):
        auth, c, w = self._to_mode3()
        h = c.handover
        self.assertEqual((h['last_old_step'], h['first_new_step']), (w.k - 1, w.k))
        self.assertIn(h['evidence_step'], (w.k - 1, w.k))
        auth2, c2 = mk()
        c2.on_commit(1, 0.01, 1, np.eye(4))
        c2.on_commit(5, 0.05, 3, np.eye(4))
        self.assertEqual(c2.abort, 'rg_handover_no_valid_evidence')
        self.assertEqual(c2.step(6, 0.06, 3, rec(6), np.eye(4), np.zeros(3), np.zeros(6), (0.0, 0.0), Q), 'rg_handover_no_valid_evidence')


if __name__ == '__main__':
    unittest.main(verbosity=1)
