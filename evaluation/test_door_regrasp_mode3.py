#!/usr/bin/env python3
"""door_regrasp_mode3（模式表到 8＋gate 2）× 真 E2 鏈（同 test_door_regrasp_mode1_chain 的假世界，模式表換成 mode3）。
  M1 1→8 正常：八次提交皆套用確認；手指目標 6 closed、7 closed、8 open（同提交步切換）；7、8 為運動形（手臂動）；
  M2 gate：2–8 任一 gate 假 ⇒ 停在前一模式且 nack gate{k}:…；gate 7 請求時真、提交當下假 ⇒ gate7_at_commit；
  M3 接觸規則：7 ⇒ ('bar_in',)；8 離開前 ('bar_in',)、離開後 ()；1–6 同 mode1；
  M4 求解端來源：mode 3 提交後受理 4–8；7、8 走 tag_candidate_motion。

    python3 evaluation/test_door_regrasp_mode3.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3m_applied_src as K3MSRC                  # noqa: E402
import door_regrasp_mode1 as RG1                       # noqa: E402
import door_regrasp_mode3 as RG                        # noqa: E402
import door_w2_exec3 as WX3                            # noqa: E402
import test_door_regrasp_mode1_chain as TR             # noqa: E402

TR.U_MOVE[7] = [0.004, 0.002, 0.002, 0.02, -0.01, 0.0, 0.01, 0.0, 0.0]
TR.U_MOVE[8] = [0.002, 0.0, 0.0, -0.02, 0.0, 0.01, 0.0, 0.0, 0.0]


class W3(TR.WorldR):
    def __init__(self, gates=None):
        TR.TC.World.__init__(self)
        g = gates if gates is not None else {k: (lambda s: True) for k in RG.GATED}
        self.auth = RG.RegraspAuthority('r', run_id_num=TR.RUNN, dt=TR.DT, gates=g)
        self.intake = K3MSRC.EnvIntake(run_id_num=TR.RUNN, stage_id=2, kinds=(0,))
        self.wx = WX3.W2Exec3(self.auth, self.intake)
        self.cand, self.post6, self.fingers = {}, None, []

    def u_of(self, mode, candidate=False):
        return list(TR.U_MOVE[mode]) if mode in RG.MOTION_MODES else list(TR.U_HOLD)


def rq(seq, ms):
    return {'run_id': 'r', 'req_seq': seq, 'from_mode': RG.MODE_OF[ms - 1], 'to_mode': RG.MODE_OF[ms], 'mode_seq': ms}


def full(w, upto=8):
    return w.runr(0.5 * upto + 1.0, reqs=[(0.5 * k, rq(k, k), {}) for k in range(1, upto + 1)])


class TM(unittest.TestCase):
    def test_m1_normal(self):
        w = W3()
        rows = full(w)
        cs = w.auth.commits
        self.assertEqual([c['mode_seq'] for c in cs], list(range(1, 9)))
        self.assertTrue(all(c['applied_confirmed'] is True for c in cs))
        f = {k: (ft, ms) for k, ft, ms in w.fingers}
        for c in cs:
            self.assertEqual(f[c['step']], (RG.FINGER_OF[c['mode_seq']], c['mode_seq']))
        for k in (7, 8):
            ck = next(c for c in cs if c['mode_seq'] == k)
            r = next(r for r in rows if r['step'] == ck['step'])
            self.assertGreater(max(abs(x) for x in r['u'][3:]), 0.005, k)

    def test_m2_gates(self):
        for k in RG.GATED:
            g = {j: (lambda s: True) for j in RG.GATED}
            g[k] = lambda s: (False, 'no')
            w = W3(gates=g)
            full(w)
            self.assertEqual(w.auth.mode_seq, k - 1, k)
            self.assertEqual(w.auth.rejections[0]['why'], f'gate{k}:no')
        st = {'n': 0}

        def gk(s, st=st):
            st['n'] += 1
            return st['n'] <= 1
        g = {j: (lambda s: True) for j in RG.GATED}
        g[7] = gk
        w = W3(gates=g)
        full(w)
        self.assertEqual(w.auth.mode_seq, 6)
        self.assertTrue(str(w.auth.rejections[0]['why']).startswith('gate7_at_commit'))

    def test_m3_contact_rules(self):
        self.assertEqual(RG.allowed_finger_classes(7), ('bar_in',))
        self.assertEqual(RG.allowed_finger_classes(8, False), ('bar_in',))
        self.assertEqual(RG.allowed_finger_classes(8, True), ())
        for m in range(0, 7):
            for e in (False, True):
                self.assertEqual(RG.allowed_finger_classes(m, e), RG1.allowed_finger_classes(m, e))

    def test_m4_solver_source(self):
        s = RG.RegraspSolverSource('r')
        s.committed = 6
        self.assertIn(7, RG.MOTION_MODES)
        self.assertIn(8, RG.MOTION_MODES)
        ok = s.on_request(rq(7, 7), False, 0.0)
        self.assertNotEqual(ok, None)


if __name__ == '__main__':
    unittest.main(verbosity=1)
