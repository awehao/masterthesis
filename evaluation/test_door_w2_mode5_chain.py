#!/usr/bin/env python3
"""door_w2_mode5（多段模式＋mode 3 准入＋HELD 交接）× 真 E2 命令鏈（CmdChainE2＋EnvIntake＋applied_source＋W2Exec2）。
合成求解端：
- 一般流＝目前已提交模式的命令；mode 0–2 為保持形；mode 3 為退離（底盤保持形、手臂 0.05 rad/s）。
- 準備中的下一模式另發旁路候選（獨立 source_seq）；mode 3 候選＝退離首筆，不改一般流內容。
  P1 正常 1→2→3：三次提交皆 applied_confirmed；mode 2 提交步手指目標由 closed 改 open（同一步）、profile release_open；
     mode 3 提交步首筆實際套用為退離候選（手臂 0.05）；之後每步 mode 3、底盤保持形。
  P2 mode 3 准入：請求時 gate 假 ⇒ nack retreat_evidence；請求時真、提交當下假 ⇒ nack retreat_evidence_at_commit、不提交、仍 mode 2。
  P3 mode 3 候選驗證：手臂 1.2 rad/s ⇒ arm_rate；手臂由 0 跳 0.3（> 19.984×0.01）⇒ arm_accel_box；底盤 0.02 m/s ⇒ base_not_hold；皆不提交。
  P4 次序：mode 1 未確認就請求 2 ⇒ previous_commit_not_confirmed；跳號請求 3 ⇒ bad_transition；from／to 名稱不符 ⇒ bad_transition。
  P5 提交後 mode 3 套用底盤非保持形 ⇒ applied_not_mode_shaped:base_not_hold、不回切。
  P6 候選被拒、或權威中止後，一般流仍為原模式保持命令：鏈實際套用值全程保持形，無一步套用到候選內容（一般流未被候選污染）。
  P7 HeldSupervisorP4b5：HELD 中 handover ⇒ HANDED_OVER、cycle 回 zero False、applied 不再判保持形；非 HELD 不能交接；outcome held_end＝handover。
  P8 舊判定器相容：註冊新模式後 W2M.PROFILE_OF_SEQ[0／1] 不變；door_w2_check_mode3 對 W2c 真鏈 C1 紀錄仍通過。

    python3 evaluation/test_door_w2_mode5_chain.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3m_applied_src as K3MSRC                  # noqa: E402
import door_w2_exec2 as WX2                            # noqa: E402
import door_w2_mode5 as M5                             # noqa: E402
import test_door_w2_exec_chain as TC                   # noqa: E402

RUNN, DT = TC.RUNN, TC.DT
U_HOLD = list(TC.U_HOLD)
U_RET = U_HOLD[:3] + [0.05, -0.03, 0.02, 0.0, 0.01, 0.0]


def rq(seq, ms):
    return {'run_id': 'r', 'req_seq': seq, 'from_mode': M5.MODE_OF_SEQ5[ms - 1], 'to_mode': M5.MODE_OF_SEQ5[ms], 'mode_seq': ms}


class World5(TC.World):
    def __init__(self, gate=lambda step: True, cand3=None, base_mode3=None):
        super().__init__()
        self.auth = M5.ModeAuthority5('r', run_id_num=RUNN, dt=DT, gate3=lambda step: self.gate(step))
        self.gate = gate
        self.intake = K3MSRC.EnvIntake(run_id_num=RUNN, stage_id=2, kinds=(0,))
        self.wx = WX2.W2Exec2(self.auth, self.intake)
        self.cand3 = cand3 or U_RET
        self.base_mode3 = base_mode3
        self.fingers = []

    def u_of(self, mode):
        if mode == 3:
            u = list(self.cand3)
            if self.base_mode3 is not None and self.auth.commit and self.auth.commit['mode_seq'] == 3 and self.k > self.auth.commit['step'] + 3:
                u[:3] = self.base_mode3
            return u
        return list(U_HOLD)

    def solver_cycle(self, t, delay=0.0, stop_mode=None, cand_delay=0.0):
        self.seq += 1
        self.inbox.append((t, TC.env(self.committed, self.seq, self.u_of(self.committed), stamp=t)))
        if self.prep is not None:
            self.seq += 1
            self.inbox.append((t, TC.env(self.prep, self.seq, self.u_of(self.prep), stamp=t)))

    def run5(self, t_end, reqs=()):
        rows = []
        reqs = sorted(reqs, key=lambda x: x[0])
        while self.k * DT <= t_end + 1e-9:
            t = round(self.k * DT, 6)
            if self.k % 5 == 0:
                self.solver_cycle(t)
            due = [r for r in reqs if r[0] <= t + 1e-9 and not r[2].get('_sent')]
            msgs = []
            for r in due:
                r[2]['_sent'] = True
                msgs.append(r[1])
            inbox = [(x[1], 0.0) for x in self.inbox if x[0] <= t + 1e-9]
            self.inbox = [x for x in self.inbox if x[0] > t + 1e-9]
            sel, relay, _ = self.wx.step_begin(self.k, t, self.chain, inbox=inbox, requests=msgs)
            for r_ in relay:
                self.prep = r_['mode_seq']
            self.fingers.append((self.k, sel.get('finger_target'), sel['mode_seq']))
            self.chain.step(t, DT, [0.0] * 6)
            u9 = list(self.chain.u_prev) if self.chain.u_prev is not None else [0.0] * 9
            rows.append(self.wx.post_step(self.k, t, self.chain, True, u9, sel))
            st = self.auth.state_msg(self.k, t)
            if self.prep is not None and st['mode_seq'] >= self.prep:
                self.committed, self.prep = self.prep, None
            elif self.prep is not None and (st['aborted'] or (self.auth.pending is None and st['mode_seq'] < self.prep)):
                self.prep = None                                              # 中止或本請求已被拒（無待決、未提交）⇒ 撤回
            self.k += 1
        return rows


def full(w):
    return w.run5(3.0, reqs=[(0.5, rq(1, 1), {}), (1.0, rq(2, 2), {}), (1.5, rq(3, 3), {})])


class TP(unittest.TestCase):
    def test_p1_normal(self):
        w = World5()
        rows = full(w)
        cs = w.auth.commits
        self.assertEqual([c['mode_seq'] for c in cs], [1, 2, 3])
        self.assertTrue(all(c['applied_confirmed'] is True for c in cs))
        k2 = cs[1]['step']
        f = {k: (ft, ms) for k, ft, ms in w.fingers}
        self.assertEqual(f[k2 - 1][0], 'closed')
        self.assertEqual(f[k2], ('open', 2))
        self.assertEqual(cs[1]['profile'], 'release_open')
        k3 = cs[2]['step']
        r3 = next(r for r in rows if r['step'] == k3)
        self.assertEqual((r3['src'], r3['mode_seq']), (cs[2]['first_source_seq'], 3))
        self.assertAlmostEqual(r3['u'][3], 0.05, places=9)
        self.assertTrue(all(r['mode_seq'] == 3 for r in rows if r['step'] >= k3 and r['src'] >= 0))
        self.assertIsNone(w.auth.aborted)
        self.assertEqual(w.auth.state_msg(0, 0.0)['mode'], 'RETREAT')

    def test_p2_gate(self):
        w = World5(gate=lambda step: False)
        full(w)
        self.assertEqual(w.auth.mode_seq, 2)
        self.assertTrue(str(w.auth.last_reject['why']).startswith('retreat_evidence:'))
        st = {'n': 0}

        def g(step):
            st['n'] += 1
            return st['n'] <= 1                                               # 請求時真、提交當下假
        w = World5(gate=g)
        full(w)
        self.assertEqual(w.auth.mode_seq, 2)
        self.assertTrue(str(w.auth.last_reject['why']).startswith('retreat_evidence_at_commit'))

    def test_p3_candidate_validation(self):
        for cand, why in ((U_HOLD[:3] + [1.2] + [0.0] * 5, 'arm_rate'), (U_HOLD[:3] + [0.3] + [0.0] * 5, 'arm_accel_box'),
                          ([0.02, 0.0, 0.0] + U_RET[3:], 'base_not_hold')):
            w = World5(cand3=cand)
            full(w)
            self.assertEqual(w.auth.mode_seq, 2, why)
            self.assertEqual(w.auth.last_reject['why'], f'candidate_invalid:{why}')

    def test_p4_order(self):
        a = M5.ModeAuthority5('r', run_id_num=RUNN, dt=DT, gate3=lambda s: True)
        self.assertEqual(a.on_request(rq(1, 3), 0, 0.0)[1]['why'], 'bad_transition')
        a.on_request(rq(1, 1), 0, 0.0)
        a.commit = {'applied_confirmed': None, 'request': rq(1, 1), 'mode_seq': 1, 'step': 0, 't': 0.0, 'first_source_seq': 1}
        a.pending, a.mode_seq = None, 1
        self.assertEqual(a.on_request(rq(2, 2), 1, 0.01)[1]['why'], 'previous_commit_not_confirmed')
        bad = dict(rq(2, 2), from_mode='PAUSE_HELD')
        a.commit['applied_confirmed'] = True
        self.assertEqual(a.on_request(bad, 1, 0.01)[1]['why'], 'bad_transition')

    def test_p5_post_commit_base(self):
        w = World5(base_mode3=[0.02, 0.0, 0.0])
        full(w)
        self.assertEqual(w.auth.aborted['why'], 'applied_not_mode_shaped:base_not_hold')
        self.assertEqual(w.auth.mode_seq, 3)

    def test_p6_normal_stream_unpolluted(self):
        for w in (World5(cand3=U_HOLD[:3] + [1.2] + [0.0] * 5), World5(gate=lambda s: False)):
            rows = full(w)
            self.assertEqual(w.auth.mode_seq, 2)
            self.assertTrue(all(max(abs(x) for x in r['u'][3:]) < 1e-9 for r in rows), '鏈實際套用到候選內容')
        w = World5()
        w.run5(1.52, reqs=[(0.5, rq(1, 1), {}), (1.0, rq(2, 2), {}), (1.5, rq(3, 3), {})])
        self.assertIsNotNone(w.auth.pending)                                  # mode 3 待決、候選未到即中止
        w.auth.abort('test_abort', w.k, 1.52)
        rows = w.run5(2.5)
        self.assertEqual(w.auth.mode_seq, 2)
        self.assertTrue(all(max(abs(x) for x in r['u'][3:]) < 1e-9 for r in rows))

    def test_p7_handover(self):
        import door_k3m_pause_sup_p4 as P4
        sup = M5.HeldSupervisorP4b5(arm_win_s=0.5, held_max_s=60.0)
        self.assertFalse(sup.handover(1.0))
        seq, x = 100, 0.6
        for k in range(int(20.0 / 0.01)):
            t = round(k * 0.01, 4)
            if k % 2 == 0:
                sup.feed_state(k, t, x, -0.53, 1.57, 0.14, [0.0] * 6, wall=t)
            if k % 5 == 0 and k > 0:
                a = sup.cycle(t, True, P4.hold_cmd_ok(U_HOLD[:3], [0.0] * 6), wall=t)
                if a['publish']:
                    seq += 1
                    sup.note_published(seq, a['zero'])
            if sup.seq.state in ('CONFIRM', 'STOPPED', 'HELD'):
                sup.applied(k, t, U_HOLD[:3], [0.0] * 6, seq)
            if sup.seq.state == 'HELD' and t > (sup.t_held or 1e9) + 1.0:
                break
        self.assertEqual(sup.seq.state, 'HELD')
        self.assertTrue(sup.handover(t))
        self.assertEqual(sup.cycle(t + 0.05, True, False, wall=t + 0.05), {'publish': True, 'zero': False, 'handed_over': True})
        sup.applied(k + 1, t + 0.01, [0.0, 0.0, 0.0], [0.3] * 6, seq)       # 手臂移動：交接後不再判保持形
        o = sup.outcome(t + 1.0)
        self.assertEqual((o['state'], o['held_end']), ('HANDED_OVER', 'handover'))
        self.assertEqual(o['transitions']['HANDOVER'], t)

    def test_p8_registry_and_old_judge(self):
        import door_w2_mode as W2M
        self.assertEqual((W2M.PROFILE_OF_SEQ[0], W2M.PROFILE_OF_SEQ[1]), ('pause_hold', 'release_prep_test'))
        import test_door_w2_exec_chain as T
        t = T.TChain()
        t.test_c1_normal()


if __name__ == '__main__':
    unittest.main(verbosity=1)
