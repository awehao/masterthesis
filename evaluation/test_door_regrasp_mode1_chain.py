#!/usr/bin/env python3
"""door_regrasp_mode1（換抓專用權威）× 真 E2 命令鏈（CmdChainE2＋EnvIntake＋W2Exec3），結構比照 test_door_w2_mode5_chain。
合成求解端：一般流＝目前已提交模式的命令；保持形模式（0、1、2、6）為保持命令；運動模式（3、4、5）為全身小速度命令；準備中的下一模式另發旁路候選。
  R1 正常 1→6：六次提交皆 applied_confirmed；手指目標 0–1 closed、2–5 open、6 closed（同提交步切換）；profile／模式名依本地表；運動模式套用底盤與手臂皆動；
  R2 准入：請求時 gate 假 ⇒ nack gate{k}；請求時真、提交當下假 ⇒ nack gate{k}_at_commit、不提交；缺 gate ⇒ gate_missing；
  R3 運動候選驗證：底盤 0.04 m/s ⇒ base_box；手臂 1.2 rad/s ⇒ arm_rate；底盤自保持跳到 0.02 m/s（> 0.5×0.01）⇒ accel_box；皆不提交；
  R4 保持形模式提交後套用到手臂運動（mode 6）⇒ applied_not_mode_shaped:arm_not_zero，不回切；
  R5 次序：前段未確認即請求 ⇒ previous_commit_not_confirmed；跳號 ⇒ bad_transition；unload=True ⇒ ValueError；
  R6 不改舊全域表：door_w2_mode.MODE_OF_SEQ／PROFILE_OF_SEQ 無 4–6、0–3 不變；mode 6 下 state_msg／stats 正常；
  R7 RegraspSolverSource：1–3 只在 HELD 受理；4–6 只在已提交 ≥ 3 後受理；保持形候選走 tag_candidate、運動候選走 tag_candidate_motion。

    python3 evaluation/test_door_regrasp_mode1_chain.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3m_applied_src as K3MSRC                  # noqa: E402
import door_regrasp_mode1 as RG                        # noqa: E402
import door_w2_exec3 as WX3                            # noqa: E402
import door_w2_mode as W2M                             # noqa: E402
import test_door_w2_exec_chain as TC                   # noqa: E402

RUNN, DT = TC.RUNN, TC.DT
U_HOLD = list(TC.U_HOLD)
U_MOVE = {3: [0.004, 0.0005, 0.003, 0.05, -0.03, 0.02, 0.0, 0.01, 0.0],
          4: [0.006, 0.002, 0.005, -0.04, 0.02, 0.0, 0.01, 0.0, 0.0],
          5: [0.003, -0.001, 0.002, 0.02, 0.0, -0.02, 0.0, 0.0, 0.01]}


def rq(seq, ms):
    return {'run_id': 'r', 'req_seq': seq, 'from_mode': RG.MODE_OF[ms - 1], 'to_mode': RG.MODE_OF[ms], 'mode_seq': ms}


class WorldR(TC.World):
    def __init__(self, gates=None, cand=None, post6=None):
        super().__init__()
        g = gates if gates is not None else {k: (lambda s: True) for k in RG.GATED}
        self.auth = RG.RegraspAuthority('r', run_id_num=RUNN, dt=DT, gates=g)
        self.intake = K3MSRC.EnvIntake(run_id_num=RUNN, stage_id=2, kinds=(0,))
        self.wx = WX3.W2Exec3(self.auth, self.intake)
        self.cand = cand or {}
        self.post6 = post6
        self.fingers = []

    def u_of(self, mode, candidate=False):
        if candidate and mode in self.cand:
            return list(self.cand[mode])
        if mode == 6 and self.post6 is not None and self.auth.commit and self.auth.commit['mode_seq'] == 6 and self.k > self.auth.commit['step'] + 3:
            return list(self.post6)
        return list(U_MOVE[mode]) if mode in RG.MOTION_MODES else list(U_HOLD)

    def solver_cycle(self, t, delay=0.0, stop_mode=None, cand_delay=0.0):
        self.seq += 1
        self.inbox.append((t, TC.env(self.committed, self.seq, self.u_of(self.committed), stamp=t)))
        if self.prep is not None:
            self.seq += 1
            self.inbox.append((t, TC.env(self.prep, self.seq, self.u_of(self.prep, candidate=True), stamp=t)))

    def runr(self, t_end, reqs=()):
        rows = []
        while self.k * DT <= t_end + 1e-9:
            t = round(self.k * DT, 6)
            if self.k % 5 == 0:
                self.solver_cycle(t)
            msgs = []
            for r in reqs:
                if r[0] <= t + 1e-9 and not r[2].get('_sent'):
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
                self.prep = None
            self.k += 1
        return rows


def full(w, upto=6):
    return w.runr(0.5 * upto + 1.0, reqs=[(0.5 * k, rq(k, k), {}) for k in range(1, upto + 1)])


class TR(unittest.TestCase):
    def test_r1_normal(self):
        w = WorldR()
        rows = full(w)
        cs = w.auth.commits
        self.assertEqual([c['mode_seq'] for c in cs], [1, 2, 3, 4, 5, 6])
        self.assertTrue(all(c['applied_confirmed'] is True for c in cs), [c['applied_confirmed'] for c in cs])
        self.assertIsNone(w.auth.aborted)
        f = {k: (ft, ms) for k, ft, ms in w.fingers}
        for c in cs:
            self.assertEqual(f[c['step']], (RG.FINGER_OF[c['mode_seq']], c['mode_seq']))
            self.assertEqual(f[c['step'] - 1][0], RG.FINGER_OF[c['mode_seq'] - 1])
            self.assertEqual(c['profile'], RG.PROFILE_OF[c['mode_seq']])
        for k in RG.MOTION_MODES:
            ck = next(c for c in cs if c['mode_seq'] == k)
            r = next(r for r in rows if r['step'] == ck['step'])
            self.assertGreater(max(abs(x) for x in r['u'][3:]), 0.01, k)
        self.assertEqual(w.auth.state_msg(0, 0.0)['mode'], 'CLOSE_INNER')

    def test_r2_gates(self):
        for k in RG.GATED:
            g = {j: (lambda s: True) for j in RG.GATED}
            g[k] = lambda s: (False, 'no_evidence')
            w = WorldR(gates=g)
            full(w)
            self.assertEqual(w.auth.mode_seq, k - 1, k)
            self.assertEqual(w.auth.rejections[0]['why'], f'gate{k}:no_evidence')
            st = {'n': 0}

            def gk(s, st=st):
                st['n'] += 1
                return st['n'] <= 1
            g[k] = gk
            w = WorldR(gates=g)
            full(w)
            self.assertEqual(w.auth.mode_seq, k - 1, k)
            self.assertTrue(str(w.auth.rejections[0]['why']).startswith(f'gate{k}_at_commit'), w.auth.rejections[0])
        g = {j: (lambda s: True) for j in RG.GATED if j != 4}
        w = WorldR(gates=g)
        full(w)
        self.assertEqual((w.auth.mode_seq, w.auth.rejections[0]['why']), (3, 'gate4:gate_missing'))

    def test_r3_motion_candidate(self):
        for k in RG.MOTION_MODES:
            base = U_HOLD if k == 3 else U_MOVE[k - 1]
            for cand, why in ((list(base[:3]) + [1.2] + [0.0] * 5, 'arm_rate'), ([0.04, 0.0, 0.0] + U_MOVE[k][3:], 'base_box'),
                              ([base[0] + 0.02, base[1], base[2]] + list(base[3:]), 'accel_box')):
                w = WorldR(cand={k: cand})
                full(w)
                self.assertEqual(w.auth.mode_seq, k - 1, (k, why))
                self.assertEqual(w.auth.rejections[0]['why'], f'candidate_invalid:{why}', (k, why))

    def test_r4_hold_mode_post_commit(self):
        w = WorldR(post6=[0.0022, -0.0013, 0.0018, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0])
        full(w)
        self.assertEqual(w.auth.mode_seq, 6)
        self.assertEqual(w.auth.aborted['why'], 'applied_not_mode_shaped:arm_not_zero')

    def test_r5_order(self):
        a = RG.RegraspAuthority('r', run_id_num=RUNN, dt=DT, gates={k: (lambda s: True) for k in RG.GATED})
        self.assertEqual(a.on_request(rq(1, 3), 0, 0.0)[1]['why'], 'bad_transition')
        a.on_request(rq(1, 1), 0, 0.0)
        a.commit = {'applied_confirmed': None, 'request': rq(1, 1), 'mode_seq': 1, 'step': 0, 't': 0.0, 'first_source_seq': 1}
        a.pending, a.mode_seq = None, 1
        self.assertEqual(a.on_request(rq(2, 2), 1, 0.01)[1]['why'], 'previous_commit_not_confirmed')
        with self.assertRaises(ValueError):
            RG.RegraspAuthority('r', run_id_num=RUNN, dt=DT, unload=True)

    def test_r6_global_tables(self):
        self.assertTrue(all(k not in W2M.MODE_OF_SEQ for k in (4, 5, 6)))
        self.assertEqual((W2M.PROFILE_OF_SEQ[0], W2M.PROFILE_OF_SEQ[1]), ('pause_hold', 'release_prep_test'))
        w = WorldR()
        full(w)
        st, s = w.auth.state_msg(1, 0.0), w.auth.stats()
        self.assertEqual((st['mode'], st['finger_target'], s['mode_name']), ('CLOSE_INNER', 'closed', 'CLOSE_INNER'))
        self.assertEqual(len(st['commits']), 6)

    def test_r7_solver_source(self):
        s = RG.RegraspSolverSource('r')
        self.assertFalse(s.on_request(rq(1, 1), False, 0.0))
        self.assertTrue(s.on_request(rq(1, 1), True, 0.0))
        self.assertEqual((s.tag_candidate(), s.tag_candidate_motion()), (1, None))
        s.on_state({'run_id': 'r', 'mode_seq': 1}, 0.05)                       # 候選 1 提交（真實時序）
        self.assertIsNone(s.prep)
        self.assertFalse(s.on_request(rq(4, 4), False, 0.06))                  # 尚未到 3：不受理 4
        s.on_state({'run_id': 'r', 'mode_seq': 3}, 0.1)
        self.assertTrue(s.on_request(rq(4, 4), False, 0.2))
        self.assertEqual((s.tag_candidate(), s.tag_candidate_motion()), (None, 4))
        s2 = RG.RegraspSolverSource('r')
        s2.on_state({'run_id': 'r', 'mode_seq': 2}, 0.1)
        self.assertFalse(s2.on_request(rq(3, 3), False, 0.2))
        self.assertTrue(s2.on_request(rq(3, 3), True, 0.2))
        self.assertEqual(s2.tag_candidate_motion(), 3)


if __name__ == '__main__':
    unittest.main(verbosity=1)
