#!/usr/bin/env python3
"""W2 r4 權威＋接線 r2 × 真 E2 命令鏈（Codex 20261025_220903 兩項必修的反例）。
  D1 Codex 重現：正常提交後持續送**新鮮** mode 1 命令、但全部被 K2 過濾擋下 ⇒ 鏈逾時、無套用來源；
     r4 以最後有效套用證據計時 ⇒ 0.25 s 內中止 post_commit_not_applied、mode_seq 仍 1（r3 的 ModeAuthority3 同情境不中止，作對照）。
  D2 持續收到但鏈拒收（輪級檢查失效）⇒ 同樣有界中止（提交步之後才失效）。
  D3 同一步「既有中止＋請求＋候選」：known_abort 先同步進權威 ⇒ 請求 nack after_abort、不轉送、不提交、不選定。
  D4 已 pending、候選已緩衝，本步 known_abort ⇒ 不提交（中止優先於 select）。
  D5 本步 select 已提交、之後才發現既有中止 ⇒ events 次序＝commit 在前、abort 在後，同一步，不倒寫；mode_seq 仍 1。
  D6 正常路徑（無中止）：提交、保持 3 s 無中止；逐步列含 sel_stamp 與 sel_t_recv 分列，sel_stamp ≤ sel_t_recv。
  D7 回歸：r3 的真鏈情境 C1–C8 以 r4／exec2 重跑仍成立（正常、過濾、拒收、限幅、斷流、Codex 重現、缺確認、老化）。

    python3 evaluation/test_door_w2_exec2_chain.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3m_applied_src as K3MSRC                  # noqa: E402
import door_w2_check_mode3 as CK3                      # noqa: E402
import door_w2_exec2 as WX2                            # noqa: E402
import door_w2_mode3 as M3                             # noqa: E402
import door_w2_mode4 as M4                             # noqa: E402
import test_door_w2_exec_chain as TC                   # noqa: E402

RUNN, DT = TC.RUNN, TC.DT
REQ = dict(TC.REQ)
env = TC.env


class World2(TC.World):
    """r3 的 World，改用 ModeAuthority4（或 3 作對照）＋W2Exec2.step_begin／sync_after；請求經 step_begin 轉送。"""

    def __init__(self, auth_cls=M4.ModeAuthority4, auth_dt=DT, wheel_ok=None, filt=None):
        super().__init__(auth_dt=auth_dt, wheel_ok=wheel_ok, filt=filt)
        self.auth = auth_cls('r', run_id_num=RUNN, dt=auth_dt)
        self.intake = K3MSRC.EnvIntake(run_id_num=RUNN, stage_id=2, kinds=(0,))
        self.wx = WX2.W2Exec2(self.auth, self.intake, filt=filt or (lambda u: u))
        self.known_abort = {}          # step → why（本步 select 前已知）
        self.abort_after = {}          # step → why（本步 select 後才發現）
        self.pending_req = []

    def run(self, t_end, req_at=None, solver_until=None, delay=0.0, cand_delay=0.0, stop_mode=None, skip_post_at_commit=False,
            filt_after_req=False):
        rows = []
        while self.k * DT <= t_end + 1e-9:
            t = round(self.k * DT, 6)
            if self.k % 5 == 0 and (solver_until is None or t < solver_until):
                self.solver_cycle(t, delay, stop_mode, cand_delay)
            reqs = []
            if req_at is not None and self.t_req is None and t >= req_at:
                reqs, self.t_req = [dict(REQ)], t
            inbox = [(x[1], 1000.0 + t) for x in self.inbox if x[0] <= t + 1e-9]
            self.inbox = [x for x in self.inbox if x[0] > t + 1e-9]
            sel, relay, _nv = self.wx.step_begin(self.k, t, self.chain, known_abort=self.known_abort.get(self.k),
                                                 inbox=inbox, requests=reqs)
            if relay:
                self.prep = 1                                   # 求解端只在轉送後準備
            self.chain.step(t, DT, [0.0] * 6)
            u9 = list(self.chain.u_prev) if self.chain.u_prev is not None else [0.0] * 9
            ev = (sel['event'] or {}).get('kind')
            if not (skip_post_at_commit and ev == 'commit'):
                rows.append(self.wx.post_step(self.k, t, self.chain, True, u9, sel))
            self.wx.sync_after(self.k, t, known_abort=self.abort_after.get(self.k))
            st = self.auth.state_msg(self.k, t)
            if self.prep is not None and st['mode_seq'] >= self.prep:
                self.committed, self.prep = self.prep, None
            elif self.prep is not None and (st['aborted'] or (st['last_reject'] or {}).get('req_seq') == 1):
                self.prep = None
            self.k += 1
        return rows


def filt_after_commit(wref):
    def f(u):
        w = wref[0]
        if w.auth.commit is not None and w.k > w.auth.commit['step']:
            return None
        return u
    return f


class TD(unittest.TestCase):
    def test_d1_codex_repro_fresh_but_filtered(self):
        wref = [None]
        w = World2(filt=filt_after_commit(wref))
        wref[0] = w
        w.run(3.0, req_at=1.0)
        c = w.auth.commit
        self.assertIs(c['applied_confirmed'], True)
        self.assertEqual(w.auth.aborted['why'], 'post_commit_not_applied')
        self.assertLessEqual(w.auth.aborted['t'] - w.auth.last_applied_t, 0.25 + 0.0101)
        self.assertEqual(w.auth.mode_seq, 1)
        self.assertGreater(w.wx.intake_acts.get('buffered_current', 0), 0)   # 新鮮 mode 1 命令持續被收下
        # 對照：r3 權威同情境不中止（Codex 重現）
        wref3 = [None]
        w3 = World2(auth_cls=M3.ModeAuthority3, filt=filt_after_commit(wref3))
        wref3[0] = w3
        rows3 = w3.run(3.0, req_at=1.0)
        no_src = [r for r in rows3 if r['step'] > w3.auth.commit['step'] and r['src'] < 0]
        self.assertGreater(len(no_src), 80)
        self.assertIsNone(w3.auth.aborted)

    def test_d2_received_but_chain_rejects(self):
        st = {'armed': False}

        def wheel(vx, vy, wz):
            return (False, 'test_reject') if st['armed'] else (True, None)
        w = World2(wheel_ok=wheel)
        rows = w.run(1.2, req_at=1.0)
        self.assertIs(w.auth.commit['applied_confirmed'], True)
        st['armed'] = True
        w.run(2.0)
        self.assertEqual(w.auth.aborted['why'], 'post_commit_not_applied')
        self.assertEqual(w.auth.mode_seq, 1)
        self.assertTrue(rows)

    def test_d3_same_step_abort_request_candidate(self):
        w = World2()
        w.run(0.99)
        k = w.k
        w.known_abort[k] = 'k2_latch:test'
        w.inbox.append((1.0, env(1, 999, stamp=1.0)))          # 同步到達的候選（無 pending ⇒ 本應違規；但中止優先）
        w.run(1.0, req_at=1.0)
        self.assertEqual(w.auth.aborted['why'], 'external:k2_latch:test')
        self.assertEqual(w.auth.aborted['step'], k)
        self.assertEqual(w.wx.relayed, [])
        self.assertEqual(w.wx.requests[-1]['result'], 'reject')
        self.assertIsNone(w.auth.commit)
        self.assertIsNone(w.prep)
        self.assertIsNone(w.wx.rows[-1]['sel_src'])
        self.assertEqual(w.auth.violations, [])                 # 中止後候選判 stale，不再追加違規

    def test_d4_abort_beats_select_with_buffered_candidate(self):
        w = World2()
        w.run(0.99)
        w.auth.on_request(dict(REQ), w.k, 1.0)
        w.wx.on_env3(env(1, 999, stamp=1.0), 1.0)
        self.assertIsNotNone(w.auth.cand)
        w.known_abort[w.k] = 'corridor_violation'
        w.run(1.05)
        self.assertIsNone(w.auth.commit)
        self.assertEqual(w.auth.aborted['why'], 'external:corridor_violation')

    def test_d5_abort_found_after_commit_same_step(self):
        twin = World2()                                    # 同設定（決定性）先找提交步
        twin.run(1.3, req_at=1.0)
        kc = twin.auth.commit['step']
        w = World2()
        w.abort_after[kc] = 'finger_readback'              # 只在提交步、select 之後才發現
        w.run(1.3, req_at=1.0)
        c = w.auth.commit
        self.assertIsNotNone(c)
        ev = [e for e in w.auth.events if e['kind'] in ('commit', 'abort')]
        self.assertEqual([e['kind'] for e in ev], ['commit', 'abort'])
        self.assertEqual(ev[0]['step'], ev[1]['step'])
        self.assertEqual(w.auth.aborted['why'], 'external:finger_readback')
        self.assertEqual(w.auth.mode_seq, 1)

    def test_d6_normal_hold(self):
        w = World2()
        rows = w.run(4.5, req_at=1.0)
        c = w.auth.commit
        self.assertIs(c['applied_confirmed'], True)
        self.assertIsNone(w.auth.aborted)
        post = [r for r in rows if r['step'] >= c['step']]
        self.assertGreaterEqual(post[-1]['t'] - post[0]['t'], 3.0)
        sel = [r for r in rows if r['sel_src'] is not None]
        self.assertTrue(all(r['sel_stamp'] is not None and r['sel_t_recv'] is not None and r['sel_stamp'] <= r['sel_t_recv'] + 1e-9
                            for r in sel))
        chk = CK3.check(rows, w.auth.stats(), w.auth.applied_rows, {'committed': 1, 'first_src_by_mode': {'1': c['first_source_seq']}},
                        t_req=w.t_req, t_obs_passed=0.5, window=300)
        self.assertTrue(chk['ok'], chk)


class TD7(TC.TChain):
    """r3 真鏈情境以 r4 權威＋exec2 重跑（World 換成 World2）。"""

    def setUp(self):
        self._orig = TC.World
        TC.World = World2

    def tearDown(self):
        TC.World = self._orig


if __name__ == '__main__':
    unittest.main(verbosity=1)
