#!/usr/bin/env python3
"""door_w2_mode 純邏輯反例（Codex 20261025_201543；設計 W2_hold_handover_design_draft2）。
HELD 監督器：
  H1 觀察通過 ⇒ OBS_PASSED＋HELD，**持續 publish／zero**（不停發）；physical_stop_confirmed／observation_passed True。
  H2 HELD 中底盤漂移 2.6 mm/s ⇒ FAIL moved_in_held；H3 HELD 中套用非保持命令 ⇒ FAIL old_command_in_held；
  H4 HELD 超過 held_max_s ⇒ HELD_TIMEOUT_END 後 DONE、停止發布，held_end＝timeout（不算任務完成）；H5 HELD 中求解失敗 ⇒ FAIL。
r2（Codex 20261025_211640）介面：on_env 只緩衝；select 在物理步選擇／提交；confirm_applied 核對實際套用。r2 反例：
  A2' 候選延遲超過命令逾時（未超過候選期限）⇒ 舊模式持續套用、無空窗，候選到達才提交；A3' 同一步連到多筆候選 ⇒ 提交＝實際選定那筆；
  A4' 提交步實際套用來源不符（進鏈拒絕／限幅）⇒ 中止 commit_not_applied、不回切；A5' 過期候選不提交、候選逾時；
  A6' 缺／非法 applied u_prev ⇒ 拒絕候選；A8' 入口身分（run、stage、kind、derived、source_seq）、未來時間、重送冪等不刷新、同序異值、跨 mode 同序、
  亂序、過期；H6 HELD 套用回報：src NaN、維度錯、步號非 int、時間 NaN、非有限 ⇒ applied_invalid（不拋例外）；H7 同步重送不刷新、同步異值 ⇒
  applied_conflict、倒退 ⇒ applied_order；S1' 一般流維持已提交序號＋候選旁路。
模式權威（r1 項目沿用）：
  A1 正常：請求 ⇒ pending；候選首筆通過驗證 ⇒ 同一處提交（mode_seq、profile、first_source_seq）；之後同序號 apply。
  A2 舊模式延遲命令：提交後 mode_seq 0 ⇒ stale、不套用、計數。
  A3 重送同一請求 ⇒ repeat（pending 中、提交後皆冪等）；不同請求 ⇒ conflict_pending／conflict_after_commit。
  A4 切換前候選不合格（手臂非零／底盤超保持界／加速度框／非有限）⇒ 拒絕、保持 mode_seq 0。
  A5 切換前候選逾時 ⇒ candidate_timeout、保持。
  A6 切換後新模式斷流 ⇒ abort post_commit_source_lost，**不回切**（之後命令一律 stale）。
  A7 中止優先：pending 中中止 ⇒ pending 拒絕、之後候選 stale、新請求 after_abort。
  A8 無 pending 的超前序號 ⇒ violation；缺 mode_seq ⇒ violation；錯 run／錯轉換 ⇒ nack。
求解端模式來源：
  S1 只在 HELD 接受請求並以新序號標記；S2 ack 遺失仍由權威狀態得知提交；S3 被拒 ⇒ 撤回原序號；S4 中止 ⇒ 撤回。

    python3 evaluation/test_door_w2_mode.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3m_pause_sup_p4 as P4                     # noqa: E402
import door_w2_mode as M                               # noqa: E402

HOLD = (0.0022, -0.0013, 0.0018)
REQ = {'run_id': 'r', 'req_seq': 1, 'from_mode': M.MODE_HELD, 'to_mode': M.MODE_RELEASE_TEST, 'mode_seq': 1}
U_HOLD = list(HOLD) + [0.0] * 6


def run_held(sup, t_end=40.0, drift_after_held=0.0, bad_cmd_in_held_at=None, solve_fail_at=None):
    seq, x = 100, 0.6
    t_held = None
    for k in range(int(t_end / 0.01)):
        t = round(k * 0.01, 4)
        if sup.state == 'HELD' and t_held is None:
            t_held = t
        if t_held is not None:
            x += drift_after_held * 0.01
        if k % 2 == 0:
            sup.feed_state(k, t, x, -0.53, 1.57, 0.14, [0.0] * 6, wall=t)
        if k % 5 == 0 and k > 0:
            ok = not (solve_fail_at is not None and t_held is not None and t >= t_held + solve_fail_at)
            a = sup.cycle(t, ok, P4.hold_cmd_ok(HOLD, [0.0] * 6), wall=t)
            if a['publish']:
                seq += 1
                sup.note_published(seq, a['zero'])
        if sup.state in ('CONFIRM', 'STOPPED', 'HELD'):
            bad = bad_cmd_in_held_at is not None and t_held is not None and t >= t_held + bad_cmd_in_held_at
            sup.applied(k, t, [0.01, 0.0, 0.0] if bad else list(HOLD), [0.0] * 6, seq)
        if sup.state in ('DONE', 'FAIL'):
            break
    return sup.outcome(t_end=t), t_held


class THeld(unittest.TestCase):
    def test_h1_held_keeps_publishing(self):
        sup = M.HeldSupervisorP4b(held_max_s=60.0)
        o, th = run_held(sup, t_end=30.0)
        self.assertEqual(o['state'], 'HELD', o)
        self.assertIs(o['observation_passed'], True)
        self.assertIs(o['physical_stop_confirmed'], True)
        self.assertIsNotNone(o['transitions']['OBS_PASSED'])
        a = sup.cycle(30.01, True, True, wall=30.01)
        self.assertEqual(a, {'publish': True, 'zero': True})
        self.assertGreater(o['held_s'], 5.0)

    def test_h2_moved_in_held(self):
        o, _ = run_held(M.HeldSupervisorP4b(), t_end=40.0, drift_after_held=0.0026)
        self.assertEqual((o['state'], o['fail_why']), ('FAIL', 'moved_in_held'), o)

    def test_h3_bad_cmd_in_held(self):
        o, _ = run_held(M.HeldSupervisorP4b(), t_end=40.0, bad_cmd_in_held_at=1.0)
        self.assertEqual(o['fail_why'], 'old_command_in_held', o)

    def test_h4_timeout_end(self):
        sup = M.HeldSupervisorP4b(held_max_s=3.0)
        o, _ = run_held(sup, t_end=40.0)
        self.assertEqual(o['state'], 'DONE')
        self.assertEqual(o['held_end'], 'timeout')
        self.assertIsNotNone(o['transitions']['HELD_TIMEOUT_END'])
        self.assertEqual(sup.cycle(41.0, True, True, wall=41.0)['publish'], False)

    def test_h5_solve_fail_in_held(self):
        o, _ = run_held(M.HeldSupervisorP4b(), t_end=40.0, solve_fail_at=1.0)
        self.assertEqual(o['fail_why'], 'solve_failed_in_held', o)


RUNN = 77.0


def env(ms, src, u=None, stamp=0.0, stage=2, kind=0, derived=True, run=RUNN):
    return {'mode_seq': ms, 'source_seq': src, 'u': list(u if u is not None else U_HOLD), 'stamp_sim_t': stamp,
            'stage_id': stage, 'kind': kind, 'derived': derived, 'run_id': run}


def auth():
    return M.ModeAuthority('r', run_id_num=RUNN)


class TAuthority(unittest.TestCase):
    def test_a1_normal_commit_at_step(self):
        a = auth()
        self.assertEqual(a.on_env(env(0, 10, stamp=1.0), 1.0), ('buffered_current', None))
        self.assertEqual(a.select(100, 1.0, U_HOLD)['apply']['source_seq'], 10)
        self.assertEqual(a.on_request(dict(REQ), 101, 1.01)[0], 'pending')
        self.assertEqual(a.on_env(env(1, 12, stamp=1.02), 1.02), ('buffered_candidate', None))     # 只緩衝，不提交
        self.assertIsNone(a.commit)
        r = a.select(103, 1.03, U_HOLD)
        self.assertEqual((r['event'], r['apply']['source_seq'], r['mode_seq'], r['profile']),
                         ({'kind': 'commit'}, 12, 1, 'release_prep_test'))
        self.assertEqual((a.commit['step'], a.commit['first_source_seq']), (103, 12))
        self.assertIs(a.confirm_applied(103, 12), True)
        self.assertIs(a.commit['applied_confirmed'], True)

    def test_a2_old_mode_supplied_while_candidate_delayed(self):
        """候選延遲超過命令逾時、但未超過候選期限：舊模式持續套用、不空窗；候選到達後才提交。"""
        a = auth()
        a.on_request(dict(REQ), 1, 0.0)
        for k in range(1, 60):
            t = k * 0.01
            if k % 5 == 0:
                a.on_env(env(0, 100 + k, stamp=t), t)                                       # 舊模式持續供應
            r = a.select(k, t, U_HOLD)
            if k >= 5:
                self.assertIsNotNone(r['apply'], k)
                self.assertEqual(r['apply']['mode_seq'], 0)
        self.assertIsNone(a.commit)
        a.on_env(env(1, 300, stamp=0.6), 0.6)
        r = a.select(60, 0.6, U_HOLD)
        self.assertEqual((r['event'], a.commit['first_source_seq']), ({'kind': 'commit'}, 300))

    def test_a3_same_step_multiple_candidates(self):
        a = auth()
        a.on_request(dict(REQ), 1, 0.0)
        a.on_env(env(1, 20, stamp=0.010), 0.010)
        a.on_env(env(1, 21, stamp=0.011), 0.011)                                            # 同一步內連到兩筆
        r = a.select(2, 0.012, U_HOLD)
        self.assertEqual((r['apply']['source_seq'], a.commit['first_source_seq']), (21, 21))

    def test_a4_chain_reject_after_commit(self):
        a = auth()
        a.on_request(dict(REQ), 1, 0.0)
        a.on_env(env(1, 20, stamp=0.01), 0.01)
        a.select(2, 0.02, U_HOLD)
        self.assertIs(a.confirm_applied(2, 19), False)                                       # 進鏈拒絕／限幅後實際來源不同
        self.assertEqual(a.aborted['why'], 'commit_not_applied')
        self.assertEqual(a.mode_seq, 1)                                                      # 不回切

    def test_a5_stale_candidate_and_timeout(self):
        a = auth()
        a.on_request(dict(REQ), 1, 0.0)
        a.on_env(env(1, 20, stamp=0.0), 0.0)
        r = a.select(30, 0.30, U_HOLD)                                                      # 候選超過命令新鮮期 ⇒ 不提交
        self.assertIsNone(a.commit)
        self.assertIsNone(r['event'])
        r = a.select(110, 1.10, U_HOLD)
        self.assertEqual(r['event'], {'kind': 'candidate_timeout'})
        self.assertEqual(a.mode_seq, 0)

    def test_a6_missing_u_prev_rejects(self):
        for up in (None, [float('nan')] * 9, [0.0] * 8):
            a = auth()
            a.on_request(dict(REQ), 1, 0.0)
            a.on_env(env(1, 20, stamp=0.01), 0.01)
            r = a.select(2, 0.02, up)
            self.assertEqual(r['event'], {'kind': 'candidate_rejected', 'why': 'u_prev_missing_or_invalid'}, up)
            self.assertIsNone(a.commit)

    def test_a7_candidate_content(self):
        for u, why in (([0.0] * 3 + [0.01] + [0.0] * 5, 'arm_not_zero'), ([0.006, 0.0, 0.0] + [0.0] * 6, 'base_not_hold')):
            a = auth()
            a.on_request(dict(REQ), 1, 0.0)
            a.on_env(env(1, 20, u, stamp=0.01), 0.01)
            self.assertEqual(a.select(2, 0.02, U_HOLD)['event']['why'], why)
        a = auth()
        a.on_request(dict(REQ), 1, 0.0)
        a.on_env(env(1, 20, [0.0049, 0.0, 0.0] + [0.0] * 6, stamp=0.01), 0.01)
        self.assertEqual(a.select(2, 0.02, [-0.03, 0.0, 0.0] + [0.0] * 6)['event']['why'], 'accel_box')

    def test_a8_intake_identity_and_seq(self):
        a = auth()
        cases = [(env(0, 1, run=1.0), 'wrong_run'), (env(0, 1, stage=0), 'wrong_stage'), (env(0, 1, kind=1), 'wrong_kind_or_not_derived'),
                 (env(0, 1, derived=False), 'wrong_kind_or_not_derived'), (env(0, -1), 'bad_source_seq'), (env(0, 1.5), 'bad_source_seq'),
                 (dict(env(0, 1), mode_seq=None), 'mode_seq_missing'), (env(0, 1, stamp=0.5), 'future_stamp'),
                 (env(0, 1, u=[float('nan')] * 9), 'bad_u'), (env(2, 1), 'mode_seq_ahead'), (env(1, 1), 'candidate_without_request')]
        for e, why in cases:
            self.assertEqual(a.on_env(e, 0.0), ('violation', why), why)
        self.assertEqual(a.on_env(env(0, 10, stamp=0.0), 0.0)[0], 'buffered_current')
        self.assertEqual(a.on_env(env(0, 10, stamp=0.0), 0.05), ('duplicate', None))                # 同序同值：冪等不刷新
        self.assertEqual(a.cur[1], 0.0)
        self.assertEqual(a.on_env(env(0, 10, u=[0.001] + [0.0] * 8, stamp=0.0), 0.05), ('violation', 'same_seq_conflict'))
        self.assertEqual(a.on_env(env(0, 9, stamp=0.05), 0.05), ('stale', 'out_of_order'))
        self.assertEqual(a.on_env(env(0, 11, stamp=0.0), 0.5), ('stale', 'old_stamp'))
        a.on_request(dict(REQ), 1, 0.06)
        self.assertEqual(a.on_env(env(1, 10, stamp=0.06), 0.06), ('violation', 'seq_mode_conflict'))

    def test_a9_stale_after_commit_and_post_commit_loss(self):
        a = auth()
        a.on_env(env(0, 5, stamp=0.0), 0.0)
        a.on_request(dict(REQ), 1, 0.0)
        a.on_env(env(1, 6, stamp=0.0), 0.0)
        a.select(1, 0.01, U_HOLD)
        self.assertEqual(a.on_env(env(0, 7, stamp=0.01), 0.01), ('stale', 'old_mode_seq'))
        self.assertIsNotNone(a.select(10, 0.10, U_HOLD)['apply'])
        r = a.select(40, 0.40, U_HOLD)
        self.assertEqual(r['event'], {'kind': 'abort'})
        self.assertEqual((a.aborted['why'], a.mode_seq), ('post_commit_source_lost', 1))
        self.assertEqual(a.on_env(env(1, 8, stamp=0.40), 0.40), ('stale', 'after_abort'))

    def test_a10_requests(self):
        a = auth()
        self.assertEqual(a.on_request(dict(REQ), 1, 0.0)[0], 'pending')
        self.assertEqual(a.on_request(dict(REQ), 2, 0.01)[0], 'repeat')
        self.assertEqual(a.on_request(dict(REQ, req_seq=2), 3, 0.02)[1]['why'], 'conflict_pending')
        a.abort('stop', 4, 0.03)
        self.assertEqual(a.last_reject['why'], 'after_abort')
        self.assertEqual(a.on_request(dict(REQ, req_seq=9), 5, 0.04)[1]['why'], 'after_abort')
        b = auth()
        for m, why in ((dict(REQ, run_id='x'), 'wrong_run'), (dict(REQ, mode_seq=2), 'bad_transition'),
                       (dict(REQ, to_mode='GRIP_OPEN'), 'bad_transition'), ({'run_id': 'r'}, 'malformed')):
            self.assertEqual(b.on_request(m, 1, 0.0)[1]['why'], why)


class THeldApplied(unittest.TestCase):
    def held(self):
        sup = M.HeldSupervisorP4b(held_max_s=60.0)
        o, _ = run_held(sup, t_end=30.0)
        self.assertEqual(sup.state, 'HELD')
        return sup

    def test_h6_invalid_types_fail_no_exception(self):
        for args in ((3001, 30.01, list(HOLD), [0.0] * 6, float('nan')), (3001, 30.01, list(HOLD)[:2], [0.0] * 6, 999),
                     (3001.0, 30.01, list(HOLD), [0.0] * 6, 999), (3001, float('nan'), list(HOLD), [0.0] * 6, 999),
                     (3001, 30.01, [float('inf'), 0, 0], [0.0] * 6, 999), (3001, 30.01, None, [0.0] * 6, 999)):
            sup = self.held()
            sup.applied(*args)
            self.assertEqual((sup.state, sup.seq.why), ('FAIL', 'applied_invalid'), args)

    def test_h7_order_and_resend(self):
        sup = self.held()
        sup.applied(3001, 30.01, list(HOLD), [0.0] * 6, 999)
        sup.applied(3001, 30.01, list(HOLD), [0.0] * 6, 999)                                 # 同步同值重送：不刷新
        self.assertEqual(sup.state, 'HELD')
        sup.applied(3001, 30.01, list(HOLD), [0.0] * 6, 998)
        self.assertEqual(sup.seq.why, 'applied_conflict')
        sup = self.held()
        sup.applied(3002, 30.02, list(HOLD), [0.0] * 6, 999)
        sup.applied(3001, 30.03, list(HOLD), [0.0] * 6, 999)
        self.assertEqual(sup.seq.why, 'applied_order')


class TSource(unittest.TestCase):
    def test_s1_only_in_held_and_dual_stream(self):
        s = M.SolverModeSource('r')
        self.assertFalse(s.on_request(dict(REQ), held=False, t=0.0))
        self.assertTrue(s.on_request(dict(REQ), held=True, t=0.1))
        self.assertEqual((s.tag_normal(), s.tag_candidate()), (0, 1))                         # 舊模式持續供應＋候選旁路
        self.assertFalse(s.on_request(dict(REQ), held=True, t=0.2))

    def test_s2_ack_lost_state_authoritative(self):
        s, a = M.SolverModeSource('r'), auth()
        a.on_request(dict(REQ), 1, 0.0)
        s.on_request(dict(REQ), True, 0.0)
        a.on_env(env(s.tag_candidate(), 5, stamp=0.0), 0.0)
        a.select(1, 0.01, U_HOLD)
        s.on_state(a.state_msg(2, 0.02), 0.02)
        self.assertEqual((s.committed, s.tag_normal(), s.tag_candidate()), (1, 1, None))

    def test_s3_revert_on_reject_and_abort(self):
        s, a = M.SolverModeSource('r'), auth()
        a.on_request(dict(REQ), 1, 0.0)
        s.on_request(dict(REQ), True, 0.0)
        a.on_env(env(1, 5, [0.0] * 3 + [0.1] + [0.0] * 5, stamp=0.0), 0.0)
        a.select(1, 0.01, U_HOLD)
        s.on_state(a.state_msg(2, 0.02), 0.02)
        self.assertEqual((s.committed, s.tag_candidate()), (0, None))
        s, a = M.SolverModeSource('r'), auth()
        a.on_request(dict(REQ), 1, 0.0)
        s.on_request(dict(REQ), True, 0.0)
        a.abort('stop', 2, 0.01)
        s.on_state(a.state_msg(3, 0.02), 0.02)
        self.assertIsNone(s.tag_candidate())


if __name__ == '__main__':
    unittest.main(verbosity=1)
