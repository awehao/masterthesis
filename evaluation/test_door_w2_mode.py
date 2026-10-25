#!/usr/bin/env python3
"""door_w2_mode 純邏輯反例（Codex 20261025_201543；設計 W2_hold_handover_design_draft2）。
HELD 監督器：
  H1 觀察通過 ⇒ OBS_PASSED＋HELD，**持續 publish／zero**（不停發）；physical_stop_confirmed／observation_passed True。
  H2 HELD 中底盤漂移 2.6 mm/s ⇒ FAIL moved_in_held；H3 HELD 中套用非保持命令 ⇒ FAIL old_command_in_held；
  H4 HELD 超過 held_max_s ⇒ HELD_TIMEOUT_END 後 DONE、停止發布，held_end＝timeout（不算任務完成）；H5 HELD 中求解失敗 ⇒ FAIL。
模式權威：
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


def env(ms, src, u=None):
    return {'mode_seq': ms, 'source_seq': src, 'u': list(u if u is not None else U_HOLD)}


class TAuthority(unittest.TestCase):
    def mk(self):
        return M.ModeAuthority('r')

    def test_a1_normal_commit(self):
        a = self.mk()
        self.assertEqual(a.on_env(env(0, 10), 100, 1.0)[0], 'apply')
        self.assertEqual(a.on_request(dict(REQ), 101, 1.01)[0], 'pending')
        self.assertEqual(a.on_env(env(0, 11), 102, 1.02)[0], 'apply')                       # 提交前舊序號照常套用（無空窗）
        k, c = a.on_env(env(1, 12), 105, 1.05, U_HOLD)
        self.assertEqual(k, 'commit')
        self.assertEqual((c['step'], c['first_source_seq'], c['mode_seq'], c['profile']), (105, 12, 1, 'release_prep_test'))
        self.assertEqual((a.mode_seq, a.profile()), (1, 'release_prep_test'))
        self.assertEqual(a.on_env(env(1, 13), 110, 1.10)[0], 'apply')
        st = a.state_msg(110, 1.10)
        self.assertEqual((st['mode_seq'], st['commit']['first_source_seq'], st['commit_req_seq']), (1, 12, 1))

    def test_a2_stale_after_commit(self):
        a = self.mk()
        a.on_request(dict(REQ), 1, 0.0)
        a.on_env(env(1, 5), 2, 0.01, U_HOLD)
        self.assertEqual(a.on_env(env(0, 4), 3, 0.02), ('stale', 'old_mode_seq'))
        self.assertEqual(a.n_stale, 1)

    def test_a3_repeat_conflict(self):
        a = self.mk()
        self.assertEqual(a.on_request(dict(REQ), 1, 0.0)[0], 'pending')
        self.assertEqual(a.on_request(dict(REQ), 2, 0.01)[0], 'repeat')
        k, n = a.on_request(dict(REQ, req_seq=2), 3, 0.02)
        self.assertEqual((k, n['why']), ('reject', 'conflict_pending'))
        a.on_env(env(1, 5), 4, 0.03, U_HOLD)
        self.assertEqual(a.on_request(dict(REQ), 5, 0.04)[0], 'repeat')
        self.assertEqual(a.on_request(dict(REQ, req_seq=3), 6, 0.05)[1]['why'], 'conflict_after_commit')
        self.assertEqual(a.mode_seq, 1)

    def test_a4_candidate_invalid(self):
        cases = [([0.0] * 3 + [0.01] + [0.0] * 5, 'arm_not_zero'), ([0.006, 0.0, 0.0] + [0.0] * 6, 'base_not_hold'),
                 ([float('nan')] + [0.0] * 8, 'shape_or_nonfinite')]
        for u, why in cases:
            a = self.mk()
            a.on_request(dict(REQ), 1, 0.0)
            self.assertEqual(a.on_env(env(1, 5, u), 2, 0.01, U_HOLD), ('candidate_rejected', why), why)
            self.assertEqual(a.mode_seq, 0)
            self.assertEqual(a.last_reject['why'], f'candidate_invalid:{why}')
            self.assertEqual(a.on_env(env(0, 6), 3, 0.02)[0], 'apply')
        a = self.mk()
        a.on_request(dict(REQ), 1, 0.0)
        self.assertEqual(a.on_env(env(1, 5, [0.0049, 0.0, 0.0] + [0.0] * 6), 2, 0.01, [-0.03, 0.0, 0.0] + [0.0] * 6)[1], 'accel_box')

    def test_a5_candidate_timeout(self):
        a = self.mk()
        a.on_request(dict(REQ), 1, 0.0)
        self.assertIsNone(a.on_step(50, 0.5))
        self.assertEqual(a.on_step(110, 1.1), {'kind': 'candidate_timeout'})
        self.assertEqual((a.mode_seq, a.pending), (0, None))
        self.assertEqual(a.on_env(env(0, 7), 111, 1.11)[0], 'apply')

    def test_a6_post_commit_loss_aborts_no_fallback(self):
        a = self.mk()
        a.on_request(dict(REQ), 1, 0.0)
        a.on_env(env(1, 5), 2, 0.01, U_HOLD)
        self.assertIsNone(a.on_step(20, 0.2))
        self.assertEqual(a.on_step(30, 0.3), {'kind': 'abort', 'why': 'post_commit_source_lost'})
        self.assertEqual(a.mode_seq, 1)                                                        # 不回切
        self.assertEqual(a.on_env(env(0, 6), 31, 0.31), ('stale', 'after_abort'))
        self.assertEqual(a.on_env(env(1, 7), 32, 0.32), ('stale', 'after_abort'))

    def test_a7_abort_priority(self):
        a = self.mk()
        a.on_request(dict(REQ), 1, 0.0)
        a.abort('stop_topic', 2, 0.01)
        self.assertEqual(a.last_reject['why'], 'after_abort')
        self.assertEqual(a.on_env(env(1, 5), 3, 0.02), ('stale', 'after_abort'))
        self.assertEqual(a.mode_seq, 0)
        self.assertEqual(a.on_request(dict(REQ, req_seq=9), 4, 0.03)[1]['why'], 'after_abort')

    def test_a8_violations_and_nacks(self):
        a = self.mk()
        self.assertEqual(a.on_env(env(1, 5), 1, 0.0), ('violation', 'mode_seq_ahead'))
        self.assertEqual(a.on_env({'source_seq': 5, 'u': U_HOLD}, 1, 0.0), ('violation', 'mode_seq_missing'))
        self.assertEqual(a.on_request(dict(REQ, run_id='x'), 1, 0.0)[1]['why'], 'wrong_run')
        self.assertEqual(a.on_request(dict(REQ, mode_seq=2), 1, 0.0)[1]['why'], 'bad_transition')
        self.assertEqual(a.on_request(dict(REQ, to_mode='GRIP_OPEN'), 1, 0.0)[1]['why'], 'bad_transition')
        self.assertEqual(a.on_request({'run_id': 'r'}, 1, 0.0)[1]['why'], 'malformed')


class TSource(unittest.TestCase):
    def test_s1_only_in_held(self):
        s = M.SolverModeSource('r')
        self.assertFalse(s.on_request(dict(REQ), held=False, t=0.0))
        self.assertEqual(s.tag(), 0)
        self.assertTrue(s.on_request(dict(REQ), held=True, t=0.1))
        self.assertEqual(s.tag(), 1)
        self.assertFalse(s.on_request(dict(REQ), held=True, t=0.2))                          # 重送不重複準備

    def test_s2_ack_lost_state_authoritative(self):
        s, a = M.SolverModeSource('r'), M.ModeAuthority('r')
        a.on_request(dict(REQ), 1, 0.0)
        s.on_request(dict(REQ), True, 0.0)
        a.on_env(env(s.tag(), 5), 2, 0.01, U_HOLD)
        s.on_state(a.state_msg(3, 0.02), 0.02)                                               # 只靠狀態流
        self.assertEqual((s.committed, s.prep, s.tag()), (1, None, 1))

    def test_s3_revert_on_reject(self):
        s, a = M.SolverModeSource('r'), M.ModeAuthority('r')
        a.on_request(dict(REQ), 1, 0.0)
        s.on_request(dict(REQ), True, 0.0)
        a.on_env(env(1, 5, [0.0] * 3 + [0.1] + [0.0] * 5), 2, 0.01, U_HOLD)
        s.on_state(a.state_msg(3, 0.02), 0.02)
        self.assertEqual((s.committed, s.prep, s.tag()), (0, None, 0))

    def test_s4_revert_on_abort(self):
        s, a = M.SolverModeSource('r'), M.ModeAuthority('r')
        a.on_request(dict(REQ), 1, 0.0)
        s.on_request(dict(REQ), True, 0.0)
        a.abort('stop', 2, 0.01)
        s.on_state(a.state_msg(3, 0.02), 0.02)
        self.assertEqual(s.tag(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=1)
