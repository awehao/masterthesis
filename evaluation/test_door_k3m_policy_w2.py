#!/usr/bin/env python3
"""door_k3m_policy_w2 純邏輯反例（父類 PAUSE 監看以替身代換：本檔只測 W2 結束條件；父類監看已有既有測試）。
  P1 正常：HELD＋觀察通過 ⇒ held_wait；1 s 後送一次請求；權威提交且 applied_confirmed ⇒ committed；保持 3 s ⇒ DONE（finish）。
     HELD 之前、held_wait 期間、committed 期間都不 finish（不送 /room/stop）。
  P2 求解端 DONE 且 held_end＝timeout ⇒ 中止 w2_held_timeout（不算完成）。
  P3 請求被拒（last_reject.req_seq＝1）⇒ w2_request_rejected；P4 applied_confirmed False ⇒ w2_commit_not_applied。
  P5 請求後 2 s 未提交 ⇒ w2_commit_timeout；P6 權威中止 ⇒ w2_mode_aborted。
  P7 committed 後權威狀態停更 > 1.5 s ⇒ w2_state_stale。P8 HELD 後監督器離開 HELD（FAIL 以外）⇒ w2_supervisor_left_held。
  P9 已中止 ⇒ take_w2_outbox 清除不發。P10 他趟 run_id 的權威狀態忽略。

    python3 evaluation/test_door_k3m_policy_w2.py
"""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3m_policy_p2 as PP2                       # noqa: E402
import door_k3m_policy_w2 as PW                        # noqa: E402


def _fake_parent_step(self, m, dt):                    # 替身：PAUSE 期間父類只回傳保持目標
    return {'target': 'T', 'grip': 0.0, 'request_handover': False, 'solver_start': False, 'stop_solver': False, 'finish': False}


def mk():
    p = object.__new__(PW.K3MW2Policy)
    p.phase, p.t_phase, p.abort, p.t_abort, p.events, p.done = 'PAUSE', 0.0, None, None, [], False
    p.c = types.SimpleNamespace(post_abort_observe_s=1.0)
    p.grip_cmd, p.pause_state, p.pause_status, p.pause_outcome = 0.0, 'active', None, None
    p.w2_run_id, p.w2_req_delay_s, p.w2_commit_timeout_s, p.w2_test_hold_s, p.w2_state_stale_s = 'r', 1.0, 2.0, 3.0, 1.5
    p.w2_state, p.w2_t, p.w2_outbox, p.w2_req, p.w2_last_state, p.w2_last_state_t, p.w2_n_state, p.w2_events = \
        'none', {}, [], None, None, None, 0, []
    return p


def status(state, **kw):
    o = {'state': state, 'observation_passed': state in ('HELD', 'DONE'), 'physical_stop_confirmed': state in ('HELD', 'DONE'),
         'transitions': {'OBS_PASSED': 10.0}}
    o.update(kw)
    return {'outcome': o}


def auth(commit=None, reject=None, aborted=None, ms=0, run='r'):
    return {'run_id': run, 'mode_seq': ms, 'commit': commit, 'commit_req_seq': 1 if commit else None, 'last_reject': reject,
            'aborted': aborted}


def drive(p, t0, t1, st_fn=None, status_fn=None):
    acts = []
    k = int(round(t0 / 0.05))
    while k * 0.05 <= t1 + 1e-9:
        t = round(k * 0.05, 4)
        if status_fn:
            p.pause_status = status_fn(t)
        if st_fn:
            s = st_fn(t)
            if s is not None:
                p.on_w2_state(s, t)
        acts.append((t, p.step(types.SimpleNamespace(t=t), 0.05)))
        k += 1
        if p.phase != 'PAUSE':
            break
    return acts


class TPW2(unittest.TestCase):
    def setUp(self):
        self._orig = PP2.K3MP2Policy.step
        PP2.K3MP2Policy.step = _fake_parent_step

    def tearDown(self):
        PP2.K3MP2Policy.step = self._orig

    def test_p1_normal(self):
        p = mk()
        C = {'step': 1, 't': 12.0, 'first_source_seq': 5, 'mode_seq': 1, 'applied_confirmed': True}
        acts = drive(p, 0.0, 30.0, status_fn=lambda t: status('HELD' if t >= 10.0 else 'CONFIRM'),
                     st_fn=lambda t: auth(C, ms=1) if t >= 11.2 else auth())
        self.assertEqual(p.phase, 'DONE')
        self.assertEqual(p.w2_t['held_wait'], 10.0)
        self.assertEqual(p.w2_t['requested'], 11.0)
        self.assertEqual(p.w2_t['committed'], 11.2)
        self.assertEqual(p.w2_t['done'], 14.2)
        fin = [t for t, a in acts if a['finish']]
        self.assertEqual(fin, [14.2])
        self.assertEqual(p.w2_req['mode_seq'], 1)
        self.assertEqual(len([e for e in p.w2_events if e.get('to') == 'requested']), 1)

    def test_p2_held_timeout(self):
        p = mk()
        drive(p, 0.0, 30.0, status_fn=lambda t: status('HELD') if t < 2.5 else status('DONE', held_end='timeout'),
              st_fn=lambda t: auth())
        self.assertEqual(p.abort, 'w2_held_timeout')

    def test_p3_p4_reject_and_not_applied(self):
        p = mk()
        drive(p, 0.0, 30.0, status_fn=lambda t: status('HELD'),
              st_fn=lambda t: auth(reject={'req_seq': 1, 'why': 'candidate_timeout'}) if t >= 1.5 else auth())
        self.assertEqual(p.abort, 'w2_request_rejected:candidate_timeout')
        p = mk()
        drive(p, 0.0, 30.0, status_fn=lambda t: status('HELD'),
              st_fn=lambda t: auth({'step': 1, 'mode_seq': 1, 'applied_confirmed': False}, ms=1) if t >= 1.5 else auth())
        self.assertEqual(p.abort, 'w2_commit_not_applied')

    def test_p5_p6_timeout_and_abort(self):
        p = mk()
        drive(p, 0.0, 30.0, status_fn=lambda t: status('HELD'), st_fn=lambda t: auth())
        self.assertEqual(p.abort, 'w2_commit_timeout')
        self.assertAlmostEqual(p.t_abort, 3.05, places=6)
        p = mk()
        drive(p, 0.0, 30.0, status_fn=lambda t: status('HELD'),
              st_fn=lambda t: auth(aborted={'why': 'post_commit_source_lost'}) if t >= 1.5 else auth())
        self.assertEqual(p.abort, 'w2_mode_aborted:post_commit_source_lost')

    def test_p7_state_stale_after_commit(self):
        p = mk()
        C = {'step': 1, 'mode_seq': 1, 'applied_confirmed': True}
        drive(p, 0.0, 30.0, status_fn=lambda t: status('HELD'), st_fn=lambda t: auth(C, ms=1) if 1.2 <= t < 2.0 else (None if t >= 2.0 else auth()))
        self.assertEqual(p.abort, 'w2_state_stale')

    def test_p8_left_held(self):
        p = mk()
        drive(p, 0.0, 30.0, status_fn=lambda t: status('HELD') if t < 0.5 else status('CONFIRM'), st_fn=lambda t: auth())
        self.assertEqual(p.abort, 'w2_supervisor_left_held:CONFIRM')

    def test_p9_p10(self):
        p = mk()
        p.w2_outbox.append({'x': 1})
        p.phase = 'ABORTED'
        self.assertEqual(p.take_w2_outbox(), [])
        p = mk()
        p.on_w2_state(auth(run='other'), 1.0)
        self.assertIsNone(p.w2_last_state)


if __name__ == '__main__':
    unittest.main(verbosity=1)
