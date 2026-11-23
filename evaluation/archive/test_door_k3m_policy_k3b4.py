#!/usr/bin/env python3
"""door_k3m_policy_k3b4 純邏輯反例（父類 PAUSE 監看與幾何以替身代換，沿用 test_door_k3m_policy_k3b3 的替身；**合成時間軸**）。
時間軸（任務 20 Hz）：HELD 10.0 → 請求 1 → 提交 11.2 → 執行端 SETTLING 12.2、UNLOADED 12.9；求解端 unload → unloaded 12.95
→ 兩端皆完成後請求 2 → 提交（t_open）→ 執行端 B_CONFIRMED t_open＋0.8；求解端 release_settled t_open＋1.0 → 請求 3 → 退離 → COMPLETE。
  P1 正常：請求序 1、2、3；mode 2 請求晚於兩端完成；不送 5 N 請求（eff_outbox 空、eff_req None）；C_CLEAR 後 eff_verify 讀回 5；DONE。
  P2 執行端 unload FAILED ⇒ k3b_unload:unload_failed:contact_lost；P3 求解端一直未 unloaded ⇒ k3b_unload:unload_deadline。
  P4 mode 3 等求解端 release_settled：B_CONFIRMED 早於 release_settled（＋1.3 s）時，請求 3 ≥ release_settled；
     ＋1.6 s 時近區退出期限（t_conf＋2.0 s）逾時 ⇒ k3b_release:near_exit_timeout（時序預算衝突，記入規格待裁定）。
  P5 讀回核對：讀回 20 ⇒ k3b_effort_verify:readback_not_5；讀回時刻過舊 ⇒ k3b_effort_verify:readback_stale_or_future。
  P6 求解端 FAIL（釋放中）⇒ w2_supervisor_fail；P7 unload=False ⇒ 與 r3 相同（送 5 N 請求、DONE）。

    python3 evaluation/test_door_k3m_policy_k3b4.py
"""
import math
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3b_release4 as KR                         # noqa: E402
import door_k3m_policy_k3b4 as PK                      # noqa: E402
import door_k3m_policy_p2 as PP2                       # noqa: E402
import test_door_k3m_policy_k3b3 as T3                 # noqa: E402


def mk(unload=True, geo=None):
    p = object.__new__(PK.K3MReleasePolicy4)
    q = T3.mk(geo)
    p.__dict__.update(q.__dict__)
    p.pair = T3.Pair(p)
    p.unload, p.effort_age_max_s, p.eff_verify_timeout_s = unload, 1.5, 1.0
    p.rel.update(unload_gate=None, eff_verify=None, eff_verify_last_why=None)
    p.rel['t']['eff_verify_from'] = None
    return p


class Sim4(T3.Sim):
    def __init__(self, ex_fail=None, solver_unloaded=True, settled_delay=1.0, rb=(5.0, 5.0), rb_age=0.3, solver_fail_at=None, **kw):
        super().__init__(**kw)
        self.ex_fail, self.solver_unloaded, self.settled_delay = ex_fail, solver_unloaded, settled_delay
        self.rb, self.rb_age, self.solver_fail_at = rb, rb_age, solver_fail_at

    def t_c(self, ms):
        return next((c['t'] for c in self.commits if c['mode_seq'] == ms), None)

    def state(self, t):
        st = super().state(t)
        cs = [c for c in self.commits if c['t'] <= t + 1e-9]                    # 只顯示已發生的提交（真權威不會回報未來提交）
        last = cs[-1] if cs else None
        st.update(commits=cs, mode_seq=max([c['mode_seq'] for c in cs] or [0]), commit=last,
                  commit_req_seq=None if last is None else last['mode_seq'], t=t)
        t1 = self.t_c(1)
        u = {'state': 'IDLE'}
        if t1 is not None:
            if self.ex_fail is not None and t >= t1 + 0.5:
                u = {'state': 'FAILED', 'why': self.ex_fail}
            elif self.t_c(2) is not None:
                u = {'state': 'HANDED_TO_RELEASE'}
            else:
                u = {'state': 'UNLOADED' if t >= t1 + 1.7 else ('SETTLING' if t >= t1 + 1.0 else 'RAMPING')}
        st['unload'] = u
        st['finger_effort'] = {'readback': list(self.rb), 't': t - self.rb_age, 'ok': True}
        return st

    def solver(self, t, held):
        t1, t2 = self.t_c(1), self.t_c(2)
        sub = 'held'
        if t1 is not None and t >= t1 + 0.02:
            sub = 'unloaded' if (self.solver_unloaded and t >= t1 + 1.75) else 'unload'
        if t2 is not None and t >= t2 + 0.02:
            sub = 'release_settled' if t >= t2 + self.settled_delay else 'release_open'
        state = 'HELD' if held else 'CONFIRM'
        if self.solver_fail_at is not None and t >= self.solver_fail_at:
            state = 'FAIL'
        return {'sim_t': t - 0.01, 'outcome': {'state': state, 'sub': sub, 'fail_why': 'x' if state == 'FAIL' else None,
                                               'observation_passed': held, 'physical_stop_confirmed': held, 'transitions': {'OBS_PASSED': 10.0}}}


def run(sim, unload=True, t_end=30.0):
    PP2.K3MP2Policy.step, orig = T3._fake_parent, PP2.K3MP2Policy.step
    try:
        p = mk(unload)
        tgt_prev = None
        sub_log = []
        for i in range(int(t_end / 0.05) + 1):
            t = round(i * 0.05, 4)
            held = t >= 10.0
            p.pause_status = sim.solver(t, held)
            sub_log.append((t, p.pause_status['outcome']['sub']))
            p.on_w2_state(sim.state(t), t)
            for r in p.take_w2_outbox():
                sim.on_req(r, t)
            for r in p.take_eff_outbox():
                sim.eff_req_t = t
            if sim.eff_req_t is not None and t >= sim.eff_req_t + 0.1:
                p.on_effort_ack({'run_id': 'r', 'result': 'apply', 'done': {'req_seq': 1, 'readback_ok': True, 'switch_step': 9999}}, t)
            T = np.asarray(tgt_prev, float) if (tgt_prev is not None and p.phase == 'RELEASE') else T3.H0 @ T3.E0
            m = types.SimpleNamespace(t=t, state_step=int(round(t * 100)), wall=t, H=T3.H0, T_tcp=T, q_door=T3.Q0, qdot_door=0.0,
                                      state_age=0.01, contacts_age=0.01, contacts_invalid=0, unexpected_max=0.0, wall_max=0.0, wrist_max=3.0,
                                      solver_failed=None, solver_up=True, solver_status_age=0.05, executor_violation=None, base_pose=(0, 0, 0),
                                      qf=np.zeros(9), q_finger=0.004, finger_q=(0.0089, 0.0089), base_drift_m=0.00002, base_drift_yaw=0.0,
                                      settle_sig={'valid': True, 'rate': 0.0})
            act = p.step(m, 0.05)
            if act.get('target') is not None and not isinstance(act.get('target'), str):
                tgt_prev = act['target']
            if p.phase in ('DONE', 'ABORTED'):
                break
        return p, sub_log
    finally:
        PP2.K3MP2Policy.step = orig


class TP(unittest.TestCase):
    def test_p1_normal(self):
        sim = Sim4()
        p, _ = run(sim)
        self.assertEqual(p.phase, 'DONE', (p.abort, p.k3b.why, p.rel.get('unload_gate')))
        self.assertEqual([ms for _, ms in sim.reqs], [1, 2, 3])
        t1 = sim.t_c(1)
        t_req2 = next(t for t, ms in sim.reqs if ms == 2)
        self.assertGreaterEqual(t_req2, t1 + 1.75)
        self.assertIsNone(p.rel['eff_req'])
        self.assertIsNone(sim.eff_req_t)
        self.assertEqual(p.rel['eff_verify']['readback'], [5.0, 5.0])
        self.assertEqual(p.k3b.state, 'COMPLETE')
        self.assertEqual(p.k3b_report()['version'], PK.VERSION)

    def test_p2_p3(self):
        p, _ = run(Sim4(ex_fail='contact_lost'))
        self.assertEqual(p.abort, 'k3b_unload:unload_failed:contact_lost')
        p, _ = run(Sim4(solver_unloaded=False))
        self.assertEqual(p.abort, 'k3b_unload:unload_deadline')

    def test_p4_mode3_waits_settled(self):
        sim = Sim4(settled_delay=1.3)                                         # B_CONFIRMED t_open＋0.8 早於 release_settled t_open＋1.3
        p, _ = run(sim)
        self.assertEqual(p.phase, 'DONE', p.abort)
        t2 = sim.t_c(2)
        t_req3 = next(t for t, ms in sim.reqs if ms == 3)
        self.assertGreaterEqual(t_req3, t2 + 1.3 - 1e-9)
        self.assertEqual(p.rel['solver_sub_at_req3'], 'release_settled')
        p, _ = run(Sim4(settled_delay=1.6))                                    # 時序預算衝突（見規格）：近區退出期限 t_conf＋2.0 s 逾時
        self.assertEqual(p.abort, 'k3b_release:near_exit_timeout')

    def test_p5_effort_verify(self):
        p, _ = run(Sim4(rb=(20.0, 20.0)))
        self.assertEqual(p.abort, 'k3b_effort_verify:readback_not_5')
        p, _ = run(Sim4(rb_age=2.0))
        self.assertEqual(p.abort, 'k3b_effort_verify:readback_stale_or_future')

    def test_p6_p7(self):
        p, _ = run(Sim4(solver_fail_at=15.5))
        self.assertTrue(str(p.abort).startswith('w2_supervisor_fail'), p.abort)
        sim = Sim4()
        p, _ = run(sim, unload=False)
        self.assertEqual(p.phase, 'DONE', p.abort)
        self.assertIsNotNone(p.rel['eff_req'])
        self.assertIsNotNone(sim.eff_req_t)

if __name__ == '__main__':
    unittest.main(verbosity=1)
