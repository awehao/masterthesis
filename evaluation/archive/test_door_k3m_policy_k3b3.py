#!/usr/bin/env python3
"""door_k3m_policy_k3b3 純邏輯反例（r3 策略＋core r4；test_door_k3m_policy_k3b2.py 的版本化複本）。以下為 r2 說明：
door_k3m_policy_k3b2 純邏輯反例（test_door_k3m_policy_k3b.py 的版本化複本；r2 策略＋core r3）。以下為 r1 說明：
door_k3m_policy_k3b 純邏輯反例（父類 PAUSE 監看與幾何以替身代換；本檔只驗 K3-B 流程與失敗出口）。
合成時間軸（任務 20 Hz）：
- HELD 10.0 → 請求 1 → 權威提交 11.2 → 測試保持 3 s → 14.2 起請求 2 → 權威提交 14.4（t_open）；
- 執行端 B_CONFIRMED 15.2 → 請求 3 → 提交 15.3；
- 之後工具完美追蹤目標（下一輪實測＝上一輪目標）。
  Q1 正常：
     - 請求序 1、2、3；RELEASE 基準取 t_open 以前最後一筆同步；mode 3 提交前目標固定 T(0)，之後 s_cmd 0.02 m/s 單調到 0.10；
     - C_CLEAR 後送 5 N 請求一次（帶 C_CLEAR 證據）；ack 讀回 True；COMPLETE 後 DONE（finish 只在最後一輪）。
  Q2 執行端 release FAILED ⇒ k3b_exec_release；Q3 mode 3 被拒 ⇒ k3b_retreat_rejected；Q4 5 N 被拒 ⇒ k3b_effort_reject；
  Q5 5 N 送出後 1 s 無完成 ⇒ k3b_effort_timeout；Q6 門速 0.06 rad/s ⇒ k3b_protect:door_rate；Q7 幾何 stop ⇒ k3b_geometry；
  Q8 mode 2 被拒 ⇒ k3b_open_rejected；Q9 測試保持期滿後（14.1 s，仍在 PAUSE）監督器離開 HELD ⇒ w2_supervisor_left_held；
  Q10 權威狀態停更 2 s（RELEASE 中）⇒ w2_state_stale；Q11 t_open 以前無同步量測 ⇒ k3b_open_baseline_missing。

    python3 evaluation/test_door_k3m_policy_k3b.py
"""
import math
import os
import sys
import types
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3b_release4 as KR                         # noqa: E402
import door_k3m_policy_k3b3 as PK                      # noqa: E402
import door_k3m_policy_p2 as PP2                       # noqa: E402

E0 = np.eye(4)
E0[:3, 3] = (-0.001, -0.0007, 0.0)
H0 = np.eye(4)
H0[:3, 3] = (0.6, -0.53, 0.85)
Q0 = math.radians(30.21)


def _fake_parent(self, m, dt):
    return {'target': 'T', 'grip': 0.0, 'request_handover': False, 'solver_start': False, 'stop_solver': False, 'finish': False}


class Geo:
    def __init__(self, stop_at=None):
        self.stop_at = stop_at

    def step(self, t, qf, q_door, q_finger, age):
        if self.stop_at is not None and t >= self.stop_at:
            return 'stop', {'why': 'geometry_clearance_nonfinger'}
        cl = {k: {'d': d, 'd_pred': d - 0.0003} for k, d in (('nonfinger', 0.008), ('finger', 0.03), ('self', 0.0108), ('base', 0.193))}
        return 'ok', {'classes': cl, 'compute_s': 0.015}


class Pair:
    def __init__(self, pol):
        self.pol = pol

    def step(self, t, qf, qd, fq, age):
        s = self.pol.rel['s_cmd']
        d = 0.002 + 0.5 * s
        return 'ok', [d, d], [d, d], {'compute_s': 0.001}


def mk(geo=None):
    p = object.__new__(PK.K3MReleasePolicy3)
    p.phase, p.t_phase, p.abort, p.t_abort, p.events, p.done = 'PAUSE', 0.0, None, None, [], False
    p.c = types.SimpleNamespace(post_abort_observe_s=1.0, stale_s=0.5, door_abort_rate=0.05, contact_abort_n=1.0, wrist_abort_n=30.0,
                                solver_stale_s=0.5, corridor_inner_mm=40.0, corridor_inner_deg=3.0, finger_open=0.0089)
    p.grip_cmd, p.pause_state, p.pause_status, p.pause_outcome = 0.0, 'active', None, None
    p.w2_run_id, p.w2_req_delay_s, p.w2_commit_timeout_s, p.w2_test_hold_s, p.w2_state_stale_s = 'r', 1.0, 2.0, 3.0, 1.5
    p.w2_state, p.w2_t, p.w2_outbox, p.w2_req, p.w2_last_state, p.w2_last_state_t, p.w2_n_state, p.w2_events = \
        'none', {}, [], None, None, None, 0, []
    p.k3b = KR.TaskReleaseMonitor()
    p.pair = Pair(p)
    p.retreat_rate, p.eff_ack_timeout_s, p.sync_hist_s = 0.02, 1.0, 1.0
    p.rel = {'state': 'none', 't': {}, 's_cmd': 0.0, 'H_rel': None, 'E_rel': None, 'q_rel': None, 'step_open': None, 'base_anchor': None,
             'eff_req': None, 'eff_ack': None, 'eff_done': None, 'n_samples': 0, 'events': []}
    p.eff_outbox, p.sync_hist = [], []
    p.geom, p.geom_last, p.geom_t = (geo or Geo()), None, None
    p.corridor = types.SimpleNamespace(check=lambda pose, q, a, b: (True, [0.0, 0.0], None))
    p.corr_last = None
    return p


class Sim:
    """權威（執行端）與執行端解除監看的替身；依時間給出狀態訊息。"""

    def __init__(self, reject3=False, reject_eff=False, no_eff=False, release_fail_at=None, reject2=False, stale_from=None,
                 left_held_at=None, door_qd=0.0, no_hist=False):
        self.reqs, self.commits, self.kw = [], [], dict(reject3=reject3, reject_eff=reject_eff, no_eff=no_eff, release_fail_at=release_fail_at,
                                                         reject2=reject2, stale_from=stale_from, left_held_at=left_held_at)
        self.door_qd, self.no_hist = door_qd, no_hist
        self.last_reject = None
        self.t_open = None
        self.eff_req_t = None

    def state(self, t):
        rel = {'state': 'IDLE'}
        if self.t_open is not None:
            fail = self.kw['release_fail_at']
            rel = {'state': 'FAILED' if fail is not None and t >= fail else ('B_CONFIRMED' if t >= self.t_open + 0.8 else 'A_OPENING'),
                   'why': 'confirm_timeout' if fail is not None and t >= fail else None, 't_open': self.t_open,
                   't_conf': self.t_open + 0.8 if t >= self.t_open + 0.8 else None}
        ms = max([c['mode_seq'] for c in self.commits] or [0])
        last = self.commits[-1] if self.commits else None                 # 同 ModeAuthority5.state_msg：commit＝最新、commit_req_seq＝其請求序號
        return {'run_id': 'r', 'mode_seq': ms, 'commits': list(self.commits), 'commit': last,
                'commit_req_seq': None if last is None else last['mode_seq'], 'last_reject': self.last_reject, 'aborted': None, 'release': rel}

    def on_req(self, r, t):
        self.reqs.append((t, r['mode_seq']))
        if r['mode_seq'] == 2 and self.kw['reject2']:
            self.last_reject = {'req_seq': 2, 'why': 'candidate_invalid:base_not_hold'}
            return
        if r['mode_seq'] == 3 and self.kw['reject3']:
            self.last_reject = {'req_seq': 3, 'why': 'retreat_evidence:not_confirmed_fresh'}
            return
        k = int(round((t + 0.2) * 100))
        self.commits.append({'mode_seq': r['mode_seq'], 'step': k, 't': t + 0.2, 'first_source_seq': 100 + r['mode_seq'], 'applied_confirmed': True})
        if r['mode_seq'] == 2:
            self.t_open = t + 0.2


def run(sim, t_end=30.0, geo=None, mod=None):
    PP2.K3MP2Policy.step, orig = _fake_parent, PP2.K3MP2Policy.step
    try:
        p = mk(geo)
        p.pair = Pair(p)
        tgt_prev = None
        targets, finishes = [], []
        for i in range(int(t_end / 0.05) + 1):
            t = round(i * 0.05, 4)
            held = t >= 10.0 and not (sim.kw['left_held_at'] is not None and t >= sim.kw['left_held_at'])
            p.pause_status = {'outcome': {'state': 'HELD' if held else 'CONFIRM', 'observation_passed': held, 'physical_stop_confirmed': held,
                                          'transitions': {'OBS_PASSED': 10.0}}}
            if not (sim.kw['stale_from'] is not None and t >= sim.kw['stale_from']):
                p.on_w2_state(sim.state(t), t)
            for r in p.take_w2_outbox():
                sim.on_req(r, t)
            for r in p.take_eff_outbox():
                sim.eff_req_t = t
            if sim.eff_req_t is not None and t >= sim.eff_req_t + 0.1 and not sim.kw['no_eff']:
                if sim.kw['reject_eff']:
                    p.on_effort_ack({'run_id': 'r', 'result': 'reject:evidence_not_clear'}, t)
                else:
                    p.on_effort_ack({'run_id': 'r', 'result': 'apply', 'done': {'req_seq': 1, 'readback_ok': True, 'switch_step': 9999}}, t)
            T = H0 @ (tgt_prev if tgt_prev is not None and p.phase == 'RELEASE' else E0) if False else \
                (np.asarray(tgt_prev, float) if (tgt_prev is not None and p.phase == 'RELEASE') else H0 @ E0)
            step = None if (sim.no_hist and sim.t_open is None) else int(round(t * 100))
            m = types.SimpleNamespace(t=t, state_step=step if step is not None else 'x', wall=t, H=H0, T_tcp=T, q_door=Q0, qdot_door=0.0,
                                      state_age=0.01, contacts_age=0.01, contacts_invalid=0, unexpected_max=0.0, wall_max=0.0, wrist_max=3.0,
                                      solver_failed=None, solver_up=True, solver_status_age=0.05, executor_violation=None, base_pose=(0, 0, 0),
                                      qf=np.zeros(9), q_finger=0.004, finger_q=(0.0089, 0.0089), base_drift_m=0.00002, base_drift_yaw=0.0,
                                      settle_sig={'valid': True, 'rate': sim.door_qd})
            if mod:
                mod(t, m, p)
            act = p.step(m, 0.05)
            if p.phase == 'RELEASE' or p.phase == 'DONE':
                targets.append((t, p.rel['s_cmd']))
            if act.get('target') is not None and not isinstance(act.get('target'), str):
                tgt_prev = act['target']
            finishes.append(bool(act.get('finish')))
            if p.phase in ('DONE', 'ABORTED'):
                break
        return p, targets, finishes
    finally:
        PP2.K3MP2Policy.step = orig


class TQ(unittest.TestCase):
    def test_q1_normal(self):
        sim = Sim()
        p, targets, fin = run(sim)
        self.assertEqual(p.phase, 'DONE', (p.abort, p.k3b.why))
        self.assertEqual([ms for _, ms in sim.reqs], [1, 2, 3])
        self.assertLessEqual(p.rel['t']['baseline'], sim.t_open + 1e-9)
        s_vals = [s for _, s in targets]
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(s_vals, s_vals[1:])))
        t_ret = p.rel['t']['retreat']
        self.assertTrue(all(s == 0.0 for t, s in targets if t < t_ret))
        self.assertAlmostEqual(max(s_vals), 0.10, places=9)
        self.assertEqual(p.rel['eff_req']['evidence']['s_meas'] >= 0.02, True)
        self.assertEqual(fin.count(True), 1)
        self.assertTrue(fin[-1])
        self.assertEqual(p.k3b.state, 'COMPLETE')
        self.assertTrue(p.k3b.ended)
        self.assertLessEqual(p.rel['anchor_record']['baseline_step'], p.rel['step_open'])

    def test_q2_to_q5(self):
        for sim, why in ((Sim(release_fail_at=14.9), 'k3b_exec_release:confirm_timeout'),
                         (Sim(reject3=True), 'k3b_retreat_rejected:retreat_evidence:not_confirmed_fresh'),
                         (Sim(reject_eff=True), 'k3b_effort_reject:evidence_not_clear'),
                         (Sim(no_eff=True), 'k3b_effort_timeout')):
            p, _, _ = run(sim)
            self.assertEqual(p.abort, why)

    def test_q6_q7(self):
        p, _, _ = run(Sim(), mod=lambda t, m, pp: setattr(m, 'qdot_door', 0.06) if pp.phase == 'RELEASE' else None)
        self.assertEqual(p.abort, 'k3b_protect:door_rate')
        p, _, _ = run(Sim(), geo=Geo(stop_at=16.0))
        self.assertEqual(p.abort, 'k3b_geometry:geometry_clearance_nonfinger')

    def test_q8_q9_q10(self):
        p, _, _ = run(Sim(reject2=True))
        self.assertEqual(p.abort, 'k3b_open_rejected:candidate_invalid:base_not_hold')
        p, _, _ = run(Sim(left_held_at=14.1))                                  # 保持期滿後第一輪（仍在 PAUSE）
        self.assertTrue(str(p.abort).startswith('w2_supervisor_left_held'), p.abort)
        p, _, _ = run(Sim(stale_from=15.0))
        self.assertEqual(p.abort, 'w2_state_stale')

    def test_q11_baseline_missing(self):
        sim = Sim()
        p, _, _ = run(sim, mod=lambda t, m, pp: setattr(m, 'state_step', None) if t < 14.6 else None)
        self.assertEqual(p.abort, 'k3b_open_baseline_missing')


if __name__ == '__main__':
    unittest.main(verbosity=1)
