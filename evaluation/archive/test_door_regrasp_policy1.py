#!/usr/bin/env python3
"""door_regrasp_policy1 純邏輯驗證（PAUSE 期父類以替身代換；換抓相位全用真類別＋真參考包 door_regrasp_ref_v1）。
合成世界：理想追蹤（實測＝目標）；假權威收到請求 0.1 s 後提交且 applied_confirmed；gate 依時間成立（gate3 提交後 0.6 s、gate4／5／6 參考走完後 1.0 s、
hold 提交 6 後 2.5 s）；v3 接觸窗口摘要預設全 0、complete。
  T1 正常：mode 1 保持 3 s 後轉 RG2，依序請求 2→6 各一次、最後 DONE（finish）；三時刻 t_commit／t_seen／t_ref0 分記；
     請求後未提交期間目標＝上一模式終點（參考不推進）；mode 3 目標＝起始實測 TCP 沿其工具 −z 退開；
  T2 指定接觸：mode 5 手指碰 bar_in 0.2 N ⇒ rg_contact_not_designated；任一模式非手指接觸 ⇒ rg_nonfinger_contact；
     mode 6 手指 bar_in 2 N 允許；mode 3 退開 ≥ 0.02 m 後外側類 0.2 N ⇒ 中止，退開前允許；
  T3 起始銜接：mode 3 提交時底盤偏 0.03 m ⇒ rg3_bridge_base；手臂偏 0.06 rad ⇒ rg3_bridge_arm；
  T4 權威狀態 gates.gate4＝FAILED ⇒ rg_exec_gate4_failed；T5 gate5 永不成立 ⇒ 參考走完後 10 s rg4_gate_timeout；
  T6 請求後無提交 ⇒ rg2_commit_timeout；T7 權威狀態停更 > 1.5 s ⇒ rg_state_stale；
  T8 淨空監看目標：mode 2–3 out、4 None、5 先 None、TCP 距預抓取 ≤ 5 mm 後 in、6 in；
  T9 門動：門角偏離解除時 > 1° ⇒ rg_door_moved；v3 摘要 complete False ⇒ rg_contact_v3_stale_or_incomplete。

    python3 evaluation/test_door_regrasp_policy1.py
"""
import json
import math
import os
import sys
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_policy_p2 as PP2                       # noqa: E402
import door_regrasp_policy1 as RP                      # noqa: E402

PKG = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v1.json')))
DT = 0.05
F1 = 'uflite_finger1'


def _fake_parent_step(self, m, dt):
    return {'target': self.last_target, 'grip': 0.0, 'request_handover': False, 'solver_start': False, 'stop_solver': False, 'finish': False}


def mk(reg=None):
    p = object.__new__(RP.RegraspPolicy)
    p.phase, p.t_phase, p.abort, p.t_abort, p.events, p.done = 'PAUSE', 0.0, None, None, [], False
    p.c = types.SimpleNamespace(post_abort_observe_s=1.0, solver_stale_s=0.5)
    p.grip_cmd, p.pause_state, p.pause_status, p.pause_outcome = 0.0, 'active', None, None
    p.w2_run_id, p.w2_req_delay_s, p.w2_commit_timeout_s, p.w2_test_hold_s, p.w2_state_stale_s = 'r', 1.0, 2.0, 3.0, 1.5
    p.w2_state, p.w2_t, p.w2_outbox, p.w2_req, p.w2_last_state, p.w2_last_state_t, p.w2_n_state, p.w2_events = \
        'committed', {'committed': 0.0}, [], None, None, None, 0, []
    p.last_target = np.asarray(PKG['segments']['3']['tcp'][0], float).reshape(4, 4)
    p.r = dict(RP.REG, **(reg or {}))
    p.ref = RP.RefRuntime(json.loads(json.dumps(PKG)))
    p.geom_v3_fn = None
    p.rg = {'k': 1, 'req': {}, 't_req': {}, 't_commit': {}, 't_seen': {}, 't_ref0': {}, 'bridge': {}, 'events': [],
            'q_rel_deg': None, 'last_T': None, 'geom_target': 'out', 'gate_wait_t0': None}
    p.rg_outbox, p.c3, p.c3_t = [], None, None
    return p


class Fake:
    """假權威＋理想追蹤世界。"""

    def __init__(self, p, gate_delay=None, gate_fail=None, no_commit=None, stale_after=None, base_off=None, arm_off=None,
                 contact=None, q_dev=None, incomplete_at=None):
        self.p, self.k, self.commits, self.req_t = p, 1, [{'step': 0, 't': 0.0, 'first_source_seq': 1, 'mode_seq': 1, 'applied_confirmed': True}], {}
        self.gate_delay = {'gate3': 0.6, 'gate4': 1.0, 'gate5': 1.0, 'gate6': 1.0, 'hold': 2.5}
        self.gate_delay.update(gate_delay or {})
        self.gate_fail, self.no_commit, self.stale_after = gate_fail, no_commit, stale_after
        self.base_off, self.arm_off, self.contact, self.q_dev, self.incomplete_at = base_off, arm_off, contact, q_dev, incomplete_at
        self.t_commit = {1: 0.0}
        self.acts = []
        seg3 = PKG['segments']['3']
        self.base = np.asarray(seg3['base'][0], float)
        self.arm = np.asarray(seg3['arm'][0], float)
        self.T = np.asarray(seg3['tcp'][0], float).reshape(4, 4)

    def gates(self, t):
        g = {}
        p = self.p
        for k, name in ((2, 'gate3'), (3, 'gate4'), (4, 'gate5'), (5, 'gate6'), (6, 'hold')):
            if k in self.t_commit:
                t0 = self.t_commit[k] if k in (2, 6) else (p.rg['t_ref0'].get(k, 1e9) + p.ref.length_s(k))
                ok = t >= t0 + self.gate_delay[name]
                g[name] = {'state': 'CONFIRMED' if ok else 'WAITING', 'fresh': ok}
                if self.gate_fail == name:
                    g[name] = {'state': 'FAILED', 'why': 'test'}
        return g

    def state(self, t):
        return {'run_id': 'r', 'mode_seq': self.k, 'commits': list(self.commits), 'commit_req_seqs': [c['mode_seq'] for c in self.commits],
                'last_reject': None, 'aborted': None, 'gates': self.gates(t)}

    def run(self, t_end):
        t = 3.0
        while t <= t_end + 1e-9 and self.p.phase not in ('DONE', 'ABORTED'):
            for r in self.p.take_w2_outbox():
                self.req_t[r['mode_seq']] = t
            for k, tr in list(self.req_t.items()):
                if k == self.k + 1 and t >= tr + 0.1 and k != self.no_commit:
                    self.k = k
                    self.t_commit[k] = t
                    self.commits.append({'step': int(t / 0.01), 't': t, 'first_source_seq': 100 + k, 'mode_seq': k, 'applied_confirmed': True})
                    if k == 3:
                        self.base = self.base + np.r_[self.base_off or 0.0, 0.0, 0.0]
                        self.arm = self.arm + (self.arm_off or 0.0)
            if self.stale_after is None or t < self.stale_after:
                self.p.on_w2_state(self.state(t), t)
            mx = {}
            if self.contact is not None:
                k_on, link, cls, f = self.contact
                if self.p.rg.get('k_committed', 1) == k_on and (k_on != 3 or (self.p.rg.get('s_meas') or 0) >= 0.0):
                    mx = {link: {cls: f}}
            comp = not (self.incomplete_at is not None and t >= self.incomplete_at)
            self.p.on_c3({'complete': comp, 'max': mx}, t)
            q = math.radians(PKG['q_ref_deg'] + (self.q_dev(t) if self.q_dev else 0.0))
            m = types.SimpleNamespace(t=round(t, 6), q_door=q, qdot_door=0.0, state_age=0.01, contacts_age=0.01, wall_max=0.0, wrist_max=1.0,
                                      solver_failed=None, solver_up=True, solver_status_age=0.05, base_pose=tuple(self.base),
                                      qf=tuple(np.r_[self.base, self.arm]), T_tcp=self.T.copy(), q_finger=0.0, executor_violation=None)
            a = self.p.step(m, DT)
            self.acts.append((t, a, self.p.phase, self.p.rg.get('geom_target')))
            if a.get('target') is not None:
                self.T = np.asarray(a['target'], float)                    # 理想追蹤
            kc = self.p.rg.get('k_committed', 1)
            if kc in (3, 4, 5) and self.p.rg.get('t_ref0', {}).get(kc) is not None:
                tau = t - self.p.rg['t_ref0'][kc]
                _, b, _ = self.p.ref.at(kc, tau)
                self.base = np.asarray(b, float)
                sg = PKG['segments'][str(kc)]
                i = min(int(round(tau / DT)), len(sg['t']) - 1)
                self.arm = np.asarray(sg['arm'][i], float)
            t = round(t + DT, 6)
        return self


class TP(unittest.TestCase):
    def setUp(self):
        self._orig = PP2.K3MP2Policy.step
        PP2.K3MP2Policy.step = _fake_parent_step

    def tearDown(self):
        PP2.K3MP2Policy.step = self._orig

    def test_t1_normal(self):
        p = mk()
        w = Fake(p).run(200.0)
        self.assertEqual(p.phase, 'DONE', p.abort)
        self.assertEqual(sorted(w.req_t), [2, 3, 4, 5, 6])
        self.assertEqual([e['request'] for e in p.rg['events'] if 'request' in e], [2, 3, 4, 5, 6])
        for k in range(2, 7):
            self.assertIsNotNone(p.rg['t_commit'][k])
            self.assertGreaterEqual(p.rg['t_seen'][k], p.rg['t_commit'][k])
            self.assertEqual(p.rg['t_ref0'][k], p.rg['t_seen'][k])
        self.assertEqual(sum(1 for _, a, _, _ in w.acts if a['finish']), 1)
        # 請求 4 之後、提交前：目標＝mode 3 終點
        t4r, t4c = w.req_t[4], w.t_commit[4]
        held = [a['target'] for t, a, _, _ in w.acts if t4r < t < t4c]
        self.assertTrue(held and all(np.allclose(h, held[0]) for h in held))
        # mode 3 目標：起始 TCP 沿其工具 −z
        T0 = p.ref.ctx[3]['T0']
        a3 = [a['target'] for t, a, ph, _ in w.acts if ph == 'RG3']
        d = np.asarray(a3[-1])[:3, 3] - T0[:3, 3]
        self.assertGreater(float(-d @ T0[:3, 2]), 0.05)
        lat = d - float(d @ T0[:3, 2]) * T0[:3, 2]
        self.assertLess(float(np.linalg.norm(lat)), 1e-9)                       # 純沿起始工具 −z
        self.assertTrue(np.allclose(np.asarray(a3[-1])[:3, :3], T0[:3, :3]))    # 姿態不變

    def test_t2_contacts(self):
        p = mk()
        Fake(p, contact=(5, F1, 'bar_in', 0.2)).run(200.0)
        self.assertEqual(p.abort, 'rg_contact_not_designated')
        p = mk()
        Fake(p, contact=(4, 'link6', 'panel', 0.2)).run(200.0)
        self.assertEqual(p.abort, 'rg_nonfinger_contact')
        p = mk()
        Fake(p, contact=(6, F1, 'bar_in', 2.0)).run(200.0)
        self.assertEqual(p.phase, 'DONE')
        p = mk()
        Fake(p, contact=(3, F1, 'bar_out', 0.2)).run(200.0)
        self.assertEqual(p.abort, 'rg_contact_not_designated')
        ev = next(e for e in p.events if e.get('abort'))
        self.assertGreaterEqual(p.rg['s_meas'], 0.02, ev)

    def test_t3_bridge(self):
        p = mk()
        Fake(p, base_off=0.03).run(200.0)
        self.assertEqual(p.abort, 'rg3_bridge_base')
        p = mk()
        Fake(p, arm_off=0.06).run(200.0)
        self.assertEqual(p.abort, 'rg3_bridge_arm')

    def test_t4_t5_gates(self):
        p = mk()
        Fake(p, gate_fail='gate4').run(200.0)
        self.assertEqual(p.abort, 'rg_exec_gate4_failed:test')
        p = mk()
        w = Fake(p, gate_delay={'gate5': 1e9}).run(200.0)
        self.assertEqual(p.abort, 'rg4_gate_timeout')
        t_ab = p.t_abort
        t_end4 = p.rg['t_ref0'][4] + p.ref.length_s(4)
        self.assertAlmostEqual(t_ab - t_end4, 10.0, delta=0.11)
        del w

    def test_t6_t7(self):
        p = mk()
        Fake(p, no_commit=2).run(50.0)
        self.assertEqual(p.abort, 'rg2_commit_timeout')
        p = mk()
        Fake(p, stale_after=10.0).run(50.0)
        self.assertEqual(p.abort, 'rg_state_stale')

    def test_t8_geom_target(self):
        p = mk()
        w = Fake(p).run(200.0)
        tg = {}
        for _, _, ph, g in w.acts:
            tg.setdefault(ph, set()).add(g)
        self.assertEqual(tg['RG3'], {'out'})
        self.assertEqual(tg['RG4'], {None})
        self.assertEqual(tg['RG5'], {None, 'in'})
        self.assertEqual(tg['RG6'], {'in'})
        self.assertTrue(any(e.get('geom_target') == 'in' for e in p.rg['events']))

    def test_t9_door_and_c3(self):
        p = mk()
        Fake(p, q_dev=lambda t: 0.0 if t < 20 else 1.5).run(200.0)
        self.assertEqual(p.abort, 'rg_door_moved')
        p = mk()
        Fake(p, incomplete_at=20.0).run(200.0)
        self.assertEqual(p.abort, 'rg_contact_v3_stale_or_incomplete')


if __name__ == '__main__':
    unittest.main(verbosity=1)
