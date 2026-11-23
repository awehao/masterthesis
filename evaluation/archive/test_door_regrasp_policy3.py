#!/usr/bin/env python3
"""door_regrasp_policy3 驗證（Codex r53 §2–§5）。假世界沿用 test_door_regrasp_policy1 的權威與 gate 排程，差異：
接觸摘要改發執行端格式（door_regrasp_contact_window：依**假權威當步模式**分組、src_step／src_t、exited3 由退開量 ≥ 0.02 單向閂鎖），
幾何為真 v3 後端＋GeometryMonitor；理想追蹤時 mode 4／5 實測構型＝RefRuntime3.cfg（含銜接與門角修正）。
  V1 權威 mode 6 提交當輪即有 bar_in 2 N（任務端尚未更新 k_committed）⇒ 不中止、DONE；
  V2 mode 2 手指 post_out0 0.2 N ⇒ rg_contact_not_designated；mode 3 離開後 bar_out 0.2 N ⇒ 中止；
  V3 摘要 violation 非空 ⇒ rg_exec_contact_violation；V4 摘要缺 by_mode ⇒ rg_contact_v3_malformed；
  V5 摘要 src_t 停在 20 s（接收仍持續）⇒ rg_contact_v3_stale_or_incomplete；V6 無幾何後端 ⇒ 進入即 rg_geometry_backend_missing；
  V7 mode 4 起點底盤偏 19 mm：起點參考＝實測（差 < 1e-9）、參考逐樣本步長最大值 ≤ 原表最大步長＋0.5 mm（無跳變）；
  V8 Δq＝+1°：mode 5 前饋＝同一條修正參考的差分（逐樣本相等；直接驗 RefRuntime3）；V9 配對預測使用實測接近率（事件含 rate、v_used ≥ v_floor）。
  V1／V7 用替身幾何（固定大距離）隔離機制；V10 真後端重現已知衝突（mode 4 A* 段 geometry_clearance_base；重新規劃前的現況紀錄）。

    python3 evaluation/test_door_regrasp_policy3.py
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_k3m_policy_p2 as PP2                       # noqa: E402
import door_regrasp_policy3 as RP3                     # noqa: E402
import test_door_regrasp_policy1 as T1                 # noqa: E402
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa: E402

KIN = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
G = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k3m_offline_geom_monitor_r3.json')))
F1 = 'uflite_finger1'


class StubGeom:
    """機制測試用替身後端：四類固定大距離（不代表淨空成立；真後端見 V10）。"""

    def __init__(self):
        self.last, self.tgt = None, 'out'

    def set_target(self, t):
        self.tgt = t

    def __call__(self, qf, q, qfi):
        self.last = {'nonfinger': (0.1, 'stub'), 'finger': (0.1, 'stub'), 'base': (0.5, 'stub'), 'self': (0.1, 'stub'),
                     'finger_bar_out': (0.04, 'stub'), 'finger_bar_in': (0.04, 'stub')}
        return self.last


def mk3(geom=True):
    p = T1.mk()
    p.__class__ = RP3.RegraspPolicy3
    p.r = dict(RP3.REG)
    p.ref = RP3.RefRuntime3(p.ref.p, KIN, T_b=p.r['bridge_T_s'])
    p.pair_prev = None
    p.geom_last, p.geom_t = None, None
    if geom == 'stub':
        fn = StubGeom()
        p.geom_v3_fn = fn
        p.geom = GM.GeometryMonitor(GM.GeomMonitorConfig(v_floor=dict(G['v_floor_registered']), ds_err=dict(G['downsampling']['ds_err_registered'])), fn)
    elif geom:
        fn = GM3.backend_v3()
        p.geom_v3_fn = fn
        p.geom = GM.GeometryMonitor(GM.GeomMonitorConfig(v_floor=dict(G['v_floor_registered']), ds_err=dict(G['downsampling']['ds_err_registered'])), fn)
    else:
        p.geom_v3_fn, p.geom = None, None
    return p


class Fake3(T1.Fake):
    def __init__(self, p, contact3=None, violation_at=None, drop_key_at=None, freeze_src_after=None, off4=None, dq=0.0, **kw):
        super().__init__(p, q_dev=(lambda t: dq) if dq else None, **kw)
        self.contact3, self.violation_at, self.drop_key_at, self.freeze_src_after, self.off4 = contact3, violation_at, drop_key_at, freeze_src_after, off4
        self.exited = False
        self.src = (0, 0.0)
        self.b_log = {}

    def summary(self, t):
        k = self.k
        s_meas = self.p.rg.get('s_meas') or 0.0
        if k >= 3 and s_meas >= 0.02:
            self.exited = True
        bm = {str(k): {}}
        if self.contact3 is not None and self.contact3[0] == k:
            _, link, cls, f = self.contact3
            bm[str(k)] = {link: {cls: f}}
        if self.freeze_src_after is None or t < self.freeze_src_after:
            self.src = (int(round(t / 0.01)), t)
        c = {'src_step': self.src[0], 'src_t': self.src[1], 'complete': True, 'by_mode': bm, 'exited3': self.exited,
             'violation': ({'step': 1, 'why': 'test'} if (self.violation_at is not None and t >= self.violation_at) else None)}
        if self.drop_key_at is not None and t >= self.drop_key_at:
            c.pop('by_mode')
        return c

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
                    if k == 4 and self.off4:
                        self.base = self.base + np.r_[self.off4, 0.0, 0.0]
            self.p.on_w2_state(self.state(t), t)
            self.p.c3, self.p.c3_t = self.summary(t), t
            q = math.radians(T1.PKG['q_ref_deg'] + (self.q_dev(t) if self.q_dev else 0.0))
            m = types.SimpleNamespace(t=round(t, 6), q_door=q, qdot_door=0.0, state_age=0.01, contacts_age=0.01, wall_max=0.0, wrist_max=1.0,
                                      solver_failed=None, solver_up=True, solver_status_age=0.05, base_pose=tuple(self.base),
                                      qf=tuple(np.r_[self.base, self.arm]), T_tcp=self.T.copy(), q_finger=0.0089 if self.k in (2, 3, 4, 5) else 0.0,
                                      executor_violation=None)
            a = self.p.step(m, T1.DT)
            self.acts.append((t, a, self.p.phase, self.p.rg.get('geom_target')))
            if a.get('target') is not None:
                self.T = np.asarray(a['target'], float)
            kc = self.p.rg.get('k_committed', 1)
            if kc in (4, 5) and self.p.rg.get('t_ref0', {}).get(kc) is not None:
                tau = t - self.p.rg['t_ref0'][kc]
                i = min(int(round(tau / T1.DT)), len(T1.PKG['segments'][str(kc)]['t']) - 1)
                x = self.p.ref.cfg(kc, i)
                self.base, self.arm = x[:3].copy(), x[3:].copy()
                self.b_log.setdefault(kc, []).append((tau, x[:3].copy()))
            elif kc == 3 and self.p.rg.get('t_ref0', {}).get(3) is not None:
                tau = t - self.p.rg['t_ref0'][3]
                _, b, _ = self.p.ref.at(3, tau)
                self.base = np.asarray(b, float)
                sg = T1.PKG['segments']['3']
                self.arm = np.asarray(sg['arm'][min(int(round(tau / T1.DT)), len(sg['t']) - 1)], float)
            t = round(t + T1.DT, 6)
        return self


class TV(unittest.TestCase):
    def setUp(self):
        self._orig = PP2.K3MP2Policy.step
        PP2.K3MP2Policy.step = T1._fake_parent_step

    def tearDown(self):
        PP2.K3MP2Policy.step = self._orig

    def test_v1(self):
        p = mk3('stub')
        w = Fake3(p, contact3=(6, F1, 'bar_in', 2.0)).run(200.0)
        self.assertEqual(p.phase, 'DONE', (p.abort, [e for e in p.events if e.get('abort')][-1:]))
        self.assertIn(6, w.t_commit)

    def test_v2(self):
        p = mk3()
        Fake3(p, contact3=(2, F1, 'post_out0', 0.2)).run(60.0)
        self.assertEqual(p.abort, 'rg_contact_not_designated')
        p = mk3()
        Fake3(p, contact3=(3, F1, 'bar_out', 0.2)).run(60.0)
        self.assertEqual(p.abort, 'rg_contact_not_designated')

    def test_v3_v4_v5_v6(self):
        p = mk3()
        Fake3(p, violation_at=20.0).run(60.0)
        self.assertEqual(p.abort, 'rg_exec_contact_violation')
        p = mk3()
        Fake3(p, drop_key_at=20.0).run(60.0)
        self.assertEqual(p.abort, 'rg_contact_v3_malformed')
        p = mk3()
        Fake3(p, freeze_src_after=20.0).run(60.0)
        self.assertEqual(p.abort, 'rg_contact_v3_stale_or_incomplete')
        p = mk3(geom=False)
        Fake3(p).run(10.0)
        self.assertEqual(p.abort, 'rg_geometry_backend_missing')

    def test_v7_bridge(self):
        p = mk3('stub')
        w = Fake3(p, off4=0.019).run(200.0)
        self.assertIn(5, w.t_commit, p.abort)
        d0 = p.rg['bridge'][4]['d0']
        self.assertAlmostEqual(d0[0], 0.019, delta=2e-3)
        b = np.array([x for _, x in w.b_log[4]])
        sg = np.array(T1.PKG['segments']['4']['base'])
        step_ref = float(np.max(np.linalg.norm(np.diff(sg[:, :2], axis=0), axis=1)))
        step_cmd = float(np.max(np.linalg.norm(np.diff(b[:, :2], axis=0), axis=1)))
        self.assertLessEqual(step_cmd, step_ref + 0.0005, (step_cmd, step_ref))
        x0 = p.ref.cfg(4, 0)
        self.assertLess(abs(x0[0] - (sg[0, 0] + d0[0])), 1e-9)

    def test_v8_ff(self):
        p = mk3('stub')
        sg5 = T1.PKG['segments']['5']
        ok, why, det = p.ref.start(5, sg5['base'][0], sg5['arm'][0], T1.PKG['q_ref_deg'] + 1.0, None)
        self.assertTrue(ok, why)
        self.assertAlmostEqual(p.ref.ctx[5]['dq_rad'], math.radians(1.0), places=9)
        n = len(T1.PKG['segments']['5']['t'])
        for i in range(0, n - 1, 7):
            _, b0, ff = p.ref.at(5, i * T1.DT)
            _, b1, _ = p.ref.at(5, (i + 1) * T1.DT)
            d = (np.asarray(b1) - np.asarray(b0)) / T1.DT
            d[2] = math.remainder((b1[2] - b0[2]), 2 * math.pi) / T1.DT
            self.assertTrue(np.allclose(ff, d, atol=1e-9), (i, ff, d))

    def test_v10_known_conflict(self):
        """已知衝突（送 Codex）：參考以「淨空 ≥ 門檻」規劃，未預留監看的預測扣減 ⇒ 真後端在 mode 4 A* 段以 geometry_clearance_base 停。重新規劃前此測試記錄現況。"""
        p = mk3()
        Fake3(p).run(200.0)
        self.assertEqual(p.abort, 'rg_geometry:geometry_clearance_base')

    def test_v9_pair_rate(self):
        p = mk3()
        Fake3(p).run(60.0)
        ev = [e for e in p.rg['events'] if 'pair_check_before_mode4' in e and e.get('detail')]
        self.assertTrue(ev)
        d = ev[-1]['detail']
        self.assertIn('rate', d)
        self.assertGreaterEqual(d['v_used'], 0.002 - 1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=1)
