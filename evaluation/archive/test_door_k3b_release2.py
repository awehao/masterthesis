#!/usr/bin/env python3
"""door_k3b_release2 反例（Codex 20261026_000706：分層＋三缺陷＋新鮮度）。
執行端 ExecReleaseMonitor：
  E1 正常：0.31 s 開到 95 %、承載 0.04 s 內降到 0 ⇒ 0.81 s 進 B_CONFIRMED；confirmed_fresh 只在逐步資料未斷時為真。
  E2 Codex 重現一：同一物理步分別帶 t＝0.01、0.51 ⇒ time_conflict_same_step（不累積、不進 B）；同步同 t 重送 ⇒ 忽略。
  E3 Codex 重現二：首次確認在 1.51 s ⇒ confirm_timeout（先查期限再轉移）；1.50 s 恰成立 ⇒ 通過。
  E4 以 20 Hz 餵入（步號 5、10、15…）⇒ 缺步、無效、0.2 s 後 data_invalid（執行端必須逐步）。
  E5 段 A 承載 > 18 N ⇒ pair_force_design_bound；確認後 0.2 N ⇒ pair_force_after_confirm；確認後開度掉 ⇒ finger_closed_after_confirm。
  E6 步號倒退、時間不遞增 ⇒ step_or_time_backward；接觸資料無效 0.15 s ⇒ 只延後確認。
任務端 TaskReleaseMonitor：
  T1 正常：ARMED →（exec B）NEAR → C_CLEAR（s ≥ 0.02、配對 ≥ 5 mm）→ RETREAT_DONE（s ≥ 0.095 且 0.5 s 內移動 ≤ 1 mm）→ COMPLETE（保持 3 s）。
  T2 Codex 重現三：COMPLETE／RETREAT_DONE 後輸入超力、淨空不足 ⇒ FAILED（READY 後監看持續到 end()）；end() 之後不再判。
  T3 停更：src_step 不前進 0.25 s ⇒ task_data_stale；結果年齡 state_age＋compute 0.25 s 連續 ⇒ task_data_stale；短暫 0.1 s ⇒ 只跳過。
  T4 退出不只靠 s：s ≥ 0.02 但配對 4 mm ⇒ 不退出、near_exit_timeout；門角帶跨界（s 0.025、Δ 0.6°）⇒ door_band。
  T5 退離期間 −1.2° ⇒ door_band（一般下界 −1°）；終點時 +2.7° ⇒ 仍在帶內（終點帶 +5°），stow_ready False（收臂前置只作旗標）。
5 N 恢復權威 EffortRestoreAuthority：
  R1 正常：mode 3、證據新鮮且 s ≥ 0.02／配對 ≥ 5 mm、執行端 B 靜窗 ≥ 0.5 s ⇒ apply（切換步記錄）；讀回回報。
  R2 證據過期 0.3 s、證據 s 0.015、配對 4 mm、band_ok False、未來來源步 ⇒ reject；req_seq 不遞增 ⇒ reject；他趟 ⇒ reject。
  R3 mode 2 時請求 ⇒ reject:mode_not_retreat；中止後 ⇒ reject:after_abort；pending 中執行端失效 ⇒ reject:exec_state；
     靜窗未滿且期限 0.5 s 到 ⇒ reject:deadline。

    python3 evaluation/test_door_k3b_release2.py
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3b_release2 as R                          # noqa: E402

FO = R.REG['finger_open_m']
E_REL = np.eye(4)
E_REL[:3, 3] = (-0.001, -0.0007, 0.0)
Q_REL = math.radians(30.21)


def fq_at(t, final=1.0):
    f = 0.2 + 0.75 * t / 0.31 if t < 0.31 else 0.95 + (t - 0.31) * 0.2
    return FO * min(final, f)


def pf_at(t):
    return max(0.0, 12.0 * (1 - t / 0.04))


def exec_run(m, t_end=2.0, every=1, mod=None, final=1.0):
    m.open(0, 0.0)
    for k in range(every, int(t_end / 0.01) + 1, every):
        t = round(k * 0.01, 4)
        a = dict(finger_q=[fq_at(t, final)] * 2, pair_force=[pf_at(t)] * 2, contacts_valid=True)
        if mod:
            a.update(mod(t) or {})
        m.step(k, t, **a)
        if m.state == 'FAILED':
            break
    return m


class TE(unittest.TestCase):
    def test_e1_normal(self):
        m = exec_run(R.ExecReleaseMonitor(), t_end=1.0)
        self.assertEqual(m.state, 'B_CONFIRMED')
        self.assertAlmostEqual(m.t['B_CONFIRMED'], 0.81, places=6)
        self.assertTrue(m.confirmed_fresh(101))
        self.assertFalse(m.confirmed_fresh(105))

    def test_e2_same_step_time_conflict(self):
        m = R.ExecReleaseMonitor()
        m.open(0, 0.0)
        m.step(1, 0.01, [FO] * 2, [0.0] * 2, True)
        m.step(1, 0.01, [FO] * 2, [0.0] * 2, True)
        self.assertEqual(m.n['dup'], 1)
        m.step(1, 0.51, [FO] * 2, [0.0] * 2, True)
        self.assertEqual(m.why, 'time_conflict_same_step')
        self.assertNotIn('B_CONFIRMED', m.t)

    def test_e3_deadline_first(self):
        # 開度在 1.01 s 才到 95 % ⇒ 持續窗滿於 1.51 s ⇒ 期限 1.5 s 已過
        m = exec_run(R.ExecReleaseMonitor(), mod=lambda t: {'finger_q': [FO * (0.5 if t < 1.01 else 1.0)] * 2})
        self.assertEqual(m.why, 'confirm_timeout')
        self.assertAlmostEqual(m.t['failed'], 1.51, places=6)
        m = exec_run(R.ExecReleaseMonitor(), mod=lambda t: {'finger_q': [FO * (0.5 if t < 1.0 - 1e-9 else 1.0)] * 2})
        self.assertEqual(m.state, 'B_CONFIRMED')
        self.assertAlmostEqual(m.t['B_CONFIRMED'], 1.50, places=6)

    def test_e4_sparse_feed(self):
        m = exec_run(R.ExecReleaseMonitor(), every=5)
        self.assertEqual(m.why, 'data_invalid')
        self.assertGreater(m.n['missing'], 0)

    def test_e5_forces(self):
        m = R.ExecReleaseMonitor()
        m.open(0, 0.0)
        m.step(1, 0.01, [FO * 0.2] * 2, [18.5, 3.0], True)
        self.assertEqual(m.why, 'pair_force_design_bound')
        m = exec_run(R.ExecReleaseMonitor(), mod=lambda t: {'pair_force': [0.2, 0.0]} if t >= 0.9 else None)
        self.assertEqual(m.why, 'pair_force_after_confirm')
        m = exec_run(R.ExecReleaseMonitor(), mod=lambda t: {'finger_q': [FO * 0.9] * 2} if t >= 0.9 else None)
        self.assertEqual(m.why, 'finger_closed_after_confirm')

    def test_e6_order_and_brief_invalid(self):
        m = R.ExecReleaseMonitor()
        m.open(0, 0.0)
        m.step(2, 0.02, [FO] * 2, [0.0] * 2, True)
        m.step(1, 0.03, [FO] * 2, [0.0] * 2, True)
        self.assertEqual(m.why, 'step_or_time_backward')
        m = exec_run(R.ExecReleaseMonitor(), mod=lambda t: {'contacts_valid': False} if 0.5 <= t < 0.65 else None)
        self.assertEqual(m.state, 'B_CONFIRMED')
        self.assertAlmostEqual(m.t['B_CONFIRMED'], 1.15, places=6)


def tsample(src, t, s, exec_state, pair=None, door_dq=0.0, door_qd=0.0, nonf=None, age=0.01, comp=0.015, E_extra=None, t_conf=0.81):
    E = E_REL.copy()
    E[:3, 3] = E_REL[:3, 3] + s * R.A_H
    if E_extra is not None:
        E[:3, 3] += E_extra
    p = 0.002 + 0.5 * s if pair is None else pair
    nf = 0.00796 + s if nonf is None else nonf
    return {'t': t, 'src_step': src, 'state_age': age, 'compute_s': comp, 'E': E.tolist(), 's_cmd': s, 'door_q': Q_REL + math.radians(door_dq),
            'door_qd': door_qd, 'pair_d': [p, p], 'pair_d_pred': [p, p], 'nonfinger_d': nf, 'nonfinger_d_pred': nf, 'self_d': 0.0108,
            'self_d_pred': 0.0108, 'base_d': 0.193, 'base_d_pred': 0.193, 'base_dev_m': 0.00002, 'base_dev_yaw': 0.0, 'wrist_n': 3.0,
            'unexpected_n': 0.0, 'exec_state': exec_state, 't_conf': t_conf}


def task_run(m, t_end=12.0, mod=None):
    m.open(0.0, 0, Q_REL, E_REL)
    s = 0.0
    for i in range(1, int(t_end / 0.05) + 1):
        t = round(i * 0.05, 4)
        es = 'B_CONFIRMED' if t >= 0.81 else 'A_OPENING'
        if m.state in ('NEAR', 'C_CLEAR') or (m.state in ('RETREAT_DONE', 'COMPLETE') and s < 0.10):
            s = min(0.10, s + 0.02 * 0.05)
        kw = (mod(t, s, m) if mod else None) or {}
        sk = {k: kw.pop(k) for k in ('door_dq', 'door_qd', 'pair', 'nonf', 'age', 'comp', 'E_extra') if k in kw}
        x = tsample(i * 5, t, s, es, **sk)
        x.update(kw)
        m.step(x)
        if m.state == 'FAILED':
            break
    return m


class TT(unittest.TestCase):
    def test_t1_normal(self):
        m = task_run(R.TaskReleaseMonitor())
        self.assertEqual(m.state, 'COMPLETE', m.why)
        self.assertEqual([e['to'] for e in m.events], ['ARMED', 'NEAR', 'C_CLEAR', 'RETREAT_DONE', 'COMPLETE'])
        self.assertGreaterEqual(m.t['COMPLETE'] - m.t['RETREAT_DONE'], 3.0 - 1e-9)
        self.assertTrue(m.flags['stow_ready'])

    def test_t2_monitor_after_ready(self):
        m = task_run(R.TaskReleaseMonitor(), mod=lambda t, s, mm: {'wrist_n': 31.0} if mm.state == 'COMPLETE' else None, t_end=14.0)
        self.assertEqual(m.why, 'wrist_force')
        m = task_run(R.TaskReleaseMonitor(), mod=lambda t, s, mm: {'nonfinger_d': 0.004} if mm.state == 'RETREAT_DONE' else None)
        self.assertEqual(m.why, 'nonfinger_clearance')
        m = task_run(R.TaskReleaseMonitor(), t_end=12.0)
        m.end(12.1)
        m.step(tsample(10 ** 6, 12.2, 0.1, 'B_CONFIRMED', nonf=0.0))
        self.assertEqual(m.state, 'COMPLETE')

    def test_t3_stale(self):
        m = task_run(R.TaskReleaseMonitor(), mod=lambda t, s, mm: {'src_step': 100} if 1.0 <= t < 1.35 else None)
        self.assertEqual(m.why, 'task_data_stale')
        m = task_run(R.TaskReleaseMonitor(), mod=lambda t, s, mm: {'age': 0.24} if 1.0 <= t < 1.3 else None)
        self.assertEqual(m.why, 'task_data_stale')
        m = task_run(R.TaskReleaseMonitor(), mod=lambda t, s, mm: {'age': 0.24} if 1.0 <= t < 1.1 else None)
        self.assertEqual(m.state, 'COMPLETE', m.why)
        self.assertGreater(m.n['stale'], 0)

    def test_t4_exit_not_by_s_alone(self):
        m = task_run(R.TaskReleaseMonitor(), mod=lambda t, s, mm: {'pair_d': [0.004] * 2, 'pair_d_pred': [0.004] * 2} if mm.state == 'NEAR' else None)
        self.assertEqual(m.why, 'near_exit_timeout')
        m = task_run(R.TaskReleaseMonitor(), mod=lambda t, s, mm: {'door_dq': 0.6} if 0.024 <= s <= 0.026 else None)
        self.assertEqual(m.why, 'door_band')

    def test_t5_lower_and_stow_flag(self):
        m = task_run(R.TaskReleaseMonitor(), mod=lambda t, s, mm: {'door_dq': -1.2} if s >= 0.05 else None)
        self.assertEqual(m.why, 'door_band')
        m = task_run(R.TaskReleaseMonitor(), mod=lambda t, s, mm: {'door_dq': 2.7} if s >= 0.095 else None)
        self.assertEqual(m.state, 'COMPLETE', m.why)
        self.assertFalse(m.flags['stow_ready'])


def ev(t, s=0.03, pair=0.008, band=True, src=10):
    return {'src_step': src, 't': t, 's_meas': s, 'pair_d_min': pair, 'band_ok': band}


def req(seq, evd, mode=3, run='r'):
    return {'run_id': run, 'req_seq': seq, 'mode_seq': mode, 'evidence': evd}


class TR(unittest.TestCase):
    def mon(self):
        m = exec_run(R.ExecReleaseMonitor(), t_end=1.0)
        return m

    def test_r1_normal(self):
        a, m = R.EffortRestoreAuthority('r'), self.mon()
        self.assertEqual(a.request(req(1, ev(1.0)), 100, 1.0, 3), 'pending')
        out = a.tick(100, 1.0, m, 3, True)
        self.assertEqual(out, 'apply')                                         # 靜窗自 0.31 起已 ≥ 0.5 s
        a.confirm(True, 100, 1.0)
        self.assertEqual(a.ack()['done']['readback_ok'], True)
        self.assertEqual(a.ack()['done']['switch_step'], 100)

    def test_r2_evidence(self):
        m = self.mon()
        for e, why in ((ev(0.7), 'evidence_stale_or_future'), (ev(1.0, s=0.015), 'evidence_not_clear'), (ev(1.0, pair=0.004), 'evidence_not_clear'),
                       (ev(1.0, band=False), 'evidence_malformed'), (ev(1.0, src=500), 'evidence_stale_or_future')):
            a = R.EffortRestoreAuthority('r')
            self.assertEqual(a.request(req(1, e), 100, 1.0, 3), f'reject:{why}')
        a = R.EffortRestoreAuthority('r')
        self.assertEqual(a.request(req(0, ev(1.0)), 100, 1.0, 3), 'reject:req_seq_not_increasing')
        self.assertEqual(a.request(req(1, ev(1.0), run='x'), 100, 1.0, 3), 'reject:wrong_run_or_malformed')
        self.assertIsNotNone(m)

    def test_r3_mode_abort_exec_deadline(self):
        m = self.mon()
        a = R.EffortRestoreAuthority('r')
        self.assertEqual(a.request(req(1, ev(1.0), mode=2), 100, 1.0, 2), 'reject:mode_not_retreat')
        a.abort('k2_latch', 100, 1.0)
        self.assertEqual(a.request(req(2, ev(1.0)), 100, 1.0, 3), 'reject:after_abort')
        a = R.EffortRestoreAuthority('r')
        a.request(req(1, ev(1.0)), 100, 1.0, 3)
        bad = R.ExecReleaseMonitor()
        bad.state = 'FAILED'
        self.assertEqual(a.tick(101, 1.01, bad, 3, True), 'reject:exec_state')
        a = R.EffortRestoreAuthority('r')
        young = R.ExecReleaseMonitor()
        young.state, young.quiet_t0 = 'B_CONFIRMED', 0.9
        a.request(req(1, ev(1.0)), 100, 1.0, 3)
        self.assertIsNone(a.tick(100, 1.0, young, 3, True))
        self.assertEqual(a.tick(151, 1.51, young, 3, True), 'reject:deadline')
        self.assertIsNotNone(m)


if __name__ == '__main__':
    unittest.main(verbosity=1)
