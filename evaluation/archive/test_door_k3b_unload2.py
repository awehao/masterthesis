#!/usr/bin/env python3
"""door_k3b_unload2 反例（r2：test_door_k3b_unload.py 的版本化複本；U6、U9 依 r2 改寫＋R1–R6 Codex 20261026_025647 必修 A、B）。以下為 r1 說明：
door_k3b_unload 反例（Codex 20261026_023226 §3；**合成資料，不是實跑結果**）。
合成：mode 1 提交 t0＝10.0；斜坡 1 s 20→5 N；接觸力 12→5 N；卸載瞬態仿 door_wide_k3b_30_1 W1e 5→20 N 切換的量級
（門 +0.24°、底盤 3 mm、手臂 0.004 rad，時間常數 0.5 s）。
  U0 正例：RAMPING→SETTLING→UNLOADED，confirmed_fresh；command() 依斜坡；handoff 後停止監看。
  U1 期限：瞬態不收斂（門持續 0.003 rad/s）⇒ timeout（先核期限：期限步即使剛好低速也失敗）。
  U2 失接觸：一指力降到 0.3 N ⇒ contact_lost。
  U3 門速保護：門 0.06 rad/s ⇒ door_rate。
  U4 底盤偏離 25 mm ⇒ base_dev。
  U5 讀回 ≠ 設定、設定 ≠ 斜坡 ⇒ readback_mismatch。
  U6 缺步：缺 30 步 ⇒ 低速窗歸零（UNLOADED 延後）；連續無效 0.25 s ⇒ data_invalid。
  U7 UNLOADED 後低速中斷 ⇒ 回 SETTLING、confirmed_fresh False；之後超過期限 ⇒ timeout。
  U8 重送同步同時 ⇒ dup 不累積；同步異時 ⇒ time_conflict；倒退 ⇒ order。
  U9 task_gate：go 須執行端 UNLOADED 且求解端子狀態 unloaded；wait／abort（任一端失敗、期限＋余裕、缺狀態）。

    python3 evaluation/test_door_k3b_unload.py
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3b_unload2 as U                           # noqa: E402

T0, K0 = 10.0, 1000
B0 = (0.3215, -0.8037, 1.3843)
A0 = [-0.8957, 0.3327, 1.9056, -1.5765, -0.557, 0.0063]
Q0 = 0.5276


def run(dur=6.0, door=None, base=None, arm=None, contact=None, rb=None, setv=None, skip=None, invalid=None, stop_at=None, abort_at=None):
    m = U.ExecUnloadMonitor()
    assert m.start(K0, T0)
    tau = 0.5
    door = door or (lambda t: Q0 + math.radians(0.24) * (1 - math.exp(-max(0.0, t - T0) / tau)))
    base = base or (lambda t: (B0[0] - 0.0022 * (1 - math.exp(-max(0.0, t - T0) / tau)), B0[1] - 0.0024 * (1 - math.exp(-max(0.0, t - T0) / tau)), B0[2]))
    arm = arm or (lambda t: [A0[0], A0[1], A0[2], A0[3] - 0.004 * (1 - math.exp(-max(0.0, t - T0) / tau)), A0[4], A0[5]])
    contact = contact or (lambda t, f: [f * 0.64, f * 0.5])
    for k in range(K0, K0 + int(dur / 0.01)):
        t = round(k * 0.01, 4)
        if skip and skip[0] <= k < skip[1]:
            continue
        f = m.command(t)
        if f is None:
            break
        sv = setv(t, f) if setv else f
        r = rb(t, sv) if rb else [sv, sv]
        ok = not (invalid and invalid[0] <= k < invalid[1])
        if abort_at is not None and t >= abort_at:
            m.abort(k, t, 'k2_latch')
            continue
        m.step(k, t, r, sv, contact(t, f), ok, door(t), base(t), arm(t))
        if stop_at and t >= stop_at:
            break
    return m


class TU(unittest.TestCase):
    def test_u0_positive(self):
        m = run()
        self.assertEqual(m.state, 'UNLOADED', (m.why, m.events, m.worst))
        self.assertLessEqual(m.t['UNLOADED'] - T0, m.r['unload_max_s'])
        self.assertTrue(m.confirmed_fresh(m.last_step)[0])
        self.assertFalse(m.confirmed_fresh(m.last_step + 3)[0])
        self.assertEqual(m.report()['t_settling'], m.t['SETTLING'])
        self.assertAlmostEqual(U.f_cmd(m.r, T0, T0 + 0.5), 12.5)
        self.assertTrue(m.handoff(m.last_step + 1, m.last_t + 0.01))
        self.assertIsNone(m.command(m.last_t + 0.02))

    def test_u1_timeout(self):
        m = run(door=lambda t: Q0 + 0.003 * max(0.0, t - T0))
        self.assertEqual((m.state, m.why), ('FAILED', 'timeout'))
        self.assertAlmostEqual(m.t['FAILED'] - T0, m.r['unload_max_s'], delta=0.011)

    def test_u2_contact_lost(self):
        m = run(contact=lambda t, f: [f * 0.64, 0.3 if t > 10.6 else f * 0.5])
        self.assertEqual((m.state, m.why), ('FAILED', 'contact_lost'))

    def test_u3_door_rate(self):
        m = run(door=lambda t: Q0 + 0.06 * max(0.0, t - 10.3))
        self.assertEqual((m.state, m.why), ('FAILED', 'door_rate'))

    def test_u4_base_dev(self):
        m = run(base=lambda t: (B0[0] + 0.025 * min(1.0, max(0.0, t - T0)), B0[1], B0[2]))
        self.assertEqual((m.state, m.why), ('FAILED', 'base_dev'))

    def test_u5_readback(self):
        m = run(rb=lambda t, sv: [sv, 20.0 if t > 10.4 else sv])
        self.assertEqual((m.state, m.why), ('FAILED', 'readback_mismatch'))
        m = run(setv=lambda t, f: 5.0)                                  # 一次降到 5 N（不依斜坡）
        self.assertEqual((m.state, m.why), ('FAILED', 'readback_mismatch'))

    def test_u6_missing_invalid(self):
        m0 = run()
        m = run(skip=(K0 + 150, K0 + 165))                                   # 缺 15 步（0.15 s ≤ 0.2）⇒ 低速窗歸零、延後
        self.assertEqual(m.state, 'UNLOADED')
        self.assertEqual(m.n['missing'], 15)
        self.assertGreater(m.t['UNLOADED'], m0.t['UNLOADED'])
        m = run(invalid=(K0 + 200, K0 + 230))
        self.assertEqual((m.state, m.why), ('FAILED', 'data_invalid'))

    def test_u7_quiet_lost_after_unloaded(self):
        m0 = run()
        tu = m0.t['UNLOADED']
        m = run(door=lambda t: Q0 + math.radians(0.24) * (1 - math.exp(-max(0.0, t - T0) / 0.5)) + 0.003 * max(0.0, t - (tu + 0.2)))
        self.assertIn('quiet_lost', [e.get('why') for e in m.events])
        self.assertEqual((m.state, m.why), ('FAILED', 'timeout'))

    def test_u8_order(self):
        m = U.ExecUnloadMonitor()
        m.start(K0, T0)
        args = ([20.0, 20.0], 20.0, [12.0, 10.0], True, Q0, B0, A0)
        m.step(K0, T0, *args)
        m.step(K0, T0, *args)
        self.assertEqual((m.n['dup'], m.state), (1, 'RAMPING'))
        m.step(K0, T0 + 0.001, *args)
        self.assertEqual((m.state, m.why), ('FAILED', 'time_conflict'))
        m = U.ExecUnloadMonitor()
        m.start(K0, T0)
        m.step(K0, T0, *args)
        m.step(K0 - 1, T0 + 0.01, *args)
        self.assertEqual((m.state, m.why), ('FAILED', 'order'))

    def test_u9_task_gate(self):
        ok = {'state': 'HELD', 'sub': 'unloaded'}
        st = lambda x: {'run_id': 'r', 'unload': {'state': x}}
        g = lambda state, tn, so=ok, sts=None, sot=None: U.task_gate(state, 10.0, tn, so, run_id='r',
                                                                      state_t=tn - 0.02 if sts is None else sts, solver_t=tn - 0.03 if sot is None else sot)
        self.assertEqual(g(st('SETTLING'), 11.0), 'wait')
        self.assertEqual(g(st('UNLOADED'), 11.0), 'go')
        self.assertEqual(g(st('UNLOADED'), 11.0, {'state': 'HELD', 'sub': 'unload'}), 'wait')
        self.assertEqual(g(st('UNLOADED'), 11.0, None), 'wait')
        self.assertEqual(g(st('UNLOADED'), 11.0, {'state': 'FAIL', 'fail_why': 'x'}), 'abort:solver_fail:x')
        self.assertEqual(g({'run_id': 'r', 'unload': {'state': 'FAILED', 'why': 'contact_lost'}}, 11.0), 'abort:unload_failed:contact_lost')
        self.assertEqual(g(st('SETTLING'), 14.6), 'abort:unload_deadline')
        self.assertEqual(g({}, 11.0), 'wait')

    def test_r1_missing_counts_as_invalid(self):                              # Codex A：缺 30 步約 0.30 s
        m = run(skip=(K0 + 250, K0 + 280))
        self.assertEqual((m.state, m.why), ('FAILED', 'data_invalid'))

    def test_r2_door_short_window(self):                                      # Codex A：初段門 0.06 rad/s 轉 0.30 s 後停
        m = run(door=lambda t: Q0 + 0.06 * min(max(t - T0, 0.0), 0.30))
        self.assertEqual((m.state, m.why), ('FAILED', 'door_rate'))
        self.assertLess(m.t['FAILED'] - T0, 0.25)

    def test_r3_confirmed_fresh(self):                                        # Codex B
        m = run()
        k, t = m.last_step, m.last_t
        self.assertTrue(m.confirmed_fresh(k, t)[0])
        self.assertTrue(m.confirmed_fresh(k + 1, t + 0.01)[0])
        for kk, tt in ((k - 100, None), (k - 1, None), (k + 2, None), (k, float('nan')), (k, t + 0.5), (k, t - 0.5), (True, None)):
            self.assertFalse(m.confirmed_fresh(kk, tt)[0], (kk, tt))

    def test_r4_task_gate_rejects_first(self):                                # Codex B：NaN、遠超期限、他趟、過期
        ok = {'state': 'HELD', 'sub': 'unloaded'}
        good = {'run_id': 'r', 'unload': {'state': 'UNLOADED'}}
        self.assertEqual(U.task_gate(good, 10.0, float('nan'), ok, run_id='r', state_t=11.0, solver_t=11.0), 'abort:time_invalid')
        self.assertEqual(U.task_gate(good, float('nan'), 11.0, ok, run_id='r', state_t=11.0, solver_t=11.0), 'abort:time_invalid')
        self.assertEqual(U.task_gate(good, 10.0, 9.0, ok, run_id='r', state_t=9.0, solver_t=9.0), 'abort:time_invalid')
        self.assertEqual(U.task_gate(good, 10.0, 30.0, ok, run_id='r', state_t=30.0, solver_t=30.0), 'abort:unload_deadline')
        self.assertEqual(U.task_gate(dict(good, run_id='other'), 10.0, 11.0, ok, run_id='r', state_t=11.0, solver_t=11.0), 'wait')
        self.assertEqual(U.task_gate(good, 10.0, 11.0, ok, run_id='r', state_t=10.5, solver_t=11.0), 'wait')     # 執行端狀態過期
        self.assertEqual(U.task_gate(good, 10.0, 11.0, ok, run_id='r', state_t=11.0, solver_t=None), 'wait')     # 求解端狀態缺時刻
        self.assertEqual(U.task_gate(good, 10.0, 11.0, ok, run_id='r', state_t=11.2, solver_t=11.0), 'wait')     # 未來時刻
        self.assertEqual(U.task_gate(good, 10.0, 11.0, ok, run_id='r', state_t=11.0, solver_t=10.95), 'go')

    def test_r5_abort_first(self):
        m = run(abort_at=10.5)
        self.assertEqual(m.state, 'ABORTED')
        self.assertIsNone(m.command(10.6))
        self.assertFalse(m.confirmed_fresh(m.last_step)[0])

    def test_r6_base0_first_valid(self):
        m = U.ExecUnloadMonitor()
        m.start(K0, T0)
        self.assertIsNone(m.base0)
        m.step(K0, T0, [20.0, 20.0], 20.0, [12.0, 10.0], True, Q0, B0, A0)
        self.assertEqual(m.base0, list(B0))

if __name__ == '__main__':
    unittest.main(verbosity=1)
