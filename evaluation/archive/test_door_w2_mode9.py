#!/usr/bin/env python3
"""door_w2_mode9／door_k3b_transient2 反例（Codex 20261026_034842 §2、§8-1；**合成資料**）。
  Y1 Codex 重現：0.90 s 前低速、0.90–1.06 s 連續無效、1.06 s 恢復 ⇒ r1 立即 CONVERGED；r2 須重新成窗＋0.5 s 才收斂；
     無效持續 0.25 s ⇒ data_invalid。
  Y2 Codex 重現：mode 2 提交後、得知前（無週期介入）手臂 6 mm 位移後回復 ⇒ r8 切換後 worst 0、接受；r9 重放 ⇒ FAIL release_transient:*。
  Y3 監督器 r9 量測失效送入瞬態：release_open 中短暫非有限同步 ⇒ 收斂延後（不沿用舊計時）。
  Y4 邊界：j2 以等速移 0.009 rad（TCP 約 3.9 mm，界內），0.44 s 停 ⇒ 1.5 s 內 CONVERGED；0.56 s 停 ⇒ 收斂需 > 1.5 s ⇒ timeout（期限不放寬）。
  R  回歸：test_door_w2_mode8 全部反例改以 HeldSupervisorP4b9 執行。

    python3 evaluation/test_door_w2_mode9.py
"""
import math
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3b_transient as TT1                       # noqa: E402
import door_k3b_transient2 as TT2                      # noqa: E402
import door_w2_mode8 as M8                             # noqa: E402
import door_w2_mode9 as M9                             # noqa: E402
import test_door_w2_mode8 as T8                        # noqa: E402

A0, B0, TCP = T8.A0, T8.B0, T8.TCP
_held8 = T8.held


def held9(cls=M9.HeldSupervisorP4b9, unload=True):
    """test_door_w2_mode8.held（凍結，不改）只在類別恰為 P4b8 時給 tcp_fn；r9 以子類別判斷。"""
    if isinstance(cls, type) and issubclass(cls, M8.HeldSupervisorP4b8) and cls is not M8.HeldSupervisorP4b8:
        orig = M8.HeldSupervisorP4b8
        try:
            T8.M8 = types.SimpleNamespace(HeldSupervisorP4b8=cls)
            return _held8(cls, unload)
        finally:
            T8.M8 = M8
    return _held8(cls, unload)


def feed_seq(tm, t0, dur, j2=lambda t: 0.0, invalid=None, dt=0.02):
    tm.start(t0, B0, A0)
    out = None
    k = 1
    while k * dt <= dur + 1e-9:
        t = round(t0 + k * dt, 4)
        if invalid and invalid[0] <= t - t0 < invalid[1]:
            tm.feed(t, (math.nan,) * 3, [math.nan] * 6)
        else:
            a = list(A0)
            a[1] += j2(t)
            tm.feed(t, B0, a)
        if tm.state != 'ACTIVE' and out is None:
            out = (t, tm.state, tm.why)
        k += 1
    return out


class TY(unittest.TestCase):
    def test_y1_invalid_breaks_quiet(self):
        r1 = feed_seq(TT1.TransientMonitor(TCP, 4.0), 10.0, 2.0, invalid=(0.90, 1.06))
        r2 = feed_seq(TT2.TransientMonitor(TCP, 4.0), 10.0, 3.0, invalid=(0.90, 1.06))
        self.assertEqual(r1[1], 'CONVERGED')
        self.assertLessEqual(r1[0], 10.0 + 1.08)                              # r1：恢復後立即沿用舊計時
        self.assertEqual(r2[1], 'CONVERGED')
        self.assertGreaterEqual(r2[0], 10.0 + 1.06 + 0.5 + 0.5 - 1e-6)        # r2：重新成窗（0.5 s 手臂窗）＋0.5 s 持續
        r3 = feed_seq(TT2.TransientMonitor(TCP, 4.0), 10.0, 2.0, invalid=(0.90, 1.16))
        self.assertEqual(r3[1:], ('FAILED', 'data_invalid'))

    def test_y2_replay_after_late_notice(self):
        def run(cls):
            s = held9(cls)
            T8.drive(s, 30.0, 2.5, pend=[(30.3, 1)], commits=[(30.52, 1, T8.TC1)], settle=[(31.53, 31.505)])
            assert s.sub == 'unloaded', s.outcome()
            tc2 = 32.705
            s.set_pending(2, True, 32.6)
            k0 = int(round(32.5 / 0.01))
            # 提交後 0.20 s 內手臂 j2 去回（峰值約 6 mm），期間不呼叫 cycle（得知在週期之前）
            for kk in range(k0, int(round((tc2 + 0.20) / 0.01))):
                t = round(kk * 0.01, 4)
                if kk % 2 == 0:
                    a = list(A0)
                    x = max(0.0, t - tc2)
                    a[1] += 0.02 * math.sin(math.pi * min(x, 0.18) / 0.18) if x > 0 else 0.0
                    s.feed_state(kk, t, B0[0], B0[1], B0[2], 0.14, a, wall=t)
            ok = s.on_commit(2, tc2, tc2 + 0.20)
            return s, ok
        s8, ok8 = run(M8.HeldSupervisorP4b8)
        self.assertTrue(ok8)
        self.assertEqual(s8.tm[2].worst['disp_m'], 0.0)                      # r8：重新初始化、漏掉
        s9, ok9 = run(M9.HeldSupervisorP4b9)
        self.assertFalse(ok9)
        self.assertTrue(str(s9.seq.why).startswith('release_transient:'), s9.seq.why)
        self.assertGreater(s9.checks[2]['replay_n'], 5)

    def test_y3_invalid_sync_delays_convergence(self):
        def run(gap):
            s = held9(M9.HeldSupervisorP4b9)
            T8.drive(s, 30.0, 2.5, pend=[(30.3, 1)], commits=[(30.52, 1, T8.TC1)], settle=[(31.53, 31.505)])
            tc2 = 32.705
            r = T8.drive(s, 32.5, 3.0, j2=lambda t: T8.tr(tc2, 0.004, 0.2, t), pend=[(32.55, 2)], commits=[(tc2 + 0.05, 2, tc2)])
            return s, r
        s, r = run(None)
        t_ok = s.outcome()['transitions']['RELEASE_SETTLED']
        self.assertIsNotNone(t_ok)
        s2 = held9(M9.HeldSupervisorP4b9)
        T8.drive(s2, 30.0, 2.5, pend=[(30.3, 1)], commits=[(30.52, 1, T8.TC1)], settle=[(31.53, 31.505)])
        tc2 = 32.705
        T8.drive(s2, 32.5, 0.7, j2=lambda t: T8.tr(tc2, 0.004, 0.2, t), pend=[(32.55, 2)], commits=[(tc2 + 0.05, 2, tc2)])
        s2.tm[2].feed(33.21, (math.nan,) * 3, [math.nan] * 6)                 # 短暫失效（送入瞬態監看）
        T8.drive(s2, 33.2, 2.8, j2=lambda t: T8.tr(tc2, 0.004, 0.2, t))
        t2 = s2.outcome()['transitions']['RELEASE_SETTLED']
        self.assertTrue(t2 is None or t2 > t_ok, (t_ok, t2, s2.seq.why))

    def test_y4_boundary_release_deadline(self):
        def run(t_stop):
            tm = TT2.TransientMonitor(TCP, TT2.T_MAX_RELEASE)
            return feed_seq(tm, 20.0, 2.5, j2=lambda t: 0.009 * min(max(t - 20.0, 0.0), t_stop) / t_stop)
        ok = run(0.44)
        late = run(0.56)
        self.assertEqual(ok[1], 'CONVERGED', ok)
        self.assertLessEqual(ok[0] - 20.0, 1.5)
        self.assertEqual(late[1:], ('FAILED', 'timeout'), late)
        self.assertAlmostEqual(late[0] - 20.0, 1.5, delta=0.021)


class R9(unittest.TestCase):
    """test_door_w2_mode8 的全部反例，改以 r9 執行。"""

    def test_mode8_cases_on_r9(self):
        shim = types.SimpleNamespace(**{k: getattr(M8, k) for k in dir(M8) if not k.startswith('__')})
        shim.HeldSupervisorP4b8 = M9.HeldSupervisorP4b9
        orig = T8.M8
        T8.M8 = shim
        try:
            orig_held = T8.held
            T8.held = lambda cls=M9.HeldSupervisorP4b9, unload=True: held9(M9.HeldSupervisorP4b9 if cls is M8.HeldSupervisorP4b8 else cls, unload)
            res = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w')).run(unittest.defaultTestLoader.loadTestsFromModule(T8))
        finally:
            T8.M8 = orig
            T8.held = orig_held
        self.assertTrue(res.wasSuccessful(), [str(x[1])[-300:] for x in res.failures + res.errors])
        self.assertEqual(res.testsRun, 9)


if __name__ == '__main__':
    unittest.main(verbosity=1)
