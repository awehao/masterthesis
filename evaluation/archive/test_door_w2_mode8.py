#!/usr/bin/env python3
"""door_w2_mode8 反例（Codex 20261026_025647 §1、§3B、§4；**合成資料**）。
驅動：run_held 進 HELD 後腳本化續跑；手臂為 door_wide_k3b_30_1 HELD 實測姿態加 j2 擾動，TCP 以 door_k2_check._tcp 計算。
  X1 Codex §1：r7 以斜坡期間的低速累積、斜坡一結束即 unloaded；r8 須得知執行端 t_settling 後才累積 ⇒ t_unloaded ≥ t_settling＋0.5。
  X2 一直未得知 t_settling ⇒ unload_timeout。X3 t_settling 早於 t_c1＋ramp_s−0.02、或晚於現在 ⇒ 拒收（不採用）。
  X4 卸載中 TCP 緩慢偏移 > 5 mm ⇒ unload_transient:displacement。X5 卸載中 TCP 速度 > 0.02 m/s ⇒ unload_transient:tcp_speed。
  X6 張爪後手臂回彈（j2 0.004 rad、τ 0.2 s）收斂 ⇒ release_settled，之後交接成功。
  X7 回彈不收斂（j2 持續 0.002 rad/s）⇒ release_transient:timeout。
  X8 release_open 中交接 ⇒ handover_before_transient_converged；未得知張爪即交接 ⇒ handover_without_release。
  X9 release_settled 後手臂再動 ⇒ moved_in_release_settled。X10 release_open 中底盤動 ⇒ moved_in_release_open。
  X11 mode 2 通知延遲 0.1 s 期間手臂已回彈：暫定瞬態監看容許 ⇒ 切換、交接；暫緩中手臂過快（0.2 rad/s）⇒ moved_in_held。
  A1 ModeAuthority8：unload 啟用缺 gate2 ⇒ 建構 ValueError；gate2 存在 ⇒ 可建構。

    python3 evaluation/test_door_w2_mode8.py
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402
import door_k3m_pause_sup_p4 as P4                     # noqa: E402
import door_w2_mode7 as M7                             # noqa: E402
import door_w2_mode8 as M8                             # noqa: E402
import test_door_w2_mode as TM                         # noqa: E402

A0 = [-0.8957, 0.3327, 1.9056, -1.5765, -0.557, 0.0063]
B0 = (0.6, -0.53, 1.57)
TCP = lambda b, a: KC._tcp(b, a)                        # noqa: E731
TC1 = 30.505


def held(cls=M8.HeldSupervisorP4b8, unload=True):
    kw = dict(tcp_fn=TCP) if cls is M8.HeldSupervisorP4b8 else {}
    sup = cls(arm_win_s=0.5, held_max_s=60.0, unload=unload, **kw)
    seq, t = 100, 0.0
    for k in range(3000):                                # 同 test_door_w2_mode.run_held，但手臂用實測姿態
        t = round(k * 0.01, 4)
        if k % 2 == 0:
            sup.feed_state(k, t, B0[0], B0[1], B0[2], 0.14, list(A0), wall=t)
        if k % 5 == 0 and k > 0:
            a = sup.cycle(t, True, P4.hold_cmd_ok(TM.HOLD, [0.0] * 6), wall=t)
            if a['publish']:
                seq += 1
                sup.note_published(seq, a['zero'])
        if sup.state in ('CONFIRM', 'STOPPED', 'HELD'):
            sup.applied(k, t, list(TM.HOLD), [0.0] * 6, seq)
    assert sup.state == 'HELD', sup.outcome()
    return sup


def drive(sup, t0, dur, j2=lambda t: 0.0, bx=lambda t: B0[0], door=lambda t: 0.14, pend=(), commits=(), settle=(), handover_at=None):
    seq = 20_000
    res = {'ok': {}, 'handed': None, 'first_nopub': None, 'settle_ok': []}
    done = set()
    for kk in range(int(round(t0 / 0.01)), int(round((t0 + dur) / 0.01))):
        t = round(kk * 0.01, 4)
        for tp, k in pend:
            if t >= tp and ('p', k) not in done:
                done.add(('p', k))
                sup.set_pending(k, True, t)
        for tk, k, tc in commits:
            if t >= tk and ('c', k) not in done and sup.state == 'HELD':
                done.add(('c', k))
                res['ok'][k] = sup.on_commit(k, tc, t)
        for tk, ts in settle:
            if t >= tk and ('s', tk) not in done:
                done.add(('s', tk))
                res['settle_ok'].append(sup.set_exec_settling(ts, t))
        if handover_at is not None and t >= handover_at and res['handed'] is None and sup.state == 'HELD':
            res['handed'] = sup.handover(t)
        if kk % 2 == 0:
            arm = list(A0)
            arm[1] += j2(t)
            sup.feed_state(kk, t, bx(t), B0[1], B0[2], door(t), arm, wall=t)
        if kk % 5 == 0:
            a = sup.cycle(t, True, P4.hold_cmd_ok(TM.HOLD, [0.0] * 6), wall=t)
            if a['publish']:
                seq += 1
                sup.note_published(seq, a['zero'])
            elif res['first_nopub'] is None:
                res['first_nopub'] = t
        if sup.state == 'HELD':
            sup.applied(kk, t, list(TM.HOLD), [0.0] * 6, seq)
        if sup.state in ('DONE', 'FAIL', 'HANDED_OVER'):
            break
    res['outcome'] = sup.outcome(t_end=t)
    return res


def tr(t0, amp, tau, t):
    return amp * (1 - math.exp(-max(0.0, t - t0) / tau))


class TX(unittest.TestCase):
    def test_x1_quiet_after_settling(self):
        s7 = held(M7.HeldSupervisorP4b7)
        drive(s7, 30.0, 3.0, pend=[(30.3, 1)], commits=[(30.52, 1, TC1)])
        s8 = held()
        r = drive(s8, 30.0, 3.0, pend=[(30.3, 1)], commits=[(30.52, 1, TC1)], settle=[(31.53, 31.505)])
        self.assertEqual((s7.sub, s8.sub), ('unloaded', 'unloaded'), r['outcome'])
        self.assertLess(s7.t_unloaded, TC1 + 1.0 + 0.06)                    # r7：斜坡一結束即 unloaded
        self.assertGreaterEqual(s8.t_unloaded, 31.505 + 0.5 - 1e-6)          # r8：t_settling 後才累積

    def test_x2_x3_settling(self):
        s = held()
        drive(s, 30.0, 5.0, pend=[(30.3, 1)], commits=[(30.52, 1, TC1)])
        self.assertEqual(s.seq.why, 'unload_timeout')
        s = held()
        r = drive(s, 30.0, 2.0, pend=[(30.3, 1)], commits=[(30.52, 1, TC1)], settle=[(31.0, 31.0), (31.6, 31.9)])
        self.assertEqual(r['settle_ok'], [False, False])
        self.assertIsNone(s.t_settling)

    def test_x4_x5_unload_transient(self):
        s = held()
        drive(s, 30.0, 4.0, j2=lambda t: 0.012 * max(0.0, t - TC1), pend=[(30.3, 1)], commits=[(30.52, 1, TC1)], settle=[(31.53, 31.505)])
        self.assertEqual(s.seq.why, 'unload_transient:displacement', s.outcome()['transient'])
        s = held()
        drive(s, 30.0, 2.0, j2=lambda t: 0.2 * max(0.0, t - 30.8), pend=[(30.3, 1)], commits=[(30.52, 1, TC1)])
        self.assertEqual(s.seq.why, 'unload_transient:tcp_speed', s.outcome()['transient'])

    def unloaded(self):
        s = held()
        drive(s, 30.0, 2.5, pend=[(30.3, 1)], commits=[(30.52, 1, TC1)], settle=[(31.53, 31.505)])
        assert s.sub == 'unloaded', s.outcome()
        return s

    def test_x6_release_settles_then_handover(self):
        s = self.unloaded()
        tc2 = 32.705
        r = drive(s, 32.5, 3.0, j2=lambda t: tr(tc2, 0.004, 0.2, t), door=lambda t: 0.14 + max(0.0, t - tc2) * 0.0175,
                  pend=[(32.55, 2)], commits=[(tc2 + 0.05, 2, tc2)], handover_at=34.6)
        o = r['outcome']
        self.assertTrue(r['ok'][2], o['checks'])
        self.assertTrue(r['handed'], o)
        self.assertEqual(o['state'], 'HANDED_OVER')
        self.assertIsNotNone(o['transitions']['RELEASE_SETTLED'])
        self.assertLessEqual(o['transient']['2']['worst']['disp_m'], 0.005)

    def test_x7_release_no_converge(self):
        s = self.unloaded()
        tc2 = 32.705
        drive(s, 32.5, 3.0, j2=lambda t: 0.002 * max(0.0, t - tc2), pend=[(32.55, 2)], commits=[(tc2 + 0.05, 2, tc2)])
        self.assertEqual(s.seq.why, 'release_transient:timeout')

    def test_x8_handover_rules(self):
        s = self.unloaded()
        tc2 = 32.705
        drive(s, 32.5, 1.0, j2=lambda t: tr(tc2, 0.004, 0.2, t), pend=[(32.55, 2)], commits=[(tc2 + 0.05, 2, tc2)], handover_at=32.9)
        self.assertEqual(s.seq.why, 'handover_before_transient_converged')
        s = held(unload=False)
        self.assertFalse(s.handover(30.1))
        self.assertEqual(s.seq.why, 'handover_without_release')

    def test_x9_x10(self):
        s = self.unloaded()
        tc2 = 32.705
        drive(s, 32.5, 4.0, j2=lambda t: tr(tc2, 0.004, 0.2, t) + 0.01 * max(0.0, t - 35.0), pend=[(32.55, 2)], commits=[(tc2 + 0.05, 2, tc2)])
        self.assertEqual(s.seq.why, 'moved_in_release_settled')
        s = self.unloaded()
        drive(s, 32.5, 2.0, bx=lambda t: B0[0] + 0.002 * max(0.0, t - 33.0), pend=[(32.55, 2)], commits=[(tc2 + 0.05, 2, tc2)])
        self.assertEqual(s.seq.why, 'moved_in_release_open')

    def test_x11_defer_with_springback(self):
        s = self.unloaded()
        tc2 = 32.705
        r = drive(s, 32.5, 3.0, j2=lambda t: tr(tc2, 0.004, 0.2, t), door=lambda t: 0.14 + max(0.0, t - tc2) * 0.0175,
                  pend=[(32.55, 2)], commits=[(tc2 + 0.1, 2, tc2)], handover_at=34.6)
        self.assertTrue(r['ok'][2], r['outcome']['checks'])
        self.assertTrue(any(x.get('defer_start') == 2 for x in r['outcome']['log7']))
        self.assertEqual(r['outcome']['state'], 'HANDED_OVER')
        s = self.unloaded()
        drive(s, 32.5, 1.0, j2=lambda t: 0.2 * max(0.0, t - tc2), pend=[(32.55, 2)], commits=[(tc2 + 0.1, 2, tc2)])
        self.assertEqual(s.seq.why, 'moved_in_held')                         # 暫緩中手臂過快 ⇒ 不容許

    def test_a1_authority(self):
        with self.assertRaises(ValueError):
            M8.ModeAuthority8('r', unload=True)
        a = M8.ModeAuthority8('r', unload=True, gate2=lambda step: True)
        self.assertEqual(a._gate2(1), (True, None))
        a.gate2 = None
        self.assertEqual(a._gate2(1), (False, 'gate2_missing'))
        with self.assertRaises(ValueError):
            M8.HeldSupervisorP4b8(arm_win_s=0.5, held_max_s=60.0)


if __name__ == '__main__':
    unittest.main(verbosity=1)
