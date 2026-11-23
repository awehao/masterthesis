#!/usr/bin/env python3
"""door_k3m_pause_sup_p4b（主動保持停止＋手臂 0.5 s 窗口）離線反例：
  B1 手臂原地微抖（各關節 ±0.1 mrad、約 7 Hz、淨位移 0；p4_1 實測量級）：p4（0.1 s）⇒ 失敗；p4b ⇒ DONE。
  B2 手臂持續移動 2e-3 rad/s ⇒ p4b 不低速（timeout），STOPPED 後才開始移動 ⇒ moved_after_stop。
  B3 手臂蠕變 0.8e-3 rad/s（低於門檻）⇒ p4b DONE。
  B4 底盤與門仍用 0.1 s：STOPPED 後底盤漂移 2.6 mm/s ⇒ moved_after_stop（與 p4 相同，手臂窗口不影響底盤）。
  B5 手臂窗口所需 0.5 s 內有斷點 ⇒ 該時刻無值；arm_win_s 範圍檢查。

    python3 evaluation/test_door_k3m_pause_sup_p4b.py
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_pause_sup_p4 as P4                     # noqa: E402
import door_k3m_pause_sup_p4b as P4B                   # noqa: E402

HOLD = (0.0022, -0.0013, 0.0018)


def run(sup, arm_fn, t_end=30.0, base_drift_after_stop=0.0, arm_from_stop=False):
    seq, x, t_stop = 100, 0.6, None
    for k in range(int(t_end / 0.01)):
        t = round(k * 0.01, 4)
        if t_stop is None and sup.state == 'STOPPED':
            t_stop = t
        if t_stop is not None:
            x += base_drift_after_stop * 0.01
        if k % 2 == 0:
            q = arm_fn(t - (t_stop if (arm_from_stop and t_stop is not None) else (1e9 if arm_from_stop else 0.0)))
            sup.feed_state(k, t, x, -0.53, 1.57, 0.14, [q] * 6, wall=t)
        if k % 5 == 0 and k > 0:
            a = sup.cycle(t, True, P4.hold_cmd_ok(HOLD, [0.0] * 6), wall=t)
            if a['publish']:
                seq += 1
                sup.note_published(seq, a['zero'])
        if sup.state in ('CONFIRM', 'STOPPED'):
            sup.applied(k, t, list(HOLD), [0.0] * 6, seq)
        if sup.state in ('DONE', 'FAIL'):
            break
    return sup.outcome()


def jitter(t):
    return 1e-4 * math.sin(2 * math.pi * 7.0 * t)


class T(unittest.TestCase):
    def test_b1_jitter(self):
        o4 = run(P4.ActiveHoldSupervisor(), jitter)
        self.assertNotEqual(o4['state'], 'DONE')
        o = run(P4B.ActiveHoldSupervisorP4b(), jitter)
        self.assertEqual(o['state'], 'DONE', o)
        self.assertTrue(o['command_stop_confirmed'] and o['physical_stop_confirmed'])
        self.assertEqual((o['arm_window_s'], o['base_door_window_s']), (0.5, 0.1))

    def test_b2_steady_motion(self):
        o = run(P4B.ActiveHoldSupervisorP4b(), lambda t: 2e-3 * t)
        self.assertEqual((o['state'], o['fail_why']), ('FAIL', 'timeout'), o)
        o = run(P4B.ActiveHoldSupervisorP4b(), lambda t: 2e-3 * max(0.0, t), arm_from_stop=True)
        self.assertEqual(o['fail_why'], 'moved_after_stop', o)

    def test_b3_creep_below_threshold(self):
        o = run(P4B.ActiveHoldSupervisorP4b(), lambda t: 0.8e-3 * t)
        self.assertEqual(o['state'], 'DONE', o)

    def test_b4_base_still_0p1(self):
        o = run(P4B.ActiveHoldSupervisorP4b(), lambda t: 0.0, base_drift_after_stop=0.0026)
        self.assertEqual(o['fail_why'], 'moved_after_stop')

    def test_b5_window_rules(self):
        ws = P4B.WindowSpeedP4b()
        for k in range(0, 42, 2):
            ws.feed(k, k * 0.01, 0.6, 0.0, 0.0, 0.1, [0.0] * 6, wall=k * 0.01)
        self.assertIsNone(ws.speeds(0.40, 0.40))                        # 只有 0.4 s 資料 ⇒ 手臂 0.5 s 窗口不足、無值
        self.assertIsNotNone(P4.PSUP.WindowSpeed.speeds(ws, 0.40, 0.40))  # 0.1 s 窗口有值
        ws = P4B.WindowSpeedP4b()
        for k in range(0, 80, 2):
            if k == 50:
                ws.feed(k, k * 0.01, float('nan'), 0.0, 0.0, 0.1, [0.0] * 6, wall=k * 0.01)   # 斷點
                continue
            ws.feed(k, k * 0.01, 0.6, 0.0, 0.0, 0.1, [0.0] * 6, wall=k * 0.01)
        self.assertIsNone(ws.speeds(0.78, 0.78))                        # 0.5 s 窗口跨過斷點 ⇒ 無值
        self.assertIsNotNone(P4.PSUP.WindowSpeed.speeds(ws, 0.78, 0.78))  # 0.1 s 窗口仍有值
        with self.assertRaises(ValueError):
            P4B.WindowSpeedP4b(arm_win_s=2.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
