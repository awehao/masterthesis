#!/usr/bin/env python3
"""door_k3m_pause_sup_p4（主動保持停止）離線反例：
  A1 靜止世界＋底盤保持命令 (2.2, −1.3) mm/s、0.0018 rad/s、手臂 0 ⇒ ZERO_PUB → CONFIRM → STOPPED → 觀察 10 s → DONE；
     STOPPED 後每週期仍發布（publish＝True）；outcome 標 stop_definition＝active_hold。
  A2 同情境用原監督器（命令歸零定義）⇒ 永遠到不了 STOPPED（定義差異的對照）。
  A3 CONFIRM 中底盤套用 6 mm/s（超出保持界 5 mm/s）⇒ 不確認 ⇒ timeout。
  A4 STOPPED 後手臂套用非零 ⇒ old_command_after_stop。
  A5 STOPPED 後底盤實際漂移 2.6 mm/s（p3_1 實測量級）⇒ moved_after_stop。
  A6 STOPPED 後套用來源序號早於保持第一筆或 −1 ⇒ old_command_after_stop。
  A7 hold_cmd_ok 邊界：線速度向量 5 mm/s 內通過、超過拒絕；|ω| 0.02 邊界；手臂 1e-6。

    python3 evaluation/test_door_k3m_pause_sup_p4.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_pause_sup as PSUP                      # noqa: E402
import door_k3m_pause_sup_p4 as P4                     # noqa: E402

HOLD = (0.0022, -0.0013, 0.0018)


def run(sup, t_end=30.0, base_applied=HOLD, arm_after_stop=0.0, drift_after_stop=0.0, src_after_stop=None):
    """物理步 0.01 s、同步 50 Hz、監督週期 0.05 s。回傳 (sup, 每週期回傳列表)。"""
    acts = []
    seq = 100
    x = 0.6
    t_stop = None
    for k in range(int(t_end / 0.01)):
        t = round(k * 0.01, 4)
        if t_stop is None and sup.state == 'STOPPED':
            t_stop = t
        if t_stop is not None:
            x += drift_after_stop * 0.01
        if k % 2 == 0:
            sup.feed_state(k, t, x, -0.53, 1.57, 0.14, [0.0] * 6, wall=t)
        if k % 5 == 0 and k > 0:
            a = sup.cycle(t, True, P4.hold_cmd_ok(HOLD, [0.0] * 6) if isinstance(sup, P4.ActiveHoldSupervisor) else False, wall=t)
            acts.append((t, sup.state, a))
            if a['publish']:
                seq += 1
                sup.note_published(seq, a['zero'])
        if sup.state in ('CONFIRM', 'STOPPED'):
            arm = [arm_after_stop] + [0.0] * 5 if sup.state == 'STOPPED' else [0.0] * 6
            src = src_after_stop if (sup.state == 'STOPPED' and src_after_stop is not None) else seq
            sup.applied(k, t, list(base_applied), arm, src)
        if sup.state in ('DONE', 'FAIL'):
            break
    return sup, acts


class T(unittest.TestCase):
    def test_a1_active_hold_done(self):
        sup, acts = run(P4.ActiveHoldSupervisor())
        o = sup.outcome()
        self.assertEqual(o['state'], 'DONE', o)
        self.assertTrue(o['command_stop_confirmed'] and o['physical_stop_confirmed'])
        self.assertEqual(o['stop_definition'], 'active_hold')
        stopped = [a for t, st, a in acts if st == 'STOPPED']
        self.assertTrue(stopped and all(a['publish'] and a['zero'] for a in stopped))   # 觀察期持續發布保持命令
        self.assertGreaterEqual(o['observation_covered_s'], 10.0 - 1e-9)

    def test_a2_original_definition_never_stops(self):
        sup, _ = run(PSUP.PauseSupervisor())
        o = sup.outcome()
        self.assertNotEqual(o['state'], 'DONE')
        self.assertFalse(o['command_stop_confirmed'])

    def test_a3_base_over_hold_bound(self):
        sup, _ = run(P4.ActiveHoldSupervisor(), base_applied=(0.006, 0.0, 0.0))
        o = sup.outcome()
        self.assertEqual((o['state'], o['fail_why']), ('FAIL', 'timeout'), o)

    def test_a4_arm_nonzero_after_stop(self):
        sup, _ = run(P4.ActiveHoldSupervisor(), arm_after_stop=0.001)
        self.assertEqual(sup.outcome()['fail_why'], 'old_command_after_stop')

    def test_a5_drift_after_stop(self):
        sup, _ = run(P4.ActiveHoldSupervisor(), drift_after_stop=0.0026)
        self.assertEqual(sup.outcome()['fail_why'], 'moved_after_stop')

    def test_a6_old_or_timeout_source_after_stop(self):
        for src in (5, -1):
            sup, _ = run(P4.ActiveHoldSupervisor(), src_after_stop=src)
            self.assertEqual(sup.outcome()['fail_why'], 'old_command_after_stop', src)

    def test_a7_hold_bounds(self):
        z = [0.0] * 6
        self.assertTrue(P4.hold_cmd_ok((0.003, 0.004, 0.0), z))            # 向量 5 mm/s
        self.assertFalse(P4.hold_cmd_ok((0.0036, 0.0036, 0.0), z))         # 向量 5.09 mm/s
        self.assertTrue(P4.hold_cmd_ok((0.0, 0.0, 0.02), z))
        self.assertFalse(P4.hold_cmd_ok((0.0, 0.0, 0.0201), z))
        self.assertFalse(P4.hold_cmd_ok((0.0, 0.0, 0.0), [2e-6] + [0.0] * 5))
        self.assertFalse(P4.hold_cmd_ok((0.0, float('nan'), 0.0), z))


if __name__ == '__main__':
    unittest.main(verbosity=2)
