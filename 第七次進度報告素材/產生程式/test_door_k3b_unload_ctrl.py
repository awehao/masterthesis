#!/usr/bin/env python3
"""door_k3b_unload_ctrl 反例（假控制器，沿用 test_door_k3b_finger_restore.Ctrl）。
  C1 暫停切換 5→20 後依斜坡逐步設定 20→5：每步 ok、讀回＝設定；之後 periodic 以 5 核對通過；restore 紀錄未被寫入。
  C2 斜坡中 periodic（每 1 s）以當下設定核對，不誤判 reverted。
  C3 只改到一指 ⇒ finger_readback_not_set、參考值不變；C4 非手指上限被改 ⇒ non_finger_max_effort_changed；
  C5 set 拋例外 ⇒ ok False；C6 無暫停切換紀錄 ⇒ no_pause_switch_record。

    python3 evaluation/test_door_k3b_unload_ctrl.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3b_unload2 as KU                          # noqa: E402
import door_k3b_unload_ctrl as UC                      # noqa: E402
import door_k3m_finger_effort as FEFF                  # noqa: E402
import test_door_k3b_finger_restore as TF              # noqa: E402

IDX = TF.IDX


def paused(mode='ok'):
    c = TF.Ctrl(mode)
    fe = FEFF.PauseFingerEffort(20.0, IDX, ['finger_joint1', 'finger_joint2'])
    r = fe.on_pause_switch(c, 100, 1.0)
    assert r['ok'] is True, r
    return c, fe


class TC(unittest.TestCase):
    def test_c1_c2(self):
        c, fe = paused()
        r = KU.REG_UNLOAD
        for k in range(0, 201):
            t = 10.0 + k * 0.01
            v = KU.f_cmd(r, 10.0, t)
            o = UC.set_ramp(fe, c, v, 1000 + k, t)
            self.assertTrue(o['ok'], o)
            self.assertEqual(o['readback'], [v, v])
            p = fe.periodic(c, 1000 + k, t)
            if p is not None:
                self.assertTrue(p['ok'], p)
        self.assertEqual(fe.value, 5.0)
        self.assertIsNone(fe.record.get('restore'))
        p = fe.periodic(c, 2000, 30.0)
        self.assertTrue(p['ok'] and p['readback'] == [5.0, 5.0], p)

    def test_c3_c6(self):
        c, fe = paused('one')
        o = UC.set_ramp(fe, c, 19.0, 1, 10.0)
        self.assertEqual((o['ok'], o['why'], fe.value), (False, 'finger_readback_not_set', 20.0))
        c, fe = paused('other')
        self.assertEqual(UC.set_ramp(fe, c, 19.0, 1, 10.0)['why'], 'non_finger_max_effort_changed')
        c, fe = paused('raise')
        o = UC.set_ramp(fe, c, 19.0, 1, 10.0)
        self.assertFalse(o['ok'])
        self.assertTrue(o['why'].startswith('set_max_efforts 例外'))
        fe0 = FEFF.PauseFingerEffort(20.0, IDX, ['finger_joint1', 'finger_joint2'])
        self.assertEqual(UC.set_ramp(fe0, TF.Ctrl(), 19.0, 1, 10.0)['why'], 'no_pause_switch_record')


if __name__ == '__main__':
    unittest.main(verbosity=1)
