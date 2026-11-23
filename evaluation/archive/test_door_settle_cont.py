#!/usr/bin/env python3
"""door_settle_cont 最小核對：准入判定（有效／新鮮／armed／連續低力）與 SETTLE 當輪 contact_min_n 換值後還原。"""
import dataclasses
import math
import types
import unittest

import door_settle_cont as SC


@dataclasses.dataclass
class Cfg:
    contact_min_n: float = 0.5
    finger_loss_s: float = 0.2
    stale_s: float = 0.5


def meas(cur=(0.0, 0.0), mx=(0.0, 0.0), armed=True, age=0.05, fbm=(0.0, 3.0)):
    return types.SimpleNamespace(finger_low_cont=cur, finger_low_cont_max=mx, finger_loss_armed=armed, contacts_age=age, finger_bar_min=fbm)


class T(unittest.TestCase):
    def test_ok_despite_window_min_zero(self):
        ok, d = SC.settle_contact_ok(meas(cur=(0.12, 0.0), mx=(0.15, 0.0)), Cfg())
        self.assertTrue(ok)
        self.assertIsNone(d['why'])

    def test_missing_invalid_stale_unarmed(self):
        c = Cfg()
        for m, why in ((meas(cur=None), 'invalid'), (meas(mx=(math.nan, 0.0)), 'invalid'), (meas(age=0.9), 'stale'),
                       (meas(age=math.inf), 'stale'), (meas(armed=None), 'not_armed'), (meas(armed=False), 'not_armed'),
                       (meas(cur=(0.2, 0.0)), 'low_load_cont'), (meas(mx=(0.0, 0.25)), 'low_load_cont')):
            ok, d = SC.settle_contact_ok(m, c)
            self.assertFalse(ok)
            self.assertEqual(d['why'], why)

    def test_mixin_swaps_and_restores(self):
        seen = []

        class Base:
            def __init__(self):
                self.c, self.phase, self.settle_last = Cfg(), 'SETTLE', None

            def step(self, m, dt):
                seen.append(self.c.contact_min_n)
                self.settle_last = {'contact_ok': min(m.finger_bar_min) >= self.c.contact_min_n}
                return 'act'

        P = SC.settle_cont(Base)()
        P.step(meas(), 0.01)
        self.assertEqual(seen[-1], -math.inf)
        self.assertTrue(P.settle_last['contact_ok'])
        self.assertEqual(P.settle_last['contact_semantics'], SC.SEMANTICS)
        P.step(meas(armed=False, fbm=(3.0, 3.0)), 0.01)
        self.assertEqual(seen[-1], math.inf)
        self.assertFalse(P.settle_last['contact_ok'])
        self.assertEqual(P.c.contact_min_n, 0.5)
        P.phase = 'OPEN'
        P.step(meas(), 0.01)
        self.assertEqual(seen[-1], 0.5)


if __name__ == '__main__':
    unittest.main()
