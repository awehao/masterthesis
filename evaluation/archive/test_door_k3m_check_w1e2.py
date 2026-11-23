#!/usr/bin/env python3
"""door_k3m_check_w1e2 測試：room_run 兩位小數時間＋執行端 float32 時間（差約 1.2e-6）⇒ w1e 誤拒、w1e2 接受；
Codex 三反例（重複 periodic、切換時間 99999、索引 [0,1]）與其他反例在 w1e2 下仍全部不通過；時間差 1e-5（超過 5e-6）仍拒。

    python3 evaluation/test_door_k3m_check_w1e2.py
"""
import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_k3m_check_w1e as E                         # noqa: E402
import door_k3m_check_w1e2 as E2                       # noqa: E402
import test_door_k3m_finger_effort as T                # noqa: E402

DT = -1.2e-6


def room_real():
    """仿實錄：room steps 兩位小數；執行端紀錄 t＝step/100 − 1.2e-6。"""
    r = T.room_ok()
    pz = r['finger_effort']['pause']
    pz['t'] = pz['step'] / 100.0 + DT
    r['k3m']['pause_switch']['switch_t'] = pz['t']
    for e in pz['periodic']:
        e['t'] = e['step'] / 100.0 + DT
    r['steps'] = [[s, round(s / 100.0, 2)] for s in range(5000, 6201)]
    return r


def chk(mod, room):
    w, st = [], {}
    orig = E._at_room
    if mod is E2:
        E._at_room = E2._at_room_r2
    try:
        return E.check_finger_effort(room, T.EC, w, st), w
    finally:
        E._at_room = orig


class T2(unittest.TestCase):
    def test_w1e_false_reject_w1e2_accept(self):
        ok1, w1 = chk(E, room_real())
        self.assertFalse(ok1)
        self.assertIn('finger_effort_not_at_pause_switch', w1)
        ok2, w2 = chk(E2, room_real())
        self.assertTrue(ok2, w2)

    def test_tolerance_bound(self):
        r = room_real()
        r['finger_effort']['pause']['periodic'][3]['t'] += 1e-5
        ok, w = chk(E2, r)
        self.assertIn('finger_effort_periodic_invalid_1', w)

    def test_codex_counterexamples_still_rejected(self):
        r = room_real()
        P = r['finger_effort']['pause']
        P['periodic'] = [copy.deepcopy(P['periodic'][0]) for _ in range(7)]
        ok, w = chk(E2, r)
        self.assertFalse(ok)
        r = room_real()
        r['finger_effort']['pause']['t'] = 99999.0
        self.assertIn('finger_effort_not_at_pause_switch', chk(E2, r)[1])
        r = room_real()
        r['finger_effort']['init']['indices'] = [0, 1]
        self.assertIn('finger_effort_init_invalid', chk(E2, r)[1])
        r = room_real()
        r['finger_effort']['pause']['ok'] = 1
        self.assertIn('finger_effort_not_effective', chk(E2, r)[1])

    def test_restores_module(self):
        before = E._at_room
        E2.judge  # noqa: B018
        self.assertIs(E._at_room, before)


if __name__ == '__main__':
    unittest.main(verbosity=1)
