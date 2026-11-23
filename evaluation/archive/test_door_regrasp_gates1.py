#!/usr/bin/env python3
"""door_regrasp_gates1 合成逐物理步驗證（10 ms）。
  E1 Gate4：工具 −z 退 0.06 m（2 s）後停住、無接觸 ⇒ 0.5 s 收斂窗滿＋條件再連續 0.5 s 後 CONFIRMED（約 2.97 s）；confirmed_fresh 只在處理到 step−1 時真；
     殘留 bar_out 0.2 N ⇒ 不成立；退到 0.03 m 即停 ⇒ 不成立；15 s 未成立 ⇒ FAILED retreat_timeout；
     0.1 s 資料缺口（status query_failed）⇒ 窗口重置、延後成立；0.3 s 缺口 ⇒ FAILED data_invalid；跳號 ⇒ 該步無效；
  E2 Gate5：底盤到 B1、手臂到 STOW、停 ⇒ CONFIRMED；偏航差 3° ⇒ 不成立；任一部位接觸 0.2 N ⇒ 不成立；
  E3 Gate6：TCP 收斂到名目抓取 1 mm 內 ⇒ 0.5 s 後 CONFIRMED；手指碰 bar_in 0.2 N ⇒ 重置；姿態差 3° ⇒ 不成立；complete False 的紀錄 ⇒ 無效；
  E4 HoldConfirm：兩指 bar_in 各 1 N、無漂移 ⇒ 2 s 後 CONFIRMED；單指 0.3 N ⇒ 不成立且 4.5 s FAILED hold_timeout；
     立柱 0.2 N ⇒ 不成立；漂移 6 mm ⇒ 不成立；基準取第一個成立步、之後不重設（緩慢漂移累積到 5 mm 即中斷）；成立後失去 ⇒ FAILED hold_lost_after_confirm。

    python3 evaluation/test_door_regrasp_gates1.py
"""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_regrasp_gates1 as GT                       # noqa: E402

DT = 0.01
F1, F2 = GT.FINGERS


def rec(pts=(), status='ok', complete=True):
    return {'status': status, 'complete': complete, 'points': [{'robot_link': l, 'class': c, 'f': f} for l, c, f in pts]}


def Tz(p, R=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = p
    return T


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


T0 = Tz([0.5, -0.6, 0.85])           # 工具 z＝世界 z（合成）；退開＝沿 −z


def run_g4(s_end=0.06, extra=(), gap=None, t_end=4.0, skip=None):
    g = GT.Gate4Exit()
    g.start(0, 0.0, T_tcp0=T0)
    k, conf_k = 0, None
    while k * DT < t_end:
        k += 1
        t = round(k * DT, 6)
        if skip is not None and k == skip:
            continue
        s = min(s_end, s_end * t / 2.0)
        r = rec(extra) if (gap is None or not (gap[0] <= t < gap[1])) else rec(status='query_failed')
        g.step(k, t, Tz(T0[:3, 3] - s * T0[:3, 2]), r)
        if g.state == 'CONFIRMED' and conf_k is None:
            conf_k = k
        if g.state == 'FAILED':
            break
    return g, conf_k


class TG(unittest.TestCase):
    def test_e1_gate4(self):
        g, ck = run_g4()
        self.assertEqual(g.state, 'CONFIRMED')
        self.assertAlmostEqual(ck * DT, 2.97, delta=0.03)                      # 2 s 到位＋0.5 s 收斂窗＋條件再連續 0.5 s
        self.assertTrue(g.confirmed_fresh(g.last[0] + 1))
        self.assertFalse(g.confirmed_fresh(g.last[0] + 3))
        self.assertEqual(run_g4(extra=[(F1, 'bar_out', 0.2)])[0].state, 'WAITING')
        self.assertEqual(run_g4(s_end=0.03)[0].state, 'WAITING')
        g, _ = run_g4(s_end=0.03, t_end=16.0)
        self.assertEqual((g.state, g.why), ('FAILED', 'retreat_timeout'))
        g, ck2 = run_g4(gap=(2.3, 2.4))
        self.assertEqual(g.state, 'CONFIRMED')
        self.assertGreater(ck2 * DT, 3.3)
        g, _ = run_g4(gap=(2.3, 2.61))
        self.assertEqual((g.state, g.why), ('FAILED', 'data_invalid'))
        g, ck3 = run_g4(skip=240)
        self.assertEqual(g.n['missing'], 1)
        self.assertGreater(ck3 * DT, 3.3)

    def run_g5(self, yaw_off=0.0, extra=()):
        g = GT.Gate5Arrive()
        bg, stow = np.array([0.6, -0.1, 4.712]), np.array([0.0, 0.0, 0.035, 0.0, -1.5708, 0.0])
        g.start(0, 0.0, base_goal=bg, arm_stow=stow)
        for k in range(1, 300):
            t = k * DT
            a = min(1.0, t / 1.5)
            b = bg + (1 - a) * np.array([0.1, -0.05, 0.1]) + np.array([0, 0, math.radians(yaw_off)])
            g.step(k, t, b, stow + (1 - a) * 0.2, rec(extra))
        return g

    def test_e2_gate5(self):
        self.assertEqual(self.run_g5().state, 'CONFIRMED')
        self.assertEqual(self.run_g5(yaw_off=3.0).state, 'WAITING')
        self.assertEqual(self.run_g5(extra=[('base_link', 'panel', 0.2)]).state, 'WAITING')

    def run_g6(self, rot=0.0, touch_at=None, incomplete=False):
        g = GT.Gate6Admit()
        g.start(0, 0.0)
        G = Tz([0.8, 0.0, 0.85], Rz(0.3))
        ck = None
        for k in range(1, 200):
            t = k * DT
            e = max(0.001, 0.02 * (1 - t / 0.6))
            T = Tz(G[:3, 3] + [e, 0, 0], Rz(0.3 + math.radians(rot)))
            pts = [(F1, 'bar_in', 0.2)] if touch_at is not None and abs(t - touch_at) < 0.005 else []
            g.step(k, t, T, G, rec(pts, complete=not incomplete))
            if g.state == 'CONFIRMED' and ck is None:
                ck = k
        return g, ck

    def test_e3_gate6(self):
        g, ck = self.run_g6()
        self.assertEqual(g.state, 'CONFIRMED')
        g2, ck2 = self.run_g6(touch_at=1.0)
        self.assertEqual(g2.state, 'CONFIRMED')
        self.assertGreater(ck2, ck)
        self.assertEqual(self.run_g6(rot=3.0)[0].state, 'WAITING')
        self.assertEqual(self.run_g6(incomplete=True)[0].state, 'FAILED')

    def run_hold(self, f1=1.0, f2=1.0, extra=(), drift=lambda t: 0.0, t_end=5.0, lose_at=None):
        g = GT.HoldConfirm()
        g.start(0, 0.0)
        H = Tz([0.8, 0.0, 0.85], Rz(0.2))
        E = Tz([0.0, 0.0068, 0.0])
        k = 0
        while k * DT < t_end:
            k += 1
            t = k * DT
            T = H @ E @ Tz([drift(t), 0, 0])
            ff1 = 0.0 if (lose_at is not None and t >= lose_at) else f1
            g.step(k, t, T, H, rec([(F1, 'bar_in', ff1), (F2, 'bar_in', f2)] + list(extra)))
            if g.state == 'FAILED':
                break
        return g

    def test_e4_hold(self):
        g = self.run_hold(t_end=2.5)
        self.assertEqual(g.state, 'CONFIRMED')
        self.assertAlmostEqual(g.t['CONFIRMED'], 2.01, delta=0.02)
        g = self.run_hold(f1=0.3)
        self.assertEqual((g.state, g.why), ('FAILED', 'hold_timeout'))
        self.assertEqual(self.run_hold(extra=[(F2, 'post_in0', 0.2)], t_end=3.0).state, 'WAITING')
        self.assertEqual(self.run_hold(drift=lambda t: 0.0 if t < 0.5 else 0.006, t_end=3.0).state, 'WAITING')
        g = self.run_hold(drift=lambda t: 0.003 * t, t_end=3.0)                # 1.67 s 累積到 5 mm：基準不重設 ⇒ 不成立
        self.assertEqual(g.state, 'WAITING')
        g = self.run_hold(t_end=3.0, lose_at=2.5)
        self.assertEqual((g.state, g.why), ('FAILED', 'hold_lost_after_confirm'))


if __name__ == '__main__':
    unittest.main(verbosity=1)
