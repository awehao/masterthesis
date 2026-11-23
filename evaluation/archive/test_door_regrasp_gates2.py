#!/usr/bin/env python3
"""door_regrasp_gates2 驗證（test_door_regrasp_gates1 的 r2 版：合成紀錄改帶真實 step／t；另加 Codex r52 反例 C1–C7）。
  C1 紀錄不屬於當步（step −100、t 不符、缺欄）⇒ 不成立（Gate4／5／6／Hold）；
  C2 確認後送一筆 query_failed ⇒ 立即 WAITING、confirmed_fresh False；之後有效資料須重新累積完整窗口；
  C3 confirmed_fresh(−100)、非整數步號 ⇒ False；
  C4 step 1 @ 0.01 → step 2 @ 0.62（相鄰步號、0.61 s 缺口）⇒ 不成立且 FAILED data_invalid；
  C5 相鄰步號但時間跳 0.05 s ⇒ 該步無效、窗口重置（延後成立）；
  C6 HoldConfirm 確認後一筆無效 ⇒ 撤銷（WAITING，非 FAILED）、恢復後重新 2 s（期限 4.5 s 自閉爪起不重設：撤銷太晚即逾時失敗）；確認後條件不成立 ⇒ FAILED hold_lost_after_confirm；
  C7 Gate3Release：張爪＋小力 0.5 s ⇒ B_CONFIRMED 且准入；無效一步 ⇒ 不准入、恢復後須再 0.5 s；缺口 0.25 s ⇒ FAILED；相鄰步時間跳號 ⇒ 該步無效。
以下為 r1 說明：
door_regrasp_gates1 合成逐物理步驗證（10 ms）。
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
import door_regrasp_gates2 as GT                       # noqa: E402

DT = 0.01
F1, F2 = GT.FINGERS


def rec(pts=(), status='ok', complete=True, step=None, t=None):
    return {'status': status, 'complete': complete, 'step': step, 't': t, 'points': [{'robot_link': l, 'class': c, 'f': f} for l, c, f in pts]}


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
        r = rec(extra, step=k, t=t) if (gap is None or not (gap[0] <= t < gap[1])) else rec(status='query_failed', step=k, t=t)
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
            t = round(k * DT, 6)
            a = min(1.0, t / 1.5)
            b = bg + (1 - a) * np.array([0.1, -0.05, 0.1]) + np.array([0, 0, math.radians(yaw_off)])
            g.step(k, t, b, stow + (1 - a) * 0.2, rec(extra, step=k, t=t))
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
            t = round(k * DT, 6)
            e = max(0.001, 0.02 * (1 - t / 0.6))
            T = Tz(G[:3, 3] + [e, 0, 0], Rz(0.3 + math.radians(rot)))
            pts = [(F1, 'bar_in', 0.2)] if touch_at is not None and abs(t - touch_at) < 0.005 else []
            g.step(k, t, T, G, rec(pts, complete=not incomplete, step=k, t=t))
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
            t = round(k * DT, 6)
            T = H @ E @ Tz([drift(t), 0, 0])
            ff1 = 0.0 if (lose_at is not None and t >= lose_at) else f1
            g.step(k, t, T, H, rec([(F1, 'bar_in', ff1), (F2, 'bar_in', f2)] + list(extra), step=k, t=t))
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


class TR2(unittest.TestCase):
    """Codex r52 反例。"""

    def g6_run(self, n, rec_fn, t_fn=lambda k: round(k * DT, 6), step_fn=lambda k: k):
        g = GT.Gate6Admit()
        g.start(0, 0.0)
        G = Tz([0.8, 0.0, 0.85])
        hist = []
        for k in range(1, n):
            st, t = step_fn(k), t_fn(k)
            g.step(st, t, G, G, rec_fn(st, t))
            hist.append((st, t, g.state, g.confirmed_fresh(st + 1)))
            if g.state == 'FAILED':
                break
        return g, hist

    def test_c1_binding(self):
        for bad in (lambda s, t: rec(step=-100, t=-1.0), lambda s, t: rec(step=s, t=t + 0.003), lambda s, t: rec(step=None, t=t),
                    lambda s, t: {'status': 'ok', 'complete': True, 'points': []}):
            g, _ = self.g6_run(120, bad)
            self.assertNotEqual(g.state, 'CONFIRMED')
        g = GT.Gate4Exit()
        g.start(0, 0.0, T_tcp0=T0)
        for k in range(1, 400):
            t = round(k * DT, 6)
            g.step(k, t, Tz(T0[:3, 3] - min(0.06, 0.03 * t) * T0[:3, 2]), rec(step=k - 1, t=t))
        self.assertNotEqual(g.state, 'CONFIRMED')

    def test_c2_c3_revoke(self):
        bad_at = {80}
        g, h = self.g6_run(200, lambda s, t: rec(status='query_failed', step=s, t=t) if s in bad_at else rec(step=s, t=t))
        conf = [x for x in h if x[2] == 'CONFIRMED']
        self.assertTrue(conf and conf[0][0] < 80)
        self.assertEqual(next(x for x in h if x[0] == 80)[2:], ('WAITING', False))
        re = next(x for x in h if x[0] > 80 and x[2] == 'CONFIRMED')
        self.assertGreaterEqual(re[1] - 0.81, 0.5 - 1e-9)                      # 恢復後重新累積完整 0.5 s
        self.assertEqual(g.n['revoked'], 1)
        self.assertFalse(g.confirmed_fresh(-100))
        self.assertFalse(g.confirmed_fresh(1.0))
        self.assertFalse(g.confirmed_fresh(True))
        self.assertTrue(g.confirmed_fresh(g.last_valid_step) and g.confirmed_fresh(g.last_valid_step + 1))
        self.assertFalse(g.confirmed_fresh(g.last_valid_step + 2))

    def test_c4_c5_gaps(self):
        g = GT.Gate6Admit()
        g.start(0, 0.0)
        G = Tz([0.8, 0.0, 0.85])
        g.step(1, 0.01, G, G, rec(step=1, t=0.01))
        g.step(2, 0.62, G, G, rec(step=2, t=0.62))
        self.assertEqual((g.state, g.why), ('FAILED', 'data_invalid'))
        g, h = self.g6_run(150, lambda s, t: rec(step=s, t=t), t_fn=lambda k: round(k * DT + (0.05 if k >= 30 else 0.0), 6))
        c = next(x for x in h if x[2] == 'CONFIRMED')
        self.assertGreater(c[0], 30 + 49)
        self.assertEqual(g.n['time_gap'], 1)

    def test_c6_hold_revoke(self):
        g = GT.HoldConfirm()
        g.start(0, 0.0)
        H = Tz([0.8, 0.0, 0.85])
        E = Tz([0.0, 0.0068, 0.0])
        ok_pts = [(F1, 'bar_in', 1.0), (F2, 'bar_in', 1.0)]
        conf2 = None
        for k in range(1, 500):
            t = round(k * DT, 6)
            r = rec(status='query_failed', step=k, t=t) if k == 210 else rec(ok_pts, step=k, t=t)
            g.step(k, t, H @ E, H, r)
            if k == 210:
                self.assertEqual(g.state, 'WAITING')
            if k > 210 and g.state == 'CONFIRMED' and conf2 is None:
                conf2 = t
        self.assertGreaterEqual(conf2 - 2.11, 2.0 - 1e-9)                  # 撤銷後重新累積完整 2 s（仍在 4.5 s 期限內）
        g = GT.HoldConfirm()
        g.start(0, 0.0)
        for k in range(1, 300):
            t = round(k * DT, 6)
            g.step(k, t, H @ E, H, rec(ok_pts if k < 250 else [(F1, 'bar_in', 1.0)], step=k, t=t))
            if g.state == 'FAILED':
                break
        self.assertEqual((g.state, g.why), ('FAILED', 'hold_lost_after_confirm'))

    def test_c7_gate3(self):
        def run(bad=(), gap=None, tj=None, n=150):
            g = GT.Gate3Release()
            g.open(0, 0.0)
            fr = []
            for k in range(1, n):
                t = round(k * DT + (0.05 if (tj is not None and k >= tj) else 0.0), 6)
                if gap is not None and gap[0] <= k < gap[1]:
                    continue
                g.step(k, t, [0.0089, 0.0089], [0.0, 0.0], k not in bad)
                fr.append((k, g.state, g.confirmed_fresh(k + 1)))
                if g.state == 'FAILED':
                    break
            return g, fr
        g, fr = run()
        first = next(k for k, _, f in fr if f)
        self.assertEqual(g.state, 'B_CONFIRMED')
        self.assertLessEqual(first, 52)
        g, fr = run(bad={80})
        self.assertFalse(next(f for k, _, f in fr if k == 80))
        re = next(k for k, _, f in fr if k > 80 and f)
        self.assertGreaterEqual(re - 81, 49)
        g, _ = run(gap=(60, 86))
        self.assertEqual((g.state, g.why), ('FAILED', 'data_invalid'))
        g, fr = run(tj=70)
        self.assertEqual(g.n['time_gap'], 1)
        self.assertFalse(next(f for k, _, f in fr if k == 70))


if __name__ == '__main__':
    unittest.main(verbosity=1)
