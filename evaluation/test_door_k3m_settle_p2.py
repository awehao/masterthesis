#!/usr/bin/env python3
"""door_k3m_settle_sig（SETTLE 完成訊號）與 door_k3m_policy_p2（只換 SETTLE 角速來源）的離線反例。

窗口（SettleRateWindow）：
  正常差分值；重送不進窗口；倒退／非有限 ⇒ 清空、需重新累積；缺步（間隔 > 0.03 s）⇒ gap；窗口不足 ⇒ short_window；
  最新樣本不是本輪步號 ⇒ step_mismatch；牆鐘停更 ⇒ stale。
策略（沿用 test_door_k3m_policy_p.PSim 合成世界；同步流以 50 Hz 餵入窗口）：
  N1 tensor 門速欄位在 SETTLE 固定 0.016 rad/s（door_k3m_pause_1 實錄的中位數）：p 版到 SETTLE 逾時中止；p2 以差分完成 SETTLE ⇒ PAUSE ⇒ DONE，
     且 m.qdot_door 未被覆寫（END_HOLD 事件記錄的 qdot 仍是 0.016）；
  N2 同步流在 SETTLE 中斷（缺樣本）⇒ 不算低速、不進 PAUSE（SETTLE 逾時中止）；
  N3 短暫低速後門再動 ⇒ 連續計時重設（記 rate），PAUSE 晚於最後一次重設 ≥ 1 s；
  N4 差分低但 tensor 門速 0.06 > 0.05 ⇒ 既有 door_rate 保護仍中止；
  N5 SETTLE 中段同步流缺步（窗口內間隔 > 0.03 s）⇒ 重設原因 sig_invalid:gap，之後重新累積 1 s；
  N6 進 PAUSE 時凍結目標＝最後一筆已發布目標（完整 4×4），握手照舊。

    python3 evaluation/test_door_k3m_settle_p2.py
"""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3m_settle_sig as SS                       # noqa: E402
import door_k3m_policy_p2 as KP2                       # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import test_door_k3a_policy as TP                      # noqa: E402
import test_door_k3m_policy as TM                      # noqa: E402
import test_door_k3m_policy_p as TPP                   # noqa: E402


class Window(unittest.TestCase):
    def feed(self, w, t0, t1, rate, q0=0.1, dt=0.02, wall0=10.0):
        t = t0
        while t <= t1 + 1e-9:
            w.on_sync(t, int(round(t * 100)), q0 + rate * (t - t0), wall0 + t)
            t = round(t + dt, 6)
        return round(t - dt, 6)

    def test_nominal_rate(self):
        w = SS.SettleRateWindow()
        tl = self.feed(w, 1.0, 1.5, 0.001)
        v = w.value(int(round(tl * 100)), 10.0 + tl)
        self.assertTrue(v['valid'], v)
        self.assertAlmostEqual(v['rate'], 0.001, places=9)
        self.assertAlmostEqual(v['t1'] - v['t0'], 0.1, places=9)

    def test_duplicate_not_in_window(self):
        w = SS.SettleRateWindow()
        tl = self.feed(w, 1.0, 1.5, 0.001)
        n = len(w.buf)
        w.on_sync(tl, int(round(tl * 100)), 0.5, 10.0 + tl)          # 同步號同時間（重送，值不同也不收）
        self.assertEqual(len(w.buf), n)
        self.assertEqual(w.anomalies['duplicate'], 1)
        self.assertAlmostEqual(w.value(int(round(tl * 100)), 10.0 + tl)['rate'], 0.001, places=9)

    def test_backward_and_nonfinite_clear(self):
        for bad in ((1.2, 120, 0.1), (1.6, 160, float('nan'))):
            w = SS.SettleRateWindow()
            tl = self.feed(w, 1.0, 1.5, 0.0)
            w.on_sync(bad[0], bad[1], bad[2], 11.6)
            self.assertEqual(len(w.buf), 0)
            w.on_sync(1.62, 162, 0.1, 11.62)
            self.assertEqual(w.value(162, 11.62)['why'], 'short_window')        # 需重新累積滿一個窗口

    def test_gap_short_mismatch_stale(self):
        w = SS.SettleRateWindow()
        self.feed(w, 1.0, 1.04, 0.0)
        self.assertEqual(w.value(104, 11.04)['why'], 'short_window')
        w = SS.SettleRateWindow()
        self.feed(w, 1.0, 1.4, 0.0)
        w.on_sync(1.46, 146, 0.1, 11.46)                                   # 1.40 → 1.46：缺兩筆
        self.assertEqual(w.value(146, 11.46)['why'], 'gap')
        w = SS.SettleRateWindow()
        tl = self.feed(w, 1.0, 1.5, 0.0)
        self.assertEqual(w.value(int(round(tl * 100)) + 2, 10.0 + tl)['why'], 'step_mismatch')
        self.assertEqual(w.value(int(round(tl * 100)), 10.0 + tl + 0.25)['why'], 'stale')
        self.assertEqual(SS.SettleRateWindow().value(1, 1.0)['why'], 'no_sample')

    def test_recompute(self):
        sq = {100: (1.0, 0.1), 110: (1.1, 0.1002)}
        self.assertAlmostEqual(SS.recompute_rate(sq, 100, 110), 0.002, places=9)
        self.assertIsNone(SS.recompute_rate(sq, 100, 111))


class P2Sim(TPP.PSim):
    """p2 版合成世界：tensor 門速欄位在 SETTLE 起可被改寫（raw_settle）；同步流 50 Hz 餵入窗口（sync_drop(t) ⇒ 該樣本不送）；
    q_bump(t)：SETTLE 期間對門角加的額外偏移（模擬門再動）。"""
    def __init__(self, raw_settle=0.016, sync_drop=None, q_bump=None, policy='p2', **kw):
        super().__init__(**kw)
        if policy == 'p2':
            gfn = (lambda q, a, b: {k: (v, 'x') for k, v in TM.NOM.items()})
            self.cfg = KP2.K3MP2Config()
            self.pol = KP2.K3MP2Policy(self.cfg, TP.T_HG(), TM.RUN, TM.COR, GM.GeometryMonitor(GM.GeomMonitorConfig(), gfn))
        self.win = SS.SettleRateWindow()
        self.raw_settle, self.sync_drop, self.q_bump = raw_settle, sync_drop, q_bump
        self.t_sync = None
        self.settle_t0 = None
        self.raw_seen = []

    def meas(self):
        m = super().meas()
        q_now = m.q_door
        if self.pol.phase in ('SETTLE', 'PAUSE'):
            self.settle_t0 = self.settle_t0 if self.settle_t0 is not None else self.t
            if self.q_bump is not None:
                q_now = m.q_door + self.q_bump(self.t - self.settle_t0)
                m.q_door = q_now
            if self.raw_settle is not None:
                m.qdot_door = self.raw_settle
        # 50 Hz 同步流：上一輪之後每 0.02 s 一筆（線性內插），最後一筆＝本輪狀態（步號＝m.state_step）
        ts = []
        if self.t_sync is not None:
            tk = round(self.t_sync + 0.02, 6)
            while tk < self.t - 1e-9:
                ts.append(tk)
                tk = round(tk + 0.02, 6)
        ts.append(self.t)
        q_last = getattr(self, 'q_sync_last', q_now)
        t_last = self.t_sync if self.t_sync is not None else self.t
        for tk in ts:
            qk = q_now if tk == self.t or self.t == t_last else q_last + (q_now - q_last) * (tk - t_last) / (self.t - t_last)
            if self.sync_drop is None or not self.sync_drop(self, tk):
                self.win.on_sync(tk, int(round(tk * 100)), qk, self.wall)
        self.t_sync, self.q_sync_last = self.t, q_now
        m.settle_sig = self.win.value(m.state_step, self.wall)
        self.raw_seen.append((self.t, self.pol.phase, m.qdot_door))
        return m


def tpause(pol):
    return next((e['t'] for e in pol.events if e.get('to') == 'PAUSE'), None)


class Policy(unittest.TestCase):
    def test_n1_raw_field_blocks_p_but_not_p2(self):
        p = P2Sim(policy='p').run()
        self.assertEqual(p.abort, 'timeout:SETTLE', p.events[-3:])
        s = P2Sim()
        pol = s.run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-4:])
        ts = next(e['t'] for e in pol.events if e.get('to') == 'SETTLE')
        self.assertGreaterEqual(tpause(pol) - ts, pol.c.settle_hold_s)
        eh = next(e for e in pol.events if e.get('to') == 'END_HOLD')
        self.assertAlmostEqual(eh['qdot'], 0.016)                         # tensor 欄位未被覆寫
        self.assertTrue(eh['settle_sig']['valid'])
        self.assertLessEqual(abs(eh['settle_sig']['rate']), pol.c.settle_rate)

    def test_n2_missing_sync(self):
        s = P2Sim(sync_drop=lambda sim, tk: sim.settle_t0 is not None)
        pol = s.run()
        self.assertIsNone(tpause(pol))
        self.assertEqual(pol.abort, 'timeout:SETTLE')

    def test_n3_low_then_moves(self):
        bump = (lambda d: 0.0 if d < 0.6 else 0.003 * (d - 0.6) if d < 1.0 else 0.0012)
        pol = P2Sim(q_bump=bump).run()
        self.assertEqual(pol.phase, 'DONE', pol.events[-4:])
        whys = [w for _, w in pol.settle_resets]
        self.assertIn('rate', whys)
        self.assertGreaterEqual(tpause(pol) - pol.settle_resets[-1][0], pol.c.settle_hold_s - 1e-9)

    def test_n4_raw_safety_still_aborts(self):
        pol = P2Sim(raw_settle=0.06).run()
        self.assertEqual(pol.abort, 'door_rate')
        self.assertIsNone(tpause(pol))

    def test_n5_intermittent_gap(self):
        def drop(sim, tk):                                                  # 連續缺 0.06 s 的中間樣本（本輪狀態照送）⇒ 窗口內出現缺步
            return sim.settle_t0 is not None and 0.5 < tk - sim.settle_t0 < 0.56 and abs(tk - sim.t) > 1e-9
        pol = P2Sim(sync_drop=drop).run()
        whys = [w for _, w in pol.settle_resets]
        self.assertTrue(any(w.startswith('sig_invalid:') for w in whys), whys)
        self.assertEqual(pol.phase, 'DONE', pol.events[-4:])
        self.assertGreaterEqual(tpause(pol) - pol.settle_resets[-1][0], pol.c.settle_hold_s - 1e-9)

    def test_n6_frozen_target_and_handshake(self):
        s = P2Sim()
        pol = s.run()
        req = next(e['k3m_pause_request'] for e in pol.events if 'k3m_pause_request' in e)
        T0 = s.pause_rows[0][1]
        np.testing.assert_allclose(np.asarray(req['frozen_target_T']).reshape(4, 4), T0, atol=1e-12)
        for _, T in s.pause_rows:
            np.testing.assert_array_equal(T, T0)
        self.assertEqual(req['seq'], 2)
        self.assertIsNotNone(pol.pwaiter.exec_ack)
        self.assertIsNotNone(pol.pwaiter.solver_ack)


if __name__ == '__main__':
    unittest.main(verbosity=2)
