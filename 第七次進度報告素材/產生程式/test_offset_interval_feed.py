#!/usr/bin/env python3
"""offset_interval_feed（StepBuffer＋IntervalFeed）的離線反例。合成執行端依模型逐物理步產生設定點與套用命令，
求解快照每 4／6 步一次（交替）。

    python3 evaluation/test_offset_interval_feed.py
"""
import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offset_interval as OI                          # noqa: E402
import offset_interval_feed as F                      # noqa: E402

ID = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
ALPHA = np.asarray(ID['alpha'], float)
DTP = float(ID['phys_dt_measured_s'])
D0 = np.array([0.0, 0.017, -0.008, -0.001, -0.001, 0.0])


class World:
    """逐物理步：先套用 u_j 更新設定點，再依模型更新 q（與 offset_interval 的 s_j 序列一致）。"""

    def __init__(self, d=D0, n_steps=400, seed=0):
        rng = np.random.default_rng(seed)
        self.steps = []
        q = np.array([-0.7, 0.45, 2.1, -1.64, -0.69, 0.09])
        s = q - d
        u = np.zeros(6)
        for j in range(n_steps):
            if j % 5 == 0:
                u = rng.normal(0, 0.03, 6)
            if j > 0:
                s = s + DTP * u
            self.steps.append({'step': 1000 + j, 't': round((1000 + j) * DTP, 9), 's': s.copy(), 'u': u.copy(), 'q': q.copy()})
            q = q + ALPHA * (s - q) + ALPHA * d

    def feed_step(self, buf, k, order='sp_first', mode=0, api=(True, True), sp_override=None, t_shift=0.0):
        x = self.steps[k]
        sp = x['s'] if sp_override is None else sp_override
        calls = [lambda: buf.on_setpoint(x['step'], x['t'] + t_shift, True, sp, mode, api[0]),
                 lambda: buf.on_applied(x['step'], x['t'], x['u'], mode, api[1])]
        for c in (calls if order == 'sp_first' else calls[::-1]):
            c()


def snap_idx(n):
    out, k = [], 10
    while k < n:
        out.append(k)
        k += 4 if len(out) % 2 else 6
    return out


def mk():
    obs = OI.IntervalOffsetObserver(ALPHA, DTP)
    buf = F.StepBuffer(cap=256)
    return obs, buf, F.IntervalFeed(obs, buf, DTP)


def run(world, feed, buf, snaps, step_feeder=None, blocked_at=None):
    recs, fed = [], 0
    for k in snaps:
        while fed <= k:
            (step_feeder or (lambda b, i: world.feed_step(b, i)))(buf, fed)
            fed += 1
        x = world.steps[k]
        recs.append(feed.on_snapshot(x['t'], x['step'], x['s'], x['q'], blocked=(blocked_at(k) if blocked_at else None)))
    return recs


class Feed(unittest.TestCase):
    def test_converges_both_arrival_orders(self):
        w = World(n_steps=1200)
        for order in ('sp_first', 'ap_first'):
            obs, buf, feed = mk()
            recs = run(w, feed, buf, snap_idx(1180), lambda b, i, o=order: w.feed_step(b, i, order=o))
            self.assertEqual(recs[0]['used'], 'anchor_set')
            self.assertTrue(all(r['used'] == 'updated' for r in recs[1:]), [r['why'] for r in recs if r['used'] == 'held'][:3])
            self.assertEqual({r['m'] for r in recs[1:]}, {4, 6})
            np.testing.assert_allclose(obs.d, D0, atol=1e-8)
            for r in recs[1:]:
                np.testing.assert_allclose(r['d_obs'], D0, atol=1e-9)

    def test_applied_arrives_after_snapshot_then_recovers(self):
        """設定點先到、applied 晚於快照才到 ⇒ 該區間 missing_applied 保持；之後從新快照恢復，不重複提交。"""
        w = World()
        obs, buf, feed = mk()
        snaps = snap_idx(200)
        late = set(range(snaps[3] - 2, snaps[3] + 1))

        def feeder(b, i):
            x = w.steps[i]
            b.on_setpoint(x['step'], x['t'], True, x['s'], 0, True)
            if i not in late:
                b.on_applied(x['step'], x['t'], x['u'], 0, True)
        recs = run(w, feed, buf, snaps, feeder)
        self.assertEqual(recs[3]['used'], 'held')
        self.assertIn('missing_applied', recs[3]['why'])
        self.assertEqual(recs[4]['used'], 'held')                    # 下一區間仍含缺 applied 的起點步
        self.assertEqual(recs[4]['step0'], recs[3]['step1'])
        self.assertTrue(all(r['used'] == 'updated' for r in recs[5:]))
        self.assertEqual(len({(r['step0'], r['step1']) for r in recs if r['used'] == 'updated'}),
                         sum(1 for r in recs if r['used'] == 'updated'))   # 每區間至多提交一次

    def test_applied_truly_late_then_next_interval_ok(self):
        """applied 在快照之後才到（真的晚到、最後會補齊）⇒ 只有當下區間保持；下一區間（起點步此時已齊）正常更新。"""
        w = World()
        obs, buf, feed = mk()
        snaps = snap_idx(200)
        late = set(range(snaps[3] - 2, snaps[3] + 1))
        recs, fed = [], 0
        for i, k in enumerate(snaps):
            while fed <= k:
                x = w.steps[fed]
                buf.on_setpoint(x['step'], x['t'], True, x['s'], 0, True)
                if fed not in late:
                    buf.on_applied(x['step'], x['t'], x['u'], 0, True)
                fed += 1
            x = w.steps[k]
            recs.append(feed.on_snapshot(x['t'], x['step'], x['s'], x['q']))
            if i == 3:
                for j in sorted(late):                                # 快照後才補到
                    y = w.steps[j]
                    buf.on_applied(y['step'], y['t'], y['u'], 0, True)
        self.assertEqual(recs[3]['used'], 'held')
        self.assertIn('missing_applied', recs[3]['why'])
        self.assertTrue(all(r['used'] == 'updated' for r in recs[4:]))

    def test_missing_step_then_recovery(self):
        w = World()
        obs, buf, feed = mk()
        snaps = snap_idx(200)
        gap = snaps[5] - 1

        def feeder(b, i):
            if i != gap:
                w.feed_step(b, i)
        recs = run(w, feed, buf, snaps, feeder)
        self.assertEqual(recs[5]['used'], 'held')
        self.assertIn('missing', recs[5]['why'])
        self.assertTrue(all(r['used'] == 'updated' for r in recs[6:]))

    def test_conflicting_duplicate_not_overwritten(self):
        w = World()
        obs, buf, feed = mk()
        snaps = snap_idx(120)
        k_bad = snaps[4] - 2

        def feeder(b, i):
            w.feed_step(b, i)
            if i == k_bad:
                x = w.steps[i]
                b.on_setpoint(x['step'], x['t'], True, x['s'] + 1e-4, 0, True)       # 同步號、內容不同
            if i == k_bad - 1:
                w.feed_step(b, i)                                                      # 完全相同的重送 ⇒ 忽略
        recs = run(w, feed, buf, snaps, feeder)
        self.assertEqual(recs[4]['used'], 'held')
        self.assertIn('conflict', recs[4]['why'])
        self.assertGreaterEqual(buf.n_dup, 2)
        self.assertEqual(buf.n_conflict, 1)
        np.testing.assert_allclose(buf.sp[w.steps[k_bad]['step']][0], w.steps[k_bad]['s'])   # 原值未被覆寫

    def test_time_mismatch_between_sources(self):
        w = World()
        obs, buf, feed = mk()
        snaps = snap_idx(120)
        k_bad = snaps[3] - 1
        recs = run(w, feed, buf, snaps, lambda b, i: w.feed_step(b, i, t_shift=(1e-6 if i == k_bad else 0.0)))
        self.assertIn('time_mismatch', recs[3]['why'])

    def test_step_spacing_wrong(self):
        w = World()
        for x in w.steps[60:]:
            x['t'] = round(x['t'] + 0.005, 9)                    # 兩來源一致地跳 5 ms
        obs, buf, feed = mk()
        recs = run(w, feed, buf, snap_idx(120))
        bad = [r for r in recs if r['used'] == 'held']
        self.assertEqual(len(bad), 1)
        self.assertEqual(bad[0]['why'], 'step_spacing')

    def test_api_not_applied_either_source(self):
        w = World()
        for which in (0, 1):
            obs, buf, feed = mk()
            snaps = snap_idx(120)
            k_bad = snaps[3] - 1
            api = [True, True]
            api[which] = False
            recs = run(w, feed, buf, snaps, lambda b, i, a=tuple(api): w.feed_step(b, i, api=(a if i == k_bad else (True, True))))
            self.assertIn('api_not_applied', recs[3]['why'])

    def test_exec_mode_abnormal_and_blocked_priority(self):
        w = World()
        obs, buf, feed = mk()
        snaps = snap_idx(160)
        k_bad = snaps[3] - 1
        recs = run(w, feed, buf, snaps, lambda b, i: w.feed_step(b, i, mode=(4 if i == k_bad else 0)))
        self.assertIn('exec_mode', recs[3]['why'])
        # 停止優先：exec_mode 全為 0，但快照標 blocked ⇒ 保持、清錨點；下一快照只重設錨點，不跨停止組區間
        obs, buf, feed = mk()
        stop_k = snaps[5]
        recs = run(w, feed, buf, snaps, blocked_at=lambda k: ('stop_rx' if k == stop_k else None))
        self.assertEqual(recs[5]['used'], 'held')
        self.assertTrue(recs[5]['why'].startswith('blocked:stop_rx'))
        self.assertEqual(recs[6]['used'], 'anchor_set')
        self.assertEqual(recs[7]['used'], 'updated')

    def test_eviction_counts_as_missing(self):
        w = World(n_steps=400)
        obs, buf, feed = mk()
        buf.cap = 8
        snaps = snap_idx(200)
        recs = run(w, feed, buf, snaps)
        # 快照前已收 > cap 步、且區間 ≤ 6 步：起點步仍在緩衝 ⇒ 正常；把 cap 縮到 3 ⇒ 起點被淘汰
        self.assertTrue(all(r['used'] == 'updated' for r in recs[1:]))
        obs, buf, feed = mk()
        buf.cap = 3
        recs = run(w, feed, buf, snaps)
        self.assertTrue(all(r['used'] == 'held' and 'evicted' in r['why'] for r in recs[1:]))

    def test_overlong_interval_and_no_accumulation(self):
        w = World()
        obs, buf, feed = mk()
        snaps = [10, 14, 40, 44, 50]                              # 14→40 為 26 步
        recs = run(w, feed, buf, snaps)
        self.assertEqual([r['used'] for r in recs], ['anchor_set', 'updated', 'held', 'updated', 'updated'])
        self.assertIn('interval_too_long', recs[2]['why'])
        self.assertEqual((recs[3]['step0'], recs[3]['m']), (w.steps[40]['step'], 4))   # 從新快照重新起算

    def test_snapshot_not_advancing_and_backwards(self):
        w = World()
        obs, buf, feed = mk()
        snaps = [10, 14, 14, 20, 16, 22]
        recs = run(w, feed, buf, snaps)
        self.assertEqual(recs[2]['why'], 'snapshot_not_advancing')
        self.assertEqual(recs[3]['used'], 'updated')
        self.assertEqual((recs[3]['step0'], recs[3]['step1']), (w.steps[14]['step'], w.steps[20]['step']))
        self.assertEqual(recs[4]['why'], 'step_backwards')
        self.assertEqual(recs[5]['used'], 'updated')

    def test_snapshot_setpoint_mismatch_and_integration(self):
        w = World()
        obs, buf, feed = mk()
        snaps = snap_idx(120)
        x = w.steps[snaps[3]]
        recs = []
        fed = 0
        for i, k in enumerate(snaps):
            while fed <= k:
                w.feed_step(buf, fed)
                fed += 1
            s = w.steps[k]['s'] + (1e-6 if i == 3 else 0.0)
            recs.append(feed.on_snapshot(w.steps[k]['t'], w.steps[k]['step'], s, w.steps[k]['q']))
        self.assertEqual(recs[3]['why'], 'snapshot_setpoint_mismatch')
        # 積分不符：某步 applied 與設定點差不一致
        obs, buf, feed = mk()
        k_bad = snaps[4] - 1

        def feeder(b, i):
            xx = w.steps[i]
            b.on_setpoint(xx['step'], xx['t'], True, xx['s'], 0, True)
            b.on_applied(xx['step'], xx['t'], xx['u'] + (0.01 if i == k_bad else 0.0), 0, True)
        recs = run(w, feed, buf, snaps, feeder)
        self.assertIn('integration_mismatch', recs[4]['why'])

    def test_invalid_snapshot(self):
        obs, buf, feed = mk()
        r = feed.on_snapshot(1.0, 5, [float('nan')] * 6, np.zeros(6))
        self.assertEqual(r['why'], 'snapshot_invalid')
        r = feed.on_snapshot(1.0, None, np.zeros(6), np.zeros(6))
        self.assertEqual(r['why'], 'snapshot_invalid')
        self.assertIsNone(feed.anchor)


if __name__ == '__main__':
    unittest.main(verbosity=2)
