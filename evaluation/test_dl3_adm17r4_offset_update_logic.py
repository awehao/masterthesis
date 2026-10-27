#!/usr/bin/env python3
"""ADM17r4（--offset-update-in-open）偏移估計更新邊界的純邏輯測試（Codex r27；不跑 Isaac）。
以 ADM17r4 節點的 `_offset_update_interval`（未改寫，直接取用）作用在最小替身物件上；觀測器、餵入器用真的
offset_interval.IntervalOffsetObserver（d0＝六軸非零）與 offset_interval_feed.IntervalFeed，並以真的 StepBuffer 餵入逐物理步設定點／套用紀錄
（設定點固定、套用速度 0、量測含 +2 mrad 下垂 ⇒ 每個合格區間都會更新）。
  M1 旗標開、free 中、未暫停：不凍結；連續 10 個合格區間都 updated，d̂ 確實改變；每輪觀測器內部 d、節點 _of_d、bias＝α⊙d̂ 同步，計數一致；
  M2 暫停生效那一輪：本輪更新之前凍結（d_frozen＝上一輪 d̂、trigger＝pause、n_upd_before＝已更新數），本輪起 blocked:offset_hold、值不再變；
  M3 free 中（暫停前）停止／失效／閘門非 ARMED 仍優先：blocked:stop_rx／stopped_on_fail／gate:*，值不變、錨點清除；解除後重新設錨再更新；
  M4 暫停與停止同時：凍結照記（trigger pause），本輪阻擋理由為 stop_rx（停止優先）；
  M5 旗標關（r3 行為）：首次 free 即凍結（trigger free），與 ADM9 L2 相同；
  M6 啟動檢查：--offset-update-in-open 未帶 --offset-hold-on-free ⇒ 拒跑（SystemExit，在 rclpy.init 之前）；
  M7 靜態：r4 對 r3 的非註解刪改只限凍結觸發條件三行（其餘皆為新增）。

    python3 evaluation/test_dl3_adm17r4_offset_update_logic.py      # 需 ROS 環境（匯入節點模組）
"""
import difflib
import importlib.util
import os
import subprocess
import sys
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offset_interval as OINT                          # noqa: E402
import offset_interval_feed as OFEED                    # noqa: E402

ALPHA = np.full(6, 0.095)
D0 = np.array([0.0007, -0.0234, 0.0070, -0.0012, 0.0005, -0.0002])   # bench_3 凍結值量級，六軸非零
SP = np.array([0.1, 0.4, 0.9, 0.0, 0.5, 0.0])
SAG = np.array([0.0, 0.002, -0.001, 0.0, 0.0, 0.0])
DT = 0.01
NODE = os.path.join(HERE, 'dl3_wgmpc_wg2_adm17r4_node.py')


def node_mod():
    spec = importlib.util.spec_from_file_location('adm17r4mod', NODE)
    m = importlib.util.module_from_spec(spec)
    sys.modules['adm17r4mod'] = m
    spec.loader.exec_module(m)
    return m


def make(m, upd_open=True):
    obs = OINT.IntervalOffsetObserver(ALPHA, DT, 0.5, 0.05, d0=D0.copy())
    feed = OFEED.IntervalFeed(obs, OFEED.StepBuffer(cap=256), DT)
    flags = {'pause': False}
    me = types.SimpleNamespace(a=types.SimpleNamespace(offset_hold_on_free=True, offset_update_in_open=upd_open), _of_hold=None,
                               _mode_sw=types.SimpleNamespace(active=False), _of_d=D0.copy(), _of_n_upd=0,
                               _stop_rx=None, _stopped_on_fail=False, _gate_state=m.ARMED, _of_feed=feed,
                               _pause_active=lambda: flags['pause'],
                               cfg=types.SimpleNamespace(arm_model=types.SimpleNamespace(alpha=ALPHA.copy(), bias=ALPHA * D0)))
    return me, flags


class World:
    """逐物理步餵 StepBuffer；每 5 步一個快照（m＝5）。"""
    def __init__(self, me):
        self.me, self.step = me, 100

    def advance(self, n=5):
        b = self.me._of_feed.buf
        for _ in range(n):
            self.step += 1
            t = self.step * DT
            b.on_setpoint(self.step, t, True, SP.tolist(), 0, True)
            b.on_applied(self.step, t, [0.0] * 6, 0, True)

    def snap(self):
        if self.step == 100:
            b = self.me._of_feed.buf
            b.on_setpoint(100, 1.0, True, SP.tolist(), 0, True)
            b.on_applied(100, 1.0, [0.0] * 6, 0, True)
        return types.SimpleNamespace(sim_t=self.step * DT, sp_step_id=self.step, s=SP.copy())

    def q0(self):
        return np.concatenate([np.zeros(3), SP + SAG])


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.m = node_mod()
        except ImportError as e:
            raise unittest.SkipTest(f'無 ROS 環境：{e}')
        cls.f = staticmethod(next(getattr(c, '_offset_update_interval') for c in vars(cls.m).values()
                                  if isinstance(c, type) and hasattr(c, '_offset_update_interval')))

    def synced(self, me):
        np.testing.assert_array_equal(np.asarray(me._of_feed.obs.d), np.asarray(me._of_d))
        np.testing.assert_array_equal(np.asarray(me.cfg.arm_model.bias), ALPHA * np.asarray(me._of_d))
        self.assertEqual(me._of_n_upd, me._of_feed.n_updated)

    def run_free(self, me, w, n):
        recs = []
        for _ in range(n):
            w.advance()
            recs.append(self.f(me, w.snap(), w.q0()))
            self.synced(me)
        return recs

    def start(self, upd_open=True):
        me, flags = make(self.m, upd_open)
        w = World(me)
        r = self.f(me, w.snap(), w.q0())
        self.assertEqual(r['used'], 'anchor_set')
        me._mode_sw.active = True
        return me, flags, w

    def test_m1_updates_in_free(self):
        me, _, w = self.start()
        recs = self.run_free(me, w, 10)
        self.assertIsNone(me._of_hold)
        self.assertEqual([r['used'] for r in recs], ['updated'] * 10, [r['why'] for r in recs])
        self.assertGreater(float(np.abs(np.asarray(me._of_d) - D0).max()), 1e-5)
        self.assertEqual(me._of_n_upd, 10)

    def test_m2_freeze_at_pause(self):
        me, flags, w = self.start()
        self.run_free(me, w, 6)
        d_avail, n_avail = np.asarray(me._of_d).copy(), me._of_n_upd
        flags['pause'] = True
        w.advance()
        r = self.f(me, w.snap(), w.q0())
        self.assertEqual(me._of_hold['trigger'], 'pause')
        np.testing.assert_array_equal(np.asarray(me._of_hold['d_frozen']), d_avail)
        self.assertEqual(me._of_hold['n_upd_before'], n_avail)
        self.assertEqual(r['why'], 'blocked:offset_hold')
        for r in self.run_free(me, w, 15):
            self.assertEqual(r['why'], 'blocked:offset_hold')
            np.testing.assert_array_equal(np.asarray(me._of_d), d_avail)
            self.assertIsNone(me._of_feed.anchor)
        self.assertEqual(me._of_n_upd, n_avail)

    def test_m3_block_priority_in_free(self):
        for attr, val, want in (('_stop_rx', {'reason': 't'}, 'blocked:stop_rx'), ('_stopped_on_fail', True, 'blocked:stopped_on_fail'),
                                ('_gate_state', 'disarmed', 'blocked:gate:disarmed')):
            with self.subTest(attr):
                me, _, w = self.start()
                self.run_free(me, w, 3)
                d_avail, n_avail = np.asarray(me._of_d).copy(), me._of_n_upd
                setattr(me, attr, val)
                for r in self.run_free(me, w, 4):
                    self.assertEqual(r['why'], want)
                    np.testing.assert_array_equal(np.asarray(me._of_d), d_avail)
                    self.assertIsNone(me._of_feed.anchor)
                self.assertEqual(me._of_n_upd, n_avail)
                self.assertIsNone(me._of_hold)
                if attr == '_gate_state':                     # 閘門恢復 ⇒ 先重設錨點，再下一個區間才更新
                    setattr(me, attr, self.m.ARMED)
                    rs = self.run_free(me, w, 2)
                    self.assertEqual([x['used'] for x in rs], ['anchor_set', 'updated'])

    def test_m4_pause_and_stop(self):
        me, flags, w = self.start()
        self.run_free(me, w, 3)
        d_avail = np.asarray(me._of_d).copy()
        flags['pause'] = True
        me._stop_rx = {'reason': 't'}
        w.advance()
        r = self.f(me, w.snap(), w.q0())
        self.assertEqual(me._of_hold['trigger'], 'pause')
        self.assertEqual(r['why'], 'blocked:stop_rx')
        np.testing.assert_array_equal(np.asarray(me._of_d), d_avail)

    def test_m5_flag_off_r3_behavior(self):
        me, _, w = self.start(upd_open=False)
        w.advance()
        r = self.f(me, w.snap(), w.q0())
        self.assertEqual(me._of_hold['trigger'], 'free')
        np.testing.assert_array_equal(np.asarray(me._of_hold['d_frozen']), D0)
        self.assertEqual(r['why'], 'blocked:offset_hold')

    def test_m6_startup_check(self):
        p = subprocess.run([sys.executable, NODE, '--offset-update-in-open', '--offset-observer', 'interval', '--run-id', 'x'],
                           capture_output=True, text=True, timeout=60)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('--offset-update-in-open 需要 --offset-hold-on-free', p.stderr + p.stdout)

    def test_m7_static_diff(self):
        A = open(os.path.join(HERE, 'dl3_wgmpc_wg2_adm17r3_node.py')).read().splitlines()
        B = open(NODE).read().splitlines()
        rm = [ln[1:] for ln in difflib.unified_diff(A, B, lineterm='', n=0) if ln.startswith('-') and not ln.startswith('---')]
        rm = [ln.strip() for ln in rm if ln.strip() and not ln.lstrip().startswith('#')]
        self.assertEqual(rm, ["if (getattr(self.a, 'offset_hold_on_free', False) and self._of_hold is None and self._mode_sw is not None",
                              "and self._mode_sw.active):                                          # ADM9：首次 free ⇒ 凍結（本輪更新之前）",
                              "'first_published': None}"], rm)


if __name__ == '__main__':
    unittest.main(verbosity=2)
