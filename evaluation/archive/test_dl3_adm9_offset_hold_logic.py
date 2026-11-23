#!/usr/bin/env python3
"""ADM9 偏移估計保持的**非零** d̂ 純邏輯測試（Codex 20261025_003515：零值假世界無法區分「保持」與「誤清零」；不跑 Isaac）。
以 ADM9 節點的 `_offset_update_interval` 方法（未改寫，直接取用）作用在最小替身物件上；觀測器、餵入器用真的
offset_interval.IntervalOffsetObserver（d0＝六軸非零）與 offset_interval_feed.IntervalFeed。
  L1 未切 free：不凍結（_of_hold 為 None）、本輪不以 offset_hold 阻擋；
  L2 首次 free：凍結值＝當時已可取得的非零 d̂（逐元素相等），凍結前更新數記錄；
  L3 凍結後連續 20 輪：每輪 why＝blocked:offset_hold；觀測器內部 obs.d、節點 _of_d、模型偏差 bias＝α⊙d̂ 都保持非零且不變；
     n_updated 與 _of_n_upd 不增加；錨點被清除（不在保持期間累積區間）；
  L4 保持中收到停止：阻擋理由為 stop_rx（停止優先），值仍保持；
  L5 未啟用 --offset-hold-on-free：切 free 也不凍結、不以 offset_hold 阻擋（重現原行為的分支）。

    python3 evaluation/test_dl3_adm9_offset_hold_logic.py      # 需 ROS 環境（匯入節點模組）
"""
import importlib.util
import os
import sys
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offset_interval as OINT                          # noqa: E402
import offset_interval_feed as OFEED                    # noqa: E402

ALPHA = np.array([0.095, 0.095, 0.095, 0.095, 0.095, 0.095])
D0 = np.array([0.0013, 0.0262, -0.0142, -0.0038, -0.0005, -0.0002])   # door_wide_w1b_30_1 中止前量級，六軸非零


def node_mod():
    spec = importlib.util.spec_from_file_location('adm9mod', os.path.join(HERE, 'dl3_wgmpc_wg2_adm9_node.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules['adm9mod'] = m
    spec.loader.exec_module(m)
    return m


def make(m, hold=True, n_upd=17):
    obs = OINT.IntervalOffsetObserver(ALPHA, 0.01, 0.5, 0.05, d0=D0.copy())
    feed = OFEED.IntervalFeed(obs, OFEED.StepBuffer(cap=256), 0.01)
    feed.n_updated = n_upd
    me = types.SimpleNamespace(a=types.SimpleNamespace(offset_hold_on_free=hold), _of_hold=None,
                               _mode_sw=types.SimpleNamespace(active=False), _of_d=D0.copy(), _of_n_upd=n_upd,
                               _stop_rx=None, _stopped_on_fail=False, _gate_state=m.ARMED, _of_feed=feed,
                               cfg=types.SimpleNamespace(arm_model=types.SimpleNamespace(alpha=ALPHA.copy(), bias=ALPHA * D0)))
    return me


def snap(k):
    return types.SimpleNamespace(sim_t=1.0 + 0.05 * k, sp_step_id=100 + 5 * k, s=np.zeros(6))


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.m = node_mod()
        except ImportError as e:
            raise unittest.SkipTest(f'無 ROS 環境：{e}')
        fn = next(getattr(c, '_offset_update_interval') for c in vars(cls.m).values()
                  if isinstance(c, type) and hasattr(c, '_offset_update_interval'))
        cls.f = staticmethod(fn)                                    # 未綁定：第一個參數傳替身物件

    def test_l1_l2_l3_hold_nonzero(self):
        me = make(self.m)
        r = self.f(me, snap(0), np.zeros(9))
        self.assertIsNone(me._of_hold)
        self.assertNotEqual(r.get('why'), 'blocked:offset_hold')
        d_avail = np.array(me._of_d, float).copy()
        me._mode_sw.active = True
        r = self.f(me, snap(1), np.zeros(9))
        self.assertIsNotNone(me._of_hold)
        np.testing.assert_array_equal(np.array(me._of_hold['d_frozen']), d_avail)
        self.assertTrue(np.all(np.abs(d_avail) > 0))
        self.assertEqual(me._of_hold['n_upd_before'], 17)
        self.assertEqual(r.get('why'), 'blocked:offset_hold')
        for k in range(2, 22):
            r = self.f(me, snap(k), np.full(9, 0.3))                # 量測大幅變化也不得改變保持值
            self.assertEqual(r.get('why'), 'blocked:offset_hold')
            np.testing.assert_array_equal(np.asarray(me._of_feed.obs.d), d_avail)
            np.testing.assert_array_equal(np.asarray(me._of_d), d_avail)
            np.testing.assert_array_equal(np.asarray(me.cfg.arm_model.bias), ALPHA * d_avail)
            self.assertEqual((me._of_feed.n_updated, me._of_n_upd), (17, 17))
            self.assertIsNone(me._of_feed.anchor)

    def test_l4_stop_priority(self):
        me = make(self.m)
        me._mode_sw.active = True
        self.f(me, snap(0), np.zeros(9))
        me._stop_rx = {'reason': 'test'}
        r = self.f(me, snap(1), np.zeros(9))
        self.assertEqual(r.get('why'), 'blocked:stop_rx')
        np.testing.assert_array_equal(np.asarray(me._of_d), D0)

    def test_l5_disabled(self):
        me = make(self.m, hold=False)
        me._mode_sw.active = True
        for k in range(3):
            r = self.f(me, snap(k), np.zeros(9))
            self.assertIsNone(me._of_hold)
            self.assertNotEqual(r.get('why'), 'blocked:offset_hold')


if __name__ == '__main__':
    unittest.main(verbosity=2)
