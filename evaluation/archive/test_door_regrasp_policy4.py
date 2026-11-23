#!/usr/bin/env python3
"""door_regrasp_policy4 驗證（Codex r60 §1、§2、§4）。假世界沿用 test_door_regrasp_policy3.Fake3（理想追蹤；權威與 gate 排程同 policy1），
參考包改 door_regrasp_ref_v3（以 T1.PKG 替換；v1／v2 測試不受影響，tearDown 還原）。
  P1 建構即核驗登錄：舊 v2 參考包 ⇒ ValueError（不進 super().__init__）；
  P2 真 v3 後端（self_mode full，同時間化 v5）＋切換包裝、理想追蹤 ⇒ DONE（policy3 V10 的已知衝突在新參考下不再出現），
     且切換事件含 out→None 與 None→in、每筆事件有前態舊／新距離與重算耗時；
  P3 張爪前預檢：門角偏 +0.8°（> 登錄 0.7°）⇒ rg_prerelease_bridge:door…，且權威從未被請求 mode 2（仍在外側持握保持中中止）；
  P4 張爪前預檢：底盤起點偏 25 mm（> 0.02 m）⇒ rg_prerelease_bridge:base…；
  P5 門角偏 +0.5°（範圍內）⇒ 預檢通過（事件 prerelease_check True）並提交到 mode 3。起始構型隨門繞鉸鏈剛體轉（r2：r1 未轉，夾持構型與偏轉門板
     重疊 ⇒ mode 1 即 geometry_clearance_nonfinger，屬假世界錯誤）；只核到 mode 3 提交（Fake3 的 mode 3 構型取未轉參考，之後不具意義）。
注意：理想追蹤＝實測構型取發布參考；不是 GMPC 追蹤通過（另由名目閉迴路核）。

    python3 evaluation/test_door_regrasp_policy4.py
"""
import json
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402
import door_k3m_policy_p2 as PP2                       # noqa: E402
import door_regrasp_geom_switch as GS                  # noqa: E402
import door_regrasp_policy3 as RP3                     # noqa: E402
import door_regrasp_policy4 as RP4                     # noqa: E402
import test_door_regrasp_policy1 as T1                 # noqa: E402
import test_door_regrasp_policy3 as T3                 # noqa: E402

V3 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v3.json')))
V2 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_v2.json')))
G = T3.G
FN = None


def mk4():
    global FN
    p = T1.mk()
    p.__class__ = RP4.RegraspPolicy4
    p.r = dict(RP3.REG)
    p.r['door_dev_deg'] = float(V3['bridge']['door_deg'])
    p.ref = RP3.RefRuntime3(p.ref.p, T3.KIN, T_b=p.r['bridge_T_s'])
    p.pair_prev = None
    p.geom_last, p.geom_t = None, None
    if FN is None:
        FN = GM3.backend_v3(self_mode='full')
    FN.set_target('out')
    p.geom_v3_fn = FN
    p.geom = GM.GeometryMonitor(GM.GeomMonitorConfig(v_floor=dict(G['v_floor_registered']), ds_err=dict(G['downsampling']['ds_err_registered'])), FN)
    p.gsw = GS.SwitchAwareGeom(p.geom, FN)
    return p


class TP(unittest.TestCase):
    def setUp(self):
        self._orig = PP2.K3MP2Policy.step
        self._pkg = T1.PKG
        PP2.K3MP2Policy.step = T1._fake_parent_step
        T1.PKG = V3

    def tearDown(self):
        PP2.K3MP2Policy.step = self._orig
        T1.PKG = self._pkg

    def test_p1_registry(self):
        with self.assertRaises(ValueError) as cm:
            RP4.RegraspPolicy4(ref_pkg=V2)
        self.assertIn('inner_grasp_missing', str(cm.exception))

    def test_p2_done_real_backend(self):
        p = mk4()
        w = T3.Fake3(p).run(200.0)
        self.assertEqual(p.phase, 'DONE', (p.abort, [e for e in p.rg['events'] if 'abort' in str(e)][-3:]))
        self.assertIn(6, w.t_commit)
        sw = [tuple(e['switch']) for e in p.gsw.events]
        self.assertIn(('out', None), sw)
        self.assertIn((None, 'in'), sw)
        for e in p.gsw.events:
            self.assertIn('prev_new', e)
            self.assertTrue(math.isfinite(e['recompute_s']))
        self.assertTrue(any(e.get('prerelease_check') for e in p.rg['events']))

    def test_p3_prerelease_door(self):
        p = mk4()
        w = T3.Fake3(p, dq=0.8).run(30.0)
        self.assertTrue((p.abort or '').startswith('rg_prerelease_bridge:door'), p.abort)
        self.assertNotIn(2, w.t_commit)
        self.assertNotIn(2, w.req_t)

    def test_p4_prerelease_base(self):
        p = mk4()
        w = T3.Fake3(p)
        w.base = w.base + np.r_[0.025, 0.0, 0.0]
        w.run(30.0)
        self.assertTrue((p.abort or '').startswith('rg_prerelease_bridge:base'), p.abort)
        self.assertNotIn(2, w.req_t)

    def test_p5_prerelease_ok_inside(self):
        p = mk4()
        w = T3.Fake3(p, dq=0.5)
        a = -math.radians(0.5)
        c, s_ = math.cos(a), math.sin(a)
        b = w.base.copy()
        w.base = np.r_[c * b[0] - s_ * b[1], s_ * b[0] + c * b[1], b[2] + a]
        w.run(25.0)
        self.assertTrue(any(e.get('prerelease_check') is True for e in p.rg['events']))
        self.assertIn(3, w.t_commit, p.abort)


if __name__ == '__main__':
    unittest.main(verbosity=1)
