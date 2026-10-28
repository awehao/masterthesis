#!/usr/bin/env python3
"""換抓任務端策略 r6（door_regrasp_policy5 的子類；Codex r64 §2「行為承接」）：換抓期間另用**獨立的 v3 監看器**。
問題：K3MP2Policy 的 self.geom 是任務節點以 v2 預設後端建的 GeometryMonitor，開門前綴（OPEN／SETTLE／PAUSE）照用；policy3／4 在換抓段也走 self.geom，
  而切換包裝以 geom_v3_fn 重算前態 ⇒ 真節點裡「前態 v3、本步 v2」混用（假世界測試直接把 self.geom 換成 v3，看不到這個問題）。
做法：建構時 geom_v3_fn 必給；另建 self.geom_rg＝GeometryMonitor(與 self.geom 相同設定的複本, geom_v3_fn)，切換包裝 gsw 改包 geom_rg；
  開門前綴的 self.geom（v2）完全不動。換抓段的監看器自進入換抓的第一輪起算（無前態 ⇒ 接近率 0、用 v_floor）。
"""
from __future__ import annotations

import copy

import door_k3m_geom_monitor as GM
import door_regrasp_geom_switch as GS
import door_regrasp_policy5 as P5

VERSION = 'door_regrasp_policy6/r6'
Meas = P5.Meas


class RegraspPolicy6(P5.RegraspPolicy5):
    def __init__(self, *a, geom_v3_fn=None, **kw):
        if geom_v3_fn is None:
            raise ValueError('policy6 需 v3 監看後端 geom_v3_fn')
        super().__init__(*a, geom_v3_fn=geom_v3_fn, **kw)
        if getattr(self, 'geom', None) is None:
            raise ValueError('policy6 需開門前綴的 GeometryMonitor（設定複本給換抓監看器）')
        self.geom_rg = GM.GeometryMonitor(copy.deepcopy(self.geom.c), geom_v3_fn)
        self.gsw = GS.SwitchAwareGeom(self.geom_rg, geom_v3_fn)

    def rg_report(self):
        r = super().rg_report()
        r['version'] = VERSION
        r['geom_rg_backend'] = getattr(self.geom_v3_fn, 'self_mode', None)
        return r
