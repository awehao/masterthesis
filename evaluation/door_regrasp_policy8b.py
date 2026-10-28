#!/usr/bin/env python3
"""換抓策略 r8b（切法 B；Codex 20261028）：door_regrasp_policy8 的子類，只改模式推進所看的執行端 gate 名：
  mode 2（夾持中放手前靜止確認）完成 ⇒ 看 'pre3'（door_regrasp_exec_core_b.PreReleaseStill）確認且新鮮才請求 mode 3；
  mode 3（張爪＋退離）完成 ⇒ 看 'gate4b'（Gate3Release 放手確認且新鮮＋Gate4Exit）才請求 mode 4。其餘（參考、監看、接觸規則、門角追蹤、mode 5–8）同 r8。
"""
from __future__ import annotations

import door_regrasp_mode4 as RG4
import door_regrasp_policy8 as RP8

VERSION = 'door_regrasp_policy8b/r8b'


class RegraspPolicy8B(RP8.RegraspPolicy8):
    RGT = RG4
    ALLOWED_FN = staticmethod(RG4.allowed_finger_classes)
    GATE_FOR = {**RP8.RegraspPolicy8.GATE_FOR, 2: 'pre3', 3: 'gate4b'}

    def rg_report(self):
        r = super().rg_report()
        r['version'] = VERSION
        r['gate_for'] = {str(k): v for k, v in self.GATE_FOR.items()}
        return r
