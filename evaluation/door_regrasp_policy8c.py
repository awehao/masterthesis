#!/usr/bin/env python3
"""換抓策略 r8c（Codex 20261028「先建立實際退離，再張爪」）：door_regrasp_policy8b 的子類。
  - 模式表／接觸規則改 door_regrasp_mode4（RegraspAuthority4T 的權威狀態；手指目標由執行端觸發，本策略不發手指命令）；
  - mode 3 門角監看上限 DOOR_DEV_M3（None＝不變，沿用登錄 door_dev_deg）：閉爪退離會帶動門、張爪後門以釋放速度滑行（預估 1.1–1.3°），
    是否放寬由 Codex 決定後以 --door-dev-m3 給值；門速上限 door_abort_rate、幾何、接觸、腕力不變。其餘同 r8b。
"""
from __future__ import annotations

import door_regrasp_mode4 as RG4
import door_regrasp_policy8b as P8B

VERSION = 'door_regrasp_policy8c/r8c'


class RegraspPolicy8C(P8B.RegraspPolicy8B):
    RGT = RG4
    DOOR_DEV_M3 = None

    def _monitor_rg(self, m, act):
        if self.DOOR_DEV_M3 is not None and self.rg.get('k_committed', 1) == 3:
            dev0 = self.r['door_dev_deg']
            self.r['door_dev_deg'] = float(self.DOOR_DEV_M3)
            try:
                return super()._monitor_rg(m, act)
            finally:
                self.r['door_dev_deg'] = dev0
        return super()._monitor_rg(m, act)

    def rg_report(self):
        r = super().rg_report()
        r['version'] = VERSION
        r['door_dev_m3'] = self.DOOR_DEV_M3
        return r
