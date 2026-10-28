#!/usr/bin/env python3
"""換抓任務端策略 r7（door_regrasp_policy6 的子類；聯測 iii 發現）：請求 mode 3 前另等**求解端釋放瞬態收斂**。
求解端 HELD 監督器（door_w2_mode9）只准自 release_settled 交接到 mode 3；policy1–6 只看執行端 gate3，聯測中 mode 3 於張爪後 0.75 s 提交，
求解端判 handover_before_transient_converged 失敗、不切命令流 ⇒ 執行端 post_commit_source_lost 中止。
規則同 K3-B 策略 r4（door_k3m_policy_k3b4）：/wgmpc/pause_status 的 outcome.state＝HELD 且 sub＝release_settled，且該訊息 sim_t 距本輪 ≤ stale_s；
outcome.state＝FAIL ⇒ 中止 rg_supervisor_fail:<fail_why>。
另（聯測 iii 第二趟）：換抓段幾何監看同一狀態時刻只算一次——任務迴圈在無新同步狀態的輪次以相同 m.t 再呼叫會被 GeometryMonitor 判
geometry_time_not_increasing；比照開門前綴 K3MPolicy 的 geom_last／geom_t 作法沿用同時刻結果。其餘同 r6。
"""
from __future__ import annotations

import door_regrasp_policy6 as P6

VERSION = 'door_regrasp_policy7/r7'
Meas = P6.Meas


class RegraspPolicy7(P6.RegraspPolicy6):
    def _extra_ready(self, k, m):
        if k == 2:
            ps = getattr(self, 'pause_status', None) or {}
            o = ps.get('outcome') or {}
            st = ps.get('sim_t')
            if not isinstance(st, (int, float)) or m.t - float(st) > self.r['stale_s']:
                return False, 'solver_status_stale'
            if o.get('state') != 'HELD' or o.get('sub') != 'release_settled':
                return False, f"solver_not_settled:{o.get('state')}/{o.get('sub')}"
            return True, None
        return super()._extra_ready(k, m)

    def _geom_step(self, m):
        if getattr(self, '_rg_geom_t', None) == m.t and getattr(self, '_rg_geom_last', None) is not None:
            return self._rg_geom_last                                          # 同一狀態時刻只算一次
        r = super()._geom_step(m)
        self._rg_geom_t, self._rg_geom_last = m.t, r
        return r

    def _rg_step(self, m, dt):
        o = ((getattr(self, 'pause_status', None) or {}).get('outcome') or {})
        if self.phase.startswith('RG') and o.get('state') == 'FAIL':
            return self._rg_abort(f"rg_supervisor_fail:{o.get('fail_why')}", m, self._act())
        return super()._rg_step(m, dt)

    def rg_report(self):
        r = super().rg_report()
        r['version'] = VERSION
        return r
