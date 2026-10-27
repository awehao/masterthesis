#!/usr/bin/env python3
"""換抓任務端策略 r2（door_regrasp_policy1 的子類；Codex r52 §4）：S_EXIT 只是進度條件，不代替實測配對淨空。
請求 mode 4（手指改用一般 5 mm 規則）前另核：當輪 v3 淨空監看後端（door_k3m_geom_monitor_v3，目標 out）算出的兩指對外側橫桿最小距離 d，
其預測值 d − v_floor[finger]·T_pred − ds_err[finger]（T_pred＝同輪 GeometryMonitor 明細 T_pred_s；v_floor、ds_err、d_min、warn_margin 皆取既有登錄設定）
≥ d_min[finger]＋warn_margin_m；後端或明細缺 ⇒ 不成立（不請求；由 gate_wait 期限有界中止）。每次判斷記入事件。
"""
from __future__ import annotations

import math

import door_regrasp_policy1 as P1

VERSION = 'door_regrasp_policy2/r2'
Meas = P1.Meas
REG = P1.REG
RefRuntime = P1.RefRuntime


class RegraspPolicy2(P1.RegraspPolicy):
    def _pair_clear(self, m):
        fn = self.geom_v3_fn
        last = getattr(fn, 'last', None) if fn is not None else None
        gl = getattr(self, 'geom_last', None)
        if last is None or gl is None or getattr(self, 'geom_t', None) != m.t:
            return False, 'pair_geometry_missing', None
        dec, det = gl
        T = det.get('T_pred_s') if isinstance(det, dict) else None
        d = (last.get('finger_bar_out') or (None,))[0]
        if not (isinstance(T, (int, float)) and math.isfinite(T) and isinstance(d, (int, float)) and math.isfinite(d)):
            return False, 'pair_geometry_invalid', None
        c = self.geom.c
        dp = float(d) - c.v_floor['finger'] * float(T) - c.ds_err['finger']
        need = c.d_min['finger'] + c.warn_margin_m
        return dp >= need, (None if dp >= need else 'pair_clearance_low'), {'d': float(d), 'd_pred': dp, 'need': need, 'T_pred': float(T)}

    def _extra_ready(self, k, m):
        if k != 3:
            return True, None
        ok, why, det = self._pair_clear(m)
        self.rg['events'].append({'t': m.t, 'pair_check_before_mode4': ok, 'why': why, 'detail': det})
        return ok, why

    def rg_report(self):
        r = super().rg_report()
        r['version'] = VERSION
        return r
