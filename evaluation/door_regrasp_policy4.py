#!/usr/bin/env python3
"""換抓任務端策略 r4（door_regrasp_policy3 的子類；Codex r60 §1、§2、§4）。
  - 參考包須先過 door_regrasp_ref_registry.validate（內側新抓法、B1、分支、來源雜湊、外側基準獨立）；不過即拒絕建構。
  - 幾何判定改走 door_regrasp_geom_switch.SwitchAwareGeom（目標切換以新集合重算前態；與離線時間化 v5 同一套）。
  - 門角相對解除時的監看界＝參考包登錄的起始銜接門角（v3：0.7°）。
  - 張爪前預檢（r60 §2「可預先檢查的條件前移至張爪前」）：請求 mode 2 前，以當輪實測核門角對 q_ref、底盤與手臂對 mode 3 參考起點的銜接範圍；
    超出 ⇒ 仍在外側持握的保持中中止 rg_prerelease_bridge:<why>（不放手）。mode 3／4／5 開始時 RefRuntime3 仍重核（聯合銜接檢查保留）。
"""
from __future__ import annotations

import math

import numpy as np

import door_regrasp_geom_switch as GS
import door_regrasp_policy1 as P1
import door_regrasp_policy3 as P3
import door_regrasp_ref_registry as REGY

VERSION = 'door_regrasp_policy4/r4'
Meas = P3.Meas


class RegraspPolicy4(P3.RegraspPolicy3):
    def __init__(self, *a, ref_pkg, **kw):
        ok, why = REGY.validate(ref_pkg)
        if not ok:
            raise ValueError(f'參考包登錄核驗不過：{why}')
        super().__init__(*a, ref_pkg=ref_pkg, **kw)
        self.r['door_dev_deg'] = float(ref_pkg['bridge']['door_deg'])
        self.gsw = GS.SwitchAwareGeom(self.geom, self.geom_v3_fn) if (self.geom is not None and self.geom_v3_fn is not None) else None

    def _geom_step(self, m):
        if self.gsw is None:
            self.gsw = GS.SwitchAwareGeom(self.geom, self.geom_v3_fn)
        return self.gsw.step(m.t, getattr(m, 'qf', None), m.q_door, getattr(m, 'q_finger', None), m.state_age, self.rg['geom_target'])

    def _prerelease(self, m):
        p = self.ref.p
        sg = p['segments']['3']
        qf = getattr(m, 'qf', None)
        if qf is None or m.q_door is None:
            return False, 'measurement_missing'
        x = np.asarray(qf, float)
        if not np.all(np.isfinite(x)) or x.shape != (9,):
            return False, 'measurement_invalid'
        dq = math.degrees(m.q_door) - float(p['q_ref_deg'])
        if abs(dq) > p['bridge']['door_deg']:
            return False, f'door:{dq:.3f}deg'
        b0, a0 = np.asarray(sg['base'][0], float), np.asarray(sg['arm'][0], float)
        db = float(np.linalg.norm(x[:2] - b0[:2]))
        dyaw = math.degrees(abs(P1.wrap(x[2] - b0[2])))
        da = float(np.max(np.abs(x[3:] - a0)))
        if db > p['bridge']['base_m'] or dyaw > p['bridge']['base_deg']:
            return False, f'base:{db * 1e3:.1f}mm/{dyaw:.2f}deg'
        if da > p['bridge']['arm_rad']:
            return False, f'arm:{da:.3f}rad'
        return True, None

    def _enter_rg(self, m, dt):
        ok, why = self._prerelease(m)
        self.rg['events'].append({'t': m.t, 'prerelease_check': ok, 'why': why})
        if not ok:
            return self._rg_abort(f'rg_prerelease_bridge:{why}', m, self._act())
        return super()._enter_rg(m, dt)

    def rg_report(self):
        r = super().rg_report()
        r['version'] = VERSION
        r['geom_switch_events'] = [] if self.gsw is None else self.gsw.events[:50]
        return r
