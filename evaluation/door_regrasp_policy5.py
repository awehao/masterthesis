#!/usr/bin/env python3
"""換抓任務端策略 r5（door_regrasp_policy4 的子類；Codex r62 §3）。mode 5 參考推進加**預抓取節點閘控**（door_regrasp_refclock.PregraspClock）：
參考時間到 A 段末（預抓取）即停住，實測 TCP 對預抓取（門角修正後）≤ pre_tol_m 5 mm 才續推插入；期限按 mode 5 進入時間計
（τ_pre＋pre_wait_s 10 s），逾時 ⇒ rg5_pregrasp_timeout 中止；量測缺 ⇒ 不放行。
做法：mode 5 提交後，每步以時鐘輸出 τ_ref 平移 rg['t_ref0'][5]（＝m.t − τ_ref），既有目標／完成判斷沿用父類（只讀 t_ref0 算 τ）；
進入時刻另存 rg['t_enter5']。參考包須有 segments['5']['stage']（door_regrasp_ref_split r2）。
"""
from __future__ import annotations

import numpy as np

import door_regrasp_policy1 as P1
import door_regrasp_policy4 as P4
import door_regrasp_refclock as RC

VERSION = 'door_regrasp_policy5/r5'
Meas = P4.Meas


class RegraspPolicy5(P4.RegraspPolicy4):
    def __init__(self, *a, ref_pkg, **kw):
        st = (ref_pkg.get('segments', {}).get('5') or {}).get('stage')
        if st is None:
            raise ValueError('參考包 mode 5 缺階段標記（stage）')
        self.i_pre = RC.pregrasp_index(st)
        super().__init__(*a, ref_pkg=ref_pkg, **kw)
        self.clock5 = None

    def _pre_pos(self):
        pre = np.asarray(self.ref.p['pregrasp_T'], float).reshape(4, 4)
        return (P1.Rz4(-self.ref.ctx[5]['dq_rad']) @ pre)[:3, 3]

    def _rg_step(self, m, dt):
        if self.rg.get('k_committed') == 5 and self.clock5 is not None and self.phase not in ('DONE', 'ABORTED'):
            T = getattr(m, 'T_tcp', None)
            p = None if T is None else np.asarray(T, float)[:3, 3]
            tau_ref, status = self.clock5.step(m.t, p, self._pre_pos())
            if status == 'pregrasp_timeout':
                return self._rg_abort('rg5_pregrasp_timeout', m, self._act(), detail={'events': self.clock5.events[-2:]})
            self.rg['t_ref0'][5] = m.t - tau_ref
            self.rg['clock5'] = status
        act = super()._rg_step(m, dt)
        if self.rg.get('k_committed') == 5 and self.clock5 is None:
            self.clock5 = RC.PregraspClock(self.i_pre, self.ref.dt, pre_tol_m=self.r['pre_tol_m'], pre_wait_s=self.r.get('pre_wait_s', 10.0))
            self.clock5.start(self.rg['t_ref0'][5])
            self.rg['t_enter5'] = self.rg['t_ref0'][5]
        return act

    def rg_report(self):
        r = super().rg_report()
        r['version'] = VERSION
        r['clock5_events'] = [] if self.clock5 is None else list(self.clock5.events)
        r['t_enter5'] = self.rg.get('t_enter5')
        return r
