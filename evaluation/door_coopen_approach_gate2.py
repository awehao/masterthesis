#!/usr/bin/env python3
"""核心段：接近進度閘控 r2（door_coopen_approach_gate.py 的版本化修正；Codex 20261027_004139 §1、§2；r1 保留不改）。

r1 的放行缺口（Codex 直接呼叫策略重現）：
  (a) 只在垂直超限時暫停 ⇒ 垂直 0、沿軸落後 10 mm 時 s 仍推進；
  (b) 父類執行後才還原 s ⇒ 已暫停、s 1 mm、位置到抓取點但姿態 0.03 rad 時，父類仍先轉入 ENGAGE；
  (c) 確認窗只要求狀態時間前進 ⇒ ALIGN_HOLD 兩筆有效資料間隔 0.6 s 被算成連續確認。
r2 修正（門檻與持續窗不變：垂直 3／2 mm、續進 0.5 s、沿軸 5 mm、姿態 0.02 rad、對準確認 0.5 s）：
  1. 每輪推進許可 advance_ok＝本輪量測有效 且 未暫停 且 |沿軸| ≤ along_resume_m 且 姿態 ≤ rot_resume_rad（垂直遲滯暫停照舊）；
     不許可 ⇒ 父類推進後把 s 還原（目標仍為本輪發布的 T(s)，控制照常發布）。沿軸／姿態超限只擋當輪推進，不另啟動 0.5 s 恢復窗。
  2. **APPROACH → ENGAGE**：advance_ok 為假（含暫停中、本輪量測無效、沿軸／姿態超限）時攔下 _go('ENGAGE')，並清除父類 settle_t0
     （閉爪前穩定 0.5 s 須在閘控許可下重新累積）；另 APPROACH 中任一輪不許可即清除 settle_t0（穩定窗只在閘控持續許可下累積）。
     父類在 _go 前設的 H_frozen 於真正轉入 ENGAGE 時會重設。
  3. 確認窗連續性：相鄰有效樣本的狀態時間間隔 > stale_s（0.2 s，既有新鮮度尺度）⇒ 清空對準／恢復累積窗（新樣本自身可重新起算）；
     非有限、倒退、重讀（狀態時間未前進）、state_age 超限 ⇒ 無效：清空累積窗、不許推進。
既有中止（碰桿、非設計接觸、腕力、幾何、資料失效）在父類 monitor 先行，閘控不遮蔽；APPROACH 總逾時不重置、不放寬。
"""
from __future__ import annotations

import math

import numpy as np

from door_coopen_approach_gate import ApproachGateConfig, _fin   # noqa: F401  設定與門檻同 r1


class ApproachGateMixin2:
    def gate_init(self, gcfg: ApproachGateConfig):
        self.gcfg = gcfg
        self.gate = {'paused': False, 'win_t0': None, 'last_t': None, 'align_t0': None, 'align_ok': False, 'advance_ok': False}
        self.gate_stats = {'n_pause': 0, 'n_resume': 0, 'n_hold_cycles': 0, 'n_invalid': 0, 'n_gap_reset': 0, 'n_align_wait': 0,
                           'n_engage_veto': 0, 'n_hold_along': 0, 'n_hold_rot': 0}

    def _valid(self, m):
        ok = _fin(m.T_tcp, m.H) and m.state_age is not None and math.isfinite(m.state_age) and m.state_age <= self.c.stale_s \
            and math.isfinite(m.t) and (self.gate['last_t'] is None or m.t > self.gate['last_t'])
        if not ok:
            self.gate_stats['n_invalid'] += 1
        return ok

    def _errs(self, m, T):
        e = m.T_tcp[:3, 3] - T[:3, 3]
        ax = np.asarray(m.H, float)[:3, :3] @ np.asarray(self.a_H, float)
        along = float(e @ ax)
        perp = float(np.linalg.norm(e - along * ax))
        R = T[:3, :3].T @ m.T_tcp[:3, :3]
        rot = math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0)))
        return along, perp, rot

    def _go(self, ph, t, **kw):
        if self.phase == 'ALIGN_HOLD' and ph == 'APPROACH' and not self.gate.get('align_ok'):
            self.gate_stats['n_align_wait'] += 1
            return
        if self.phase == 'APPROACH' and ph == 'ENGAGE' and not self.gate.get('advance_ok'):
            self.gate_stats['n_engage_veto'] += 1
            self.settle_t0 = None                                      # 閉爪前穩定窗須在許可下重新累積
            return
        super()._go(ph, t, **kw)

    def step(self, m, dt):
        g, gc = self.gate, self.gcfg
        ph_before, s_before = self.phase, getattr(self, 's', None)
        valid = self._valid(m)
        if valid and g['last_t'] is not None and m.t - g['last_t'] > self.c.stale_s:
            g['align_t0'], g['win_t0'] = None, None                    # 資料空白：清空累積窗（本筆可重新起算）
            self.gate_stats['n_gap_reset'] += 1
        if not valid:
            g['align_t0'], g['win_t0'] = None, None
        if ph_before == 'ALIGN_HOLD':
            ok = valid and float(np.linalg.norm(m.T_tcp[:3, 3] - self.target(m.H, self.c.s_pre)[:3, 3])) < self.c.pre_tol_m
            if ok:
                g['align_t0'] = m.t if g['align_t0'] is None else g['align_t0']
            else:
                g['align_t0'] = None
            g['align_ok'] = g['align_t0'] is not None and m.t - g['align_t0'] >= gc.align_confirm_s
        info, g['advance_ok'] = None, False
        if ph_before == 'APPROACH' and s_before is not None and valid:
            along, perp, rot = self._errs(m, self.target(m.H, s_before))
            info = {'along_m': along, 'perp_m': perp, 'rot_rad': rot, 's': s_before}
            if not g['paused'] and perp > gc.perp_pause_m:
                g['paused'], g['win_t0'] = True, None
                self.gate_stats['n_pause'] += 1
                self.events.append({'t': m.t, 'approach_gate_pause': True, **info})
            elif g['paused']:
                good = perp <= gc.perp_resume_m and abs(along) <= gc.along_resume_m and rot <= gc.rot_resume_rad
                if good:
                    g['win_t0'] = m.t if g['win_t0'] is None else g['win_t0']
                    if m.t - g['win_t0'] >= gc.resume_hold_s:
                        g['paused'], g['win_t0'] = False, None
                        self.gate_stats['n_resume'] += 1
                        self.events.append({'t': m.t, 'approach_gate_resume': True, **info})
                else:
                    g['win_t0'] = None
            along_ok, rot_ok = abs(along) <= gc.along_resume_m, rot <= gc.rot_resume_rad
            self.gate_stats['n_hold_along'] += int(not along_ok)
            self.gate_stats['n_hold_rot'] += int(not rot_ok)
            g['advance_ok'] = (not g['paused']) and along_ok and rot_ok
        if valid:
            g['last_t'] = m.t
        act = super().step(m, dt)
        if ph_before == 'APPROACH' and self.phase == 'APPROACH' and not g['advance_ok']:
            self.s = s_before                                          # 不許可：不推進（本輪目標已是 T(s_before)）
            self.settle_t0 = None                                      # 閉爪前穩定窗只在閘控持續許可下累積
            self.gate_stats['n_hold_cycles'] += 1
        act['gate'] = {'paused': g['paused'], 'advance_ok': g['advance_ok'], 'info': info, 'align_ok': g.get('align_ok')}
        return act


def gated2(policy_cls):
    """回傳把 r2 閘控疊在 policy_cls 上的類別；建構後須呼叫 gate_init(cfg)。"""
    return type('Gated2' + policy_cls.__name__, (ApproachGateMixin2, policy_cls), {})
