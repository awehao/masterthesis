#!/usr/bin/env python3
"""核心段：接近進度閘控（Codex 20261027_003024 §2；控制策略變更，只改接近參考的推進管理）。

以 mixin 疊在既有策略（K2Policy 衍生，如 K3APolicy、K3MP2Policy）上，不改父類檔案：
  G-A 對準確認：ALIGN_HOLD → APPROACH 須「連續有效量測、TCP 對預抓取目標位置誤差 < pre_tol_m」累積 align_confirm_s（以狀態時間計）才放行；
      原版為單筆 < pre_tol_m 即推進。未滿足時攔下 _go('APPROACH')（父類照常發布預抓取目標，W-GMPC 繼續追蹤）。
  G-B 推進凍結：APPROACH 中，以本輪發布目標 T(s)（s＝本輪推進前值）計 TCP 誤差，分解為沿接近軸（世界 H_R·a_H）分量、垂直分量與姿態角：
      垂直 > perp_pause_m ⇒ 暫停：本輪父類推進後把 s 還原（目標停在 T(s)，**控制照常發布、W-GMPC 繼續追蹤當前目標**，不凍結手臂設定點）；
      暫停中「垂直 ≤ perp_resume_m 且 |沿軸| ≤ along_resume_m 且 姿態 ≤ rot_resume_rad」以有效量測連續累積 resume_hold_s 才續進。
  有效量測：T_tcp／H 有限、state_age ≤ stale_s、且狀態時間 m.t 嚴格前進（重讀同一狀態不累積）；無效即清空累積窗（不當作滿足）。
  APPROACH 總逾時不重置、不放寬（父類 timeouts['APPROACH'] 自進入 APPROACH 起算，暫停期間照計）；提前碰桿、非設計接觸、腕力、幾何、
  資料失效等既有中止在父類 monitor 先行，閘控不遮蔽（中止後相位已非 APPROACH，閘控不作用）。
事件：approach_gate_pause／approach_gate_resume／align_confirm_wait 記入 pol.events（含誤差分量）；gate_stats 記筆數。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ApproachGateConfig:
    perp_pause_m: float = 0.003          # Codex 20261027_003024 登錄單一候選
    perp_resume_m: float = 0.002         # 同上
    resume_hold_s: float = 0.5           # 同上
    along_resume_m: float = 0.005        # ＝K2Config.pre_tol_m（既有預抓取容許）
    rot_resume_rad: float = 0.02         # ＝求解器 reach_rot_rad（既有到位姿態容許）
    align_confirm_s: float = 0.5         # ＝K2Config.grip_settle_s（既有閉爪前穩定時間）


def _fin(*a):
    return all(np.all(np.isfinite(np.asarray(x, float))) for x in a)


class ApproachGateMixin:
    def gate_init(self, gcfg: ApproachGateConfig):
        self.gcfg = gcfg
        self.gate = {'paused': False, 'win_t0': None, 'last_t': None, 'align_t0': None, 'align_ok': False}
        self.gate_stats = {'n_pause': 0, 'n_resume': 0, 'n_paused_cycles': 0, 'n_invalid': 0, 'n_align_wait': 0}

    def _valid(self, m):
        ok = _fin(m.T_tcp, m.H) and m.state_age is not None and math.isfinite(m.state_age) and m.state_age <= self.c.stale_s \
            and (self.gate['last_t'] is None or m.t > self.gate['last_t'])
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
            return                                                     # G-A：對準未連續確認，不放行
        super()._go(ph, t, **kw)

    def step(self, m, dt):
        g, gc = self.gate, self.gcfg
        ph_before, s_before = self.phase, getattr(self, 's', None)
        valid = self._valid(m)
        if ph_before == 'ALIGN_HOLD':
            ok = False
            if valid:
                e = float(np.linalg.norm(m.T_tcp[:3, 3] - self.target(m.H, self.c.s_pre)[:3, 3]))
                ok = e < self.c.pre_tol_m
            if ok:
                g['align_t0'] = m.t if g['align_t0'] is None else g['align_t0']
            else:
                g['align_t0'] = None
            g['align_ok'] = g['align_t0'] is not None and m.t - g['align_t0'] >= gc.align_confirm_s
        info = None
        if ph_before == 'APPROACH' and s_before is not None:
            T = self.target(m.H, s_before)
            if valid:
                along, perp, rot = self._errs(m, T)
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
            else:
                g['win_t0'] = None                                     # 無效量測不累積恢復窗
        if valid:
            g['last_t'] = m.t
        act = super().step(m, dt)
        if ph_before == 'APPROACH' and self.phase == 'APPROACH' and g['paused']:
            self.s = s_before                                          # G-B：凍結推進（本輪目標已是 T(s_before)）
            self.gate_stats['n_paused_cycles'] += 1
        act['gate'] = {'paused': g['paused'], 'info': info, 'align_ok': g.get('align_ok')}
        return act


def gated(policy_cls):
    """回傳把閘控疊在 policy_cls 上的類別；建構後須呼叫 gate_init(cfg)。"""
    return type('Gated' + policy_cls.__name__, (ApproachGateMixin, policy_cls), {})
