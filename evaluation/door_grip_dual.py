#!/usr/bin/env python3
"""持握雙指目標（純邏輯；Codex 20261029 手指候選核可）：縮小持握位置誤差，嘗試降低持續受限的驅動需求、改善接觸穩定性
（不保證進入驅動線性區；目標偏移 Δ 下的比例項 kp·Δ 不是實際接觸力保證）。
  - 基準登錄（只一次、之後固定不追隨量測）：手指命令閉合，且兩指接觸力皆 ≥ f_min 連續 arm_hold_s（＝逐步失接觸判定的同一啟用規則）；
    該確認窗內的有效開度平均為各指基準 q_basis,i；目標 q_target,i＝q_basis,i − Δ。
  - 適用範圍：只在登錄後、未張爪前；權威／任務張爪（純量命令 > 閉合值）⇒ 清除且不再登錄（不套到之後的內側換抓）。
  - 輸出：targets(cmd, latched)——閉合命令且已登錄未清除 ⇒ 雙指 (q_target,1, q_target,2)；否則兩指＝純量命令。
    閂鎖（中止）⇒ 凍結最後實際下達的雙指目標（不回到純量凍結值）。純量命令的來源與權威（k2_latch／GripGate）不變，本類只在其後轉成雙指。
"""
from __future__ import annotations

import math

VERSION = 'door_grip_dual/r1'


class GripDual:
    def __init__(self, delta, finger_closed, f_min=0.5, arm_hold_s=2.0, dt=0.01):
        self.delta, self.closed, self.f_min, self.arm_hold_s, self.dt = float(delta), float(finger_closed), float(f_min), float(arm_hold_s), float(dt)
        self.win = []
        self.basis = None
        self.registered = None
        self.cleared = None
        self.last = None
        self.frozen = None
        self.n = {'targets_dual': 0, 'targets_scalar': 0}

    def _is_closed(self, cmd):
        return cmd is not None and float(cmd) <= self.closed + 1e-9

    def observe(self, t, q1, q2, f1, f2, valid, grip_cmd):
        """每物理步（與接觸窗口同一筆量測）；登錄前累積確認窗。"""
        if self.basis is not None or self.cleared is not None:
            return
        ok = bool(valid) and all(x is not None and math.isfinite(float(x)) for x in (q1, q2, f1, f2))
        if not self._is_closed(grip_cmd) or not ok or min(float(f1), float(f2)) < self.f_min:
            self.win = []
            return
        self.win.append((float(t), float(q1), float(q2)))
        if self.win[-1][0] - self.win[0][0] + self.dt >= self.arm_hold_s - 1e-9:
            n = len(self.win)
            b = (sum(w[1] for w in self.win) / n, sum(w[2] for w in self.win) / n)
            self.basis = b
            self.registered = {'t': float(t), 'window_s': [self.win[0][0], self.win[-1][0]], 'n': n, 'basis_m': list(b),
                               'target_m': [b[0] - self.delta, b[1] - self.delta]}
            self.win = []

    def targets(self, cmd, latched=False):
        if latched:
            if self.frozen is None:
                self.frozen = self.last if self.last is not None else (float(cmd), float(cmd))
            return self.frozen
        if self.basis is not None and self.cleared is None and not self._is_closed(cmd):
            self.cleared = {'cmd': float(cmd)}
        if self.basis is not None and self.cleared is None:
            out = (self.basis[0] - self.delta, self.basis[1] - self.delta)
            self.n['targets_dual'] += 1
        else:
            out = (float(cmd), float(cmd))
            self.n['targets_scalar'] += 1
        self.last = out
        return out

    def report(self):
        return {'version': VERSION, 'delta_m': self.delta, 'f_min': self.f_min, 'arm_hold_s': self.arm_hold_s, 'registered': self.registered,
                'cleared': self.cleared, 'frozen': None if self.frozen is None else list(self.frozen), 'n': dict(self.n)}
