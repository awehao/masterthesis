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


class GripDual2(GripDual):
    """r2（Codex 20261029 extf_grip_1 回覆）：登錄改由**任務端真正的 HOLD 確認**觸發（執行端收到 /door/k2_phase＝HOLD 的那一步呼叫 trigger）。
    確認窗定義不變（閉合命令＋兩指 ≥ f_min 連續；滾動保留最近 arm_hold_s）；觸發時窗已滿 ⇒ 當步以窗內平均登錄；未滿 ⇒ 待窗滿時登錄。
    只登錄一次、之後固定；觸發前不改目標（純量命令照舊）。r1 自行以接觸滿 2 s 觸發（extf_grip_1：4.00 s 仍在 ENGAGE），已撤換。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.trigger_t = None

    def trigger(self, t, why='task_hold'):
        if self.trigger_t is None and self.basis is None and self.cleared is None:
            self.trigger_t = {'t': float(t), 'why': why}

    def observe(self, t, q1, q2, f1, f2, valid, grip_cmd):
        if self.basis is not None or self.cleared is not None:
            return
        ok = bool(valid) and all(x is not None and math.isfinite(float(x)) for x in (q1, q2, f1, f2))
        if not self._is_closed(grip_cmd) or not ok or min(float(f1), float(f2)) < self.f_min:
            self.win = []
            return
        self.win.append((float(t), float(q1), float(q2)))
        while self.win and self.win[-1][0] - self.win[0][0] + self.dt > self.arm_hold_s + 1e-9:
            self.win.pop(0)
        full = self.win[-1][0] - self.win[0][0] + self.dt >= self.arm_hold_s - 1e-9
        if self.trigger_t is not None and full:
            n = len(self.win)
            b = (sum(w[1] for w in self.win) / n, sum(w[2] for w in self.win) / n)
            self.basis = b
            self.registered = {'t': float(t), 'trigger': dict(self.trigger_t), 'window_s': [self.win[0][0], self.win[-1][0]], 'n': n,
                               'basis_m': list(b), 'target_m': [b[0] - self.delta, b[1] - self.delta]}
            self.win = []

    def report(self):
        r = super().report()
        r.update(version='door_grip_dual/r2', trigger=self.trigger_t)
        return r
