#!/usr/bin/env python3
"""K3-M 受控暫停 p4b：主動保持停止＋**手臂速度改用 0.5 s 窗口**（Howard 2026-10-24 選定；規格 K3M_pause_p4b_arm_window.md）。
door_k3m_pause_sup.py／door_k3m_pause_sup_p4.py 不改。

依據（door_k3m_pause_p4_1，STOPPED 後 1.1 s）：手臂零命令下原地微抖，各關節範圍 ≤ 0.3 mrad、淨位移 ≤ 0.14 mrad；
以 0.1 s 窗口算有 28% 樣本 > 1e-3 rad/s，0.2 s 窗口 1%，0.5 s 窗口 0%。
差異只有手臂：最大關節速度＝最新合法樣本與「至少 arm_win_s 之前」最近樣本的差分；有效性規則與底盤／門窗口相同
（相鄰樣本步號遞增、時間間隔 ≤ max_gap_s、不跨斷點、最新樣本新鮮）。手臂窗口不足 ⇒ 該時刻無值（不補零）。
**門檻不變**（手臂 1e-3 rad/s）；底盤與門仍用 0.1 s 窗口。報告須寫明手臂窗口為 0.5 s。
"""
from __future__ import annotations

import door_k3m_pause_sup as PSUP
import door_k3m_pause_sup_p4 as P4

ARM_WIN_S = 0.5


class WindowSpeedP4b(PSUP.WindowSpeed):
    def __init__(self, arm_win_s=ARM_WIN_S, **kw):
        super().__init__(**kw)
        if not (self.win_s <= arm_win_s <= self.keep_s):
            raise ValueError('arm_win_s 必須介於 win_s 與 keep_s 之間')
        self.arm_win_s = float(arm_win_s)

    def _start(self, win_s):
        b = self.buf[-1]
        prev = b
        for s in reversed(list(self.buf)[:-1]):
            if s is None or prev[1] - s[1] > self.max_gap_s or not (s[0] < prev[0]):
                return None
            prev = s
            if b[1] - s[1] >= win_s - 1e-9:
                return s
        return None

    def speeds(self, t_now=None, wall_now=None):
        v = super().speeds(t_now, wall_now)                       # 新鮮度、底盤、門（0.1 s）
        if v is None:
            return None
        a = self._start(self.arm_win_s)
        if a is None:
            return None
        b = self.buf[-1]
        arm = max(abs(u - w) for u, w in zip(b[6], a[6])) / (b[1] - a[1])
        return (v[0], v[1], v[2], arm)


class ActiveHoldSupervisorP4b(P4.ActiveHoldSupervisor):
    stop_definition = 'active_hold_arm_win_0.5s'

    def __init__(self, *a, arm_win_s=ARM_WIN_S, **kw):
        super().__init__(*a, **kw)
        self.ws = WindowSpeedP4b(arm_win_s=arm_win_s)

    def outcome(self, t_end=None):
        o = super().outcome(t_end)
        o['arm_window_s'] = self.ws.arm_win_s
        o['base_door_window_s'] = self.ws.win_s
        return o
