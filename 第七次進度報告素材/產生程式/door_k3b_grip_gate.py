#!/usr/bin/env python3
"""K3-B 手指命令閘門（純邏輯；ev7 用；Codex 20261026_010158 必修 1：中止不得重新閉爪）。

既有 K2 閂鎖分支在閂鎖當下以任務端 node.grip_cmd 取凍結值（K2AbortLatch.grip_frozen）。
mode 2 已由權威模式下達張開、任務端舊命令仍為閉合時，舊規則會把中止凍結成「閉合」⇒ 重新閉爪。
本閘門：
  - 未閂鎖：mode ≥ 1 時手指命令由權威模式決定（open ⇒ finger_open；closed ⇒ finger_closed）；mode 0 沿用既有命令。
    每步記住**最後實際下達**的命令。
  - 一旦閂鎖（任何入口：K2 中止訊息、入口違規、走廊、手指讀回、W2 中止…）：第一次看到時把 grip_frozen 改為最後實際下達值，
    之後恆回傳該值。
    - 閂鎖若發生在本步命令之後，下一步即修正，與該步已下達的值相同，不會在兩步間閉合。
  - 每次修正記事件（原凍結值、改後值、閂鎖原因、步）。
"""
from __future__ import annotations

VERSION = 'door_k3b_grip_gate/1'


class GripGate:
    def __init__(self, finger_open, finger_closed):
        self.finger_open, self.finger_closed = float(finger_open), float(finger_closed)
        self.last = None
        self.fixed = None
        self.events = []

    def command(self, step, t, latch, mode_seq, finger_target, base_cmd):
        """base_cmd：既有路徑算出的命令（k2_latch.grip(...)）。回傳本步實際下達的手指命令。"""
        if latch.latched is not None:
            if self.fixed is None:
                old = latch.grip_frozen
                if self.last is not None:
                    latch.grip_frozen = self.last
                self.fixed = {'step': int(step), 't': float(t), 'latched': dict(latch.latched), 'frozen_before': old, 'frozen_after': latch.grip_frozen}
                self.events.append(dict(self.fixed))
            return latch.grip_frozen
        if mode_seq is not None and mode_seq >= 1:
            cmd = self.finger_open if finger_target == 'open' else self.finger_closed
        else:
            cmd = float(base_cmd)
        self.last = cmd
        return cmd

    def report(self):
        return {'version': VERSION, 'last': self.last, 'fixed': self.fixed, 'events': self.events[:20]}
