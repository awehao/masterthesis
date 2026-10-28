#!/usr/bin/env python3
"""手指失接觸的逐物理步連續判定（純邏輯；Codex 20261029 extf_1 回覆）：原定義 0.5 N／0.2 s 改在執行端逐物理步計時。
停止判準**時間語義變更**：原任務端以發布窗口最小力（finger_bar_min）跨窗口累計時間，窗口之間的實際恢復未計入（extf_1：最長低力事件僅 1 步 0.01 s，仍被累計成 0.2 s）。
  - 啟用（armed）：手指命令為閉合，且兩指皆 ≥ f_min 連續 arm_hold_s（夾持已建立）；手指命令改為非閉合 ⇒ 解除並清零。
  - 計時：啟用中，某指有效樣本 < f_min ⇒ 自首個低力步起算連續時長（含本步 dt）；有效樣本 ≥ f_min ⇒ 清零。
    無效樣本（讀不到）不清零也不加時（資料失效由既有保護處理）。
  - 觸發：任一指連續時長 ≥ t_loss ⇒ 回傳中止原因（呼叫端走既有 K2 閂鎖）；只觸發一次。
  - 窗口摘要：cur_low_s（窗口結束時仍持續的低力時長）、max_low_s（窗口內最長），供任務端讀取；窗口最小力另由原欄位保存。
"""
from __future__ import annotations

import math

VERSION = 'door_finger_loss_cont/r1'


class FingerLossCont:
    def __init__(self, f_min=0.5, t_loss=0.2, arm_hold_s=2.0, dt=0.01):
        self.f_min, self.t_loss, self.arm_hold_s, self.dt = float(f_min), float(t_loss), float(arm_hold_s), float(dt)
        self.armed = False
        self.arm_t0 = None
        self.t0 = [None, None]
        self.dur = [0.0, 0.0]
        self.fired = None
        self.win_max = [0.0, 0.0]
        self.n = {'steps': 0, 'armed_steps': 0, 'low_events': 0, 'invalid': 0}
        self.events = []

    def step(self, t, f1, f2, valid, grip_closed):
        self.n['steps'] += 1
        t = float(t)
        f = (f1, f2)
        ok = bool(valid) and all(x is not None and math.isfinite(float(x)) for x in f)
        if not grip_closed:
            if self.armed:
                self.events.append({'t': t, 'disarm': 'grip_not_closed'})
            self.armed, self.arm_t0, self.t0, self.dur = False, None, [None, None], [0.0, 0.0]
            return None
        if not self.armed:
            if ok and min(float(x) for x in f) >= self.f_min:
                self.arm_t0 = self.arm_t0 if self.arm_t0 is not None else t
                if t - self.arm_t0 + self.dt >= self.arm_hold_s - 1e-9:
                    self.armed = True
                    self.events.append({'t': t, 'armed': True})
            else:
                self.arm_t0 = None
            return None
        self.n['armed_steps'] += 1
        if not ok:
            self.n['invalid'] += 1
            return None
        for k in range(2):
            if float(f[k]) < self.f_min:
                if self.t0[k] is None:
                    self.t0[k] = t
                    self.n['low_events'] += 1
                self.dur[k] = t - self.t0[k] + self.dt
                self.win_max[k] = max(self.win_max[k], self.dur[k])
                if self.fired is None and self.dur[k] >= self.t_loss - 1e-9:
                    self.fired = {'t': t, 'finger': k + 1, 'dur_s': self.dur[k], 't0': self.t0[k]}
                    self.events.append(dict(self.fired, fire=True))
                    return f'finger_contact_lost_cont:f{k + 1}:{self.dur[k]:.3f}s'
            else:
                self.t0[k], self.dur[k] = None, 0.0
        return None

    def take_window(self):
        out = {'finger_low_cont_s': [float(x) for x in self.dur], 'finger_low_cont_max_s': [float(x) for x in self.win_max],
               'finger_loss_armed': bool(self.armed)}
        self.win_max = [float(x) for x in self.dur]
        return out

    def report(self):
        return {'version': VERSION, 'f_min': self.f_min, 't_loss': self.t_loss, 'arm_hold_s': self.arm_hold_s, 'armed': self.armed,
                'fired': self.fired, 'n': dict(self.n), 'events': self.events[:200]}
