#!/usr/bin/env python3
"""K2 接線用的純邏輯（不依賴 ROS／Isaac，可離線測試；Codex 20261021 K2 實作審查）。

  SyncTracker        同步狀態的新鮮度：以**獨立牆鐘**記錄接收時間；步號須嚴格遞增（重複／倒退記異常）；
                     年齡＝現在牆鐘 − 最後一筆「新」狀態的接收牆鐘（停更時持續增加）；從未收到 ⇒ 年齡 None。
  ContactsAggregator 接觸窗口合併：自上次消費以來收到的所有窗口取 max（力）／min（橫桿最小）／加總（讀不到步數），
                     消費後清空——短脈衝不會被下一個正常窗口覆蓋；沒有新窗口 ⇒ 消費結果 None。
  K2AbortLatch       執行端的 K2 中止閂鎖：收到後不再把全身命令送進命令鏈、夾爪命令凍結在閂鎖當下值。
  classify_contacts  門扇接觸點分類：bar（橫桿自由段、徑向 ≤ r＋tol）、ambiguous（支柱 u 範圍內且靠近橫桿軸，或超出橫桿端）、
                     panel（其餘）。ambiguous 一律算非允許接觸。
"""
from __future__ import annotations

import math

import numpy as np


class SyncTracker:
    def __init__(self):
        self.last_step = None
        self.last_t = None
        self.last_wall = None
        self.n_dup = 0
        self.n_back = 0
        self.anomalies = []

    def on_msg(self, step, sim_t, wall_now):
        if not (math.isfinite(step) and math.isfinite(sim_t)):
            self.anomalies.append(('nonfinite', wall_now))
            return False
        if self.last_step is not None:
            if step == self.last_step:
                self.n_dup += 1
                return False
            if step < self.last_step or sim_t < self.last_t:
                self.n_back += 1
                self.anomalies.append(('backward', wall_now, self.last_step, step))
                return False
        self.last_step, self.last_t, self.last_wall = step, sim_t, wall_now
        return True

    def age(self, wall_now):
        return None if self.last_wall is None else wall_now - self.last_wall

    def ok(self):
        return self.n_back == 0 and not any(a[0] == 'nonfinite' for a in self.anomalies)


class ContactsAggregator:
    KEYS_MAX = ('unexpected_max_n', 'wall_max_n', 'wrist_max_n')

    def __init__(self):
        self.pending = None
        self.last_wall = None
        self.last_sim_t = None

    def on_msg(self, d, wall_now):
        try:
            fbmax = [float(x) for x in d['finger_bar_max_n']]
            fbmin = [float(x) for x in d['finger_bar_min_n']]
            vals = {k: float(d[k]) for k in self.KEYS_MAX}
            inv = int(d['n_invalid'])
            st = float(d['sim_t'])
        except (KeyError, TypeError, ValueError):
            fbmax, fbmin, vals, inv, st = [math.nan] * 2, [math.nan] * 2, {k: math.nan for k in self.KEYS_MAX}, 1, math.nan
        p = self.pending
        if p is None:
            p = {'finger_bar_max_n': fbmax, 'finger_bar_min_n': fbmin, **vals, 'n_invalid': inv, 'unexpected_who': d.get('unexpected_who'),
                 'n_windows': 1}
        else:
            p['finger_bar_max_n'] = [max(a, b) if math.isfinite(a) and math.isfinite(b) else math.nan for a, b in zip(p['finger_bar_max_n'], fbmax)]
            p['finger_bar_min_n'] = [min(a, b) if math.isfinite(a) and math.isfinite(b) else math.nan for a, b in zip(p['finger_bar_min_n'], fbmin)]
            for k in self.KEYS_MAX:
                if not (math.isfinite(p[k]) and math.isfinite(vals[k])):
                    p[k] = math.nan
                elif vals[k] > p[k]:
                    p[k] = vals[k]
                    if k == 'unexpected_max_n':
                        p['unexpected_who'] = d.get('unexpected_who')
            p['n_invalid'] += inv
            p['n_windows'] += 1
        self.pending = p
        self.last_wall, self.last_sim_t = wall_now, st

    def consume(self):
        p, self.pending = self.pending, None
        return p

    def age(self, wall_now):
        return None if self.last_wall is None else wall_now - self.last_wall


class K2AbortLatch:
    def __init__(self):
        self.latched = None
        self.grip_frozen = None

    def on_msg(self, why, step, sim_t, grip_now):
        if self.latched is None:
            self.latched = {'why': why, 'step': step, 'sim_t': sim_t}
            self.grip_frozen = grip_now
        return self.latched

    @staticmethod
    def expire_chain(chain):
        """閂鎖當步：使命令鏈內**既有**命令立即過期 ⇒ 之後每步走鏈既有的逾時停止路徑（手臂速率立即歸零、設定點凍結；
        底盤經輪級限制減速，limit9_timeout）。不等舊命令自然逾時。回傳是否有命令被過期。"""
        snap = getattr(chain, 'snap', None)
        if snap is None:
            return False
        snap.recv_sim_t = float('-inf')
        return True

    def filter_wb(self, cmd):
        """閂鎖後全身命令一律不進命令鏈（回 None）。"""
        return None if self.latched is not None else cmd

    def grip(self, cmd):
        return self.grip_frozen if self.latched is not None else cmd


def classify_contacts(P, f, T_WH, bar_r, bar_half, post_u, post_half_u, tol):
    """P：(N,3) 接觸點；f：(N,) 力大小；T_WH：把手 prim 位姿（x＝沿橫桿）。回傳 {'bar','ambiguous','panel'} 力和。"""
    P = np.atleast_2d(np.asarray(P, float))
    f = np.asarray(f, float).reshape(-1)
    c, u = T_WH[:3, 3], T_WH[:3, 0]
    s = (P - c) @ u
    sc = np.clip(s, -bar_half, bar_half)
    radial = np.linalg.norm(P - (c + sc[:, None] * u), axis=1)
    near_axis = radial <= bar_r + tol
    in_post = np.zeros(len(P), bool)
    for pu in post_u:
        in_post |= np.abs(s - pu) <= post_half_u + tol
    beyond_end = np.abs(s) > bar_half
    bar = near_axis & ~in_post & ~beyond_end
    amb = near_axis & (in_post | beyond_end)
    return {'bar': float(f[bar].sum()), 'ambiguous': float(f[amb].sum()), 'panel': float(f[~near_axis].sum())}
