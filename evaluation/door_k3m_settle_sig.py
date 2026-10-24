#!/usr/bin/env python3
"""K3-M SETTLE 完成訊號：門角因果後向差分（Codex 20261024_101421 (a)）。

只取代 SETTLE 完成判斷的角速子條件；原 tensor 門速 m.qdot_door 不被覆寫，0.05 rad/s 門速保護照舊。

輸入：執行端 /door/sync_state（50 Hz）逐筆 (t, step, q, wall)，只收步號與時間嚴格遞增、全部有限的樣本。
輸出 value(step_now, wall_now)：
  {'valid', 'rate', 'why', 't0', 't1', 'step0', 'step1', 'q0', 'q1', 'n'}
  rate＝(q1 − q0)/(t1 − t0)；樣本 1＝最新樣本（必須正是本輪量測所用的步號），
  樣本 0＝t ≤ t1 − window_s 的最晚樣本。
不成立（valid=False，不得算低速）：
  no_sample、step_mismatch（最新樣本不是本輪步號）、stale（最新樣本牆鐘年齡 > stale_wall_s）、
  short_window（沒有 t ≤ t1 − window_s 的樣本）、long_window（t1 − t0 > window_s + gap_max_s）、
  gap（窗口內相鄰樣本間隔 > gap_max_s，即缺步）、nonfinite。
異常樣本（非有限、步號或時間不遞增＝重送／倒退）不進窗口、記入 anomalies；非有限與倒退另清空窗口
（之後要重新累積滿一個窗口）。
"""
from __future__ import annotations

import collections
import math


class SettleRateWindow:
    def __init__(self, window_s=0.1, gap_max_s=0.03, stale_wall_s=0.2, keep_s=0.5):
        self.window_s = float(window_s)
        self.gap_max_s = float(gap_max_s)
        self.stale_wall_s = float(stale_wall_s)
        self.keep_s = float(keep_s)
        self.buf = collections.deque()          # (t, step, q, wall)
        self.anomalies = collections.Counter()

    def on_sync(self, t, step, q, wall):
        try:
            t, step, q, wall = float(t), int(step), float(q), float(wall)
        except (TypeError, ValueError, OverflowError):
            self.anomalies['nonfinite'] += 1
            self.buf.clear()
            return False
        if not all(math.isfinite(v) for v in (t, q, wall)):
            self.anomalies['nonfinite'] += 1
            self.buf.clear()
            return False
        if self.buf:
            t_l, s_l = self.buf[-1][0], self.buf[-1][1]
            if step == s_l and t == t_l:
                self.anomalies['duplicate'] += 1          # 重送：不進窗口（不製造零差分）
                return False
            if step <= s_l or t <= t_l:
                self.anomalies['backward'] += 1
                self.buf.clear()
                return False
        self.buf.append((t, step, q, wall))
        while self.buf and self.buf[0][0] < t - self.keep_s:
            self.buf.popleft()
        return True

    def value(self, step_now, wall_now):
        out = {'valid': False, 'rate': None, 'why': None, 't0': None, 't1': None,
               'step0': None, 'step1': None, 'q0': None, 'q1': None, 'n': 0}
        if not self.buf:
            out['why'] = 'no_sample'
            return out
        t1, s1, q1, w1 = self.buf[-1]
        out.update(t1=t1, step1=s1, q1=q1)
        if step_now is None or int(step_now) != s1:
            out['why'] = 'step_mismatch'
            return out
        if not math.isfinite(float(wall_now)) or float(wall_now) - w1 > self.stale_wall_s:
            out['why'] = 'stale'
            return out
        i0 = None
        for i in range(len(self.buf) - 1, -1, -1):
            if self.buf[i][0] <= t1 - self.window_s + 1e-9:
                i0 = i
                break
        if i0 is None:
            out['why'] = 'short_window'
            return out
        t0, s0, q0, _ = self.buf[i0]
        out.update(t0=t0, step0=s0, q0=q0, n=len(self.buf) - i0)
        if t1 - t0 > self.window_s + self.gap_max_s:
            out['why'] = 'long_window'
            return out
        for i in range(i0, len(self.buf) - 1):
            if self.buf[i + 1][0] - self.buf[i][0] > self.gap_max_s + 1e-9:
                out['why'] = 'gap'
                return out
        r = (q1 - q0) / (t1 - t0)
        if not math.isfinite(r):
            out['why'] = 'nonfinite'
            return out
        out.update(valid=True, rate=r, why='ok')
        return out


def recompute_rate(steps_q, step0, step1):
    """判定器用：由房間逐步紀錄 {step: (t, q)} 獨立重算窗口兩端的差分。"""
    if step0 not in steps_q or step1 not in steps_q:
        return None
    (t0, q0), (t1, q1) = steps_q[step0], steps_q[step1]
    if t1 <= t0:
        return None
    return (q1 - q0) / (t1 - t0)
