#!/usr/bin/env python3
"""換抓：v3 淨空監看的「目標橫桿切換」包裝（Codex r60 §4）。door_k3m_geom_monitor.GeometryMonitor（凍結）本體不改。
問題：目標由 out 改 None（或 None 改 in）時，手指類距離的組成改變（納入／排除某支橫桿），GeometryMonitor 以前後兩輪差分算接近率，
  會把「集合改變」誤當「接近」（離線實例：finger2—bar_out 52 mm、假接近率 0.26 m/s ⇒ d_pred −24 mm）。
做法（線上任務端與離線時間化／核對共用同一個類別）：
  - 每輪記下完整前態（qf、門角、手指開度、模擬時刻）；
  - 本輪目標與上一輪不同時，先以**新目標集合**對上一輪完整前態重算距離，只用它覆寫監看器的差分基準（prev）；
    不受目標影響的類別重算值與原值相同，真實接近率照常由本輪與前態（同一集合）差分得出——不清零；
  - 前態缺或重算失敗／非有限 ⇒ 該輪 stop（geometry_switch_prev_invalid），不補零通過；
  - 重算耗時加進本輪 state_age（因而進入 GeometryMonitor 的結果年齡核與 T_pred）；
  - 事件記錄：切換前後集合、前態時刻、前態舊／新距離、重算耗時，供判定器獨立重算。
"""
from __future__ import annotations

import math
import time

import numpy as np

import door_k3m_geom_monitor as GM

VERSION = 'door_regrasp_geom_switch/r1'


class SwitchAwareGeom:
    def __init__(self, monitor: GM.GeometryMonitor, fn):
        self.m, self.fn = monitor, fn
        self.target = None
        self.has_target = False
        self.prev_state = None              # (t, qf, q_door, q_finger)
        self.events = []

    def step(self, t, qf, q_door, q_finger, state_age, target):
        extra = 0.0
        if self.has_target and target != self.target and self.m.prev is not None:
            ps = self.prev_state
            ok = ps is not None
            d_prev = None
            if ok:
                t0 = time.perf_counter()
                try:
                    self.fn.set_target(target)
                    d_prev = self.fn(np.asarray(ps[1], float), float(ps[2]), float(ps[3]))
                    ok = isinstance(d_prev, dict) and all(k in d_prev and math.isfinite(float(d_prev[k][0])) for k in GM.CLASSES)
                except Exception:                                      # noqa: BLE001
                    ok = False
                extra = time.perf_counter() - t0
            if not ok:
                self.events.append({'t': t, 'switch': [self.target, target], 'why': 'prev_invalid'})
                self.target, self.has_target = target, True
                self.m.log_last = ('stop', {'why': 'geometry_switch_prev_invalid'})
                return 'stop', {'why': 'geometry_switch_prev_invalid'}
            old = dict(self.m.prev[1])
            new = {k: float(d_prev[k][0]) for k in GM.CLASSES}
            self.m.prev = (self.m.prev[0], new)
            self.events.append({'t': t, 'switch': [self.target, target], 't_prev': ps[0], 'prev_old': old, 'prev_new': new, 'recompute_s': extra})
        self.fn.set_target(target)
        dec, det = self.m.step(t, qf, q_door, q_finger, float(state_age) + extra)
        if isinstance(det, dict):
            det['switch_recompute_s'] = extra
        try:
            self.prev_state = (float(t), np.asarray(qf, float).copy(), float(q_door), float(q_finger))
        except (TypeError, ValueError):
            self.prev_state = None
        self.target, self.has_target = target, True
        return dec, det
