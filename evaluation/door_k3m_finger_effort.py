#!/usr/bin/env python3
"""W1e：暫停期提高手指關節驅動輸出上限（max effort）的純邏輯（設定、讀回、定期核對）。規格 W1e_pause_finger_effort_spec.md
（Codex 20261025_023608）。執行端 isaac_door_k2_sim_ev5.py 以 ArticulationController 呼叫；本模組只依賴 get_max_efforts()／
set_max_efforts(values, joint_indices) 介面，可用假控制器離線測試。

只改兩指的有效驅動輸出上限；閉合目標、kp／kd、摩擦、手臂／底盤控制與所有判準不變。這不是直接設定接觸法向力。
讀回與設定值差 > tol ⇒ ok＝False（不宣稱介入成立；執行端照常運行，由判定器判不通過）。
"""
from __future__ import annotations

import math

TOL = 1e-3


def _read(ctrl, idx):
    v = ctrl.get_max_efforts()
    if v is None:
        return None
    try:
        return [float(v[i]) for i in idx]
    except (TypeError, IndexError, ValueError):
        return None


def read_init(ctrl, idx, joints):
    """啟動時讀回（不改值）。"""
    rb = _read(ctrl, idx)
    return {'joints': list(joints), 'indices': list(idx), 'readback': rb,
            'ok': rb is not None and all(math.isfinite(x) for x in rb)}


class PauseFingerEffort:
    def __init__(self, value, idx, joints, period_s=1.0):
        self.value = float(value)
        self.idx, self.joints = list(idx), list(joints)
        self.period = float(period_s)
        self.record = None
        self._next = None

    @property
    def enabled(self):
        return self.value > 0.0

    def on_pause_switch(self, ctrl, step, t):
        """暫停切換那一步（world.step 之前）呼叫一次；之後再呼叫不重設。回傳 record。"""
        if not self.enabled or self.record is not None:
            return self.record
        old = _read(ctrl, self.idx)
        ctrl.set_max_efforts([self.value] * len(self.idx), joint_indices=list(self.idx))
        rb = _read(ctrl, self.idx)
        ok = rb is not None and all(abs(x - self.value) <= TOL for x in rb)
        self.record = {'step': int(step), 't': float(t), 'joints': self.joints, 'indices': self.idx, 'old': old,
                       'set': self.value, 'readback': rb, 'ok': ok, 'effective_from': 'this_step_world_step',
                       'periodic': []}
        self._next = float(t) + self.period
        return self.record

    def periodic(self, ctrl, step, t):
        """切換後每 period_s 讀回一次（不改值）。"""
        if self.record is None or t < self._next - 1e-9:
            return None
        rb = _read(ctrl, self.idx)
        ent = {'step': int(step), 't': float(t), 'readback': rb,
               'ok': rb is not None and all(abs(x - self.value) <= TOL for x in rb)}
        self.record['periodic'].append(ent)
        self._next = float(t) + self.period
        return ent
