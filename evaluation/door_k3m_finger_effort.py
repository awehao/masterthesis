#!/usr/bin/env python3
"""W1e：暫停期提高手指關節驅動輸出上限（max effort）的純邏輯（設定、讀回、定期核對）。規格 W1e_pause_finger_effort_spec.md
（Codex 20261025_023608；r2＝Codex 20261025_024339 必修①③）。執行端 isaac_door_k2_sim_ev5.py 以 ArticulationController 呼叫；
本模組只依賴 get_max_efforts()／set_max_efforts(values, joint_indices) 介面，可用假控制器離線測試。

只改兩指的有效驅動輸出上限；閉合目標、kp／kd、摩擦、手臂／底盤控制與所有判準不變。這不是直接設定接觸法向力；
上限讀回只確認設定成立，不能單獨證明驅動已飽和或法向力達到某值。
r2：讀回異常不得當正常實驗——
  - read_init：兩指索引合法且不同、名稱映射＝(finger_joint1, finger_joint2)、兩指讀回有限且＝原值（5）⇒ ok True；否則 ok False＋why（執行端拒跑）；
  - on_pause_switch／periodic：記完整 DOF 上限向量與名稱映射；兩指讀回有限且＝設定值、非手指上限與切換前相同 ⇒ ok True；
    否則 ok False＋why（執行端走既有中止閂鎖、不回覆正常暫停 ack）。不自動重試、不加力、不放爪。
"""
from __future__ import annotations

import math

TOL = 1e-3


def _vec(ctrl):
    try:
        v = ctrl.get_max_efforts()
    except Exception as e:                                                       # noqa: BLE001
        return None, f'get_max_efforts 例外：{type(e).__name__}'
    if v is None:
        return None, 'get_max_efforts 回傳 None'
    try:
        out = [float(x) for x in list(v)]
    except (TypeError, ValueError):
        return None, 'get_max_efforts 非數值'
    return out, None


def _idx_ok(idx, names, joints):
    return (isinstance(idx, (list, tuple)) and len(idx) == len(joints) == 2 and all(isinstance(i, int) for i in idx)
            and len(set(idx)) == 2 and all(0 <= i < len(names) for i in idx) and [names[i] for i in idx] == list(joints))


def _fingers_eq(vec, idx, x):
    return vec is not None and all(math.isfinite(vec[i]) and abs(vec[i] - x) <= TOL for i in idx)


def _others_same(a, b, idx):
    if a is None or b is None or len(a) != len(b):
        return False
    return all((i in idx) or (a[i] == b[i]) or (math.isnan(a[i]) and math.isnan(b[i])) for i in range(len(a)))


def read_init(ctrl, idx, joints, names, original):
    """啟動時讀回（不改值）。"""
    names = list(names)
    vec, why = _vec(ctrl)
    if why is None and not _idx_ok(list(idx), names, joints):
        why = 'finger_index_or_name_mapping_invalid'
    if why is None and len(vec) != len(names):
        why = 'max_effort_vector_length_mismatch'
    if why is None and not _fingers_eq(vec, idx, float(original)):
        why = 'finger_init_readback_not_original'
    return {'joints': list(joints), 'indices': list(idx), 'names': names, 'original': float(original), 'full': vec,
            'readback': None if vec is None or not _idx_ok(list(idx), names, joints) else [vec[i] for i in idx],
            'ok': why is None, 'why': why}


class PauseFingerEffort:
    def __init__(self, value, idx, joints, period_s=1.0):
        self.value = float(value)
        self.idx, self.joints = list(idx), list(joints)
        self.period = float(period_s)
        self.record = None
        self._next = None
        self._ref = None

    @property
    def enabled(self):
        return self.value > 0.0

    def on_pause_switch(self, ctrl, step, t):
        """暫停切換那一步（world.step 之前）呼叫一次；之後再呼叫不重設。回傳 record（ok 為嚴格 bool）。"""
        if not self.enabled or self.record is not None:
            return self.record
        before, why = _vec(ctrl)
        set_err = None
        if why is None:
            try:
                ctrl.set_max_efforts([self.value] * len(self.idx), joint_indices=list(self.idx))
            except Exception as e:                                               # noqa: BLE001
                set_err = f'set_max_efforts 例外：{type(e).__name__}'
        after, why2 = _vec(ctrl)
        why = why or set_err or why2
        if why is None and not _fingers_eq(after, self.idx, self.value):
            why = 'finger_readback_not_set_value'
        if why is None and not _others_same(before, after, self.idx):
            why = 'non_finger_max_effort_changed'
        self.record = {'step': int(step), 't': float(t), 'joints': self.joints, 'indices': self.idx,
                       'old': None if before is None else [before[i] for i in self.idx], 'set': self.value,
                       'readback': None if after is None else [after[i] for i in self.idx],
                       'full_before': before, 'full_after': after, 'ok': why is None, 'why': why,
                       'effective_from': 'this_step_world_step', 'periodic': []}
        self._ref = after
        self._next = float(t) + self.period
        return self.record

    def periodic(self, ctrl, step, t):
        """切換後每 period_s 讀回一次（不改值）。回傳該筆（ok 嚴格 bool）或 None（未到時間）。"""
        if self.record is None or t < self._next - 1e-9:
            return None
        vec, why = _vec(ctrl)
        if why is None and not _fingers_eq(vec, self.idx, self.value):
            why = 'finger_readback_reverted'
        if why is None and not _others_same(self._ref, vec, self.idx):
            why = 'non_finger_max_effort_changed'
        ent = {'step': int(step), 't': float(t), 'readback': None if vec is None else [vec[i] for i in self.idx],
               'others_same': vec is not None and _others_same(self._ref, vec, self.idx), 'ok': why is None, 'why': why}
        self.record['periodic'].append(ent)
        self._next = float(t) + self.period
        return ent
