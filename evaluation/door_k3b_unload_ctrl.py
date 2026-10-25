#!/usr/bin/env python3
"""K3-B 卸載斜坡的控制器介面（純函式＋ArticulationController 介面；Codex 20261026_025647 §4 整合）。
door_k3m_finger_effort.PauseFingerEffort（W1e，凍結）與 door_k3b_finger_restore（5 N 一次恢復）不改；本函式在 PauseFingerEffort 物件上
每步設定一次斜坡值：
  - set_max_efforts 兩指 → value，當步讀回：兩指＝value、非手指上限與參考向量（暫停切換後的完整向量）相同；
  - 成功 ⇒ 物件的定期讀回參考值改為 value（之後 periodic 以目前斜坡值核對，避免把正常卸載誤判 finger_readback_reverted）；
  - 失敗 ⇒ 不改參考值；回傳 ok False 與原因（呼叫端走既有中止）。
  - 前提：暫停切換已成功（fe.record ok True）；否則 'no_pause_switch_record'。
  - 只改兩指；不改 fe.record['restore']（卸載不是 5 N 恢復動作，不偽造恢復紀錄）。
"""
from __future__ import annotations

import door_k3m_finger_effort as FEFF

VERSION = 'door_k3b_unload_ctrl/1'


def set_ramp(fe, ctrl, value, step, t):
    if fe is None or fe.record is None or fe.record.get('ok') is not True:
        return {'ok': False, 'why': 'no_pause_switch_record', 'step': int(step), 't': float(t), 'set': float(value), 'readback': None}
    set_err = None
    try:
        ctrl.set_max_efforts([float(value)] * len(fe.idx), joint_indices=list(fe.idx))
    except Exception as e:                                                       # noqa: BLE001
        set_err = f'set_max_efforts 例外：{type(e).__name__}'
    after, why = FEFF._vec(ctrl)
    why = set_err or why
    if why is None and not FEFF._fingers_eq(after, fe.idx, float(value)):
        why = 'finger_readback_not_set'
    if why is None and not FEFF._others_same(fe._ref, after, fe.idx):
        why = 'non_finger_max_effort_changed'
    if why is None:
        fe.value = float(value)
    return {'ok': why is None, 'why': why, 'step': int(step), 't': float(t), 'set': float(value),
            'readback': None if after is None else [after[i] for i in fe.idx]}
