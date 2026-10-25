#!/usr/bin/env python3
"""K3-B 暫停期手指驅動上限恢復（20 → 5 N；純邏輯＋控制器介面；規格 K3B_release_clear_spec_draft2.md §6；Codex 20261026_000706 D4）。
door_k3m_finger_effort.PauseFingerEffort（W1e，凍結）不改；本函式在其物件上執行一次恢復：
  - set_max_efforts 兩指 → value（原值 5 N），當步讀回：兩指＝value、非手指上限不變（與切換前完整向量比）；
  - 成功 ⇒ 物件的定期讀回參考改為 value（之後 periodic 以 5 N 核對，避免把正常恢復誤判 finger_readback_reverted）；
  - 失敗 ⇒ 不改參考（之後 periodic 仍以 20 N 核，讀回不符走既有中止）；回傳 ok False 與原因。
只呼叫一次；第二次呼叫 ⇒ 'restore_twice'（不再改值）。
"""
from __future__ import annotations

import door_k3m_finger_effort as FEFF

VERSION = 'door_k3b_finger_restore/1'


def restore(fe, ctrl, value, step, t):
    if fe is None or fe.record is None:
        return {'ok': False, 'why': 'no_pause_switch_record', 'step': int(step), 't': float(t)}
    if fe.record.get('restore') is not None:
        return {'ok': False, 'why': 'restore_twice', 'step': int(step), 't': float(t)}
    before, why = FEFF._vec(ctrl)
    set_err = None
    if why is None:
        try:
            ctrl.set_max_efforts([float(value)] * len(fe.idx), joint_indices=list(fe.idx))
        except Exception as e:                                                   # noqa: BLE001
            set_err = f'set_max_efforts 例外：{type(e).__name__}'
    after, why2 = FEFF._vec(ctrl)
    why = why or set_err or why2
    if why is None and not FEFF._fingers_eq(after, fe.idx, float(value)):
        why = 'finger_readback_not_restored'
    if why is None and not FEFF._others_same(before, after, fe.idx):
        why = 'non_finger_max_effort_changed'
    rec = {'step': int(step), 't': float(t), 'from': fe.value, 'to': float(value),
           'readback': None if after is None else [after[i] for i in fe.idx], 'ok': why is None, 'why': why}
    fe.record['restore'] = rec
    if why is None:
        fe.value = float(value)
        fe._ref = after
    return rec
