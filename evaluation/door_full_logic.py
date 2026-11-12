#!/usr/bin/env python3
"""整合執行器的段別守門邏輯（純函式，離線可測；Codex 20261113_020612）。"""
import math

TASK_REF_MAX_AGE_S = 30.0                                            # 任務參考最長沿用時間（模擬時鐘）
TASK_REF_DOOR_DEG = 0.05                                             # 擷取後門角變化上限（同 A0 門角監看）


def task_ref_check(t_now_sim, t_capture_sim, door_now_rad, door_capture_rad, latched):
    """已接受任務參考是否仍有效：回 (ok, 資訊)。失效條件：前段任何鎖存、門角自擷取起變化超過 TASK_REF_DOOR_DEG、年齡超過 TASK_REF_MAX_AGE_S。"""
    age = float(t_now_sim) - float(t_capture_sim)
    dd = abs(math.degrees(float(door_now_rad) - float(door_capture_rad)))
    why = []
    if latched:
        why.append('latched_before')
    if dd > TASK_REF_DOOR_DEG:
        why.append('door_moved_since_capture')
    if age > TASK_REF_MAX_AGE_S:
        why.append('too_old')
    if age < 0.0:
        why.append('future_capture')
    return (not why), dict(age_s=round(age, 3), door_change_deg=round(dd, 5), why=why, max_age_s=TASK_REF_MAX_AGE_S, door_tol_deg=TASK_REF_DOOR_DEG)


class PhaseGuard:
    """禁止接觸鎖存一經發生即持續：
      - 有活動控制器 ⇒ 交給它的停止處理（forward）；
      - 沒有活動控制器（例：張爪／觀察）⇒ 無可驗證處置 ⇒ 模擬中止；
      - 任何新段開始前 may_start() 必須為真（鎖存後一律不得啟動新段）。"""
    def __init__(self):
        self.latched, self.event, self.aborted, self.blocked = None, None, None, []

    def on_latch(self, latch, active_kind):
        if self.latched is None:
            self.latched, self.event = dict(latch), dict(latch, active=active_kind)
        if active_kind is None and self.aborted is None:
            self.aborted = dict(why='forbidden_contact_without_active_controller', latch=self.latched)
        return 'forward' if active_kind is not None else 'abort'

    def may_start(self, phase):
        if self.latched is not None:
            self.blocked.append(phase)
            return False
        return True
