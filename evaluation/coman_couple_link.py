"""連接／解除的握手與**確認**（純邏輯，不含 ROS 與 Isaac，可離線測試）。

三件事分開，互不代表
--------------------
  **請求**  任務流程說「現在要解除」（release_requested）
  **執行**  呼叫解除函式（disable joint）——**這不是確認**
  **確認**  由執行端**讀回**固定關節狀態得知確實已解除

確認的條件（兩者都要）
  1. 讀回顯示關節已停用（或本來就沒有建立過連接）
  2. 距離「執行」至少 `min_steps_after_execute` 個物理步 ——
     同一步的讀回可能還沒反映物理端的狀態

正常解除需要**資格 ＋ 請求同時成立**；
**緊急解除不看資格也不看請求**，且要求**閂鎖到確認完成**才撤除。
每個階段各自記錄 `physics_step_id` 與模擬時間。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stamp:
    step: int
    t: float
    note: str = ''

    def as_list(self):
        return [int(self.step), float(self.t), self.note]


class CoupleLink:
    def __init__(self, min_steps_after_execute: int = 1):
        self.min_steps = int(min_steps_after_execute)
        self.attached_ever = False
        self.request: Stamp | None = None
        self.execute: Stamp | None = None
        self.confirm: Stamp | None = None
        self.emergency_latched = False
        self.emergency_stamp: Stamp | None = None
        self.kind: str = ''            # 'normal' | 'emergency'
        self.blocked: list = []        # 被擋下的嘗試，保留為證據

    # ---------------- 連接 ----------------
    def mark_attached(self, step: int, t: float) -> None:
        self.attached_ever = True
        self.attach_stamp = Stamp(step, t, 'attach')

    # ---------------- 請求 ----------------
    def request_release(self, step: int, t: float, source: str = 'task_flow') -> None:
        """任務流程的明確請求；**只記錄，不執行**。"""
        if self.request is None:
            self.request = Stamp(step, t, source)

    # ---------------- 執行 ----------------
    def may_release_now(self, capability: bool, step: int, t: float) -> bool:
        """正常解除：資格與請求**同一週期**都成立才允許。"""
        ok = bool(capability) and self.request is not None
        if not ok:
            self.blocked.append([int(step), float(t),
                                 f'capability={bool(capability)}, '
                                 f'requested={self.request is not None}'])
        return ok

    def mark_executed(self, step: int, t: float, kind: str) -> None:
        """記錄**呼叫解除函式**的時刻。呼叫本身不構成確認。"""
        if self.execute is None:
            self.execute = Stamp(step, t, kind)
            self.kind = kind

    # ---------------- 緊急 ----------------
    def emergency(self, step: int, t: float, reason: str) -> None:
        """不看資格、不看請求；閂鎖到確認完成。"""
        self.emergency_latched = True
        if self.emergency_stamp is None:
            self.emergency_stamp = Stamp(step, t, reason)

    # ---------------- 確認 ----------------
    def poll_confirm(self, joint_exists: bool, joint_enabled: bool,
                     step: int, t: float) -> bool:
        """由**讀回**決定是否確認已解除。

        `joint_exists=False` 只有在**從未建立連接**時才算已解除；
        建立過卻讀不到關節，視為讀回異常，不予確認。
        """
        if self.confirm is not None:
            return True
        if self.execute is None:
            return False                       # 還沒執行，談不上確認
        if step - self.execute.step < self.min_steps:
            return False                       # 同一步的讀回不採信
        if joint_exists:
            released = not joint_enabled
        else:
            released = not self.attached_ever
        if not released:
            return False
        self.confirm = Stamp(step, t, 'readback')
        self.emergency_latched = False          # 確認後才撤除緊急要求
        return True

    # ---------------- 輸出 ----------------
    def record(self) -> dict:
        f = lambda s: (s.as_list() if s is not None else None)
        return {
            'kind': self.kind,
            'attached_ever': self.attached_ever,
            'request': f(self.request),
            'execute': f(self.execute),
            'confirm': f(self.confirm),
            'emergency': f(self.emergency_stamp),
            'emergency_latched': self.emergency_latched,
            'blocked_attempts': self.blocked,
            'stamp_cols': ['physics_step_id', 'sim_time', 'note'],
            'semantics': ('execute = 呼叫解除函式；confirm = 讀回確認已解除。'
                          '兩者不可互相代表。'),
        }


# ------------------------------------------------------------------ 離線測試
def selftest() -> int:
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f'  {name:52s} {"ok" if cond else "**錯**"}')
        bad += not cond

    # 1 正常流程：資格 ＋ 請求
    L = CoupleLink()
    L.mark_attached(10, 0.10)
    check('未請求時不得執行正常解除', not L.may_release_now(True, 11, 0.11))
    L.request_release(12, 0.12)
    check('有請求但無資格 → 不得執行', not L.may_release_now(False, 13, 0.13))
    check('資格 ＋ 請求 → 允許執行', L.may_release_now(True, 14, 0.14))
    L.mark_executed(14, 0.14, 'normal')
    check('執行當下不得直接算確認',
          not L.poll_confirm(True, False, 14, 0.14) and L.confirm is None)
    check('讀回仍為啟用 → 不得確認',
          not L.poll_confirm(True, True, 15, 0.15))
    check('讀回已停用且已過一步 → 確認',
          L.poll_confirm(True, False, 15, 0.15) and L.confirm is not None)
    r = L.record()
    check('請求／執行／確認各自記步序與時間',
          r['request'][0] == 12 and r['execute'][0] == 14 and r['confirm'][0] == 15
          and r['request'][1] == 0.12)
    check('被擋下的嘗試保留為證據', len(r['blocked_attempts']) == 2)

    # 2 「呼叫過解除函式」不等於確認
    L2 = CoupleLink()
    L2.mark_attached(1, 0.01)
    L2.request_release(2, 0.02)
    L2.may_release_now(True, 3, 0.03)
    L2.mark_executed(3, 0.03, 'normal')
    for s in range(4, 9):
        L2.poll_confirm(True, True, s, s * 0.01)     # 讀回一直是啟用
    check('讀回未變 → 始終不確認', L2.confirm is None)

    # 3 建立過連接卻讀不到關節 → 視為讀回異常，不確認
    L3 = CoupleLink()
    L3.mark_attached(1, 0.01)
    L3.request_release(2, 0.02)
    L3.mark_executed(3, 0.03, 'normal')
    check('曾建立連接卻讀不到關節 → 不確認',
          not L3.poll_confirm(False, False, 5, 0.05) and L3.confirm is None)

    # 4 從未連接 → 讀不到關節即算已解除
    L4 = CoupleLink()
    L4.mark_executed(3, 0.03, 'normal')
    check('從未建立連接時，讀不到關節即算已解除',
          L4.poll_confirm(False, False, 5, 0.05))

    # 5 緊急：不看資格與請求，閂鎖到確認
    L5 = CoupleLink()
    L5.mark_attached(1, 0.01)
    L5.emergency(7, 0.07, 'monitor_failed_test')
    check('緊急不需資格與請求即可閂鎖',
          L5.emergency_latched and L5.request is None)
    L5.mark_executed(7, 0.07, 'emergency')
    check('緊急執行後未確認前維持閂鎖',
          not L5.poll_confirm(True, True, 8, 0.08) and L5.emergency_latched)
    check('確認後才撤除緊急要求',
          L5.poll_confirm(True, False, 9, 0.09) and not L5.emergency_latched)
    check('緊急的種類與時刻有記錄',
          L5.record()['emergency'][0] == 7 and L5.record()['kind'] == 'emergency')

    # 6 正常資格不得影響緊急
    L6 = CoupleLink()
    L6.mark_attached(1, 0.01)
    L6.emergency(5, 0.05, 'injected')
    check('緊急時即使資格為假也不阻擋（資格只管正常路徑）',
          L6.emergency_latched and not L6.may_release_now(False, 5, 0.05))

    print('連接／解除握手離線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(selftest())
