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


CAPABILITY_SOURCES = ('state_machine', 'diagnostic')


def evaluate_capability(mode: str, machine_allowed: bool = False,
                        machine_wired: bool = False,
                        diagnostic_allowed: bool = False) -> tuple[bool, str]:
    """決定釋放資格，並回報**資格來源**。

    * `mode='production'`：資格**只能**來自狀態機的 normal_release_allowed。
      狀態機尚未接妥（`machine_wired=False`）⇒ **禁止正常釋放**，
      不得退回「已連接且位姿有效」這類簡化條件。
    * `mode='diagnostic'`：明確標記的診斷資格，只用於零行程接線測試，
      **與正式操作隔離**，不得用來宣稱操作合格。
    """
    if mode == 'diagnostic':
        return bool(diagnostic_allowed), 'diagnostic'
    if mode != 'production':
        return False, f'unknown_mode:{mode}'
    if not machine_wired:
        return False, 'state_machine_not_wired'
    return bool(machine_allowed), 'state_machine'


def freeze_target(last_applied_q, measured_q):
    """決定保持目標：**只能用最後一次實際套用的命令設定點**。

    不得改用實測角 —— 實測與命令之間的穩態差是維持負載所需的追蹤誤差，
    把它一次抽掉等於在命令上製造跳變。
    尚無任何已套用命令時**拒絕凍結**，不以實測角替代。

    回傳 (target, source, delta_mrad)；拒絕時 target 為 None。
    """
    if last_applied_q is None:
        return None, 'refused_no_applied_cmd', None
    tgt = [float(v) for v in last_applied_q]
    delta = None
    if measured_q is not None and len(measured_q) == len(tgt):
        delta = [round(1000.0 * (t_ - float(m_)), 4)
                 for t_, m_ in zip(tgt, measured_q)]
    return tgt, 'last_applied_command', delta


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
        self.capability_source: str = ''

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
    def may_release_now(self, capability: bool, step: int, t: float,
                        source: str = 'unspecified') -> bool:
        """正常解除：資格與請求**同一週期**都成立才允許。

        `source` 記錄資格來源（state_machine／diagnostic／未接妥的原因），
        讓趟後可以分辨這是正式資格還是診斷資格。
        """
        ok = bool(capability) and self.request is not None
        self.capability_source = source
        if not ok:
            self.blocked.append([int(step), float(t),
                                 f'capability={bool(capability)}, '
                                 f'requested={self.request is not None}, '
                                 f'source={source}'])
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
            'capability_source': self.capability_source,
            'stamp_cols': ['physics_step_id', 'sim_time', 'note'],
            'semantics': ('execute = 呼叫解除函式；confirm = **停用屬性讀回確認**'
                          '（JointEnabledAttr=False 並經過至少一個物理步）。'
                          '兩者不可互相代表；confirm **不等同**獨立證明物理約束已卸載。'),
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

    # 7 正式路徑未接妥狀態機 ⇒ 禁止正常釋放，不得退回簡化條件
    ok, src = evaluate_capability('production', machine_allowed=True,
                                  machine_wired=False)
    check('狀態機未接妥 → 正式資格為假且標明原因',
          not ok and src == 'state_machine_not_wired')
    ok, src = evaluate_capability('production', machine_allowed=False,
                                  machine_wired=True)
    check('狀態機已接但不允許 → 資格為假', not ok and src == 'state_machine')
    ok, src = evaluate_capability('production', machine_allowed=True,
                                  machine_wired=True)
    check('狀態機已接且允許 → 資格為真', ok and src == 'state_machine')
    ok, src = evaluate_capability('diagnostic', diagnostic_allowed=True)
    check('診斷資格獨立標記', ok and src == 'diagnostic')
    L7 = CoupleLink()
    L7.mark_attached(1, 0.01)
    L7.request_release(2, 0.02)
    cap, src = evaluate_capability('production', machine_allowed=True,
                                   machine_wired=False)
    check('未接妥狀態機時正式釋放被擋下',
          not L7.may_release_now(cap, 3, 0.03, src) and L7.execute is None)
    check('被擋原因記錄資格來源',
          'state_machine_not_wired' in L7.record()['blocked_attempts'][-1][2])

    # 8 凍結目標只能取最後實際套用的命令設定點
    tgt, src, delta = freeze_target(None, [1.0, 2.0])
    check('尚無已套用命令 → 拒絕凍結，不以實測角替代',
          tgt is None and src == 'refused_no_applied_cmd')
    applied = [0.0, 1.05390, 1.72386, 0.0, -0.5, 0.0]
    measured = [0.0, 1.05175, 1.72459, 0.0, -0.5, 0.0]
    tgt, src, delta = freeze_target(applied, measured)
    check('凍結目標等於最後套用命令，不等於實測角',
          tgt == applied and tgt != measured and src == 'last_applied_command')
    check('記錄命令與實測之差（joint2 約 +2.15 mrad）',
          abs(delta[1] - 2.15) < 0.01 and abs(delta[2] + 0.73) < 0.01)

    # 9 超力解除也要走同一套紀錄，且不得偽裝成正常釋放
    L8 = CoupleLink()
    L8.mark_attached(100, 1.00)
    L8.emergency(110, 1.10, 'stop_contact_force')
    L8.mark_executed(110, 1.10, 'emergency_contact_force')
    check('超力解除：緊急閂鎖並記錄執行',
          L8.emergency_latched and L8.record()['execute'][0] == 110)
    check('超力解除不得出現正常請求紀錄', L8.record()['request'] is None)
    check('超力解除確認前維持閂鎖',
          not L8.poll_confirm(True, True, 111, 1.11) and L8.emergency_latched)
    check('超力解除經讀回確認後撤除閂鎖',
          L8.poll_confirm(True, False, 112, 1.12) and not L8.emergency_latched)
    check('停止原因保留在種類欄位',
          L8.record()['kind'] == 'emergency_contact_force')

    print('連接／解除握手離線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(selftest())
