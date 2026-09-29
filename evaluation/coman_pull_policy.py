"""協同拉動的**任務語意決策**（純邏輯，不含 ROS，可離線測試）。

為什麼另立一支：既有 `wholebody_pregrasp.run()` 是為**固定目標**寫的 ——
誤差小於容差就回報完成。拉動任務若沿用，**到達接近點就會結束整個任務**。

本模組只決定「相位如何推進、任務何時真的結束、何時不得推進」，
不碰 QP、不碰命令。
"""
from __future__ import annotations

from dataclasses import dataclass

PHASES = ('APPROACH', 'ENGAGE_WAIT', 'PULL', 'HOLD', 'RELEASE_WAIT',
          'RETREAT', 'DONE')


@dataclass
class TaskState:
    """執行端回報的量測與旗標（求解器**不自行認定**）。"""
    sim_t: float
    state_age_s: float          # 這份狀態自身的年齡
    attached: bool
    handover_pass: bool
    hold_tracking_pass: bool
    decouple_confirmed: bool
    pos_err_m: float            # 對**當前相位目標**的誤差
    rot_err_rad: float
    cmd_max_abs: float          # 上一筆命令的最大分量（停滯判定用）
    retreat_signed_m: float = 0.0
    emergency: bool = False


class PullTaskPolicy:
    """相位推進與結束條件。**完成只在退出量達標之後**。"""

    def __init__(self, *, pull_duration_s, state_max_age_s=0.2,
                 retreat_clear_m=0.0233, stall_cmd=2e-3, stall_cycles=40,
                 tol_p=0.005, tol_r=0.02):
        self.pull_s = float(pull_duration_s)
        self.state_max_age = float(state_max_age_s)
        self.retreat_clear = float(retreat_clear_m)
        self.stall_cmd = float(stall_cmd)
        self.stall_cycles = int(stall_cycles)
        self.tol_p, self.tol_r = float(tol_p), float(tol_r)
        self.phase = 'APPROACH'
        self.t_pull0 = None
        self.n_stall = 0
        self.stamps = {}
        self.blocked = []           # 因狀態過期而未推進的紀錄
        self.abort = None           # **閂鎖**：中止一旦成立就不再改寫
        self.done = False

    def _mark(self, k, t):
        self.stamps.setdefault(k, t)

    def step(self, s: TaskState) -> dict:
        """回傳 {phase, done, abort, reason}。**不因接近點到達而完成。**"""
        out = {'phase': self.phase, 'done': self.done, 'abort': self.abort,
               'reason': ''}
        if self.abort is not None:
            out['reason'] = f'已中止（{self.abort}），不再推進'
            return out
        if self.done:
            out['reason'] = '已完成，不再推進'
            return out
        if s.emergency:
            self.phase, self.abort = 'DONE', 'emergency'
            return {**out, 'phase': 'DONE', 'abort': 'emergency',
                    'reason': '緊急處置，任務中止'}
        # **狀態過期 ⇒ 不推進相位**（也不完成、不中止；由上層決定是否停止）
        if s.state_age_s > self.state_max_age:
            self.blocked.append([round(s.sim_t, 4), round(s.state_age_s, 4),
                                 self.phase])
            out['reason'] = f'狀態過期 {s.state_age_s:.3f}s，不推進'
            return out

        # 停滯只在**需要移動的相位**判定；HOLD 命令本來就接近零
        if self.phase in ('APPROACH', 'PULL', 'RETREAT'):
            if s.cmd_max_abs < self.stall_cmd:
                self.n_stall += 1
            else:
                self.n_stall = 0
            if self.n_stall > self.stall_cycles:
                self.phase, self.abort = 'DONE', 'stall'
                return {**out, 'phase': 'DONE', 'abort': 'stall',
                        'reason': f'{self.phase} 相位命令連續 {self.n_stall} 週期近零'}
        else:
            self.n_stall = 0

        p = self.phase
        if p == 'APPROACH':
            # **到達接近點不是任務完成**，只是可以等待連接
            if s.pos_err_m < self.tol_p and s.rot_err_rad < self.tol_r \
                    and s.handover_pass:
                self.phase = 'ENGAGE_WAIT'
                self._mark('approach_reached', s.sim_t)
                out['reason'] = '到達接近點，等待連接（**任務未完成**）'
        elif p == 'ENGAGE_WAIT':
            if s.attached:
                self.phase = 'PULL'
                self.t_pull0 = s.sim_t
                self._mark('attached', s.sim_t)
        elif p == 'PULL':
            if self.t_pull0 is not None and s.sim_t - self.t_pull0 >= self.pull_s:
                self.phase = 'HOLD'
                self._mark('pull_done', s.sim_t)
        elif p == 'HOLD':
            if s.hold_tracking_pass:
                self.phase = 'RELEASE_WAIT'
                self._mark('hold_pass', s.sim_t)
        elif p == 'RELEASE_WAIT':
            # **解除確認之前不得退出**
            if s.decouple_confirmed:
                self.phase = 'RETREAT'
                self._mark('decouple_confirmed', s.sim_t)
            else:
                out['reason'] = '尚未確認解除，維持保持命令'
        elif p == 'RETREAT':
            if s.retreat_signed_m >= self.retreat_clear:
                self.phase = 'DONE'
                self._mark('retreat_done', s.sim_t)
                self.done = True
                out['done'] = True
                out['reason'] = '退出量達標，任務完成'
        out['phase'] = self.phase
        return out


# ------------------------------------------------------------------ 離線測試
def selftest() -> int:
    bad = 0

    def check(name, cond):
        nonlocal bad
        print(f'  {name:52s} {"ok" if cond else "**錯**"}')
        bad += not cond

    def st(t, **kw):
        base = dict(sim_t=t, state_age_s=0.01, attached=False,
                    handover_pass=False, hold_tracking_pass=False,
                    decouple_confirmed=False, pos_err_m=0.5, rot_err_rad=0.5,
                    cmd_max_abs=0.02, retreat_signed_m=0.0)
        base.update(kw)
        return TaskState(**base)

    # 1 到達接近點**不得**結束任務
    P = PullTaskPolicy(pull_duration_s=4.0)
    r = P.step(st(0.0, pos_err_m=0.001, rot_err_rad=0.001, handover_pass=True))
    check('到達接近點 → 進入 ENGAGE_WAIT，任務未完成',
          r['phase'] == 'ENGAGE_WAIT' and not r['done'])
    for t in (0.05, 0.10, 0.15):
        r = P.step(st(t, pos_err_m=0.0005, rot_err_rad=0.0005))
    check('在接近點停留多週期仍不完成', not r['done'] and P.phase == 'ENGAGE_WAIT')

    # 2 連接 → PULL → HOLD
    r = P.step(st(0.20, attached=True))
    check('連接後進入 PULL', r['phase'] == 'PULL')
    r = P.step(st(2.0, attached=True))
    check('拉動未滿時長不得進入 HOLD', r['phase'] == 'PULL')
    r = P.step(st(4.25, attached=True))
    check('拉動滿時長 → HOLD', r['phase'] == 'HOLD')
    r = P.step(st(4.30, attached=True, cmd_max_abs=0.0))
    check('HOLD 命令近零**不得**判為停滯', r['abort'] is None)

    # 3 解除確認前不得退出
    r = P.step(st(4.35, attached=True, hold_tracking_pass=True))
    check('保持合格 → RELEASE_WAIT', r['phase'] == 'RELEASE_WAIT')
    for t in (4.40, 4.45, 4.50):
        r = P.step(st(t, attached=False))
    check('未確認解除 → 不得進入 RETREAT',
          r['phase'] == 'RELEASE_WAIT' and '尚未確認解除' in r['reason'])
    r = P.step(st(4.55, attached=False, decouple_confirmed=True))
    check('確認解除後才進入 RETREAT', r['phase'] == 'RETREAT')

    # 4 退出量達標才完成
    r = P.step(st(4.60, decouple_confirmed=True, retreat_signed_m=0.009))
    check('退出量不足不得完成', not r['done'] and r['phase'] == 'RETREAT')
    r = P.step(st(4.70, decouple_confirmed=True, retreat_signed_m=0.0233))
    check('退出量達標 → 任務完成', r['done'] and r['phase'] == 'DONE')

    # 5 狀態過期不得推進
    P2 = PullTaskPolicy(pull_duration_s=4.0)
    r = P2.step(st(0.0, state_age_s=0.5, pos_err_m=0.001, rot_err_rad=0.001,
                   handover_pass=True))
    check('狀態過期 → 不推進相位',
          r['phase'] == 'APPROACH' and '狀態過期' in r['reason'])
    check('過期事件被記錄', len(P2.blocked) == 1)

    # 6 需移動相位的停滯仍要抓
    P3 = PullTaskPolicy(pull_duration_s=4.0, stall_cycles=3)
    first = None
    for k in range(6):
        r = P3.step(st(0.05 * k, cmd_max_abs=0.0))
        if first is None and r['abort']:
            first = r
    check('APPROACH 命令連續近零 → 判停滯中止', first is not None
          and first['abort'] == 'stall')
    check('中止**閂鎖**：其後各步仍回報同一中止',
          r['abort'] == 'stall' and '已中止' in r['reason'])

    # 7 緊急立即中止
    P4 = PullTaskPolicy(pull_duration_s=4.0)
    r = P4.step(st(1.0, emergency=True))
    check('緊急 → 立即中止', r['abort'] == 'emergency' and r['phase'] == 'DONE')

    print('拉動任務語意離線測試：' + ('全部通過' if bad == 0 else f'**{bad} 項失敗**'))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(selftest())
