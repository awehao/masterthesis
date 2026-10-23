#!/usr/bin/env python3
"""K3-M 受控暫停 r2：命令停止與實際停止分開判定（純函式；Codex 20261024_014938 第 2 節）。door_k3m_pause.py 不改。

WindowSpeed：由量測樣本（同步狀態：步號、時間、底盤 x／y／yaw、門角、實測手臂 6 關節）計算**位姿差分窗口速度**——
  以最新樣本與「時間上至少 win_s 之前」的最近樣本相減；窗口內任兩相鄰樣本步號必須遞增、時間間隔 ≤ max_gap_s、全有限，
  否則該時刻「無值」（不補零、不湊連續）。沿用長窗診斷的停止判準數值（底盤 1 mm/s、0.005 rad/s、門 0.002 rad/s、
  實測手臂 1e-3 rad/s），作完成判定用途，**不替換原門速安全保護**。

PauseSupervisor：
  命令段（沿用 door_k3m_pause.PauseSequencer 的狀態名）：
    LOWSPEED → ZERO_PUB（實際低速連續 hold_s）→ CONFIRM（發布的底盤命令已為 0）→ STOPPED（套用回報：連續、不同的實際物理步，
    來源序號 ≥ 零速第一筆，見 PauseSequencer.applied）→ 停止發布；
  實際停止：從 ZERO_PUB 起，**每個週期**都要實際低速且資料有效；CONFIRM 與 STOPPED 後觀察 observe_s（10 s）期間亦然。
    任一時刻不低速 ⇒ FAIL 'moved_*'；資料無效 ⇒ FAIL 'measurement_invalid'；停止發布前求解失敗 ⇒ FAIL 'solve_failed'；
    到達 STOPPED 前超過 timeout_s ⇒ FAIL 'timeout'（觀察期不計入逾時）。
  結果分報（outcome()）：
    command_stop_confirmed（到 STOPPED）、physical_stop_confirmed（STOPPED 後觀察滿 observe_s 且全程低速）、
    observation_covered_s、fail_why、各轉換時刻。觀察未滿（例如外部結束）⇒ physical_stop_confirmed＝False、標 observation_incomplete。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from door_k3m_pause import PauseCfg, PauseSequencer


@dataclass
class StopTh:
    base_lin: float = 0.001
    base_ang: float = 0.005
    door: float = 0.002
    arm: float = 1e-3


def _fin(*xs):
    try:
        return all(math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


class WindowSpeed:
    def __init__(self, win_s=0.1, max_gap_s=0.035, keep_s=1.0):
        self.win_s, self.max_gap_s, self.keep_s = win_s, max_gap_s, keep_s
        self.buf = deque()
        self.n_bad = 0

    def feed(self, step, t, x, y, yaw, q_door, q_arm):
        """回傳 False 表示樣本被拒（非有限、步號或時間不遞增）；被拒的樣本會讓之後跨越它的窗口無值（斷點）。"""
        ok = _fin(step, t, x, y, yaw, q_door, *list(q_arm)) and len(list(q_arm)) == 6
        if ok and self.buf and not (step > self.buf[-1][0] and t > self.buf[-1][1]):
            ok = False
        if not ok:
            self.n_bad += 1
            self.buf.append(None)                               # 斷點
        else:
            self.buf.append((int(step), float(t), float(x), float(y), float(yaw), float(q_door), [float(v) for v in q_arm]))
        while len(self.buf) > 2 and self.buf[0] is not None and self.buf[-1] is not None and self.buf[-1][1] - self.buf[0][1] > self.keep_s:
            self.buf.popleft()
        while len(self.buf) > 200:
            self.buf.popleft()
        return ok

    def speeds(self):
        """(底盤線速, 底盤角速, 門速, 手臂最大關節速) 或 None。"""
        if not self.buf or self.buf[-1] is None:
            return None
        b = self.buf[-1]
        a = None
        prev = b
        for s in reversed(list(self.buf)[:-1]):
            if s is None or prev[1] - s[1] > self.max_gap_s or not (s[0] < prev[0]):
                return None
            prev = s
            if b[1] - s[1] >= self.win_s - 1e-9:
                a = s
                break
        if a is None:
            return None
        dt = b[1] - a[1]
        dyaw = math.atan2(math.sin(b[4] - a[4]), math.cos(b[4] - a[4]))
        return (math.hypot(b[2] - a[2], b[3] - a[3]) / dt, abs(dyaw) / dt, abs(b[5] - a[5]) / dt,
                max(abs(u - v) for u, v in zip(b[6], a[6])) / dt)

    @staticmethod
    def low(v, th: StopTh):
        return v is not None and v[0] <= th.base_lin and v[1] <= th.base_ang and v[2] <= th.door and v[3] <= th.arm


class PauseSupervisor:
    def __init__(self, cfg: PauseCfg | None = None, th: StopTh | None = None, observe_s=10.0, timeout_s=8.0):
        c = cfg or PauseCfg()
        c.observe_s, c.timeout_s = observe_s, timeout_s
        self.c, self.th = c, th or StopTh()
        self.seq = PauseSequencer(c)
        self.ws = WindowSpeed()
        self.t_first_low = None
        self.events = []
        self.last_speeds = None

    @property
    def state(self):
        return self.seq.state

    def feed_state(self, step, t, x, y, yaw, q_door, q_arm):
        return self.ws.feed(step, t, x, y, yaw, q_door, q_arm)

    def _fail(self, why, t):
        self.seq._fail(why, t)
        self.events.append((t, 'FAIL', why))

    def cycle(self, t, solve_ok, base_cmd_zero):
        """每控制週期呼叫（在 feed_state 之後）。回傳 {'publish', 'zero'}。"""
        s = self.seq
        if s.t0 is None:
            s.t0 = t
        if s.state in ('DONE', 'FAIL'):
            return {'publish': False, 'zero': False}
        v = self.ws.speeds()
        self.last_speeds = v
        if v is None:
            if s.state == 'LOWSPEED':
                s.since = None                                    # 低速計時從頭來；不判失敗（暫停起始資料可能尚未成窗）
                return {'publish': True, 'zero': False}
            self._fail('measurement_invalid', t)
            return {'publish': False, 'zero': False}
        low = WindowSpeed.low(v, self.th)
        if s.state in ('LOWSPEED', 'ZERO_PUB', 'CONFIRM') and not solve_ok:
            self._fail('solve_failed', t)
            return {'publish': False, 'zero': False}
        if s.state in ('LOWSPEED', 'ZERO_PUB', 'CONFIRM') and t - s.t0 > self.c.timeout_s + 1e-9:
            self._fail('timeout', t)
            return {'publish': False, 'zero': False}
        if s.state == 'LOWSPEED':
            if low:
                self.t_first_low = t if self.t_first_low is None else self.t_first_low
                s.since = t if s.since is None else s.since
                if t - s.since >= self.c.hold_s - 1e-9:
                    s.state = 'ZERO_PUB'
                    s.log.append((t, 'ZERO_PUB', None))
            else:
                s.since = None
            return {'publish': True, 'zero': s.state == 'ZERO_PUB'}
        if not low:                                               # ZERO_PUB／CONFIRM／STOPPED：全程必須實際低速
            self._fail({'ZERO_PUB': 'moved_during_zero_pub', 'CONFIRM': 'moved_during_confirm',
                        'STOPPED': 'moved_after_stop'}[s.state], t)
            return {'publish': False, 'zero': False}
        if s.state == 'ZERO_PUB':
            if base_cmd_zero:
                s.state = 'CONFIRM'
                s.log.append((t, 'CONFIRM', None))
            return {'publish': True, 'zero': True}
        if s.state == 'CONFIRM':
            return {'publish': True, 'zero': True}
        if s.state == 'STOPPED':
            if t - s.t_stop >= self.c.observe_s - 1e-9:
                s.state = 'DONE'
                s.log.append((t, 'DONE', None))
            return {'publish': False, 'zero': False}
        return {'publish': False, 'zero': False}

    def note_published(self, seq, zero):
        self.seq.note_published(seq, zero)

    def applied(self, phys_step, t, base3, arm6, src_seq):
        """套用回報。CONFIRM 中交給 PauseSequencer 計數；STOPPED 後（停止發布）任何非零套用值，或來源序號 ≥ 0 但早於零速第一筆
        （舊命令重新生效）⇒ FAIL 'old_command_after_stop'。來源序號 −1（逾時減速／停止）只要值為零就接受。"""
        s = self.seq
        if s.state == 'STOPPED':
            try:
                vals = [abs(float(v)) for v in list(base3) + list(arm6)]
            except (TypeError, ValueError):
                return self._fail('applied_invalid', t)
            src_old = src_seq is not None and src_seq >= 0 and (s.zero_seq0 is None or src_seq < s.zero_seq0)
            if src_old or max(vals) > self.c.applied_tol:
                return self._fail('old_command_after_stop', t)
            return
        self.seq.applied(phys_step, t, base3, arm6, src_seq)

    def outcome(self, t_end=None):
        s = self.seq
        tr = {name: next((x[0] for x in s.log if x[1] == name), None) for name in ('ZERO_PUB', 'CONFIRM', 'STOPPED', 'DONE', 'FAIL')}
        cmd_ok = tr['STOPPED'] is not None
        covered = None if tr['STOPPED'] is None else ((tr['DONE'] if tr['DONE'] is not None else (t_end if t_end is not None else tr['STOPPED']))
                                                       - tr['STOPPED'])
        phys_ok = s.state == 'DONE'
        out = {'state': s.state, 'fail_why': s.why, 'transitions': tr, 'first_low_t': self.t_first_low,
               'command_stop_confirmed': cmd_ok, 'physical_stop_confirmed': phys_ok,
               'observation_required_s': self.c.observe_s, 'observation_covered_s': covered,
               'thresholds': vars(self.th), 'n_rejected_samples': self.ws.n_bad}
        if cmd_ok and not phys_ok and s.state not in ('FAIL',):
            out['observation_incomplete'] = True
        return out
