#!/usr/bin/env python3
"""PARK_FIXED 的**靜止閘門**與**保持監看**（純邏輯，可離線測試；執行端每個物理步呼叫一次）。

規格：evaluation/results/motm_speed/park_fixed_vs_motm_spec.md；接線：park_fixed_impl_plan.md
（Codex reviews/20261005_113349_reply.md 已併入）。

量測來源一律是**同一物理步**的真值底盤位姿與**實際寫進 API** 的底盤命令；速度由相鄰物理步的位姿差分／實際
時間差算（yaw 解包），不用可能停更的 twist，也不拿「已送零命令」代替實測停止。

  StaticGate  控制者仍是停車控制器時，判定「已停住」：連續 hold_s（= max(0.5 s, max_cmd_age_s)）的每一步都滿足
              線速度 ≤ 1 mm/s、yaw 速率 ≤ 0.01 rad/s、套用底盤命令各軸 ≤ 1e-6、位姿在入場容差內（≤ 10 mm、≤ 0.05 rad）、
              手臂仍收攏。通過即閂鎖，錨點＝通過當步的實測位姿（之後不更新）。通過後仍逐步回報 ok_now，供切換
              提交當步核「停車條件仍成立」。
  HoldMonitor 從**實際切換給全身**到**實際交還導航**，逐步核：線速度、yaw 速率、套用命令、相對錨點的平移 ≤ 1 mm、
              yaw ≤ 0.5°。第一次違規閂鎖（步、時間、相位、原因）；不重設錨點、不以連續計數延後。
              **邊界**（Codex 20261005_115350）：開始時承接切換前一物理步的位姿（prev），切換當步的速度才核得到；
              交還當步呼叫 update(cmd3=None)——核最後一個全身控制區間（位姿 k−1→k），不核導航首筆命令——再 close；
              只有交還時手臂已收攏才算正常結束（否則 INSUFFICIENT）。

證據不足（任一）：物理步不連續、時間不前進、非有限值 ⇒ StaticGate 重新起算；HoldMonitor 閂鎖 evidence_insufficient
（判定不得為 PASS），但不當成底盤移動。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _finite(*xs) -> bool:
    return all(math.isfinite(float(x)) for x in xs)


@dataclass
class ParkCfg:
    park_x: float
    park_y: float
    park_yaw: float
    hold_s: float = 0.5                  # 呼叫端給 max(0.5, max_cmd_age_s)
    entry_pos_tol: float = 0.010
    entry_yaw_tol: float = 0.05
    v_lin_max: float = 0.001             # m/s
    w_max: float = 0.01                  # rad/s
    cmd_max: float = 1e-6                # 套用底盤命令各軸（m/s、rad/s）
    dev_pos_max: float = 0.001           # 保持期相對錨點
    dev_yaw_max: float = math.radians(0.5)


class _Diff:
    """相鄰物理步差分；回傳 (ok, v_lin, w, why)。"""

    def __init__(self):
        self.prev = None

    def update(self, step, t, x, y, yaw):
        if not _finite(t, x, y, yaw):
            self.prev = None
            return False, None, None, 'nonfinite'
        p = self.prev
        self.prev = (int(step), float(t), float(x), float(y), float(yaw))
        if p is None:
            return False, None, None, 'no_prev'
        if int(step) != p[0] + 1:
            return False, None, None, f'step_gap {p[0]}→{step}'
        dt = float(t) - p[1]
        if not dt > 0.0:
            return False, None, None, f'time_not_increasing dt={dt}'
        v = math.hypot(float(x) - p[2], float(y) - p[3]) / dt
        w = abs(wrap(float(yaw) - p[4])) / dt
        return True, v, w, None


class StaticGate:
    def __init__(self, cfg: ParkCfg):
        self.cfg = cfg
        self.diff = _Diff()
        self.run_t0 = None
        self.passed = False
        self.pass_step = None
        self.pass_t = None
        self.anchor = None               # (x, y, yaw)
        self.window = None               # (t0, t1) of the passing window
        self.ok_now = False
        self.why_now = None
        self.n_evidence_reset = 0
        self.last = None

    def update(self, step, t, x, y, yaw, cmd3, arm_stowed: bool):
        c = self.cfg
        ok_d, v, w, why = self.diff.update(step, t, x, y, yaw)
        reasons = []
        if not ok_d:
            if why != 'no_prev':
                self.n_evidence_reset += 1
            reasons.append(why)
        else:
            if v > c.v_lin_max:
                reasons.append(f'v_lin {v:.6f} > {c.v_lin_max}')
            if w > c.w_max:
                reasons.append(f'yaw_rate {w:.6f} > {c.w_max}')
        if not _finite(*cmd3):
            reasons.append('cmd_nonfinite')
        elif max(abs(float(u)) for u in cmd3) > c.cmd_max:
            reasons.append(f'cmd {max(abs(float(u)) for u in cmd3):.3e} > {c.cmd_max}')
        if _finite(x, y, yaw):
            dp = math.hypot(float(x) - c.park_x, float(y) - c.park_y)
            dyaw = abs(wrap(float(yaw) - c.park_yaw))
            if dp > c.entry_pos_tol:
                reasons.append(f'entry_pos {dp:.4f} > {c.entry_pos_tol}')
            if dyaw > c.entry_yaw_tol:
                reasons.append(f'entry_yaw {dyaw:.4f} > {c.entry_yaw_tol}')
        if not arm_stowed:
            reasons.append('arm_not_stowed')
        self.ok_now = not reasons
        self.why_now = None if self.ok_now else '; '.join(reasons)
        self.last = {'step': int(step), 't': float(t), 'v_lin': v, 'yaw_rate': w}
        if not self.ok_now:
            self.run_t0 = None
            return self.state()
        if self.run_t0 is None:
            self.run_t0 = float(t)
        if (not self.passed) and float(t) - self.run_t0 >= c.hold_s - 1e-9:
            self.passed = True
            self.pass_step, self.pass_t = int(step), float(t)
            self.anchor = (float(x), float(y), float(yaw))
            self.window = (self.run_t0, float(t))
        return self.state()

    def state(self):
        return {'passed': self.passed, 'ok_now': self.ok_now, 'why_now': self.why_now,
                'pass_step': self.pass_step, 'pass_t': self.pass_t, 'anchor': self.anchor,
                'window': self.window, 'run_s': (None if self.run_t0 is None or self.last is None
                                                 else self.last['t'] - self.run_t0),
                'n_evidence_reset': self.n_evidence_reset}


class HoldMonitor:
    def __init__(self, cfg: ParkCfg, anchor, start_step: int, start_t: float, prev=None):
        """prev = 切換前一物理步的 (step, t, x, y, yaw)；給了才核得到切換當步的速度。"""
        self.cfg = cfg
        self.anchor = tuple(float(a) for a in anchor)
        self.start = (int(start_step), float(start_t))
        self.diff = _Diff()
        if prev is not None and _finite(*prev[1:]):
            self.diff.prev = (int(prev[0]), float(prev[1]), float(prev[2]), float(prev[3]), float(prev[4]))
        self.seeded = prev is not None
        self.handback_stowed = None
        self.violation = None            # 第一次違規（閂鎖）
        self.evidence_insufficient = None
        self.n = 0
        self.max = {'v_lin': 0.0, 'yaw_rate': 0.0, 'cmd': 0.0, 'dev_pos': 0.0, 'dev_yaw': 0.0}
        self.end = None

    def update(self, step, t, x, y, yaw, cmd3, phase=None):
        """回傳本步新發生的違規（只在第一次），否則 None。cmd3 = None ⇒ 本步不核命令（交還當步）。"""
        c = self.cfg
        self.n += 1
        ok_d, v, w, why = self.diff.update(step, t, x, y, yaw)
        if not ok_d and why != 'no_prev' and self.evidence_insufficient is None:
            self.evidence_insufficient = {'step': int(step), 't': float(t), 'why': why, 'phase': phase}
        reasons = []
        if ok_d:
            self.max['v_lin'] = max(self.max['v_lin'], v)
            self.max['yaw_rate'] = max(self.max['yaw_rate'], w)
            if v > c.v_lin_max:
                reasons.append(f'v_lin {v:.6f} > {c.v_lin_max}')
            if w > c.w_max:
                reasons.append(f'yaw_rate {w:.6f} > {c.w_max}')
        if cmd3 is None:
            pass
        elif _finite(*cmd3):
            m = max(abs(float(u)) for u in cmd3)
            self.max['cmd'] = max(self.max['cmd'], m)
            if m > c.cmd_max:
                reasons.append(f'cmd {m:.3e} > {c.cmd_max}')
        elif self.evidence_insufficient is None:
            self.evidence_insufficient = {'step': int(step), 't': float(t), 'why': 'cmd_nonfinite',
                                          'phase': phase}
        if _finite(x, y, yaw):
            dp = math.hypot(float(x) - self.anchor[0], float(y) - self.anchor[1])
            dyaw = abs(wrap(float(yaw) - self.anchor[2]))
            self.max['dev_pos'] = max(self.max['dev_pos'], dp)
            self.max['dev_yaw'] = max(self.max['dev_yaw'], dyaw)
            if dp > c.dev_pos_max:
                reasons.append(f'dev_pos {dp * 1e3:.3f} mm > {c.dev_pos_max * 1e3:.1f} mm')
            if dyaw > c.dev_yaw_max:
                reasons.append(f'dev_yaw {math.degrees(dyaw):.3f}° > {math.degrees(c.dev_yaw_max):.2f}°')
        if reasons and self.violation is None:
            self.violation = {'step': int(step), 't': float(t), 'phase': phase, 'why': '; '.join(reasons)}
            return self.violation
        return None

    def close(self, step, t, stowed=None):
        self.end = (int(step), float(t))
        self.handback_stowed = None if stowed is None else bool(stowed)

    def verdict(self):
        if self.violation is not None:
            return 'VIOLATION'
        if (self.evidence_insufficient is not None or self.end is None or self.n == 0
                or not self.seeded or self.handback_stowed is not True):
            return 'INSUFFICIENT'
        return 'PASS'

    def report(self):
        return {'anchor': self.anchor, 'start': self.start, 'end': self.end, 'n_steps': self.n,
                'seeded_with_pre_switch_pose': self.seeded, 'handback_stowed': self.handback_stowed,
                'verdict': self.verdict(), 'first_violation': self.violation,
                'evidence_insufficient': self.evidence_insufficient,
                'max': {'v_lin_mps': self.max['v_lin'], 'yaw_rate_rps': self.max['yaw_rate'],
                        'cmd_abs': self.max['cmd'], 'dev_pos_mm': self.max['dev_pos'] * 1e3,
                        'dev_yaw_deg': math.degrees(self.max['dev_yaw'])}}
