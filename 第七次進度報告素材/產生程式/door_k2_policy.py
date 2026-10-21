#!/usr/bin/env python3
"""K2 相位機與監看（純邏輯，可離線測試；ROS 節點 door_k2_task_node.py 只負責收量測、執行動作）。

規格：evaluation/results/specs/K2_door_grasp_spec.md（draft-2＋draft-3）。
相位：WAIT_HANDOVER → UNFOLD → ALIGN_PRE（目標位置自交棒時 TCP 以 0.020 m/s 斜坡到 s = 0.10）→ ALIGN_HOLD（s = 0.10、誤差 < 5 mm）
      → APPROACH（s 0.10 → 0，0.020 m/s；0.030 只記檢查點）→ ENGAGE（誤差 < 2 mm 持續 0.5 s ⇒ **凍結把手位姿**、1.0 s 閉爪斜坡、
      雙指持續接觸證據 2.0 s ⇒ 夾持成立）→ HOLD（3.0 s）→ RELEASE（張爪；每指 < 0.1 N 且張開 ≥ 80% 持續 0.5 s）
      → RETREAT（s 0 → 0.10，0.020 m/s；s = 0.10 且誤差 < 5 mm）→ DONE。
**閉爪開始即凍結把手位姿**（不再跟實測把手走：motm_cal20_074420 正回授把抽屜拉開 25.69 mm 的教訓）。
監看自 UNFOLD 起到 DONE 全程生效；任一中止 ⇒ ABORTED：凍結目標斜坡、閉爪斜坡與相位，夾爪保持中止當下命令，
不放爪、不退離、停止發布新目標並要求停止求解器；觀察 3 s 模擬時間後結束（牆鐘上限由節點負責）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class K2Config:
    approach_rate: float = 0.020
    s_pre: float = 0.10
    s_check: float = 0.030
    pre_tol_m: float = 0.005
    grip_tol_m: float = 0.002
    grip_settle_s: float = 0.5
    close_ramp_s: float = 1.0
    contact_min_n: float = 0.5
    attach_hold_s: float = 2.0
    hold_s: float = 3.0
    release_max_n: float = 0.1
    release_open_frac: float = 0.8
    release_hold_s: float = 0.5
    finger_open: float = 0.0089
    finger_closed: float = 0.0
    # 中止門檻
    door_abort_deg: float = 2.0
    door_abort_rate: float = 0.05
    drift_abort_m: float = 0.005
    finger_loss_s: float = 0.2
    contact_abort_n: float = 0.1
    wrist_abort_n: float = 30.0
    base_drift_m: float = 0.020
    base_drift_yaw_rad: float = math.radians(2.0)
    stale_s: float = 0.2
    solver_stale_s: float = 0.5
    post_abort_observe_s: float = 3.0
    timeouts: dict = field(default_factory=lambda: {
        'WAIT_HANDOVER': 30.0, 'UNFOLD': 30.0, 'ALIGN_PRE': 20.0, 'ALIGN_HOLD': 20.0, 'APPROACH': 15.0,
        'ENGAGE': 10.0, 'RELEASE': 10.0, 'RETREAT': 15.0})


ACTIVE_MONITOR = ('UNFOLD', 'ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH', 'ENGAGE', 'HOLD', 'RELEASE', 'RETREAT')
FINGER_BAR_ALLOWED = ('ENGAGE', 'HOLD', 'RELEASE')


@dataclass
class Meas:
    """一個控制週期的量測快照（節點組裝）。缺值用 None；非有限值視同缺。"""
    t: float
    owner: str | None = None             # 執行端控制者
    unfold_done: bool = False            # 展開節點已回報完成（UNFOLD_DONE／RELEASED）
    solver_up: bool = False
    solver_failed: str | None = None
    solver_status_age: float | None = None
    state_age: float | None = None       # /door/sync_state 年齡
    contacts_age: float | None = None
    q_door: float | None = None          # rad
    qdot_door: float | None = None
    H: np.ndarray | None = None          # 實測把手位姿 4×4
    T_tcp: np.ndarray | None = None      # 實測 TCP 4×4
    finger_bar_min: tuple | None = None  # 窗口內每指對橫桿接觸力最小值
    finger_bar_max: tuple | None = None
    unexpected_max: float | None = None
    unexpected_who: str | None = None
    wall_max: float | None = None
    wrist_max: float | None = None
    contacts_invalid: int | None = None
    contact_evidence_ok: bool = False    # 執行端逐物理步連續接觸證據（physics_contact_hold）
    finger_q: tuple | None = None        # 兩指實測位置
    base_drift_m: float | None = None
    base_drift_yaw: float | None = None


def _finite(*xs):
    for x in xs:
        if x is None:
            return False
        a = np.asarray(x, float)
        if not np.isfinite(a).all():
            return False
    return True


class K2Policy:
    def __init__(self, cfg: K2Config, T_HG: np.ndarray, a_H=(0.0, -1.0, 0.0)):
        self.c = cfg
        self.T_HG = np.asarray(T_HG, float)
        self.a_H = np.asarray(a_H, float)
        self.phase = 'WAIT_HANDOVER'
        self.t_phase = None
        self.events = []
        self.abort = None
        self.t_abort = None
        self.s = cfg.s_pre
        self.pre_p = None                # ALIGN_PRE 斜坡中的目標位置
        self.H_frozen = None
        self.G_T_H = None
        self.grip_cmd = cfg.finger_open
        self.close_t0 = None
        self.settle_t0 = None
        self.release_t0 = None
        self.loss_t0 = [None, None]
        self.last_target = None
        self.checkpoint_logged = False
        self.max_abs_q = 0.0
        self.done = False

    # ------------------------------------------------------------ 幾何
    def target(self, H, s):
        T = np.eye(4)
        T[:3, 3] = s * self.a_H
        return H @ T @ self.T_HG

    def _go(self, ph, t, **kw):
        self.events.append({'t': t, 'from': self.phase, 'to': ph, **kw})
        self.phase, self.t_phase = ph, t

    def _abort(self, why, t, **kw):
        if self.abort is None:
            self.abort, self.t_abort = why, t
            self.events.append({'t': t, 'abort': why, 'phase': self.phase, **kw})
            self.phase = 'ABORTED'

    # ------------------------------------------------------------ 監看
    def monitor(self, m: Meas):
        c = self.c
        if self.phase not in ACTIVE_MONITOR:
            return
        if not _finite(m.state_age, m.contacts_age) or m.state_age > c.stale_s or m.contacts_age > c.stale_s:
            return self._abort('monitor_stale_or_missing', m.t, state_age=m.state_age, contacts_age=m.contacts_age)
        if not _finite(m.q_door, m.qdot_door, m.H, m.T_tcp, m.finger_bar_min, m.finger_bar_max, m.unexpected_max, m.wall_max,
                       m.wrist_max, m.base_drift_m, m.base_drift_yaw) or m.contacts_invalid is None or m.contacts_invalid > 0:
            return self._abort('monitor_nonfinite_or_invalid', m.t)
        self.max_abs_q = max(self.max_abs_q, abs(m.q_door))
        if abs(m.q_door) > math.radians(c.door_abort_deg) or abs(m.qdot_door) > c.door_abort_rate:
            return self._abort('door_moving', m.t, q_deg=math.degrees(m.q_door), qdot=m.qdot_door)
        if m.unexpected_max > c.contact_abort_n:
            return self._abort('unexpected_contact', m.t, n=m.unexpected_max, who=m.unexpected_who)
        if m.wall_max > c.contact_abort_n:
            return self._abort('wall_contact', m.t, n=m.wall_max)
        if self.phase not in FINGER_BAR_ALLOWED and max(m.finger_bar_max) > c.contact_abort_n:
            return self._abort('finger_bar_outside_allowed_phase', m.t, n=max(m.finger_bar_max))
        if m.wrist_max >= c.wrist_abort_n:
            return self._abort('wrist_force', m.t, n=m.wrist_max)
        if m.base_drift_m > c.base_drift_m or abs(m.base_drift_yaw) > c.base_drift_yaw_rad:
            return self._abort('base_drift', m.t, d=m.base_drift_m, yaw=m.base_drift_yaw)
        if m.solver_failed:
            return self._abort('solver_failed', m.t, detail=m.solver_failed)
        if m.solver_up and (m.solver_status_age is None or m.solver_status_age > c.solver_stale_s):
            return self._abort('solver_status_stale', m.t, age=m.solver_status_age)
        to = c.timeouts.get(self.phase)
        if to is not None and self.t_phase is not None and m.t - self.t_phase > to:
            return self._abort(f'timeout:{self.phase}', m.t)

    # ------------------------------------------------------------ 推進
    def step(self, m: Meas, dt: float):
        """回傳動作 dict：target（4×4 或 None＝不發布）、grip、request_handover、solver_start、stop_solver、finish。"""
        c = self.c
        act = {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False,
               'stop_solver': False, 'finish': False}
        if self.t_phase is None:
            self.t_phase = m.t
        if self.phase == 'WAIT_HANDOVER':
            if m.owner == 'wholebody':
                self._go('UNFOLD', m.t)
            else:
                act['request_handover'] = True
                to = c.timeouts['WAIT_HANDOVER']
                if m.t - self.t_phase > to:
                    self._abort('timeout:WAIT_HANDOVER', m.t)
        self.monitor(m)
        if self.phase == 'ABORTED':
            act['stop_solver'] = True
            act['grip'] = self.grip_cmd                   # 保持中止當下命令（不放爪、不續閉）
            if m.t - self.t_abort >= c.post_abort_observe_s:
                act['finish'] = True
            return act
        if self.phase == 'UNFOLD':
            if m.unfold_done and _finite(m.T_tcp, m.H):
                self.pre_p = m.T_tcp[:3, 3].copy()
                self._go('ALIGN_PRE', m.t)
        ph0 = self.phase                                   # 本輪開始時的相位：相位切換後下一輪才評估新相位
        if self.phase in ('ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH'):
            Tt = self.target(m.H, c.s_pre if self.phase != 'APPROACH' else self.s)
            if self.phase == 'ALIGN_PRE':
                d = Tt[:3, 3] - self.pre_p
                n = float(np.linalg.norm(d))
                stepm = c.approach_rate * dt
                self.pre_p = Tt[:3, 3].copy() if n <= stepm else self.pre_p + d / n * stepm
                T = Tt.copy()
                T[:3, 3] = self.pre_p
                if n <= stepm:
                    self._go('ALIGN_HOLD', m.t)
            else:
                T = Tt
            act['target'] = T
            act['solver_start'] = True
            err = float(np.linalg.norm(m.T_tcp[:3, 3] - T[:3, 3]))
            if ph0 == 'ALIGN_HOLD' and m.solver_up and err < c.pre_tol_m:
                self._go('APPROACH', m.t, err=err)
                self.s = c.s_pre
            elif ph0 == 'APPROACH':
                self.s = max(0.0, self.s - c.approach_rate * dt)
                if not self.checkpoint_logged and self.s <= c.s_check:
                    self.checkpoint_logged = True
                    self.events.append({'t': m.t, 'checkpoint_s': c.s_check, 'err': err})
                if self.s <= 0.0:
                    e0 = float(np.linalg.norm(m.T_tcp[:3, 3] - self.target(m.H, 0.0)[:3, 3]))
                    if e0 < c.grip_tol_m:
                        self.settle_t0 = self.settle_t0 or m.t
                        if m.t - self.settle_t0 >= c.grip_settle_s:
                            self.H_frozen = np.asarray(m.H, float).copy()        # **閉爪開始即凍結把手位姿**
                            self.close_t0 = m.t
                            self._go('ENGAGE', m.t, err=e0)
                    else:
                        self.settle_t0 = None
        elif self.phase == 'ENGAGE':
            frac = min(1.0, (m.t - self.close_t0) / c.close_ramp_s)
            self.grip_cmd = c.finger_open + frac * (c.finger_closed - c.finger_open)
            act['target'] = self.target(self.H_frozen, 0.0)
            if frac >= 1.0 and m.contact_evidence_ok:
                self.G_T_H = np.linalg.inv(m.T_tcp) @ m.H
                self._go('HOLD', m.t)
        elif self.phase == 'HOLD':
            act['target'] = self.target(self.H_frozen, 0.0)
            drift = float(np.linalg.norm((m.T_tcp @ self.G_T_H)[:3, 3] - m.H[:3, 3]))
            if drift > c.drift_abort_m:
                self._abort('grasp_drift', m.t, drift=drift)
            for k in range(2):
                if m.finger_bar_min[k] < c.contact_min_n:
                    self.loss_t0[k] = self.loss_t0[k] if self.loss_t0[k] is not None else m.t
                    if m.t - self.loss_t0[k] >= c.finger_loss_s:
                        self._abort('finger_contact_lost', m.t, finger=k + 1)
                else:
                    self.loss_t0[k] = None
            if self.phase == 'HOLD' and m.t - self.t_phase >= c.hold_s:
                self.grip_cmd = c.finger_open
                self._go('RELEASE', m.t)
        elif self.phase == 'RELEASE':
            act['target'] = self.target(self.H_frozen, 0.0)
            self.grip_cmd = c.finger_open
            opened = _finite(m.finger_q) and min(m.finger_q) >= c.release_open_frac * c.finger_open
            if max(m.finger_bar_max) < c.release_max_n and opened:
                self.release_t0 = self.release_t0 or m.t
                if m.t - self.release_t0 >= c.release_hold_s:
                    self.s = 0.0
                    self._go('RETREAT', m.t)
            else:
                self.release_t0 = None
        elif self.phase == 'RETREAT':
            self.s = min(c.s_pre, self.s + c.approach_rate * dt)
            T = self.target(self.H_frozen, self.s)
            act['target'] = T
            if self.s >= c.s_pre and float(np.linalg.norm(m.T_tcp[:3, 3] - T[:3, 3])) < c.pre_tol_m:
                self._go('DONE', m.t)
        if self.phase == 'ABORTED':                       # 本輪推進中觸發（漂移、失接觸）
            act['target'] = None
            act['stop_solver'] = True
            act['grip'] = self.grip_cmd
            return act
        if self.phase == 'DONE':
            act['stop_solver'] = True
            act['finish'] = True
            self.done = True
        act['grip'] = self.grip_cmd
        if act['target'] is not None:
            self.last_target = act['target']
        return act
