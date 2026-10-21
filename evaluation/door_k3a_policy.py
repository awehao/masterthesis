#!/usr/bin/env python3
"""K3-A 相位機與監看（純邏輯，可離線測試；door_k2_policy.py 不改，接近／夾持段直接沿用 K2Policy）。

規格：evaluation/results/specs/K3_door_small_open_spec.md（draft-1～3）；範圍依 Codex 20261021 draft-3 審查縮為 **K3-A**：
  夾持 → 開到 θ_goal（8°）→ SETTLE → **保留抓持、受控停止並封存**；**不張爪、不退離**（解除清空另列 K3-B）。
相位：WAIT_HANDOVER → UNFOLD → ALIGN_PRE → ALIGN_HOLD → APPROACH → ENGAGE（以上同 K2）
      → HOLD（夾持保持 1.0 s）→ OPEN（DoorTargetK3：R∩L∩G、首步不推進、首筆＝上一筆已發布目標；
        目標＝上一筆已發布目標繞模型鉸鏈轉 (q_ref − q_entry)，只由門角決定）
      → SETTLE（q_ref 保持 θ_goal；穩定條件持續 1.0 s）→ END_HOLD（目標固定為最後一筆已發布目標，觀察 3.0 s）→ DONE。
抓取漂移一律以**夾持成立時量測的 G_T_H_grasp**為基準（不被 OPEN 初始化覆蓋）：位置 ≤ 5 mm、轉角 ≤ 2°。
門角：OPEN 以前 |q| ≤ 2°（同 K2）；OPEN 起 −1° ≤ q ≤ θ_goal＋1°；|q̇| ≤ 0.05 rad/s 全程。
手指—橫桿設計接觸允許相位：ENGAGE、HOLD、OPEN、SETTLE、END_HOLD、DONE（夾持保留）。
中止處置同 K2：凍結相位與夾爪命令，不放爪、不退離，停止發布新目標並要求停止求解器；觀察 3 s 後結束。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import door_k2_policy as K2
import door_target_k3 as T3
from door_target import rotate_about_hinge

Meas = K2.Meas
_finite = K2._finite
K2_PHASES = ('WAIT_HANDOVER', 'UNFOLD', 'ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH', 'ENGAGE')
ACTIVE_MONITOR = ('UNFOLD', 'ALIGN_PRE', 'ALIGN_HOLD', 'APPROACH', 'ENGAGE', 'HOLD', 'OPEN', 'SETTLE', 'END_HOLD')
FINGER_BAR_ALLOWED = ('ENGAGE', 'HOLD', 'OPEN', 'SETTLE', 'END_HOLD', 'DONE')
GRASP_PHASES = ('HOLD', 'OPEN', 'SETTLE', 'END_HOLD')
OPENING = ('OPEN', 'SETTLE', 'END_HOLD')


@dataclass
class K3AConfig(K2.K2Config):
    hold_s: float = 1.0                       # 夾持保持（K2 為 3.0）
    theta_goal_deg: float = 8.0
    r_max_dps: float = 1.0
    r_min_dps: float = 0.25
    d_slow_deg: float = 2.0
    lead_deg: float = 0.5
    q_min_deg: float = -1.0
    over_goal_deg: float = 1.0                # OPEN 起上界 θ_goal＋1°
    settle_tol_deg: float = 0.5
    settle_rate: float = 0.002                # rad/s（Codex 同意的事前候選）
    settle_hold_s: float = 1.0
    end_hold_s: float = 3.0
    drift_rot_abort_rad: float = math.radians(2.0)
    hinge_point: tuple = (0.0, 0.0, 0.0)
    hinge_axis: tuple = (0.0, 0.0, -1.0)      # door_local_v2：q > 0 ＝ 繞 −z
    timeouts: dict = field(default_factory=lambda: {
        'WAIT_HANDOVER': 30.0, 'UNFOLD': 30.0, 'ALIGN_PRE': 20.0, 'ALIGN_HOLD': 20.0, 'APPROACH': 15.0,
        'ENGAGE': 10.0, 'OPEN': 30.0, 'SETTLE': 10.0})

    def target_cfg(self):
        return T3.K3TargetConfig(hinge_point_world=self.hinge_point, hinge_axis_world=self.hinge_axis,
                                 theta_goal_rad=math.radians(self.theta_goal_deg), q_min_rad=math.radians(self.q_min_deg),
                                 r_max_rps=math.radians(self.r_max_dps), r_min_rps=math.radians(self.r_min_dps),
                                 d_slow_rad=math.radians(self.d_slow_deg), lead_max_rad=math.radians(self.lead_deg))


def grasp_drift(T_tcp, G_T_H, H):
    """實測 TCP 經夾持成立時的抓取關係推得的把手位姿 vs 實測把手：（位置 m、轉角 rad）。"""
    E = np.linalg.inv(np.asarray(H, float)) @ (np.asarray(T_tcp, float) @ np.asarray(G_T_H, float))
    c = (float(np.trace(E[:3, :3])) - 1.0) / 2.0
    return float(np.linalg.norm(E[:3, 3])), math.acos(max(-1.0, min(1.0, c)))


class K3APolicy(K2.K2Policy):
    def __init__(self, cfg: K3AConfig, T_HG, a_H=(0.0, -1.0, 0.0)):
        super().__init__(cfg, T_HG, a_H)
        self.c = cfg
        self.tcfg = cfg.target_cfg()
        self.tcfg.validate()
        self.dtk = None
        self.q_ref = None
        self.ref_info = None
        self.settle_ok_t0 = None
        self.open_entry = None            # {'t', 'T_last', 'H_entry', 'G_T_H_open'}
        self.max_drift = (0.0, 0.0)

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
        if self.phase in OPENING:
            lo, hi = math.radians(c.q_min_deg), math.radians(c.theta_goal_deg + c.over_goal_deg)
            if not (lo <= m.q_door <= hi):
                return self._abort('door_angle_out_of_bounds', m.t, q_deg=math.degrees(m.q_door))
        elif abs(m.q_door) > math.radians(c.door_abort_deg):
            return self._abort('door_moving', m.t, q_deg=math.degrees(m.q_door), qdot=m.qdot_door)
        if abs(m.qdot_door) > c.door_abort_rate:
            return self._abort('door_rate', m.t, q_deg=math.degrees(m.q_door), qdot=m.qdot_door)
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

    def _grasp_checks(self, m: Meas):
        c = self.c
        dp, dr = grasp_drift(m.T_tcp, self.G_T_H, m.H)
        self.max_drift = (max(self.max_drift[0], dp), max(self.max_drift[1], dr))
        if dp > c.drift_abort_m or dr > c.drift_rot_abort_rad:
            self._abort('grasp_drift', m.t, drift=dp, rot=dr)
            return dp
        for k in range(2):
            if m.finger_bar_min[k] < c.contact_min_n:
                self.loss_t0[k] = self.loss_t0[k] if self.loss_t0[k] is not None else m.t
                if m.t - self.loss_t0[k] >= c.finger_loss_s:
                    self._abort('finger_contact_lost', m.t, finger=k + 1)
                    return dp
            else:
                self.loss_t0[k] = None
        return dp

    def _ref_target(self, m: Meas, dt: float):
        # 目標只由門角決定：把手位姿取 OPEN 進入時的實測把手繞模型鉸鏈轉 (q_meas − q_entry)；
        # 實測把手 prim 位姿只用於漂移監看（不讓 prim 位姿雜訊進入目標）
        e = self.open_entry
        H_use = rotate_about_hinge(np.asarray(e['H_entry'], float), np.asarray(self.c.hinge_point, float),
                                   np.asarray(self.c.hinge_axis, float), m.q_door - e['q_meas'])
        try:
            T, info = self.dtk.step(H_use, m.q_door, dt)
        except T3.DoorTargetK3Conflict as e:
            self._abort('door_target_conflict', m.t, detail=str(e))
            return None
        self.q_ref, self.ref_info = self.dtk.q_ref, info
        return T

    # ------------------------------------------------------------ 推進
    def step(self, m: Meas, dt: float):
        if self.phase in K2_PHASES or self.phase == 'ABORTED':
            return super().step(m, dt)                    # 接近／夾持段與中止處置：K2 原邏輯（含 G_T_H 量測）
        c = self.c
        act = {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False,
               'stop_solver': False, 'finish': False}
        self.monitor(m)
        if self.phase == 'ABORTED':
            return self._aborted_act(act, m)
        ph0 = self.phase
        if ph0 in GRASP_PHASES:
            dp = self._grasp_checks(m)
            if self.phase == 'ABORTED':
                return self._aborted_act(act, m)
        if ph0 == 'HOLD':
            act['target'] = self.target(self.H_frozen, 0.0)
            if m.t - self.t_phase >= c.hold_s:
                self._go('OPEN', m.t)
        elif ph0 == 'OPEN':
            if self.dtk is None:                           # 進 OPEN 的第一輪才建立：首筆＝上一筆已發布目標
                self.dtk = T3.DoorTargetK3(self.last_target, m.H, self.tcfg, G_T_H_grasp=self.G_T_H)
                self.open_entry = {'t': m.t, 'T_last': np.asarray(self.last_target).tolist(), 'H_entry': np.asarray(m.H).tolist(),
                                   'G_T_H_open': self.dtk.G_T_H.tolist(), 'q_meas': m.q_door}
            T = self._ref_target(m, dt)
            if T is None:
                return self._aborted_act(act, m)
            act['target'] = T
            if self.dtk.reached:
                self._go('SETTLE', m.t, q_ref=self.q_ref)
        elif ph0 == 'SETTLE':
            T = self._ref_target(m, dt)
            if T is None:
                return self._aborted_act(act, m)
            act['target'] = T
            ok = (abs(m.q_door - self.tcfg.theta_goal_rad) <= math.radians(c.settle_tol_deg) and abs(m.qdot_door) <= c.settle_rate
                  and dp <= c.drift_abort_m and min(m.finger_bar_min) >= c.contact_min_n)
            if ok:
                self.settle_ok_t0 = self.settle_ok_t0 if self.settle_ok_t0 is not None else m.t
                if m.t - self.settle_ok_t0 >= c.settle_hold_s:
                    self._go('END_HOLD', m.t, q=m.q_door, qdot=m.qdot_door)
            else:
                self.settle_ok_t0 = None
        elif ph0 == 'END_HOLD':
            act['target'] = self.last_target               # 受控停止：固定最後一筆已發布目標
            if m.t - self.t_phase >= c.end_hold_s:
                self._go('DONE', m.t)
        if self.phase == 'ABORTED':
            return self._aborted_act(act, m)
        if self.phase == 'DONE':
            act['target'] = None
            act['stop_solver'] = True
            act['finish'] = True
            self.done = True
        act['grip'] = self.grip_cmd                        # 夾爪保持閉合命令（K3-A 不張爪）
        if act['target'] is not None:
            self.last_target = act['target']
        return act

    def _aborted_act(self, act, m):
        act['target'] = None
        act['stop_solver'] = True
        act['grip'] = self.grip_cmd
        if m.t - self.t_abort >= self.c.post_abort_observe_s:
            act['finish'] = True
        return act
