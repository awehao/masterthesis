#!/usr/bin/env python3
"""K3 門角參考產生器（純幾何，可離線測試；door_target.py 不改）。

q_ref 每步取三個區間的**交集**（Codex 20261021 K3 審查）：
  限速區間  R = [q_ref_prev − r(q_ref_prev)·dt, q_ref_prev + r(q_ref_prev)·dt]
  超前區間  L = [q_meas − lead, q_meas + lead]
  登錄範圍  G = [q_min, θ_goal]               （q_ref ≤ θ_goal 恆成立）
交集為空 ⇒ 拋 DoorTargetK3Conflict（**不最後裁切、不重建**）。非空 ⇒ 取交集上界（朝 θ_goal 前進的最大允許值）。
減速律（依 **q_ref**，事前登錄）：r(q_ref) = r_min + (r_max − r_min)·clip((θ_goal − q_ref)/d_slow, 0, 1)。
到達＝q_ref 由同一規則走到 θ_goal（不跳設）；之後保持（交集上界即 θ_goal）。
目標把手位姿＝實測把手位姿繞鉸鏈轉 (q_ref − q_meas)；夾爪目標＝目標把手位姿 × G_T_H⁻¹（抓取關係量一次固定）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from door_target import rotate_about_hinge


class DoorTargetK3Conflict(RuntimeError):
    pass


@dataclass
class K3TargetConfig:
    hinge_point_world: tuple
    hinge_axis_world: tuple          # 右手定則 q 增加＝開門
    theta_goal_rad: float
    q_min_rad: float = 0.0
    r_max_rps: float = math.radians(1.0)
    r_min_rps: float = math.radians(0.25)
    d_slow_rad: float = math.radians(2.0)
    lead_max_rad: float = math.radians(0.5)

    def validate(self):
        for nm in ('theta_goal_rad', 'q_min_rad', 'r_max_rps', 'r_min_rps', 'd_slow_rad', 'lead_max_rad'):
            if not math.isfinite(float(getattr(self, nm))):
                raise ValueError(f'{nm} 非有限')
        if not (0.0 < self.r_min_rps <= self.r_max_rps and self.d_slow_rad > 0 and self.lead_max_rad > 0 and self.q_min_rad < self.theta_goal_rad):
            raise ValueError('K3TargetConfig 參數不合法')
        a = np.asarray(self.hinge_axis_world, float)
        if a.shape != (3,) or not np.isfinite(a).all() or float(np.linalg.norm(a)) < 1e-9:
            raise ValueError('hinge_axis_world 不合法')

    def rate(self, q_ref):
        f = min(1.0, max(0.0, (self.theta_goal_rad - q_ref) / self.d_slow_rad))
        return self.r_min_rps + (self.r_max_rps - self.r_min_rps) * f


class DoorTargetK3:
    def __init__(self, W_T_G0, W_T_H0, cfg: K3TargetConfig):
        cfg.validate()
        self.cfg = cfg
        self.G_T_H = np.linalg.inv(np.asarray(W_T_G0, float)) @ np.asarray(W_T_H0, float)
        self.q_ref = None
        self.reached = False
        self.last = {}

    def step_ref(self, q_meas, dt):
        c = self.cfg
        if not (math.isfinite(q_meas) and math.isfinite(dt) and dt >= 0.0):
            raise ValueError('q_meas／dt 非有限或 dt < 0')
        if self.q_ref is None:
            self.q_ref = float(q_meas)                     # 第一輪由實測建立
        r = c.rate(self.q_ref) * dt
        lo = max(self.q_ref - r, q_meas - c.lead_max_rad, c.q_min_rad)
        hi = min(self.q_ref + r, q_meas + c.lead_max_rad, c.theta_goal_rad)
        if lo > hi + 1e-12:
            raise DoorTargetK3Conflict(f'交集為空：限速 [{self.q_ref - r:.6f}, {self.q_ref + r:.6f}]、超前 [{q_meas - c.lead_max_rad:.6f}, '
                                       f'{q_meas + c.lead_max_rad:.6f}]、範圍 [{c.q_min_rad:.6f}, {c.theta_goal_rad:.6f}]')
        q_new = hi
        self.last = {'q_meas': q_meas, 'q_ref_prev': self.q_ref, 'q_ref': q_new, 'rate': c.rate(self.q_ref), 'lo': lo, 'hi': hi}
        self.q_ref = q_new
        self.reached = abs(self.q_ref - c.theta_goal_rad) <= 1e-12
        return self.q_ref

    def step(self, W_T_H_meas, q_meas, dt):
        q_ref = self.step_ref(q_meas, dt)
        T_h = rotate_about_hinge(np.asarray(W_T_H_meas, float), np.asarray(self.cfg.hinge_point_world, float),
                                 np.asarray(self.cfg.hinge_axis_world, float) / np.linalg.norm(self.cfg.hinge_axis_world), q_ref - q_meas)
        return T_h @ np.linalg.inv(self.G_T_H), dict(self.last)
