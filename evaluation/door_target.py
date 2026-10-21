#!/usr/bin/env python3
"""鉸鏈門的把手位姿與**由實測門角驅動**的夾爪目標（純幾何，可離線測試；不依賴模擬器）。

`hinged_handle_pose` 是 DL1 `opened_handle_pose`（滑軌）的旋轉版：
    T_WH(q) = [R_a(q)·R_WH0 | c + R_a(q)(p_WH0 − c)]
其中 c 為鉸鏈軸上一點、a 為鉸鏈軸單位向量（**右手定則下 q 增加＝開門方向**）、q 為門角（rad）。
把手方向隨門扇一起轉，不能沿用抽屜「夾爪方向不變」的軌跡。

`DoorTarget` 與 `DrawerTarget` 同結構：抓取關係 G_T_H 在連接當下量一次固定；
參考門角 q_ref 閉在**實測門角**上（有界超前 lead_max_rad、限速 rate_max_rps）；
目標把手位姿＝**實測**把手位姿繞鉸鏈軸再轉 (q_ref − q_meas)；夾爪目標＝目標把手位姿 × G_T_H⁻¹。
門角是物體狀態，不是門的致動命令。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from object_target_geometry import Reject, rigid, scalar, unit


def rot_axis(a, q):
    a = np.asarray(a, float)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)


def rotate_about_hinge(T, c, a, dq):
    R = rot_axis(a, dq)
    out = np.eye(4)
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = c + R @ (T[:3, 3] - c)
    return out


def hinged_handle_pose(T_WH0, hinge_point_W, hinge_axis_W, q, q_min, q_max):
    """T_WH0＝門角 0 的把手位姿；q 須在 [q_min, q_max]（rad）。回傳 {'ok', 'T_WH' | 'why'}。"""
    try:
        T0 = rigid(T_WH0, 'T_WH0')
        c = np.asarray(hinge_point_W, float)
        if c.shape != (3,) or not np.isfinite(c).all():
            raise Reject('hinge_point_invalid')
        a = unit(hinge_axis_W, 'hinge_axis_W')
        qmin = scalar(q_min, 'q_min', -math.pi, math.pi)
        qmax = scalar(q_max, 'q_max', -math.pi, math.pi)
        if qmax < qmin:
            raise Reject('q_range_inverted')
        q = scalar(q, 'q', qmin, qmax)
        return {'ok': True, 'T_WH': rotate_about_hinge(T0, c, a, q)}
    except Reject as r:
        return {'ok': False, 'why': r.why}


@dataclass
class DoorTargetConfig:
    hinge_point_world: tuple
    hinge_axis_world: tuple            # 右手定則下 q 增加＝開門
    q_min_rad: float
    q_max_rad: float
    lead_max_rad: float                # 有界超前：q_ref 與 q_meas 的差不得超過
    rate_max_rps: float                # q_ref 每秒最多推進

    def validate(self):
        c = np.asarray(self.hinge_point_world, float)
        if c.shape != (3,) or not np.isfinite(c).all():
            raise ValueError('hinge_point_world 必須是三個有限值')
        a = np.asarray(self.hinge_axis_world, float)
        if a.shape != (3,) or not np.isfinite(a).all() or float(np.linalg.norm(a)) < 1e-9:
            raise ValueError('hinge_axis_world 必須是非零有限向量')
        if not (math.isfinite(self.q_min_rad) and math.isfinite(self.q_max_rad) and self.q_min_rad < self.q_max_rad):
            raise ValueError('q_min_rad 必須小於 q_max_rad')
        for nm in ('lead_max_rad', 'rate_max_rps'):
            v = float(getattr(self, nm))
            if not (math.isfinite(v) and v > 0.0):
                raise ValueError(f'{nm} 必須為正有限值')

    def axis(self):
        a = np.asarray(self.hinge_axis_world, float)
        return a / float(np.linalg.norm(a))


def _rigid_or_value_error(T, name):
    try:
        return rigid(T, name)
    except Reject as r:
        raise ValueError(f'{name}：{r.why}') from None


class DoorTarget:
    """連接當下記下抓取關係，之後由實測門角產生夾爪目標。"""

    def __init__(self, W_T_G0, W_T_H0, cfg: DoorTargetConfig):
        cfg.validate()
        self.cfg = cfg
        W_T_G0 = _rigid_or_value_error(W_T_G0, '連接當下的夾爪位姿')
        W_T_H0 = _rigid_or_value_error(W_T_H0, '連接當下的把手位姿')
        self.G_T_H = np.linalg.inv(W_T_G0) @ W_T_H0
        self.q_ref = None
        self.n_step = 0
        self.last = {}

    def step(self, W_T_H_meas, q_meas, q_goal, dt):
        cfg = self.cfg
        W_T_H_meas = _rigid_or_value_error(W_T_H_meas, '實測把手位姿')
        for nm, v in (('q_meas', q_meas), ('q_goal', q_goal), ('dt', dt)):
            if v is None or not math.isfinite(float(v)):
                raise ValueError(f'{nm} 必須是有限值，收到 {v!r}')
        if float(dt) <= 0.0:
            raise ValueError(f'dt 必須為正，收到 {dt}')
        q_meas = float(q_meas)
        q_goal = min(max(float(q_goal), cfg.q_min_rad), cfg.q_max_rad)
        if self.q_ref is None:
            self.q_ref = q_meas                       # 第一輪由實測建立
        step_max = cfg.rate_max_rps * float(dt)
        delta = q_goal - self.q_ref
        q_ref = self.q_ref + max(-step_max, min(step_max, delta))
        lo, hi = q_meas - cfg.lead_max_rad, q_meas + cfg.lead_max_rad
        q_clamped = min(max(q_ref, lo), hi)
        lead_clamped = q_clamped != q_ref
        q_final = min(max(q_clamped, cfg.q_min_rad), cfg.q_max_rad)
        range_clamped = q_final != q_clamped
        self.q_ref = q_final
        T_h = rotate_about_hinge(W_T_H_meas, np.asarray(cfg.hinge_point_world, float), cfg.axis(), self.q_ref - q_meas)
        T_g = T_h @ np.linalg.inv(self.G_T_H)
        self.n_step += 1
        self.last = {'q_meas': q_meas, 'q_ref': self.q_ref, 'q_goal': q_goal, 'lead_rad': self.q_ref - q_meas,
                     'lead_clamped': bool(lead_clamped), 'range_clamped': bool(range_clamped), 'rate_limited': abs(delta) > step_max,
                     'basis': '參考門角閉在實測門角上；目標把手位姿由實測位姿繞鉸鏈軸旋轉'}
        return T_g, dict(self.last)

    def drift_vs(self, W_T_G_meas, W_T_H_meas):
        """夾爪實測位姿套抓取關係所預期的把手位置，對實測把手位置的距離（m）——滑脫判定用。"""
        W_T_G_meas = _rigid_or_value_error(W_T_G_meas, '實測夾爪位姿')
        W_T_H_meas = _rigid_or_value_error(W_T_H_meas, '實測把手位姿')
        return float(np.linalg.norm((W_T_G_meas @ self.G_T_H)[:3, 3] - W_T_H_meas[:3, 3]))
