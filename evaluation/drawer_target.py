#!/usr/bin/env python3
"""由**實測開度**驅動的夾爪目標（純幾何，可離線測試）。

與既有 `coman_pull_target.py` 的差別
-----------------------------------
那一支的行程 `s(t)` 是**時間的 smoothstep** —— 開迴路。參考值會按時間往前走，
不管抽屜實際到哪裡；抽屜若被卡住，參考值仍然前進，夾爪就一路拉，
最後不是滑脫就是硬扯。凍結判準也明文禁止以時間到期代替實測。

本檔改成**閉在量測上**：
    參考開度 d_ref = 實測開度 d_meas ＋ 有界超前量 lead，並朝相位目標推進
    目標把手位姿 = **實測**把手位姿沿滑軌平移 (d_ref − d_meas)
    夾爪目標     = 目標把手位姿 × 抓取關係的逆

超前量有界是重點：參考值永遠不會比實際多走 `lead_max`。抽屜卡住時
d_meas 不動，d_ref 也就停在 d_meas + lead_max，不會愈拉愈遠。

抓取關係 `G_T_H` 在**連接當下**量一次就固定 —— 之後夾爪目標都由把手目標反推，
抓取關係整段不變，這正是理想固定連接所要求的。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _rigid_ok(T, name):
    T = np.asarray(T, float)
    if T.shape != (4, 4):
        raise ValueError(f'{name} 必須是 4×4，收到 {T.shape}')
    if not np.isfinite(T).all():
        raise ValueError(f'{name} 含非有限值')
    R = T[:3, :3]
    if abs(float(np.linalg.det(R)) - 1.0) > 1e-6:
        raise ValueError(f'{name} 的旋轉部分行列式不是 +1')
    if np.abs(R.T @ R - np.eye(3)).max() > 1e-6:
        raise ValueError(f'{name} 的旋轉部分不是正交矩陣')
    return T


@dataclass
class DrawerTargetConfig:
    """滑軌幾何與推進設定。"""
    axis_world: tuple = (0.0, -1.0, 0.0)   # 開啟方向（世界）
    travel_min_m: float = 0.0
    travel_max_m: float = 0.220
    # **有界超前**：參考開度最多比實測多走這麼多
    lead_max_m: float = 0.010
    # 參考開度每秒最多推進多少（與底盤速度框同量級，不是另一個自由參數）
    rate_max_mps: float = 0.035255

    def validate(self):
        a = np.asarray(self.axis_world, float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError('axis_world 必須是三個有限值')
        if float(np.linalg.norm(a)) < 1e-9:
            raise ValueError('axis_world 長度為零')
        if not (self.travel_min_m < self.travel_max_m):
            raise ValueError('travel_min_m 必須小於 travel_max_m')
        for nm in ('lead_max_m', 'rate_max_mps'):
            if float(getattr(self, nm)) <= 0.0:
                raise ValueError(f'{nm} 必須為正')

    def axis(self):
        a = np.asarray(self.axis_world, float)
        return a / float(np.linalg.norm(a))


class DrawerTarget:
    """連接當下記下抓取關係，之後由實測開度產生夾爪目標。"""

    def __init__(self, W_T_G0, W_T_H0, cfg: DrawerTargetConfig | None = None):
        cfg = cfg or DrawerTargetConfig()
        cfg.validate()
        self.cfg = cfg
        W_T_G0 = _rigid_ok(W_T_G0, '連接當下的夾爪位姿')
        W_T_H0 = _rigid_ok(W_T_H0, '連接當下的把手位姿')
        # **抓取關係只量一次**（夾爪 → 把手）
        self.G_T_H = np.linalg.inv(W_T_G0) @ W_T_H0
        self.d_ref = None          # 參考開度；第一次 step 時由實測建立
        self.n_step = 0
        self.last = {}

    # ------------------------------------------------------------ 推進
    def step(self, W_T_H_meas, d_meas: float, d_goal: float, dt: float):
        """回傳 (夾爪目標 4×4, 診斷 dict)。

        `W_T_H_meas` 與 `d_meas` 都是**這一輪讀回來的實測值**。
        """
        cfg = self.cfg
        W_T_H_meas = _rigid_ok(W_T_H_meas, '實測把手位姿')
        for nm, v in (('d_meas', d_meas), ('d_goal', d_goal), ('dt', dt)):
            if v is None or not math.isfinite(float(v)):
                raise ValueError(f'{nm} 必須是有限值，收到 {v!r}')
        if float(dt) <= 0.0:
            raise ValueError(f'dt 必須為正，收到 {dt}')
        d_meas = float(d_meas)
        d_goal = min(max(float(d_goal), cfg.travel_min_m), cfg.travel_max_m)

        if self.d_ref is None:
            # **第一輪由實測建立**，不從零或從目標起算
            self.d_ref = d_meas
        # 朝目標推進，受速率上限
        step_max = cfg.rate_max_mps * float(dt)
        delta = d_goal - self.d_ref
        d_ref = self.d_ref + max(-step_max, min(step_max, delta))
        # **有界超前**：不得比實測多走 lead_max（兩個方向都限）
        lo = d_meas - cfg.lead_max_m
        hi = d_meas + cfg.lead_max_m
        d_ref_clamped = min(max(d_ref, lo), hi)
        lead_clamped = (d_ref_clamped != d_ref)
        # 行程上下限
        d_ref_final = min(max(d_ref_clamped, cfg.travel_min_m),
                          cfg.travel_max_m)
        travel_clamped = (d_ref_final != d_ref_clamped)
        self.d_ref = d_ref_final

        # 目標把手位姿 = **實測**把手位姿沿滑軌平移 (d_ref − d_meas)
        T_h = W_T_H_meas.copy()
        T_h[:3, 3] = T_h[:3, 3] + cfg.axis() * (self.d_ref - d_meas)
        T_g = T_h @ np.linalg.inv(self.G_T_H)

        self.n_step += 1
        self.last = {
            'd_meas': d_meas, 'd_ref': self.d_ref, 'd_goal': d_goal,
            'lead_m': self.d_ref - d_meas,
            'lead_clamped': bool(lead_clamped),
            'travel_clamped': bool(travel_clamped),
            'rate_limited': abs(delta) > step_max,
            'basis': '參考開度閉在**實測開度**上；目標把手位姿由實測位姿平移',
        }
        return T_g, dict(self.last)

    # ------------------------------------------------------------ 核對
    def grasp_drift(self, W_T_G_meas) -> float:
        """夾爪實測位姿對抓取關係的**位置**漂移（m）。

        把連接當下的抓取關係套到現在的夾爪位姿，得到「把手應該在哪」，
        再與實測把手位姿比。回傳給相位機當滑脫判定用。
        """
        W_T_G_meas = _rigid_ok(W_T_G_meas, '實測夾爪位姿')
        return W_T_G_meas @ self.G_T_H

    def drift_vs(self, W_T_G_meas, W_T_H_meas) -> float:
        expect = self.grasp_drift(W_T_G_meas)
        W_T_H_meas = _rigid_ok(W_T_H_meas, '實測把手位姿')
        return float(np.linalg.norm(expect[:3, 3] - W_T_H_meas[:3, 3]))
