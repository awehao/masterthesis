#!/usr/bin/env python3
"""有限區間等效偏移觀測（純邏輯候選；**未接入求解節點**）。規格：evaluation/results/specs/K3_offset_interval_observer_spec.md。

手臂模型（每物理步、逐軸）：q⁺ = q + α(s − q) + α·d，s⁺ = s + dt_p·u。d 在區間內近似恆定時，m 步後 d 對 q 的貢獻為
d·[1 − (1 − α)^m]（線性），因此以前一估計 d̂ 預測區間終點 q_pred 後，等效偏移觀測為

    d_obs = d̂ + (q_meas − q_pred) / (1 − (1 − α)^m)

這是「區間內近似恆定的**等效**偏移」，不是真實負載。更新：d̂ ← clip(d̂ + (m·dt_p/τ)(d_obs − d̂), ±max)。
- 只在區間結束、資料完整（q_k、s_k、各步套用速度、q_{k+1} 皆有限且 m ≥ 1）時更新；否則保持並記錄原因。
- 預測一律用**更新前**的 d̂（前一區間得到的估計）；回報的 pred_residual 是樣本外殘差，不得以同區間反解後的近零殘差當改善證據。
- 沿用現有 τ 與限幅；不把殘差當固定前饋。
"""
from __future__ import annotations

import math

import numpy as np


class IntervalOffsetObserver:
    def __init__(self, alpha, phys_dt, tau_s=0.5, max_rad=0.05, d0=None):
        self.alpha = np.asarray(alpha, float).reshape(6)
        self.dtp = float(phys_dt)
        if not (np.all(self.alpha > 0) and np.all(self.alpha < 1) and self.dtp > 0 and tau_s > 0 and max_rad > 0):
            raise ValueError('alpha 須在 (0,1)；phys_dt、tau、max 須為正')
        self.tau, self.max = float(tau_s), float(max_rad)
        self.d = np.zeros(6) if d0 is None else np.clip(np.asarray(d0, float).reshape(6), -max_rad, max_rad)
        self.n_upd = 0
        self.n_held = 0

    @staticmethod
    def _vec(x):
        v = np.asarray(x, float).reshape(6)
        if not np.isfinite(v).all():
            raise ValueError('nonfinite')
        return v

    def predict(self, q0, s0, applied, d=None):
        q = self._vec(q0).copy()
        s = self._vec(s0).copy()
        d = self.d if d is None else np.asarray(d, float)
        for u in applied:
            u = self._vec(u)
            q = q + self.alpha * (s - q) + self.alpha * d
            s = s + self.dtp * u
        return q, s

    def update(self, q0, s0, applied, q1):
        """一個完整區間。回傳紀錄：used（updated／held）、why、m、pred_residual（樣本外，更新前 d̂）、d_obs、d_hat_before／after。"""
        rec = {'used': 'held', 'why': None, 'm': None, 'pred_residual': None, 'd_obs': None, 'd_hat_before': self.d.tolist()}
        try:
            applied = list(applied) if applied is not None else None
            if not applied:
                raise ValueError('no_steps')
            q_pred, _ = self.predict(q0, s0, applied)
            q1 = self._vec(q1)
        except (TypeError, ValueError) as e:
            self.n_held += 1
            rec.update(why=f'incomplete:{e}', d_hat_after=self.d.tolist())
            return rec
        m = len(applied)
        gain = 1.0 - (1.0 - self.alpha) ** m
        res = q1 - q_pred
        d_obs = self.d + res / gain
        self.d = np.clip(self.d + (m * self.dtp / self.tau) * (d_obs - self.d), -self.max, self.max)
        self.n_upd += 1
        rec.update(used='updated', m=m, pred_residual=res.tolist(), d_obs=d_obs.tolist(), d_hat_after=self.d.tolist())
        return rec
