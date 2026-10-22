#!/usr/bin/env python3
"""有限區間等效偏移觀測（純邏輯候選；**未接入求解節點**）。規格：evaluation/results/specs/K3_offset_interval_observer_spec.md。

手臂模型（每物理步、逐軸）：q⁺ = q + α(s − q) + α·d，s⁺ = s + dt_p·u。d 在區間內近似恆定時，m 步後 d 對 q 的貢獻為
d·[1 − (1 − α)^m]（線性），因此以前一估計 d̂ 預測區間終點 q_pred 後，等效偏移觀測為

    d_obs = d̂ + (q_meas − q_pred) / (1 − (1 − α)^m)

這是「區間內近似恆定的**等效**偏移」，不是真實負載。更新：d̂ ← clip(d̂ + (m·dt_p/τ)(d_obs − d̂), ±max)。
- 只在區間結束、資料完整（q_k、s_k、各步套用速度、q_{k+1} 皆有限且 m ≥ 1）時更新；否則保持並記錄原因。
- 預測一律用**更新前**的 d̂（前一區間得到的估計）；回報的 pred_residual 是樣本外殘差，不得以同區間反解後的近零殘差當改善證據。
- 沿用現有 τ 與限幅；不把殘差當固定前饋。
- （Codex 20261022 必修）設定與初值必須有限，否則拒絕建構；運算結果非有限 ⇒ 保持、不提交。
- 可接受區間長度 m ∈ [1, m_max]（事前登錄 m_max＝10 物理步＝名目 5 步的兩倍；須滿足 m_max·dt_p/τ ≤ 1，
  否則 Euler 增益會超過 1）；超出 ⇒ 保持並記錄 interval_too_long，不改濾波公式或時間常數。
"""
from __future__ import annotations

import math

import numpy as np


class IntervalOffsetObserver:
    def __init__(self, alpha, phys_dt, tau_s=0.5, max_rad=0.05, d0=None, m_max=10):
        self.alpha = np.asarray(alpha, float).reshape(6)
        self.dtp, self.tau, self.max = float(phys_dt), float(tau_s), float(max_rad)
        if not (np.isfinite(self.alpha).all() and all(math.isfinite(x) for x in (self.dtp, self.tau, self.max))):
            raise ValueError('alpha／phys_dt／tau／max 須為有限值')
        if not (np.all(self.alpha > 0) and np.all(self.alpha < 1) and self.dtp > 0 and self.tau > 0 and self.max > 0):
            raise ValueError('alpha 須在 (0,1)；phys_dt、tau、max 須為正')
        if not (isinstance(m_max, int) and m_max >= 1 and m_max * self.dtp / self.tau <= 1.0):
            raise ValueError('m_max 須為正整數且 m_max·dt_p/τ ≤ 1')
        self.m_max = m_max
        if d0 is None:
            self.d = np.zeros(6)
        else:
            d0 = np.asarray(d0, float).reshape(6)
            if not np.isfinite(d0).all():
                raise ValueError('d0 須為有限值')
            self.d = np.clip(d0, -self.max, self.max)
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
            with np.errstate(over='ignore', invalid='ignore'):          # 溢位交給下方有限性檢查（不提交）
                q_pred, _ = self.predict(q0, s0, applied)
            q1 = self._vec(q1)
        except (TypeError, ValueError) as e:
            self.n_held += 1
            rec.update(why=f'incomplete:{e}', d_hat_after=self.d.tolist())
            return rec
        m = len(applied)
        rec['m'] = m
        if m > self.m_max:
            self.n_held += 1
            rec.update(why=f'interval_too_long（m={m} > {self.m_max}）', d_hat_after=self.d.tolist())
            return rec
        gain = 1.0 - (1.0 - self.alpha) ** m
        with np.errstate(over='ignore', invalid='ignore'):
            res = q1 - q_pred
            d_obs = self.d + res / gain
            d_new = np.clip(self.d + (m * self.dtp / self.tau) * (d_obs - self.d), -self.max, self.max)
        if not (np.isfinite(res).all() and np.isfinite(d_obs).all() and np.isfinite(d_new).all()):
            self.n_held += 1
            rec.update(why='nonfinite_result', d_hat_after=self.d.tolist())
            return rec
        self.d = d_new
        self.n_upd += 1
        rec.update(used='updated', m=m, pred_residual=res.tolist(), d_obs=d_obs.tolist(), d_hat_after=self.d.tolist())
        return rec
