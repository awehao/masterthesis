#!/usr/bin/env python3
"""K3-A-B 相位機：OPEN 期間依兩指承載差作有界夾合軸目標微調（規格 K3_finger_balance_compliance_spec.md draft-2＋Codex 20261023）。
door_k3a_policy.py 不改；繼承 K3APolicy（不繼承 K3-A-U／X，HOLD 不作用）。

時序（以本輪開始時的相位 ph0 定義）：
  - HOLD 與 OPEN 第一輪：Δ ≡ 0；OPEN 第一輪記 s0＝本輪狀態步號並丟棄 ≤ s0 的配對（不積分）⇒ 首筆 OPEN 發布目標＝最後 HOLD 發布目標。
  - OPEN 第二輪起至 SETTLE：先執行既有監看與夾持檢查（父類），再檢查配對量測（牆鐘年齡 ≤ stale_s、模擬時間年齡 ≤ 0.10 s、
    緩衝無 fault），取步號 ≤ 本輪狀態步號的配對逐步更新 Δ；任何停止 ⇒ 中止 balance_meas_invalid（不補值、不恢復）。
  - 發布目標＝基礎目標（DoorTargetK3，只由門角決定，不含 Δ）⊕ 平移 Δ·R_base·e₁₂ᵀ。
  - END_HOLD：父類固定發布最後一筆**實際已發布**的完整目標（含 Δ）；Δ 不再更新。DONE：照父類停止發布並停止求解器。
r2（Codex 20261023_194140 第 2、3 項）：
  - **SETTLE → END_HOLD 轉換輪**（本輪開始為 SETTLE、父類在本輪內切到 END_HOLD）即不消費配對，發布上一筆實際已發布的完整目標。
  - 消費的配對 sim_t 不得晚於本輪狀態 sim_t（＋dt_tol）⇒ 否則停止（future_time）。
抓取漂移與安全監看全部沿用父類（對原始 G_T_H_grasp，不重新初始化）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import door_k3a_policy as P
import door_k3ab_balance as B

Meas = P.Meas
BALANCE_PHASES = ('OPEN', 'SETTLE')


@dataclass
class K3ABConfig(P.K3AConfig):
    balance: B.BalanceConfig = field(default_factory=B.BalanceConfig)


class K3ABPolicy(P.K3APolicy):
    def __init__(self, cfg: K3ABConfig, T_HG, a_H=(0.0, -1.0, 0.0)):
        super().__init__(cfg, T_HG, a_H)
        self.pairs = B.PairBuffer()
        self.bal = B.BalanceLaw(cfg.balance)
        self.last_base = None
        self.row = None                     # 本輪紀錄：ph0、base、Δ、e12、本輪消費步號範圍

    @staticmethod
    def e12_world(T):
        return np.asarray(T, float)[:3, :3] @ np.asarray(B.E12_TCP, float)

    def _balance_fail(self, m, act, why):
        self.bal.stop(why)
        self._abort('balance_meas_invalid', m.t, detail=self.bal.stopped)
        self.row['consumed'] = None
        return self._aborted_act(act, m)

    def step(self, m: Meas, dt: float):
        ph0 = self.phase
        first_open = ph0 == 'OPEN' and self.dtk is None
        pub_prev, base_prev = self.last_target, self.last_base
        act = super().step(m, dt)
        self.row = {'ph0': ph0, 'base': None, 'delta': self.bal.delta, 'e12': None, 'consumed': None}
        if act['target'] is None or self.phase == 'ABORTED':
            return act
        base = np.asarray(act['target'], float)
        if ph0 == 'END_HOLD':                                 # 父類已給最後實際發布目標；基礎目標同樣凍結
            self.row.update(base=self.last_base, e12=None if self.last_base is None else self.e12_world(self.last_base))
            return act
        if ph0 == 'SETTLE' and self.phase == 'END_HOLD':               # 轉換輪：即凍結上一筆實際已發布目標
            act['target'] = np.array(pub_prev, float)
            self.last_target = act['target']
            self.row.update(base=base_prev, e12=None if base_prev is None else self.e12_world(base_prev))
            return act
        if ph0 not in BALANCE_PHASES:
            self.last_base = base
            self.row.update(base=base, e12=self.e12_world(base))
            return act
        c = self.c.balance
        state_step = getattr(m, 'state_step', None)
        if state_step is None:
            return self._balance_fail(m, act, 'state_step_missing')
        if first_open:
            self.bal.start(state_step, m.t)
            self.pairs.discard_upto(state_step)
            self.events.append({'t': m.t, 'balance_start': {'s0': int(state_step), 't0': m.t}})
        else:
            wall_age = getattr(m, 'pairs_age_wall', None)
            if wall_age is None or not math.isfinite(wall_age) or wall_age > self.c.stale_s:
                return self._balance_fail(m, act, 'stale_wall')
            if self.pairs.fault is not None:
                return self._balance_fail(m, act, f'buffer_{self.pairs.fault}')
            n0 = len(self.bal.records)
            entries = self.pairs.take(int(state_step))
            if any(e[1] > m.t + c.dt_tol_s for e in entries):
                return self._balance_fail(m, act, 'future_time')
            self.bal.update(entries)
            if self.bal.stopped is not None:
                self._abort('balance_meas_invalid', m.t, detail=self.bal.stopped)
                return self._aborted_act(act, m)
            t_last = self.bal.last_t if self.bal.last_t is not None else self.bal.t0
            if m.t - t_last > c.stale_sim_s + 1e-9:
                return self._balance_fail(m, act, 'stale_sim')
            new = self.bal.records[n0:]
            self.row['consumed'] = None if not new else [new[0]['step'], new[-1]['step']]
        e12 = self.e12_world(base)
        pub = base.copy()
        pub[:3, 3] = base[:3, 3] + self.bal.delta * e12
        act['target'] = pub
        self.last_target = pub
        self.last_base = base
        self.row.update(base=base, delta=self.bal.delta, e12=e12)
        return act
