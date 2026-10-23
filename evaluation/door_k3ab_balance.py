#!/usr/bin/env python3
"""K3-A-B 兩指承載平衡：逐步配對緩衝與有界調整律（純邏輯，可離線測試；規格 K3_finger_balance_compliance_spec.md draft-2＋Codex 20261023）。

**量測**：執行端 /door/finger_pair 每 50 Hz 一筆 JSON {'pairs': [[step, sim_t, f1, f2, valid], ...]}，列出窗口內每個物理步的
  finger1／finger2 橫桿法向力幅值（與 room leaf_contacts 同一步、同一來源、同為 4 位小數）。
**PairBuffer**（入口驗證，原始值不轉型前先檢查）：
  - 訊息須為 dict 且 'pairs' 為 list；每筆須為 5 元素，step 為有限整數值、sim_t 有限、valid 為 bool、f1／f2 為有限數或 None。
    不合格 ⇒ fault＝'malformed'（之後柔順更新即停止）。
  - 已消費或已待處理的步號重送：值完全相同 ⇒ 忽略（重送不重積分）；不同 ⇒ fault＝'conflict'。
  - take(upto)：只取步號 ≤ upto（本輪狀態步號）的待處理筆，依步號遞增回傳並移除；較晚的步留在緩衝（不提前使用、不丟棄）。
  - discard_upto(s0)：OPEN 第一輪把 ≤ s0 的筆標為已消費但不積分；之後到達的 ≤ s0 筆只計數、不積分。
**BalanceLaw**（每個來源步只消費一次，依步號順序；登錄值固定）：
  - start(s0, t0)：首個可更新步＝s0＋1；之前 Δ ≡ 0。
  - 每步：步號須＝上一步＋1、sim_t 差＝dt_p（容差 1e-4）、valid 且 f1、f2 有限 ≥ 0；否則停止（stop），之後不再更新、不恢復、不跨缺口補積分。
  - e＝f2 − f1；濾波 ẽ：第一個積分步 ẽ＝e（登錄初值），之後 ẽ ← ẽ＋β(e − ẽ)，β＝1 − exp(−dt_p/τ_f)。
  - 死區 u＝sign(ẽ)·max(|ẽ| − e_db, 0)；速率 v＝clip(k_i·u, ±v_max)；Δ ← clip(Δ＋v·dt_p, ±Δ_max)（條件積分＝截斷，反向立即可動）。
  - Δ > 0 ⇒ 目標沿 e₁₂（finger1 → finger2，TCP 座標 (0, −1, 0)）平移。
  - 逐步紀錄 {step, sim_t, f1, f2, e, ef, u, v, delta, sat} 供判定器獨立重算。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

E12_TCP = (0.0, -1.0, 0.0)


@dataclass
class BalanceConfig:
    tau_f_s: float = 0.05
    dt_p_s: float = 0.01
    deadband_n: float = 0.5
    k_i_m_per_ns: float = 0.25e-3
    v_max_mps: float = 1.0e-3
    delta_max_m: float = 1.0e-3
    dt_tol_s: float = 1e-4
    stale_sim_s: float = 0.10

    @property
    def beta(self):
        return 1.0 - math.exp(-self.dt_p_s / self.tau_f_s)


def _is_int_value(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return False
    return math.isfinite(float(x)) and float(x) == int(x)


def _num_or_none(x):
    return x is None or (not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(float(x)))


def parse_entry(e):
    """原始一筆 → (step, sim_t, f1, f2, valid)；不合格 ⇒ None。"""
    if not isinstance(e, (list, tuple)) or len(e) != 5:
        return None
    s, t, f1, f2, v = e
    if not _is_int_value(s) or isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(float(t)):
        return None
    if not isinstance(v, bool) or not _num_or_none(f1) or not _num_or_none(f2):
        return None
    return (int(s), float(t), None if f1 is None else float(f1), None if f2 is None else float(f2), v)


class PairBuffer:
    def __init__(self):
        self.pending = {}
        self.seen = {}            # 已消費（含丟棄）的值，用於重送比對
        self.hwm = None           # 已消費的最高步號
        self.fault = None
        self.last_wall = None
        self.n_dup = 0
        self.n_late_discarded = 0

    def push(self, msg, wall):
        if not isinstance(msg, dict) or not isinstance(msg.get('pairs'), list):
            self.fault = self.fault or 'malformed'
            return
        got = False
        for raw in msg['pairs']:
            e = parse_entry(raw)
            if e is None:
                self.fault = self.fault or 'malformed'
                continue
            got = True
            s = e[0]
            old = self.seen.get(s, self.pending.get(s))
            if old is not None:
                if old == e:
                    self.n_dup += 1
                else:
                    self.fault = self.fault or 'conflict'
                continue
            if self.hwm is not None and s <= self.hwm:
                self.n_late_discarded += 1                        # 已越過（丟棄區）的遲到筆：不積分
                self.seen[s] = e
                continue
            self.pending[s] = e
        if got:
            self.last_wall = wall

    def discard_upto(self, s0):
        for s in sorted(k for k in self.pending if k <= s0):
            self.seen[s] = self.pending.pop(s)
        self.hwm = s0 if self.hwm is None else max(self.hwm, s0)

    def take(self, upto):
        out = []
        for s in sorted(k for k in self.pending if k <= upto):
            e = self.pending.pop(s)
            self.seen[s] = e
            out.append(e)
        if out:
            self.hwm = max(self.hwm if self.hwm is not None else out[-1][0], out[-1][0])
        return out

    def age(self, wall):
        return math.inf if self.last_wall is None else wall - self.last_wall


class BalanceLaw:
    def __init__(self, cfg: BalanceConfig | None = None):
        self.c = cfg or BalanceConfig()
        self.s0 = None
        self.t0 = None
        self.last_step = None
        self.last_t = None
        self.ef = None
        self.delta = 0.0
        self.stopped = None
        self.records = []

    def start(self, s0, t0):
        self.s0, self.t0 = int(s0), float(t0)
        self.last_step, self.last_t = int(s0), None

    def stop(self, why, **kw):
        if self.stopped is None:
            self.stopped = {'why': why, **kw}

    def update(self, entries):
        c = self.c
        for (s, t, f1, f2, valid) in entries:
            if self.stopped is not None:
                return
            if self.last_step is None:
                self.stop('not_started', step=s)
                return
            if s != self.last_step + 1:
                self.stop('gap', expected=self.last_step + 1, got=s)
                return
            if self.last_t is not None and abs((t - self.last_t) - c.dt_p_s) > c.dt_tol_s:
                self.stop('time_step', step=s, dt=t - self.last_t)
                return
            if not valid or f1 is None or f2 is None or f1 < 0.0 or f2 < 0.0:
                self.stop('invalid', step=s)
                return
            e = f2 - f1
            self.ef = e if self.ef is None else self.ef + c.beta * (e - self.ef)
            u = math.copysign(max(abs(self.ef) - c.deadband_n, 0.0), self.ef)
            v = max(-c.v_max_mps, min(c.v_max_mps, c.k_i_m_per_ns * u))
            raw = self.delta + v * c.dt_p_s
            d = max(-c.delta_max_m, min(c.delta_max_m, raw))
            self.delta = d
            self.last_step, self.last_t = s, t
            self.records.append({'step': s, 'sim_t': t, 'f1': f1, 'f2': f2, 'e': e, 'ef': self.ef, 'u': u, 'v': v,
                                 'delta': d, 'sat': raw != d})
