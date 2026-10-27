#!/usr/bin/env python3
"""持握閉合修正引導 r2（door_coopen_closure_guidance.py 的版本化修正；Codex r36 §2；純邏輯、不依賴 ROS／Isaac）。
r1 缺口（Codex 重現）：界限加在內部狀態 c，真正加到參考上的是 s·c（兩者變化率不同）；先逐軸限速、再範數截斷會破壞逐軸限速
  （c＝(8/√2, 8/√2) mm、dt 0.05、目標 (20, 0) mm ⇒ y 每輪 −0.100868 mm ＞ 0.1 mm）。
r2：
  - 狀態＝**實際輸出** o（加到參考位姿平移上的量，世界 xy，公尺）。目標 o*＝s(q)·e（e＝p_target − p_tcp），s(q) 依實測門角 smoothstep。
  - 每輪：Δ＝(dt/τ)·(o* − o)；以**範數**限速 |Δ| ≤ rate·dt；o′＝Π(o＋Δ)，Π＝投影到 |o| ≤ c_max 的球。
    因 o 本在球內且投影非擴張 ⇒ |o′ − o| ≤ |Δ| ≤ rate·dt，|o′| ≤ c_max（兩個界限都落在實際輸出上）。s 的變化也經由 o* 進入同一限速。
    登錄的變化率 2 mm/s 由 r1 的「每軸」改為「向量範數」（數值不變、定義從嚴，揭露）。
  - 同步步號：step 須為整數且嚴格大於上一個已處理步，否則本輪不更新、不刷新新鮮度（why＝step_not_advanced）。
  - 無效輸入（非有限、維度錯、dt≤0）：不更新 o（沿用），why＝input_invalid；自首次無效起（模擬時間）連續 ≥ invalid_hold_s ⇒ timeout＝True
    （呼叫端須走既有中止路徑，理由 closure_input_invalid）。有效輸入一到即清除。
  - closure_cycle()：任務節點每輪呼叫的純函式，回傳 (參考表＝ShiftedTable 或原表, 紀錄, 中止理由或 None)。只在原引導作用的相位
    （OPEN、SETTLE）且底盤已切 free、抓取基準已建立時更新；其他相位不更新、輸出沿用（PAUSE 起原引導整體關閉，既有行為）。
參數 CFG（A/B 前固定，數值同 r1；新增 invalid_hold_s 0.2 s＝既有監看時效）。

    python3 evaluation/test_door_coopen_closure_guidance2.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

GUIDE_PHASES = ('OPEN', 'SETTLE')


@dataclass
class ClosureCfg2:
    q_on_deg: float = 36.0
    q_full_deg: float = 40.0
    tau_s: float = 1.0
    rate_norm_m_s: float = 0.002
    c_max_m: float = 0.008
    invalid_hold_s: float = 0.2


CFG = ClosureCfg2()


def smoothstep(q_deg, a, b):
    if q_deg <= a:
        return 0.0
    if q_deg >= b:
        return 1.0
    x = (q_deg - a) / (b - a)
    return x * x * (3.0 - 2.0 * x)


def _vec2(v):
    a = np.asarray(v, float).reshape(-1)
    if a.shape != (2,) or not np.all(np.isfinite(a)):
        raise ValueError('需要兩個有限值')
    return a


@dataclass
class ClosureCorrector2:
    cfg: ClosureCfg2 = field(default_factory=ClosureCfg2)
    o: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_step: int | None = None
    invalid_since: float | None = None
    n_update: int = 0
    n_invalid: int = 0
    n_step_repeat: int = 0
    n_rate_limited: int = 0
    n_mag_projected: int = 0

    def update(self, step, t, dt, q_door_deg, p_tcp_xy, p_target_xy):
        o_prev = self.o.copy()
        rec = {'step': step, 't': t, 'o_prev': o_prev.tolist(), 'o': o_prev.tolist(), 's': None, 'e': None, 'why': None, 'timeout': False}
        if not isinstance(step, int) or isinstance(step, bool) or (self.last_step is not None and step <= self.last_step):
            self.n_step_repeat += 1
            rec['why'] = 'step_not_advanced'
            return rec
        try:
            if not (math.isfinite(float(t)) and math.isfinite(float(dt)) and float(dt) > 0 and math.isfinite(float(q_door_deg))):
                raise ValueError('t／dt／門角無效')
            pt, pg = _vec2(p_tcp_xy), _vec2(p_target_xy)
        except (TypeError, ValueError):
            self.n_invalid += 1
            tt = float(t) if isinstance(t, (int, float)) and math.isfinite(float(t)) else None
            if self.invalid_since is None:
                self.invalid_since = tt
            rec['why'] = 'input_invalid'
            rec['timeout'] = (tt is not None and self.invalid_since is not None and tt - self.invalid_since >= self.cfg.invalid_hold_s - 1e-9)
            return rec
        self.invalid_since = None
        self.last_step = step
        s = smoothstep(float(q_door_deg), self.cfg.q_on_deg, self.cfg.q_full_deg)
        e = pg - pt
        target = s * e
        d = (float(dt) / self.cfg.tau_s) * (target - self.o)
        lim = self.cfg.rate_norm_m_s * float(dt)
        nd = float(np.linalg.norm(d))
        if nd > lim:
            d = d * (lim / nd)
            self.n_rate_limited += 1
        o = self.o + d
        no = float(np.linalg.norm(o))
        if no > self.cfg.c_max_m:
            o = o * (self.cfg.c_max_m / no)
            self.n_mag_projected += 1
        self.o = o
        self.n_update += 1
        rec.update(o=o.tolist(), s=s, e=e.tolist())
        return rec


class ShiftedTable:
    """參考表薄包裝：只把 pose() 的平移加上 offset；dpose_dq（前饋）、arm_nom 與其餘屬性原樣轉交（door_wide_guidance.guidance 不改）。"""

    def __init__(self, table, offset_xy=(0.0, 0.0)):
        self._t = table
        self.offset = np.asarray(offset_xy, float).reshape(2)

    def pose(self, q):
        pr = self._t.pose(q)
        return (float(pr[0]) + float(self.offset[0]), float(pr[1]) + float(self.offset[1]), float(pr[2]))

    def __getattr__(self, name):
        return getattr(self._t, name)


def closure_cycle(corr, table, phase, mode_state, G_T_H, step, t, dt, q_door_rad, T_tcp, H):
    """任務節點每輪呼叫。回傳 (table_used, rec, abort_why)。
    只有 phase ∈ GUIDE_PHASES、mode_state＝free、G_T_H 已建立時才呼叫 corr.update；其他情形不更新、輸出沿用（rec why＝not_active）。
    參考表一律包成 ShiftedTable(table, corr.o)（o＝0 時與原表 pose 相同）。rec 另記原參考與修正後參考（判定器重算用）。"""
    rec = None
    if phase in GUIDE_PHASES and mode_state == 'free' and G_T_H is not None:
        try:
            Tt = np.asarray(H, float).reshape(4, 4) @ np.linalg.inv(np.asarray(G_T_H, float).reshape(4, 4))
            p_tg, p_tcp = Tt[:2, 3], np.asarray(T_tcp, float).reshape(4, 4)[:2, 3]
        except (TypeError, ValueError, np.linalg.LinAlgError):
            p_tg = p_tcp = None
        rec = corr.update(step, t, dt, math.degrees(q_door_rad) if isinstance(q_door_rad, (int, float)) else float('nan'), p_tcp, p_tg)
    else:
        rec = {'step': step, 't': t, 'o_prev': corr.o.tolist(), 'o': corr.o.tolist(), 's': None, 'e': None, 'why': 'not_active', 'timeout': False}
    tbl = ShiftedTable(table, corr.o)
    try:
        rec['ref_orig'] = [float(v) for v in table.pose(q_door_rad)]
        rec['ref_mod'] = [float(v) for v in tbl.pose(q_door_rad)]
    except (TypeError, ValueError):
        rec['ref_orig'] = rec['ref_mod'] = None
    return tbl, rec, ('closure_input_invalid' if rec.get('timeout') else None)
