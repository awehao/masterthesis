#!/usr/bin/env python3
"""長窗隔離診斷的觀察期實驗保護（純函式；規格 K3_stop_longwindow_diag_spec.md draft-2；依 Codex 20261023_230237 第 2 節）。

定位：**實驗保護**，觸發 ⇒ 提前結束本次觀察（記為觀察提前終止），**不是**控制器成功停住，也不恢復任何控制。
與原策略獨立：原策略中止後已離開 ACTIVE_MONITOR，本模組不依賴它。只在收到中止（arm）之後生效。

觸發條件（任一，首次觸發即鎖定，之後不再改寫）：
  - door_range：實測門角 ≥ 5°（本次隔離診斷的範圍上限；不是安全依據）
  - door_rate：|q̇_door| > 0.05 rad/s（原 K2Config.door_abort_rate）
  - unexpected_contact：非設計接觸窗口最大值 > 0.1 N（原 contact_abort_n；**不要求持續時間**）
  - wall_contact：牆接觸窗口最大值 > 0.1 N
  - wrist：腕力窗口最大值 ≥ 30 N（原 wrist_abort_n）
  - contacts_invalid：接觸窗口 n_invalid > 0、欄位缺失或非有限
  - state_invalid：狀態非有限、長度錯、步號或時間倒退
  - state_stale／contacts_stale：最後一筆有效訊息的牆鐘年齡 > 0.2 s（原 stale_s）；arm 之後從未收到 ⇒ 以 arm 時刻起算
接觸窗口是執行端「窗口內逐物理步最大值」，每收到一個窗口就立即判定（不先合併再取用），短脈衝不會被漏掉。
狀態重複步號只略過（不更新年齡），不觸發。
正常結束（任務節點或執行端自行發 /room/stop）時呼叫 close()：之後停更不算保護觸發。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class LWGuardConfig:
    door_max_deg: float = 5.0
    door_rate_max: float = 0.05
    contact_max_n: float = 0.1
    wrist_max_n: float = 30.0
    stale_s: float = 0.2


@dataclass
class LWGuard:
    cfg: LWGuardConfig = field(default_factory=LWGuardConfig)
    armed_wall: float | None = None
    armed_info: dict | None = None
    trip: dict | None = None
    last_state: tuple | None = None          # (step, sim_t, wall)
    last_ct_wall: float | None = None
    n_state: int = 0
    n_ct: int = 0
    n_dup: int = 0
    closed: dict | None = None               # 正常結束（他方 /room/stop）後關閉：之後不再觸發

    def close(self, wall_now, why):
        if self.closed is None and self.trip is None:
            self.closed = {'wall': float(wall_now), 'why': why}

    def arm(self, wall_now, info=None):
        if self.armed_wall is None:
            self.armed_wall = float(wall_now)
            self.armed_info = info

    def _trip(self, why, wall_now, **kw):
        if self.trip is None and self.armed_wall is not None and self.closed is None:
            self.trip = {'why': why, 'wall_trip': float(wall_now), **kw}
        return self.trip

    def on_state(self, d, wall_now):
        """d：/door/sync_state 的 29 欄。"""
        if len(d) != 29 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in d):
            return self._trip('state_invalid', wall_now, detail='length_or_nonfinite')
        step, sim_t, q, qdot = d[1], d[0], d[2], d[28]
        if self.last_state is not None:
            if step == self.last_state[0]:
                self.n_dup += 1
                return self.trip
            if step < self.last_state[0] or sim_t < self.last_state[1]:
                return self._trip('state_invalid', wall_now, detail='backward', src_step=step, src_sim_t=sim_t)
        self.last_state = (step, sim_t, float(wall_now))
        self.n_state += 1
        if self.armed_wall is None:
            return None
        if abs(q) >= math.radians(self.cfg.door_max_deg):
            return self._trip('door_range', wall_now, src_step=step, src_sim_t=sim_t, wall_recv=wall_now, q_deg=math.degrees(q))
        if abs(qdot) > self.cfg.door_rate_max:
            return self._trip('door_rate', wall_now, src_step=step, src_sim_t=sim_t, wall_recv=wall_now, qdot=qdot)
        return None

    def on_contacts(self, d, wall_now):
        """d：/door/contacts 的 JSON（一個窗口）。"""
        try:
            vals = {k: float(d[k]) for k in ('unexpected_max_n', 'wall_max_n', 'wrist_max_n', 'sim_t')}
            inv, step = int(d['n_invalid']), d.get('step')
        except (KeyError, TypeError, ValueError):
            return self._trip('contacts_invalid', wall_now, detail='fields')
        self.last_ct_wall = float(wall_now)
        self.n_ct += 1
        if self.armed_wall is None:
            return None
        src = {'src_step': step, 'src_sim_t': vals['sim_t'], 'wall_recv': wall_now}
        if inv > 0 or not all(math.isfinite(v) for v in vals.values()):
            return self._trip('contacts_invalid', wall_now, n_invalid=inv, **src)
        if vals['unexpected_max_n'] > self.cfg.contact_max_n:
            return self._trip('unexpected_contact', wall_now, value_n=vals['unexpected_max_n'], who=d.get('unexpected_who'), **src)
        if vals['wall_max_n'] > self.cfg.contact_max_n:
            return self._trip('wall_contact', wall_now, value_n=vals['wall_max_n'], **src)
        if vals['wrist_max_n'] >= self.cfg.wrist_max_n:
            return self._trip('wrist', wall_now, value_n=vals['wrist_max_n'], **src)
        return None

    def check(self, wall_now):
        """週期呼叫：新鮮度。"""
        if self.armed_wall is None or self.closed is not None:
            return None
        ref_s = self.last_state[2] if self.last_state is not None else None
        ref_s = self.armed_wall if ref_s is None or ref_s < self.armed_wall else ref_s
        if wall_now - ref_s > self.cfg.stale_s:
            return self._trip('state_stale', wall_now, age_s=wall_now - ref_s)
        ref_c = self.last_ct_wall
        ref_c = self.armed_wall if ref_c is None or ref_c < self.armed_wall else ref_c
        if wall_now - ref_c > self.cfg.stale_s:
            return self._trip('contacts_stale', wall_now, age_s=wall_now - ref_c)
        return None
