#!/usr/bin/env python3
"""長窗隔離診斷的觀察期實驗保護（純函式；規格 K3_stop_longwindow_diag_spec_draft3.md；Codex 20261023_230237／231357）。

定位：**實驗保護**，觸發 ⇒ 提前結束本次觀察（記為觀察提前終止），**不是**控制器成功停住，也不恢復任何控制。
與原策略獨立：原策略中止後已離開 ACTIVE_MONITOR，本模組不依賴它。只在收到中止（arm）之後觸發；arm 之前的異常只記錄。

觸發條件（任一，首次觸發即鎖定，之後不再改寫）：
  - door_range：實測門角 ≥ 5°（本次隔離診斷的範圍上限；不是安全依據）
  - door_rate：|q̇_door| > 0.05 rad/s（原 K2Config.door_abort_rate）
  - unexpected_contact：非設計接觸窗口最大值 > 0.1 N（原 contact_abort_n；**不要求持續時間**）
  - wall_contact：牆接觸窗口最大值 > 0.1 N
  - wrist：腕力窗口最大值 ≥ 30 N（原 wrist_abort_n）
  - contacts_invalid：欄位缺失、非有限、步號／n_invalid 不是合法非負整數、n_invalid > 0、
    同步號內容衝突、來源步號或時間倒退
  - state_invalid：長度錯、非有限、同步號內容衝突、步號或時間倒退
  - state_stale／contacts_stale：最後一筆**合法新**訊息的牆鐘年齡 > 0.2 s（原 stale_s）；arm 後從未收到 ⇒ 以 arm 時刻起算
新鮮度只由**合法且來源前進**的新訊息刷新；同一步號、內容相同的重送只計數（n_dup），不刷新年齡。
接觸窗口是執行端「窗口內逐物理步最大值」，每收到一個就立即判定，短脈衝不會被漏掉。

結束（r2 依 Codex 231357 第 3 項）：
  - closing(why)：收到他方結束請求（/room/stop）。之後**只停用停更告警**（執行端照預期停發），硬界與無效資料仍照判；
  - finish(why)：有界收尾完成（觀察器收到運行器在執行端行程結束後送的 TERM）⇒ closed。closed 之後不再判定。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

INT_MAX = 2 ** 53


def _num(x):
    """合法有限實數；否則 None（不拋例外）。"""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    try:
        v = float(x)
    except (OverflowError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _nonneg_int(x):
    v = _num(x)
    if v is None or v < 0 or v > INT_MAX or v != int(v):
        return None
    return int(v)


@dataclass
class LWGuardConfig:
    door_max_deg: float = 5.0
    door_rate_max: float = 0.05
    contact_max_n: float = 0.1
    wrist_max_n: float = 30.0
    stale_s: float = 0.2


CT_KEYS = ('unexpected_max_n', 'wall_max_n', 'wrist_max_n', 'sim_t')


@dataclass
class LWGuard:
    cfg: LWGuardConfig = field(default_factory=LWGuardConfig)
    armed_wall: float | None = None
    armed_info: dict | None = None
    trip: dict | None = None
    last_state: tuple | None = None          # (step, sim_t, wall, data)
    last_ct: tuple | None = None             # (step, sim_t, wall, 內容)
    n_state: int = 0
    n_ct: int = 0
    n_dup: int = 0
    n_ct_dup: int = 0
    anomalies: list = field(default_factory=list)
    closing: dict | None = None
    closed: dict | None = None

    # ---- 生命週期 ----
    def arm(self, wall_now, info=None):
        if self.armed_wall is None:
            self.armed_wall = float(wall_now)
            self.armed_info = info

    def close(self, wall_now, why):
        """收到結束請求：只停用停更告警（硬界照判）。"""
        if self.closing is None:
            self.closing = {'wall': float(wall_now), 'why': why}

    def finish(self, wall_now, why):
        """有界收尾完成：之後不再判定。"""
        if self.closed is None:
            self.closed = {'wall': float(wall_now), 'why': why, 'after_closing': self.closing is not None}

    def _active(self):
        return self.armed_wall is not None and self.closed is None

    def _bad(self, kind, why, wall_now, **kw):
        self.anomalies.append({'kind': kind, 'why': why, 'wall': float(wall_now), **kw})
        if len(self.anomalies) > 200:
            self.anomalies = self.anomalies[:100] + self.anomalies[-100:]
        return self._trip(kind, wall_now, detail=why, **kw)

    def _trip(self, why, wall_now, **kw):
        if self.trip is None and self._active():
            self.trip = {'why': why, 'wall_trip': float(wall_now), 'during_closing': self.closing is not None, **kw}
        return self.trip

    # ---- 輸入 ----
    def on_state(self, d, wall_now):
        """d：/door/sync_state 的 29 欄。"""
        try:
            d = list(d)
        except TypeError:
            return self._bad('state_invalid', 'not_sequence', wall_now)
        vals = [_num(x) for x in d]
        if len(d) != 29 or any(v is None for v in vals):
            return self._bad('state_invalid', 'length_or_nonfinite', wall_now)
        step, sim_t, q, qdot = vals[1], vals[0], vals[2], vals[28]
        if _nonneg_int(step) is None:
            return self._bad('state_invalid', 'step_not_int', wall_now)
        if self.last_state is not None:
            if step == self.last_state[0]:
                if vals == self.last_state[3]:
                    self.n_dup += 1
                    return self.trip
                return self._bad('state_invalid', 'same_step_conflict', wall_now, src_step=step)
            if step < self.last_state[0] or sim_t < self.last_state[1]:
                return self._bad('state_invalid', 'backward', wall_now, src_step=step, src_sim_t=sim_t)
        self.last_state = (step, sim_t, float(wall_now), vals)
        self.n_state += 1
        if not self._active():
            return self.trip
        src = {'src_step': int(step), 'src_sim_t': sim_t, 'wall_recv': float(wall_now)}
        if abs(q) >= math.radians(self.cfg.door_max_deg):
            return self._trip('door_range', wall_now, q_deg=math.degrees(q), **src)
        if abs(qdot) > self.cfg.door_rate_max:
            return self._trip('door_rate', wall_now, qdot=qdot, **src)
        return self.trip

    def on_contacts(self, d, wall_now):
        """d：/door/contacts 的 JSON（一個窗口）。"""
        if not isinstance(d, dict):
            return self._bad('contacts_invalid', 'not_dict', wall_now)
        vals = {k: _num(d.get(k)) for k in CT_KEYS}
        step, inv = _nonneg_int(d.get('step')), _nonneg_int(d.get('n_invalid'))
        if any(v is None for v in vals.values()) or step is None or inv is None:
            return self._bad('contacts_invalid', 'fields_nonfinite_or_not_int', wall_now)
        key = (step, vals['sim_t'], inv, tuple(vals[k] for k in CT_KEYS), repr(d.get('unexpected_who')))
        if self.last_ct is not None:
            if step == self.last_ct[0]:
                if key == self.last_ct[3]:
                    self.n_ct_dup += 1                                      # 重送：不刷新年齡
                    return self.trip
                return self._bad('contacts_invalid', 'same_step_conflict', wall_now, src_step=step)
            if step < self.last_ct[0] or vals['sim_t'] < self.last_ct[1]:
                return self._bad('contacts_invalid', 'backward', wall_now, src_step=step, src_sim_t=vals['sim_t'])
        self.last_ct = (step, vals['sim_t'], float(wall_now), key)
        self.n_ct += 1
        if not self._active():
            return self.trip
        src = {'src_step': step, 'src_sim_t': vals['sim_t'], 'wall_recv': float(wall_now)}
        if inv > 0:
            return self._trip('contacts_invalid', wall_now, detail='n_invalid', n_invalid=inv, **src)
        if vals['unexpected_max_n'] > self.cfg.contact_max_n:
            return self._trip('unexpected_contact', wall_now, value_n=vals['unexpected_max_n'], who=d.get('unexpected_who'), **src)
        if vals['wall_max_n'] > self.cfg.contact_max_n:
            return self._trip('wall_contact', wall_now, value_n=vals['wall_max_n'], **src)
        if vals['wrist_max_n'] >= self.cfg.wrist_max_n:
            return self._trip('wrist', wall_now, value_n=vals['wrist_max_n'], **src)
        return self.trip

    def check(self, wall_now):
        """週期呼叫：新鮮度（closing 之後停用）。"""
        if not self._active() or self.closing is not None:
            return self.trip
        for kind, last in (('state_stale', self.last_state), ('contacts_stale', self.last_ct)):
            ref = last[2] if last is not None else None
            ref = self.armed_wall if ref is None or ref < self.armed_wall else ref
            if wall_now - ref > self.cfg.stale_s:
                return self._trip(kind, wall_now, age_s=wall_now - ref)
        return self.trip

    def report(self):
        return {'armed_wall': self.armed_wall, 'armed_info': self.armed_info, 'trip': self.trip, 'closing': self.closing, 'closed': self.closed,
                'n_state': self.n_state, 'n_contacts': self.n_ct, 'n_state_dup': self.n_dup, 'n_contacts_dup': self.n_ct_dup,
                'anomalies': self.anomalies[:50], 'n_anomalies': len(self.anomalies)}


@dataclass
class HeartbeatWatch:
    """任務節點端：觀察器心跳監看。就緒＝收過一筆合法心跳；之後年齡 > max_age_s ⇒ unhealthy。
    心跳 seq 必須嚴格遞增；重複／倒退不刷新。"""
    max_age_s: float = 1.0
    last_seq: int | None = None
    last_wall: float | None = None
    ready_wall: float | None = None
    bad: list = field(default_factory=list)

    def on_beat(self, d, wall_now):
        seq = _nonneg_int(d.get('seq')) if isinstance(d, dict) else None
        if seq is None:
            self.bad.append(('invalid', float(wall_now)))
            return
        if self.last_seq is not None and seq <= self.last_seq:
            self.bad.append(('not_increasing', float(wall_now), seq))
            return
        self.last_seq, self.last_wall = seq, float(wall_now)
        if self.ready_wall is None:
            self.ready_wall = float(wall_now)

    def ready(self):
        return self.ready_wall is not None

    def healthy(self, wall_now):
        return self.ready() and wall_now - self.last_wall <= self.max_age_s
