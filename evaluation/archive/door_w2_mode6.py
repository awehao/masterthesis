#!/usr/bin/env python3
"""W2 HELD 監督器 r6：mode 2（RELEASE_OPEN）實際提交點切換為釋放保持監看（純邏輯；Codex 20261026_021025 §2）。
door_w2_mode*.py 不改；本檔以子類別擴充 door_w2_mode5.HeldSupervisorP4b5。

背景：door_wide_k3b_30_1 中張爪後門扇轉動，求解端 HELD 監督器仍以「門必須靜止」（0.002 rad/s）於提交後 0.05 s 判 moved_in_held 並停發。

r6 規則：
- 外層狀態仍為 HELD，求解端既有的請求受理（只在 HELD）與 mode 3 交接（只自 HELD）不變；內部以 sub＝'held'／'release_open' 區分。
- 進入 release_open：release_open(t_commit, t_now)，由求解端在得知 mode 2 已提交時呼叫（權威狀態 mode_seq ≥ 2，提交時刻取權威 commits 列）。
  - 提交前門須靜止：以速度窗緩衝中「結束於 t_commit 以前」的窗口重算門速，> 舊 HELD 門檻 ⇒ FAIL moved_before_release；無足夠樣本 ⇒ 不核（記錄）。
  - 若已在暫緩中（見下），須 t_commit ≤ 首次超限時刻，否則 FAIL moved_before_release。
- release_open 中，每週期：
  - 時間有效、求解成功、HELD 時限照舊；
  - 量測新鮮（不新鮮 ⇒ FAIL measurement_stale／invalid）；
  - 底盤線速、角速與手臂速度 ≤ 舊 HELD 門檻（不變；mode 2 手臂仍應靜止、底盤主動保持、持續發布）；
  - 門速改為保護上限 door_rate_release（＝任務端已登錄 door_abort_rate 0.05 rad/s）：> 上限 ⇒ FAIL door_rate_in_release；
  - 門角帶、幾何、資料新鮮度由任務端／執行端照常負責（本監督器不取代）。
- 權威狀態延遲／ack 遺失（Codex：不能等得知切換後才開始承接）：
  - HELD（sub held）中只有門超過舊門檻、底盤與手臂仍低、門 ≤ door_rate_release，且求解端已受理 mode 2 請求（pending2）⇒ 暫緩，不立即 FAIL；
    暫緩期間照常發布保持命令（與 mode 2 候選同形）。
  - 暫緩起點起 defer_max_s（0.2 s，同任務端狀態年齡上限）內得知 mode 2 提交且符合上列條件 ⇒ 轉 release_open；逾時 ⇒ FAIL moved_in_held。
  - 未受理 mode 2 請求 ⇒ 照舊立即 FAIL（不暫緩）。
- 套用證據（applied）：release_open 中沿用 HELD 的保持形核對（mode 2 候選＝保持形）。
- mode 3 交接沿用 r5（handover 只在外層 HELD；release_open 亦屬 HELD）。
"""
from __future__ import annotations

import math

import door_k3m_pause_sup as PSUP
import door_w2_mode5 as M5

VERSION = 'door_w2_mode6/r6'
DOOR_RATE_RELEASE = 0.05
DEFER_MAX_S = 0.2


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class HeldSupervisorP4b6(M5.HeldSupervisorP4b5):
    stop_definition = 'active_hold_arm_win_0.5s_held_release_open_handover'

    def __init__(self, *a, door_rate_release=DOOR_RATE_RELEASE, defer_max_s=DEFER_MAX_S, **kw):
        super().__init__(*a, **kw)
        self.door_rate_release, self.defer_max_s = float(door_rate_release), float(defer_max_s)
        self.sub = 'held'
        self.pending2 = False
        self.t_defer0 = None
        self.t_release_commit = None
        self.t_release_known = None
        self.defer_log = []
        self.release_checks = None

    # ---- 求解端回報
    def set_pending2(self, flag, t=None):
        flag = bool(flag)
        if flag != self.pending2:
            self.defer_log.append({'t': t, 'pending2': flag})
        self.pending2 = flag

    def _door_rate_before(self, t_c):
        """結束於 t_c 以前（含）的最近連續窗口門速；樣本不足 ⇒ None。"""
        buf = list(self.ws.buf)
        idx = [i for i, s in enumerate(buf) if s is not None and s[1] <= t_c + 1e-9]
        if not idx:
            return None
        j = idx[-1]
        b = buf[j]
        prev = b
        for s in reversed(buf[:j]):
            if s is None or prev[1] - s[1] > self.ws.max_gap_s or not (s[0] < prev[0]):
                return None
            prev = s
            if b[1] - s[1] >= self.ws.win_s - 1e-9:
                return abs(b[5] - s[5]) / (b[1] - s[1])
        return None

    def release_open(self, t_commit, t_now):
        """得知 mode 2 已提交 ⇒ 切換為釋放保持監看。回傳是否切換。只在外層 HELD、sub held 有效。"""
        s = self.seq
        if s.state != 'HELD' or self.sub != 'held':
            return False
        if not _fin(t_commit) or not _fin(t_now) or t_commit > t_now + 1e-9:
            self._fail('release_commit_time_invalid', t_now if _fin(t_now) else float('nan'))
            return False
        pre = self._door_rate_before(t_commit)
        self.release_checks = {'t_commit': float(t_commit), 't_known': float(t_now), 'door_rate_pre': pre, 't_defer0': self.t_defer0}
        if pre is not None and pre > self.th.door:
            self._fail('moved_before_release', t_now)
            return False
        if self.t_defer0 is not None and t_commit > self.t_defer0 + 1e-9:
            self._fail('moved_before_release', t_now)
            return False
        self.sub = 'release_open'
        self.t_release_commit, self.t_release_known = float(t_commit), float(t_now)
        s.log.append((float(t_now), 'RELEASE_OPEN', None))
        self.t_defer0 = None
        return True

    # ---- 每週期
    def cycle(self, t, solve_ok, hold_cmd_ok_now, wall=None):
        s = self.seq
        if s.state != 'HELD':
            return super().cycle(t, solve_ok, hold_cmd_ok_now, wall)
        if not PSUP._fin(t):
            self._fail('time_invalid', t)
            return {'publish': False, 'zero': False}
        if not solve_ok:
            self._fail('solve_failed_in_held', t)
            return {'publish': False, 'zero': False}
        if t - self.t_held > self.held_max_s + 1e-9:
            return super().cycle(t, solve_ok, hold_cmd_ok_now, wall)              # 時限結束照 r4
        v = self.ws.speeds(t, wall)
        self.last_speeds = v
        if v is None:
            self._fail('measurement_invalid' if self.ws.fresh(t, wall) else 'measurement_stale', t)
            return {'publish': False, 'zero': False}
        others_low = v[0] <= self.th.base_lin and v[1] <= self.th.base_ang and v[3] <= self.th.arm
        if self.sub == 'release_open':
            if not others_low:
                self._fail('moved_in_release_open', t)
                return {'publish': False, 'zero': False}
            if v[2] > self.door_rate_release:
                self._fail('door_rate_in_release', t)
                return {'publish': False, 'zero': False}
            return {'publish': True, 'zero': True}
        if PSUP.WindowSpeed.low(v, self.th):
            if self.t_defer0 is not None:
                self.defer_log.append({'t': t, 'defer_cleared': True, 't_defer0': self.t_defer0})
                self.t_defer0 = None
            return {'publish': True, 'zero': True}
        if others_low and v[2] <= self.door_rate_release and self.pending2:
            if self.t_defer0 is None:
                self.t_defer0 = float(t)
                self.defer_log.append({'t': t, 'defer_start': True, 'door_rate': v[2]})
            if t - self.t_defer0 > self.defer_max_s + 1e-9:
                self._fail('moved_in_held', t)
                self.defer_log.append({'t': t, 'defer_expired': True, 't_defer0': self.t_defer0})
                return {'publish': False, 'zero': False}
            return {'publish': True, 'zero': True}
        self._fail('moved_in_held', t)
        return {'publish': False, 'zero': False}

    def applied(self, phys_step, t, base3, arm6, src_seq):
        return super().applied(phys_step, t, base3, arm6, src_seq)               # 外層 HELD ⇒ 沿用保持形核對（release_open 亦然）

    def outcome(self, t_end=None):
        o = super().outcome(t_end)
        o['transitions']['RELEASE_OPEN'] = self.t_release_known
        o.update(version=VERSION, sub=self.sub, release_commit_t=self.t_release_commit, release_known_t=self.t_release_known,
                 release_checks=self.release_checks, door_rate_release=self.door_rate_release, defer_max_s=self.defer_max_s,
                 defer_log=self.defer_log[:50])
        return o


def commit_t_of(state_msg, mode_seq):
    """權威狀態訊息中指定 mode_seq 的提交時刻；無 ⇒ None。"""
    for c in (state_msg or {}).get('commits') or []:
        if isinstance(c, dict) and c.get('mode_seq') == mode_seq and _fin(c.get('t')):
            return float(c['t'])
    return None
