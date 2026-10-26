#!/usr/bin/env python3
"""W2 r9：求解端監督器 r9（純邏輯；Codex 20261026_034842 §2）。door_w2_mode8 不改；以子類別擴充。

HeldSupervisorP4b9(HeldSupervisorP4b8)：
  1. 瞬態監看改 door_k3b_transient2（無效樣本中斷收斂證據、重新成窗、重新計時；連續無效 > 0.2 s 失敗）。
  2. 量測失效也送進瞬態監看：feed_state 收到非有限樣本時，以無效樣本餵入（重送／倒退仍忽略，不算無效）。
  3. **延遲得知提交時補核**：on_commit 成功後，瞬態監看由提交前最後一筆樣本起算，並**依序重放**緩衝內已收到的提交後樣本
     （含失效斷點）；重放中超限 ⇒ 當下 FAIL <unload|release>_transient:<why>，不重新初始化而漏掉。
     暫緩期暫定監看的紀錄（worst）一併存入 checks[k]['provisional']。
  4. 暫緩期暫定監看亦改用 transient2。
  提交前運動、暫緩期限、首次超限不清除、中止優先等規則照 r7／r8。
ModeAuthority9＝ModeAuthority8（不變；另名以利登錄）。
"""
from __future__ import annotations

import math

import door_k3b_transient2 as TT2
import door_w2_mode7 as M7
import door_w2_mode8 as M8

VERSION = 'door_w2_mode9/r9'
NAN = float('nan')


def _all_fin(*xs):
    try:
        return all(math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


class HeldSupervisorP4b9(M8.HeldSupervisorP4b8):
    stop_definition = 'active_hold_arm_win_0.5s_held_unload_release_transient2_settled_handover'

    def __init__(self, *a, t_max_unload=TT2.T_MAX_UNLOAD, t_max_release=TT2.T_MAX_RELEASE, **kw):
        super().__init__(*a, t_max_unload=t_max_unload, t_max_release=t_max_release, **kw)
        self.tm = {1: TT2.TransientMonitor(self.tcp_fn, t_max_unload, name='unload'), 2: TT2.TransientMonitor(self.tcp_fn, t_max_release, name='release')}

    def feed_state(self, step, t, x, y, yaw, q_door, q_arm, wall=None):
        r = M7.HeldSupervisorP4b7.feed_state(self, step, t, x, y, yaw, q_door, q_arm, wall)
        k = 1 if self.sub == 'unload' else 2 if self.sub == 'release_open' else None
        if k is not None and self.tm[k].state == 'ACTIVE':
            try:
                qa = list(q_arm)
            except TypeError:
                qa = []
            if r:
                self.tm[k].feed(float(t), (x, y, yaw), qa)
            elif not (_all_fin(t, x, y, yaw, q_door, *qa) and len(qa) == 6):          # 失效（非重送）⇒ 中斷收斂證據
                self.tm[k].feed(float(t) if _all_fin(t) else NAN, (NAN, NAN, NAN), [NAN] * 6)
        return r

    def _prov_check(self, t, k, v):
        if self.prov is None:
            s0 = self._sample_before(float(t) - self.ws.win_s)
            if v[2] > self.door_rate_protect:
                return False
            self.prov = TT2.TransientMonitor(self.tcp_fn, self.defer_max_s + self.ws.win_s + 0.05, name=f'defer{k}')
            if s0 is None or not self.prov.start(s0[1], (s0[2], s0[3], s0[4]), s0[6]):
                return False
            self._prov_fed = s0[1]
        return super()._prov_check(t, k, v)

    def on_commit(self, k, t_c, t_now):
        prov = None if self.prov is None else self.prov.report()
        ok = M7.HeldSupervisorP4b7.on_commit(self, k, t_c, t_now)
        if not ok:
            return False
        self.prov, self._prov_fed = None, None
        if k not in self.tm:
            return True
        name = 'unload' if k == 1 else 'release'
        s0 = self._sample_before(t_c)
        if s0 is None or not self.tm[k].start(s0[1], (s0[2], s0[3], s0[4]), s0[6]):
            self._fail(f'{name}_transient:start_invalid', t_now)
            return False
        buf = list(self.ws.buf)
        try:
            i0 = next(i for i in range(len(buf) - 1, -1, -1) if buf[i] is s0)
        except StopIteration:
            i0 = len(buf) - 1
        later = [(NAN, (NAN, NAN, NAN), [NAN] * 6) if x is None else (x[1], (x[2], x[3], x[4]), x[6]) for x in buf[i0 + 1:]]
        self.tm[k].feed_many(later)
        self.checks[k].update(replay_n=len(later), provisional=prov, replay_worst=dict(self.tm[k].worst))
        if self.tm[k].state == 'FAILED':
            self._fail(f'{name}_transient:{self.tm[k].why}', t_now)
            return False
        return True

    def outcome(self, t_end=None):
        o = super().outcome(t_end)
        o['version'] = VERSION
        return o


ModeAuthority9 = M8.ModeAuthority8
exec_unload_of = M8.exec_unload_of
