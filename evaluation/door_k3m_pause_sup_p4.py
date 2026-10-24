#!/usr/bin/env python3
"""K3-M 受控暫停 p4：**主動保持停止**監督器（Howard 2026-10-24 選定；規格 K3M_pause_p4_active_hold_stop.md）。
door_k3m_pause_sup.py／door_k3m_pause.py 不改；本檔繼承 PauseSupervisor，只換「命令停止」的定義與 STOPPED 之後的發布。

依據（door_k3m_pause_p3_1、p3b_1）：夾持中底盤命令一改成零，0.1 s 內底盤約 2.6 mm/s、門約 0.003 rad/s 漂移；
底盤主動保持＋手臂零命令時四項速度可持續低於門檻。所以本版的停止＝「手臂零命令＋底盤主動保持命令」下的實際靜止，
**不是**「命令歸零即停住」；報告須寫明為主動保持停止。

與 r2（PauseSupervisor）的差異：
  - ZERO_PUB（狀態名沿用，語意＝**保持發布**）：發布手臂 0、底盤＝主動保持命令（錨點伺服，向量限幅 ≤ HOLD_V、|ω| ≤ HOLD_W）。
  - CONFIRM：cycle 的第三參數改為 hold_cmd_ok（已發布的手臂為 0 且底盤在保持界內），取代 base_cmd_zero。
  - STOPPED：套用回報連續 n_confirm 個不同物理步滿足「手臂 ≤ applied_tol、底盤在保持界內、來源序號 ≥ 保持第一筆」。
  - STOPPED 之後**繼續發布保持命令**（publish=True、zero=True），觀察 observe_s（10 s）期間每週期都要實際低速；
    套用回報出現舊序號、手臂非零或底盤超出保持界 ⇒ FAIL 'old_command_after_stop'。
不變：四項門檻（底盤 1 mm/s、0.005 rad/s、門 0.002 rad/s、手臂 1e-3 rad/s）、窗口與新鮮度、LOWSPEED 的 hold_s、逾時 8 s、
  觀察 10 s、FAIL 理由名稱、outcome 欄位（另加 stop_definition＝'active_hold'）。
"""
from __future__ import annotations

import math

import door_k3m_pause_sup as PSUP

HOLD_V = 0.005               # m/s：停車伺服合線速度上限（park_fixed hold_v_max，同一口徑）
HOLD_W = 0.02                # rad/s：hold_w_max


def hold_cmd_ok(base3, arm6, tol=1e-6):
    """手臂全部 ≤ tol 且底盤在主動保持界內（線速度向量 ≤ HOLD_V、|ω| ≤ HOLD_W，各加 tol）。"""
    try:
        b = [float(v) for v in base3]
        a = [float(v) for v in arm6]
    except (TypeError, ValueError):
        return False
    if len(b) != 3 or len(a) != 6 or not all(math.isfinite(v) for v in b + a):
        return False
    return max(abs(v) for v in a) <= tol and math.hypot(b[0], b[1]) <= HOLD_V + tol and abs(b[2]) <= HOLD_W + tol


class ActiveHoldSupervisor(PSUP.PauseSupervisor):
    stop_definition = 'active_hold'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.n_hold_bad_after_stop = 0

    def cycle(self, t, solve_ok, hold_cmd_ok_now, wall=None):
        s = self.seq
        if s.t0 is None:
            s.t0 = t
        if s.state in ('DONE', 'FAIL'):
            return {'publish': False, 'zero': False}
        if not PSUP._fin(t):
            self._fail('time_invalid', t)
            return {'publish': False, 'zero': False}
        if s.state in ('LOWSPEED', 'ZERO_PUB', 'CONFIRM', 'STOPPED') and not solve_ok:
            self._fail('solve_failed', t)                         # p4：STOPPED 後仍在發布保持命令 ⇒ 求解失敗同樣判失敗
            return {'publish': False, 'zero': False}
        if s.state in ('LOWSPEED', 'ZERO_PUB', 'CONFIRM') and t - s.t0 > self.c.timeout_s + 1e-9:
            self._fail('timeout', t)
            return {'publish': False, 'zero': False}
        v = self.ws.speeds(t, wall)
        self.last_speeds = v
        if v is None:
            if s.state == 'LOWSPEED':
                s.since = None
                return {'publish': True, 'zero': False}
            self._fail('measurement_invalid' if self.ws.fresh(t, wall) else 'measurement_stale', t)
            return {'publish': False, 'zero': False}
        low = PSUP.WindowSpeed.low(v, self.th)
        if s.state == 'LOWSPEED':
            if low:
                self.t_first_low = t if self.t_first_low is None else self.t_first_low
                s.since = t if s.since is None else s.since
                if t - s.since >= self.c.hold_s - 1e-9:
                    s.state = 'ZERO_PUB'
                    s.log.append((t, 'ZERO_PUB', None))
            else:
                s.since = None
            return {'publish': True, 'zero': s.state == 'ZERO_PUB'}
        if not low:
            self._fail({'ZERO_PUB': 'moved_during_zero_pub', 'CONFIRM': 'moved_during_confirm',
                        'STOPPED': 'moved_after_stop'}[s.state], t)
            return {'publish': False, 'zero': False}
        if s.state == 'ZERO_PUB':
            if hold_cmd_ok_now:
                s.state = 'CONFIRM'
                s.log.append((t, 'CONFIRM', None))
            return {'publish': True, 'zero': True}
        if s.state == 'CONFIRM':
            return {'publish': True, 'zero': True}
        if s.state == 'STOPPED':
            if t - s.t_stop >= self.c.observe_s - 1e-9:
                s.state = 'DONE'
                s.log.append((t, 'DONE', None))
                return {'publish': False, 'zero': False}
            return {'publish': True, 'zero': True}                 # p4：觀察期持續發布保持命令
        return {'publish': False, 'zero': False}

    def applied(self, phys_step, t, base3, arm6, src_seq):
        s = self.seq
        if s.state in ('DONE', 'FAIL'):
            return
        try:
            vals = [float(v) for v in list(base3) + list(arm6)]
            ok = (len(list(base3)) == 3 and len(list(arm6)) == 6 and all(math.isfinite(v) for v in vals)
                  and isinstance(phys_step, int) and not isinstance(phys_step, bool) and PSUP._fin(t)
                  and (src_seq is None or (isinstance(src_seq, int) and not isinstance(src_seq, bool))))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return self._fail('applied_invalid', t if PSUP._fin(t) else float('nan'))
        if s.state == 'STOPPED':
            src_old = src_seq is None or src_seq < 0 or (s.zero_seq0 is None or src_seq < s.zero_seq0)
            if src_old or not hold_cmd_ok(base3, arm6, self.c.applied_tol):
                return self._fail('old_command_after_stop', t)
            return
        if s.state != 'CONFIRM':
            return
        if s.last_phys is not None and phys_step == s.last_phys:
            return                                                    # 重送不累計
        if s.last_phys is not None and (phys_step < s.last_phys or not t > s.last_t):
            return self._fail('applied_order', t)
        good = (s.zero_seq0 is not None and src_seq is not None and src_seq >= s.zero_seq0
                and hold_cmd_ok(base3, arm6, self.c.applied_tol))
        consecutive = s.last_phys is not None and phys_step == s.last_phys + 1
        s.n_ok = (s.n_ok + 1 if consecutive else 1) if good else 0
        s.last_phys, s.last_t = phys_step, t
        if s.n_ok >= self.c.n_confirm:
            s.state, s.t_stop = 'STOPPED', t
            s.log.append((t, 'STOPPED', None))

    def outcome(self, t_end=None):
        o = super().outcome(t_end)
        o['stop_definition'] = self.stop_definition
        o['hold_bounds'] = {'v_lin': HOLD_V, 'w': HOLD_W, 'arm_tol': self.c.applied_tol}
        return o
