#!/usr/bin/env python3
"""拉關終點停穩後才放手（2026-10-30；door_m4r_13 RG8 放手時門仍以 −0.0066 rad/s 慣性滑動，q_rel8 2.099° → 1.398° 超 0.7° ⇒ rg_door_moved；
Codex m4r 選 A）。
  任務端 close_settle(cls)：mode 7 參考走完後（done_ref）保持終點目標與抓握，實測 |qdot_door| ≤ 0.002 rad/s **連續 0.5 s** 才准請求 RG8；
    期限自參考結束（第一次 done_ref）起算 5 s，不重設；逾時 ⇒ 中止 rg7_close_settle_timeout。
  執行端 RegraspExecCoreCS：gate8（提交 mode 8）另核同一條件（以執行端每步門角差分、模擬時間）；不足 ⇒ gate8 不成立。
  HoldConfirm、E0 漂移基準、接觸、幾何保護與 RG8 的 0.7° 門檻全不變。
"""
import math

VERSION = 'door_regrasp_close_settle/r1'
QDOT_MAX = 0.002       # rad/s
HOLD_S = 0.5           # 連續
DEADLINE_S = 5.0       # 自 mode 7 參考結束起算


class SettleWindow:
    """連續低門速窗：輸入 (t, qdot)；任一筆非有限或超限 ⇒ 重新起算。"""

    def __init__(self, qdot_max=QDOT_MAX, hold_s=HOLD_S):
        self.qdot_max, self.hold_s = qdot_max, hold_s
        self.t0 = None
        self.t_last = None

    def feed(self, t, qdot):
        if not (isinstance(qdot, (int, float)) and math.isfinite(qdot) and isinstance(t, (int, float)) and math.isfinite(t)) \
                or abs(qdot) > self.qdot_max:
            self.t0 = None
        elif self.t0 is None:
            self.t0 = t
        self.t_last = t

    def ok(self, t):
        return self.t0 is not None and self.t_last is not None and t - self.t_last <= 0.05 + 1e-9 and t - self.t0 >= self.hold_s - 1e-9

    def held(self, t):
        return 0.0 if self.t0 is None else t - self.t0


def close_settle(cls):
    class CloseSettle(cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.cs = {'version': VERSION, 'qdot_max': QDOT_MAX, 'hold_s': HOLD_S, 'deadline_s': DEADLINE_S,
                       't_ref_end7': None, 't_ready': None, 'timeout': None}
            self.cs_win = SettleWindow()

        def _monitor_rg(self, m, act):
            if self.rg.get('k_committed') == 7 and self.rg.get('k_target') != 8:
                self.cs_win.feed(m.t, m.qdot_door)
                t0 = self.cs['t_ref_end7']
                if t0 is not None and m.t - t0 > DEADLINE_S and self.cs['timeout'] is None:
                    self.cs['timeout'] = {'t': m.t, 'held_s': self.cs_win.held(m.t), 'qdot': m.qdot_door}
                    return self._rg_abort('rg7_close_settle_timeout', m, act, **self.cs['timeout'])
            return super()._monitor_rg(m, act)

        def _extra_ready(self, k, m):
            if k == 7:
                if self.cs['t_ref_end7'] is None:
                    self.cs['t_ref_end7'] = m.t
                    self.rg['events'].append({'t': m.t, 'close_ref_end': True, 'qdot': m.qdot_door})
                if not self.cs_win.ok(m.t):
                    return False, f'door_not_settled:held={self.cs_win.held(m.t):.2f}s qdot={m.qdot_door}'
                if self.cs['t_ready'] is None:
                    self.cs['t_ready'] = m.t
                    self.rg['events'].append({'t': m.t, 'close_settled': True, 'qdot': m.qdot_door,
                                              'q_deg': math.degrees(m.q_door), 'wait_s': m.t - self.cs['t_ref_end7']})
            return super()._extra_ready(k, m)

        def rg_report(self):
            r = super().rg_report()
            r['close_settle'] = dict(self.cs)
            return r
    CloseSettle.__name__ = f'CloseSettle_{cls.__name__}'
    return CloseSettle


def exec_core_cs(RGX):
    """執行端：RegraspExecCore 子類；gate8 另核門角差分速度連續 0.5 s ≤ 0.002 rad/s。"""
    class RegraspExecCoreCS(RGX.RegraspExecCore):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.cs_win = SettleWindow()
            self.cs_prev = None
            self.cs_t = None
            self.cs_log = {'version': VERSION, 'gate8_refused_unsettled': 0, 'gate8_first_ok': None}

        def step(self, step, t, mode_sel, rec, T_tcp, base, arm_q, finger_q, door_q, base_cmd=None, arm_cmd=None):
            try:
                q, tt = float(door_q), float(t)
                if self.cs_prev is not None and tt > self.cs_prev[0]:
                    self.cs_win.feed(tt, (q - self.cs_prev[1]) / (tt - self.cs_prev[0]))
                self.cs_prev, self.cs_t = (tt, q), tt
            except (TypeError, ValueError):
                self.cs_win.feed(float('nan'), float('nan'))
            return super().step(step, t, mode_sel, rec, T_tcp, base, arm_q, finger_q, door_q, base_cmd=base_cmd, arm_cmd=arm_cmd)

        def _gate8(self, step):
            ok, why = super()._gate8(step)
            if not ok:
                return ok, why
            if self.cs_t is None or not self.cs_win.ok(self.cs_t):
                self.cs_log['gate8_refused_unsettled'] += 1
                return False, f'door_not_settled:held={0.0 if self.cs_t is None else self.cs_win.held(self.cs_t):.2f}s'
            if self.cs_log['gate8_first_ok'] is None:
                self.cs_log['gate8_first_ok'] = {'step': step, 't': self.cs_t, 'held_s': self.cs_win.held(self.cs_t)}
            return True, None
    return RegraspExecCoreCS
