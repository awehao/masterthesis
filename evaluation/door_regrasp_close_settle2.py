#!/usr/bin/env python3
"""r2（2026-10-30；door_m4r_15：門在 2.023° 已停（RG7 末 3 s 位置只變 0.001°）但回報 qdot_door 持續 ≈ −0.003 rad/s（與位置變化不符）⇒ 收斂窗不成立、
逾時中止；且逾時中止參數鍵 t 與 _abort(t) 衝突 ⇒ TypeError 任務節點當掉；Codex m4t）：r1（door_regrasp_close_settle）的版本化複本，只改：
  (1) 停穩訊號改「門角位置差分」：任務端用既有 m.settle_sig（door_k3m_settle_sig.SettleRateWindow：同步流 0.1 s 後向差分，含缺口／步號／重送／倒退檢查），
      無效即連續窗重算；執行端 RateFromPos（0.1 s 跨距，步號須遞增、缺口 > 0.05 s 重置）；回報 qdot_door 只記錄；
  (2) SettleWindow 加缺口上限（任務端 0.3 s、執行端 0.05 s），缺口／無效／倒退重算，重送不增加成立時間；
  (3) 逾時中止參數鍵改 t_timeout。
  判準值（0.002 rad/s、連續 0.5 s、期限 5 s）與原回報門速 0.05 rad/s 中止保護、接觸、漂移、幾何、RG8 0.7° 全不變。
r1：拉關終點停穩後才放手（door_m4r_13 RG8 放手時門仍慣性滑動 ⇒ rg_door_moved；Codex m4r 選 A）。
  任務端 close_settle(cls)：mode 7 參考走完後（done_ref）保持終點目標與抓握，門速 ≤ 0.002 rad/s 連續 0.5 s 才准請求 RG8；
    期限自參考結束（第一次 done_ref）起算 5 s，不重設；逾時 ⇒ 中止 rg7_close_settle_timeout。
  執行端 RegraspExecCoreCS：gate8（提交 mode 8）另核同一條件（執行端每步門角、模擬時間）；不足 ⇒ gate8 不成立。
"""
import math

VERSION = 'door_regrasp_close_settle/r2'
QDOT_MAX = 0.002       # rad/s
HOLD_S = 0.5           # 連續
DEADLINE_S = 5.0       # 自 mode 7 參考結束起算
SPAN_S = 0.1           # 執行端位置差分跨距


class RateFromPos:
    """執行端門角位置差分速度：以 ≥ SPAN_S 前的最近樣本為基準；步號須遞增；缺口／倒退／非有限 ⇒ 重來（回傳 nan）；重送／跨距不足 ⇒ None。"""

    def __init__(self, span_s=SPAN_S, gap_max_s=0.05):
        self.span_s, self.gap_max_s = span_s, gap_max_s
        self.buf = []
        self.last = None

    def feed(self, t, q, step=None):
        if not (isinstance(t, (int, float)) and isinstance(q, (int, float)) and math.isfinite(t) and math.isfinite(q)):
            self.buf = []
            return float('nan')
        if self.buf:
            t_l, _q, s_l = self.buf[-1]
            back = t < t_l or (step is not None and s_l is not None and step < s_l)
            dup = t == t_l or (step is not None and s_l is not None and step == s_l)
            if back:
                self.buf = []
                return float('nan')
            if dup:
                return None
            if t - t_l > self.gap_max_s + 1e-9:
                self.buf = [(t, q, step)]
                return float('nan')
        self.buf.append((t, q, step))
        while len(self.buf) > 2 and self.buf[1][0] <= t - self.span_s:
            self.buf.pop(0)
        t0, q0, s0 = self.buf[0]
        if t - t0 < self.span_s - 1e-9:
            return None
        self.last = {'t0': t0, 't1': t, 'q0': q0, 'q1': q, 'step0': s0, 'step1': step}
        return (q - q0) / (t - t0)


class SettleWindow:
    """連續低門速窗：輸入 (t, rate)；非有限、超限、缺口 > gap_max_s 或倒退 ⇒ 重新起算；重送（t 不增）不增加成立時間。"""

    def __init__(self, qdot_max=QDOT_MAX, hold_s=HOLD_S, gap_max_s=0.3):
        self.qdot_max, self.hold_s, self.gap_max_s = qdot_max, hold_s, gap_max_s
        self.t0 = None
        self.t_last = None

    def feed(self, t, rate):
        if not (isinstance(t, (int, float)) and math.isfinite(t)):
            self.t0 = None
            return
        if self.t_last is not None and t <= self.t_last:
            if t < self.t_last:
                self.t0, self.t_last = None, None
            return
        gap = self.t_last is not None and t - self.t_last > self.gap_max_s + 1e-9
        if gap or not (isinstance(rate, (int, float)) and math.isfinite(rate)) or abs(rate) > self.qdot_max:
            self.t0 = None
        elif self.t0 is None:
            self.t0 = t
        self.t_last = t

    def ok(self, t):
        return self.t0 is not None and self.t_last is not None and t - self.t_last <= self.gap_max_s + 1e-9 and t - self.t0 >= self.hold_s - 1e-9

    def held(self, t):
        return 0.0 if self.t0 is None else t - self.t0


SIG_KEYS = ('valid', 'why', 'rate', 't0', 't1', 'q0', 'q1', 'step0', 'step1', 'n')


def close_settle(cls):
    class CloseSettle(cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.cs = {'version': VERSION, 'qdot_max': QDOT_MAX, 'hold_s': HOLD_S, 'deadline_s': DEADLINE_S, 'signal': 'm.settle_sig（位置差分）',
                       't_ref_end7': None, 't_ready': None, 'timeout': None}
            self.cs_win = SettleWindow(gap_max_s=0.3)
            self.cs_sig = None

        def _monitor_rg(self, m, act):
            if self.rg.get('k_committed') == 7 and self.rg.get('k_target') != 8:
                sg = getattr(m, 'settle_sig', None) or {}
                rate = sg.get('rate') if sg.get('valid') is True else float('nan')
                self.cs_win.feed(m.t, rate)
                self.cs_sig = {k: sg.get(k) for k in SIG_KEYS}
                t0 = self.cs['t_ref_end7']
                if t0 is not None and m.t - t0 > DEADLINE_S and self.cs['timeout'] is None:
                    self.cs['timeout'] = {'t_timeout': m.t, 'held_s': self.cs_win.held(m.t), 'sig': self.cs_sig, 'qdot_reported': m.qdot_door}
                    return self._rg_abort('rg7_close_settle_timeout', m, act, **self.cs['timeout'])
            return super()._monitor_rg(m, act)

        def _extra_ready(self, k, m):
            if k == 7:
                if self.cs['t_ref_end7'] is None:
                    self.cs['t_ref_end7'] = m.t
                    self.rg['events'].append({'t': m.t, 'close_ref_end': True, 'sig': self.cs_sig, 'qdot_reported': m.qdot_door})
                if not self.cs_win.ok(m.t):
                    return False, f"door_not_settled:held={self.cs_win.held(m.t):.2f}s sig={(self.cs_sig or {}).get('rate')}"
                if self.cs['t_ready'] is None:
                    self.cs['t_ready'] = m.t
                    self.rg['events'].append({'t': m.t, 'close_settled': True, 'sig': self.cs_sig, 'qdot_reported': m.qdot_door,
                                              'q_deg': math.degrees(m.q_door), 'wait_s': m.t - self.cs['t_ref_end7']})
            return super()._extra_ready(k, m)

        def rg_report(self):
            r = super().rg_report()
            r['close_settle'] = dict(self.cs)
            return r
    CloseSettle.__name__ = f'CloseSettle_{cls.__name__}'
    return CloseSettle


def exec_core_cs(RGX):
    """執行端：RegraspExecCore 子類；gate8 另核門角位置差分速度連續 0.5 s ≤ 0.002 rad/s。"""
    class RegraspExecCoreCS(RGX.RegraspExecCore):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.cs_win = SettleWindow(gap_max_s=0.05)
            self.cs_rate = RateFromPos(gap_max_s=0.05)
            self.cs_t = None
            self.cs_log = {'version': VERSION, 'gate8_refused_unsettled': 0, 'gate8_first_ok': None}

        def step(self, step, t, mode_sel, rec, T_tcp, base, arm_q, finger_q, door_q, base_cmd=None, arm_cmd=None):
            try:
                q, tt = float(door_q), float(t)
                v = self.cs_rate.feed(tt, q, step=int(step))
                if v is not None:
                    self.cs_win.feed(tt, v)
                self.cs_t = tt
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
                self.cs_log['gate8_first_ok'] = {'step': step, 't': self.cs_t, 'held_s': self.cs_win.held(self.cs_t), 'diff': dict(self.cs_rate.last or {})}
            return True, None
    return RegraspExecCoreCS
