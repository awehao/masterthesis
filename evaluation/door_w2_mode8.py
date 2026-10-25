#!/usr/bin/env python3
"""W2 r8：求解端監督器 r8＋權威 r8（純邏輯；Codex 20261026_025647 §1、§3B、§4）。door_w2_mode7 不改；以子類別擴充。

HeldSupervisorP4b8(HeldSupervisorP4b7)：
  1. 卸載重新收斂窗與執行端統一：須先得知執行端已驗證的 t_settling（斜坡完成且讀回＝5 N；來自本趟權威狀態 unload 欄，
     t_c1＋ramp_s − 0.02 ≤ t_settling ≤ 現在），之後才累積 quiet_s；斜坡期間的低速不計入。另須卸載瞬態監看 CONVERGED。
  2. 瞬態監看（door_k3b_transient）：
     - unload：mode 1 提交起，以瞬態規則（TCP 位移 ≤ 5 mm／2°、TCP 速度 ≤ 0.02 m/s、t_max 4.0 s 內收斂）取代手臂停止門檻；
     - release_open：mode 2 提交起同規則（t_max 1.5 s）；收斂後 sub＝release_settled，手臂回到舊 HELD 門檻 0.001 rad/s。
     - 底盤在兩段仍為：unload 偏離 ≤ base_dev、release 舊 HELD 速度門檻（主動保持）。門：保護上限（門角差分）。
     - 瞬態失敗 ⇒ FAIL unload_transient:<why>／release_transient:<why>。
  2b. 暫緩期間（狀態延遲：已受理 k 的請求、提交通知未到）手臂可能已開始回彈：以**暫定瞬態監看**（同一位移／速度界，
      起點＝首次超限前 win_s 的最後一筆樣本）約束手臂，門 ≤ 保護、底盤（k＝2 舊門檻；k＝1 偏離 ≤ base_dev）；
      暫緩期限 0.2 s 與首次超限時刻不清除沿用 r7。暫定監看失敗 ⇒ FAIL moved_in_held。
  3. mode 3 交接只准自 release_settled：release_open 中交接 ⇒ FAIL handover_before_transient_converged；sub held（未得知張爪）⇒ FAIL handover_without_release。
ModeAuthority8(ModeAuthority7)：unload=True 時必須提供 gate2，否則建構即 ValueError；執行期 gate2 缺失 ⇒ 拒絕准入（不沿用「未設即放行」）。
"""
from __future__ import annotations

import math

import door_k3b_transient as TT
import door_k3m_pause_sup as PSUP
import door_w2_mode7 as M7

VERSION = 'door_w2_mode8/r8'


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class HeldSupervisorP4b8(M7.HeldSupervisorP4b7):
    stop_definition = 'active_hold_arm_win_0.5s_held_unload_release_transient_settled_handover'

    def __init__(self, *a, tcp_fn=None, t_max_unload=TT.T_MAX_UNLOAD, t_max_release=TT.T_MAX_RELEASE, **kw):
        super().__init__(*a, **kw)
        if tcp_fn is None:
            raise ValueError('HeldSupervisorP4b8 需要 tcp_fn（瞬態監看）')
        self.tcp_fn = tcp_fn
        self.tm = {1: TT.TransientMonitor(tcp_fn, t_max_unload, name='unload'), 2: TT.TransientMonitor(tcp_fn, t_max_release, name='release')}
        self.t_settling = None
        self.settling_log = []
        self.prov = None                     # 暫緩期間的暫定瞬態監看
        self._prov_fed = None

    # ---- 執行端卸載狀態（求解端已驗證本趟後才呼叫）
    def set_exec_settling(self, t_s, t_now):
        if self.sub != 'unload' or self.t_settling is not None:
            return False
        t1 = self.t_commit.get(1)
        ok = _fin(t_s) and _fin(t_now) and t1 is not None and t1 + self.ramp_s - 0.02 - 1e-9 <= t_s <= t_now + 1e-9
        if len(self.settling_log) < 20:
            self.settling_log.append({'t_s': t_s, 't_now': t_now, 'ok': bool(ok)})
        if ok:
            self.t_settling = float(t_s)
        return ok

    def _sample_before(self, t_c):
        for s in reversed(self.ws.buf):
            if s is not None and s[1] <= t_c + 1e-9:
                return s
        return None

    def on_commit(self, k, t_c, t_now):
        ok = super().on_commit(k, t_c, t_now)
        if ok:
            self.prov, self._prov_fed = None, None
        if ok and k in self.tm:
            s = self._sample_before(t_c)
            if s is None or not self.tm[k].start(s[1], (s[2], s[3], s[4]), s[6]):
                self._fail(f"{'unload' if k == 1 else 'release'}_transient:start_invalid", t_now)
                return False
        return ok

    def feed_state(self, step, t, x, y, yaw, q_door, q_arm, wall=None):
        r = super().feed_state(step, t, x, y, yaw, q_door, q_arm, wall)
        k = 1 if self.sub == 'unload' else 2 if self.sub == 'release_open' else None
        if r and k is not None and self.tm[k].state == 'ACTIVE':
            self.tm[k].feed(float(t), (x, y, yaw), list(q_arm))
        return r

    def _prov_check(self, t, k, v):
        """暫緩中的容許：門 ≤ 保護、底盤（k＝2 舊門檻；k＝1 偏離 ≤ base_dev）、手臂以暫定瞬態監看。回傳是否容許。"""
        if v[2] > self.door_rate_protect:
            return False
        if k == 2 and (v[0] > self.th.base_lin or v[1] > self.th.base_ang):
            return False
        if k == 1:
            ref = next((x for x in self.ws.buf if x is not None), None)
            if ref is None or not self._base_dev_ok((ref[2], ref[3], ref[4])):
                return False
        if self.prov is None:
            s0 = self._sample_before(float(t) - self.ws.win_s)
            self.prov = TT.TransientMonitor(self.tcp_fn, self.defer_max_s + self.ws.win_s + 0.05, name=f'defer{k}')
            if s0 is None or not self.prov.start(s0[1], (s0[2], s0[3], s0[4]), s0[6]):
                return False
            self._prov_fed = s0[1]
        for x in list(self.ws.buf):
            if x is not None and x[1] > self._prov_fed + 1e-9:
                self.prov.feed(x[1], (x[2], x[3], x[4]), x[6])
                self._prov_fed = x[1]
        return self.prov.state != 'FAILED'

    def cycle(self, t, solve_ok, hold_cmd_ok_now, wall=None):
        s = self.seq
        if s.state == 'HELD' and self.sub in ('held', 'unloaded') and self.unload is not None:
            return self._cycle_held(t, solve_ok, hold_cmd_ok_now, wall)
        if s.state != 'HELD' or self.sub not in ('unload', 'release_open', 'release_settled'):
            return super().cycle(t, solve_ok, hold_cmd_ok_now, wall)
        if not PSUP._fin(t):
            self._fail('time_invalid', t)
            return {'publish': False, 'zero': False}
        if not solve_ok:
            self._fail('solve_failed_in_held', t)
            return {'publish': False, 'zero': False}
        if t - self.t_held > self.held_max_s + 1e-9:
            return M7.M5.HeldSupervisorP4b5.cycle(self, t, solve_ok, hold_cmd_ok_now, wall)
        v = self.ws.speeds(t, wall)
        self.last_speeds = v
        if v is None:
            self._fail('measurement_invalid' if self.ws.fresh(t, wall) else 'measurement_stale', t)
            return {'publish': False, 'zero': False}
        no = {'publish': False, 'zero': False}
        if self.sub == 'unload':
            tm = self.tm[1]
            if tm.state == 'FAILED':
                self._fail(f'unload_transient:{tm.why}', t)
                return no
            if t - self.t_commit[1] > self.unload_max_s + 1e-9:
                self._fail('unload_timeout', t)
                return no
            if v[2] > self.door_rate_protect:
                self._fail('door_rate_in_unload', t)
                return no
            if not self._base_dev_ok(self.base_c1):
                self._fail('base_dev_in_unload', t)
                return no
            low = PSUP.WindowSpeed.low(v, self.th)
            if low and self.t_settling is not None and t >= self.t_settling - 1e-9:
                self.t_quiet0 = self.t_quiet0 if self.t_quiet0 is not None else float(t)
                if t - self.t_quiet0 >= self.quiet_s - 1e-9 and tm.state == 'CONVERGED':
                    self.sub, self.t_unloaded = 'unloaded', float(t)
                    s.log.append((float(t), 'UNLOADED', None))
            else:
                self.t_quiet0 = None
            return {'publish': True, 'zero': True}
        tm = self.tm[2]
        if tm.state == 'FAILED':
            self._fail(f'release_transient:{tm.why}', t)
            return no
        if v[0] > self.th.base_lin or v[1] > self.th.base_ang:
            self._fail('moved_in_release_open', t)
            return no
        if v[2] > self.door_rate_protect:
            self._fail('door_rate_in_release', t)
            return no
        if self.sub == 'release_open' and tm.state == 'CONVERGED':
            self.sub = 'release_settled'
            s.log.append((float(t), 'RELEASE_SETTLED', None))
        if self.sub == 'release_settled' and v[3] > self.th.arm:
            self._fail('moved_in_release_settled', t)
            return no
        return {'publish': True, 'zero': True}

    def _cycle_held(self, t, solve_ok, hold_cmd_ok_now, wall):
        """sub held／unloaded：停止確認（舊門檻）＋有界暫緩（r7 規則，手臂改以暫定瞬態監看）。"""
        s = self.seq
        if not PSUP._fin(t):
            self._fail('time_invalid', t)
            return {'publish': False, 'zero': False}
        if not solve_ok:
            self._fail('solve_failed_in_held', t)
            return {'publish': False, 'zero': False}
        if t - self.t_held > self.held_max_s + 1e-9:
            return M7.M5.HeldSupervisorP4b5.cycle(self, t, solve_ok, hold_cmd_ok_now, wall)
        v = self.ws.speeds(t, wall)
        self.last_speeds = v
        if v is None:
            self._fail('measurement_invalid' if self.ws.fresh(t, wall) else 'measurement_stale', t)
            return {'publish': False, 'zero': False}
        if self.t_exc_first is not None and t - self.t_exc_first > self.defer_max_s + 1e-9:
            self._fail('moved_in_held', t)
            self.log7.append({'t': t, 'defer_expired': True, 't_exc_first': self.t_exc_first})
            return {'publish': False, 'zero': False}
        if PSUP.WindowSpeed.low(v, self.th) and self.t_exc_first is None:
            return {'publish': True, 'zero': True}
        k = self.next_mode()
        if k is not None and self.pending.get(k) and (PSUP.WindowSpeed.low(v, self.th) or self._prov_check(t, k, v)):
            if self.t_exc_first is None:
                self.t_exc_first = float(t)
                self.log7.append({'t': t, 'defer_start': k, 'speeds': list(v)})
            return {'publish': True, 'zero': True}
        self._fail('moved_in_held', t)
        return {'publish': False, 'zero': False}

    def handover(self, t):
        if self.seq.state != 'HELD':
            return False
        if self.sub == 'release_open':
            self._fail('handover_before_transient_converged', t)
            return False
        if self.sub != 'release_settled':
            self._fail('handover_without_release', t)
            return False
        return super().handover(t)

    def outcome(self, t_end=None):
        o = super().outcome(t_end)
        o['transitions']['RELEASE_SETTLED'] = next((x[0] for x in self.seq.log if x[1] == 'RELEASE_SETTLED'), None)
        o.update(version=VERSION, t_settling=self.t_settling, settling_log=self.settling_log,
                 transient={str(k): m.report() for k, m in self.tm.items()})
        return o


class ModeAuthority8(M7.ModeAuthority7):
    def __init__(self, *a, unload=False, **kw):
        super().__init__(*a, **kw)
        self.unload = bool(unload)
        if self.unload and self.gate2 is None:
            raise ValueError('ModeAuthority8：unload 啟用時必須提供 gate2')

    def _gate2(self, step):
        if self.unload and self.gate2 is None:
            return False, 'gate2_missing'
        return super()._gate2(step)


def exec_unload_of(st):
    """本趟權威狀態訊息（已由呼叫端核 run_id）中的執行端卸載摘要；格式不合 ⇒ None。"""
    u = (st or {}).get('unload') if isinstance(st, dict) else None
    return u if isinstance(u, dict) and isinstance(u.get('state'), str) else None
