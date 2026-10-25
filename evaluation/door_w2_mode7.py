#!/usr/bin/env python3
"""W2 r7：mode 2 交接 r2（補 Codex 20261026_023226 §1 三項誤收）＋閉爪卸載候選的求解端子狀態與 mode 2 准入（純邏輯）。
door_w2_mode5／mode6 不改；HeldSupervisorP4b7 直接繼承 door_w2_mode5.HeldSupervisorP4b5（不沿用 r6 的放行規則）。

一、HeldSupervisorP4b7（外層仍為 HELD；內部子狀態）
  sub：held →（unload 啟用時）unload → unloaded → release_open；未啟用卸載時 held → release_open。
  - next_mode()：held 且啟用卸載 ⇒ 1；held（未啟用）或 unloaded ⇒ 2；其餘 None。
  - on_commit(k, t_c, t_now)：由求解端以**已驗證**的本趟權威提交（validated_commit）呼叫：
    (a) k 必須＝next_mode()，否則 FAIL commit_order；
    (b) **通知當下先核暫緩期限**：已有超限記錄且 t_now − t_exc_first > defer_max_s ⇒ FAIL moved_in_held（不靠週期檢查）；
    (c) 已有超限且 t_c > t_exc_first ⇒ FAIL moved_before_commit；
    (d) **提交前證據**：速度窗緩衝中，最後一筆 ≤ t_c 的樣本距 t_c ≤ max_gap_s，且其前連續涵蓋 0.1 s（底盤、門）與 0.5 s（手臂），
        四項皆 ≤ 舊 HELD 門檻；不足或不連續 ⇒ FAIL pre_commit_evidence_missing；超限 ⇒ FAIL moved_before_commit；
    (e) k＝2 且啟用卸載：t_c ≥ t_unloaded，否則 FAIL commit_before_reconverge；
    (f) 通過 ⇒ k＝1：sub unload（記錄 t_c 與提交前底盤）；k＝2：sub release_open。超限記錄只在成功切換後清除。
  - 暫緩（狀態延遲／ack 遺失）：sub held／unloaded 中速度超舊門檻、且已受理 next_mode() 的請求（pending[k]），
    且超限在 k 的容許範圍內（k＝2：底盤、手臂低、門 ≤ 保護；k＝1：門 ≤ 保護、底盤偏離 ≤ base_dev）⇒ 不立即 FAIL，照常發布保持命令；
    t_exc_first 為受理後**首次**超限時刻，**不因之後恢復低速而清除**；t − t_exc_first > defer_max_s 仍未切換 ⇒ FAIL moved_in_held。
  - sub unload：門 ≤ 保護（door_rate_in_unload）、底盤相對提交前偏離 ≤ base_dev（base_dev_in_unload）、t − t_c1 ≤ unload_max_s（unload_timeout）；
    手臂不以停止門檻核（卸載瞬態；任務端漂移、幾何監看照舊）；連續 quiet_s 四項 ≤ 舊門檻且 t ≥ t_c1＋ramp_s ⇒ sub unloaded。
  - sub release_open：同 r6（底盤、手臂舊門檻；門 ≤ 保護）。
  - 門速「保護」為同步狀態門角差分窗口速度；執行端／任務端原有的 tensor 門速保護保留，兩者不是同一訊號。
二、validated_commit(st, run_id, k, req_seq, t_lo, t_now)：只接受本趟（run_id 相符）、未中止、mode_seq ≥ k、commits 列有 mode_seq＝k 且
   applied_confirmed is True、時刻有限且 t_lo ≤ t_c ≤ t_now、該筆對應的 commit_req_seqs＝求解端受理的 req_seq。否則 (None, why)。
三、ModeAuthority7：ModeAuthority5＋gate2（mode 2 准入；請求時與提交當下都核，比照 gate3）。
"""
from __future__ import annotations

import math

import door_k3m_pause_sup as PSUP
import door_w2_mode5 as M5

VERSION = 'door_w2_mode7/r7'
DOOR_RATE_PROTECT = 0.05
DEFER_MAX_S = 0.2


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class HeldSupervisorP4b7(M5.HeldSupervisorP4b5):
    stop_definition = 'active_hold_arm_win_0.5s_held_unload_release_open_handover'

    def __init__(self, *a, unload=False, door_rate_protect=DOOR_RATE_PROTECT, defer_max_s=DEFER_MAX_S, base_dev_m=0.02, base_dev_deg=2.0,
                 unload_max_s=4.0, ramp_s=1.0, quiet_s=0.5, **kw):
        super().__init__(*a, **kw)
        self.unload = bool(unload)
        self.door_rate_protect, self.defer_max_s = float(door_rate_protect), float(defer_max_s)
        self.base_dev_m, self.base_dev_rad = float(base_dev_m), math.radians(float(base_dev_deg))
        self.unload_max_s, self.ramp_s, self.quiet_s = float(unload_max_s), float(ramp_s), float(quiet_s)
        self.sub = 'held'
        self.pending = {1: False, 2: False}
        self.t_exc_first = None
        self.t_commit = {}
        self.t_known = {}
        self.base_c1 = None
        self.t_quiet0 = None
        self.t_unloaded = None
        self.checks = {}
        self.log7 = []

    # ---- 求解端回報
    def next_mode(self):
        if self.sub == 'held':
            return 1 if self.unload else 2
        if self.sub == 'unloaded':
            return 2
        return None

    def set_pending(self, k, flag, t=None):
        flag = bool(flag)
        if self.pending.get(k) != flag and len(self.log7) < 100:
            self.log7.append({'t': t, 'pending': k, 'flag': flag})
        self.pending[k] = flag

    def _window_before(self, t_c):
        """結束於 t_c 以前的連續窗口：(底盤線速, 角速, 門速, 手臂) 或 (None, why)。"""
        buf = list(self.ws.buf)
        idx = [i for i, s in enumerate(buf) if s is not None and s[1] <= t_c + 1e-9]
        if not idx:
            return None, 'no_sample_before_commit'
        j = idx[-1]
        b = buf[j]
        if t_c - b[1] > self.ws.max_gap_s + 1e-9:
            return None, 'last_sample_too_old'
        a = aa = None
        prev = b
        for s in reversed(buf[:j]):
            if s is None or prev[1] - s[1] > self.ws.max_gap_s or not (s[0] < prev[0]):
                break
            prev = s
            if a is None and b[1] - s[1] >= self.ws.win_s - 1e-9:
                a = s
            if b[1] - s[1] >= self.ws.arm_win_s - 1e-9:
                aa = s
                break
        if a is None or aa is None:
            return None, 'window_not_covered'
        dt, dta = b[1] - a[1], b[1] - aa[1]
        dyaw = math.atan2(math.sin(b[4] - a[4]), math.cos(b[4] - a[4]))
        return (math.hypot(b[2] - a[2], b[3] - a[3]) / dt, abs(dyaw) / dt, abs(b[5] - a[5]) / dt,
                max(abs(u - w) for u, w in zip(b[6], aa[6])) / dta), (b[2], b[3], b[4])

    def on_commit(self, k, t_c, t_now):
        s = self.seq
        if s.state != 'HELD':
            return False
        if not (_fin(t_c) and _fin(t_now)) or t_c > t_now + 1e-9:
            self._fail('commit_time_invalid', t_now if _fin(t_now) else float('nan'))
            return False
        chk = {'k': k, 't_commit': float(t_c), 't_known': float(t_now), 't_exc_first': self.t_exc_first, 'sub_before': self.sub}
        self.checks[k] = chk
        if k != self.next_mode():
            self._fail('commit_order', t_now)
            return False
        if self.t_exc_first is not None and t_now - self.t_exc_first > self.defer_max_s + 1e-9:
            chk['why'] = 'notice_after_defer_deadline'
            self._fail('moved_in_held', t_now)
            return False
        if self.t_exc_first is not None and t_c > self.t_exc_first + 1e-9:
            chk['why'] = 'exceed_before_commit'
            self._fail('moved_before_commit', t_now)
            return False
        v, extra = self._window_before(t_c)
        chk['pre_speeds'] = v if v is None else list(v)
        if v is None:
            chk['why'] = extra
            self._fail('pre_commit_evidence_missing', t_now)
            return False
        if not PSUP.WindowSpeed.low(v, self.th):
            chk['why'] = 'pre_commit_not_low'
            self._fail('moved_before_commit', t_now)
            return False
        if k == 2 and self.unload and (self.t_unloaded is None or t_c < self.t_unloaded - 1e-9):
            self._fail('commit_before_reconverge', t_now)
            return False
        self.t_commit[k], self.t_known[k] = float(t_c), float(t_now)
        self.t_exc_first = None
        if k == 1:
            self.sub, self.base_c1, self.t_quiet0 = 'unload', list(extra), None
            s.log.append((float(t_now), 'UNLOAD', None))
        else:
            self.sub = 'release_open'
            s.log.append((float(t_now), 'RELEASE_OPEN', None))
        chk['ok'] = True
        return True

    def _base_dev_ok(self, ref):
        b = next((x for x in reversed(self.ws.buf) if x is not None), None)
        if b is None or ref is None:
            return False
        dyaw = math.atan2(math.sin(b[4] - ref[2]), math.cos(b[4] - ref[2]))
        return math.hypot(b[2] - ref[0], b[3] - ref[1]) <= self.base_dev_m and abs(dyaw) <= self.base_dev_rad

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
            return super().cycle(t, solve_ok, hold_cmd_ok_now, wall)
        v = self.ws.speeds(t, wall)
        self.last_speeds = v
        if v is None:
            self._fail('measurement_invalid' if self.ws.fresh(t, wall) else 'measurement_stale', t)
            return {'publish': False, 'zero': False}
        low = PSUP.WindowSpeed.low(v, self.th)
        ok_pub = {'publish': True, 'zero': True}
        if self.sub == 'release_open':
            if not (v[0] <= self.th.base_lin and v[1] <= self.th.base_ang and v[3] <= self.th.arm):
                self._fail('moved_in_release_open', t)
                return {'publish': False, 'zero': False}
            if v[2] > self.door_rate_protect:
                self._fail('door_rate_in_release', t)
                return {'publish': False, 'zero': False}
            return ok_pub
        if self.sub == 'unload':
            if t - self.t_commit[1] > self.unload_max_s + 1e-9:
                self._fail('unload_timeout', t)
                return {'publish': False, 'zero': False}
            if v[2] > self.door_rate_protect:
                self._fail('door_rate_in_unload', t)
                return {'publish': False, 'zero': False}
            if not self._base_dev_ok(self.base_c1):
                self._fail('base_dev_in_unload', t)
                return {'publish': False, 'zero': False}
            if low:
                self.t_quiet0 = self.t_quiet0 if self.t_quiet0 is not None else float(t)
                if t - self.t_quiet0 >= self.quiet_s - 1e-9 and t >= self.t_commit[1] + self.ramp_s - 1e-9:
                    self.sub, self.t_unloaded = 'unloaded', float(t)
                    s.log.append((float(t), 'UNLOADED', None))
            else:
                self.t_quiet0 = None
            return ok_pub
        # sub held／unloaded：停止確認（舊門檻）＋有界暫緩
        if self.t_exc_first is not None and t - self.t_exc_first > self.defer_max_s + 1e-9:
            self._fail('moved_in_held', t)
            self.log7.append({'t': t, 'defer_expired': True, 't_exc_first': self.t_exc_first})
            return {'publish': False, 'zero': False}
        if low:
            return ok_pub
        k = self.next_mode()
        if k == 2:
            allow = v[0] <= self.th.base_lin and v[1] <= self.th.base_ang and v[3] <= self.th.arm and v[2] <= self.door_rate_protect
        elif k == 1:
            ref = next((x for x in self.ws.buf if x is not None), None)
            allow = v[2] <= self.door_rate_protect and self._base_dev_ok(None if ref is None else (ref[2], ref[3], ref[4]))
        else:
            allow = False
        if allow and self.pending.get(k):
            if self.t_exc_first is None:
                self.t_exc_first = float(t)
                self.log7.append({'t': t, 'defer_start': k, 'speeds': list(v)})
            return ok_pub
        self._fail('moved_in_held', t)
        return {'publish': False, 'zero': False}

    def outcome(self, t_end=None):
        o = super().outcome(t_end)
        o['transitions'].update(UNLOAD=self.t_known.get(1), UNLOADED=self.t_unloaded, RELEASE_OPEN=self.t_known.get(2))
        o.update(version=VERSION, sub=self.sub, unload=self.unload, commit_t={str(k): v for k, v in self.t_commit.items()},
                 known_t={str(k): v for k, v in self.t_known.items()}, checks={str(k): v for k, v in self.checks.items()},
                 door_rate_protect=self.door_rate_protect, door_rate_signal='sync_state 門角差分窗口速度（非 tensor 門速）',
                 defer_max_s=self.defer_max_s, log7=self.log7[:60])
        return o


def validated_commit(st, run_id, k, req_seq, t_lo, t_now):
    if not isinstance(st, dict) or st.get('run_id') != run_id:
        return None, 'wrong_run_or_malformed'
    if st.get('aborted'):
        return None, 'aborted'
    ms = st.get('mode_seq')
    if type(ms) is not int or ms < k:
        return None, 'not_committed'
    cs, rs = st.get('commits'), st.get('commit_req_seqs')
    if not isinstance(cs, list) or not isinstance(rs, list) or len(cs) != len(rs):
        return None, 'commits_missing'
    hit = [(c, r) for c, r in zip(cs, rs) if isinstance(c, dict) and c.get('mode_seq') == k]
    if len(hit) != 1:
        return None, 'commit_k_missing_or_dup'
    c, r = hit[0]
    if c.get('applied_confirmed') is not True:
        return None, 'not_applied_confirmed'
    if type(req_seq) is not int or r != req_seq:
        return None, 'req_seq_mismatch'
    tc = c.get('t')
    if not _fin(tc) or not _fin(t_lo) or not _fin(t_now) or tc < t_lo - 1e-9 or tc > t_now + 1e-9:
        return None, 'commit_time_inconsistent'
    return float(tc), None


class ModeAuthority7(M5.ModeAuthority5):
    def __init__(self, *a, gate2=None, **kw):
        super().__init__(*a, **kw)
        self.gate2 = gate2

    def _gate2(self, step):
        if self.gate2 is None:
            return True, None                                                 # 未設 gate2 ⇒ 與 r5 相同
        try:
            r = self.gate2(step)
        except Exception as e:                                                # noqa: BLE001
            return False, f'gate_error:{e!r}'
        if isinstance(r, tuple):
            return bool(r[0]), r[1]
        return bool(r), None if r else 'not_unloaded_fresh'

    def on_request(self, msg, step, t):
        if isinstance(msg, dict) and msg.get('mode_seq') == 2 and msg.get('run_id') == self.run_id and self.aborted is None \
                and self.pending is None and self.mode_seq == 1 and not any(self._key(msg) == self._key(c['request']) for c in self.commits):
            ok, why = self._gate2(step)
            if not ok:
                return self._nack(f'unload_evidence:{why}', msg, step, t)
        return super().on_request(msg, step, t)

    def select(self, step, t, u_prev_applied):
        if self.aborted is None and self.await_confirm is None and self.pending is not None \
                and self.pending['request']['mode_seq'] == 2 and self.cand is not None:
            ok, why = self._gate2(step)
            if not ok:
                req = self.pending['request']
                self.pending, self.cand = None, None
                self._nack(f'unload_evidence_at_commit:{why}', req, step, t)
        return super().select(step, t, u_prev_applied)
