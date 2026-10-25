#!/usr/bin/env python3
"""K3-B 卸載監看 r2（door_k3b_unload.py〔r1〕的版本化複本；Codex 20261026_025647 必修 A、B）。r2 差異：
  A1 缺步計入連續無效時間：下一筆有效資料到達時若距上一筆有效 > invalid_max_s ⇒ FAILED data_invalid（不洗掉）。
  A2 門 0.1 s 保護獨立於手臂 0.5 s 長窗（門速窗齊即核）；原 tensor 門速保護不在本模組、照舊保留。
  B1 confirmed_fresh(step, t)：步號須 ∈ [last_step, last_step＋fresh_steps]、時間有限且 last_t ≤ t ≤ last_t＋fresh_steps·dt；過去步、未來步、非法 ⇒ False。
  B2 task_gate：先拒非法／倒退時間、逾時、他趟、過期狀態，再判兩端完成。
  B3 abort(step, t, why)：中止優先；ABORTED 後 command() 回 None（不再設定上限）。
  B4 base0 改取第一筆有效步的實測底盤（與 room step0 列同源）。
  B5 report 增 t_settling（斜坡完成且讀回＝f_lo 的時刻），供求解端起算重新收斂窗。
以下為 r1 說明：
K3-B 候選：閉爪下有界卸載，確認低速後才准張爪（純邏輯；Codex 20261026_023226 §3；規格 K3B_unload_spec_draft1.md）。
**待驗證候選**：不預設卸載能消除張爪後門運動，也不把接觸衝量當成已確認根因。

一、ExecUnloadMonitor（執行端，逐物理步；mode 1 提交步 start）
  - 手指驅動上限依登錄斜坡由 f_hi 線性降到 f_lo：f_cmd(t)＝f_hi＋(f_lo−f_hi)·min(1, (t−t0)/ramp_s)；每步讀回須＝本步設定（容差 TOL）。
  - 狀態：RAMPING →（斜坡結束且讀回＝f_lo）SETTLING →（連續 quiet_s 低速）UNLOADED；任何不合格 ⇒ FAILED（終止，不重試、不加力）。
  - 每步核（**先核期限再轉移**，同 ExecReleaseMonitor）：
    - 期限：t − t0 > unload_max_s 而尚未 UNLOADED ⇒ FAILED timeout；
    - 資料：步號連續、時間遞增；缺步／無效 ⇒ 該步無效、低速窗歸零；連續無效 > invalid_max_s ⇒ FAILED data_invalid；
    - 失接觸：兩指—橫桿力（bar＋ambiguous）皆 ≥ contact_min_n，否則 FAILED contact_lost；
    - 門速保護：0.1 s 門角差分 ≤ door_rate_protect，否則 FAILED door_rate；
    - 底盤偏離：相對 t0 實測底盤 ≤ base_dev_m／base_dev_deg，否則 FAILED base_dev；
    - 讀回：≠ 本步設定 ⇒ FAILED readback_mismatch。
  - 低速（quiet）：底盤線速／角速、門速（0.1 s 窗）與手臂最大關節速（0.5 s 窗）皆 ≤ 舊 HELD 門檻（door_k3m_pause_sup.StopTh）。
  - UNLOADED 後到 mode 2 提交前持續監看：保護違規 ⇒ FAILED；低速中斷 ⇒ 回 SETTLING 重新累積（期限照舊）。
  - mode 2 提交步 handoff()：停止本監看（交給張爪後的 ExecReleaseMonitor）。
  - confirmed_fresh(step)：UNLOADED、最後有效步＝step 或 step−1、目前仍低速。供 mode 2 准入（請求時與提交當下都核）。
二、task_gate(state, t_commit1, t_now, solver_outcome)：任務端決定 'wait'／'go'／'abort:<why>'；'go' 須執行端 UNLOADED 且求解端子狀態 unloaded；
   任一端失敗 ⇒ abort；超過 unload_max_s＋余裕仍未 go ⇒ abort。
"""
from __future__ import annotations

import math

import door_k3m_pause_sup as PSUP

VERSION = 'door_k3b_unload/2'
TOL = 1e-3
_TH = PSUP.StopTh()
REG_UNLOAD = {
    'f_hi_n': 20.0, 'f_lo_n': 5.0,                    # 暫停期上限（W1e）→ 原上限
    'ramp_s': 1.0,                                    # 候選：斜坡時間
    'unload_max_s': 4.0,                              # 候選：mode 1 提交起到 UNLOADED 的上限（斜坡 1 s＋收斂 ≤ 3 s；W1e 5→20 N 切換後約 2 s 收斂）
    'quiet_s': 0.5,                                   # 低速持續窗（同 HELD 手臂長窗 0.5 s）
    'contact_min_n': 0.5,                             # 既有 GRIP_MIN_N（door_k2_check／door_k3a_check）
    'door_rate_protect': 0.05,                        # 任務端已登錄 door_abort_rate（rad/s；此處為門角差分，非 tensor 門速）
    'base_dev_m': 0.02, 'base_dev_deg': 2.0,          # door_k3b_release REG（底盤偏離）
    'invalid_max_s': 0.2, 'fresh_steps': 1, 'dt': 0.01,
    'quiet_base_lin': _TH.base_lin, 'quiet_base_ang': _TH.base_ang, 'quiet_door': _TH.door, 'quiet_arm': _TH.arm,
    'win_s': 0.1, 'arm_win_s': 0.5,
    'task_margin_s': 0.5, 'state_age_max_s': 0.2,
}


def _fin(*xs):
    try:
        return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


def f_cmd(r, t0, t):
    return r['f_hi_n'] + (r['f_lo_n'] - r['f_hi_n']) * min(1.0, max(0.0, (t - t0) / r['ramp_s']))


class ExecUnloadMonitor:
    def __init__(self, reg=None):
        self.r = dict(REG_UNLOAD, **(reg or {}))
        self.state = 'IDLE'
        self.why = None
        self.t0 = self.step0 = None
        self.base0 = None
        self.t = {}
        self.events = []
        self.hist = []                      # (step, t, x, y, yaw, q_door, arm6) 有效步
        self.last_step = None
        self.last_t = None
        self.last_valid_t = None
        self.t_quiet0 = None
        self.set_log = []                   # (step, t, set, readback)
        self.n = {'steps': 0, 'invalid': 0, 'missing': 0, 'dup': 0}
        self.worst = {'contact_min_n': math.inf, 'door_rate': 0.0, 'base_dev_m': 0.0, 'base_dev_deg': 0.0}

    def _go(self, to, step, t, **kw):
        self.state = to
        self.t[to] = t
        self.events.append(dict({'t': t, 'step': step, 'to': to}, **kw))

    def _fail(self, why, step, t):
        if self.state != 'FAILED':
            self.why = why
            self._go('FAILED', step, t, why=why)
        return self.state

    def start(self, step, t, base=None):
        if self.state != 'IDLE':
            return False
        if type(step) is not int or not _fin(t):
            self._fail('start_invalid', step, t)
            return False
        self.t0, self.step0, self.base0 = float(t), int(step), None                   # r2：base0＝第一筆有效步實測
        self.last_step, self.last_t, self.last_valid_t = step - 1, float(t) - self.r['dt'], float(t)
        self._go('RAMPING', step, t)
        return True

    def command(self, t):
        """本步要設定的手指上限（呼叫端照此設定並讀回）；非 RAMPING／SETTLING／UNLOADED（含 ABORTED、FAILED）⇒ None（不改設定）。"""
        if self.state not in ('RAMPING', 'SETTLING', 'UNLOADED') or self.t0 is None or not _fin(t):
            return None
        return f_cmd(self.r, self.t0, t)

    def abort(self, step, t, why):
        """r2：中止優先（呼叫端在本步設定上限之前呼叫）。"""
        if self.state in ('RAMPING', 'SETTLING', 'UNLOADED'):
            self.why = f'abort:{why}'
            self._go('ABORTED', step, t, why=self.why)
            return True
        return False

    def _speeds(self):
        h = self.hist
        if len(h) < 2:
            return None, None
        b = h[-1]

        def start(win):
            prev = b
            for s in reversed(h[:-1]):
                if prev[0] - s[0] != 1:
                    return None
                prev = s
                if b[1] - s[1] >= win - 1e-9:
                    return s
            return None
        a, aa = start(self.r['win_s']), start(self.r['arm_win_s'])
        if a is None:
            return None, None
        dt = b[1] - a[1]
        dyaw = math.atan2(math.sin(b[4] - a[4]), math.cos(b[4] - a[4]))
        short = (math.hypot(b[2] - a[2], b[3] - a[3]) / dt, abs(dyaw) / dt, abs(b[5] - a[5]) / dt)
        if aa is None:
            return short, None                                                     # r2：門／底盤短窗獨立回傳
        return short, short + (max(abs(u - v) for u, v in zip(b[6], aa[6])) / (b[1] - aa[1]),)

    def step(self, step, t, readback, set_value, contacts, contacts_valid, door_q, base, arm6):
        """readback／set_value：本步讀回與本步設定（兩指）；contacts：兩指 bar＋ambiguous 力。"""
        if self.state not in ('RAMPING', 'SETTLING', 'UNLOADED'):
            return self.state
        r = self.r
        if type(step) is not int or not _fin(t):
            return self._fail('step_or_time_invalid', step, t)
        if step == self.last_step:
            if t == self.last_t:
                self.n['dup'] += 1
                return self.state
            return self._fail('time_conflict', step, t)
        if step < self.last_step or not t > self.last_t:
            return self._fail('order', step, t)
        if t - self.t0 > r['unload_max_s'] + 1e-9 and self.state != 'UNLOADED':          # 先核期限
            return self._fail('timeout', step, t)
        gap = step - self.last_step - 1
        if gap > 0:
            self.n['missing'] += gap
            self.hist = []
            self.t_quiet0 = None
        self.last_step, self.last_t = step, float(t)
        if t - self.last_valid_t > r['invalid_max_s'] + 1e-9:                          # r2 A1：缺步／停更計入連續無效時間
            return self._fail('data_invalid', step, t)
        self.n['steps'] += 1
        ok = (contacts_valid is True and contacts is not None and len(list(contacts)) == 2 and _fin(*contacts)
              and _fin(door_q) and base is not None and len(list(base)) == 3 and _fin(*base)
              and arm6 is not None and len(list(arm6)) == 6 and _fin(*arm6)
              and readback is not None and len(list(readback)) == 2 and _fin(*readback) and _fin(set_value))
        if not ok:
            self.n['invalid'] += 1
            self.hist = []
            self.t_quiet0 = None
            if t - self.last_valid_t > r['invalid_max_s'] + 1e-9:
                return self._fail('data_invalid', step, t)
            return self.state
        self.last_valid_t = float(t)
        want = f_cmd(r, self.t0, t)
        self.set_log.append((step, float(t), float(set_value), [float(x) for x in readback]))
        if abs(float(set_value) - want) > TOL or any(abs(float(x) - float(set_value)) > TOL for x in readback):
            return self._fail('readback_mismatch', step, t)
        fmin = min(float(x) for x in contacts)
        self.worst['contact_min_n'] = min(self.worst['contact_min_n'], fmin)
        if fmin < r['contact_min_n']:
            return self._fail('contact_lost', step, t)
        b = [float(x) for x in base]
        if self.base0 is None:
            self.base0 = list(b)
        dev = math.hypot(b[0] - self.base0[0], b[1] - self.base0[1])
        ddeg = abs(math.degrees(math.atan2(math.sin(b[2] - self.base0[2]), math.cos(b[2] - self.base0[2]))))
        self.worst['base_dev_m'], self.worst['base_dev_deg'] = max(self.worst['base_dev_m'], dev), max(self.worst['base_dev_deg'], ddeg)
        if dev > r['base_dev_m'] or ddeg > r['base_dev_deg']:
            return self._fail('base_dev', step, t)
        self.hist.append((step, float(t), b[0], b[1], b[2], float(door_q), [float(x) for x in arm6]))
        while len(self.hist) > 2 and self.hist[-1][1] - self.hist[0][1] > r['arm_win_s'] + 0.05:
            self.hist.pop(0)
        vs, v = self._speeds()
        if vs is not None:                                                             # r2 A2：門 0.1 s 保護不等手臂長窗
            self.worst['door_rate'] = max(self.worst['door_rate'], vs[2])
            if vs[2] > r['door_rate_protect']:
                return self._fail('door_rate', step, t)
        if self.state == 'RAMPING' and t - self.t0 >= r['ramp_s'] - 1e-9 and all(abs(float(x) - r['f_lo_n']) <= TOL for x in readback):
            self._go('SETTLING', step, t)
            self.t_quiet0 = None                                                       # 斜坡完成、讀回 f_lo 後才起算重新收斂窗
        quiet = v is not None and v[0] <= r['quiet_base_lin'] and v[1] <= r['quiet_base_ang'] and v[2] <= r['quiet_door'] and v[3] <= r['quiet_arm']
        if self.state in ('SETTLING', 'UNLOADED'):
            if quiet:
                if self.t_quiet0 is None:
                    self.t_quiet0 = float(t)
                if self.state == 'SETTLING' and t - self.t_quiet0 >= r['quiet_s'] - 1e-9:
                    self._go('UNLOADED', step, t, speeds=list(v))
            else:
                self.t_quiet0 = None
                if self.state == 'UNLOADED':
                    self._go('SETTLING', step, t, why='quiet_lost')
        self.last_speeds = v
        return self.state

    def confirmed_fresh(self, step, t=None):
        """r2 B1：步號 ∈ [last_step, last_step＋fresh_steps]；給 t 時另核 last_t ≤ t ≤ last_t＋fresh_steps·dt。"""
        if self.state != 'UNLOADED' or self.last_step is None or type(step) is not int or isinstance(step, bool):
            return False, f'unload_state:{self.state}'
        if not (self.last_step <= step <= self.last_step + self.r['fresh_steps']) or self.last_valid_t != self.last_t:
            return False, 'unload_data_not_fresh'
        if t is not None and (not _fin(t) or t < self.last_t - 1e-9 or t > self.last_t + self.r['fresh_steps'] * self.r['dt'] + 1e-9):
            return False, 'unload_time_not_fresh'
        return True, None

    def handoff(self, step, t):
        if self.state == 'UNLOADED':
            self._go('HANDED_TO_RELEASE', step, t)
            return True
        return False

    def report(self):
        return {'version': VERSION, 'state': self.state, 'why': self.why, 't': dict(self.t), 't0': self.t0, 'step0': self.step0,
                't_settling': self.t.get('SETTLING'), 'last_step': self.last_step, 'last_t': self.last_t,
                'base0': self.base0, 'events': self.events[:50], 'n': dict(self.n), 'worst': dict(self.worst), 'reg': dict(self.r),
                'set_log': self.set_log}

    def state_brief(self):
        """權威狀態訊息用（小）。"""
        return {'state': self.state, 'why': self.why, 't0': self.t0, 't_settling': self.t.get('SETTLING'), 't_unloaded': self.t.get('UNLOADED'),
                'last_step': self.last_step, 'last_t': self.last_t}


def task_gate(state, t_commit1, t_now, solver_outcome=None, run_id=None, state_t=None, solver_t=None, reg=None):
    """任務端：mode 1 提交後決定是否請求 mode 2。r2 B2：先拒非法、逾時、他趟、過期，再判兩端完成。
    state＝執行端權威狀態訊息（含 run_id、unload 欄）；state_t／solver_t＝兩訊息的來源模擬時間；solver_outcome＝求解端 outcome。"""
    r = dict(REG_UNLOAD, **(reg or {}))
    if not (_fin(t_commit1) and _fin(t_now)) or t_now < t_commit1 - 1e-9:
        return 'abort:time_invalid'
    if t_now - t_commit1 > r['unload_max_s'] + r['task_margin_s']:
        return 'abort:unload_deadline'
    so = solver_outcome if isinstance(solver_outcome, dict) else {}
    if so.get('state') == 'FAIL':
        return f"abort:solver_fail:{so.get('fail_why')}"
    if not isinstance(state, dict):
        return 'wait'
    if run_id is not None and state.get('run_id') != run_id:
        return 'wait'                                                         # 他趟訊息不採用（不 go；期限照算）
    u = state.get('unload')
    if isinstance(u, dict) and u.get('state') in ('FAILED', 'ABORTED'):
        return f"abort:unload_failed:{u.get('why')}"
    for ts in (state_t, solver_t):
        if ts is None or not _fin(ts) or ts > t_now + 1e-9 or t_now - ts > r['state_age_max_s'] + 1e-9:
            return 'wait'
    if isinstance(u, dict) and u.get('state') == 'UNLOADED' and so.get('state') == 'HELD' and so.get('sub') == 'unloaded':
        return 'go'
    return 'wait'
