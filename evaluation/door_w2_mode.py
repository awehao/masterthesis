#!/usr/bin/env python3
"""W2 保持／待交接：HELD 監督器、執行端模式權威、求解端模式來源（純邏輯；Codex 20261025_201543；設計 W2_hold_handover_design_draft2.md §1–§2、§8）。

**本包範圍**：同一 W-GMPC 命令流內的模式轉換機制。只實作一個**測試模式** RELEASE_PREP_TEST。
- 它的命令與 HELD 完全相同：手臂零、底盤錨點伺服、手指不變。
- **不驅動真實張爪或退離**；張爪、退離與 K3-B 判準留到受控釋放套件。

一、HeldSupervisorP4b（求解端；ActiveHoldSupervisorP4b 的子類，門檻不變）
- 觀察滿 observe_s 時，原本進入 DONE、停止發布；本類改為記 OBS_PASSED 並進入 **HELD**。
- HELD 期間持續回傳 publish／zero，並持續監看：
  - 窗口速度不低 ⇒ FAIL moved_in_held；
  - 量測失效 ⇒ FAIL；求解失敗 ⇒ FAIL solve_failed_in_held；
  - 套用命令非保持形或來源早於零速段 ⇒ FAIL old_command_in_held。
- HELD 只由 (a) 執行端模式提交後的下一段接手、(b) 中止／失效、(c) **有界超時 held_max_s** 結束。
  - (c) 記 HELD_TIMEOUT_END，再進 DONE（停止發布）。這不算任務完成，也不代表撤除保持後安全停止。

二、ModeAuthority（執行端權威；純類別，假世界與之後的 ev6 共用）
- mode_seq 0＝暫停／HELD 命令流；1＝RELEASE_PREP_TEST。
- on_request：
  - 本趟 run_id、from_mode PAUSE_HELD、to_mode RELEASE_PREP_TEST、mode_seq＝目前＋1 ⇒ pending；
  - 同一請求重送 ⇒ repeat（冪等）；
  - 不同請求 ⇒ nack（conflict_pending／conflict_after_commit）；
  - 已中止 ⇒ nack after_abort。
- on_env（每筆進入執行端的封裝；mode_seq 與 source_seq 同封裝，不以「最新模式」猜歸屬）：
  - mode_seq＝目前 ⇒ apply；
  - mode_seq < 目前 ⇒ stale（不套用、計數）；
  - mode_seq＝目前＋1 且有 pending ⇒ 驗證候選首筆：有限、保持形（底盤 ≤ hold_v／hold_w、手臂 ≤ arm_tol）、相對實際 applied u_prev 的加速度框。
    - 通過 ⇒ **原子提交**：mode_seq、監看 profile 與套用來源在同一處同時切換，記 commit（step、t、first_source_seq）；
    - 不通過 ⇒ 拒絕候選、保持目前模式。
  - 無 pending 的超前序號 ⇒ violation。
- on_step：
  - 切換前候選逾時 ⇒ 拒絕、繼續保持；
  - 切換後目前模式命令斷流超過 post_commit_timeout ⇒ **中止（不回切）**。
- abort(why)：中止優先，pending 一律拒絕；之後所有命令視為 stale。
- state_msg()：每步發布的權威狀態。求解端與任務端以此為準，**不以 ack 是否送達推定**。

三、SolverModeSource（求解端）
- 只在 HELD 時接受請求，開始以新 mode_seq 標記發布的命令（準備首筆）。
- 看到權威狀態已提交 ⇒ 新序號成為目前序號；看到本請求被拒絕 ⇒ 撤回、回到目前序號。
- ack 可能遺失，一律以 state_msg 為準。
"""
from __future__ import annotations

import math

import door_k3m_pause_sup as PSUP
import door_k3m_pause_sup_p4 as P4
import door_k3m_pause_sup_p4b as P4B

MODE_HELD, MODE_RELEASE_TEST = 'PAUSE_HELD', 'RELEASE_PREP_TEST'
MODE_OF_SEQ = {0: MODE_HELD, 1: MODE_RELEASE_TEST}
PROFILE_OF_SEQ = {0: 'pause_hold', 1: 'release_prep_test'}    # 本包兩者監看內容相同（測試模式不張爪）；分開標示以核對切換步


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ---------------------------------------------------------------- 一、HELD 監督器
class HeldSupervisorP4b(P4B.ActiveHoldSupervisorP4b):
    stop_definition = 'active_hold_arm_win_0.5s_held'

    def __init__(self, *a, held_max_s=60.0, **kw):
        super().__init__(*a, **kw)
        self.held_max_s = float(held_max_s)
        self.t_held = None
        self.held_end = None

    def cycle(self, t, solve_ok, hold_cmd_ok_now, wall=None):
        s = self.seq
        if s.state == 'HELD':
            if not PSUP._fin(t):
                self._fail('time_invalid', t)
                return {'publish': False, 'zero': False}
            if not solve_ok:
                self._fail('solve_failed_in_held', t)
                return {'publish': False, 'zero': False}
            if t - self.t_held > self.held_max_s + 1e-9:
                s.state = 'DONE'
                s.log.append((t, 'HELD_TIMEOUT_END', None))
                self.held_end = 'timeout'
                return {'publish': False, 'zero': False}
            v = self.ws.speeds(t, wall)
            self.last_speeds = v
            if v is None:
                self._fail('measurement_invalid' if self.ws.fresh(t, wall) else 'measurement_stale', t)
                return {'publish': False, 'zero': False}
            if not PSUP.WindowSpeed.low(v, self.th):
                self._fail('moved_in_held', t)
                return {'publish': False, 'zero': False}
            return {'publish': True, 'zero': True}
        was_stopped = s.state == 'STOPPED'
        r = super().cycle(t, solve_ok, hold_cmd_ok_now, wall)
        if was_stopped and s.state == 'DONE':                      # 觀察通過：改進 HELD，不停發
            s.log[-1] = (s.log[-1][0], 'OBS_PASSED', None)
            s.state = 'HELD'
            s.log.append((t, 'HELD', None))
            self.t_held = t
            return {'publish': True, 'zero': True}
        return r

    def applied(self, phys_step, t, base3, arm6, src_seq):
        s = self.seq
        if s.state == 'HELD':
            try:
                vals = [float(v) for v in list(base3) + list(arm6)]
                ok = len(vals) == 9 and all(math.isfinite(v) for v in vals) and PSUP._fin(t)
            except (TypeError, ValueError):
                ok = False
            if not ok:
                return self._fail('applied_invalid', t if PSUP._fin(t) else float('nan'))
            src_old = src_seq is None or src_seq < 0 or (s.zero_seq0 is None or src_seq < s.zero_seq0)
            if src_old or not P4.hold_cmd_ok(base3, arm6, self.c.applied_tol):
                return self._fail('old_command_in_held', t)
            return
        return super().applied(phys_step, t, base3, arm6, src_seq)

    def outcome(self, t_end=None):
        s = self.seq
        names = ('ZERO_PUB', 'CONFIRM', 'STOPPED', 'OBS_PASSED', 'HELD', 'HELD_TIMEOUT_END', 'DONE', 'FAIL')
        tr = {n: next((x[0] for x in s.log if x[1] == n), None) for n in names}
        cmd_ok = tr['STOPPED'] is not None
        obs_ok = tr['OBS_PASSED'] is not None
        covered = None if tr['STOPPED'] is None else ((tr['OBS_PASSED'] if obs_ok else (t_end if t_end is not None else tr['STOPPED']))
                                                       - tr['STOPPED'])
        held_s = None if tr['HELD'] is None else ((tr['HELD_TIMEOUT_END'] or tr['FAIL'] or t_end or tr['HELD']) - tr['HELD'])
        return {'state': s.state, 'fail_why': s.why, 'transitions': tr, 'first_low_t': self.t_first_low,
                'command_stop_confirmed': cmd_ok, 'physical_stop_confirmed': obs_ok, 'observation_passed': obs_ok,
                'observation_required_s': self.c.observe_s, 'observation_covered_s': covered,
                'held_s': held_s, 'held_end': self.held_end, 'held_max_s': self.held_max_s,
                'thresholds': vars(self.th), 'n_rejected_samples': self.ws.n_bad,
                'stop_definition': self.stop_definition, 'arm_window_s': getattr(self, 'arm_win_s', None)}


# ---------------------------------------------------------------- 二、執行端模式權威
class ModeAuthority:
    def __init__(self, run_id, cand_timeout_s=1.0, post_commit_timeout_s=0.25, hold_v=0.005, hold_w=0.02, arm_tol=1e-6,
                 a_lin=0.5, a_ang=2.0, a_arm=None, dt=0.05, tol=1e-9):
        self.run_id = str(run_id)
        self.cand_timeout_s, self.post_commit_timeout_s = float(cand_timeout_s), float(post_commit_timeout_s)
        self.hold_v, self.hold_w, self.arm_tol = float(hold_v), float(hold_w), float(arm_tol)
        self.a_lin, self.a_ang, self.a_arm, self.dt, self.tol = float(a_lin), float(a_ang), a_arm, float(dt), float(tol)
        self.mode_seq = 0
        self.pending = None
        self.commit = None
        self.aborted = None
        self.last_reject = None
        self.last_cur_t = None
        self.n_stale = 0
        self.events = []
        self.rejections = []

    # 請求
    def _nack(self, why, msg, step, t):
        n = {'run_id': self.run_id, 'req_seq': (msg or {}).get('req_seq') if isinstance(msg, dict) else None, 'why': why,
             'step': step, 't': t}
        self.rejections.append(n)
        self.last_reject = n
        self.events.append(dict(n, kind='nack'))
        return 'reject', n

    @staticmethod
    def _key(m):
        return (m.get('run_id'), m.get('req_seq'), m.get('from_mode'), m.get('to_mode'), m.get('mode_seq'))

    def on_request(self, msg, step, t):
        if not isinstance(msg, dict) or not all(k in msg for k in ('run_id', 'req_seq', 'from_mode', 'to_mode', 'mode_seq')):
            return self._nack('malformed', msg, step, t)
        if msg['run_id'] != self.run_id:
            return self._nack('wrong_run', msg, step, t)
        if self.aborted is not None:
            return self._nack('after_abort', msg, step, t)
        if self.commit is not None:
            if self._key(msg) == self._key(self.commit['request']):
                return 'repeat', self.state_msg(step, t)
            return self._nack('conflict_after_commit', msg, step, t)
        if self.pending is not None:
            if self._key(msg) == self._key(self.pending['request']):
                return 'repeat', self.state_msg(step, t)
            return self._nack('conflict_pending', msg, step, t)
        if msg['from_mode'] != MODE_OF_SEQ[self.mode_seq] or msg['to_mode'] != MODE_RELEASE_TEST \
                or type(msg['mode_seq']) is not int or msg['mode_seq'] != self.mode_seq + 1:
            return self._nack('bad_transition', msg, step, t)
        self.pending = {'request': dict(msg), 'step': step, 't': t}
        self.events.append({'kind': 'pending', 'step': step, 't': t, 'req_seq': msg['req_seq'], 'mode_seq': msg['mode_seq']})
        return 'pending', self.state_msg(step, t)

    # 候選驗證
    def validate(self, u, u_prev):
        try:
            u = [float(x) for x in u]
        except (TypeError, ValueError):
            return False, 'nonnumeric'
        if len(u) != 9 or not all(math.isfinite(x) for x in u):
            return False, 'shape_or_nonfinite'
        if math.hypot(u[0], u[1]) > self.hold_v + self.tol or abs(u[2]) > self.hold_w + self.tol:
            return False, 'base_not_hold'
        if max(abs(x) for x in u[3:]) > self.arm_tol:
            return False, 'arm_not_zero'
        if u_prev is not None:
            up = [float(x) for x in u_prev]
            if len(up) != 9 or not all(math.isfinite(x) for x in up):
                return False, 'u_prev_invalid'
            if max(abs(u[0] - up[0]), abs(u[1] - up[1])) > self.a_lin * self.dt + self.tol or abs(u[2] - up[2]) > self.a_ang * self.dt + self.tol:
                return False, 'accel_box'
        return True, None

    def on_env(self, env, step, t, u_prev_applied=None):
        """回傳 (動作, 資訊)：apply／commit 才可進鏈；stale／candidate_rejected／violation 一律不進鏈。"""
        ms = env.get('mode_seq')
        if type(ms) is not int:
            return 'violation', 'mode_seq_missing'
        if self.aborted is not None:
            self.n_stale += 1
            return 'stale', 'after_abort'
        if ms == self.mode_seq:
            self.last_cur_t = t
            return 'apply', None
        if ms < self.mode_seq:
            self.n_stale += 1
            return 'stale', 'old_mode_seq'
        if ms == self.mode_seq + 1 and self.pending is not None and self.pending['request']['mode_seq'] == ms:
            ok, why = self.validate(env.get('u'), u_prev_applied)
            if not ok:
                req = self.pending['request']
                self.pending = None
                self._nack(f'candidate_invalid:{why}', req, step, t)
                return 'candidate_rejected', why
            req = self.pending['request']
            self.pending = None
            self.mode_seq = ms                                                  # 原子提交：模式序號、監看 profile、套用來源同處切換
            self.commit = {'step': step, 't': t, 'first_source_seq': env.get('source_seq'), 'mode_seq': ms,
                           'profile': PROFILE_OF_SEQ[ms], 'request': req}
            self.last_cur_t = t
            self.events.append({'kind': 'commit', 'step': step, 't': t, 'first_source_seq': env.get('source_seq'), 'mode_seq': ms})
            return 'commit', self.commit
        return 'violation', 'mode_seq_ahead'

    def on_step(self, step, t):
        """每物理步呼叫；回傳事件或 None。"""
        if self.aborted is not None:
            return None
        if self.pending is not None and t - self.pending['t'] > self.cand_timeout_s + 1e-9:
            req = self.pending['request']
            self.pending = None
            self._nack('candidate_timeout', req, step, t)
            return {'kind': 'candidate_timeout'}
        if self.commit is not None and self.last_cur_t is not None and t - self.last_cur_t > self.post_commit_timeout_s + 1e-9:
            self.abort('post_commit_source_lost', step, t)
            return {'kind': 'abort', 'why': 'post_commit_source_lost'}
        return None

    def abort(self, why, step, t):
        if self.aborted is not None:
            return
        self.aborted = {'why': why, 'step': step, 't': t}
        if self.pending is not None:
            req = self.pending['request']
            self.pending = None
            self._nack('after_abort', req, step, t)
        self.events.append({'kind': 'abort', 'why': why, 'step': step, 't': t})

    def profile(self):
        return PROFILE_OF_SEQ[self.mode_seq]

    def state_msg(self, step, t):
        return {'run_id': self.run_id, 'step': step, 't': t, 'mode_seq': self.mode_seq, 'mode': MODE_OF_SEQ[self.mode_seq],
                'profile': self.profile(), 'pending_req_seq': None if self.pending is None else self.pending['request']['req_seq'],
                'commit': None if self.commit is None else {k: self.commit[k] for k in ('step', 't', 'first_source_seq', 'mode_seq')},
                'commit_req_seq': None if self.commit is None else self.commit['request']['req_seq'],
                'last_reject': self.last_reject, 'aborted': self.aborted, 'n_stale': self.n_stale}


# ---------------------------------------------------------------- 三、求解端模式來源
class SolverModeSource:
    def __init__(self, run_id):
        self.run_id = str(run_id)
        self.committed = 0
        self.prep = None
        self.events = []

    def on_request(self, msg, held, t):
        if not isinstance(msg, dict) or msg.get('run_id') != self.run_id:
            self.events.append({'kind': 'ignored_bad_request', 't': t})
            return False
        if not held:
            self.events.append({'kind': 'ignored_not_held', 't': t, 'req_seq': msg.get('req_seq')})
            return False
        if self.prep is not None or msg.get('mode_seq') != self.committed + 1:
            self.events.append({'kind': 'ignored_dup_or_bad_seq', 't': t, 'req_seq': msg.get('req_seq')})
            return False
        self.prep = dict(msg)
        self.events.append({'kind': 'prepare', 't': t, 'req_seq': msg['req_seq'], 'mode_seq': msg['mode_seq']})
        return True

    def on_state(self, st, t):
        """權威狀態為準（ack 可能遺失）。"""
        if not isinstance(st, dict) or st.get('run_id') != self.run_id:
            return
        ms = st.get('mode_seq')
        if type(ms) is int and ms > self.committed:
            self.committed = ms
            if self.prep is not None and self.prep.get('mode_seq') == ms:
                self.events.append({'kind': 'committed_seen', 't': t, 'mode_seq': ms})
                self.prep = None
        rj = st.get('last_reject')
        if self.prep is not None and isinstance(rj, dict) and rj.get('req_seq') == self.prep.get('req_seq'):
            self.events.append({'kind': 'reverted', 't': t, 'why': rj.get('why')})
            self.prep = None
        if st.get('aborted') and self.prep is not None:
            self.events.append({'kind': 'reverted', 't': t, 'why': 'aborted'})
            self.prep = None

    def tag(self):
        return self.prep['mode_seq'] if self.prep is not None else self.committed
