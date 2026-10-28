#!/usr/bin/env python3
"""換抓＋關門模式權威 r4（door_regrasp_mode1 的版本化複本，只擴模式表到 8 並併入 door_regrasp_mode2 的 gate 2；Codex r67）：
  7 PASS_CLOSE        通過中拉關：夾持內側橫桿、全身形（底盤自由）、手指閉；准入 gate 7＝HoldConfirm 確認且新鮮；
  8 RELEASE_RETREAT   張爪、沿工具 −z 退出、收臂：全身形、手指開；准入 gate 8＝夾持仍有效且新鮮＋門角到拉關終點（執行端核心 r4）。
  接觸規則：7 ('bar_in',)；8 離開前 ('bar_in',)、離開（退出量 ≥ S_EXIT，單向閂鎖）後 ()。其餘同 mode1／mode2。
以下為 mode1 說明：
換抓專用模式權威（純邏輯；Codex r50；規格 REGRASP_inner_contact_prep_draft2.md §3、§7）。door_w2_mode*.py 全部不改、**不改其全域對照表**。

重用 W2／K3-B 的協定與核驗機制（請求、候選、原子提交、逐步套用確認、提交後斷流／未套用有界中止、中止優先）：
  RegraspAuthority 繼承 door_w2_mode8.ModeAuthority8（unload＝False；不引入 20→5 N 卸載），**覆寫** on_request／select／validate／_hold_shaped／
  profile／finger_target／state_msg／stats，改用本檔的對照表（r3 的 select 與基底 state_msg 直接查全域表，mode ≥ 4 會 KeyError，故以本地表重寫；
  select 本體＝r3 核心＋r4 套用逾時＋r5 提交紀錄，逐段搬移，邏輯不變）；confirm_step（r3＋r4）沿用父類，依本檔 _hold_shaped 逐模式核形。
  HELD 監督沿用 door_w2_mode9.HeldSupervisorP4b9（不改）：HELD → mode 2 張爪（release_open 瞬態）→ mode 3 提交時交接（只准自 release_settled）。

模式（每段須上一段提交且套用確認 True 才可請求下一段；單趟只走一次）：
  0 PAUSE_HELD          保持（手臂零、底盤錨點伺服、手指不變＝閉）
  1 RELEASE_PREP_TEST   保持形測試段（沿用 W2；不卸載）
  2 RELEASE_OPEN        保持形；手指目標＝開；提交步＝t_open
  3 RETREAT_WB          **全身受控退開**（r50 §2）：TCP 沿解除時實測工具 −z 0.06 m（方向固定），底盤與手臂同一條已驗路徑參考；手指開
  4 STOW_MOVE           收臂＋底盤移到門洞入口 B1（retime_v3 參考 τ 查表＋起始銜接）；手指開
  5 APPROACH_INNER      內側接近＋插入（目標依當時實測內側把手位姿）；手指開
  6 CLOSE_INNER         准入後閉爪＋短保持確認（短保持是本模式的計時子狀態，不另握手）；手指目標＝閉
候選與套用形狀（validate／_hold_shaped）：
  1、2、6：保持形（底盤 ≤ hold_v／hold_w、手臂 ≤ arm_tol）；
  3、4、5：全身形——底盤車體每軸 ≤ v_lin（求解器 0.035255）、|ω| ≤ v_ang（0.1999）、世界合速度 ≤ v_world（E2 數值 0.05）；手臂 ≤ arm_rate_max（1.0）；
          候選另核相對實際套用 u_prev 的加速度框（底盤 a_lin／a_ang×dt、手臂 a_arm×dt）。
准入 gate（請求時與提交當下都核；呼叫端給 gates={k: fn(step) → bool 或 (bool, why)}；缺 gate ⇒ 拒絕）：3（解除確認新鮮）、4（離開外側橫桿且留餘裕＋退開終點）、
  5（到達銜接範圍＋收臂完成）、6（內側閉爪准入）。gate 的內容由執行端提供（door_regrasp_gates*.py），本類只負責「沒有證據就不提交」。
另提供：DESIGNATED[k]（指定接觸橫桿）、GEOM_TARGET[k]（v3 淨空監看目標橫桿）、PERMIT[k]（底盤／手臂／手指許可），供任務端、判定器共用。
"""
from __future__ import annotations

import math

import door_w2_mode as W2M
import door_w2_mode8 as M8

VERSION = 'door_regrasp_mode3/r4'
MODE_OF = {0: 'PAUSE_HELD', 1: 'RELEASE_PREP_TEST', 2: 'RELEASE_OPEN', 3: 'RETREAT_WB', 4: 'STOW_MOVE', 5: 'APPROACH_INNER', 6: 'CLOSE_INNER',
           7: 'PASS_CLOSE', 8: 'RELEASE_RETREAT'}
PROFILE_OF = {0: 'pause_hold', 1: 'release_prep_test', 2: 'release_open', 3: 'regrasp_retreat_wb', 4: 'regrasp_stow_move',
              5: 'regrasp_approach_inner', 6: 'regrasp_close_inner', 7: 'regrasp_pass_close', 8: 'regrasp_release_retreat'}
FINGER_OF = {0: 'closed', 1: 'closed', 2: 'open', 3: 'open', 4: 'open', 5: 'open', 6: 'closed', 7: 'closed', 8: 'open'}
DESIGNATED = {0: 'bar_out', 1: 'bar_out', 2: 'bar_out', 3: 'bar_out_exiting', 4: None, 5: 'bar_in_preclose_protected', 6: 'bar_in', 7: 'bar_in',
              8: 'bar_in_exiting'}
GEOM_TARGET = {0: 'out', 1: 'out', 2: 'out', 3: 'out', 4: None, 5: 'in_after_pregrasp', 6: 'in', 7: 'in', 8: 'in'}
PERMIT = {0: ('hold', 'zero', 'keep'), 1: ('hold', 'zero', 'keep'), 2: ('hold', 'zero', 'open'), 3: ('move', 'move', 'open'),
          4: ('move', 'move', 'open'), 5: ('move', 'move', 'open'), 6: ('hold', 'zero', 'close'), 7: ('move', 'move', 'close'), 8: ('move', 'move', 'open')}
HOLD_MODES, MOTION_MODES, GATED = (1, 2, 6), (3, 4, 5, 7, 8), (2, 3, 4, 5, 6, 7, 8)
MAX_MODE = 8
A_ARM = 19.984
# 舊全域表不被改動（door_w2_mode5 只登錄 0–3；本檔不登錄）
assert all(k not in W2M.MODE_OF_SEQ for k in (4, 5, 6, 7, 8))


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def allowed_finger_classes(mode, exited=False):
    """手指允許接觸的 v3 類別（Codex r53 §2；執行端保護與任務端監看共用，規則唯一來源）。
    指定持握只認橫桿；立柱、panel、ambiguous 一律依非預期接觸保護（解除確認時保守加總它們的力，不等於允許接觸）。
    mode 0–2：('bar_out',)；mode 3：未離開前 ('bar_out',)，離開（exited，單向閂鎖，由執行端 Gate4 的退開量 ≥ S_EXIT 首次成立時設定）後 ()；
    mode 4、5：()（閉爪前不允許任何接觸）；mode 6：('bar_in',)。非手指部位任何類別都不允許。"""
    if mode in (0, 1, 2):
        return ('bar_out',)
    if mode == 3:
        return () if exited else ('bar_out',)
    if mode in (6, 7):
        return ('bar_in',)
    if mode == 8:
        return () if exited else ('bar_in',)
    return ()


class RegraspAuthority(M8.ModeAuthority8):
    def __init__(self, *a, gates=None, arm_rate_max=1.0, a_arm=A_ARM, v_lin=0.035255, v_ang=0.1999, v_world=0.05, **kw):
        kw.setdefault('unload', False)
        if kw['unload']:
            raise ValueError('換抓權威不啟用卸載（r50 §3）')
        super().__init__(*a, **kw)
        self.gates = dict(gates or {})
        self.arm_rate_max, self.a_arm = float(arm_rate_max), float(a_arm)
        self.v_lin, self.v_ang, self.v_world = float(v_lin), float(v_ang), float(v_world)
        self.max_mode = MAX_MODE
        self.commits = []

    # ---- 准入
    def _gate_k(self, k, step):
        fn = self.gates.get(k)
        if fn is None:
            return False, 'gate_missing'
        try:
            r = fn(step)
        except Exception as e:                                                # noqa: BLE001
            return False, f'gate_error:{e!r}'
        if isinstance(r, tuple):
            return bool(r[0]), r[1]
        return bool(r), None if r else 'not_confirmed_fresh'

    # ---- 請求（r5 規則，本地表、通用 gate）
    def on_request(self, msg, step, t):
        if not isinstance(msg, dict) or not all(k in msg for k in ('run_id', 'req_seq', 'from_mode', 'to_mode', 'mode_seq')):
            return self._nack('malformed', msg, step, t)
        if msg['run_id'] != self.run_id:
            return self._nack('wrong_run', msg, step, t)
        if self.aborted is not None:
            return self._nack('after_abort', msg, step, t)
        for c in self.commits:
            if self._key(msg) == self._key(c['request']):
                return 'repeat', self.state_msg(step, t)
        if self.pending is not None:
            if self._key(msg) == self._key(self.pending['request']):
                return 'repeat', self.state_msg(step, t)
            return self._nack('conflict_pending', msg, step, t)
        ms = msg['mode_seq']
        if type(ms) is not int or ms != self.mode_seq + 1 or ms > self.max_mode \
                or msg['from_mode'] != MODE_OF[self.mode_seq] or msg['to_mode'] != MODE_OF[ms]:
            return self._nack('bad_transition', msg, step, t)
        if self.commit is not None and self.commit.get('applied_confirmed') is not True:
            return self._nack('previous_commit_not_confirmed', msg, step, t)
        if ms in GATED:
            ok, why = self._gate_k(ms, step)
            if not ok:
                return self._nack(f'gate{ms}:{why}', msg, step, t)
        self.pending = {'request': dict(msg), 'step': step, 't': t}
        self.events.append({'kind': 'pending', 'step': step, 't': t, 'req_seq': msg['req_seq'], 'mode_seq': ms})
        return 'pending', self.state_msg(step, t)

    # ---- 形狀
    def _shape(self, u, k):
        if len(u) != 9 or not all(_fin(x) for x in u):
            return False, 'shape_or_nonfinite'
        if k in MOTION_MODES:
            if max(abs(u[0]), abs(u[1])) > self.v_lin + self.tol or abs(u[2]) > self.v_ang + self.tol:
                return False, 'base_box'
            if math.hypot(u[0], u[1]) > self.v_world + self.tol:
                return False, 'base_world_speed'
            if max(abs(x) for x in u[3:]) > self.arm_rate_max + self.tol:
                return False, 'arm_rate'
            return True, None
        if math.hypot(u[0], u[1]) > self.hold_v + self.tol or abs(u[2]) > self.hold_w + self.tol:
            return False, 'base_not_hold'
        if max(abs(x) for x in u[3:]) > self.arm_tol:
            return False, 'arm_not_zero'
        return True, None

    def validate(self, u, u_prev):
        tgt = self.pending['request']['mode_seq'] if self.pending is not None else self.mode_seq
        try:
            u = [float(x) for x in u]
            up = [float(x) for x in u_prev] if u_prev is not None else None
        except (TypeError, ValueError):
            return False, 'nonnumeric'
        ok, why = self._shape(u, tgt)
        if not ok:
            return ok, why
        if up is None or len(up) != 9 or not all(math.isfinite(x) for x in up):
            return False, 'u_prev_missing_or_invalid'
        if max(abs(u[0] - up[0]), abs(u[1] - up[1])) > self.a_lin * self.dt + self.tol or abs(u[2] - up[2]) > self.a_ang * self.dt + self.tol:
            return False, 'accel_box'
        if max(abs(a - b) for a, b in zip(u[3:], up[3:])) > self.a_arm * self.dt + self.tol:
            return False, 'arm_accel_box'
        return True, None

    def _hold_shaped(self, u):
        return self._shape(list(u), self.mode_seq)

    # ---- 物理步選擇（r7／r5 gate 再核 → r4 套用逾時 → r3 核心 → r5 提交紀錄）
    def select(self, step, t, u_prev_applied):
        if self.aborted is None and self.await_confirm is None and self.pending is not None and self.cand is not None:
            k = self.pending['request']['mode_seq']
            if k in GATED:
                ok, why = self._gate_k(k, step)
                if not ok:
                    req = self.pending['request']
                    self.pending, self.cand = None, None
                    self._nack(f'gate{k}_at_commit:{why}', req, step, t)
        if self.aborted is None and self.await_confirm is None and self.commit is not None \
                and self.commit.get('applied_confirmed') is True:                                   # r4
            la = self.last_applied_t
            if la is None or t - la > self.applied_timeout_s + 1e-9:
                self.abort('post_commit_not_applied', step, t)
                return self._out(None, {'kind': 'abort', 'why': 'post_commit_not_applied'})
        n0 = len(self.events)
        out = self._select_core(step, t, u_prev_applied)
        if any(e.get('kind') == 'commit' for e in self.events[n0:]):
            self.commit['profile'] = PROFILE_OF[self.mode_seq]
            self.commit['finger_target'] = FINGER_OF[self.mode_seq]
            self.commits.append(self.commit)
        out['profile'] = self.profile()
        out['finger_target'] = self.finger_target()
        return out

    def _select_core(self, step, t, u_prev_applied):
        """door_w2_mode3.ModeAuthority3.select 逐段搬移；唯一差異：提交紀錄的 profile 查本地表。"""
        if self.aborted is not None:
            return self._out(None, None)
        if self.await_confirm is not None:
            self.abort('confirm_missing', step, t)
            return self._out(None, {'kind': 'abort', 'why': 'confirm_missing'})
        ev = None
        if self.pending is not None:
            req = self.pending['request']
            if t - self.pending['t'] > self.cand_timeout_s + 1e-9:
                self.pending, self.cand = None, None
                self._nack('candidate_timeout', req, step, t)
                ev = {'kind': 'candidate_timeout'}
            elif self.cand is not None:
                ce = self.cand[0]
                if not self._fresh(ce, t):
                    self.cand = None
                    self.n_cand_stale_at_select += 1
                elif ce['source_seq'] <= self.last_fed_seq:
                    self.cand = None
                    self.n_cand_not_newer += 1
                else:
                    ok, why = self.validate(ce.get('u'), u_prev_applied)
                    if ok:
                        c = self.cand
                        self.pending, self.cand = None, None
                        self.mode_seq = req['mode_seq']
                        self.cur = c
                        self.commit = {'step': step, 't': t, 'first_source_seq': ce['source_seq'], 'mode_seq': self.mode_seq,
                                       'profile': PROFILE_OF[self.mode_seq], 'request': req, 'applied_confirmed': None,
                                       'stamp_sim_t': ce['stamp_sim_t'], 't_recv': c[1]}
                        self.events.append({'kind': 'commit', 'step': step, 't': t, 'first_source_seq': ce['source_seq'],
                                            'mode_seq': self.mode_seq})
                        ev = {'kind': 'commit'}
                    else:
                        self.pending, self.cand = None, None
                        self._nack(f'candidate_invalid:{why}', req, step, t)
                        ev = {'kind': 'candidate_rejected', 'why': why}
        if self.commit is not None:
            ce = None if self.cur is None else self.cur[0]
            if ce is None or not _fin(ce.get('stamp_sim_t')) or t - ce['stamp_sim_t'] > self.post_commit_timeout_s + 1e-9:
                self.abort('post_commit_source_lost', step, t)
                return self._out(None, {'kind': 'abort', 'why': 'post_commit_source_lost'})
        apply = None
        if self.cur is not None:
            if self._fresh(self.cur[0], t) and self.cur[0]['source_seq'] >= self.last_fed_seq:
                apply = self.cur[0]
            else:
                self.n_cur_stale_at_select += 1
        if apply is not None:
            self.last_fed_seq = max(self.last_fed_seq, int(apply['source_seq']))
        self.await_confirm = {'step': step, 'sel_src': None if apply is None else int(apply['source_seq']), 'mode_seq': self.mode_seq}
        return self._out(apply, ev)

    def _out(self, apply, ev):
        return {'apply': apply, 'mode_seq': self.mode_seq, 'profile': self.profile(), 'event': ev}

    def profile(self):
        return PROFILE_OF[self.mode_seq]

    def finger_target(self):
        return FINGER_OF[self.mode_seq]

    def state_msg(self, step, t):
        m = {'run_id': self.run_id, 'step': step, 't': t, 'mode_seq': self.mode_seq, 'mode': MODE_OF[self.mode_seq],
             'profile': self.profile(), 'finger_target': self.finger_target(),
             'pending_req_seq': None if self.pending is None else self.pending['request']['req_seq'],
             'commit': None if self.commit is None else {k: self.commit[k] for k in ('step', 't', 'first_source_seq', 'mode_seq', 'applied_confirmed')},
             'commit_req_seq': None if self.commit is None else self.commit['request']['req_seq'],
             'commits': [{k: c.get(k) for k in ('step', 't', 'first_source_seq', 'mode_seq', 'applied_confirmed')} for c in self.commits],
             'commit_req_seqs': [c['request']['req_seq'] for c in self.commits],
             'last_reject': self.last_reject, 'aborted': self.aborted, 'n_stale': self.n_stale, 'n_violations': len(self.violations),
             'version': VERSION}
        ex = getattr(self, 'extra_state', None)
        if ex is not None:
            m.update(ex())
        return m

    def stats(self):
        return {'version': VERSION, 'n_stale': self.n_stale, 'n_dup': self.n_dup, 'n_violations': len(self.violations),
                'n_cand_stale_at_select': self.n_cand_stale_at_select, 'n_cand_not_newer': self.n_cand_not_newer,
                'n_cur_stale_at_select': self.n_cur_stale_at_select, 'n_src_ne_selected': self.n_src_ne_selected,
                'n_post_commit_no_src': self.n_post_commit_no_src, 'last_fed_seq': self.last_fed_seq,
                'applied_timeout_s': self.applied_timeout_s, 'last_applied_t': self.last_applied_t, 'last_applied_step': self.last_applied_step,
                'commits': [{k: v for k, v in c.items() if k != 'request'} for c in self.commits], 'mode_name': MODE_OF[self.mode_seq],
                'aborted': self.aborted, 'violations': self.violations[:50], 'rejections': self.rejections[:50], 'events': self.events[:400]}


class RegraspSolverSource(W2M.SolverModeSource):
    """求解端模式來源：HELD 中受理 1–3（同 W2）；mode 3 提交（交接）後受理 4–6（committed＋1）。
    tag_candidate()：保持形候選（1、2、6）＝一般命令內容的旁路；tag_candidate_motion()：運動模式候選（3、4、5）由求解端以新目標另解。"""

    def on_request(self, msg, held, t):
        ok_state = held if (isinstance(msg, dict) and type(msg.get('mode_seq')) is int and msg['mode_seq'] <= 3) else self.committed >= 3   # 4–8
        return super().on_request(msg, ok_state, t)

    def tag_candidate(self):
        m = super().tag_candidate()
        return m if m in HOLD_MODES else None

    def tag_candidate_motion(self):
        m = super().tag_candidate()
        return m if m in MOTION_MODES else None
