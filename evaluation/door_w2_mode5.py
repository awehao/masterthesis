#!/usr/bin/env python3
"""W2 模式權威 r5＋HELD 交接（純邏輯；K3-B 受控釋放；Codex 20261026_000706 D1；設計 K3B_impl_design_draft1.md §2）。
door_w2_mode*.py 不改；本檔以子類別擴充為多段轉移。

模式序列（每段須上一段提交且套用確認為 True 才可請求下一段）：
  0 PAUSE_HELD →1 RELEASE_PREP_TEST →2 RELEASE_OPEN →3 RETREAT
- mode 1、2 候選＝保持形（同 r4）。
  - mode 2 提交步即 t_open：mode_seq、監看 profile（release_open）、手指目標（開）同一步切換。
  - 手指目標由 finger_target() 依權威模式給出，執行端據此放行張爪；任務端的夾爪命令只作參考。
- mode 3 候選＝真正退離目標求解的首筆：
  - 有限；底盤保持形（≤ hold_v／hold_w）；
  - 手臂 |u| ≤ arm_rate_max；
  - 相對實際套用 u_prev 的加速度框：底盤 a_lin／a_ang×dt；手臂 a_arm×dt，a_arm＝19.984 rad/s²（W-GMPC LITE6_SAFE）、dt＝物理步。
- mode 3 准入（請求時與提交當下都核）：gate3(step) 為真，即執行端 ExecReleaseMonitor.confirmed_fresh(step)。
  - 新鮮的解除確認；不能只因目前在 mode 2 就退離。
- 提交後套用確認：mode 1／2 保持形、mode 3 為「底盤保持形＋手臂速率界內」（_hold_shaped 依模式分派）。
- 單趟只走一次序列；同一請求重送冪等、不同請求衝突（沿用 r2 規則）。

HeldSupervisorP4b5：HELD 中由 mode 3 提交交接 → HANDED_OVER。
- 結束 HELD 的速度、保持形監看，之後回傳 publish＝True、zero＝False（求解端恢復追蹤）。
- held_end＝handover；交接時刻記入轉移。
- 交接不重設任何錨點（錨點由求解端原樣沿用）。
"""
from __future__ import annotations

import math

import door_w2_mode as W2M
import door_w2_mode4 as M4

VERSION = 'door_w2_mode5/r5'
MODE_OF_SEQ5 = {0: 'PAUSE_HELD', 1: 'RELEASE_PREP_TEST', 2: 'RELEASE_OPEN', 3: 'RETREAT'}
PROFILE_OF_SEQ5 = {0: 'pause_hold', 1: 'release_prep_test', 2: 'release_open', 3: 'release_retreat'}
FINGER_OF_SEQ5 = {0: 'closed', 1: 'closed', 2: 'open', 3: 'open'}
A_ARM = 19.984
# 共用對照表註冊新模式：door_w2_mode 的 MODE_OF_SEQ／PROFILE_OF_SEQ 為 r2–r4 共用的同一物件。
# 只新增 2、3 鍵、不改 0／1，舊判定器（door_w2_check_mode C4）對 0／1 的行為不變。
for _k in (2, 3):
    W2M.MODE_OF_SEQ.setdefault(_k, MODE_OF_SEQ5[_k])
    W2M.PROFILE_OF_SEQ.setdefault(_k, PROFILE_OF_SEQ5[_k])
assert all(W2M.MODE_OF_SEQ[k] == MODE_OF_SEQ5[k] and W2M.PROFILE_OF_SEQ[k] == PROFILE_OF_SEQ5[k] for k in range(4))


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class ModeAuthority5(M4.ModeAuthority4):
    def __init__(self, *a, gate3=None, arm_rate_max=1.0, a_arm=A_ARM, max_mode=3, **kw):
        super().__init__(*a, **kw)
        self.gate3 = gate3
        self.arm_rate_max, self.a_arm, self.max_mode = float(arm_rate_max), float(a_arm), int(max_mode)
        self.commits = []

    # ---- 請求
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
                or msg['from_mode'] != MODE_OF_SEQ5[self.mode_seq] or msg['to_mode'] != MODE_OF_SEQ5[ms]:
            return self._nack('bad_transition', msg, step, t)
        if self.commit is not None and self.commit.get('applied_confirmed') is not True:
            return self._nack('previous_commit_not_confirmed', msg, step, t)
        if ms == 3:
            ok, why = self._gate(step)
            if not ok:
                return self._nack(f'retreat_evidence:{why}', msg, step, t)
        self.pending = {'request': dict(msg), 'step': step, 't': t}
        self.events.append({'kind': 'pending', 'step': step, 't': t, 'req_seq': msg['req_seq'], 'mode_seq': ms})
        return 'pending', self.state_msg(step, t)

    def _gate(self, step):
        if self.gate3 is None:
            return False, 'gate_missing'
        try:
            r = self.gate3(step)
        except Exception as e:                                                # noqa: BLE001
            return False, f'gate_error:{e!r}'
        if isinstance(r, tuple):
            return bool(r[0]), r[1]
        return bool(r), None if r else 'not_confirmed_fresh'

    # ---- 候選驗證（依目標模式）
    def validate(self, u, u_prev):
        tgt = self.pending['request']['mode_seq'] if self.pending is not None else self.mode_seq
        if tgt in (1, 2):
            return super().validate(u, u_prev)
        try:
            u = [float(x) for x in u]
            up = [float(x) for x in u_prev] if u_prev is not None else None
        except (TypeError, ValueError):
            return False, 'nonnumeric'
        if len(u) != 9 or not all(math.isfinite(x) for x in u):
            return False, 'shape_or_nonfinite'
        if up is None or len(up) != 9 or not all(math.isfinite(x) for x in up):
            return False, 'u_prev_missing_or_invalid'
        if math.hypot(u[0], u[1]) > self.hold_v + self.tol or abs(u[2]) > self.hold_w + self.tol:
            return False, 'base_not_hold'
        if max(abs(x) for x in u[3:]) > self.arm_rate_max + self.tol:
            return False, 'arm_rate'
        if max(abs(u[0] - up[0]), abs(u[1] - up[1])) > self.a_lin * self.dt + self.tol or abs(u[2] - up[2]) > self.a_ang * self.dt + self.tol:
            return False, 'accel_box'
        if max(abs(a - b) for a, b in zip(u[3:], up[3:])) > self.a_arm * self.dt + self.tol:
            return False, 'arm_accel_box'
        return True, None

    def _hold_shaped(self, u):
        if self.mode_seq != 3:
            return super()._hold_shaped(u)
        if len(u) != 9 or not all(_fin(x) for x in u):
            return False, 'shape_or_nonfinite'
        if math.hypot(u[0], u[1]) > self.hold_v + self.tol or abs(u[2]) > self.hold_w + self.tol:
            return False, 'base_not_hold'
        if max(abs(x) for x in u[3:]) > self.arm_rate_max + self.tol:
            return False, 'arm_rate'
        return True, None

    # ---- 物理步選擇：mode 3 提交當下再核准入
    def select(self, step, t, u_prev_applied):
        if self.aborted is None and self.await_confirm is None and self.pending is not None \
                and self.pending['request']['mode_seq'] == 3 and self.cand is not None:
            ok, why = self._gate(step)
            if not ok:
                req = self.pending['request']
                self.pending, self.cand = None, None
                self._nack(f'retreat_evidence_at_commit:{why}', req, step, t)
        n0 = len(self.events)
        out = super().select(step, t, u_prev_applied)
        if any(e.get('kind') == 'commit' for e in self.events[n0:]):
            self.commit['profile'] = PROFILE_OF_SEQ5[self.mode_seq]
            self.commit['finger_target'] = FINGER_OF_SEQ5[self.mode_seq]
            self.commits.append(self.commit)
        out['profile'] = self.profile()
        out['finger_target'] = self.finger_target()
        return out

    def profile(self):
        return PROFILE_OF_SEQ5[self.mode_seq]

    def finger_target(self):
        return FINGER_OF_SEQ5[self.mode_seq]

    def state_msg(self, step, t):
        m = super().state_msg(step, t)
        m['mode'] = MODE_OF_SEQ5[self.mode_seq]
        m['profile'] = self.profile()
        m['finger_target'] = self.finger_target()
        m['commits'] = [{k: c.get(k) for k in ('step', 't', 'first_source_seq', 'mode_seq', 'applied_confirmed')} for c in self.commits]
        m['commit_req_seqs'] = [c['request']['req_seq'] for c in self.commits]
        return m

    def stats(self):
        s = super().stats()
        s.update(version=VERSION, commits=[{k: v for k, v in c.items() if k != 'request'} for c in self.commits],
                 mode_name=MODE_OF_SEQ5[self.mode_seq])
        return s


class HeldSupervisorP4b5(W2M.HeldSupervisorP4b):
    stop_definition = 'active_hold_arm_win_0.5s_held_handover'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.t_handover = None

    def handover(self, t):
        """mode 3 提交（求解端由權威狀態得知）⇒ 結束 HELD 監看；只在 HELD 中有效。回傳是否交接。"""
        if self.seq.state != 'HELD' or not _fin(t):
            return False
        self.seq.state = 'HANDED_OVER'
        self.seq.log.append((t, 'HANDOVER', None))
        self.t_handover, self.held_end = float(t), 'handover'
        return True

    def cycle(self, t, solve_ok, hold_cmd_ok_now, wall=None):
        if self.seq.state == 'HANDED_OVER':
            return {'publish': True, 'zero': False, 'handed_over': True}
        return super().cycle(t, solve_ok, hold_cmd_ok_now, wall)

    def applied(self, phys_step, t, base3, arm6, src_seq):
        if self.seq.state == 'HANDED_OVER':
            return                                                            # 交接後由 K3-B 監看負責
        return super().applied(phys_step, t, base3, arm6, src_seq)

    def outcome(self, t_end=None):
        o = super().outcome(t_end)
        o['transitions']['HANDOVER'] = self.t_handover
        if self.t_handover is not None:
            o['held_end'] = 'handover'
            o['state'] = 'HANDED_OVER'
            if o.get('transitions', {}).get('HELD') is not None:
                o['held_s'] = self.t_handover - o['transitions']['HELD']
        return o


class SolverModeSource5(W2M.SolverModeSource):
    """求解端模式來源 r5：ADM11 的旁路候選（候選＝一般命令內容）只用於保持形模式 1、2；
    mode 3（退離）的候選由 ADM12 以退離目標另解並另行發布 ⇒ tag_candidate() 對 3 回 None，改由 tag_candidate3() 提供。"""

    def tag_candidate(self):
        m = super().tag_candidate()
        return None if m == 3 else m

    def tag_candidate3(self):
        m = super().tag_candidate()
        return 3 if m == 3 else None
