#!/usr/bin/env python3
"""換抓執行端核心（純邏輯；Codex r63／r64）。isaac_door_k2_sim_ev9r8 --regrasp 與假世界聯測的假執行端**共用本類別**，聯測驗的就是實跑那份程式。

職責（每物理步一次，資料＝同一份 v3 接觸快照的分類紀錄 rec、同步讀回的 TCP／底盤／手臂／手指／門角）：
  1. gate 與權威准入：RegraspAuthority 的 gates＝{3: Gate3Release, 4: Gate4Exit, 5: Gate5Arrive, 6: Gate6Admit}.confirmed_fresh；
     HoldConfirm＝mode 6 提交後的計時子狀態。各 gate 自「前一模式提交」起逐步更新，**到它負責准入的模式提交為止**（之後結束、保存判定；
     例：Gate3Release 確認後持續要求張爪小力，若延續到 mode 6 閉爪必誤判失敗）：
       mode 2 提交 ⇒ Gate3Release.open；mode 3 提交 ⇒ Gate3 結束、Gate4Exit.start(T_tcp0＝當步實測)；
       mode 4 提交 ⇒ Gate4 結束、Gate5Arrive.start(B1, STOW＝參考包登錄值)；mode 5 提交 ⇒ Gate5 結束、Gate6Admit.start；
       mode 6 提交 ⇒ Gate6 結束、HoldConfirm.start。Gate6／HoldConfirm 的名目抓取＝H_in(q_meas)·參考包登錄 inner_grasp.T_HG（建構時 registry.validate）。
  2. 接觸保護：換抓啟用後（mode ≥ 1 提交起）每步 ContactWindow.step(mode＝本步權威選定模式, exited＝Gate4 閂鎖)；第一筆違規 ⇒ 中止原因。
  3. 失效處置：mode ≥ 1 期間 rec 不綁定當步（缺、query_failed、不完整、invalid）連續 > invalid_max_s 0.2 s ⇒ 中止 rg_contact_data_invalid
     （不當作無接觸）；進行中 gate／HoldConfirm 轉 FAILED ⇒ 中止 rg_<gate>:<why>。中止原因由呼叫端走既有 K2 閂鎖（不另設停止路徑）。
  4. 保護交接：mode 3 提交步 k 起 K3-M 走廊／包絡監看由呼叫端結束——最後一次舊判定＝k−1（handover.last_old_step），本類保護自 k 起
     （first_new_step；ContactWindow 自 mode 1 提交起已逐步在跑、Gate4 自 k 起）。r2（Codex r65 §3）：提交當下要求新保護有有效證據——
     Gate3 最後有效紀錄步 ∈ {k−1, k}（同 confirmed_fresh 的同一份紀錄）且接觸資料無進行中的失效；不符 ⇒ 中止 rg_handover_no_valid_evidence。
  5. 對任務端：gates_state(step)（權威狀態 extra_state 的 'gates'）與 ContactWindow.take_summary()（每週期摘要）。
  6. r3（Codex r66 §3）停止准入 gate 2（張爪前，逐物理步、有效證據）：Gate2StopObs——判定器 door_k3m_check_p4b 的 P2／P3 同一套程式
     （window_speeds、low、hold_cmd_ok、TH／WIN／N_CONFIRM／OBS_S）逐步重算：窗起點＝求解端宣告的 ZERO_PUB（/wgmpc/pause_status，set_zero），
     命令停止時刻 t_cmd＝自 ZERO_PUB 起套用命令連續 N_CONFIRM 步保持形；確認＝t ≥ t_cmd＋OBS_S＋MARGIN（0.2 s，使線上窗涵蓋判定器窗：
     判定器 t_cmd 另要求來源 ≥ 零速段首序號，至多晚一條命令管線延遲）；窗內任一步速度超標 ⇒ FAILED stop_obs_moved、缺步或非有限 ⇒
     FAILED stop_obs_data_invalid（不跳過、不補）；確認後至 mode 2 提交仍逐步核（失去即 FAILED）。ZERO_PUB 未宣告 ⇒ 不確認。
     權威改 door_regrasp_mode2.RegraspAuthority2（gates[2]＝本 gate）；mode 2 提交即結束（保存判定）。gate 2 FAILED 不觸發中止
     （開門前綴暫停期行為不變）：只使 mode 2 請求／提交被拒（gate2:<why>），由任務端依既有規則中止換抓。
另：bench_release_w2——台架首筆在 W2 唯一入口（v3 封裝）下經 W2Exec.on_env3→pre_step 進鏈（同 BENCH.release_intake 的檢核）。
"""
from __future__ import annotations

import numpy as np

import collections

import door_k3m_check_p as CP
import door_k3m_pause_sup_p4 as P4
import door_regrasp_contact_window as RCW
import door_regrasp_gates3 as GT
import door_regrasp_ref_registry as REGY

VERSION = 'door_regrasp_exec_core/r3'
GATE_KEY = {2: 'gate2', 3: 'gate3', 4: 'gate4', 5: 'gate5', 6: 'gate6'}
MARGIN_S = 0.2


class Gate2StopObs:
    """張爪前停止准入（判定器 P2／P3 同一套程式逐物理步重算）。狀態 WAIT → CONFIRMED／FAILED。"""
    name = 'gate2_stop_obs'

    def __init__(self, dt=0.01):
        self.dt = float(dt)
        self.buf = collections.deque(maxlen=CP.WIN + 1)
        self.I = {'step': 0, 'sim_t': 1, 'base_xyth': 2, 'q_arm_meas': 3, 'door_q_rad': 4}
        self.t_zero = self.t_cmd = None
        self.run = 0
        self.prev_step = None
        self.state, self.why = 'WAIT', None
        self.t, self.events = {}, []
        self.last_valid_step = None
        self.n = {'steps': 0, 'checked': 0, 'max_v': [0.0, 0.0, 0.0, 0.0]}

    def set_zero(self, t_zero):
        if self.t_zero is None and isinstance(t_zero, (int, float)) and not isinstance(t_zero, bool) and np.isfinite(t_zero):
            self.t_zero = float(t_zero)
            self.events.append({'t_zero': self.t_zero})

    def _fail(self, why, step, t, **kw):
        if self.state != 'FAILED':
            self.state, self.why = 'FAILED', why
            self.t['failed'] = t
            self.events.append(dict({'step': step, 't': t, 'to': 'FAILED', 'why': why}, **kw))
        return self.state

    def step(self, step, t, base3, door_q, arm6, base_cmd3, arm_cmd6):
        if self.state == 'FAILED':
            return self.state
        self.n['steps'] += 1
        self.buf.append([step, float(t), list(base3), list(arm6), float(door_q)])
        if self.t_zero is None or float(t) < self.t_zero - 1e-9:
            return self.state
        v = CP.window_speeds(list(self.buf), self.I, len(self.buf) - 1)
        self.n['checked'] += 1
        if v is None:
            return self._fail('stop_obs_data_invalid', step, t)
        self.n['max_v'] = [max(a, b) for a, b in zip(self.n['max_v'], v)]
        if not CP.low(v):
            return self._fail('stop_obs_moved', step, t, v=[float(x) for x in v])
        self.last_valid_step = step
        if self.t_cmd is None:
            ok = P4.hold_cmd_ok(base_cmd3, arm_cmd6, CP.TOL_APPLIED)
            consec = self.prev_step is not None and step == self.prev_step + 1
            self.run = (self.run + 1 if consec else 1) if ok else 0
            if self.run >= CP.N_CONFIRM:
                self.t_cmd = float(t)
                self.events.append({'step': step, 't_cmd': self.t_cmd})
        self.prev_step = step
        if self.state == 'WAIT' and self.t_cmd is not None and float(t) >= self.t_cmd + CP.OBS_S + MARGIN_S - 1e-9:
            self.state = 'CONFIRMED'
            self.t['CONFIRMED'] = float(t)
            self.events.append({'step': step, 't': t, 'to': 'CONFIRMED'})
        return self.state

    def confirmed_fresh(self, step):
        return self.state == 'CONFIRMED' and self.last_valid_step in (step - 1, step)

    def report(self):
        return {'gate': self.name, 'state': self.state, 'why': self.why, 't_zero': self.t_zero, 't_cmd': self.t_cmd, 't': dict(self.t),
                'events': self.events[:50], 'n': dict(self.n), 'reg': {'TH': dict(CP.TH), 'WIN': CP.WIN, 'N_CONFIRM': CP.N_CONFIRM,
                                                                       'OBS_S': CP.OBS_S, 'MARGIN_S': MARGIN_S, 'HOLD_V': P4.HOLD_V, 'HOLD_W': P4.HOLD_W}}


class RegraspExecCore:
    def __init__(self, auth, pkg, H_in_fn, dt=0.01, invalid_max_s=0.2, reg=None):
        ok, why = REGY.validate(pkg)
        if not ok:
            raise ValueError(f'參考包登錄核驗不過：{why}')
        self.auth, self.H_in_fn, self.dt, self.invalid_max_s = auth, H_in_fn, float(dt), float(invalid_max_s)
        self.T_HG = np.asarray(pkg['inner_grasp']['T_HG'], float).reshape(4, 4)
        self.B1, self.STOW = np.asarray(pkg['B1'], float), np.asarray(pkg['stow'], float)
        reg = dict(reg or {}, dt=self.dt)
        self.g = {2: Gate2StopObs(dt=self.dt), 3: GT.Gate3Release(reg=reg), 4: GT.Gate4Exit(reg=reg), 5: GT.Gate5Arrive(reg=reg), 6: GT.Gate6Admit(reg=reg)}
        self.hold = GT.HoldConfirm(reg=reg)
        self.active = {2}                         # 正在逐步更新的 gate（2–6）與 'hold'；gate 2 自建構起緩衝、至 mode 2 提交
        self.ended = {}                           # gate → {'step','t','state'}
        self.cw = RCW.ContactWindow(f_max=GT.REG['contact_abort_n'], dt=self.dt)
        self.engaged = False                      # mode ≥ 1 提交後
        self.inv_t0 = None
        self.abort = None
        self.commits = []
        self.handover = None
        auth.gates = {k: (lambda s, _k=k: self._fresh(_k, s)) for k in (2, 3, 4, 5, 6)}

    def _fresh(self, k, step):
        g = self.g[k]
        if k not in self.active:
            return False, f'{GATE_KEY[k]}_inactive'
        return (True, None) if g.confirmed_fresh(step) else (False, f'{GATE_KEY[k]}_{g.state}')

    def _end(self, key, step, t):
        if key in self.active:
            self.active.discard(key)
            obj = self.hold if key == 'hold' else self.g[key]
            self.ended[key] = {'step': step, 't': t, 'state': obj.state}

    def on_commit(self, step, t, mode, T_tcp):
        """權威提交事件（W2Exec select 的 event kind＝commit）當步、world.step 之前呼叫。"""
        self.commits.append({'step': step, 't': t, 'mode': mode})
        if mode >= 1:
            self.engaged = True
        if mode == 2:
            self._end(2, step, t)
            self.g[3].open(step, t)
            self.active.add(3)
        elif mode in (3, 4, 5, 6):
            self._end(mode, step, t)
            if mode == 3:
                ev = self.g[3].last_valid_step
                self.handover = {'step': step, 't': t, 'last_old_step': step - 1, 'first_new_step': step, 'evidence_step': ev,
                                 'what': 'K3-M 走廊／包絡監看最後判定於 k−1；換抓保護（ContactWindow、Gate4、權威全身形）自 k 起'}
                if ev not in (step - 1, step) or self.inv_t0 is not None:
                    self._abort('rg_handover_no_valid_evidence', step, t)
                self.g[4].start(step, t, T_tcp0=np.asarray(T_tcp, float))
                self.active.add(4)
            elif mode == 4:
                self.g[5].start(step, t, base_goal=self.B1, arm_stow=self.STOW)
                self.active.add(5)
            elif mode == 5:
                self.g[6].start(step, t)
                self.active.add(6)
            else:
                self.hold.start(step, t)
                self.active.add('hold')

    def step(self, step, t, mode_sel, rec, T_tcp, base, arm_q, finger_q, door_q, base_cmd=None, arm_cmd=None):
        """回傳本步中止原因（None＝無）。第一個中止原因閂鎖。"""
        if self.abort is not None:
            return self.abort
        lcf = GT.link_class_force(rec, step, t)
        valid = lcf is not None
        if self.engaged:
            exited = bool(getattr(self.g[4], 'exited', False))
            self.cw.step(step, t, int(mode_sel) if mode_sel is not None else -1, rec, exited)
            v = self.cw.violation
            if v is not None:
                return self._abort(f"rg_contact:{v['link']}/{v['cls']}/{v['n']:.3f}N@mode{v['mode']}", step, t)
            if valid:
                self.inv_t0 = None
            else:
                self.inv_t0 = self.inv_t0 if self.inv_t0 is not None else float(t)
                if float(t) - self.inv_t0 > self.invalid_max_s + 1e-9:
                    return self._abort('rg_contact_data_invalid', step, t)
        H_in = self.H_in_fn(float(door_q))
        if 2 in self.active:
            self.g[2].step(step, t, base, door_q, arm_q, base_cmd if base_cmd is not None else [float('nan')] * 3,
                           arm_cmd if arm_cmd is not None else [float('nan')] * 6)
        if 3 in self.active:
            pf = None if not valid else [GT.finger_force(lcf, f, GT.OUTER) for f in GT.FINGERS]
            self.g[3].step(step, t, finger_q, pf, valid, door_q)
        if 4 in self.active:
            self.g[4].step(step, t, T_tcp, rec)
        if 5 in self.active:
            self.g[5].step(step, t, base, arm_q, rec)
        if 6 in self.active:
            self.g[6].step(step, t, T_tcp, H_in @ self.T_HG, rec)
        if 'hold' in self.active:
            self.hold.step(step, t, T_tcp, H_in, rec)
        for k in sorted(x for x in self.active if x not in ('hold', 2)):              # gate 2 失敗不在此中止：只使 mode 2 永不准入（權威 nack）
            if self.g[k].state == 'FAILED':
                return self._abort(f'rg_{GATE_KEY[k]}:{self.g[k].why}', step, t)
        if 'hold' in self.active and self.hold.state == 'FAILED':
            return self._abort(f'rg_hold:{self.hold.why}', step, t)
        return None

    def _abort(self, why, step, t):
        self.abort = why
        self.abort_at = {'step': step, 't': t, 'why': why}
        return why

    def hold_confirmed(self, step):
        return 'hold' in self.active and self.hold.confirmed_fresh(step)

    def gates_state(self, step):
        out = {}
        for k, key in GATE_KEY.items():
            g = self.g[k]
            st = g.state if (k in self.active or k in self.ended) else 'IDLE'
            if st == 'B_CONFIRMED':
                st = 'CONFIRMED'
            out[key] = {'state': st, 'fresh': bool(k in self.active and g.confirmed_fresh(step)), 'why': g.why, 'active': k in self.active}
        h = self.hold
        out['hold'] = {'state': h.state if ('hold' in self.active or 'hold' in self.ended) else 'IDLE',
                       'fresh': bool('hold' in self.active and h.confirmed_fresh(step)), 'why': h.why, 'active': 'hold' in self.active,
                       'E0_set': getattr(h, 'E0', None) is not None}
        return out

    def summary(self):
        return self.cw.take_summary()

    def report(self):
        return {'version': VERSION, 'abort': getattr(self, 'abort_at', None), 'commits': self.commits, 'handover': self.handover,
                'ended': self.ended, 'gates': {GATE_KEY[k]: self.g[k].report() for k in self.g}, 'hold': self.hold.report(),
                'contact_window': {'n': dict(self.cw.n), 'violation': self.cw.violation}}


def bench_release_w2(w2x, chain, env3, announced_seq, step0, t0, aborted):
    """台架首筆（W2 唯一入口）：只把宣告序號的 v3 封裝經 W2Exec 進鏈。回傳 (ok, why, record, sel)。檢核同 door_bench_start2.release_intake。"""
    if aborted:
        return False, 'aborted_same_iteration', None, None
    if env3 is None:
        return False, 'no_env', None, None
    if announced_seq is None or env3.get('source_seq') != announced_seq:
        return False, 'seq_not_announced', None, None
    if env3.get('mode_seq') != 0:
        return False, 'mode_seq_not_0', None, None
    rej0 = int(getattr(chain, 'n_rejected', 0))
    act, _ = w2x.on_env3(env3, float(t0))
    sel = w2x.pre_step(step0, float(t0), chain)
    fed = sel.get('fed')
    e = sel.get('apply')
    if fed != 'fed' or e is None or e.get('source_seq') != announced_seq:
        return False, f'intake:{act}/{fed}', None, sel
    if int(getattr(chain, 'n_rejected', 0)) != rej0:
        return False, 'chain_rejected', None, sel
    if getattr(chain, 'fail', None) is not None:
        return False, 'chain_failed', None, sel
    return True, None, {'source_seq': int(announced_seq), 'n_recv': int(chain.n_recv), 'recv_sim_t': float(t0), 'path': 'w2_env3'}, sel
