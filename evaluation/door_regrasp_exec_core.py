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
另：bench_release_w2——台架首筆在 W2 唯一入口（v3 封裝）下經 W2Exec.on_env3→pre_step 進鏈（同 BENCH.release_intake 的檢核）。
"""
from __future__ import annotations

import numpy as np

import door_regrasp_contact_window as RCW
import door_regrasp_gates3 as GT
import door_regrasp_ref_registry as REGY

VERSION = 'door_regrasp_exec_core/r2'
GATE_KEY = {3: 'gate3', 4: 'gate4', 5: 'gate5', 6: 'gate6'}


class RegraspExecCore:
    def __init__(self, auth, pkg, H_in_fn, dt=0.01, invalid_max_s=0.2, reg=None):
        ok, why = REGY.validate(pkg)
        if not ok:
            raise ValueError(f'參考包登錄核驗不過：{why}')
        self.auth, self.H_in_fn, self.dt, self.invalid_max_s = auth, H_in_fn, float(dt), float(invalid_max_s)
        self.T_HG = np.asarray(pkg['inner_grasp']['T_HG'], float).reshape(4, 4)
        self.B1, self.STOW = np.asarray(pkg['B1'], float), np.asarray(pkg['stow'], float)
        reg = dict(reg or {}, dt=self.dt)
        self.g = {3: GT.Gate3Release(reg=reg), 4: GT.Gate4Exit(reg=reg), 5: GT.Gate5Arrive(reg=reg), 6: GT.Gate6Admit(reg=reg)}
        self.hold = GT.HoldConfirm(reg=reg)
        self.active = set()                       # 正在逐步更新的 gate（3–6）與 'hold'
        self.ended = {}                           # gate → {'step','t','state'}
        self.cw = RCW.ContactWindow(f_max=GT.REG['contact_abort_n'], dt=self.dt)
        self.engaged = False                      # mode ≥ 1 提交後
        self.inv_t0 = None
        self.abort = None
        self.commits = []
        self.handover = None
        auth.gates = {k: (lambda s, _k=k: self._fresh(_k, s)) for k in (3, 4, 5, 6)}

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

    def step(self, step, t, mode_sel, rec, T_tcp, base, arm_q, finger_q, door_q):
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
        for k in sorted(x for x in self.active if x != 'hold'):
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
