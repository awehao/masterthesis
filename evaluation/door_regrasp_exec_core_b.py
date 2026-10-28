#!/usr/bin/env python3
"""換抓執行端核心 r6（切法 B；Codex 20261028）：door_regrasp_exec_core.RegraspExecCore 的子類，搭配權威 door_regrasp_mode4（mode 2 夾持、mode 3 張爪＋退離）。
與 r5 的差異只在 mode 2–4 的准入與 gate 生命週期：
  - mode 3 准入（gates[3]）＝**放手前原地靜止確認** PreReleaseStill：mode 2 提交步起，同 Gate2StopObs 的判準（p4b 窗速度全部 ≤ CP 門檻、
    本步套用命令為保持形、步號連續有限），最近連續 PRE3_OBS_S（1.0 s）每步合格才確認；不合格撤銷並清窗、不 FAILED；確認後至 mode 3 提交仍逐步核。
    前模式套用確認（權威 applied_confirmed）與求解端交接條件（HeldSupervisor release_settled）照舊，不在此。
  - mode 3 提交步：PreReleaseStill 結束（保存）；放手監看 Gate3Release.open（自該步起算；張爪與退離同步開始）與 Gate4Exit.start 同步啟動。
    保護交接證據改為 PreReleaseStill 最後有效步 ∈ {k−1, k} 且接觸資料無進行中的失效（不符 ⇒ 中止 rg_handover_no_valid_evidence）。
  - mode 4 准入（gates[4]）＝Gate3Release 確認且新鮮 **且** Gate4Exit 確認且新鮮（請求與提交當下皆由權威核）；退離量不能代替接觸解除。
    mode 4 提交步：Gate3Release、Gate4Exit 一起結束（保存判定）。
  - Gate3Release／Gate4Exit 失效、逾時、FAILED 照 r5 中止（rg_gate3:<why>／rg_gate4:<why>）；接觸規則（mode 3 離開前 bar_out）、資料失效處置不變。
"""
from __future__ import annotations

import door_k3m_check_p as CP
import door_k3m_check_p4b as P4B
import door_k3m_pause_sup_p4 as P4
import door_regrasp_exec_core as RX
import door_regrasp_mode4 as RG4

VERSION = 'door_regrasp_exec_core_b/r6'
PRE3_OBS_S = 1.0


class PreReleaseStill(RX.Gate2StopObs):
    name = 'pre3_release_still'

    def __init__(self, dt=0.01, obs_s=PRE3_OBS_S):
        super().__init__(dt=dt)
        self.obs_s = float(obs_s)

    def step(self, step, t, base3, door_q, arm6, base_cmd3, arm_cmd6):
        self.n['steps'] += 1
        self.buf.append([step, float(t), list(base3), list(arm6), float(door_q)])
        if self.t_zero is None or float(t) < self.t_zero - 1e-9:
            return self.state
        self.n['checked'] += 1
        v = P4B.window_speeds(list(self.buf), self.I, len(self.buf) - 1)
        if v is None:
            self._reset(step, t, 'data_invalid_or_gap')
            return self.state
        self.n['max_v'] = [max(a, b) for a, b in zip(self.n['max_v'], v)]
        if not CP.low(v):
            self._reset(step, t, 'moved', v)
            return self.state
        if not P4.hold_cmd_ok(base_cmd3, arm_cmd6, CP.TOL_APPLIED):
            self._reset(step, t, 'applied_not_hold')
            return self.state
        self.win0 = self.win0 if self.win0 is not None else float(t)
        self.last_valid_step = step
        if self.state == 'WAIT' and float(t) - self.win0 >= self.obs_s - 1e-9:
            self.state = 'CONFIRMED'
            self.t['CONFIRMED'] = float(t)
            self.t['win0'] = self.win0
            self.events.append({'step': step, 't': t, 'to': 'CONFIRMED', 'win0': self.win0})
        return self.state

    def report(self):
        r = super().report()
        r.update(gate=self.name, version='r6', obs_s=self.obs_s)
        return r


class RegraspExecCoreB(RX.RegraspExecCore):
    def __init__(self, auth, pkg, H_in_fn, dt=0.01, invalid_max_s=0.2, reg=None):
        if not isinstance(auth, RG4.RegraspAuthority4):
            raise ValueError('切法 B 核心需權威 door_regrasp_mode4.RegraspAuthority4')
        super().__init__(auth, pkg, H_in_fn, dt=dt, invalid_max_s=invalid_max_s, reg=reg)
        self.pre3 = PreReleaseStill(dt=self.dt)
        auth.gates[3] = self._gate_pre3
        auth.gates[4] = self._gate4b

    def _gate_pre3(self, step):
        if 'pre3' not in self.active:
            return False, 'pre3_inactive'
        return (True, None) if self.pre3.confirmed_fresh(step) else (False, f'pre3_{self.pre3.state}')

    def _gate4b(self, step):
        g3, g4 = self.g[3], self.g[4]
        if 3 not in self.active or not g3.confirmed_fresh(step):
            return False, f'gate3_release_{g3.state if 3 in self.active else "inactive"}'
        if 4 not in self.active or not g4.confirmed_fresh(step):
            return False, f'gate4_{g4.state if 4 in self.active else "inactive"}'
        return True, None

    def _end(self, key, step, t):
        if key == 'pre3':
            if key in self.active:
                self.active.discard(key)
                self.ended[key] = {'step': step, 't': t, 'state': self.pre3.state}
            return
        super()._end(key, step, t)

    def on_commit(self, step, t, mode, T_tcp):
        if mode not in (2, 3, 4):
            return super().on_commit(step, t, mode, T_tcp)
        self.commits.append({'step': step, 't': t, 'mode': mode})
        self.engaged = True
        if mode == 2:
            self._end(2, step, t)
            self.pre3.set_zero(float(t))
            self.active.add('pre3')
        elif mode == 3:
            self._end('pre3', step, t)
            ev = self.pre3.last_valid_step
            self.handover = {'step': step, 't': t, 'last_old_step': step - 1, 'first_new_step': step, 'evidence_step': ev, 'evidence': 'pre3_release_still',
                             'what': 'K3-M 走廊／包絡監看最後判定於 k−1；換抓保護（ContactWindow、Gate3Release、Gate4、權威全身形）自 k 起；張爪與退離同步自 k 起'}
            if ev not in (step - 1, step) or self.inv_t0 is not None:
                self._abort('rg_handover_no_valid_evidence', step, t)
            self.g[3].open(step, t)
            self.active.add(3)
            self.g[4].start(step, t, T_tcp0=RX.np.asarray(T_tcp, float))
            self.active.add(4)
        else:
            self._end(3, step, t)
            self._end(4, step, t)
            self.g[5].start(step, t, base_goal=self.B1, arm_stow=self.STOW)
            self.active.add(5)

    def step(self, step, t, mode_sel, rec, T_tcp, base, arm_q, finger_q, door_q, base_cmd=None, arm_cmd=None):
        if self.abort is None and 'pre3' in self.active:
            self.pre3.step(step, t, base, door_q, arm_q, base_cmd if base_cmd is not None else [float('nan')] * 3,
                           arm_cmd if arm_cmd is not None else [float('nan')] * 6)
        return super().step(step, t, mode_sel, rec, T_tcp, base, arm_q, finger_q, door_q, base_cmd=base_cmd, arm_cmd=arm_cmd)

    def gates_state(self, step):
        out = super().gates_state(step)
        p = self.pre3
        out['pre3'] = {'state': p.state if ('pre3' in self.active or 'pre3' in self.ended) else 'IDLE',
                       'fresh': bool('pre3' in self.active and p.confirmed_fresh(step)), 'why': p.why, 'active': 'pre3' in self.active}
        g4b = self._gate4b(step)
        out['gate4b'] = {'state': 'CONFIRMED' if g4b[0] else 'WAITING', 'fresh': bool(g4b[0]), 'why': g4b[1], 'active': 4 in self.active}
        return out

    def report(self):
        r = super().report()
        r.update(version=VERSION, base_version=RX.VERSION, pre3=self.pre3.report(), semantics=RG4.SEMANTICS)
        return r
