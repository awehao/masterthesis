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


V_TRIG = 0.025          # m/s：實測 TCP 沿退離方向速度門檻（Codex 單一候選）
N_TRIG = 3              # 連續步數
K_DIFF = 5              # 差分跨距（步；0.05 s）
T_TRIG_MAX = 1.5        # s：mode 3 提交後未建立退離 ⇒ 中止（不張爪硬試）


class RegraspExecCoreBT(RegraspExecCoreB):
    """r7（Codex 20261028 pb1_6 回覆）：mode 3 提交後**先閉爪退離**，以實測 TCP 沿提交時工具 −z 的速度（K_DIFF 步差分）連續 N_TRIG 步
    ≥ V_TRIG 確認運動已建立，才呼叫權威 trigger_open（同一權威發出張爪）並自該步開啟放手監看 Gate3Release（計時自真正張爪起）。
    Gate4Exit 仍自 mode 3 提交步起算（退離量、收斂、離開後無接觸）；mode 4 准入同 r6（Gate3Release 確認新鮮＋Gate4Exit）。
    提交後 T_TRIG_MAX 內未建立 ⇒ 中止 rg_retreat_not_established；張爪前 Gate4 已離開閂鎖 ⇒ 中止 rg_exited_before_open（不繞過接觸規則）。"""

    def __init__(self, auth, pkg, H_in_fn, dt=0.01, invalid_max_s=0.2, reg=None):
        if not isinstance(auth, RG4.RegraspAuthority4T):
            raise ValueError('r7 核心需權威 door_regrasp_mode4.RegraspAuthority4T')
        super().__init__(auth, pkg, H_in_fn, dt=dt, invalid_max_s=invalid_max_s, reg=reg)
        self.rt = None

    def on_commit(self, step, t, mode, T_tcp):
        if mode != 3:
            return super().on_commit(step, t, mode, T_tcp)
        self.commits.append({'step': step, 't': t, 'mode': mode})
        self.engaged = True
        self._end('pre3', step, t)
        ev = self.pre3.last_valid_step
        self.handover = {'step': step, 't': t, 'last_old_step': step - 1, 'first_new_step': step, 'evidence_step': ev, 'evidence': 'pre3_release_still',
                         'what': 'K3-M 走廊／包絡監看最後判定於 k−1；換抓保護自 k 起；閉爪退離自 k 起、張爪待實測退離建立'}
        if ev not in (step - 1, step) or self.inv_t0 is not None:
            self._abort('rg_handover_no_valid_evidence', step, t)
        T0 = RX.np.asarray(T_tcp, float)
        self.rt = {'t0': float(t), 'step0': int(step), 'p0': T0[:3, 3].copy(), 'z0': T0[:3, 2].copy(), 'hist': [], 'n_ok': 0, 'trigger': None, 'v_max': 0.0}
        self.g[4].start(step, t, T_tcp0=T0)
        self.active.add(4)

    def step(self, step, t, mode_sel, rec, T_tcp, base, arm_q, finger_q, door_q, base_cmd=None, arm_cmd=None):
        rt = self.rt
        if self.abort is None and rt is not None and rt['trigger'] is None and T_tcp is not None:
            p = RX.np.asarray(T_tcp, float)[:3, 3]
            s = float((rt['p0'] - p) @ rt['z0'])
            rt['hist'].append((int(step), float(t), s))
            v = None
            if len(rt['hist']) > K_DIFF:
                (k0, t0, s0), (k1, t1, s1) = rt['hist'][-1 - K_DIFF], rt['hist'][-1]
                if k1 - k0 == K_DIFF and t1 > t0:
                    v = (s1 - s0) / (t1 - t0)
            if v is not None:
                rt['v_max'] = max(rt['v_max'], v)
            rt['n_ok'] = rt['n_ok'] + 1 if (v is not None and v >= V_TRIG) else 0
            if bool(getattr(self.g[4], 'exited', False)):
                return self._abort('rg_exited_before_open', step, t)
            if rt['n_ok'] >= N_TRIG:
                info = {'v_mps': v, 's_m': s, 'dt_since_commit_s': float(t) - rt['t0']}
                if self.auth.trigger_open(step, t, info):
                    rt['trigger'] = dict(info, step=int(step), t=float(t))
                    self.g[3].open(step, t)
                    self.active.add(3)
            elif float(t) - rt['t0'] > T_TRIG_MAX + 1e-9:
                return self._abort('rg_retreat_not_established', step, t)
        return super().step(step, t, mode_sel, rec, T_tcp, base, arm_q, finger_q, door_q, base_cmd=base_cmd, arm_cmd=arm_cmd)

    def report(self):
        r = super().report()
        r['version'] = 'door_regrasp_exec_core_b/r7'
        r['retreat_trigger'] = None if self.rt is None else {k: v for k, v in self.rt.items() if k not in ('hist', 'p0', 'z0')}
        r['retreat_trigger_reg'] = {'V_TRIG': V_TRIG, 'N_TRIG': N_TRIG, 'K_DIFF': K_DIFF, 'T_TRIG_MAX': T_TRIG_MAX}
        return r
