#!/usr/bin/env python3
"""命令鏈 E2＋任期（mode 4 重設計 R1–R3：全身 → 導航 → 全身；Codex 20261029_145418 §交棒 1–3）。wb_cmd_chain_e2 不改，以子類別擴充。

任期＝全身連續持有控制權的一段。第一段沿用既有承接（seed_from_handover，u_prev 須為 None）。
  close_tenure(sim_t, step)      執行端在「全身 → 直寫路徑」切換當步呼叫：封存本段（u_prev、設定點、快照序號），
                                 **清除快照**（上一段的命令不得在下一段被沿用）、停止積分。
  dry_run_handover(...)          已封存時在深拷上先清基準再走既有預核（同一條計算路徑）。
  rebase_from_handover(...)      再次進入：已封存才准；清 u_prev／設定點後走既有 seed_from_handover，任期＋1。
不清的狀態：fail（失效閂鎖跨任期保留）、計數與事件。晚到命令由入口依任期標記丟棄（door_k3m_applied_src_t.EnvIntakeT），本層不判斷命令屬於哪一段。
"""
from __future__ import annotations

import copy

from wb_cmd_chain_e2 import CmdChainE2

VERSION_T = 'wb_cmd_chain_e2t/1'


class CmdChainE2T(CmdChainE2):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.tenure = 0
        self.tenure_open = True
        self.tenures = []

    def close_tenure(self, sim_t, step):
        if not self.tenure_open:
            raise RuntimeError(f'任期 {self.tenure} 已封存，不得重複封存')
        self.tenures.append({'tenure': self.tenure, 'closed_sim_t': float(sim_t), 'closed_step': int(step),
                             'u_prev': None if self.u_prev is None else [float(x) for x in self.u_prev],
                             'setpoint': None if self.setpoint is None else [float(x) for x in self.setpoint],
                             'last_recv_seq': None if self.snap is None else int(self.snap.recv_seq)})
        self.tenure_open = False
        self.snap = None
        self.applied = None
        self.integrating = False
        self.events.append((round(float(sim_t), 4), 'tenure_closed', f'任期 {self.tenure} 封存（物理步 {int(step)}）；快照清除'))
        return dict(self.tenures[-1])

    def _cleared(self):
        p = copy.deepcopy(self)
        p.u_prev = None
        p.setpoint = None
        return p

    def dry_run_handover(self, u_applied, arm_setpoint, sim_t, physics_step_id, dt, q_arm_measured, n_steps=None):
        if self.u_prev is not None and not self.tenure_open:
            return CmdChainE2.dry_run_handover(self._cleared(), u_applied, arm_setpoint, sim_t, physics_step_id, dt,
                                               q_arm_measured, n_steps=n_steps)
        return super().dry_run_handover(u_applied, arm_setpoint, sim_t, physics_step_id, dt, q_arm_measured, n_steps=n_steps)

    def seed_from_handover(self, u_applied, arm_setpoint, sim_t, physics_step_id, source=''):
        """已封存 ⇒ 走再次進入（rebase）；否則沿用既有承接（第一段）。執行層的切換程式不必分辨是第幾段。"""
        if not self.tenure_open:
            return self.rebase_from_handover(u_applied, arm_setpoint, sim_t, physics_step_id, source=source)
        return CmdChainE2.seed_from_handover(self, u_applied, arm_setpoint, sim_t, physics_step_id, source=source)

    def rebase_from_handover(self, u_applied, arm_setpoint, sim_t, physics_step_id, source=''):
        if self.tenure_open:
            raise RuntimeError(f'任期 {self.tenure} 未封存，不得重設基準')
        if self.fail is not None:
            raise RuntimeError(f'命令鏈已失效，不得再進入：{self.fail}')
        old = (self.u_prev, self.setpoint)
        self.u_prev, self.setpoint = None, None
        try:
            r = CmdChainE2.seed_from_handover(self, u_applied, arm_setpoint, sim_t, physics_step_id, source=source)
        except (ValueError, RuntimeError):
            self.u_prev, self.setpoint = old
            raise
        self.tenure += 1
        self.tenure_open = True
        self.events.append((round(float(sim_t), 4), 'tenure_opened', f'任期 {self.tenure}（來源 {source or "?"}）'))
        return r

    def summary(self):
        d = super().summary()
        d.update({'version_t': VERSION_T, 'tenure': self.tenure, 'tenure_open': self.tenure_open, 'tenures': list(self.tenures)})
        return d
