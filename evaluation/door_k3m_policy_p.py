#!/usr/bin/env python3
"""K3-M 相位機＋受控暫停（規格 K3M_pause_integration_spec_draft1.md §1–§3）。door_k3m_policy.py／door_k3a_policy.py 不改。

繼承 K3MPolicy；差異只在 SETTLE 之後：
  - P1 觸發：父類的**完整 SETTLE 條件**（|q−8°|≤0.5°、|q̇|≤0.002、抓取漂移、兩指接觸，連續 1 s；每週期的 ACTIVE_MONITOR 與 K3-M 走廊／幾何監看照舊）
    成立 ⇒ 父類轉入 END_HOLD 的同一輪改轉 **PAUSE**，以**最後一筆已發布目標**（frozen_target_seq＝節點的已發布序號）提出暫停請求
    （TaskPauseWaiter；放進 pause_outbox，由節點經 take_pause_outbox() 發布——已中止則清除不發）。
  - PAUSE 期間：
      * 監看與夾持檢查**以 END_HOLD 身分**執行（父類模組常數 ACTIVE_MONITOR／OPENING／FINGER_BAR_ALLOWED／GRASP_PHASES 不能改；
        END_HOLD 的監看集合＝門角界、雙指接觸、新鮮度、非設計／牆接觸、腕力、門速、K3-M 走廊與幾何）；任何違規 ⇒ 既有中止（K2 閂鎖）。
      * 目標＝最後一筆已發布目標（不推進參考）。
      * 握手：nack、不一致、逾時、凍結目標不符 ⇒ 中止 pause_switch:<why>。
      * 求解端 /wgmpc/pause_status：FAIL ⇒ 中止 pause_failed:<fail_why>；DONE（命令停止＋實際停止＋觀察 10 s）⇒ DONE。
      * 暫停總時長上限 pause_total_s（模擬時間）⇒ 中止 pause_total_timeout。
  正常暫停通過**不等於**異常停止風險解除；異常一律走既有閂鎖。
"""
from __future__ import annotations

from dataclasses import dataclass

import door_k3m_pause_mode as PMODE
import door_k3m_policy as PM

Meas = PM.Meas


@dataclass
class K3MPConfig(PM.K3MConfig):
    pause_timeout_s: float = 1.0
    pause_wall_timeout_s: float = 2.0
    pause_total_s: float = 25.0


class K3MPPolicy(PM.K3MPolicy):
    def __init__(self, cfg: K3MPConfig, T_HG, run_id: str, corridor, geom_monitor, a_H=(0.0, -1.0, 0.0)):
        super().__init__(cfg, T_HG, run_id, corridor, geom_monitor, a_H)
        self.pwaiter = PMODE.TaskPauseWaiter(run_id, cfg.pause_timeout_s, cfg.pause_wall_timeout_s)
        self.pause_outbox = []
        self.pause_state = 'none'           # none → waiting → active
        self.pause_t0 = None
        self.pause_status = None
        self.pause_outcome = None
        self.pub_seq = 0                    # 節點每輪設定：目前已發布目標的序號

    def on_pause_executor(self, msg, t_recv, wall_recv):
        self.pwaiter.on_executor(msg, t_recv, wall_recv)

    def on_pause_solver(self, msg, t_recv, wall_recv):
        self.pwaiter.on_solver(msg, t_recv, wall_recv)

    def on_pause_status(self, msg):
        if isinstance(msg, dict) and isinstance(msg.get('outcome'), dict):
            self.pause_status = msg

    def take_pause_outbox(self):
        if self.phase == 'ABORTED':
            self.pause_outbox.clear()
            return []
        out, self.pause_outbox = list(self.pause_outbox), []
        return out

    def _as_end_hold(self, fn, m):
        """以 END_HOLD 身分呼叫父類監看／夾持檢查；中止則保留 ABORTED，否則還原 PAUSE。"""
        self.phase = 'END_HOLD'
        try:
            fn(m)
        finally:
            if self.phase == 'END_HOLD':
                self.phase = 'PAUSE'
            elif self.phase == 'ABORTED' and self.events and self.events[-1].get('abort') and self.events[-1].get('phase') == 'END_HOLD':
                self.events[-1]['phase'] = 'PAUSE'                     # 中止紀錄標實際相位（監看以 END_HOLD 身分執行）
                self.events[-1]['monitored_as'] = 'END_HOLD'

    def step(self, m: Meas, dt: float):
        if self.phase != 'PAUSE':
            act = super().step(m, dt)
            if self.phase == 'END_HOLD':                  # 父類剛以完整 SETTLE 條件轉入 END_HOLD ⇒ 改轉 PAUSE
                self._go('PAUSE', m.t)
                self.pause_t0 = m.t
                T = self.last_target
                req = self.pwaiter.make_request(int(getattr(m, 'state_step')), m.t, float(getattr(m, 'wall')),
                                                int(self.pub_seq), [float(v) for v in T[:3, 3]])
                self.pause_outbox.append(req)
                self.pause_state = 'waiting'
                self.events.append({'t': m.t, 'k3m_pause_request': dict(req)})
                act['target'] = T
            return act
        act = {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False,
               'stop_solver': False, 'finish': False}
        self._as_end_hold(self.monitor, m)
        if self.phase == 'ABORTED':
            self.pause_outbox.clear()
            return self._aborted_act(act, m)
        self._as_end_hold(self._grasp_checks, m)
        if self.phase == 'ABORTED':
            self.pause_outbox.clear()
            return self._aborted_act(act, m)
        if m.t - self.pause_t0 > self.c.pause_total_s:
            self._abort('pause_total_timeout', m.t)
            return self._aborted_act(act, m)
        st = self.pwaiter.status(m.t, float(getattr(m, 'wall')))
        if isinstance(st, tuple):
            self._abort(f'pause_switch:{st[1]}', m.t)
            return self._aborted_act(act, m)
        if st == 'confirmed' and self.pause_state != 'active':
            self.pause_state = 'active'
            self.events.append({'t': m.t, 'k3m_pause_confirmed': {'exec': self.pwaiter.exec_ack, 'solver': self.pwaiter.solver_ack}})
        if self.pause_state == 'active' and self.pause_status is not None:
            o = self.pause_status['outcome']
            if o.get('state') == 'FAIL':
                self.pause_outcome = o
                self._abort(f"pause_failed:{o.get('fail_why')}", m.t, outcome=o)
                return self._aborted_act(act, m)
            if o.get('state') == 'DONE' and o.get('command_stop_confirmed') and o.get('physical_stop_confirmed'):
                self.pause_outcome = o
                self._go('DONE', m.t, pause_outcome=o)
                act.update(stop_solver=True, finish=True)
                self.done = True
                return act
        act['target'] = self.last_target
        return act
