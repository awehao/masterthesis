#!/usr/bin/env python3
"""K3-M 相位機：底盤可動的小角度協同開門（規格 K3_mobile_coop_small_open_spec.md draft-3＋Codex 20261023_204354）。door_k3a_policy.py 不改。

繼承 K3APolicy（不疊加 B／U／X）。UNFOLD～HOLD 完全同 K3-A。差異只在 OPEN 起：
  - **OPEN 第一輪**：提出模式切換請求（TaskModeWaiter；放進 outbox 由節點發布），之後保持最後已發布目標、**不建立 DoorTargetK3、不推進參考**，
    直到收到一致的執行端＋求解器 ack（confirmed）；nack、不一致、逾時（模擬 1.0 s／牆鐘 2.0 s）、中止 ⇒ 中止 mode_switch:<why>。
  - confirmed 的那一輪起交回父類 OPEN 邏輯（首輪 DoorTargetK3 不推進 ⇒ 首筆目標＝最後 HOLD 目標）。
  - **監看替換**：收到有效的執行端 ack（執行端已在明確物理步關閉 HoldMonitor）起，父類的底盤 20 mm／2° 漂移檢查改由
    (1) 走廊第二層（door_k3m_corridor；內側帶 20 mm／0.8°，以實測底盤位姿與實測門角）與
    (2) 實測 FK 幾何距離（door_k3m_geom_monitor；stop ⇒ 中止 geometry:<why>；warn 只記錄）取代；其他監看全部照舊。
  - 執行端越界（/park/violation，m.executor_violation）⇒ 中止 executor_violation。
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import door_k3a_policy as P
import door_k3m_mode as MODE

Meas = P.Meas


@dataclass
class K3MConfig(P.K3AConfig):
    corridor_inner_mm: float = 20.0
    corridor_inner_deg: float = 0.8
    mode_timeout_s: float = 1.0
    mode_wall_timeout_s: float = 2.0


class K3MPolicy(P.K3APolicy):
    def __init__(self, cfg: K3MConfig, T_HG, run_id: str, corridor, geom_monitor, a_H=(0.0, -1.0, 0.0)):
        super().__init__(cfg, T_HG, a_H)
        if not run_id:
            raise ValueError('run_id 必須非空')
        self.run_id = run_id
        self.corridor = corridor
        self.geom = geom_monitor
        self.waiter = MODE.TaskModeWaiter(run_id, cfg.mode_timeout_s, cfg.mode_wall_timeout_s)
        self.outbox = []
        self.mode_state = 'hold'            # hold → waiting → free
        self.geom_last = None
        self.geom_t = None
        self.corr_last = None
        self.n_geom_warn = 0

    # ---- 節點轉入的 ack（t_recv＝收到時的最新狀態模擬時間）
    def on_executor_ack(self, msg, t_recv):
        self.waiter.on_executor(msg, t_recv)

    def on_solver_ack(self, msg, t_recv):
        self.waiter.on_solver(msg, t_recv)

    @property
    def exec_switched(self):
        return self.waiter.exec_ack is not None

    def monitor(self, m: Meas):
        if self.phase not in P.ACTIVE_MONITOR:
            return
        if getattr(m, 'executor_violation', None) is not None:
            return self._abort('executor_violation', m.t, detail=m.executor_violation)
        if not self.exec_switched:
            return super().monitor(m)
        m2 = dataclasses.replace(m, base_drift_m=0.0, base_drift_yaw=0.0)         # 底盤 20 mm／2° 由走廊與幾何取代
        super().monitor(m2)
        if self.phase == 'ABORTED':
            return
        c = self.c
        ok, e, why = self.corridor.check(getattr(m, 'base_pose', None), None if m.q_door is None else math.degrees(m.q_door),
                                         c.corridor_inner_mm * 1e-3, math.radians(c.corridor_inner_deg))
        self.corr_last = e
        if not ok:
            return self._abort('base_corridor', m.t, kind=why, dev=e)
        if self.geom_last is not None and self.geom_t == m.t:
            dec, det = self.geom_last                        # 同一輪（同一狀態時刻）只算一次
        else:
            dec, det = self.geom.step(m.t, getattr(m, 'qf', None), m.q_door, getattr(m, 'q_finger', None), m.state_age)
            self.geom_last, self.geom_t = (dec, det), m.t
            if dec == 'warn':
                self.n_geom_warn += 1
        if dec == 'stop':
            return self._abort(f"geometry:{det.get('why')}", m.t, detail=det)

    def step(self, m: Meas, dt: float):
        if self.phase != 'OPEN' or self.mode_state == 'free':
            return super().step(m, dt)
        act = {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False,
               'stop_solver': False, 'finish': False}
        if self.mode_state == 'hold':
            self.outbox.append(self.waiter.make_request(int(getattr(m, 'state_step')), m.t, float(getattr(m, 'wall'))))
            self.mode_state = 'waiting'
            self.events.append({'t': m.t, 'k3m_mode_request': dict(self.waiter.request)})
        self.monitor(m)
        if self.phase == 'ABORTED':
            return self._aborted_act(act, m)
        self._grasp_checks(m)
        if self.phase == 'ABORTED':
            return self._aborted_act(act, m)
        st = self.waiter.status(m.t, float(getattr(m, 'wall')))
        if isinstance(st, tuple):
            self._abort(f'mode_switch:{st[1]}', m.t)
            return self._aborted_act(act, m)
        if st == 'confirmed':
            self.mode_state = 'free'
            self.events.append({'t': m.t, 'k3m_mode_confirmed': {'exec': self.waiter.exec_ack, 'solver': self.waiter.solver_ack}})
            return super().step(m, dt)                       # 本輪起交回父類 OPEN（首輪 DoorTargetK3 不推進）
        act['target'] = self.last_target                    # 等待中：保持最後已發布目標、不推進參考
        act['grip'] = self.grip_cmd
        return act
