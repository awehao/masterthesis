#!/usr/bin/env python3
"""K3-A-U 相位機：夾持後接近軸參考卸載（介入候選；規格 K3_hold_approach_unload_spec.md＋修訂 1）。door_k3a_policy.py 不改。

只改 HOLD 內的參考，其餘（接近／夾持段、OPEN／SETTLE／END_HOLD、所有監看與中止）沿用 K3APolicy：
  HOLD 起 0.5 s 之後，第一個滿足擷取條件的週期**只擷取一次**（截止 HOLD 起 1.5 s，逾時 ⇒ 中止 unload_capture_timeout）：
    |q̇| ≤ 0.002 rad/s、兩指 ≥ 0.5 N、抓取漂移 ≤ 5 mm／2°（監看與夾持檢查本輪已通過）。
  擷取量：T_tcp（同步快照 FK）、T_ref0（當下 HOLD 參考）、a＝T_ref0 工具 z 軸（單位、世界；指向門內，**不是**退讓軸 a_H）、
    δ＝a·(p_tcp − p_ref0)。界限 −3 mm ≤ δ ≤ 0；δ > 0 ⇒ 中止 unload_offset_positive；δ < −3 mm ⇒ unload_offset_out_of_range；
    非有限 ⇒ unload_offset_nonfinite。δ＝0 ⇒ 零介入（照常走完斜坡與保持）。
  斜坡：t ∈ [t_cap, t_cap＋0.5]，p_ref＝p_ref0＋a·δ·σ(u)，σ＝3u²−2u³；姿態與垂直於 a 的分量不變。
  保持：[t_cap＋0.5, t_cap＋1.0]，p_ref＝p_ref0＋a·δ。滿 t_cap＋1.0 才進 OPEN（不受父類 hold_s 計時提前切換）。
  HOLD 最長＝1.5＋1.0＝2.5 s（加離散週期誤差）。既有監看與夾持檢查每輪優先執行；中止後不發新目標、不進 OPEN。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import door_k3a_policy as P

Meas = P.Meas


@dataclass
class K3AUConfig(P.K3AConfig):
    unload_settle_s: float = 0.5
    unload_capture_deadline_s: float = 1.5
    unload_ramp_s: float = 0.5
    unload_hold_s: float = 0.5
    unload_delta_min_m: float = -0.003
    unload_delta_max_m: float = 0.0


def smoothstep(u):
    u = min(1.0, max(0.0, u))
    return 3.0 * u * u - 2.0 * u * u * u


class K3AUPolicy(P.K3APolicy):
    def __init__(self, cfg: K3AUConfig, T_HG, a_H=(0.0, -1.0, 0.0)):
        super().__init__(cfg, T_HG, a_H)
        self.unload = None              # 擷取事件 dict（只寫一次）
        self.unload_marks = {}          # ramp_end_t、hold_end_t

    def unload_target(self, t):
        u = self.unload
        T = np.array(u['T_ref0'], float)
        a = np.array(u['a'], float)
        tau = t - u['t']
        T[:3, 3] = T[:3, 3] + a * u['delta'] * smoothstep(tau / self.c.unload_ramp_s)
        return T

    def _try_capture(self, m: Meas, T_ref0):
        c = self.c
        if not (abs(m.qdot_door) <= c.settle_rate and min(m.finger_bar_min) >= c.contact_min_n):
            return False
        dp, dr = P.grasp_drift(m.T_tcp, self.G_T_H, m.H)
        if dp > c.drift_abort_m or dr > c.drift_rot_abort_rad:
            return False
        z = np.asarray(T_ref0, float)[:3, 2]
        n = float(np.linalg.norm(z))
        a = z / n
        p_tcp, p_ref = np.asarray(m.T_tcp, float)[:3, 3], np.asarray(T_ref0, float)[:3, 3]
        delta = float(a @ (p_tcp - p_ref))
        ev = {'t': m.t, 'step': getattr(m, 'step', None), 'T_tcp': np.asarray(m.T_tcp, float).tolist(),
              'T_ref0': np.asarray(T_ref0, float).tolist(), 'a': a.tolist(), 'delta': delta}
        self.events.append({'t': m.t, 'unload_capture': ev})
        if not (math.isfinite(delta) and np.isfinite(a).all()):
            self._abort('unload_offset_nonfinite', m.t, delta=delta)
            return True
        if delta > c.unload_delta_max_m:
            self._abort('unload_offset_positive', m.t, delta=delta)
            return True
        if delta < c.unload_delta_min_m:
            self._abort('unload_offset_out_of_range', m.t, delta=delta)
            return True
        self.unload = ev
        return True

    def step(self, m: Meas, dt: float):
        if self.phase != 'HOLD':
            return super().step(m, dt)
        c = self.c
        act = {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False,
               'stop_solver': False, 'finish': False}
        self.monitor(m)                                     # 既有監看優先
        if self.phase == 'ABORTED':
            return self._aborted_act(act, m)
        self._grasp_checks(m)
        if self.phase == 'ABORTED':
            return self._aborted_act(act, m)
        T_ref0 = self.target(self.H_frozen, 0.0)
        tau_h = m.t - self.t_phase
        if self.unload is None and tau_h >= c.unload_settle_s:
            if tau_h > c.unload_capture_deadline_s:
                self._abort('unload_capture_timeout', m.t)
                return self._aborted_act(act, m)
            captured = self._try_capture(m, T_ref0)
            if self.phase == 'ABORTED':
                return self._aborted_act(act, m)
            if not captured:
                act['target'] = T_ref0
        if self.unload is None:
            act['target'] = T_ref0
        else:
            act['target'] = self.unload_target(m.t)
            tau = m.t - self.unload['t']
            if tau >= c.unload_ramp_s and 'ramp_end_t' not in self.unload_marks:
                self.unload_marks['ramp_end_t'] = m.t
                self.events.append({'t': m.t, 'unload_ramp_end': True})
            if tau >= c.unload_ramp_s + c.unload_hold_s:
                self.unload_marks['hold_end_t'] = m.t
                self.events.append({'t': m.t, 'unload_hold_end': True})
                self._go('OPEN', m.t)
        act['grip'] = self.grip_cmd
        self.last_target = act['target']
        return act
