#!/usr/bin/env python3
"""K3-A-X 相位機：沿桿單軸參考重對齊（規格 K3_bar_axis_realign_spec.md；Codex 20261022 核可）。door_k3au_policy.py 不改。

繼承 K3AUPolicy（時序、界限、斜坡、保持、中止處置、監看優先全部沿用），**只換擷取時的軸**：
  axis_mode＝'bar'：a＝H_frozen 旋轉第一欄（把手 +x、桿軸；單位、世界），**不是** T_ref0 的工具 z（接近軸）、工具 +x 或退讓軸 a_H。
  δ＝a·(p_tcp − p_ref0)，限 [−3, 0] mm；斜坡 p_ref＝p_ref0＋a·δ·σ ⇒ 位移只沿 a，垂直於 a 的分量與姿態不變。
不做接近軸（y）卸載；抓取漂移基準（G_T_H_grasp）不重新初始化。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import door_k3au_policy as U

P = U.P
Meas = U.Meas
smoothstep = U.smoothstep


@dataclass
class K3AXConfig(U.K3AUConfig):
    axis_mode: str = 'bar'


class K3AXPolicy(U.K3AUPolicy):
    def __init__(self, cfg: K3AXConfig, T_HG, a_H=(0.0, -1.0, 0.0)):
        if getattr(cfg, 'axis_mode', None) != 'bar':
            raise ValueError("K3AXPolicy 只支援 axis_mode='bar'")
        super().__init__(cfg, T_HG, a_H)

    def bar_axis(self):
        x = np.asarray(self.H_frozen, float)[:3, 0]
        return x / np.linalg.norm(x)

    def _try_capture(self, m: Meas, T_ref0):
        c = self.c
        if not (abs(m.qdot_door) <= c.settle_rate and min(m.finger_bar_min) >= c.contact_min_n):
            return False
        dp, dr = P.grasp_drift(m.T_tcp, self.G_T_H, m.H)
        if dp > c.drift_abort_m or dr > c.drift_rot_abort_rad:
            return False
        a = self.bar_axis()
        p_tcp, p_ref = np.asarray(m.T_tcp, float)[:3, 3], np.asarray(T_ref0, float)[:3, 3]
        delta = float(a @ (p_tcp - p_ref))
        ev = {'t': m.t, 'step': getattr(m, 'step', None), 'axis_mode': 'bar', 'T_tcp': np.asarray(m.T_tcp, float).tolist(),
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
