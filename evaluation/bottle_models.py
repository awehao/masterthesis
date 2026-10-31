#!/usr/bin/env python3
"""瓶子對照 C2：**獨立理想速度模型**（Codex 20261031_165248 §2；不改凍結核心、不以 α＝1 替代）。
保留 15 維介面 z＝[q(9), s(6)]：
  手臂實測角預測  q_a⁺ ＝ q_a ＋ dt·u_a        （不含伺服追蹤、不含偏差）
  設定點          s⁺   ＝ s   ＋ dt·u_a        （照舊獨立積分，供設定點限位核對）
  底盤            與原核心相同（顯式 Euler）
實作：wgmpc_core_sp 的手臂區塊完全由 cfg.composed() 的 (P, Q, G, h) 決定（step_sp、affine_model_sp 都只讀它），
故 C2 只覆寫 composed() 為 (1, 0, dt, 0)。延遲前推另給 ideal_phys_step（逐物理步同一理想模型）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ammr_wholebody_mpc import wgmpc_core_sp as S
from ammr_wholebody_mpc.wgmpc_core import body_to_world


@dataclass
class WGMPCConfigSPIdeal(S.WGMPCConfigSP):
    """C2：手臂以理想速度積分預測。arm_model 仍保留（只為介面相容；預測不使用其 α、b）。"""

    def composed(self):
        kp = int(round(self.dt / self.arm_model.phys_dt))
        return (np.ones(S.NS), np.zeros(S.NS), np.full(S.NS, self.dt), np.zeros(S.NS), kp)


def ideal_phys_step(q, s, u, dt_p):
    """理想模型逐物理步（延遲前推用）：q_a、s 都以 u_a 積分；底盤同原核心。"""
    q = np.asarray(q, float).copy()
    s = np.asarray(s, float).copy()
    u = np.asarray(u, float)
    q[:3] = q[:3] + dt_p * (body_to_world(float(q[2])) @ u)[:3]
    q[3:] = q[3:] + dt_p * u[3:]
    s = s + dt_p * u[3:]
    return q, s
