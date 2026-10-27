#!/usr/bin/env python3
"""持握閉合修正的底盤引導（C2；Codex r35：保留原出口引導，加入方向明確、可界限的閉合修正；純邏輯、不依賴 ROS／Isaac）。

原引導（door_wide_guidance.guidance）：參考位姿 pr＝表位姿（依實測門角）；世界速度 vw＝表前饋＋k_lin·(pr − 實測底盤)，再逐軸限幅。
C2 只改參考位姿的**平移**：pr′＝pr＋s(q)·c，c 為閉合修正（世界 x、y，公尺）。前饋、偏航、k_lin／k_yaw、v_cap／w_cap 限幅全不變
⇒ 加上修正後的總參考仍受既有速度限制。

閉合誤差：e＝p_target − p_tcp（世界水平 x、y）；p_target＝(H·G_T_H⁻¹) 的平移，H＝當輪同步量測的把手位姿、G_T_H＝原抓取基準（不回填、不重設）；
p_tcp＝實測底盤＋實測手臂角的 TCP（正運動學）。直覺：手臂不動、底盤平移 e 時 TCP 也平移 e。偏航與鉛直不修（另報）。
修正狀態 c 的更新（每輪 dt）：
  c_des＝e；c ← c＋clip((dt/τ)·(c_des − c), ±rate·dt)（各軸，一階＋變化率界限）；再把 |c| 截到 c_max。
啟用權重 s(q)：門角 < q_on ⇒ 0；q_on–q_full 以 smoothstep 平滑上升；≥ q_full ⇒ 1（只依實測門角，不依時間）。
  s(q)＝0 時 c 不更新、維持 0（不在啟用前累積）。
失效處置：任一輸入非有限或維度錯 ⇒ 本輪不更新 c（沿用上一個有效 c）、n_invalid＋1、回傳 why；不拋例外、不新增命令來源。
相位：只在原引導作用的相位（OPEN、SETTLE）使用；c 在 SETTLE 持續沿用（不撤除，避免跳變）；PAUSE 以後原引導整體關閉（既有行為）。
參數（A/B 前固定，登錄後不得看結果再改）：DEFAULT。

    python3 evaluation/test_door_coopen_closure_guidance.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClosureCfg:
    q_on_deg: float = 36.0       # 開始平滑啟用的門角
    q_full_deg: float = 40.0     # 全量啟用的門角
    tau_s: float = 1.0           # 一階時間常數
    rate_m_s: float = 0.002      # 每軸變化率界限（m/s）
    c_max_m: float = 0.008       # 修正幅度上限（m，水平範數）


DEFAULT = ClosureCfg()


def smoothstep(q_deg, a, b):
    if q_deg <= a:
        return 0.0
    if q_deg >= b:
        return 1.0
    x = (q_deg - a) / (b - a)
    return x * x * (3.0 - 2.0 * x)


@dataclass
class ClosureCorrector:
    cfg: ClosureCfg = field(default_factory=ClosureCfg)
    c: np.ndarray = field(default_factory=lambda: np.zeros(2))
    n_update: int = 0
    n_invalid: int = 0
    n_clip_mag: int = 0
    n_rate_limited: int = 0
    s: float = 0.0               # 最近一個有效輪的啟用權重

    def update(self, dt, q_door_deg, p_tcp_xy, p_target_xy):
        """回傳 (s, c 世界 xy, info)。"""
        try:
            pt, pg = np.asarray(p_tcp_xy, float).reshape(2), np.asarray(p_target_xy, float).reshape(2)
            ok = math.isfinite(float(dt)) and float(dt) > 0 and math.isfinite(float(q_door_deg)) and np.all(np.isfinite(pt)) and np.all(np.isfinite(pg))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            self.n_invalid += 1
            return self.s, self.c.copy(), {'why': 'input_invalid'}
        s = smoothstep(float(q_door_deg), self.cfg.q_on_deg, self.cfg.q_full_deg)
        e = pg - pt
        if s <= 0.0:
            self.c, self.s = np.zeros(2), 0.0
            return 0.0, self.c.copy(), {'why': None, 'e': e.tolist(), 'active': False}
        step = (float(dt) / self.cfg.tau_s) * (e - self.c)
        lim = self.cfg.rate_m_s * float(dt)
        if np.any(np.abs(step) > lim + 1e-15):
            self.n_rate_limited += 1
        c = self.c + np.clip(step, -lim, lim)
        n = float(np.linalg.norm(c))
        if n > self.cfg.c_max_m:
            c = c * (self.cfg.c_max_m / n)
            self.n_clip_mag += 1
        self.c, self.s = c, s
        self.n_update += 1
        return s, self.c.copy(), {'why': None, 'e': e.tolist(), 'active': True}

    def ref_offset(self):
        """加到參考位姿平移上的量（世界 xy）＝最近有效權重 × 修正狀態。"""
        return (self.s * self.c).copy()


class ShiftedTable:
    """包裝 door_wide_guidance.RefTable：只把 pose() 的平移加上 offset；dpose_dq（前饋）、arm_nom 與其餘屬性原樣轉交。
    使既有 door_wide_guidance.guidance 不改一行即可使用修正後的參考位姿。"""

    def __init__(self, table, offset_xy=(0.0, 0.0)):
        self._t = table
        self.offset = np.asarray(offset_xy, float).reshape(2)

    def pose(self, q):
        pr = self._t.pose(q)
        return (float(pr[0]) + float(self.offset[0]), float(pr[1]) + float(self.offset[1]), float(pr[2]))

    def __getattr__(self, name):
        return getattr(self._t, name)
