#!/usr/bin/env python3
"""接近段成本切換（純邏輯；Codex 20261027_011742／r13：登錄 r7 approach_cost，單一候選 w_a 0.01）。
任務節點每輪在 door_wide_guidance.phase_coord 之後呼叫 ApproachCost.apply(coord, phase, t)：
  - 啟用條件：phase ∈ cfg.phases（r7：只有 APPROACH）且未閂鎖，且本輪引導關閉（coord 與 off_coord 的 w_vref／w_qn 皆 0）⇒ coord[INDEX]＝w_a；
  - 其他 ⇒ coord 原樣（off_coord 的 [12] 為 NaN ⇒ 求解端沿用啟動值）。
  - 閂鎖：曾啟用、之後任一輪 phase 不在 cfg.phases ⇒ exited；此後永不再送候選（相位回跳也不重啟）。
  - 引導開啟中（w_vref 或 w_qn ≠ 0）卻遇到啟用相位 ⇒ 不覆寫，記 conflict（防禦；r7 的 APPROACH 不在 GUIDE_PHASES，正常不會發生）。
coord 版面同 dl3_wgmpc_wg2_adm9_node.parse_coord v1：[1, w_vref, vx, vy, wz, w_qn, q×6, w_a, w_p]。
"""
from __future__ import annotations

import math

INDEX = 12
COORD_LEN = 14
KEYS = {'w_a', 'phases', 'latch_after_exit', 'coord_index', 'launch_w_a', 'restore_deadline_s'}


class ApproachCost:
    def __init__(self, reg: dict):
        if not isinstance(reg, dict) or set(reg) != KEYS:
            raise ValueError(f'approach_cost 鍵不符：{sorted(reg) if isinstance(reg, dict) else reg}')
        w = float(reg['w_a'])
        if not (math.isfinite(w) and w > 0.0):
            raise ValueError(f'approach_cost.w_a 必須為正有限值：{w}')
        if int(reg['coord_index']) != INDEX or reg['latch_after_exit'] is not True:
            raise ValueError('approach_cost：coord_index 須為 12、latch_after_exit 須為 True')
        self.w_a = w
        self.phases = tuple(str(p) for p in reg['phases'])
        self.reg = dict(reg)
        self.active = False
        self.exited = False
        self.t_first_on = None
        self.t_exit = None
        self.exit_phase = None
        self.n_on = 0
        self.n_off = 0
        self.n_conflict = 0

    def apply(self, coord, phase, t):
        """回傳 (coord_out, 本輪紀錄)。coord_out 為新 list；輸入不改。"""
        c = [float(x) for x in coord]
        if len(c) != COORD_LEN:
            raise ValueError(f'coord 長度 {len(c)} 不是 {COORD_LEN}')
        want = str(phase) in self.phases and not self.exited
        if self.active and not want:
            self.active = False
            self.exited = True
            self.t_exit, self.exit_phase = float(t), str(phase)
        if want:
            if c[1] != 0.0 or c[5] != 0.0:
                self.n_conflict += 1
                self.n_off += 1
                return c, {'w_a': None, 'on': False, 'why': 'guidance_on'}
            if not self.active:
                self.active = True
                if self.t_first_on is None:
                    self.t_first_on = float(t)
            c[INDEX] = self.w_a
            self.n_on += 1
            return c, {'w_a': self.w_a, 'on': True, 'why': None}
        self.n_off += 1
        return c, {'w_a': None if math.isnan(c[INDEX]) else c[INDEX], 'on': False, 'why': 'latched' if self.exited else 'phase'}

    def stats(self):
        return {'t_first_on': self.t_first_on, 't_exit': self.t_exit, 'exit_phase': self.exit_phase, 'exited': self.exited,
                'n_on': self.n_on, 'n_off': self.n_off, 'n_conflict': self.n_conflict}
