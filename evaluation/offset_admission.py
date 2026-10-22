#!/usr/bin/env python3
"""手臂偏移估計 d̂ 的更新準入（純邏輯；**尚未接入求解節點**，dl3_wgmpc_wg2_node.py／offset_moving.py 不改）。

依 Codex 20261022 door_k3a_1 診斷 v2 複核：OPEN 期間有 7 次「運動觀測器判為非等速、不給觀測，static 分支仍更新 d̂」。
本模組把求解節點的更新規則抽出成可離線測試的類別，兩種準入：

- admission='legacy'（預設）：**逐位元**照 dl3_wgmpc_wg2_node.py 1157–1193 行：
    設定點逐週期變化 < gate ⇒ static 更新 d̂ ← d̂ + (dt/τ)((q − s) − d̂)；
    否則運動觀測器准入 ⇒ moving 更新（d_obs）；兩者皆否 ⇒ 不更新。限幅 ±max。
- admission='consistent'：**只有運動觀測器准入**才更新（觀測＝同時刻 q − s 加速度補償 k·ṡ；靜止時 ṡ＝0 ⇒ 補償為零，
  與 static 觀測相同）；觀測器未准入（非等速、窗口未滿、缺資料、時間不前進）⇒ **保持 d̂**，不由 static 繞過。
  需要運動觀測器（moving=True）。
init_static（第一輪以 q − s 為初值）兩模式相同。輸入非有限 ⇒ consistent 保持並記錄；legacy 照原式（呼叫端負責）。
"""
from __future__ import annotations

import math

import numpy as np

from offset_moving import MovingCfg, MovingOffsetObserver


class OffsetEstimator:
    def __init__(self, alpha, phys_dt, dt, tau_s=0.5, gate_rad=0.002, max_rad=0.05, init_static=False, moving=True,
                 admission='legacy', moving_cfg: MovingCfg | None = None):
        if admission not in ('legacy', 'consistent'):
            raise ValueError(f'admission 不明：{admission}')
        if admission == 'consistent' and not moving:
            raise ValueError("admission='consistent' 需要運動觀測器（moving=True）")
        if not (dt > 0 and tau_s > 0 and gate_rad > 0 and max_rad > 0):
            raise ValueError('dt／tau／gate／max 須為正')
        self.alpha = np.asarray(alpha, float).reshape(6)
        self.dt, self.tau, self.gate, self.max = float(dt), float(tau_s), float(gate_rad), float(max_rad)
        self.init_static = bool(init_static)
        self.admission = admission
        self.obs = MovingOffsetObserver(moving_cfg or MovingCfg(), self.alpha, phys_dt) if moving else None
        self.d = np.zeros(6)
        self.s_prev = None
        self.n_upd = 0
        self.n_upd_moving = 0
        self.n_held = 0
        self.init = None

    @property
    def bias(self):
        return self.alpha * self.d

    def update(self, t, s, q):
        """一個求解週期。s＝同步量測的設定點、q＝同時刻實測手臂角（皆 6 軸）。回傳紀錄 dict。"""
        rec = {'used': 'none', 'why': None, 'om': None}
        s_ok = q_ok = False
        try:
            s_now = np.asarray(s, float).reshape(6)
            q_now = np.asarray(q, float).reshape(6)
            s_ok, q_ok = bool(np.isfinite(s_now).all()), bool(np.isfinite(q_now).all())
        except (TypeError, ValueError):
            s_now = q_now = None
        if self.admission == 'consistent' and not (s_ok and q_ok and t is not None and math.isfinite(float(t))):
            self.n_held += 1
            rec.update(used='held', why='missing_or_nonfinite', d_hat_after=self.d.tolist())
            return rec
        e = q_now - s_now
        if self.init_static and self.n_upd == 0 and self.s_prev is None:
            self.d = np.clip(e, -self.max, self.max)
            self.init = self.d.tolist()
        gate_ok = self.s_prev is not None and float(np.max(np.abs(s_now - self.s_prev))) < self.gate
        om = None if self.obs is None else self.obs.observe(t, s_now, q_now)
        rec['om'] = om
        if self.admission == 'legacy':
            if gate_ok:
                self.d += (self.dt / self.tau) * (e - self.d)
                self.n_upd += 1
                rec['used'] = 'static'
            elif om is not None and om['ok']:
                self.d += (self.dt / self.tau) * (np.asarray(om['d_obs'], float) - self.d)
                self.n_upd += 1
                self.n_upd_moving += 1
                rec['used'] = 'moving'
        else:
            if om is not None and om['ok']:
                self.d += (self.dt / self.tau) * (np.asarray(om['d_obs'], float) - self.d)
                self.n_upd += 1
                self.n_upd_moving += 1
                rec['used'] = 'admitted'
            else:
                self.n_held += 1
                rec['used'] = 'held'
                rec['why'] = None if om is None else om.get('why')
        self.d = np.clip(self.d, -self.max, self.max)
        self.s_prev = s_now.copy()
        rec['gate_ok'] = gate_ok
        rec['d_hat_after'] = self.d.tolist()
        return rec
