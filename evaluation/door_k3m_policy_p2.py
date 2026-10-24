#!/usr/bin/env python3
"""K3-M 受控暫停 p2：只更換 SETTLE 完成判斷的門速來源（Codex 20261024_101421 (a)）。door_k3m_policy_p.py 以上不改。

差異只在 SETTLE 相位：
  - 角速子條件改看 m.settle_sig（door_k3m_settle_sig.SettleRateWindow.value()：50 Hz 同步流的 0.1 s 因果後向門角差分），
    |rate| ≤ settle_rate（0.002 rad/s）；valid=False（缺樣本、窗口不足、缺步、停更、步號不符、非有限）⇒ 不算低速、重設連續計時。
  - 其餘子條件（|q−8°|≤0.5°、抓取漂移、兩指接觸 ≥ contact_min_n）、連續 1 s、全部監看（含 tensor 門速 0.05 rad/s 保護、
    走廊、幾何、K2 閂鎖）與夾持檢查都照舊；m.qdot_door 不被覆寫。
  - 條件成立 ⇒ 與 p 版相同：轉 PAUSE、以最後一筆已發布目標（完整 4×4）提出暫停請求，握手不變。
SETTLE 分支是 K3APolicy.step 的 SETTLE 路徑逐行複本（監看 → 夾持檢查 → 目標 → 完成判斷），只換角速子條件。
"""
from __future__ import annotations

import math

import door_k3m_policy_p as PP
import door_k3a_policy as P

Meas = PP.Meas
K3MP2Config = PP.K3MPConfig


class K3MP2Policy(PP.K3MPPolicy):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.settle_last = None             # 本輪完成判斷：{'sig', 'angle_ok', 'rate_ok', 'drift_ok', 'contact_ok', 'ok', 'held_s'}
        self.settle_resets = []             # (t, 原因)：連續計時被重設

    def step(self, m: Meas, dt: float):
        if self.phase != 'SETTLE':
            return super().step(m, dt)
        c = self.c
        act = {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False,
               'stop_solver': False, 'finish': False}
        self.monitor(m)                                   # K3MPolicy.monitor：執行端違規、tensor 門速 0.05、走廊、幾何…
        if self.phase == 'ABORTED':
            return self._aborted_act(act, m)
        dp = self._grasp_checks(m)
        if self.phase == 'ABORTED':
            return self._aborted_act(act, m)
        T = self._ref_target(m, dt)
        if T is None:
            return self._aborted_act(act, m)
        act['target'] = T
        sig = getattr(m, 'settle_sig', None)
        valid = isinstance(sig, dict) and sig.get('valid') is True and isinstance(sig.get('rate'), float) and math.isfinite(sig['rate'])
        angle_ok = abs(m.q_door - self.tcfg.theta_goal_rad) <= math.radians(c.settle_tol_deg)
        rate_ok = valid and abs(sig['rate']) <= c.settle_rate
        drift_ok = dp <= c.drift_abort_m
        contact_ok = min(m.finger_bar_min) >= c.contact_min_n
        ok = angle_ok and rate_ok and drift_ok and contact_ok
        if ok:
            self.settle_ok_t0 = self.settle_ok_t0 if self.settle_ok_t0 is not None else m.t
        else:
            if self.settle_ok_t0 is not None:
                why = ('sig_invalid:' + str(None if not isinstance(sig, dict) else sig.get('why'))) if not valid else \
                      ('rate' if not rate_ok else 'angle' if not angle_ok else 'drift' if not drift_ok else 'contact')
                self.settle_resets.append((m.t, why))
            self.settle_ok_t0 = None
        held = 0.0 if self.settle_ok_t0 is None else m.t - self.settle_ok_t0
        self.settle_last = {'sig': sig, 'angle_ok': angle_ok, 'rate_ok': rate_ok, 'drift_ok': drift_ok,
                            'contact_ok': contact_ok, 'ok': ok, 'held_s': held}
        act['grip'] = self.grip_cmd
        self.last_target = T
        if ok and held >= c.settle_hold_s:
            self._go('END_HOLD', m.t, q=m.q_door, qdot=m.qdot_door, settle_sig=dict(sig))
            self._go('PAUSE', m.t)                        # 與 p 版相同：END_HOLD 當輪改轉 PAUSE
            self.pause_t0 = m.t
            req = self.pwaiter.make_request(int(getattr(m, 'state_step')), m.t, float(getattr(m, 'wall')),
                                            int(self.pub_seq), T)
            self.pause_outbox.append(req)
            self.pause_state = 'waiting'
            self.events.append({'t': m.t, 'k3m_pause_request': dict(req)})
        return act


assert P.OPENING == ('OPEN', 'SETTLE', 'END_HOLD')       # 父類模組常數不變（SETTLE 屬 OPENING／GRASP_PHASES）
