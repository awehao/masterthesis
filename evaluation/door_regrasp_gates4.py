#!/usr/bin/env python3
"""HoldConfirm r4（leader_sc_3 hold_timeout；比照 Codex 20261029_053702 已核可的 SETTLE 逐步連續低承載准入語義；gates3 保留不改）。
gates3.HoldConfirm 的「兩指各自 bar_in ≥ contact_min_n」為逐步嚴格條件：finger1 單一物理步瞬間 < 0.5 N（leader_sc_3：25／454 步，每次 0.01 s）
即重設 2.0 s 連續窗 ⇒ hold_timeout；mode 7 中確認後同樣會因單步掉力判 hold_lost_after_confirm。
r4 只換這一個子條件（**明確放寬時間語義；非原判準通過，不表示持握穩定**）：
  （Codex 20261029_074108 核可語義）
  - 正向接觸證據先行：E0 仍只由首個有效且兩指 bar_in 皆 ≥ contact_min_n、other 成立的樣本建立（嚴格，只設一次）；E0 建立前 held＝原嚴格條件。
  - E0 建立後：每指逐物理步記 low_t0（bar_in < contact_min_n 的連續起點；≥ 即清除）；held ⇔ 兩指皆（≥ contact_min_n 或 t − low_t0 < finger_loss_s〔0.2 s〕）。
  - 確認前連續低力達 0.2 s ⇒ held False ⇒ 重設確認窗（不重設 E0、不重設 4.5 s 總期限）。
  - 確認後及整段 mode 7（同一物件延續、mode 6→7 不清計時）：連續低力達 0.2 s ⇒ 失去確認 ⇒ FAILED hold_lost_after_confirm；單步低力不再失去確認。
  - 缺步／無效／過期照 gates3 _seq／_invalid（在 held 計算前返回，不當作恢復接觸）；low_t0 不因無效步清除。
  其他類別 ≤ 0.1 N、非手指無接觸、E0 基準（第一個 held 且 other 成立步）、漂移 5 mm／2°、連續 attach_hold_s 2.0 s、t_hold_max_s 4.5 s、
  確認後失去 ⇒ hold_lost_after_confirm——全部同 gates3。另記 n_low_tolerated（以短暫低承載放行的步數）與 low_max_s。
"""
from __future__ import annotations

import numpy as np

import door_regrasp_gates3 as GT
from door_regrasp_gates3 import ALL, FINGERS, finger_force, link_class_force, nonfinger_force, rot_angle

VERSION = 'door_regrasp_gates4/r1'
SEMANTICS = 'per_step_continuous_low_load<finger_loss_s(after_strict_E0)'


class HoldConfirmCont(GT.HoldConfirm):
    def __init__(self, *a, finger_loss_s=0.2, **kw):
        super().__init__(*a, **kw)
        self.finger_loss_s = float(finger_loss_s)
        self._low_t0 = {f: None for f in FINGERS}
        self.n_low_tolerated = 0
        self.low_max_s = 0.0

    def start(self, *a, **kw):
        self._low_t0 = {f: None for f in FINGERS}
        return super().start(*a, **kw)

    def _held_cont(self, lcf, t):
        cmin = self.r['contact_min_n']
        if getattr(self, 'E0', None) is None:                               # 基準前：原嚴格條件
            return all(finger_force(lcf, f, ('bar_in',)) >= cmin for f in FINGERS)
        ok, tolerated = True, False
        for f in FINGERS:
            fb = finger_force(lcf, f, ('bar_in',))
            if fb >= cmin:
                self._low_t0[f] = None
                continue
            if self._low_t0[f] is None:
                self._low_t0[f] = float(t)
            dur = float(t) - self._low_t0[f]
            self.low_max_s = max(self.low_max_s, dur)
            if dur < self.finger_loss_s - 1e-9:
                tolerated = True
            else:
                ok = False
        if ok and tolerated:
            self.n_low_tolerated += 1
        return ok

    def step(self, step, t, T_tcp, H_in, rec):                              # gates3.HoldConfirm.step 逐行複本，只換 held
        go, missing = self._seq(step, t)
        if not go:
            return self.state
        if self.state != 'CONFIRMED' and float(t) - self.t['start'] > self.r['t_hold_max_s'] + 1e-9:
            return self._fail('hold_timeout', t, step)
        lcf = link_class_force(rec, step, t, self.r['time_tol'])
        try:
            T, Hm = np.asarray(T_tcp, float), np.asarray(H_in, float)
            okv = T.shape == (4, 4) and Hm.shape == (4, 4) and np.all(np.isfinite(T)) and np.all(np.isfinite(Hm))
        except (TypeError, ValueError):
            okv = False
        if missing or lcf is None or not okv:
            return self._invalid(step, t)
        E = np.linalg.inv(Hm) @ T
        held = self._held_cont(lcf, t)
        other = all(finger_force(lcf, f, tuple(c for c in ALL if c != 'bar_in')) <= self.r['unexpected_max_n'] for f in FINGERS) \
            and nonfinger_force(lcf) <= self.r['unexpected_max_n']
        if held and other and getattr(self, 'E0', None) is None:
            self.E0 = E.copy()
            self.events.append({'t': t, 'step': step, 'baseline': True})
        drift_ok = True
        if getattr(self, 'E0', None) is not None:
            dp = float(np.linalg.norm(Hm[:3, :3] @ (E[:3, 3] - self.E0[:3, 3])))
            dr = rot_angle(self.E0[:3, :3], E[:3, :3])
            drift_ok = dp <= self.r['drift_m'] and dr <= self.r['drift_rad']
            self.last_drift = (dp, dr)
        ok = held and other and drift_ok
        was = self.state
        st = self._hold(ok, step, t)
        if was == 'CONFIRMED' and st == 'WAITING':
            return self._fail('hold_lost_after_confirm', t, step)
        return st

    def report(self):
        r = super().report()
        r = dict(r) if isinstance(r, dict) else {'base': r}
        r['cont'] = {'version': VERSION, 'semantics': SEMANTICS, 'finger_loss_s': self.finger_loss_s, 'n_low_tolerated': self.n_low_tolerated,
                     'low_max_s': self.low_max_s, 'E0_set': getattr(self, 'E0', None) is not None}
        return r
