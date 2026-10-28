#!/usr/bin/env python3
"""夾爪主從（mimic）跟隨核對（純邏輯；Codex 20261029 核可單一驅動主從配置）：只驅動 leader（finger_joint1），follower（finger_joint2）
kp＝kd＝0、不下目標、由 NewtonMimicAPI 連動帶動。用**原閉爪斜坡中無接觸的部分**核對 follower 確有跟隨（不能只靠最後兩指碰到橫桿判定）。
  - 起點：手指命令首次低於張開值（閉爪斜坡開始）的那一步，記 q1₀、q2₀（呼叫端以 grip_closed＝命令 < 張開值傳入）。
  - 累積：之後每步有效樣本，兩指接觸力皆 ≤ f_contact（0.05 N）時記 max|q1−q2|；
  - 判定（只一次）：leader 行程 Δ1＝q1₀−q1 首次 ≥ min_travel（2 mm），且仍無接觸 ⇒ 通過需 |Δ2−Δ1| ≤ rel_tol·Δ1（20 %）且 max|q1−q2| ≤ abs_tol（0.5 mm）；
    若在 Δ1 達 min_travel 前已出現接觸、或資料無效 ⇒ 判「無法確認」失敗（不放行開門）。
回傳 None（尚未判定）／'pass'／'fail:<原因>'。
"""
from __future__ import annotations

import math

VERSION = 'door_mimic_follow/r1'


class MimicFollowCheck:
    def __init__(self, min_travel=0.002, rel_tol=0.2, abs_tol=0.0005, f_contact=0.05):
        self.min_travel, self.rel_tol, self.abs_tol, self.f_contact = float(min_travel), float(rel_tol), float(abs_tol), float(f_contact)
        self.q0 = None
        self.max_diff = 0.0
        self.result = None
        self.detail = {}

    def step(self, t, q1, q2, f1, f2, valid, grip_closed):
        if self.result is not None:
            return None
        if self.q0 is None:
            if grip_closed and all(x is not None and math.isfinite(float(x)) for x in (q1, q2)):
                self.q0 = (float(q1), float(q2), float(t))
            return None
        ok = bool(valid) and all(x is not None and math.isfinite(float(x)) for x in (q1, q2, f1, f2))
        if not ok:
            return self._done('fail:data_invalid', t)
        d1, d2 = self.q0[0] - float(q1), self.q0[1] - float(q2)
        contact = max(float(f1), float(f2)) > self.f_contact
        if contact:
            return self._done(f'fail:contact_before_travel(d1={d1 * 1e3:.3f}mm)', t, d1=d1, d2=d2)
        self.max_diff = max(self.max_diff, abs(float(q1) - float(q2)))
        if d1 >= self.min_travel:
            if abs(d2 - d1) <= self.rel_tol * d1 and self.max_diff <= self.abs_tol:
                return self._done('pass', t, d1=d1, d2=d2)
            return self._done(f'fail:not_following(d1={d1 * 1e3:.3f}mm,d2={d2 * 1e3:.3f}mm,max_diff={self.max_diff * 1e3:.3f}mm)', t, d1=d1, d2=d2)
        return None

    def _done(self, res, t, **kw):
        self.result = res
        self.detail = dict(kw, t=float(t), q0=list(self.q0) if self.q0 else None, max_diff=self.max_diff)
        return res

    def report(self):
        return {'version': VERSION, 'min_travel': self.min_travel, 'rel_tol': self.rel_tol, 'abs_tol': self.abs_tol, 'f_contact': self.f_contact,
                'result': self.result, 'detail': self.detail}
