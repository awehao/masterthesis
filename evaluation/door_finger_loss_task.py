#!/usr/bin/env python3
"""任務端讀取執行端逐物理步失接觸判定（Codex 20261029 extf_1 回覆；搭配 door_finger_loss_cont／ev9r8f --finger-loss-cont）。
  - ContactsAggregatorC：door_k2_io.ContactsAggregator 的子類，另併 finger_low_cont_s（取最新窗口）、finger_low_cont_max_s（未消費窗口取最大）、finger_loss_armed（最新）；
    缺欄位 ⇒ 記 NaN（不當成無低力）。原 finger_bar_min／max 照舊保存（報告用）。
  - cont_loss(Base)：策略 mixin，覆寫 door_k3a_policy._grasp_checks 的失接觸部分——不再以窗口最小力跨窗口累計時間，
    改讀 m.finger_low_cont_max（執行端逐步連續時長）：任一指 ≥ finger_loss_s ⇒ 中止 finger_contact_lost（附 cont 秒數）；欄位缺或非有限 ⇒ 中止 finger_loss_cont_missing。
    持握漂移判定（grasp_drift）照原規則。執行端已在超時當步走 K2 閂鎖；任務端此判定是同一來源的讀取，不另計時。
"""
from __future__ import annotations

import math

import door_k2_io as KIO
import door_k3a_policy as K3A

VERSION = 'door_finger_loss_task/r1'


class ContactsAggregatorC(KIO.ContactsAggregator):
    def on_msg(self, d, wall_now):
        prev = self.pending
        super().on_msg(d, wall_now)
        try:
            cur = [float(x) for x in d['finger_low_cont_s']]
            mx = [float(x) for x in d['finger_low_cont_max_s']]
            armed = bool(d['finger_loss_armed'])
        except (KeyError, TypeError, ValueError):
            cur, mx, armed = [math.nan] * 2, [math.nan] * 2, None
        p = self.pending
        if prev is not None and 'finger_low_cont_max_s' in prev:
            mx = [max(a, b) if math.isfinite(a) and math.isfinite(b) else math.nan for a, b in zip(prev['finger_low_cont_max_s'], mx)]
        p['finger_low_cont_s'], p['finger_low_cont_max_s'], p['finger_loss_armed'] = cur, mx, armed


def cont_loss(Base):
    class ContLoss(Base):
        def _grasp_checks(self, m):
            c = self.c
            dp, dr = K3A.grasp_drift(m.T_tcp, self.G_T_H, m.H)
            self.max_drift = (max(self.max_drift[0], dp), max(self.max_drift[1], dr))
            if dp > c.drift_abort_m or dr > c.drift_rot_abort_rad:
                self._abort('grasp_drift', m.t, drift=dp, rot=dr)
                return dp
            mx = getattr(m, 'finger_low_cont_max', None)
            if mx is None or len(mx) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in mx):
                self._abort('finger_loss_cont_missing', m.t, value=mx)
                return dp
            for k in range(2):
                if mx[k] >= c.finger_loss_s - 1e-9:
                    self._abort('finger_contact_lost', m.t, finger=k + 1, cont_s=mx[k], semantics='executor_per_step_continuous')
                    return dp
            return dp
    ContLoss.__name__ = f'ContLoss_{Base.__name__}'
    return ContLoss
