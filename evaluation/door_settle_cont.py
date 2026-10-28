#!/usr/bin/env python3
"""SETTLE 接觸准入改用逐物理步連續低承載語義（Codex 20261029_053702 核可 (A)；搭配 node10d --settle-cont、ev9r8j --finger-loss-cont）。
**這是明確放寬 SETTLE 接觸准入的時間語義**：容許短暫低承載，不再要求整個窗口每一步雙指皆 ≥ contact_min_n；不是原判準通過，也不表示持握已穩定。
  - 原 p2（door_k3m_policy_p2.K3MP2Policy）SETTLE 的接觸子條件是 min(m.finger_bar_min) ≥ c.contact_min_n（窗口最小值）。
  - 本 mixin 只在 SETTLE 相位、只換這一個子條件：contact_ok ⇔ 逐步失接觸摘要**有效、新鮮、已 armed**，且兩指目前連續低力時長皆 < finger_loss_s。
      有效：m.finger_low_cont（目前值）與 m.finger_low_cont_max（未消費窗口最大）皆為兩個有限數；
      新鮮：m.contacts_age 有限且 ≤ c.stale_s；armed：m.finger_loss_armed is True。
    缺資料、未 armed、無效值 ⇒ contact_ok＝False（不當成「沒有失接觸」），照 p2 規則重設 SETTLE 連續計時。
    任一指連續低於 contact_min_n 達 finger_loss_s ⇒ 仍由 cont_loss mixin 照既有規則中止 finger_contact_lost（本模組不改、不清除執行端累積時間）。
  - 實作：SETTLE 當輪暫把 self.c 換成 contact_min_n＝−inf（准入成立）或 +inf（不成立）的複本呼叫原 step，結束後還原；
    SETTLE 相位內 contact_min_n 只被 p2 第 50 行使用（_grasp_checks 由 cont_loss 覆寫、K3A 原 SETTLE 分支不經過），其餘門角、門速、漂移、保持時間、
    監看與後續停止觀察／gate 2／幾何／腕力保護全部照舊。
  - settle_last 另記 contact_semantics、contact_detail（cont、cont_max、armed、contacts_age、窗口最小值原值），供紀錄標明准入語義。
"""
from __future__ import annotations

import dataclasses
import math

VERSION = 'door_settle_cont/r1'
SEMANTICS = 'executor_per_step_continuous_low_load<finger_loss_s(valid,fresh,armed)'


def _pair(v):
    return v is not None and len(v) == 2 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)


def settle_contact_ok(m, c):
    cur, mx = getattr(m, 'finger_low_cont', None), getattr(m, 'finger_low_cont_max', None)
    armed, age = getattr(m, 'finger_loss_armed', None), getattr(m, 'contacts_age', None)
    fresh = isinstance(age, (int, float)) and math.isfinite(age) and age <= c.stale_s
    valid = _pair(cur) and _pair(mx)
    ok = bool(valid and fresh and armed is True and max(cur) < c.finger_loss_s - 1e-9 and max(mx) < c.finger_loss_s - 1e-9)
    why = None if ok else ('invalid' if not valid else 'stale' if not fresh else 'not_armed' if armed is not True else 'low_load_cont')
    return ok, {'cont': None if cur is None else list(cur), 'cont_max': None if mx is None else list(mx), 'armed': armed,
                'contacts_age': age, 'finger_bar_min': None if m.finger_bar_min is None else list(m.finger_bar_min), 'why': why}


def settle_cont(Base):
    class SettleCont(Base):
        def step(self, m, dt):
            if self.phase != 'SETTLE':
                return super().step(m, dt)
            ok, det = settle_contact_ok(m, self.c)
            c0, sl0 = self.c, getattr(self, 'settle_last', None)
            self.c = dataclasses.replace(c0, contact_min_n=-math.inf if ok else math.inf)
            try:
                act = super().step(m, dt)
            finally:
                self.c = c0
            sl = getattr(self, 'settle_last', None)
            if isinstance(sl, dict) and sl is not sl0:                 # 只標本輪 p2 新寫的判斷（中止提前返回時不改舊紀錄）
                sl['contact_semantics'], sl['contact_detail'] = SEMANTICS, det
            return act
    SettleCont.__name__ = f'SettleCont_{Base.__name__}'
    return SettleCont
