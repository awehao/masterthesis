#!/usr/bin/env python3
"""指定抓取初態的隔離驗證相位（mixin；Howard 2026-10-27 確認「先驗證核心動作」、Codex r15 補充裁定）。
用途：手臂以 --stow-q 直接生成在既有候選的門關閉抓取構型（coopen_core_h0_r4.json qa_grasp，手指張開），**跳過自主接近**；
以實際閉爪＋既有接觸證據建立持握，之後全部沿用原策略（ENGAGE → HOLD → OPEN → SETTLE → PAUSE…）。

init_grasp(Base) 回傳 Base 的子類，只改兩處：
  (I1) UNFOLD 完成（展開節點回報，m.unfold_done 且 T_tcp、H 有限）⇒ 轉入 **INIT_CHECK**（不經 ALIGN_PRE／ALIGN_HOLD／APPROACH，紀錄中不出現這三個相位）。
  (I2) INIT_CHECK：目標＝target(H, 0)（與 K2 APPROACH 終點同一式），求解照常啟動；
       位置誤差 < K2Config.grip_tol_m 連續 grip_settle_s ⇒ 凍結把手位姿（H_frozen）、記閉爪起點，轉入 ENGAGE（K2 原閉爪、接觸證據、G_T_H 量測）。
       誤差超限即清除持續窗。逾時沿用 cfg.timeouts['INIT_CHECK']（由節點自登錄設定；缺 ⇒ 建構時拒絕）。
  (I3) 監看：父類 monitor 以 door_k3a_policy.ACTIVE_MONITOR（K3A 與 K3M 皆讀此模組常數）決定是否執行，原集合不含 INIT_CHECK。
       本類只在 phase＝INIT_CHECK 呼叫 monitor 的期間把該集合暫時擴為「原集合＋INIT_CHECK」，呼叫後立即還原（不改原模組檔、不影響其他相位）。
       因此 INIT_CHECK 受與 APPROACH 相同的全部監看：門角／門速、意外接觸、牆、手指—橫桿（INIT_CHECK 不在 FINGER_BAR_ALLOWED ⇒
       閉爪前接觸 > contact_abort_n 即中止）、腕力、底盤漂移、求解失效、逾時。不放寬任何門檻、不固定附著、不偽造接觸證據。
紀錄：events 另記 init_check（進入時刻、TCP 位置誤差、結束時誤差）；init_stats 統計輪數與持續窗重置次數。
"""
from __future__ import annotations

import contextlib

import numpy as np

import door_k3a_policy as KAP

PHASE = 'INIT_CHECK'


@contextlib.contextmanager
def _monitored():
    orig = KAP.ACTIVE_MONITOR
    KAP.ACTIVE_MONITOR = tuple(orig) + (PHASE,)
    try:
        yield
    finally:
        KAP.ACTIVE_MONITOR = orig


def _fin(*xs):
    return all(x is not None and np.isfinite(np.asarray(x, float)).all() for x in xs)


def init_grasp(Base):
    class InitGrasp(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if PHASE not in self.c.timeouts:
                raise ValueError('init_grasp：cfg.timeouts 缺 INIT_CHECK')
            self.init_stats = {'n_cycles': 0, 'n_window_reset': 0, 't_enter': None, 't_engage': None, 'err_enter_m': None, 'err_engage_m': None}

        def monitor(self, m):                                                              # (I3)
            if self.phase != PHASE:
                return super().monitor(m)
            with _monitored():
                return super().monitor(m)

        def _base_act(self):
            return {'target': None, 'grip': self.grip_cmd, 'request_handover': False, 'solver_start': False, 'stop_solver': False, 'finish': False}

        def step(self, m, dt):
            if self.t_phase is None:
                self.t_phase = m.t
            if self.phase == 'UNFOLD' and m.unfold_done and _fin(m.T_tcp, m.H):              # (I1)
                self.monitor(m)
                if self.phase == 'ABORTED':
                    return super().step(m, dt)
                self._go(PHASE, m.t)
                e = float(np.linalg.norm(m.T_tcp[:3, 3] - self.target(m.H, 0.0)[:3, 3]))
                self.init_stats.update(t_enter=m.t, err_enter_m=e)
                self.events.append({'t': m.t, 'init_check': 'enter', 'err_m': e})
                self.settle_t0 = None
                act = self._base_act()
                act['target'], act['solver_start'] = self.target(m.H, 0.0), True
                return act
            if self.phase == PHASE:                                                          # (I2)
                self.monitor(m)
                if self.phase == 'ABORTED':
                    return super().step(m, dt)
                c = self.c
                self.init_stats['n_cycles'] += 1
                T = self.target(m.H, 0.0)
                act = self._base_act()
                act['target'], act['solver_start'] = T, True
                if _fin(m.T_tcp):
                    e0 = float(np.linalg.norm(m.T_tcp[:3, 3] - T[:3, 3]))
                    if e0 < c.grip_tol_m:
                        self.settle_t0 = self.settle_t0 or m.t
                        if m.t - self.settle_t0 >= c.grip_settle_s:
                            self.H_frozen = np.asarray(m.H, float).copy()                     # 閉爪開始即凍結把手位姿（同 K2）
                            self.close_t0 = m.t
                            self.init_stats.update(t_engage=m.t, err_engage_m=e0)
                            self.events.append({'t': m.t, 'init_check': 'pass', 'err_m': e0})
                            self._go('ENGAGE', m.t, err=e0)
                    else:
                        if self.settle_t0 is not None:
                            self.init_stats['n_window_reset'] += 1
                        self.settle_t0 = None
                else:
                    self.settle_t0 = None
                return act
            return super().step(m, dt)
    InitGrasp.__name__ = f'InitGrasp_{Base.__name__}'
    return InitGrasp
