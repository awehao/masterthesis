#!/usr/bin/env python3
"""台架初態起始相位 r3（Codex 20261029_054218 核可；r2 door_coopen_bench_grasp2 保留不改）：只補「首筆接觸窗口晚於狀態」的有界等待。
r2 缺口（leader_sc_1）：BENCH_START 物理一前進就完整監看，首筆接觸窗口尚未送達（contacts_age＝None）⇒ 立即 monitor_stale_or_missing。
r3：bench_grasp3(Base) 外包在 bench_grasp2 之外：
  (W1) 只在 BENCH_START、物理已前進（非 r2 (H1) 凍結樣本）、且**從未收到過接觸窗口**（m.contacts_age is None）時等待：
       保持原目標 T(H,0)、不轉 INIT_CHECK、不閉爪、不累積准入時間；不呼叫完整 monitor，**不把缺接觸資料補成零接觸**。
  (W2) 等待期限自首次物理前進樣本起固定計時：模擬時間 > cfg.stale_s（0.2 s）或牆鐘 > WAIT_WALL_MAX_S（1.0 s）⇒ 中止 bench_contacts_unconfirmed。
  (W3) 等待中仍檢查可取得者：執行端違規、狀態新鮮度（state_age ≤ stale_s）、門角／門速、底盤漂移（未切換執行端時；同 K3A 規則）、求解端失敗／狀態過期、
       bench_state＝aborted（交回 r2 走 bench_release_failed）；執行端原保護照常運作。
  (W4) 收到首筆接觸窗口後永久交回 r2：當輪即完整監看（含接觸有效／新鮮），監看通過才可能轉 INIT_CHECK；之後不得再享等待期，
       遺失／無效／過期照原規則處理。
  紀錄：bench_wait＝{'version','t0','wall0','n_wait','t_contacts_first','wall_max_s','sim_max_s'}。
"""
from __future__ import annotations

import math

from door_coopen_bench_grasp2 import PHASE, bench_grasp2
from door_coopen_init_grasp2 import _fin

VERSION = 'door_coopen_bench_grasp3/r1'
WAIT_WALL_MAX_S = 1.0


def bench_grasp3(Base):
    B2 = bench_grasp2(Base)

    class BenchGrasp3(B2):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.bench_wait = {'version': VERSION, 't0': None, 'wall0': None, 'n_wait': 0, 't_contacts_first': None,
                               'wall_max_s': WAIT_WALL_MAX_S, 'sim_max_s': float(self.c.stale_s)}
            self._ct_seen = False

        def _wait_checks(self, m):
            c = self.c
            if getattr(m, 'executor_violation', None) is not None:
                return self._abort('executor_violation', m.t, detail=m.executor_violation)
            if not _fin(m.state_age) or m.state_age > c.stale_s:
                return self._abort('monitor_stale_or_missing', m.t, state_age=m.state_age, contacts_age=None, bench_wait=True)
            if not _fin(m.q_door, m.qdot_door, m.H, m.T_tcp, m.base_drift_m, m.base_drift_yaw):
                return self._abort('monitor_nonfinite_or_invalid', m.t, bench_wait=True)
            if abs(m.q_door) > math.radians(c.door_abort_deg):
                return self._abort('door_moving', m.t, q_deg=math.degrees(m.q_door), qdot=m.qdot_door)
            if abs(m.qdot_door) > c.door_abort_rate:
                return self._abort('door_rate', m.t, q_deg=math.degrees(m.q_door), qdot=m.qdot_door)
            if not getattr(self, 'exec_switched', False) and (m.base_drift_m > c.base_drift_m or abs(m.base_drift_yaw) > c.base_drift_yaw_rad):
                return self._abort('base_drift', m.t, d=m.base_drift_m, yaw=m.base_drift_yaw)
            if m.solver_failed:
                return self._abort('solver_failed', m.t, detail=m.solver_failed)
            if m.solver_up and (m.solver_status_age is None or m.solver_status_age > c.solver_stale_s):
                return self._abort('solver_status_stale', m.t, age=m.solver_status_age)

        def step(self, m, dt):
            if self.phase != PHASE or self._ct_seen:
                return super().step(m, dt)
            if m.contacts_age is not None:                                       # (W4) 首筆窗口：永久交回 r2，當輪完整監看
                self._ct_seen = True
                self.bench_wait['t_contacts_first'] = m.t
                return super().step(m, dt)
            if getattr(m, 'bench_state', None) == 'aborted':
                return super().step(m, dt)                                        # r2 (H3)
            t0 = getattr(m, 'bench_t0', None)
            if t0 is not None:
                self._b_t0 = float(t0)
            step = getattr(m, 'state_step', None)
            if step == 0 and (self._b_t0 is None or (m.t is not None and float(m.t) <= self._b_t0 + 1e-9)):
                return super().step(m, dt)                                        # r2 (H1) 凍結樣本：照原樣跳過監看
            w = self.bench_wait
            if w['t0'] is None:
                w['t0'], w['wall0'] = float(m.t), float(getattr(m, 'wall'))
                self.events.append({'t': m.t, 'bench_wait': 'contacts_first_pending'})
            if self.bench_stats['t_first_advanced'] is None:
                self.bench_stats['t_first_advanced'] = m.t                       # r2 釋放確認期限仍自首次前進起算
            self._wait_checks(m)
            if self.phase == 'ABORTED':
                return super().step(m, dt)
            if float(m.t) - w['t0'] > w['sim_max_s'] or float(getattr(m, 'wall')) - w['wall0'] > w['wall_max_s']:
                self._abort('bench_contacts_unconfirmed', m.t, sim_wait=float(m.t) - w['t0'], wall_wait=float(getattr(m, 'wall')) - w['wall0'])
                return super().step(m, dt)
            w['n_wait'] += 1
            act = self._base_act()
            if _fin(m.H):
                act['target'], act['solver_start'] = self.target(m.H, 0.0), True
            return act
    BenchGrasp3.__name__ = f'BenchGrasp3_{Base.__name__}'
    return BenchGrasp3
