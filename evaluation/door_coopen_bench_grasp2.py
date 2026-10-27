#!/usr/bin/env python3
"""台架初態啟動的任務端起始相位 r2（door_coopen_bench_grasp.py 的版本化修正；Codex r21 §2；r1 保留不改）。
r1 缺口：BENCH_START 一律不監看、須收到 released 才離開 ⇒ 狀態通知遺失／延遲時，物理已前進仍跳過保護；轉入 INIT_CHECK 當輪也未監看。
r2：bench_grasp2(Base)，Base＝init_grasp2(gated2(K3MP2Policy))：
  (H1) 只有**確認仍凍結**（m.state_step＝0 且 m.t ≤ bench_t0；bench_t0 未知時以第一筆凍結樣本時刻為 t0）才跳過監看（計 monitor_skipped_cycles）。
  (H2) 物理已前進（步號 ≥ 1 或 m.t > t0）⇒ 每輪先執行監看（ACTIVE_MONITOR 暫時擴為原集合＋BENCH_START，try/finally 還原；同 init_grasp2 作法）；
       監看中止優先。之後：已收到經驗證的 released（任務節點 v8 以 door_bench_start2.StateFilter 驗本趟、t0、次序）⇒ 轉入 INIT_CHECK（本輪已監看）；
       否則自第一筆前進樣本起超過 cfg.stale_s（0.2 s）仍無釋放證明 ⇒ 中止 bench_release_unconfirmed（有界，不無監看等待）。
  (H3) bench_state＝aborted ⇒ 中止 bench_release_failed（任何時刻）。
  起始相位事件、目標 T(H,0)、solver_start 同 r1。
"""
from __future__ import annotations

import contextlib

import numpy as np

import door_k3a_policy as KAP
from door_coopen_init_grasp2 import PHASE as INIT_PHASE, _fin

PHASE = 'BENCH_START'


@contextlib.contextmanager
def _monitored():
    orig = KAP.ACTIVE_MONITOR
    KAP.ACTIVE_MONITOR = tuple(orig) + (PHASE,)
    try:
        yield
    finally:
        KAP.ACTIVE_MONITOR = orig


def bench_grasp2(Base):
    class BenchGrasp2(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.phase = PHASE
            self.bench_stats = {'monitor_skipped_cycles': 0, 'monitored_cycles': 0, 'n_wait': 0, 't_release_seen': None, 't0': None,
                                't_first_advanced': None}
            self._b_t0 = None

        def monitor(self, m):
            if self.phase != PHASE:
                return super().monitor(m)
            with _monitored():
                return super().monitor(m)

        def step(self, m, dt):
            if self.phase != PHASE:
                return super().step(m, dt)
            if self.t_phase is None:
                self.t_phase = m.t
                self.events.append({'t': m.t, 'from': None, 'to': PHASE, 'bench': True})
            st, t0 = getattr(m, 'bench_state', None), getattr(m, 'bench_t0', None)
            if t0 is not None:
                self._b_t0 = float(t0)
            act = self._base_act()
            if st == 'aborted':
                self._abort('bench_release_failed', m.t)
                return super().step(m, dt)
            if _fin(m.H):
                act['target'], act['solver_start'] = self.target(m.H, 0.0), True
            step = getattr(m, 'state_step', None)
            if self._b_t0 is None and step == 0 and m.t is not None:
                self._b_t0 = float(m.t)                                            # 第一筆凍結樣本
            frozen = step == 0 and self._b_t0 is not None and m.t is not None and float(m.t) <= self._b_t0 + 1e-9
            if frozen:                                                            # (H1)
                self.bench_stats['monitor_skipped_cycles'] += 1
                self.bench_stats['n_wait'] += 1
                return act
            self.bench_stats['monitored_cycles'] += 1                             # (H2)
            if self.bench_stats['t_first_advanced'] is None:
                self.bench_stats['t_first_advanced'] = m.t
            self.monitor(m)
            if self.phase == 'ABORTED':
                return super().step(m, dt)
            if st == 'released' and _fin(m.T_tcp, m.H):
                self.bench_stats.update(t_release_seen=m.t, t0=self._b_t0)
                self._go(INIT_PHASE, m.t, bench_released=True)
                e = float(np.linalg.norm(m.T_tcp[:3, 3] - self.target(m.H, 0.0)[:3, 3]))
                self.init_stats.update(t_enter=m.t, err_enter_m=e)
                self.events.append({'t': m.t, 'init_check': 'enter', 'err_m': e, 'from_bench': True})
                self.settle_t0 = None
                act['init'] = {'valid': None, 'why': 'enter', 'err_p_m': e, 'err_r_rad': None, 'good': False, 'win_t0': None}
                return act
            if m.t is not None and float(m.t) - float(self.bench_stats['t_first_advanced']) > self.c.stale_s:
                self._abort('bench_release_unconfirmed', m.t)
                return super().step(m, dt)
            self.bench_stats['n_wait'] += 1
            return act
    BenchGrasp2.__name__ = f'BenchGrasp2_{Base.__name__}'
    return BenchGrasp2
