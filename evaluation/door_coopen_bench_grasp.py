#!/usr/bin/env python3
"""台架初態啟動的任務端起始相位（mixin；Codex r19／r20；搭配執行端 ev9、求解端 ADM17、door_bench_start）。
bench_grasp(Base)：Base＝init_grasp2(gated2(K3MP2Policy))（INIT_CHECK 連續確認 r2 與監看擴充不變）；本層只改起始：
  (G1) 起始相位＝BENCH_START（取代 WAIT_HANDOVER；控制權自始為 wholebody，不經交棒、不經 UNFOLD；紀錄中不出現 WAIT_HANDOVER／UNFOLD／ALIGN／APPROACH）。
  (G2) BENCH_START：目標＝T(H, 0)、solver_start＝True（求解端需要目標才能在 t0 算出首解）；**不呼叫 monitor**——凍結期物理不演化、
       執行端不發接觸流，監看資料不存在；此限制記入 bench_stats.monitor_skipped_cycles（揭露，不冒充已監看）。
  (G3) 轉入 INIT_CHECK：m.bench_state＝'released' 且 m.t > t0（模擬時間已前進）且 T_tcp、H 有限 ⇒ 進入（之後全部監看照 init_grasp2）。
       m.bench_state＝'aborted' ⇒ 中止 bench_release_failed。其他 ⇒ 繼續等待（凍結期逾時由執行端判定並回報 aborted）。
m.bench_state／m.bench_t0 由任務節點 v7 從 /bench/state 填入 Meas（動態屬性）。
"""
from __future__ import annotations

import numpy as np

from door_coopen_init_grasp2 import PHASE as INIT_PHASE, _fin

PHASE = 'BENCH_START'


def bench_grasp(Base):
    class BenchGrasp(Base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.phase = PHASE
            self.bench_stats = {'monitor_skipped_cycles': 0, 'n_wait': 0, 't_release_seen': None, 't0': None}

        def step(self, m, dt):
            if self.phase != PHASE:
                return super().step(m, dt)
            if self.t_phase is None:
                self.t_phase = m.t
                self.events.append({'t': m.t, 'from': None, 'to': PHASE, 'bench': True})   # 起始相位事件（實際時刻；判定窗口起點）
            st, t0 = getattr(m, 'bench_state', None), getattr(m, 'bench_t0', None)
            self.bench_stats['monitor_skipped_cycles'] += 1
            act = self._base_act()
            if st == 'aborted':
                self._abort('bench_release_failed', m.t)
                return super().step(m, dt)
            if _fin(m.H):
                act['target'], act['solver_start'] = self.target(m.H, 0.0), True
            if st == 'released' and t0 is not None and m.t is not None and m.t > float(t0) + 1e-9 and _fin(m.T_tcp, m.H):
                self.bench_stats.update(t_release_seen=m.t, t0=float(t0))
                self._go(INIT_PHASE, m.t, bench_released=True)
                e = float(np.linalg.norm(m.T_tcp[:3, 3] - self.target(m.H, 0.0)[:3, 3]))
                self.init_stats.update(t_enter=m.t, err_enter_m=e)
                self.events.append({'t': m.t, 'init_check': 'enter', 'err_m': e, 'from_bench': True})
                self.settle_t0 = None
                act['init'] = {'valid': None, 'why': 'enter', 'err_p_m': e, 'err_r_rad': None, 'good': False, 'win_t0': None}
                return act
            self.bench_stats['n_wait'] += 1
            return act
    BenchGrasp.__name__ = f'BenchGrasp_{Base.__name__}'
    return BenchGrasp
