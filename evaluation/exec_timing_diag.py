#!/usr/bin/env python3
"""執行端逐步牆鐘與 GC 計時診斷（Codex r29：只觀測；純邏輯、不依賴 Isaac）。
- StepTimer：預先配置的有界陣列（cap 列 × 標記數），每個主迴圈輪次一列：步號、模擬時刻、各標記的單調牆鐘（time.monotonic，
  與任務端 enq_wall_mono 同一系統時鐘，可跨行程配對）。輪次超過 cap ⇒ 停止記錄並計 n_overflow（不擴充、不配置）。
- GC：只在 gc.callbacks 掛一個回呼，記 start／stop、generation、collected、uncollectable、當下列號；有界（cap_gc），超過計數。
  **不呼叫 gc.collect／disable／freeze／set_threshold**，不改 GC 行為門檻。
- 任何診斷錯誤都吞掉並計數（n_err），不得影響控制；stop() 卸除回呼。
- 結束時 save(path) 寫 npz（原始陣列）；summary() 回傳可放進 room_run 的摘要：逐區段耗時分位數、最慢 20 輪的區段拆解、> 10 ms 的 GC。
區段定義（相鄰標記差；最後一段 = 下一輪起點 − 本輪最後標記）：見 MARKS／SECTIONS。

    python3 evaluation/test_exec_timing_diag.py
"""
from __future__ import annotations

import gc
import time

import numpy as np

MARKS = ('loop_start', 'after_callbacks', 'before_exec_step', 'after_exec_step', 'after_reads', 'after_evidence', 'before_world_step',
         'after_world_step')
SECTIONS = ('callbacks', 'command_processing', 'exec_step', 'contact_state_reads', 'evidence', 'record_publish_pre', 'world_step',
            'post_step_publish')
GC_SLOW_S = 0.010


class StepTimer:
    def __init__(self, cap=20000, cap_gc=50000, clock=time.monotonic):
        self.clock = clock
        self.cap, self.cap_gc = int(cap), int(cap_gc)
        self.T = np.full((self.cap, len(MARKS)), np.nan)
        self.meta = np.full((self.cap, 2), np.nan)                 # step_id, sim_t
        self.G = np.full((self.cap_gc, 6), np.nan)                 # wall, phase(0 start/1 stop), generation, collected, uncollectable, row
        self.row = -1
        self.n_gc = 0
        self.n_overflow = 0
        self.n_gc_overflow = 0
        self.n_err = 0
        self.active = True
        self._cb = None
        self._mi = {m: i for i, m in enumerate(MARKS)}

    def start_gc(self):
        def cb(phase, info):
            try:
                if not self.active:
                    return
                if self.n_gc >= self.cap_gc:
                    self.n_gc_overflow += 1
                    return
                self.G[self.n_gc] = (self.clock(), 0.0 if phase == 'start' else 1.0, info.get('generation', -1),
                                     info.get('collected', -1) if phase == 'stop' else -1,
                                     info.get('uncollectable', -1) if phase == 'stop' else -1, self.row)
                self.n_gc += 1
            except Exception:                                      # noqa: BLE001  診斷不得影響控制
                self.n_err += 1
        try:
            gc.callbacks.append(cb)
            self._cb = cb
        except Exception:                                          # noqa: BLE001
            self.n_err += 1

    def begin(self, step_id, sim_t):
        try:
            if self.row + 1 >= self.cap:
                self.n_overflow += 1
                return
            self.row += 1
            self.T[self.row, 0] = self.clock()
            self.meta[self.row] = (step_id, sim_t)
        except Exception:                                          # noqa: BLE001
            self.n_err += 1

    def mark(self, name):
        try:
            if 0 <= self.row < self.cap and self.n_overflow == 0:
                self.T[self.row, self._mi[name]] = self.clock()
        except Exception:                                          # noqa: BLE001
            self.n_err += 1

    def stop(self):
        self.active = False
        try:
            if self._cb is not None and self._cb in gc.callbacks:
                gc.callbacks.remove(self._cb)
        except Exception:                                          # noqa: BLE001
            self.n_err += 1

    def sections(self):
        n = self.row + 1
        T = self.T[:n]
        S = np.full((n, len(SECTIONS)), np.nan)
        S[:, :-1] = np.diff(T, axis=1)
        if n > 1:
            S[:-1, -1] = T[1:, 0] - T[:-1, -1]
        return S

    def gc_events(self):
        """配對 start／stop ⇒ [(wall_start, dur_s, generation, collected, row)]。"""
        out, pend = [], None
        for w, ph, gen, col, unc, row in self.G[:self.n_gc]:
            if ph == 0.0:
                pend = (w, gen, row)
            elif pend is not None:
                out.append((float(pend[0]), float(w - pend[0]), int(pend[1]), int(col), int(pend[2])))
                pend = None
        return out

    def save(self, path):
        try:
            np.savez_compressed(path, T=self.T[:self.row + 1], meta=self.meta[:self.row + 1], G=self.G[:self.n_gc],
                                marks=np.array(MARKS), sections=np.array(SECTIONS))
            return True
        except Exception:                                          # noqa: BLE001
            self.n_err += 1
            return False

    def summary(self, top=20):
        try:
            S = self.sections()
            n = len(S)
            tot = np.nansum(S, axis=1) if n else np.zeros(0)
            idx = np.argsort(-np.nan_to_num(tot))[:top] if n else []
            ev = self.gc_events()
            q = {}
            for j, s in enumerate(SECTIONS):
                v = S[:, j][np.isfinite(S[:, j])]
                q[s] = None if not len(v) else {'p50': float(np.percentile(v, 50)), 'p99': float(np.percentile(v, 99)), 'max': float(v.max()),
                                                'argmax_step': float(self.meta[int(np.nanargmax(np.where(np.isfinite(S[:, j]), S[:, j], -1))), 0])}
            return {'n_rows': n, 'n_overflow': self.n_overflow, 'n_gc_events': len(ev), 'n_gc_overflow': self.n_gc_overflow, 'n_err': self.n_err,
                    'marks': list(MARKS), 'sections': list(SECTIONS), 'clock': 'time.monotonic（與任務端 enq_wall_mono 同系統時鐘）',
                    'section_quantiles_s': q,
                    'slowest_rows': [{'step': float(self.meta[i, 0]), 'sim_t': float(self.meta[i, 1]), 'wall_start': float(self.T[i, 0]),
                                      'total_s': float(tot[i]), 'sections_s': {s: (None if not np.isfinite(S[i, j]) else float(S[i, j]))
                                                                               for j, s in enumerate(SECTIONS)}} for i in idx],
                    'gc_slow': [{'wall_start': w, 'dur_s': d, 'generation': g, 'collected': c, 'row': r,
                                 'step': (None if not (0 <= r < n) else float(self.meta[r, 0]))} for w, d, g, c, r in ev if d > GC_SLOW_S],
                    'gc_by_generation': {str(g): {'n': sum(1 for e in ev if e[2] == g), 'max_s': max((e[1] for e in ev if e[2] == g), default=None)}
                                         for g in (0, 1, 2)}}
        except Exception as e:                                     # noqa: BLE001
            self.n_err += 1
            return {'error': type(e).__name__, 'n_err': self.n_err}
