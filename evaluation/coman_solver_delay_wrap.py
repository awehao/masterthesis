"""以**注入求解延遲**的方式跑實際的求解端節點（測試用外殼）。

延遲包在 `WholeBody.solve` 外面，所以它**被計入 `_solve_ms`**，
與 main5 的 301／256 ms 等效。生產程式碼未加任何測試鉤子 ——
這裡用 monkeypatch，而且是在 `build()` 取用基底類別**之前**完成。

    COMAN_SOLVE_DELAY_MS="300,260"   # 逐次循環套用

延遲是 **CPU-bound 忙迴圈**（持有 GIL），不是 `time.sleep` ——
`sleep` 釋放 GIL，背景 executor 執行緒照跑，重現不出真實長求解的條件。
"""
from __future__ import annotations
import itertools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import coman_pull_solver_node as P   # noqa: E402

_ms = [float(x) for x in os.environ.get('COMAN_SOLVE_DELAY_MS', '0').split(',')
       if x.strip()]
_cyc = itertools.cycle(_ms or [0.0])
_orig_load = P.load_base


def _load_patched():
    M = _orig_load()
    _solve = M.WholeBody.solve

    def slow(self, T_des):
        out = _solve(self, T_des)
        d = next(_cyc) / 1e3
        if d > 0:
            # **CPU-bound、持有 GIL** —— `time.sleep()` 會**釋放** GIL，
            # 背景執行緒（TransformListener 的專用 executor 在轉本節點）
            # 照樣處理回呼，那就重現不出真實長求解的條件。
            # 真實的 `_constraints` 是 Python 迴圈組 1370 列，整段持有 GIL。
            t_end = time.monotonic() + d
            x = 0
            while time.monotonic() < t_end:
                for _ in range(2000):
                    x += 1              # 純 Python，不放 GIL
        return out

    M.WholeBody.solve = slow
    return M


P.load_base = _load_patched
print(f'[wrap] 注入求解延遲 {_ms} ms（循環）', flush=True)
raise SystemExit(P.main())
