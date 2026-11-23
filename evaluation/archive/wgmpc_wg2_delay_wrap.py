"""以**注入求解延遲**跑實際的 WG2 節點（測試用外殼）。

延遲包在 `wgmpc_core.solve` 外面，所以它落在「求解返回」之前 ——
與真實的長求解等效：**單一 executor 在這段期間不處理回呼**，
返回時佇列裡會積著 /clock、/joint_states、/odom 等訊息。

延遲是 **CPU-bound 忙迴圈**；`time.sleep` 會釋放 GIL，雖然本節點是
單執行緒、沒有背景 executor，但用忙迴圈才與真實計算一致。

    COMAN_WG2_DELAY_MS="300"   # 逐次循環套用
"""
from __future__ import annotations
import itertools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src/ammr_wholebody_mpc'))

import ammr_wholebody_mpc.wgmpc_core as C   # noqa: E402

_ms = [float(x) for x in os.environ.get('COMAN_WG2_DELAY_MS', '0').split(',')
       if x.strip()]
_cyc = itertools.cycle(_ms or [0.0])
_orig = C.solve


def _slow(*a, **kw):
    r = _orig(*a, **kw)
    d = next(_cyc) / 1e3
    if d > 0:
        t_end = time.monotonic() + d
        x = 0
        while time.monotonic() < t_end:
            for _ in range(2000):
                x += 1
        # 把延遲也算進核心回報的 total，否則下游看不到真實耗時
        try:
            r.timing_ms = dict(r.timing_ms)
            r.timing_ms['total'] = round(r.timing_ms.get('total', 0.0)
                                         + d * 1e3, 4)
        except Exception:      # noqa: BLE001
            pass
    return r


C.solve = _slow
import wgmpc_wg2_node as NODE          # noqa: E402
NODE.solve = _slow                      # 節點是 from-import，要一併換
print(f'[wrap] 注入求解延遲 {_ms} ms（循環）', flush=True)
raise SystemExit(NODE.main())
