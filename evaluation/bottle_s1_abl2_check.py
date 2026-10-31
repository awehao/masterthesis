#!/usr/bin/env python3
"""消融「無在途狀態前推」最小驗收：關閉時命令同凍結 G；開啟時求解初態為當下狀態、安全停止核對仍用佇列；名目離線一次。假受控對象。
    python3 evaluation/bottle_s1_abl2_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_groups_check as CK                     # noqa: E402
import test_bottle_s0b as T                             # noqa: E402


def main():
    c = SC.make_lidar(no_inflight=True)
    seen, rest = [], []
    so, sr = c.mpc.solve, c._stop_rest

    def wrap(*a, **k):
        arm = k.get('arm') or {}
        if arm.get('mode') == 'active':
            seen.append((float(np.abs(np.asarray(arm['q_arm']) - c.q_now[3:]).max()), float(np.abs(np.asarray(arm['s_arm']) - c.s).max())))
        return so(*a, **k)

    def wrap_r(q):
        out = sr(q)
        rest.append(float(np.abs(out[0][:3] - q[:3]).max()))
        return out
    c.mpc.solve, c._stop_rest = wrap, wrap_r
    orig = c._solve

    def sv(q):
        c.q_now = np.asarray(q, float)
        return orig(q)
    c._solve = sv
    f = CK.fake(c, 45.0)
    r = CK.summary('無在途狀態前推', c, f)
    print(f'  求解初態與當下差峰 q {max(x[0] for x in seen):.2e}、s {max(x[1] for x in seen):.2e}（應為 0）；安全核對前推底盤峰 {max(rest, default=0.0) * 1e3:.2f} mm（{len(rest)} 次）（應 >0，佇列仍計入）')
    assert max(x[0] for x in seen) == 0.0 and max(x[1] for x in seen) == 0.0
    assert not rest or max(rest) > 0.0
    print('瓶淨空', {k: round(v * 1e3, 2) for k, v in r['clear_bottle'].items() if k in ('meas_min', 'pred_min', 'next_min') and v is not None}, '成功', r['success'])


if __name__ == '__main__':
    sys.exit(main())
