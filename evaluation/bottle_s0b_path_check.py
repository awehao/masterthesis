#!/usr/bin/env python3
"""S0b 全程路徑核對（無 Isaac）：假受控對象跑 test_bottle_s0b 正常流程（x −0.55→、v 0.03），對實測與設定點構型逐 0.05 s 算非手指／手指 ↔ 桌面、桌腳、瓶，底盤 ↔ 桌腳，持瓶 ↔ 桌面、桌腳、非手指；自碰每 0.5 s 一點，相位接點前後逐週期。結果只代表模型軌跡。
    python3 evaluation/bottle_s0b_path_check.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s1_geom as G                              # noqa: E402
import bottle_wb_geom as WG                             # noqa: E402
import test_bottle_s0b as T                             # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402


def check(f):
    """對假受控對象紀錄 f.hist 算最小距離與自碰；回傳 (mins, self_min, n_self, 速率峰, 餘裕)。"""
    boxes, legs, BP = G.table_boxes(), [b for b in G.table_boxes() if b[0].startswith('leg')], G.bottle_points()
    H = f.hist[::5]
    ph = [h['phase'] for h in H]
    near = set()
    for i in range(1, len(ph)):
        if ph[i] != ph[i - 1]:
            near |= set(range(max(i - 6, 0), min(i + 7, len(ph))))
    mins, rel, sm, ns, marg, rate = {}, None, (float('inf'), None, None, None), 0, 9.0, 0.0
    lo, hi = np.asarray(LITE6_SAFE.lower), np.asarray(LITE6_SAFE.upper)
    for h, h2 in zip(f.hist, f.hist[1:]):
        rate = max(rate, float(np.abs(h2['s'] - h['s']).max()) / 0.01)
        marg = min(marg, float(min((h['q'][3:] - lo).min(), (hi - h['q'][3:]).min())))
    for i, h in enumerate(H):
        p, t, q = h['phase'], h['t'], h['q']
        q_sp = np.r_[q[:3], h['s']]
        q_f = G.Q_OPEN if p in WG.OPEN_PH else 0.0038
        held = p in ('LIFT', 'CARRY')
        Tb = None
        if held:
            Tt = T.K.fk(q, 'link_tcp')
            if rel is None:
                T0 = np.eye(4)
                T0[:3, 3] = [T.BX, T.BY, T.TOP]
                rel = np.linalg.inv(Tt) @ T0
            Tb = Tt @ rel
        WG.env_scan(q, p, t, q_f, mins, boxes, legs, 'meas', not held, Tb, BP)
        WG.env_scan(q_sp, p, t, q_f, mins, boxes, legs, 'sp', not held, (T.K.fk(q_sp, 'link_tcp') @ rel) if held else None, BP)
        if i % 10 == 0 or i in near:
            for tag, qq in (('meas', q), ('sp', q_sp)):
                d, who = G.C.self_clear(qq, q_f)
                ns += 1
                if d < sm[0]:
                    sm = (float(d), round(float(t), 2), f'{tag} {p}', who)
    return mins, sm, ns, rate, marg


def main():
    f = T.Fake(T.ctrl()).run(40.0)
    mins, sm, ns, rate, marg = check(f)
    for k, v in sorted(mins.items()):
        print(f'{k:<42} {v[0] * 1e3:8.1f} mm @ {v[1]} s {v[2]}')
    print(f'自碰最小 {sm[0] * 1e3:.1f} mm @ {sm[1]} s {sm[2]} {sm[3]}（{ns} 點）；設定點速率峰 {rate:.3f} rad/s；關節餘裕最小 {marg:.3f} rad；底盤 x {f.hist[0]["x"]:.3f}→{f.hist[-1]["x"]:.3f}')
    json.dump({'min_m': {k: list(v) for k, v in mins.items()}, 'self_min': list(sm), 'rate_max': rate, 'margin_min': marg},
              open(os.path.join(HERE, 'results', 'bottle', 's0b_path_check_r2.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    sys.exit(main())
