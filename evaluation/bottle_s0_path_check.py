#!/usr/bin/env python3
"""S0a 全程路徑核對（無 Isaac）：假受控對象跑 test_bottle_s0a 的正常流程，逐 0.05 s 算非手指／手指對桌面、桌腳、瓶身距離（瓶固定於桌，直到抓持），
手臂自碰（door_k3m 全點雲），以及最大設定點速率與關節餘裕。結果只代表模型軌跡。
    python3 evaluation/bottle_s0_path_check.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402
import test_bottle_s0a as T                             # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE    # noqa: E402


def main():
    f = T.Fake(T.ctrl()).run(14.0)
    boxes = G.table_boxes()
    lo, hi = np.asarray(LITE6_SAFE.lower), np.asarray(LITE6_SAFE.upper)
    mins, rate_max, marg_min = {}, 0.0, 9.0
    prev = None

    def upd(k, d, t, ph):
        if k not in mins or d < mins[k][0]:
            mins[k] = (float(d), round(float(t), 2), str(ph))
    tg = f.c.judge.t_grasp
    for i, h in enumerate(f.hist):
        t, ph, _, _, s, q = h
        if prev is not None:
            rate_max = max(rate_max, float(np.abs(s - prev).max()) / 0.01)
        prev = s
        marg_min = min(marg_min, float(min((q[3:] - lo).min(), (hi - q[3:]).min())))
        if i % 5:
            continue
        q_f = 0.0089 if ph in ('CRUISE', 'PREGRASP', 'INSERT') else 0.0038
        for n, P in G.link_world(q, q_f).items():
            fing = n in G.C.FINGERS
            tag = 'finger' if fing else 'nonfinger'
            for nm, c, sz in boxes:
                upd(tag + '↔' + ('table_top' if nm == 'table_top' else 'leg'), G.box_dist(P, c, sz).min(), t, ph)
            if tg is None or t < tg:
                d = G.cyl_dist(P, 0.0, 0.42, 0.60, 0.80, 0.015).min()
                if not fing:
                    upd('nonfinger↔bottle', d, t, ph)
                elif ph in ('CRUISE', 'PREGRASP'):
                    upd('finger↔bottle（INSERT 前）', d, t, ph)
    for k, v in sorted(mins.items()):
        print(f'{k:<30} {v[0] * 1e3:8.1f} mm @ {v[1]} s {v[2]}')
    print(f'設定點速率峰 {rate_max:.3f} rad/s（上限 1.0）；關節餘裕最小 {marg_min:.3f} rad；事件 {[(e["t"], e.get("to", e["what"])) for e in f.c.events]}')
    json.dump({'min_m': {k: list(v) for k, v in mins.items()}, 'rate_max': rate_max, 'margin_min': marg_min},
              open(os.path.join(HERE, 'results', 'bottle', 's0_path_check.json'), 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    sys.exit(main())
