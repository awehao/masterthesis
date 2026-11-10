#!/usr/bin/env python3
"""觀看視角生成器輸出範圍的離線核對（Codex 20261111_053311 第 1 項：觀看姿態與到達軌跡核碰撞與關節限制）。純離線。
視角＝V_HIGH 腕部樣板＋q1（bottle_view_gen.view_q1 的輸出族）；到達軌跡＝收攏 → 視角的關節 min-jerk（21 取樣）。
底盤放在遠離桌子處（自碰撞與 q 有關、與底盤位置無關；桌碰撞在抬高時底盤離區域 ≥ 1 m，另由實際路徑核）；報每個 q1 的路徑最小自碰撞與關節裕度，得可用 q1 範圍。
    python3 evaluation/view_domain_check.py OUT.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_r0_approach as R0  # noqa: E402


def main():
    rows = []
    for q1 in np.linspace(-1.5, 1.5, 13):
        qv = R0.Q_VHIGH.copy()
        qv[0] = q1
        sm, jm = np.inf, np.inf
        for s in np.linspace(0, 1, 21):
            m = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            qa = R0.Q_TUCK + (qv - R0.Q_TUCK) * m
            _, sc, _ = R0.env(np.r_[-6.0, -3.0, 0.0, qa], [])
            sm, jm = min(sm, sc[0]), min(jm, R0.jmargin(qa))
        rows.append(dict(q1=round(float(q1), 3), self_min=round(float(sm), 4), joint_margin_min=round(float(jm), 4), ok=bool(sm >= R0.SELF_M and jm >= R0.LIM_M)))
        print(rows[-1], flush=True)
    okq = [r['q1'] for r in rows if r['ok']]
    json.dump(dict(rows=rows, q1_ok_range=[min(okq), max(okq)] if okq else None, note='V_HIGH 腕部樣板＋q1；收攏→視角 min-jerk；自碰撞門檻 5 mm、關節裕度 0.05 rad'),
              open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
    print('可用 q1', [min(okq), max(okq)] if okq else None)


if __name__ == '__main__':
    main()
