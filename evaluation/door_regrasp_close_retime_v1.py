#!/usr/bin/env python3
"""通過中拉關＋解除收臂的時間化 v1（來源 door_regrasp_close_plan_v1）。時間參數化與換抓時間化 v7 同一套（door_coopen_pass_close_inner_retime_v3 的
LSPB、接點混合、底盤車體座標每軸上限、0.995 倍加速度）；速度比例 0.5（同 v3 自由段）、偏航速度上限 0.0738 rad/s（同 v7）、門角速率 ≤ 0.6 °/s（登錄）。
輸出 20 Hz 參考（[t, x9, q_door_deg, q_finger, stage]），供 W-GMPC 名目閉迴路與後續參考包。

    python3 -u evaluation/door_regrasp_close_retime_v1.py   # 寫 results/door/door_regrasp_close_retime_v1.json（不覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_retime_v3 as RT3   # noqa: E402
import door_regrasp_retime_v7 as RV7                   # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_regrasp_close_plan_v1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_close_retime_v1.json')
RATIO = 0.5


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    src = json.load(open(SRC))
    if not src.get('ok'):
        raise SystemExit('關門規劃不成立')
    rows = src['rows']
    P = [np.array(r['x'], float) for r in rows]
    Q = [float(r['q']) for r in rows]
    F = [float(r['qf']) for r in rows]
    st = [r['stage'] for r in rows]
    for k in range(1, len(P)):
        P[k][2] = P[k - 1][2] + math.remainder(P[k][2] - P[k - 1][2], 2 * math.pi)
    ratio = np.full(len(P) - 1, RATIO)
    dt, tau, v, vq = RV7.plan(P, Q, F, ratio)
    S, tw = RT3.sample(P, Q, F, dt, tau, v, vq, st)
    X = np.array([s['x'] for s in S])
    yaw_rate = float(np.max(np.abs(np.diff(X[:, 2]))) / 0.05)
    qd_rate = float(np.max(np.abs(np.diff([s['q'] for s in S]))) / 0.05)
    out = {'schema': 'door_regrasp_close_retime/1', 'source': os.path.relpath(SRC, HERE), 'ratio': RATIO, 'yaw_ref_max': RV7.YAW_REF_MAX,
           't_end_s': S[-1]['t'], 'n_samples': len(S), 'yaw_rate_max': yaw_rate, 'door_rate_max_dps': qd_rate,
           'stage_t_start': {s_: next((float(x['t']) for x in S if x['stage'] == s_), None) for s_ in ('G', 'P', 'R_open', 'R_slide', 'R_stow')},
           'reference_20hz': {'cols': ['t', 'x9', 'q_door_deg', 'q_finger', 'stage'],
                              'rows': [[s['t'], [float(z) for z in s['x']], s['q'], s['qf'], s['stage']] for s in S]}}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('t_end_s', 'n_samples', 'yaw_rate_max', 'door_rate_max_dps', 'stage_t_start')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
