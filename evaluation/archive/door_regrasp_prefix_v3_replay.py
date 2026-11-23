#!/usr/bin/env python3
"""開門前綴的雙把手幾何核對（Codex r65 §3b；離線重播，非實跑）：bench_6 實錄（runs/door_coopen_core_bench_6/room_run.json）的逐物理步
底盤、手臂、門角、手指開度，每 STRIDE 步以兩個後端各算一次四類淨空：
  v2＝door_k3m_geom_monitor.default_backend（開門前綴任務端實際使用、看不到內側把手）；
  v3＝door_k3m_geom_monitor_v3.backend_v3(self_mode full)，目標 'out'（開門持握外側橫桿，同前綴分階段規則）。
報：各類兩後端最小值與時刻／部位；v3 最小值比 v2 小（v3 新增幾何較近）的樣本數與最大差；v3 最近部位屬內側把手（bar_in／post_in*）的樣本數與其最小距離。
只回答「bench_6 前綴期間，內側橫桿／立柱是否曾成為最近障礙、離多近」，不代表其他趟次或線上監看已涵蓋。

    python3 -u evaluation/door_regrasp_prefix_v3_replay.py   # 寫 results/door/door_regrasp_prefix_v3_replay.json（不覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402
import door_k3m_geom_monitor_v3 as GM3                 # noqa: E402

RUN = os.path.join(HERE, 'runs', 'door_coopen_core_bench_6', 'room_run.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_prefix_v3_replay.json')
STRIDE = 10
CL = ('nonfinger', 'finger', 'base', 'self')
INNER = ('bar_in', 'post_in')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0w = time.time()
    r = json.load(open(RUN))
    C = r['steps_cols']
    I = {c: C.index(c) for c in ('step', 'sim_t', 'base_xyth', 'q_arm_meas', 'door_q_rad', 'q_finger')}
    f2, f3 = GM.default_backend(), GM3.backend_v3(self_mode='full')
    f3.set_target('out')
    res = {'v2': {c: {'d': math.inf} for c in CL}, 'v3': {c: {'d': math.inf} for c in CL}}
    n, n_closer, d_closer, n_inner, inner_min = 0, {c: 0 for c in CL}, {c: 0.0 for c in CL}, {c: 0 for c in CL}, {c: math.inf for c in CL}
    skipped = 0
    for k, s in enumerate(r['steps']):
        if k % STRIDE:
            continue
        if k % 2000 == 0:
            TJ.cool()
        try:
            x = np.r_[np.asarray(s[I['base_xyth']], float), np.asarray(s[I['q_arm_meas']], float)]
            q = float(s[I['door_q_rad']])
            qf = s[I['q_finger']]
            qfi = 0.5 * (float(qf[0]) + float(qf[-1])) if isinstance(qf, (list, tuple)) else float(qf)
        except (TypeError, ValueError, IndexError):
            skipped += 1
            continue
        if not (np.all(np.isfinite(x)) and math.isfinite(q) and math.isfinite(qfi)):
            skipped += 1
            continue
        d2, d3 = f2(x, q, qfi), f3(x, q, qfi)
        n += 1
        for c in CL:
            for tag, d in (('v2', d2), ('v3', d3)):
                if float(d[c][0]) < res[tag][c]['d']:
                    res[tag][c] = {'d': float(d[c][0]), 'who': d[c][1], 't': float(s[I['sim_t']]), 'step': int(s[I['step']])}
            diff = float(d2[c][0]) - float(d3[c][0])
            if diff > 1e-9:
                n_closer[c] += 1
                d_closer[c] = max(d_closer[c], diff)
            if any(z in str(d3[c][1]) for z in INNER):
                n_inner[c] += 1
                inner_min[c] = min(inner_min[c], float(d3[c][0]))
    out = {'schema': 'door_regrasp_prefix_v3_replay/1', 'run': os.path.relpath(RUN, HERE), 'stride': STRIDE, 'n_samples': n, 'n_skipped': skipped,
           'sim_t_range': [r['steps'][0][I['sim_t']], r['steps'][-1][I['sim_t']]], 'min_m': res,
           'v3_closer_than_v2': {'n': n_closer, 'max_diff_m': d_closer},
           'v3_nearest_is_inner_handle': {'n': n_inner, 'min_m': {c: (v if math.isfinite(v) else None) for c, v in inner_min.items()}},
           'note': '重播 bench_6 前綴實錄；只說明該趟內側把手是否曾成為最近障礙，不代表線上監看已涵蓋', 'wall_s': time.time() - t0w}
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps({k: out[k] for k in ('n_samples', 'n_skipped', 'min_m', 'v3_closer_than_v2', 'v3_nearest_is_inner_handle')}, ensure_ascii=False, default=float))


if __name__ == '__main__':
    main()
