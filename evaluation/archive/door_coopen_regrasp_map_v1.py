#!/usr/bin/env python3
"""A 包：門洞內重新抓取的可行域（v4 阻點後的有界替代：開門到位放手 → 底盤進門洞 → 重新以繞緣夾法抓把手 → 邊穿越邊拉門關）。
門角 47.7°（bench_6 開門末）與 40°；底盤格點 x 0.40–0.95、y −0.45–0.20（0.05 m）、偏航 0–360°（30°）；TCP＝繞緣夾法 φ60 s0.85 roll 0；
以 door_coopen_traj_v1.check(full=True)（多種子 IK＋底盤 ≥ 20 mm＋關節餘裕 ≥ 0.05＋非手指／手指／自碰 ≥ 5 mm）判可行。
報各門角的可行格數、最接近門洞（y 最大）的可行狀態與其淨空；不含放手／退臂與移動過程（另核）。

    python3 evaluation/door_coopen_regrasp_map_v1.py   # 寫 results/door/door_coopen_regrasp_map_v1.json（不覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_regrasp_map_v1.json')
GRASP = (60.0, 0.85, 0.0)


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    res = {'grasp': GRASP, 'per_q': []}
    for qd in (47.7, 40.0):
        qr = math.radians(qd)
        T = EG.grasp_T(qr, *GRASP)
        feas = []
        for x in np.round(np.arange(0.40, 0.951, 0.05), 3):
            TJ.cool()
            for y in np.round(np.arange(-0.45, 0.201, 0.05), 3):
                for yd in range(0, 360, 30):
                    r = TJ.check([float(x), float(y), math.radians(yd)], qr, T, None, full=True)
                    if r is not None:
                        feas.append({'x': float(x), 'y': float(y), 'yaw_deg': yd, 'qa': [float(v) for v in r['qa']], 'jm': r['jm'], 'clear': r['clear']})
        top = sorted(feas, key=lambda f: -f['y'])[:5]
        res['per_q'].append({'q_deg': qd, 'n_feasible': len(feas), 'y_max': max((f['y'] for f in feas), default=None), 'top_by_y': top,
                             'cells': [[f['x'], f['y'], f['yaw_deg']] for f in feas]})
        print(f'門 {qd}°：可行 {len(feas)} 格；最大 y {res["per_q"][-1]["y_max"]}；', [(f['x'], f['y'], f['yaw_deg'], round(min(f['clear'].values()) * 1e3, 1)) for f in top], flush=True)
    res['wall_s'] = time.time() - t0
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
