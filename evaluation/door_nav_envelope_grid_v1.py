#!/usr/bin/env python3
"""導航包絡：關節偏差格點核算＋線上核對成本（接 door_nav_envelope_v1；鏈式界 Σ L≈3.4 m 太鬆不可用，改報格點最大值，並據以採「線上逐週期核實測構型」）。
幾何同 v1（全部頂點不抽樣）；每連桿先取凸包頂點（水平半徑最大值必在凸包頂點，與全頂點結果相同）。
格點：q0＋δ·s、s∈{−1,0,1}^6（729 組）、δ∈{0.02, 0.05}；手指張開（v1 已示張≥合）。格點最大值**不是**嚴格上界，只示敏感度。
    python3 evaluation/door_nav_envelope_grid_v1.py   # 寫 results/door/door_nav_envelope_grid_v1.json（不覆寫）
"""
import itertools
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_release_plan_v1 as RP               # noqa: E402
import door_nav_envelope_v1 as E                       # noqa: E402
import door_offline_check_v1 as C                      # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_nav_envelope_grid_v1.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    CL = E.full_clouds(open(C.URDF).read())
    arm = [n for n in CL if n.startswith('link') or n.startswith('uflite')]
    Hv = {n: CL[n][ConvexHull(CL[n]).vertices] for n in arm}

    def rho(qa, qfi):
        qf, b = [0.0, 0.0, 0.0] + list(qa), 0.0
        for n in arm:
            T = np.asarray(C.K.fk(qf, n))
            P = Hv[n] + (C.FINGER_AXIS[n] * qfi if n in C.FINGER_AXIS else 0.0)
            W = (T[:3, :3] @ P.T).T + T[:3, 3]
            b = max(b, float(np.hypot(W[:, 0], W[:, 1]).max()))
        return b

    res = {'schema': 'door_nav_envelope_grid/1', 'n_hull_vertices': {n: int(len(v)) for n, v in Hv.items()}, 'poses': {}}
    for nm, q0 in E.POSES.items():
        t = time.perf_counter()
        r0 = rho(q0, RP.OPEN)
        row = {'q': q0, 'rho_nominal_m': r0, 'eval_ms': (time.perf_counter() - t) * 1e3, 'grid': {}}
        for d in (0.02, 0.05):
            row['grid'][str(d)] = max(rho(np.array(q0) + d * np.array(s), RP.OPEN) for s in itertools.product((-1, 0, 1), repeat=6))
        res['poses'][nm] = row
        print(f'{nm}：名目 {r0:.4f} m；格點最大 δ0.02 {row["grid"]["0.02"]:.4f}、δ0.05 {row["grid"]["0.05"]:.4f}；單次 {row["eval_ms"]:.2f} ms', flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print('寫', OUT)


if __name__ == '__main__':
    main()
