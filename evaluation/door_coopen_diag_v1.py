#!/usr/bin/env python3
"""連續協同通行：去程（拉門）斜向切入門縫的最小門角 v1（無 Isaac、無 IK；靜態門角下的底盤圓盤路徑存在性）。

door_coopen_pass_v2 只核平行門洞中線的直行路線（需 72.5–88°）。本檔放寬路線形狀：門角固定為 q 時，
在底盤位置格點（x −0.60–1.60、y −1.40–0.70，每 0.02 m）上，以 8 連通 BFS 找一條底盤圓盤（r 0.30）對門扇／立柱／兩側牆
全程 ≥ 20 mm 的路徑，起點區＝門扇掃掠圈外的門閂側（x ≥ 1.20、y ∈ [−1.0, −0.5]），終點＝y ≥ 0.45（通過）。
求最小 q（每 1°，30–90°）與該門角下的一條路徑（含最窄處淨空）。偏航不計（圓盤）；手臂、外凸物不在內。
**用途**：決定「拉到多大、放手後可否不再推門直接斜向切入」，以及若仍要推門，推到多大即可。

    python3 evaluation/door_coopen_diag_v1.py   # 寫 results/door/door_coopen_diag_v1.json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_v1 as P1                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_diag_v1.json')
XS = [round(-0.60 + 0.02 * k, 2) for k in range(0, 111)]
YS = [round(-1.40 + 0.02 * k, 2) for k in range(0, 106)]
NAMES = ('leaf', 'post0', 'post1', 'wall_hinge_side', 'wall_latch_side')


def solve(q):
    F = np.array([[P1.disc_clear(x, y, q, NAMES) for y in YS] for x in XS])
    ok = F >= P1.BASE_M
    start = [(i, j) for i, x in enumerate(XS) for j, y in enumerate(YS) if x >= 1.20 and -1.0 <= y <= -0.5 and ok[i, j]]
    prev = {s: None for s in start}
    dq = deque(start)
    goal = None
    while dq:
        i, j = dq.popleft()
        if YS[j] >= 0.45:
            goal = (i, j)
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (di or dj) and 0 <= ni < len(XS) and 0 <= nj < len(YS) and ok[ni, nj] and (ni, nj) not in prev:
                    prev[(ni, nj)] = (i, j)
                    dq.append((ni, nj))
    if goal is None:
        return None
    path, c = [], goal
    while c is not None:
        path.append(c)
        c = prev[c]
    path.reverse()
    return {'path': [[XS[i], YS[j]] for i, j in path], 'min_clear_m': float(min(F[i, j] for i, j in path))}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    res, q_min = [], None
    for qd in range(30, 91):
        r = solve(float(qd))
        res.append({'q_deg': qd, 'passable': r is not None, 'min_clear_m': None if r is None else r['min_clear_m']})
        if r is not None and q_min is None:
            q_min, path = qd, r
            print(f'最小斜向可通過門角 {qd}°，路徑 {len(r["path"])} 點，最窄淨空 {r["min_clear_m"]:.3f} m', flush=True)
            break
    json.dump({'schema': 'door_coopen_diag/1', 'q_min_diag_deg': q_min, 'path_at_q_min': None if q_min is None else path['path'],
               'min_clear_at_q_min_m': None if q_min is None else path['min_clear_m'], 'scan': res, 'base_m': P1.BASE_M,
               'not_modeled': ['手臂與外凸物', '偏航（圓盤）', '取樣格點間']}, open(OUT, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
