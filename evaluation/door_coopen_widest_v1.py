#!/usr/bin/env python3
"""連續協同通行：放手後通過的瓶頸淨空 v1（後處理；無 IK）。

door_coopen_arc_v1.bfs_path 求的是格點最短路徑，會貼著 20 mm 門檻走（edgearc_v1 報的最窄 0.020 即此）。本檔改求「最寬路徑」
（使路徑上最小底盤圓盤淨空最大；同一格點、8 連通、Dijkstra 變形），報 door_coopen_edgearc_v1 各種子最佳放手狀態的瓶頸淨空，
另列同底盤位置在門角 47／50° 的值做敏感度參考（**後者底盤位置不在已驗證持握軌跡上，只是參考**）。

    python3 evaluation/door_coopen_widest_v1.py   # 寫 results/door/door_coopen_widest_v1.json（不覆寫）
"""
from __future__ import annotations

import heapq
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_traj_post_v1 as PO                  # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_edgearc_v1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_widest_v1.json')
XS, YS = PO.DG.XS, PO.DG.YS


def widest(qd, b):
    F = PO.grid(float(qd))
    i0 = int(np.argmin([abs(x - b[0]) for x in XS]))
    j0 = int(np.argmin([abs(y - b[1]) for y in YS]))
    best = {(i0, j0): F[i0, j0]}
    pq, done = [(-F[i0, j0], i0, j0)], set()
    while pq:
        w, i, j = heapq.heappop(pq)
        w = -w
        if (i, j) in done:
            continue
        done.add((i, j))
        if YS[j] >= 0.45:
            return float(w)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (di or dj) and 0 <= ni < len(XS) and 0 <= nj < len(YS):
                    nw = min(w, F[ni, nj])
                    if nw > best.get((ni, nj), -1e9):
                        best[(ni, nj)] = nw
                        heapq.heappush(pq, (-nw, ni, nj))
    return None


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    d = json.load(open(SRC))
    out = []
    for s in d['per_seed']:
        b = s.get('best')
        if not b:
            continue
        base = b['trajectory'][-1]['base']
        q = b['q_rel_deg']
        rec = {'seed_q_deg': s['seed_q_deg'], 'roll': s['seed']['roll'], 'q_rel_deg': q, 'base': base[:2],
               'bottleneck_m': widest(q, base), 'shortest_path_min_m': b['release']['min_clear_m'],
               'sensitivity_same_base': {str(qq): widest(qq, base) for qq in (q + 1, q + 4)}}
        out.append(rec)
        print(f"種子 q{rec['seed_q_deg']} roll {rec['roll']}：放手 {q:.0f}° 瓶頸 {rec['bottleneck_m']:.3f} m（最短路徑 {rec['shortest_path_min_m']:.3f}）；"
              f"同位置 +1°／+4°：{rec['sensitivity_same_base']}", flush=True)
    json.dump({'schema': 'door_coopen_widest/1', 'source': os.path.relpath(SRC, HERE), 'per_seed': out,
               'note': '敏感度列的底盤位置不在已驗證持握軌跡上'}, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
