#!/usr/bin/env python3
"""連續協同通行：去程候選的後處理 v1——推門段中途放手後可否直接斜向通過（無 Isaac、無 IK）。
讀 door_coopen_traj_v1_sweepA.json 每個 q_A 候選的 B 相（推門）軌跡；對每一步 (q, 底盤位置)：
  在門角固定為 q 時，以 door_coopen_diag_v1 同一格點與 8 連通 BFS，從**該步底盤位置**（最近格點）找到 y ≥ 0.45 的底盤圓盤路徑（≥ 20 mm）。
報：最早可放手直接通過的門角 q_release（及該步底盤位置、到通過的路徑長、最窄淨空），以及到 q_release 為止的推門段
「底盤朝門縫前進」的重疊（底盤到門縫入口〔門閂側牆端面 (1.0, 0)〕距離縮短的門角範圍）。
**界線**：放手後門視為靜止（場景門無閉門器；實跑曾見張爪後門自轉，另需處置）；手臂於通過時須收攏（另核）；圓盤近似。

    python3 evaluation/door_coopen_traj_post_v1.py   # 寫 results/door/door_coopen_traj_post_v1.json（不覆寫）
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
import door_coopen_diag_v1 as DG                       # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_traj_v1_sweepA.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_traj_post_v1.json')
_CACHE = {}


def grid(q):
    if q not in _CACHE:
        F = np.array([[DG.P1.disc_clear(x, y, q, DG.NAMES) for y in DG.YS] for x in DG.XS])
        _CACHE[q] = F
    return _CACHE[q]


def bfs_from(q, x0, y0):
    F = grid(float(q))
    ok = F >= DG.P1.BASE_M
    i0 = int(np.argmin([abs(x - x0) for x in DG.XS]))
    j0 = int(np.argmin([abs(y - y0) for y in DG.YS]))
    if not ok[i0, j0]:
        return None
    prev = {(i0, j0): None}
    dq = deque([(i0, j0)])
    while dq:
        i, j = dq.popleft()
        if DG.YS[j] >= 0.45:
            path, c = [], (i, j)
            while c is not None:
                path.append(c)
                c = prev[c]
            path.reverse()
            L = sum(math.hypot(DG.XS[a[0]] - DG.XS[b[0]], DG.YS[a[1]] - DG.YS[b[1]]) for a, b in zip(path, path[1:]))
            return {'len_m': L, 'min_clear_m': float(min(F[a, b] for a, b in path)), 'n': len(path)}
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (di or dj) and 0 <= ni < len(DG.XS) and 0 <= nj < len(DG.YS) and ok[ni, nj] and (ni, nj) not in prev:
                    prev[(ni, nj)] = (i, j)
                    dq.append((ni, nj))
    return None


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    d = json.load(open(SRC))
    out = []
    for c in d['candidates']:
        B = c.get('phase_B') or []
        rel = None
        rows = []
        for p in B:
            r = bfs_from(p['q_deg'], p['base'][0], p['base'][1])
            gap = math.hypot(p['base'][0] - 1.0, p['base'][1] - 0.0)
            rows.append({'q_deg': p['q_deg'], 'base': p['base'], 'dist_to_latch_jamb_m': gap, 'release_pass': r})
            if r is not None and rel is None:
                rel = rows[-1]
        seg = [r_ for r_ in rows if rel is None or r_['q_deg'] <= rel['q_deg']]
        closer = [b['q_deg'] for a, b in zip(seg, seg[1:]) if b['dist_to_latch_jamb_m'] < a['dist_to_latch_jamb_m'] - 1e-9]
        out.append({'q_A': c['q_A'], 'q_release_deg': None if rel is None else rel['q_deg'], 'release_step': rel,
                    'push_overlap_toward_gap_q_deg': [min(closer), max(closer)] if closer else None, 'n_closer_steps': len(closer), 'rows': rows})
        rs = '' if rel is None else f"（底盤 {np.round(rel['base'][:2], 2)}、路徑 {rel['release_pass']['len_m']:.2f} m、最窄 {rel['release_pass']['min_clear_m']:.3f} m）"
        print(f"q_A {c['q_A']}°：最早可放手直接通過 {None if rel is None else rel['q_deg']}°{rs}；推門中朝門縫前進 {len(closer)} 步 {out[-1]['push_overlap_toward_gap_q_deg']}",
              flush=True)
    json.dump({'schema': 'door_coopen_traj_post/1', 'source': os.path.relpath(SRC, HERE), 'per_qA': out,
               'not_modeled': ['放手後門的餘轉', '手臂收攏', '圓盤近似']}, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
