#!/usr/bin/env python3
"""連續協同通行：離線可行性盤點 v1 之二——「開門須領先通行多少」（Codex 20261026_141054 §3；無 Isaac、無 IK、無接觸）。

對平行於門洞中線（+y 方向）的直行路線 x＝x_c（0.40–0.70 m，每 0.05 m），沿 y（−1.20 → +0.60 m，每 0.02 m）逐點求：
  - 牆淨空：底盤圓盤（r 0.30）對兩側牆 ≥ 20 mm 是否可能（與門角無關；不成立 ⇒ 該 x_c 無法直行穿門）；
  - q_min(y)：底盤圓盤對門扇與兩根立柱淨空 ≥ 20 mm 所需的最小門角（0–110°、每 0.5°；門扇把手側＝機器人側）。
    「q_min 單調可達」＝q 由小到大時該點淨空恢復後不再失去；本版逐點取最小可行 q，並記錄可行 q 是否為一段連續區間。
彙整：
  - q_pass(x_c)＝max_y q_min(y)：以 x_c 直行穿過門洞所需的門角；
  - 開門—通行重疊：門角到 q 時底盤最遠可前進到的 y_reach(q)（＝ q_min ≤ q 的連續前綴的最後一點），
    即「門開到 q，底盤已可走到 y_reach(q)」；通行進展與開門進展可同時推進的區間由此讀出。
**界線**：只核底盤圓盤（不含手臂、外凸物）、靜態門角、直行路線；不代表手臂構型或接觸可行。

    python3 evaluation/door_coopen_pass_v1.py   # 寫 results/door/door_coopen_pass_v1.json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_geom_v2 as G2                              # noqa: E402
import door_k2_offline_v2 as V2                        # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_v1.json')
C, K1 = V2.C, V2.K1
D2 = K1.D2
BASE_M = 0.02
QS = [round(0.5 * k, 1) for k in range(0, 221)]


def disc_clear(x, y, q, names):
    best = math.inf
    for name, c, R, hs in G2.shapes(D2, math.radians(q), K1.H):
        if name in names:
            best = min(best, float(G2.disc_clear(x, y, c[2], C.RB, c, R, hs)))
    return best


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    walls, door = ('wall_hinge_side', 'wall_latch_side'), ('leaf', 'post0', 'post1')
    ys = [round(-1.20 + 0.02 * k, 2) for k in range(0, 91)]
    res = []
    for xc in [round(0.40 + 0.05 * k, 2) for k in range(0, 7)]:
        rows = []
        for y in ys:
            wc = disc_clear(xc, y, 0.0, walls)
            ok = [q for q in QS if disc_clear(xc, y, q, door) >= BASE_M]
            qmin = ok[0] if ok else None
            contiguous = bool(ok) and all(abs(b - a - 0.5) < 1e-9 for a, b in zip(ok, ok[1:])) and ok[-1] == QS[-1]
            rows.append({'y': y, 'wall_clear_m': wc, 'wall_ok': wc >= BASE_M, 'q_min_deg': qmin, 'q_ok_contiguous_to_110': contiguous})
        passable = all(r['wall_ok'] and r['q_min_deg'] is not None for r in rows)
        q_pass = max((r['q_min_deg'] for r in rows if r['q_min_deg'] is not None), default=None) if passable else None
        reach = []
        for q in range(0, 111, 5):
            last = None
            for r in rows:
                if r['wall_ok'] and r['q_min_deg'] is not None and r['q_min_deg'] <= q:
                    last = r['y']
                else:
                    break
            reach.append({'q_deg': q, 'y_reach': last})
        res.append({'x_c': xc, 'passable_walls': all(r['wall_ok'] for r in rows), 'q_pass_deg': q_pass, 'y_reach_by_q': reach, 'rows': rows})
        print(f"x_c={xc:.2f} 牆可通過 {all(r['wall_ok'] for r in rows)} q_pass {q_pass} "
              f"y_reach@30/45/60/75/90 {[next(z['y_reach'] for z in reach if z['q_deg'] == q) for q in (30, 45, 60, 75, 90)]}", flush=True)
    json.dump({'schema': 'door_coopen_pass/1', 'base_m': BASE_M, 'robot_radius_m': C.RB, 'per_xc': res,
               'not_modeled': ['手臂與外凸物', '門角變動中的掃掠（逐門角靜態）', '非直行路線']}, open(OUT, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
