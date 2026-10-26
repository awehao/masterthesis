#!/usr/bin/env python3
"""連續協同通行：滿足誤差預算的放手狀態集合 v1（Codex 20261026_183611 §3-2「預算不成立就改路徑／放手位置，不降門檻」；離線靜態）。

door_coopen_budget_v1：q45 候選在 10 mm 底盤誤差下不成立——持握段 4° 底盤餘裕 2 mm、放手位姿即瓶頸（46° 時 28.3 mm，門回轉 1° 即 21.8 mm）。
本檔改找放手位置：門角 q_rel ∈ 40–60°（每 1°），門餘轉區間 [q_rel − 2°, q_rel + 5°] 的逐點最小淨空場 F，
  1. 底盤格點（0.02 m）上，放手位姿淨空 F ≥ 30 mm（＝門檻 20 ＋ 誤差 10 mm），且由此到通過（y ≥ 0.45）的最寬路徑瓶頸 ≥ 30 mm
     （對全場一次做「到目標的最大瓶頸」反向 Dijkstra）；
  2. 上述位置中（再取 0.06 m 子格），繞緣夾法 φ 60°、s 0.85、roll 0 在門角 q_rel 可持握：IK（多種子）、關節餘裕 ≥ 0.05、
     非手指／手指／自碰 ≥ 5 mm、底盤（門角 q_rel 當下）≥ 30 mm；偏航取「底盤→抓取點」方位 ±60°（每 15°），粗篩同 edgegrasp_v1。
報：每個 q_rel 的預算可通過格點數、可持握放手狀態數與最佳者（瓶頸最大），作為下一步反向持握搜尋的終點集合。
**界線**：靜態；圓盤近似；門餘轉以區間包絡代替動力學；只用主候選夾法。

    python3 evaluation/door_coopen_release_set_v1.py   # 寫 results/door/door_coopen_release_set_v1.json（不覆寫）
"""
from __future__ import annotations

import heapq
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_post_v1 as PO                  # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_release_set_v1.json')
C, V2 = RV.C, RV.V2
XS, YS = PO.DG.XS, PO.DG.YS
MARGIN = 0.030
DMINUS, DPLUS = 2.0, 5.0
GRASP = {'phi': 60, 's': 0.85, 'roll': 0}


def field(qr):
    F = None
    for q in np.arange(qr - DMINUS, qr + DPLUS + 1e-9, 0.5):
        G = PO.grid(float(round(q, 1)))
        F = G.copy() if F is None else np.minimum(F, G)
    return F


def to_goal_bottleneck(F):
    """W[i,j]＝自 (i,j) 到 y ≥ 0.45 的路徑中，最小淨空的最大值。"""
    W = np.full(F.shape, -np.inf)
    pq = []
    for i in range(len(XS)):
        for j in range(len(YS)):
            if YS[j] >= 0.45:
                W[i, j] = F[i, j]
                heapq.heappush(pq, (-F[i, j], i, j))
    while pq:
        w, i, j = heapq.heappop(pq)
        w = -w
        if w < W[i, j]:
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (di or dj) and 0 <= ni < len(XS) and 0 <= nj < len(YS):
                    nw = min(w, F[ni, nj])
                    if nw > W[ni, nj]:
                        W[ni, nj] = nw
                        heapq.heappush(pq, (-nw, ni, nj))
    return W


def grasp_ok(x, y, q, Tg):
    pg, az = Tg[:3, 3], Tg[:3, 2]
    brg = math.atan2(pg[1] - y, pg[0] - x)
    out = []
    for dyaw in range(-60, 61, 15):
        yaw = brg + math.radians(dyaw)
        m = np.array([x + 0.11 * math.cos(yaw), y + 0.11 * math.sin(yaw), 0.574])
        v = pg - m
        if not (0.12 <= math.hypot(v[0], v[1]) <= 0.50) or float(az @ v) / (np.linalg.norm(v) + 1e-12) < math.cos(math.radians(80)):
            continue
        TJ.cool()
        r = TJ.check(np.array([x, y, yaw]), q, Tg, None, full=True)
        if r is not None and r['clear']['base'] >= MARGIN:
            out.append({'yaw_deg': math.degrees(yaw), 'qa': [float(v_) for v_ in r['qa']], 'jm': r['jm'], 'clear': r['clear']})
    return out


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    res = []
    for qr in range(40, 61):
        TJ.cool()
        F = field(float(qr))
        W = to_goal_bottleneck(F)
        okb = [(XS[i], YS[j], float(F[i, j]), float(W[i, j])) for i in range(len(XS)) for j in range(len(YS))
               if F[i, j] >= MARGIN and W[i, j] >= MARGIN and YS[j] < 0.0]
        q = math.radians(qr)
        Tg = EG.grasp_T(q, GRASP['phi'], GRASP['s'], GRASP['roll'])
        sub = [c for c in okb if abs(round(c[0] / 0.06) * 0.06 - c[0]) < 1e-6 and abs(round(c[1] / 0.06) * 0.06 - c[1]) < 1e-6]
        feas = []
        for x, y, f, w in sub:
            for g in grasp_ok(x, y, q, Tg):
                feas.append({'x': x, 'y': y, 'F_m': f, 'bottleneck_m': w, **g})
        best = max(feas, key=lambda z: min(z['F_m'], z['bottleneck_m']), default=None)
        res.append({'q_rel_deg': qr, 'n_budget_cells': len(okb), 'n_sub': len(sub), 'n_feasible': len(feas),
                    'best': best, 'feasible': feas})
        print(f"q_rel {qr}°：預算格點 {len(okb)}（子格 {len(sub)}）、可持握放手 {len(feas)}"
              + ('' if best is None else f"；最佳 ({best['x']:.2f}, {best['y']:.2f}) 偏航 {best['yaw_deg']:.0f}° F {best['F_m'] * 1000:.0f} mm 瓶頸 {best['bottleneck_m'] * 1000:.0f} mm 底盤 {best['clear']['base'] * 1000:.0f} mm")
              + f"；{time.time() - t0:.0f} s", flush=True)
    json.dump({'schema': 'door_coopen_release_set/1', 'margin_m': MARGIN, 'delta_minus_deg': DMINUS, 'delta_plus_deg': DPLUS, 'grasp': GRASP,
               'per_q': res, 'not_modeled': ['門餘轉動力學（區間包絡）', '圓盤近似', '格點間']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
