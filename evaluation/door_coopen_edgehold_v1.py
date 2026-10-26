#!/usr/bin/env python3
"""連續協同通行：「繞門緣握把手」持握軌跡 v1（Howard 2026-10-26 選定；無 Isaac、無接觸力；單一程序＋溫度暫停）。

接續 door_coopen_edgegrasp_v1：門洞側可行抓持只在 q 45°（底盤 (1.0,−0.8)）與 60°（(0.7,−1.1)），夾法 φ 60°、s 0.85。
本檔從這些種子出發，夾法固定（剛性持握，不滑、不換抓），門角每 1° 一步往兩個方向延伸：
  開大（q 增加至 100°）：底盤 Δx、Δy ∈ {−0.04…0.04}（0.02 步）、Δ偏航 ±4°，評分＝往門縫入口 (0.515, −0.10) 靠近；
  關小（q 減少至 0°）：同一組步長，評分＝往門前側 (0.5, −1.0) 靠近（看這個夾法最早能在哪個門角開始用）。
IK 以上一步為種子、相鄰步手臂 ≤ 0.10 rad；判準同 door_coopen_traj_v1（非手指／手指／自碰 ≥ 5 mm、底盤 ≥ 20 mm、關節餘裕 ≥ 0.05 rad）。
報：持握可延伸的門角範圍、持握中底盤離門縫入口最近的狀態（門角、位置、距離），以及在該狀態放手後、門角不動時
  底盤到通過（y ≥ 0.45）的最短路徑（door_coopen_traj_post_v1 同一格點 BFS）與其中「往回退」的量（路徑上 y 的最小值低於放手點多少）。
另報可達上界：底盤中心 y ≥ y0 時，安裝點到把手中心的最短水平距離（不論夾法、不論 IK），用來判斷「握著把手完成通過」是否可能。
**界線**：取樣步長、靜態、剛性夾持；放手後門視為靜止；不含換抓過程（由正面夾法換到繞緣夾法）、夾持力與動態。

    python3 evaluation/door_coopen_edgehold_v1.py   # 寫 results/door/door_coopen_edgehold_v1.json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_traj_post_v1 as PO                  # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k2_check as KC                             # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_edgegrasp_v1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_edgehold_v1.json')
GAP = np.array([0.515, -0.10])
FRONT = np.array([0.5, -1.0])
STEPS = (-0.04, -0.02, 0.0, 0.02, 0.04)


def run(seed, q0, q1, goal, width=5, per_parent=8):
    step = 1 if q1 > q0 else -1
    T_of = lambda q: EG.grasp_T(q, seed['phi'], seed['s'], seed['roll'])   # noqa: E731
    st0 = {'base': np.array([seed['x'], seed['y'], math.radians(seed['yaw_deg'])]), 'qa': np.array(seed['qa']),
           'jm': seed['jm'], 'clear': seed['clear']}
    layer = [dict(st0, path=[dict(st0, q=q0)])]
    for qd in range(q0 + step, q1 + step, step):
        TJ.cool()
        q = math.radians(qd)
        cand = []
        for st in layer:
            nb = []
            for dx in STEPS:
                for dy in STEPS:
                    for dyaw in (-4, 0, 4):
                        b = np.array([st['base'][0] + dx, st['base'][1] + dy, st['base'][2] + math.radians(dyaw)])
                        nb.append((-float(np.hypot(*(b[:2] - goal))), b))
            nb.sort(key=lambda z: -z[0])
            k = 0
            for sc, b in nb:
                if k >= per_parent:
                    break
                r = TJ.check(b, q, T_of(q), st['qa'])
                if r is None:
                    continue
                k += 1
                ns = {'base': b, 'qa': r['qa'], 'score': sc, 'jm': r['jm'], 'clear': r['clear']}
                ns['path'] = st['path'] + [dict(ns, q=qd)]
                cand.append(ns)
        if not cand:
            return layer, qd - step, 'no_feasible_child'
        cand.sort(key=lambda z: -z['score'])
        seen, layer = set(), []
        for c_ in cand:
            key = (round(c_['base'][0], 2), round(c_['base'][1], 2), round(math.degrees(c_['base'][2]) / 3))
            if key not in seen:
                seen.add(key)
                layer.append(c_)
            if len(layer) >= width:
                break
    return layer, q1, 'end'


def release_pass(qd, b):
    return PO.bfs_from(float(qd), float(b[0]), float(b[1]))


def retreat_of(qd, b):
    """BFS 路徑上 y 的最小值低於放手點多少（> 0 即須往回退）；PO.bfs_from 不回傳路徑，這裡重跑一份帶路徑的。"""
    F = PO.grid(float(qd))
    ok = F >= PO.DG.P1.BASE_M
    XS, YS = PO.DG.XS, PO.DG.YS
    i0 = int(np.argmin([abs(x - b[0]) for x in XS]))
    j0 = int(np.argmin([abs(y - b[1]) for y in YS]))
    if not ok[i0, j0]:
        return None

    prev = {(i0, j0): None}
    dq = deque([(i0, j0)])
    while dq:
        i, j = dq.popleft()
        if YS[j] >= 0.45:
            ys, c = [], (i, j)
            while c is not None:
                ys.append(YS[c[1]])
                c = prev[c]
            return float(max(0.0, YS[j0] - min(ys)))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (di or dj) and 0 <= ni < len(XS) and 0 <= nj < len(YS) and ok[ni, nj] and (ni, nj) not in prev:
                    prev[(ni, nj)] = (i, j)
                    dq.append((ni, nj))
    return None


def reach_bound():
    """底盤中心 y ≥ y0、底盤圓盤不撞門／牆時，安裝點（偏航任意，0.11 m 前置）到把手中心的最短水平距離。"""
    out = []
    for qd in (45, 60, 75, 90):
        q = math.radians(qd)
        hc = KC._H_of(q)[:3, 3]
        for y0 in (-0.40, -0.20, 0.0, 0.45):
            best = None
            for x in np.arange(-0.6, 1.61, 0.02):
                for y in np.arange(y0, 1.0, 0.02):
                    if PO.DG.P1.disc_clear(float(x), float(y), float(qd), PO.DG.NAMES) < PO.DG.P1.BASE_M:
                        continue
                    d = max(0.0, math.hypot(hc[0] - x, hc[1] - y) - 0.11)
                    if best is None or d < best[0]:
                        best = (d, float(x), float(y))
            out.append({'q_deg': qd, 'y0': y0, 'min_mount_to_handle_horiz_m': None if best is None else best[0],
                        'at_base': None if best is None else best[1:]})
    return out


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    src = json.load(open(SRC))
    seeds = [(r['q_deg'], f) for r in src['per_q'] for f in r['feasible']]
    rb = reach_bound()
    print('可達上界（安裝點到把手中心最短水平距離，m）：' +
          '；'.join(f"q{r['q_deg']} y≥{r['y0']}: {r['min_mount_to_handle_horiz_m']:.2f}" for r in rb if r['min_mount_to_handle_horiz_m'] is not None), flush=True)
    res = []
    for qs, sd in seeds:
        rec = {'seed_q_deg': qs, 'seed': {k: sd[k] for k in ('x', 'y', 'yaw_deg', 'phi', 's', 'roll')}}
        for name, q1, goal in (('open', 100, GAP), ('close', 0, FRONT)):
            layer, qend, why = run(sd, qs, q1, goal)
            P = layer[0]['path']
            js = [float(np.max(np.abs(np.array(b['qa']) - np.array(a['qa'])))) for a, b in zip(P, P[1:])]
            r = {'q_end_deg': qend, 'why': why, 'n_steps': len(P) - 1, 'end_base': [float(v) for v in P[-1]['base']],
                 'joint_step_max_rad': max(js) if js else None, 'joint_margin_min': min(p['jm'] for p in P),
                 'clear_min': {k: min(p['clear'][k] for p in P) for k in ('nonfinger', 'finger', 'base', 'self')},
                 'trajectory': [{'q_deg': p['q'], 'base': [float(v) for v in p['base']], 'qa': [float(v) for v in p['qa']],
                                 'jm': p['jm'], 'clear': p['clear']} for p in P]}
            if name == 'open':
                cl = min(P, key=lambda p: float(np.hypot(*(p['base'][:2] - GAP))))
                rp = release_pass(cl['q'], cl['base'])
                r['closest_to_gap'] = {'q_deg': cl['q'], 'base': [float(v) for v in cl['base']],
                                       'dist_to_gap_entry_m': float(np.hypot(*(cl['base'][:2] - GAP))),
                                       'release_pass': rp, 'retreat_m': retreat_of(cl['q'], cl['base']) if rp else None}
            rec[name] = r
            msg = ''
            if name == 'open':
                c = r['closest_to_gap']
                tail = '不可' if c['release_pass'] is None else f"{c['release_pass']['len_m']:.2f} m、退 {c['retreat_m']:.2f} m"
                msg = f"；離門縫入口最近 {c['dist_to_gap_entry_m']:.2f} m @ {c['q_deg']}°，放手後通過 {tail}"
            print(f"種子 q{qs} roll {sd['roll']}｜{name}：{why}，持握到 {qend}°，底盤 {np.round(P[-1]['base'][:2], 2)}{msg}；{time.time() - t0:.0f} s",
                  flush=True)
        res.append(rec)
    json.dump({'schema': 'door_coopen_edgehold/1', 'source': os.path.relpath(SRC, HERE), 'gap_entry': GAP.tolist(),
               'reach_bound': rb, 'per_seed': res,
               'not_modeled': ['換抓過程（正面→繞緣）', '夾持力、門動力學', '取樣間碰撞', '放手後門的餘轉', '手臂收攏']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
