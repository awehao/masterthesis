#!/usr/bin/env python3
"""連續協同通行：去程「邊開邊繞」軌跡 v1（Howard 2026-10-26 釐清；無 Isaac、無接觸力；單一程序＋溫度暫停）。

Howard：不必握著通過，而是開門的過程中底盤就沿著出去的軌跡動——門往左開（自由端往 −y、往鉸鏈側掃），
機器人往右後走（往門閂側 +x、往 −y），繞著門扇自由端，形成「開門、同時繞門」的趨勢；放手時底盤已在切入門縫的路上，不停、不退。
（door_coopen_edgehold_v1 的可達上界：底盤中心過牆面 y ≥ 0 時把手已在 0.48–0.78 m 外，過門 y ≥ 0.45 時 ≥ 0.93 m，
  超出 Lite6 伸距，故握著把手完成通過不可能，放手是必要的；本檔問的是放手前後運動能否連成一條弧。）

夾法：正面夾法（T_HG，實跑 door_wide_w2c_30_1），剛性持握，不換抓。起點＝door_wide_offline_sweep_90 第一列（q 0）。
搜尋：門角每 1° 一步到 70°；底盤 Δx、Δy ∈ {−0.03, 0, 0.03}、Δ偏航 ±3°；評分＝x − 0.3·max(0, −0.95 − y)（往門閂側繞、不往後退太深）；
  IK 以上一步為種子、相鄰步手臂 ≤ 0.10 rad；判準同 door_coopen_traj_v1。
每一步（q ≥ 40°）評估「此刻放手」：門角固定為 q，以 door_coopen_traj_post_v1 同一格點做 8 連通 BFS 到 y ≥ 0.45，報
  路徑長、最窄淨空、路徑上 y 低於放手點的量（後退量）、持握段最後 3 步的底盤運動方向與放手後前 0.30 m 路徑方向的夾角（轉向角）。
報：最早可「不後退、轉向角 ≤ 90°」放手直通的門角；持握段中底盤往 +x（門閂側）移動的步數與停住（位移 < 1 mm）的步數。
**界線**：同 traj_v1；BFS 是格點最短路徑而非平滑軌跡；放手後門視為靜止（場景門無閉門器）；手臂收攏與張爪後門餘轉另核。

    python3 evaluation/door_coopen_arc_v1.py   # 寫 results/door/door_coopen_arc_v1.json（不覆寫）
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
import door_coopen_traj_post_v1 as PO                  # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k2_check as KC                             # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_arc_v1.json')
Q_END = 70
TURN_MAX_DEG = 90.0


def bfs_path(qd, b):
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
            path, c = [], (i, j)
            while c is not None:
                path.append(c)
                c = prev[c]
            path.reverse()
            return [(XS[a], YS[b_]) for a, b_ in path], float(min(F[a, b_] for a, b_ in path))
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni, nj = i + di, j + dj
                if (di or dj) and 0 <= ni < len(XS) and 0 <= nj < len(YS) and ok[ni, nj] and (ni, nj) not in prev:
                    prev[(ni, nj)] = (i, j)
                    dq.append((ni, nj))
    return None


def release_eval(P, k):
    """持握路徑 P 第 k 步放手。"""
    qd, b = P[k]['q'], P[k]['base']
    r = bfs_path(qd, b)
    if r is None:
        return None
    pts, mc = r
    L = sum(math.hypot(a[0] - c[0], a[1] - c[1]) for a, c in zip(pts, pts[1:]))
    v_in = np.array(b[:2]) - np.array(P[max(0, k - 3)]['base'][:2])
    acc, j = 0.0, 1
    while j < len(pts) - 1 and acc < 0.30:
        acc += math.hypot(pts[j][0] - pts[j - 1][0], pts[j][1] - pts[j - 1][1])
        j += 1
    v_out = np.array(pts[min(j, len(pts) - 1)]) - np.array(pts[0])
    turn = None
    if np.linalg.norm(v_in) > 1e-3 and np.linalg.norm(v_out) > 1e-6:
        turn = math.degrees(math.acos(float(np.clip(v_in @ v_out / np.linalg.norm(v_in) / np.linalg.norm(v_out), -1, 1))))
    return {'len_m': L, 'min_clear_m': mc, 'retreat_m': float(max(0.0, pts[0][1] - min(p[1] for p in pts))),
            'turn_deg': turn, 'v_in': v_in.tolist(), 'path': [list(p) for p in pts]}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    sw = json.load(open(os.path.join(HERE, 'results', 'door', 'door_wide_offline_sweep_90.json')))
    q0 = np.array(sw['trajectory'][0]['q'], float)
    r0 = TJ.check(q0[:3], 0.0, KC._H_of(0.0) @ TJ.THG, q0[3:])
    assert r0 is not None, '起點握持不可行'
    start = {'base': q0[:3].copy(), 'qa': r0['qa'], 'extra': {}, 'jm': r0['jm'], 'clear': r0['clear']}
    gT = lambda q, ext: KC._H_of(q) @ TJ.THG                                     # noqa: E731
    dA = lambda st: [(dx, dy, dyaw, {}) for dx in (-0.03, 0.0, 0.03) for dy in (-0.03, 0.0, 0.03) for dyaw in (-3, 0, 3)]   # noqa: E731
    sA = lambda b, ext, qd: float(b[0] - 0.3 * max(0.0, -0.95 - b[1]))          # noqa: E731
    layer, qend, why = TJ.beam(start, 0, Q_END, gT, sA, dA, width=6, per_parent=8)
    P = layer[0]['path']
    print(f'持握：{why}，到 {qend}°，底盤 {np.round(P[-1]["base"][:2], 2)}，{time.time() - t0:.0f} s', flush=True)
    mv = [np.array(b['base'][:2]) - np.array(a['base'][:2]) for a, b in zip(P, P[1:])]
    rel, first = [], None
    for k, p in enumerate(P):
        if p['q'] < 40:
            continue
        TJ.cool()
        e = release_eval(P, k)
        rel.append({'q_deg': p['q'], 'base': [float(v) for v in p['base']], 'release': e})
        if first is None and e is not None and e['retreat_m'] <= 1e-9 and e['turn_deg'] is not None and e['turn_deg'] <= TURN_MAX_DEG:
            first = rel[-1]
        s = '不可' if e is None else f"路徑 {e['len_m']:.2f} m、退 {e['retreat_m']:.2f} m、轉向 {e['turn_deg'] if e['turn_deg'] is None else round(e['turn_deg'])}°、最窄 {e['min_clear_m']:.3f}"
        print(f"  q {p['q']}° 底盤 {np.round(p['base'][:2], 2)} 放手直通：{s}", flush=True)
    js = [float(np.max(np.abs(np.array(b['qa']) - np.array(a['qa'])))) for a, b in zip(P, P[1:])]
    rec = {'schema': 'door_coopen_arc/1', 'grasp': 'front (T_HG, door_wide_w2c_30_1)', 'q_end_deg': qend, 'why': why,
           'n_steps': len(P) - 1, 'n_steps_dx_pos': sum(1 for d in mv if d[0] > 1e-3), 'n_steps_dx_neg': sum(1 for d in mv if d[0] < -1e-3),
           'n_steps_stopped': sum(1 for d in mv if np.linalg.norm(d) < 1e-3),
           'joint_step_max_rad': max(js) if js else None, 'joint_margin_min': min(p['jm'] for p in P),
           'clear_min': {k: min(p['clear'][k] for p in P) for k in ('nonfinger', 'finger', 'base', 'self')},
           'first_smooth_release': first, 'turn_max_deg': TURN_MAX_DEG, 'release_by_q': rel, 'trajectory': TJ.ser(P, 'hold'),
           'not_modeled': ['接觸力、夾持柔順', '取樣間碰撞', 'BFS 為格點最短路徑（非平滑）', '放手後門的餘轉', '手臂收攏', '控制器追蹤與延遲']}
    f = first
    print(f"最早平順放手直通：{None if f is None else f['q_deg']}°" +
          ('' if f is None else f"，底盤 {np.round(f['base'][:2], 2)}，放手後 {f['release']['len_m']:.2f} m、轉向 {f['release']['turn_deg']:.0f}°") +
          f"；持握段 +x {rec['n_steps_dx_pos']}／−x {rec['n_steps_dx_neg']}／停 {rec['n_steps_stopped']} 步；{time.time() - t0:.0f} s", flush=True)
    json.dump(rec, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
