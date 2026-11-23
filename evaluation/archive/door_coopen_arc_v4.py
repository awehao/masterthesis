#!/usr/bin/env python3
"""連續協同通行：去程「邊開邊繞」軌跡 v4——DP 只走持握可行域的內部節點（無 Isaac、無接觸力；單一程序＋溫度暫停）。

**揭露**：v3（door_coopen_arc_v3.json）放手 45／50／55° 分別在 18／22／27° 前後驗證失敗，原因全為 IK；診斷顯示失敗點的內插底盤位姿
連多種子完整 IK 也無解——DP 節點貼著可行域邊緣（成本把底盤往 +x 壓到極限），格點節點可行而節點間內插已出界。
本版改兩處，其餘（細分＋時間縮放驗證、終點判準、起點）同 v3：
  1. 節點只取「內部節點」：同一門角、同一偏航下 x ±0.05、y ±0.05 四鄰也都可行（侵蝕一格）；起點節點不受此限。
  2. 驗證失敗時禁用失敗段的終點節點（v2／v3 只禁該邊），至多 60 次；偏航變化成本由 0.005 降為 0.002 /°（讓底盤可轉向配合手臂）。

    python3 evaluation/door_coopen_arc_v4.py   # 寫 results/door/door_coopen_arc_v4.json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_arc_v2 as A2                        # noqa: E402
import door_coopen_arc_v3 as A3                        # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_arc_v4.json')
GRID = 0.05
N_ITER = 60


def edge_cost(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    return L + 10.0 * max(0.0, -dx) + (0.2 if L < 0.025 else 0.0) + 0.002 * A2.dyaw(a[2], b[2])


def interior(layer):
    s = {(round(x / GRID), round(y / GRID), round(yw)) for x, y, yw in layer}
    return [n for n in layer if all((round(n[0] / GRID) + i, round(n[1] / GRID) + j, round(n[2])) in s
                                    for i, j in ((1, 0), (-1, 0), (0, 1), (0, -1)))]


def dp(layers, banned, q_r, term_cache):
    K = q_r // 5
    s0 = min(range(len(layers[0])), key=lambda i: math.hypot(layers[0][i][0] - A2.START[0], layers[0][i][1] - A2.START[1]) + 0.01 * A2.dyaw(layers[0][i][2], A2.START[2]))
    cost = [{s0: (0.0, None)}]
    for k in range(1, K + 1):
        cur = {}
        for i, b in enumerate(layers[k]):
            if (k, i) in banned:
                continue
            best = None
            for j, (c, _) in cost[k - 1].items():
                a = layers[k - 1][j]
                if math.hypot(b[0] - a[0], b[1] - a[1]) > A2.STEP_M or A2.dyaw(a[2], b[2]) > A2.STEP_YAW:
                    continue
                cc = c + edge_cost(a, b)
                if best is None or cc < best[0]:
                    best = (cc, j)
            if best is not None:
                cur[i] = best
        cost.append(cur)
        if not cur:
            return None
    best = None
    for i, (c, j) in cost[K].items():
        key = (q_r, i, j)
        if key not in term_cache:
            term_cache[key] = A2.terminal(q_r, layers[K][i], layers[K - 1][j])
        t = term_cache[key]
        if t is not None and (best is None or c + t['turn_deg'] / 90.0 + 0.2 * t['len_m'] < best[0]):
            best = (c + t['turn_deg'] / 90.0 + 0.2 * t['len_m'], i, t)
    if best is None:
        return None
    chain, i = [best[1]], best[1]
    for k in range(K, 0, -1):
        i = cost[k][i][1]
        chain.append(i)
    chain.reverse()
    return best[0], chain, best[2]


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    d = json.load(open(A2.SRC))
    raw = [[(f['x'], f['y'], float(f['yaw_deg'])) for f in r['feasible']] for r in d['per_q']]
    layers = [raw[0]] + [interior(L) for L in raw[1:]]
    print('內部節點數（每 5°）：' + ' '.join(str(len(L)) for L in layers[:13]), flush=True)
    res, term_cache = [], {}
    for q_r in A2.Q_RS:
        banned, why, path, sol, fails = set(), 'dp_no_path', None, None, {}
        for it in range(N_ITER):
            sol = dp(layers, banned, q_r, term_cache)
            if sol is None:
                why = 'dp_no_path'
                break
            path, bad, qbad, reason = A3.verify(layers, sol[1])
            if bad is None:
                why = 'verified'
                break
            banned.add((bad[0], bad[2]))
            fails[reason] = fails.get(reason, 0) + 1
            why = f'verify_fail_last_at_{qbad}_{reason}'
        rec = {'q_release_deg': q_r, 'status': why, 'n_dp_iters': it + 1, 'verify_fail_reasons': fails}
        if why == 'verified':
            mv = [np.array(b['base'][:2]) - np.array(a['base'][:2]) for a, b in zip(path, path[1:])]
            js = [float(np.max(np.abs(b['qa'] - a['qa']))) for a, b in zip(path, path[1:])]
            dts = A3.timing(path)
            t = sol[2]
            rec.update({'cost': sol[0], 'nodes': [list(layers[k][i]) for k, i in enumerate(sol[1])], 'n_points': len(path),
                        'n_subdivided_points': sum(1 for p in path if abs(p['q'] - round(p['q'])) > 1e-9),
                        'n_steps_dx_pos': sum(1 for m in mv if m[0] > 1e-3), 'n_steps_dx_neg': sum(1 for m in mv if m[0] < -1e-3),
                        'n_steps_stopped': sum(1 for m in mv if np.linalg.norm(m) < 1e-3),
                        'hold_duration_s': float(sum(dts)), 'slowest_step': {'q_deg': path[int(np.argmax(dts)) + 1]['q'], 'dt_s': float(max(dts))},
                        'release': {k: t[k] for k in ('len_m', 'min_clear_m', 'retreat_m', 'turn_deg')}, 'release_path': t['path'],
                        'joint_step_max_rad': max(js), 'joint_margin_min': min(p['jm'] for p in path),
                        'clear_min': {c: min(p['clear'][c] for p in path) for c in ('nonfinger', 'finger', 'base', 'self')},
                        'trajectory': TJ.ser(path, 'hold')})
            print(f"放手 {q_r}°：已驗證（DP {it + 1} 次，細分點 {rec['n_subdivided_points']}）；持握 +x {rec['n_steps_dx_pos']}／−x {rec['n_steps_dx_neg']}／停 {rec['n_steps_stopped']} 步；"
                  f"終點 {np.round(path[-1]['base'][:2], 2)} 偏航 {math.degrees(path[-1]['base'][2]):.0f}°；持握 {rec['hold_duration_s']:.0f} s；"
                  f"放手轉向 {t['turn_deg']:.0f}°、後續 {t['len_m']:.2f} m、最窄 {t['min_clear_m']:.3f}；淨空 {({c: round(v, 3) for c, v in rec['clear_min'].items()})}；{time.time() - t0:.0f} s",
                  flush=True)
        else:
            print(f'放手 {q_r}°：{why}（DP {it + 1} 次，失敗原因 {fails}）；{time.time() - t0:.0f} s', flush=True)
        res.append(rec)
    json.dump({'schema': 'door_coopen_arc/4', 'source': os.path.relpath(A2.SRC, HERE), 'start': A2.START, 'interior_erosion_m': GRID,
               'joint_step_rad': A3.STEP_RAD, 'subdivide_depth': A3.DEPTH, 'rates': A3.RATE, 'per_q_release': res,
               'disclosure': 'v3 失敗點內插位姿完整 IK 無解（節點貼可行域邊緣）；本版只走內部節點、禁節點重做；看過 v3 結果後的規劃方式變更',
               'not_modeled': ['格點與內插間碰撞', '接觸力、夾持柔順', 'BFS 為格點最短路徑（非平滑）', '放手後門的餘轉', '手臂收攏', '控制器追蹤與延遲']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
