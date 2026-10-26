#!/usr/bin/env python3
"""連續協同通行：去程「邊開邊繞」軌跡 v2——在持握可行域上做 DP，再以 1° 細步驗證（無 Isaac、無接觸力；單一程序＋溫度暫停）。

v1（door_coopen_arc_v1，貪婪 beam、評分往 +x）在 26° 走進死路；本版改為整體規劃：
  節點＝door_coopen_reach_v1 每 5° 的正面夾法持握可行底盤位姿（x、y 每 0.05 m、偏航每 30°）；
  起點＝q 0 最接近 W-GMPC 實跑起點 (0.60, −0.53, 90°) 的節點；
  邊（相鄰 5°）：底盤位移 ≤ 0.15 m、偏航變化 ≤ 30°；成本＝位移長 ＋ 10·往 −x 的量（往左跟門走＝不是在繞）＋ 0.2·停住（位移 < 0.025 m）＋ 0.005·|Δ偏航°|；
  終點（放手門角 Q_R ∈ {45, 50, 55, 60}）：門角固定為 Q_R 時以 door_coopen_arc_v1.bfs_path 可直通（y ≥ 0.45，底盤 ≥ 20 mm）、
    後退量 ≤ 0.05 m（格點 0.02 m 的量化）；終點成本＝轉向角/90 ＋ 0.2·放手後路徑長。
驗證：最佳路徑的相鄰節點間以線性內插（x、y、偏航）每 1° 一步，door_coopen_traj_v1.check（IK 以上一步為種子、相鄰步手臂 ≤ 0.10 rad、
  全部淨空判準）；某段不通即禁用該邊重做 DP（至多 40 次）。
報：每個 Q_R 的已驗證路徑（或失敗原因）、持握段底盤往 +x／−x／停住的 1° 步數、放手時轉向角、放手後路徑長與最窄淨空、
  最小淨空、關節餘裕、相鄰步最大關節變化。
**界線**：格點取樣（取樣間未核）、剛性夾持、靜態；BFS 為格點最短路徑（非平滑）；放手後門視為靜止；手臂收攏與張爪後門餘轉另核。

    python3 evaluation/door_coopen_arc_v2.py   # 寫 results/door/door_coopen_arc_v2.json（不覆寫）
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
import door_coopen_arc_v1 as AR                       # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k2_check as KC                             # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_reach_v1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_arc_v2.json')
START = (0.60, -0.53, 90.0)
Q_RS = (45, 50, 55, 60)
STEP_M, STEP_YAW = 0.15, 30.0


def dyaw(a, b):
    return abs((b - a + 180.0) % 360.0 - 180.0)


def edge_cost(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    return L + 10.0 * max(0.0, -dx) + (0.2 if L < 0.025 else 0.0) + 0.005 * dyaw(a[2], b[2])


def terminal(qd, node, prev):
    r = AR.bfs_path(qd, node)
    if r is None:
        return None
    pts, mc = r
    retreat = max(0.0, node[1] - min(p[1] for p in pts))
    if retreat > 0.05:
        return None
    L = sum(math.hypot(a[0] - c[0], a[1] - c[1]) for a, c in zip(pts, pts[1:]))
    acc, j = 0.0, 1
    while j < len(pts) - 1 and acc < 0.30:
        acc += math.hypot(pts[j][0] - pts[j - 1][0], pts[j][1] - pts[j - 1][1])
        j += 1
    v_out = np.array(pts[min(j, len(pts) - 1)]) - np.array(pts[0])
    v_in = np.array(node[:2]) - np.array(prev[:2])
    turn = 90.0 if np.linalg.norm(v_in) < 1e-6 else math.degrees(math.acos(float(np.clip(v_in @ v_out / np.linalg.norm(v_in) / np.linalg.norm(v_out), -1, 1))))
    return {'len_m': L, 'min_clear_m': mc, 'retreat_m': retreat, 'turn_deg': turn, 'path': [list(p) for p in pts]}


def dp(layers, banned, q_r, term_cache):
    """layers[k]＝q 5k 的節點 list；回傳 (成本, 節點索引鏈, 終點資訊)。"""
    K = q_r // 5
    s0 = min(range(len(layers[0])), key=lambda i: math.hypot(layers[0][i][0] - START[0], layers[0][i][1] - START[1]) + 0.01 * dyaw(layers[0][i][2], START[2]))
    cost = [{s0: (0.0, None)}]
    for k in range(1, K + 1):
        cur = {}
        for i, b in enumerate(layers[k]):
            best = None
            for j, (c, _) in cost[k - 1].items():
                a = layers[k - 1][j]
                if (k, j, i) in banned or math.hypot(b[0] - a[0], b[1] - a[1]) > STEP_M or dyaw(a[2], b[2]) > STEP_YAW:
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
            term_cache[key] = terminal(q_r, layers[K][i], layers[K - 1][j])
        t = term_cache[key]
        if t is None:
            continue
        tc = c + t['turn_deg'] / 90.0 + 0.2 * t['len_m']
        if best is None or tc < best[0]:
            best = (tc, i, t)
    if best is None:
        return None
    chain, i = [best[1]], best[1]
    for k in range(K, 0, -1):
        i = cost[k][i][1]
        chain.append(i)
    chain.reverse()
    return best[0], chain, best[2]


def verify(layers, chain):
    seed, out = None, []
    for k in range(len(chain) - 1):
        a, b = layers[k][chain[k]], layers[k + 1][chain[k + 1]]
        for s in range(0 if k == 0 else 1, 6):
            TJ.cool()
            qd = 5 * k + s
            yaw = a[2] + ((b[2] - a[2] + 180.0) % 360.0 - 180.0) * s / 5.0
            base = np.array([a[0] + (b[0] - a[0]) * s / 5.0, a[1] + (b[1] - a[1]) * s / 5.0, math.radians(yaw)])
            q = math.radians(qd)
            r = TJ.check(base, q, KC._H_of(q) @ TJ.THG, seed, full=seed is None)
            if r is None:
                return out, (k + 1, chain[k], chain[k + 1]), qd
            seed = r['qa']
            out.append({'q': qd, 'base': base, 'qa': r['qa'], 'jm': r['jm'], 'clear': r['clear']})
    return out, None, None


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    d = json.load(open(SRC))
    layers = [[(f['x'], f['y'], float(f['yaw_deg'])) for f in r['feasible']] for r in d['per_q']]
    assert [r['q_deg'] for r in d['per_q']][:13] == list(range(0, 65, 5))
    res, term_cache = [], {}
    for q_r in Q_RS:
        banned, why, path, sol = set(), 'dp_no_path', None, None
        for it in range(40):
            sol = dp(layers, banned, q_r, term_cache)
            if sol is None:
                why = 'dp_no_path'
                break
            path, bad, qbad = verify(layers, sol[1])
            if bad is None:
                why = 'verified'
                break
            banned.add(bad)
            why = f'verify_fail_last_at_{qbad}'
        rec = {'q_release_deg': q_r, 'status': why, 'n_dp_iters': it + 1}
        if why == 'verified':
            mv = [np.array(b['base'][:2]) - np.array(a['base'][:2]) for a, b in zip(path, path[1:])]
            js = [float(np.max(np.abs(np.array(b['qa']) - np.array(a['qa'])))) for a, b in zip(path, path[1:])]
            t = sol[2]
            rec.update({'cost': sol[0], 'nodes': [list(layers[k][i]) for k, i in enumerate(sol[1])],
                        'n_steps_dx_pos': sum(1 for m in mv if m[0] > 1e-3), 'n_steps_dx_neg': sum(1 for m in mv if m[0] < -1e-3),
                        'n_steps_stopped': sum(1 for m in mv if np.linalg.norm(m) < 1e-3),
                        'release': {k: t[k] for k in ('len_m', 'min_clear_m', 'retreat_m', 'turn_deg')}, 'release_path': t['path'],
                        'joint_step_max_rad': max(js), 'joint_margin_min': min(p['jm'] for p in path),
                        'clear_min': {c: min(p['clear'][c] for p in path) for c in ('nonfinger', 'finger', 'base', 'self')},
                        'trajectory': TJ.ser(path, 'hold')})
            print(f"放手 {q_r}°：已驗證（DP {it + 1} 次）；持握 +x {rec['n_steps_dx_pos']}／−x {rec['n_steps_dx_neg']}／停 {rec['n_steps_stopped']} 步；"
                  f"終點 {np.round(path[-1]['base'][:2], 2)}；放手轉向 {t['turn_deg']:.0f}°、後續 {t['len_m']:.2f} m、最窄 {t['min_clear_m']:.3f}；"
                  f"淨空 {({c: round(v, 3) for c, v in rec['clear_min'].items()})}；關節步 {rec['joint_step_max_rad']:.3f}；{time.time() - t0:.0f} s", flush=True)
        else:
            print(f'放手 {q_r}°：{why}（DP {it + 1} 次）；{time.time() - t0:.0f} s', flush=True)
        res.append(rec)
    json.dump({'schema': 'door_coopen_arc/2', 'source': os.path.relpath(SRC, HERE), 'start': START, 'step_limits': {'m_per_5deg': STEP_M, 'yaw_deg_per_5deg': STEP_YAW},
               'per_q_release': res,
               'not_modeled': ['格點與內插間碰撞', '接觸力、夾持柔順', 'BFS 為格點最短路徑（非平滑）', '放手後門的餘轉', '手臂收攏', '控制器追蹤與延遲']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
