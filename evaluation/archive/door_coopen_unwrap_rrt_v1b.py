#!/usr/bin/env python3
"""（v1b＝v1 加大預算：15000 次、延伸步 0.10 rad；v1 4000 次未找到）連續協同通行：放手後手臂繞回門扇自由端到 K2_READY 的關節空間路徑 v1（RRT-connect；離線靜態，無 Isaac）。

背景：door_coopen_pass_smooth_v2 的張爪＋沿工具 −z 退 60 mm 已成立（δ=0），但沿門扇外移在 5 cm 內 IK 失敗；
固定或放寬夾爪姿態的直線退法（scratchpad 探測）都在外移第 1 步失敗。退開後手臂 j1 2.82 rad（上限 2.89），收到 STOW 需 j1 回 0。
本檔：起點＝pass_smooth_v2 退開 60 mm 的最後一點（底盤固定於該點位置與偏航）、終點＝TJ.STOW；門角 q_rel。
  RRT-connect（6 維，關節界限 V2.ELO＋0.05／V2.EHI−0.05，取樣偏向終點 10%，延伸步 0.15 rad，邊每 0.03 rad 檢核），
  碰撞判準：非手指／手指（張爪開度）／自碰 ≥ 5 mm（門角 q_rel）；找到後做隨機捷徑（200 次），再以門角 q_rel 與 q_rel − 2°
  每 0.02 rad 重核整條路徑，報最小淨空、關節總行程、各關節最大行程與以 1.0 rad/s 計的最短時長。亂數種子固定（可重現）。
**界線**：底盤固定（收臂期間底盤是否同時前進另核）；靜態；不含動力學與控制器追蹤。

    python3 evaluation/door_coopen_unwrap_rrt_v1b.py   # 寫 results/door/door_coopen_unwrap_rrt_v1b.json（不覆寫）
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
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_smooth_v2.json')
SRC_P = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_smooth_v1.json')
SRC_H = os.path.join(HERE, 'results', 'door', 'door_coopen_edgehold_v2.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_unwrap_rrt_v1b.json')
C, V2 = RV.C, RV.V2
LO, HI = V2.ELO + 0.05, V2.EHI - 0.05
TH = 0.005
STEP, RES = 0.10, 0.03
N_ITER = 15000


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    rng = np.random.default_rng(0)
    d = json.load(open(SRC))
    pv = json.load(open(SRC_P))
    h = json.load(open(SRC_H))
    q_rel = d['q_rel_deg']
    yaw = next(r for r in h['per_end'] if r['q_rel_deg'] == q_rel)['trajectory'][-1]['base'][2]
    r0 = [p for p in d['per_point'] if p['phase'] == 'retract'][-1]
    k = min(int(round(r0['s_m'] / 0.01)), len(pv['path']) - 1)
    base = [pv['path'][k][0], pv['path'][k][1], yaw]
    qs, qg = np.array(r0['qa']), TJ.STOW.copy()
    n_chk = [0]

    def free(qa, dq=0.0):
        n_chk[0] += 1
        if np.any(qa < V2.ELO) or np.any(qa > V2.EHI):
            return False
        g = TJ.GEO(base + list(qa), math.radians(q_rel + dq), C.FINGER_OPEN)
        return min(g['nonfinger'][0], g['finger'][0], g['self'][0]) >= TH

    def edge_free(a, b, dq=0.0, res=RES):
        n = max(1, int(math.ceil(np.max(np.abs(b - a)) / res)))
        return all(free(a + (b - a) * t, dq) for t in np.linspace(0, 1, n + 1)[1:])

    assert free(qs), '起點不可行'
    assert free(qg), '終點不可行'
    TA, TB = [(qs, -1)], [(qg, -1)]

    def nearest(T, x):
        return int(np.argmin([np.max(np.abs(n_[0] - x)) for n_ in T]))

    def extend(T, x):
        i = nearest(T, x)
        a = T[i][0]
        v = x - a
        L = np.max(np.abs(v))
        b = x if L <= STEP else a + v * (STEP / L)
        if edge_free(a, b):
            T.append((b, i))
            return b, (L <= STEP)
        return None, False

    def connect(T, x):
        while True:
            TJ.cool()
            b, reached = extend(T, x)
            if b is None:
                return False
            if reached:
                return True

    path = None
    for it in range(N_ITER):
        TJ.cool()
        x = qg if rng.random() < 0.10 else rng.uniform(LO, HI)
        b, _ = extend(TA, x)
        if b is not None and connect(TB, b):
            def chain(T, i):
                out = []
                while i != -1:
                    out.append(T[i][0])
                    i = T[i][1]
                return out
            pa = chain(TA, len(TA) - 1)[::-1]
            pb = chain(TB, len(TB) - 1)
            path = pa + pb[1:]
            if np.allclose(path[0], qg):
                path = path[::-1]
            break
        TA, TB = TB, TA
    if path is None:
        print(f'RRT 未找到（{N_ITER} 次，檢核 {n_chk[0]}）；{time.time() - t0:.0f} s', flush=True)
        json.dump({'schema': 'door_coopen_unwrap_rrt/1', 'found': False, 'n_iter': N_ITER, 'n_checks': n_chk[0]}, open(OUT, 'w'), ensure_ascii=False, indent=1)
        return
    if not np.allclose(path[0], qs):
        path = path[::-1]
    print(f'RRT 找到：{len(path)} 節點、{it + 1} 次、檢核 {n_chk[0]}；{time.time() - t0:.0f} s', flush=True)
    for _ in range(200):
        TJ.cool()
        if len(path) <= 2:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if edge_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    dense = [path[0]]
    for a, b in zip(path, path[1:]):
        n = max(1, int(math.ceil(np.max(np.abs(b - a)) / 0.02)))
        dense += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    mins = {}
    for dq in (0.0, -2.0):
        m = {'nonfinger': (1.0, None), 'finger': (1.0, None), 'self': (1.0, None)}
        for qa in dense:
            TJ.cool()
            g = TJ.GEO(base + list(qa), math.radians(q_rel + dq), C.FINGER_OPEN)
            for c in m:
                if g[c][0] < m[c][0]:
                    m[c] = (float(g[c][0]), g[c][1])
        mins[f'q{q_rel + dq:.0f}'] = m
    travel = np.sum(np.abs(np.diff(np.array(dense), axis=0)), axis=0)
    rec = {'schema': 'door_coopen_unwrap_rrt/1', 'found': True, 'q_rel_deg': q_rel, 'base': base, 'start_qa': qs.tolist(), 'goal_qa': qg.tolist(),
           'n_waypoints': len(path), 'waypoints': [p.tolist() for p in path], 'n_dense': len(dense), 'min_clear': mins,
           'joint_travel_rad': travel.tolist(), 'time_lb_s_at_1rad_s': float(np.max(travel) / 1.0),
           'time_lb_s_at_0p5rad_s': float(np.max(travel) / 0.5), 'n_checks': n_chk[0], 'seed': 0,
           'not_modeled': ['底盤固定（未與通過段同時）', '動力學與追蹤', '取樣間碰撞（0.02 rad）']}
    print(f"捷徑後 {len(path)} 節點；最小淨空 {({k_: {c: round(v[0] * 1000, 1) for c, v in m.items()} for k_, m in mins.items()})} mm；"
          f"關節行程 {np.round(travel, 2)}；最短時長 {rec['time_lb_s_at_1rad_s']:.1f} s（1 rad/s）；{time.time() - t0:.0f} s", flush=True)
    json.dump(rec, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
