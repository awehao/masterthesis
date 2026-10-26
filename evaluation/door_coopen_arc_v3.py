#!/usr/bin/env python3
"""連續協同通行：去程「邊開邊繞」軌跡 v3——v2 的 DP 規劃不變，驗證改為「細分＋時間縮放」（無 Isaac、無接觸力；單一程序＋溫度暫停）。

**揭露**：v2（door_coopen_arc_v2.json）四個放手門角都驗證失敗；診斷（45° 的第一條 DP 路徑）顯示失敗於 17°：底盤開始往 +x 移 0.01 m 時
手臂一步變化 0.124 rad，超過 door_coopen_traj_v1 的 joint_step_rad 0.10（每 1° 門角）；IK、淨空、關節餘裕都通過。
0.10 rad/° 原是連續性代理（門 1°/s 時手臂 ≤ 0.10 rad/s，為手臂上限 1.0 rad/s 的 10%），不是物理限制。
本版**不放寬**每步 0.10 rad，而是把超過的 1° 步二分細分（最多 4 層＝1/16°），每個子步以上一子步為種子解 IK（only_prev）、
各子點重核全部淨空與關節餘裕、子步手臂變化 ≤ 0.10 rad；細分到底仍超過即判失敗（視為分支跳躍或真不連續）。
時間：每子步耗時＝max(門角變化 / 1 °/s, 手臂最大變化 / 0.10 rad/s, 底盤位移 / 0.10 m/s, 偏航變化 / 15 °/s)；報總時長與最慢段。
其餘（節點、起點、邊、成本、終點、禁邊重做 DP 至多 40 次）同 door_coopen_arc_v2。

    python3 evaluation/door_coopen_arc_v3.py   # 寫 results/door/door_coopen_arc_v3.json（不覆寫）
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
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k2_check as KC                             # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_arc_v3.json')
C, V2 = RV.C, RV.V2
STEP_RAD = TJ.LIM['joint_step_rad']
DEPTH = 4
RATE = {'door_deg_s': 1.0, 'arm_rad_s': 0.10, 'base_m_s': 0.10, 'yaw_deg_s': 15.0}


def pose_ok(base, qd, seed):
    q = math.radians(qd)
    if not RV.disc_ok(float(base[0]), float(base[1]), q):
        return None, 'disc'
    qf9 = C.ik(base, KC._H_of(q) @ TJ.THG, seed, only_prev=seed is not None)
    if qf9 is None:
        return None, 'ik'
    qa = np.asarray(qf9, float)[C.IDX]
    jm = float(min(min(qa - V2.ELO + 0.05), min(V2.EHI + 0.05 - qa)))
    if jm < TJ.LIM['joint_margin_rad']:
        return None, 'joint_margin'
    d = TJ.GEO([float(v) for v in list(base) + list(qa)], q, RV.Q_FINGER)
    for k, lim in (('nonfinger', TJ.LIM['nonfinger_m']), ('finger', TJ.LIM['finger_m']), ('base', TJ.LIM['base_m']), ('self', TJ.LIM['self_m'])):
        if d[k][0] < lim:
            return None, k
    return {'qa': qa, 'jm': jm, 'clear': {k: float(d[k][0]) for k in ('nonfinger', 'finger', 'base', 'self')}}, None


def seg(a, b, qa_, qb, seed, depth):
    """a、b＝(x, y, yaw_rad)；從 seed（a 的手臂解）走到 b；回傳 [子點...] 或 (None, 原因)。"""
    r, why = pose_ok(np.array(b), qb, seed)
    if r is not None and float(np.max(np.abs(r['qa'] - seed))) <= STEP_RAD:
        return [dict(r, q=qb, base=np.array(b))], None
    if depth >= DEPTH:
        return None, why or 'joint_step'
    dy = (b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi
    m = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), a[2] + 0.5 * dy)
    qm = 0.5 * (qa_ + qb)
    first, why = seg(a, m, qa_, qm, seed, depth + 1)
    if first is None:
        return None, why
    second, why = seg(m, b, qm, qb, first[-1]['qa'], depth + 1)
    if second is None:
        return None, why
    return first + second, None


def verify(layers, chain):
    a0 = layers[0][chain[0]]
    b0 = (a0[0], a0[1], math.radians(a0[2]))
    r0 = TJ.check(np.array(b0), 0.0, KC._H_of(0.0) @ TJ.THG, None, full=True)
    if r0 is None:
        return [], (1, chain[0], chain[1]), 0, 'start'
    out = [dict(r0, q=0.0, base=np.array(b0))]
    for k in range(len(chain) - 1):
        a, b = layers[k][chain[k]], layers[k + 1][chain[k + 1]]
        for s in range(1, 6):
            TJ.cool()
            dyw = (b[2] - a[2] + 180.0) % 360.0 - 180.0
            pb = (a[0] + (b[0] - a[0]) * s / 5.0, a[1] + (b[1] - a[1]) * s / 5.0, math.radians(a[2] + dyw * s / 5.0))
            pa = tuple(out[-1]['base'])
            sub, why = seg(pa, pb, out[-1]['q'], 5.0 * k + s, out[-1]['qa'], 0)
            if sub is None:
                return out, (k + 1, chain[k], chain[k + 1]), 5 * k + s, why
            out.extend(sub)
    return out, None, None, None


def timing(path):
    dts = []
    for a, b in zip(path, path[1:]):
        dyw = abs(math.degrees((b['base'][2] - a['base'][2] + math.pi) % (2 * math.pi) - math.pi))
        dts.append(max(abs(b['q'] - a['q']) / RATE['door_deg_s'], float(np.max(np.abs(b['qa'] - a['qa']))) / RATE['arm_rad_s'],
                       math.hypot(b['base'][0] - a['base'][0], b['base'][1] - a['base'][1]) / RATE['base_m_s'], dyw / RATE['yaw_deg_s']))
    return dts


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    d = json.load(open(A2.SRC))
    layers = [[(f['x'], f['y'], float(f['yaw_deg'])) for f in r['feasible']] for r in d['per_q']]
    res, term_cache = [], {}
    for q_r in A2.Q_RS:
        banned, why, path, sol, fails = set(), 'dp_no_path', None, None, {}
        for it in range(40):
            sol = A2.dp(layers, banned, q_r, term_cache)
            if sol is None:
                why = 'dp_no_path'
                break
            path, bad, qbad, reason = verify(layers, sol[1])
            if bad is None:
                why = 'verified'
                break
            banned.add(bad)
            fails[reason] = fails.get(reason, 0) + 1
            why = f'verify_fail_last_at_{qbad}_{reason}'
        rec = {'q_release_deg': q_r, 'status': why, 'n_dp_iters': it + 1, 'verify_fail_reasons': fails}
        if why == 'verified':
            mv = [np.array(b['base'][:2]) - np.array(a['base'][:2]) for a, b in zip(path, path[1:])]
            js = [float(np.max(np.abs(b['qa'] - a['qa']))) for a, b in zip(path, path[1:])]
            dts = timing(path)
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
    json.dump({'schema': 'door_coopen_arc/3', 'source': os.path.relpath(A2.SRC, HERE), 'start': A2.START, 'joint_step_rad': STEP_RAD,
               'subdivide_depth': DEPTH, 'rates': RATE, 'per_q_release': res,
               'disclosure': 'v2 驗證以每 1° 手臂 ≤ 0.10 rad 判定，全數失敗；本版不放寬該值，改為細分步並以時間縮放計時（看過 v2 結果後的驗證方式變更）',
               'not_modeled': ['格點與內插間碰撞', '接觸力、夾持柔順', 'BFS 為格點最短路徑（非平滑）', '放手後門的餘轉', '手臂收攏', '控制器追蹤與延遲']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
