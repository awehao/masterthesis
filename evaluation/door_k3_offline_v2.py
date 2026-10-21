#!/usr/bin/env python3
"""K3 前置離線核對 v2（依 Codex 20261021 K3 draft-1 審查；v1 輸出 door_k3_offline.json 保留，其 +1°/+3° 結果改標「轉後重新對準的退離核對」）。

情境＝**門獨立餘轉、機械臂不跟門走**：
  手臂姿態只由「釋放時門角 θ_goal 下的退離目標」決定：T(s) = H(θ_goal)·Trans(s·a_H)·T_HG，s ∈ {0, 0.01, …, 0.10}（張爪）；
  門角另外取 θ_goal + Δ，Δ ∈ {0, 0.25, 0.5, …, 5.0}°；核對手臂各連桿（**含手指—橫桿，張爪後不再豁免**）、夾爪殼、相機對門扇／立柱／橫桿／牆，
  以及底盤圓盤對門扇。輸出淨空地圖 clearance(s, Δ) 與：
    Δ_safe(s)＝在退離量 s 下仍保持環境 ≥ 5 mm、底盤 ≥ 2 cm 的最大餘轉角；
    s_clear(Δ_max)＝要承受 Δ_max 餘轉所需的最小退離量。
θ_goal＝8°；名目停位＋27 組停車誤差（±2 cm、±2°）。近似：點雲／膠囊體（橫桿）／圓盤，取樣間未核。

    python3 evaluation/door_k3_offline_v2.py     # 輸出 results/door/door_k3_offline_v2.json（已存在則拒絕覆寫）
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_offline_v2 as V2                       # noqa: E402
import door_geom_v2 as G2                             # noqa: E402

C, K1 = V2.C, V2.K1
OUT = os.path.join(HERE, 'results', 'door', 'door_k3_offline_v2.json')
THETA = 8.0
S_GRID = [round(x, 3) for x in np.linspace(0.0, 0.10, 11)]
D_GRID = [round(x, 2) for x in np.arange(0.0, 5.0001, 0.25)]
ENV_M, BASE_M = 0.005, 0.02


def clearance(qf, q_door):
    """手臂全部連桿（張爪）對門扇各部（含手指—橫桿）、牆；底盤圓盤對門扇／牆。"""
    W = C.world(qf, C.ARM_LINKS, C.FINGER_OPEN)
    sh = G2.shapes(K1.D2, q_door, K1.H)
    env, who = np.inf, None
    for n, P in W.items():
        for name, c, R, hs in sh:
            d = float(G2.box_dist(P, c, R, hs).min())
            if d < env:
                env, who = d, f'{n}|{name}'
        d = float(G2.bar_capsule_dist(K1.D2, P, q_door, K1.H).min())
        if d < env:
            env, who = d, f'{n}|bar'
    bc = min(G2.disc_clear(qf[0], qf[1], c[2], C.RB, c, R, hs) for name, c, R, hs in sh)
    return env, who, bc


def case(off):
    base = V2.base_of(*off)
    q2 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k2_offline_v2.json')))['Q_PRE_DOOR']
    q0 = C.ik(base, C.target(0.0, 0.0), np.array(q2))
    prev = q0[C.IDX]
    for qd in np.arange(0.1, THETA + 1e-9, 0.1):
        prev = C.ik(base, C.target(math.radians(qd), 0.0), prev, only_prev=True)[C.IDX]
    grid, arm = [], {}
    for s in S_GRID:
        qf = C.ik(base, C.target(math.radians(THETA), s), prev, only_prev=True)
        if qf is None:
            grid.append({'s': s, 'ik': False})
            continue
        prev = qf[C.IDX]
        arm[s] = prev.tolist()
        row = {'s': s, 'ik': True, 'joint_margin': C.jm(prev)[0], 'by_delta': []}
        for dd in D_GRID:
            env, who, bc = clearance(qf, math.radians(THETA + dd))
            row['by_delta'].append({'delta_deg': dd, 'env': env, 'env_who': who, 'base_clear': bc, 'ok': env >= ENV_M and bc >= BASE_M})
        okd = [x['delta_deg'] for x in row['by_delta']]
        # Δ_safe(s)：自 0 起連續可行的最大 Δ
        dsafe = None
        for x in row['by_delta']:
            if x['ok']:
                dsafe = x['delta_deg']
            else:
                break
        row['delta_safe_deg'] = dsafe
        grid.append(row)
    s_clear = {}
    for dmax in (0.5, 1.0, 2.0, 3.0, 5.0):
        s_clear[f'{dmax:g}'] = next((r['s'] for r in grid if r.get('ik') and r['delta_safe_deg'] is not None and r['delta_safe_deg'] >= dmax), None)
    return {'offset': off, 'grid': grid, 's_clear_for_delta': s_clear, 'arm_q_by_s': arm}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    res = {'schema': 'door_k3_offline/2', 'theta_goal_deg': THETA, 's_grid': S_GRID, 'delta_grid_deg': D_GRID,
           'spec_sha256': hashlib.sha256(open(K1.SPEC_P, 'rb').read()).hexdigest(), 'cases': [],
           'v1_relabel': 'door_k3_offline.json 的 +1°/+3° 結果＝「轉後重新對準的退離核對」，不支持手臂不跟門的解除後情境'}
    for off in V2.OFFS:
        c = case(off)
        res['cases'].append(c)
        if off == (0.0, 0.0, 0.0):
            print('名目停位 Δ_safe(s)：', {r['s']: r.get('delta_safe_deg') for r in c['grid']}, flush=True)
            r0 = c['grid'][0]
            print('  s=0（原地）各 Δ 環境淨空 mm：', {x['delta_deg']: round(x['env'] * 1e3, 2) for x in r0['by_delta'][:6]}, r0['by_delta'][2]['env_who'], flush=True)
    worst = {}
    for k in ('0.5', '1', '2', '3', '5'):
        vals = [c['s_clear_for_delta'][k] for c in res['cases']]
        worst[k] = None if any(v is None for v in vals) else max(vals)
    res['summary'] = {'s_clear_worst_over_27_m': worst,
                      'delta_safe_at_s0_min_deg': min((c['grid'][0].get('delta_safe_deg') if c['grid'][0].get('delta_safe_deg') is not None else -1) for c in res['cases'])}
    print('27 組最差：承受餘轉 Δ 所需最小退離量 s_clear（m）', worst, '；原地（s=0）可承受最小 Δ', res['summary']['delta_safe_at_s0_min_deg'], flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
