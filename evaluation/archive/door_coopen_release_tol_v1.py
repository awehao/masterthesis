#!/usr/bin/env python3
"""連續協同通行：張爪／收臂期間可容許的門餘轉 v1（讀 door_coopen_pass_smooth_v1.json；離線靜態）。

pass_smooth_v1 以門角包絡 [q_rel − 2°, q_rel + 5°] 核收臂，得非手指／手指 −20 mm（門角 60°）——門在張爪時仍往開的方向續轉，
橫桿與門扇會推向尚未退開的夾爪。本檔對收臂每個取樣點（張爪 3 點、沿 −z 退 6 點、內插到 STOW 30 點）逐一以門角 q_rel + δ
（δ ∈ {−2, −1, −0.5, 0, 0.5, 1, 2, 3, 5}°）核非手指／手指／自碰 ≥ 5 mm，報每點可容許的 δ 範圍（自 0 連續成立的最大正／負值），
以及各階段的最小可容許值。用途：決定放手前門角速度要降到多少、或張爪與退開要多快，才能讓門餘轉留在可容許範圍內。

    python3 evaluation/door_coopen_release_tol_v1.py   # 寫 results/door/door_coopen_release_tol_v1.json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_traj_v1 as TJ                       # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_smooth_v1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_release_tol_v1.json')
DS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
TH = {'nonfinger': 0.005, 'finger': 0.005, 'self': 0.005}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    d = json.load(open(SRC))
    q_rel, P = d['q_rel_deg'], d['path']
    rows = [a for a in d['arm_retract'] if 'qa' in a]
    src_h = json.load(open(os.path.join(HERE, 'results', 'door', 'door_coopen_edgehold_v2.json')))
    cand = next(r for r in src_h['per_end'] if r['q_rel_deg'] == q_rel)
    yaw = cand['trajectory'][-1]['base'][2]
    out = []
    for a in rows:
        TJ.cool()
        k = int(round(a['s_m'] / 0.01))
        qf = [P[k][0], P[k][1], yaw] + list(a['qa'])
        ok = {}
        for dq in DS:
            g = TJ.GEO(qf, math.radians(q_rel + dq), a['q_finger'])
            ok[dq] = all(g[c][0] >= TH[c] for c in TH)
        pos = 0.0
        for dq in sorted(x for x in DS if x > 0):
            if not ok[dq]:
                break
            pos = dq
        neg = 0.0
        for dq in sorted((x for x in DS if x < 0), reverse=True):
            if not ok[dq]:
                break
            neg = dq
        out.append({'phase': a['phase'], 's_m': a['s_m'], 'ok_at_0': ok[0.0], 'delta_plus_ok_deg': pos, 'delta_minus_ok_deg': neg})
    summ = {}
    for ph in ('open', 'retract', 'stow'):
        r = [o for o in out if o['phase'] == ph]
        if r:
            summ[ph] = {'n': len(r), 'all_ok_at_0': all(o['ok_at_0'] for o in r), 'delta_plus_ok_min_deg': min(o['delta_plus_ok_deg'] for o in r),
                        'delta_minus_ok_max_deg': max(o['delta_minus_ok_deg'] for o in r)}
            print(f"{ph}：{len(r)} 點；δ=0 全過 {summ[ph]['all_ok_at_0']}；可容許續轉（開門方向）最小 {summ[ph]['delta_plus_ok_min_deg']}°、回轉 {summ[ph]['delta_minus_ok_max_deg']}°", flush=True)
    first_free = next((o for o in out if o['delta_plus_ok_deg'] >= 5.0), None)
    print(f"可容許 +5° 續轉的第一點：{None if first_free is None else (first_free['phase'], first_free['s_m'])}", flush=True)
    json.dump({'schema': 'door_coopen_release_tol/1', 'source': os.path.relpath(SRC, HERE), 'q_rel_deg': q_rel, 'deltas_deg': DS,
               'per_point': out, 'summary': summ, 'first_point_tolerating_plus5': first_free,
               'note': '續轉＝放手後門往開的方向多轉；回轉＝往關的方向'}, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
