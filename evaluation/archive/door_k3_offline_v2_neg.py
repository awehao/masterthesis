#!/usr/bin/env python3
"""K3 離線核對 v2 的負向補表：門獨立**回關** Δ ∈ {0, −0.25, …, −2.0}°（手臂同 v2：沿 θ_goal＝8° 下退離路徑、張爪、不跟門）。
沿用 door_k3_offline_v2 的 case()／clearance()／判準（環境 ≥ 5 mm、底盤 ≥ 2 cm），只換 Δ 網格；v2 輸出不動。

    python3 evaluation/door_k3_offline_v2_neg.py   # 輸出 results/door/door_k3_offline_v2_neg.json（已存在則拒絕覆寫）
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3_offline_v2 as V                         # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_k3_offline_v2_neg.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    V.D_GRID = [-x for x in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)]
    res = {'schema': 'door_k3_offline_neg/1', 'theta_goal_deg': V.THETA, 's_grid': V.S_GRID, 'delta_grid_deg': V.D_GRID, 'cases': []}
    for off in V.V2.OFFS:
        c = V.case(off)
        c.pop('s_clear_for_delta', None)
        res['cases'].append(c)
    rows = []
    for i, s in enumerate(V.S_GRID):
        d = [c['grid'][i].get('delta_safe_deg') for c in res['cases']]
        rows.append({'s': s, 'delta_safe_neg_worst_deg': None if any(x is None for x in d) else max(d),
                     'env_mm_worst_by_delta': {str(dd): round(min(c['grid'][i]['by_delta'][j]['env'] for c in res['cases']) * 1e3, 2)
                                               for j, dd in enumerate(V.D_GRID)}})
    res['summary'] = rows
    for r in rows:
        print(r['s'], '回關可承受（最差）', r['delta_safe_neg_worst_deg'], '° ', r['env_mm_worst_by_delta'], flush=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
