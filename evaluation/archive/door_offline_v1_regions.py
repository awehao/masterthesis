#!/usr/bin/env python3
"""door_offline_v1 區段統計更正（Codex 20261021；只讀 v1 輸出、不改）：
  * 「≤10°」區段納入 q = 0（關門抓取姿態）作為起點；
  * 最大關節跳變只算區段內相鄰可行點（不含首次失敗點），並另報「到首次失敗點為止」的跳變；
  * 標籤：殼／相機／自碰為點雲近似，不是完整碰撞安全證明；橫桿距離為膠囊體近似。

    python3 evaluation/door_offline_v1_regions.py    # 輸出 results/door/door_offline_v1_regions.json（已存在則拒絕覆寫）
"""
import json
import os

import math
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_offline_check_v1 as C                      # noqa: E402
SRC = os.path.join(HERE, 'results', 'door', 'door_offline_v1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_offline_v1_regions.json')
KEYS = ('joint_margin', 'env', 'shell_min', 'camera_stand_min', 'base_clear', 'self_clear')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    d = json.load(open(SRC))
    rows = []
    for s in d['stands']:
        start = dict(s['grasp'], q_deg=0.0, ok=True)
        seq = [start] + s['fixed_rows']
        okseq = []
        for r in seq:
            if not r['ok']:
                break
            okseq.append(r)
        out = {'forward': s['forward'], 'lateral': s['lateral'], 'feasible_to_deg': okseq[-1]['q_deg']}
        for name, lim in (('le_10deg', 10.0), ('feasible_region', 1e9)):
            reg = [r for r in okseq if r['q_deg'] <= lim]
            covered = bool(reg) and reg[-1]['q_deg'] >= lim - 1e-9
            out[name] = {'q_from': 0.0, 'q_to': reg[-1]['q_deg'] if reg else None, 'n_points': len(reg),
                         'covers_full_range': covered if lim < 1e9 else True,
                         'min': {k: float(min(r[k] for r in reg)) for k in KEYS}}
        # v1 輸出未逐點保存關節角 ⇒ 用 v1 的同一套確定性函式（同種子順序、同延續規則）重算 0 → 10°，只取區段內可行點的相鄰跳變
        T0 = C.target(0.0, 0.0)
        base = np.array([T0[0, 3] + s['lateral'], T0[1, 3] - s['forward'], math.pi / 2])
        qf = C.ik(base, T0, None)
        prev, steps, same = qf[C.IDX], [], True
        for r in s['fixed_rows']:
            if r['q_deg'] > 10.0 + 1e-9:
                break
            q2 = C.ik(base, C.target(math.radians(r['q_deg']), 0.0), prev, only_prev=True)
            ok2, _ = C.evaluate(q2, math.radians(r['q_deg']), 0.0)
            same &= (ok2 == r['ok'])
            if not ok2:
                break
            steps.append(float(np.abs(q2[C.IDX] - prev).max()))
            prev = q2[C.IDX]
        out['le_10deg']['max_joint_step_rad_in_region'] = max(steps) if steps else None
        out['le_10deg']['recompute_matches_v1_ok_flags'] = bool(same)
        out['max_joint_step_rad_v1_incl_first_fail'] = s['fixed_sweep']['max_joint_step_rad']
        rows.append(out)
    res = {'schema': 'door_offline_v1_regions/1', 'source': os.path.relpath(SRC, os.path.dirname(HERE)), 'stands': rows,
           'labels': ['殼、相機、自碰為點雲近似結果，不是完整碰撞安全證明', '橫桿距離為膠囊體近似（端面外低估間距）',
                      '固定底盤可行角為所選停位與搜尋方法下的結果', '取樣點之間的碰撞與速度／加速度未核']}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    for r in rows:
        m = r['le_10deg']['min']
        print(r['forward'], r['lateral'], '可行至', r['feasible_to_deg'], '≤10°(含 q=0) 覆蓋', r['le_10deg']['covers_full_range'],
              '關節', round(m['joint_margin'], 3), '底盤', round(m['base_clear'], 3), '自碰', round(m['self_clear'] * 1e3, 1), 'mm',
              '殼', round(m['shell_min'] * 1e3, 1), '相機', round(m['camera_stand_min'] * 1e3, 1),
              '區段跳變', r['le_10deg']['max_joint_step_rad_in_region'], '重算一致', r['le_10deg']['recompute_matches_v1_ok_flags'])


if __name__ == '__main__':
    main()
