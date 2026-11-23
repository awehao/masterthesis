#!/usr/bin/env python3
"""door_offline_v1 補充（只讀 v1 輸出、不改）：各停位抓取姿態的 j5 與手臂 6×6 Jacobian 最小奇異值，說明固定底盤段 IK 延續失敗的原因。

    python3 evaluation/door_offline_v1_singularity.py   # 輸出 results/door/door_offline_v1_singularity.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_offline_check_v1 as C                      # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_offline_v1_singularity.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    d = json.load(open(C.OUT))
    T0 = C.target(0.0, 0.0)
    rows = []
    for s in d['stands']:
        base = np.array([T0[0, 3] + s['lateral'], T0[1, 3] - s['forward'], math.pi / 2])
        qf = C.ik(base, T0, None)                      # 與 v1 相同的確定性種子順序 ⇒ 同一解
        J = C.K.jacobian(qf, 'link_tcp')[:, C.IDX]
        sv = np.linalg.svd(J, compute_uv=False)
        rows.append({'forward': s['forward'], 'lateral': s['lateral'], 'q_arm': qf[C.IDX].tolist(), 'j5_rad': float(qf[C.IDX][4]),
                     'sigma_min': float(sv[-1]), 'fixed_ok_to_deg': s['fixed_sweep']['q_ok_to_deg'], 'first_fail': s['fixed_sweep']['first_fail']})
        print(rows[-1]['forward'], rows[-1]['lateral'], 'j5', round(rows[-1]['j5_rad'], 3), 'σmin', round(rows[-1]['sigma_min'], 4), '→', rows[-1]['fixed_ok_to_deg'])
    json.dump({'schema': 'door_offline_v1_singularity/1', 'source': os.path.relpath(C.OUT, C.WS), 'rows': rows,
               'note': 'σ_min 為 link_tcp 6×6 Jacobian（手臂六軸）最小奇異值；j5≈0 為腕部奇異（j4、j6 共軸）'}, open(OUT, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
