#!/usr/bin/env python3
"""兩個 INSERT 停止核對工具的對齊（Codex 20261112 第 17 輪）：固定同一組假世界實際紀錄（每拍 q、設定點 s、在途命令 pipe、u_last），
用**同一個距離函式**（離線工具的 TO.bottle_min，豁免手指）比較兩種停止模型：
  real  ：C21.stop_traj(q, s, pipe, u_last)（假世界核對用：真設定點、真在途命令）
  ideal ：C21.stop_traj(q, s＝q[3:9], pipe＝[u_last]×D_STEPS, u_last)（離線工具 stop_sweep 的近似：設定點取當下構型、在途命令取該拍命令重複）
另列靜態（不停止）的當拍距離。回答：差異來自輸入軌跡，還是停止計算。
    python3 evaluation/insert_tools_compare.py STATES.npz OUT.json"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src', 'ammr_wholebody_mpc'))
import bottle_s0_ctrl as C0                                       # noqa: E402
import bottle_s1_ctrl as S1                                       # noqa: E402
import bottle_w1_compactcarry21_ctrl as C21                       # noqa: E402
import tpa_offline_check as TO                                    # noqa: E402
import tpas_offline_check as TSO                                  # noqa: E402

Z = np.load(sys.argv[1], allow_pickle=True)
xb, yb = [float(v) for v in Z['bxy']]
zs = float(Z['z1'])
rows = []
for i in range(len(Z['t'])):
    ph = str(Z['phase'][i])
    q, s, pipe, u = Z['q'][i], Z['s'][i], [p_ for p_ in Z['pipe'][i]], Z['u_last'][i]
    skip = TO.FING if ph == 'INSERT' else ()
    st = TO.bottle_min(q, xb, yb, zs, skip)
    out = dict(t=round(float(Z['t'][i]), 2), phase=ph, static=round(float(st[0]), 5), static_link=str(st[1]))
    for nm, (s_, pp) in dict(real=(s, pipe), ideal=(q[3:9].copy(), [u.copy() for _ in range(C0.D_STEPS)])).items():
        w = (9.9, None)
        for e in (0, 1):
            Q = C21.stop_traj(q, s_, pp, u, TSO.CFG, extra_ticks=e)
            for qq in Q:
                b = TO.bottle_min(qq, xb, yb, zs, skip)
                if b[0] < w[0]:
                    w = (float(b[0]), f'e{e}:{b[1]}')
        out[nm], out[nm + '_link'] = round(w[0], 5), w[1]
    rows.append(out)
res = dict(states=sys.argv[1], lim=S1.CLEAR_REJECT, n=len(rows))
for ph in ('PREGRASP', 'INSERT'):
    R = [r for r in rows if r['phase'] == ph]
    if R:
        res[ph] = dict(n=len(R), static_min=min(r['static'] for r in R), real_min=min(r['real'] for r in R), ideal_min=min(r['ideal'] for r in R),
                       real_link=min(R, key=lambda r: r['real'])['real_link'], ideal_link=min(R, key=lambda r: r['ideal'])['ideal_link'],
                       max_ideal_minus_real_mm=round(max(abs(r['ideal'] - r['real']) for r in R) * 1e3, 3))
res['rows'] = rows
json.dump(res, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, ensure_ascii=False))
