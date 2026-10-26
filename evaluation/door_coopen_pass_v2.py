#!/usr/bin/env python3
"""連續協同通行：離線可行性盤點之二 v2——門角—底盤可行域與「開門須領先通行多少」（無 Isaac、無 IK、無接觸）。
v1（door_coopen_pass_v1.py，輸出保留）錯把「門角 ≥ q_min 即可行」當成單調：實際上門扇在中間門角甩進機器人側，
可行性對門角不單調，v1 的 y_reach 結果無效。v2 改為：

對平行於門洞中線（+y）的直行路線 x＝x_c（0.40–0.65 m，每 0.05 m）：
  1. 可行域 F(q, y)：底盤圓盤（r 0.30）對門扇、兩根立柱與兩側牆淨空 ≥ 20 mm（q 每 0.5°、0–110°；y 每 0.02 m、−1.40 → +0.60 m）。
  2. 動態規劃：門角只增不減（每步 +0.5°）；底盤每步 |Δy| ≤ DY（＝底盤 0.1 m/s 對門 1°/s 換算，0.05 m／0.5°），可前可後；
     相鄰兩步的四個端點組合（舊／新位置 × 舊／新門角）都須在 F 內（以 0.5° 取樣近似掃掠，取樣間未核）。起點 (q＝0, y0＝−0.53)（現停位的 y）。
     求：最早抵達 y ≥ 0.45（底盤後緣離開牆面＋20 mm 的通過點）的門角 q_through；每個門角的最遠可達 y（frontier，
     ＝開門到 q 時底盤沿此線最遠可以在哪裡）；以及沿一條見證軌跡的
       - q_enter：底盤前緣越過牆面（y ≥ −0.32）時的門角；
       - （見證軌跡的父節點選法任意，其後退深度與前進起點只是見證，不代表最佳；以 frontier 為準。）
  3. 靜態可通過門角 q_static：門角固定時整條直行路線 y ∈ [−1.40, 0.60] 皆在 F 內的最小門角（門不再動、只走底盤的情形）。
**界線**：只核底盤圓盤（不含手臂、外凸物）；直行路線；門角由外部帶動（不含如何帶動：抓持、推或其他接觸另核）。

    python3 evaluation/door_coopen_pass_v2.py   # 寫 results/door/door_coopen_pass_v2.json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_v1 as P1                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_v2.json')
QS = P1.QS
YS = [round(-1.40 + 0.02 * k, 2) for k in range(0, 101)]
DY = 0.05
Y0, Y_THROUGH, Y_ENTER = -0.53, 0.45, -0.32


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    names = ('leaf', 'post0', 'post1', 'wall_hinge_side', 'wall_latch_side')
    res = []
    for xc in [round(0.40 + 0.05 * k, 2) for k in range(0, 6)]:
        F = np.array([[P1.disc_clear(xc, y, q, names) >= P1.BASE_M for y in YS] for q in QS])
        q_static = next((QS[i] for i in range(len(QS)) if F[i].all()), None)
        j0 = int(np.argmin([abs(y - Y0) for y in YS]))
        dj = int(round(DY / 0.02))
        reach = np.zeros_like(F)
        parent = -np.ones(F.shape, dtype=int)
        if F[0, j0]:
            reach[0, j0] = True
        for i in range(1, len(QS)):
            for j in range(len(YS)):
                if not F[i, j]:
                    continue
                for jj in range(max(0, j - dj), min(len(YS), j + dj + 1)):
                    # 掃掠近似：新位置在前一門角、舊位置在新門角也都須可行（兩端點交叉皆淨空）
                    if reach[i - 1, jj] and F[i - 1, j] and F[i, jj]:
                        reach[i, j] = True
                        parent[i, j] = jj
                        break
        jt = [j for j, y in enumerate(YS) if y >= Y_THROUGH]
        it = next((i for i in range(len(QS)) if reach[i, jt].any()), None)
        traj = None
        if it is not None:
            j = max(jj for jj in jt if reach[it, jj])
            path = []
            for i in range(it, -1, -1):
                path.append((QS[i], YS[j]))
                j = parent[i, j] if i > 0 else j
            path.reverse()
            traj = path
        q_enter = next((q for q, y in traj if y >= Y_ENTER), None) if traj else None
        # 前進區間：y 自最後一次下降後開始單調前進的門角
        q_fwd = None
        if traj:
            last_drop = max((k for k in range(1, len(traj)) if traj[k][1] < traj[k - 1][1] - 1e-9), default=0)
            q_fwd = traj[last_drop][0]
        y_min = min(y for q, y in traj) if traj else None
        front = [{'q_deg': QS[i], 'y_max_reach': (max(YS[j] for j in range(len(YS)) if reach[i, j]) if reach[i].any() else None),
                  'y_min_reach': (min(YS[j] for j in range(len(YS)) if reach[i, j]) if reach[i].any() else None)} for i in range(0, len(QS), 4)]
        q_front_rise = next((f['q_deg'] for f in front if f['y_max_reach'] is not None and f['y_max_reach'] > Y0 + 0.02), None)
        res.append({'x_c': xc, 'q_static_deg': q_static, 'q_through_deg': None if it is None else QS[it], 'q_enter_deg': q_enter,
                    'q_forward_start_deg_path_artifact': q_fwd, 'y_min_m_path_artifact': y_min, 'trajectory_q_y_one_witness': traj,
                    'frontier_by_q': front, 'q_frontier_above_start_deg': q_front_rise,
                    'feasible_fraction_by_q': [{'q_deg': QS[i], 'frac': float(F[i].mean())} for i in range(0, len(QS), 10)]})
        fr = {f['q_deg']: f['y_max_reach'] for f in front}
        print(f"x_c={xc:.2f} 靜態可通過 {q_static}°；最遠可達 y＞起點自 {q_front_rise}°；前緣過牆面 {q_enter}°、通過 {res[-1]['q_through_deg']}°；"
              f"y_max@20/40/60/70/80 {[fr.get(q) for q in (20.0, 40.0, 60.0, 70.0, 80.0)]}", flush=True)
    json.dump({'schema': 'door_coopen_pass/2', 'base_m': P1.BASE_M, 'robot_radius_m': P1.C.RB, 'dy_per_half_deg_m': DY, 'y0': Y0,
               'y_through': Y_THROUGH, 'y_enter': Y_ENTER, 'per_xc': res,
               'not_modeled': ['手臂與外凸物', '取樣間掃掠', '非直行路線', '門角如何被帶動（抓持／推）']}, open(OUT, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
