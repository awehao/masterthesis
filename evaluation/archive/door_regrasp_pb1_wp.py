#!/usr/bin/env python3
"""pb1 的 M 段加速變體（10/28：pb1 整機 A* 由 W1 末態直搜 B1 超過 40 分鐘未結束）：唯一差異＝A* 經過點 W＝ref v4 來源（bench_6 放手末端）
(1.235, −0.959)——原 v8 自該點以同偏航 270° 整機 A* 到 B1 已成立（3813 節點）。M 段＝起點原地轉 → A*(起點→W) → A*(W→B1)，節點條件、
門角包絡 ±0.7°、邊中間點、full＋PI.check 逐點覆核全同 v8；其餘（放手列、N0／N1、經過點、時間化、建包）同 door_regrasp_pb1。輸出檔名加 _wp。

    python3 -u evaluation/door_regrasp_pb1_wp.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_regrasp_pb1 as PB                          # noqa: E402
import door_regrasp_replan_v8 as RV8                   # noqa: E402

W = np.array([1.235, -0.959])
_astar = RV8.astar_wb


def astar_via(b0, b1, yaw, q_deg, res=0.0125):
    p1, s1 = _astar(b0, W, yaw, q_deg, res)
    if p1 is None:
        return None, dict(s1, leg='start_to_W')
    p2, s2 = _astar(W, b1, yaw, q_deg, res)
    if p2 is None:
        return None, dict(s2, leg='W_to_B1')
    return list(p1) + list(p2[1:]), {'leg1': s1, 'leg2': s2, 'via': W.tolist(), 'n_node_checks': s1.get('n_node_checks', 0) + s2.get('n_node_checks', 0)}


RV8.astar_wb = astar_via
PB.F = {k: v.replace('.json', '_wp.json') for k, v in PB.F.items()}

if __name__ == '__main__':
    PB.main()
