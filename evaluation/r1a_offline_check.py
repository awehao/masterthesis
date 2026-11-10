#!/usr/bin/env python3
"""R1a 離線核對（Codex 20261111_011111）：
1 依量測生成：同一工作者（多種子、不含 q_ready）在不同量測瓶位（設計域角點與中心）回傳的窗起點預抓分支 q_pre_lo 是否不同。
2 轉移可行：從 V_HIGH（實際觀察構型）以 min-jerk（v_arm 1.0、a_arm 19.98）轉到 q_pre_lo，底盤同時沿目標專屬車道以 0.35 m/s 前進（全身），
  核手臂點雲對第一桌 ≥ 0.02 m、自碰撞 ≥ 0.005 m（同 bottle_r0_approach 判準與相鄰／剛性群組排除）；報轉移時間、離名目 q_ready 最小關節距離。
    python3 evaluation/r1a_offline_check.py OUT.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_plan_worker as PW  # noqa: E402
import bottle_r0_approach as R0  # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L  # noqa: E402

Q_VHIGH = R0.Q_VHIGH
Q_READY_NOM = R0.Q_READY_NOM


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(7)
    rs = [rng.uniform(L.lower * 0.8, L.upper * 0.8) for _ in range(6)]
    rows = []
    for dx in (-0.08, 0.0, 0.08):
        for dy in (-0.025, 0.04):
            xb, yb, zs = dx, 0.4275 + dy - 0.0275 + 0.0, 0.58          # 設計域：瓶 y＝0.42＋dy（區域；中心 0.4275 對應 dy 中點）
            yb = 0.42 + dy
            yl = yb - 0.4275
            bear = math.atan2(yb - yl, 0.35)
            qb = Q_VHIGH.copy()
            qb[0] = bear
            inp = dict(xy=[xb, yb], z_support=zs, band=0.10, pre_m=0.06, lift_m=0.05, win_rel=[-0.35, 0.15], x_nom=0.0, y_lane=yl,
                       q_seeds=[s.tolist() for s in [Q_VHIGH, qb] + rs], q_now=Q_VHIGH.tolist(), q_seed=None)
            ok, why, r = PW.plan_grasp(inp)
            row = dict(dx=dx, dy=dy, xy=[round(xb, 4), round(yb, 4)], y_lane=round(yl, 4), ok=ok, why=why)
            if ok:
                qp = np.asarray(r['q_pre_lo'], float)
                dq = qp - Q_VHIGH
                T = R0.min_jerk_T(dq)
                boxes = R0.table_boxes(zs)
                x_acc = xb - 2.0                                         # 承接時底盤位置（名目趟 8.7 s 約在量測 x − 2.0）
                we, wsf, dmin_ready = (np.inf, None), (np.inf, None), np.inf
                for k in range(21):
                    s = k / 20
                    sm = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
                    qa = Q_VHIGH + sm * dq
                    qf = np.r_[x_acc + 0.35 * s * T, yl, 0.0, qa]
                    e, sc, _ = R0.env(qf, boxes)
                    we = min(we, e, key=lambda z: z[0])
                    wsf = min(wsf, sc, key=lambda z: z[0])
                    dmin_ready = min(dmin_ready, float(np.abs(qa - Q_READY_NOM).max()))
                row.update(seed_index=r['seed_index'], margin=round(r['margin'], 4), q_pre_lo=np.round(qp, 3).tolist(), T_transition_s=round(T, 2),
                           env_min=(round(we[0], 4), we[1]), self_min=(round(wsf[0], 4), wsf[1]), min_dist_to_nominal_ready_maxabs=round(dmin_ready, 3),
                           end_dist_to_nominal_ready_maxabs=round(float(np.abs(qp - Q_READY_NOM).max()), 3),
                           ok_all=bool(we[0] >= R0.ENV_M and wsf[0] >= R0.SELF_M))
            rows.append(row)
            print(row if not ok else {k: row[k] for k in ('dx', 'dy', 'seed_index', 'T_transition_s', 'env_min', 'self_min', 'end_dist_to_nominal_ready_maxabs', 'min_dist_to_nominal_ready_maxabs', 'ok_all')})
    Q = [np.array(r['q_pre_lo']) for r in rows if r.get('ok')]
    spread = float(max(np.abs(a - b).max() for a in Q for b in Q)) if len(Q) > 1 else 0.0
    json.dump(dict(rows=rows, q_pre_lo_pairwise_max_diff=round(spread, 3), note='設計域 6 點；種子不含 q_ready；轉移起點 V_HIGH、底盤同時前進'), open(out, 'w'), ensure_ascii=False, indent=1)
    print('六個量測位置的預抓分支兩兩最大關節差', round(spread, 3), 'rad')


if __name__ == '__main__':
    main()
