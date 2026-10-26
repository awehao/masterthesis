#!/usr/bin/env python3
"""連續協同通行：離線可行性盤點 v1 之一——持續抓持把手時的底盤可行域（Codex 20261026_141054 §3；無 Isaac、無接觸力）。

逐門角 q（0–100°，每 5°）在底盤位姿格點（x 每 0.05 m、y 每 0.05 m、偏航每 15°）上檢查「握住把手」是否可行：
  1. 底盤圓盤（r 0.30）對門扇／立柱／兩側牆淨空 ≥ 20 mm（door_geom_v2.disc_clear，先篩）；
  2. 粗篩（只為省時，偏寬）：手臂安裝點（link1，底盤前方 0.11 m、高 0.574 m）到抓取點水平距離 0.12–0.48 m、
     偏航與「底盤→抓取點」方位差 ≤ 75°；
  3. 抓取位姿 TCP＝H(q)·T_HG（T_HG 取實跑 door_wide_w2c_30_1）以 IK 求解（door_offline_check_v1.ik 多種子），
     關節餘裕 ≥ 0.05 rad；
  4. 全連桿淨空（door_k3m_geom_monitor.default_backend，與線上同一套）：非手指 ≥ 5 mm、手指（不含設計接觸）≥ 5 mm、
     底盤 ≥ 20 mm、自碰 ≥ 5 mm。
輸出每個 q 的可行底盤位姿與幾項摘要：
  - x_max：可行底盤中心的最大 x（越大越靠門閂側＝門洞所在側）；
  - y_max、door_side：可行底盤中心是否可能位於門扇的門洞側（以門扇所在直線分側：門洞側＝門扇背面法向 −n 一側）；
  - 通行間隙：門扇自由端到門閂側牆端面的距離。
**界線**：格點取樣（取樣間未核）、剛性夾持、無接觸力與柔順、底盤以圓盤近似；只用來判斷「持續抓持能否把機器人帶到門洞側」。

    python3 evaluation/door_coopen_reach_v1.py   # 寫 results/door/door_coopen_reach_v1.json（不覆寫）
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
import door_geom_v2 as G2                              # noqa: E402
import door_k2_check as KC                             # noqa: E402
import door_k2_offline_v2 as V2                        # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_reach_v1.json')
C, K1 = V2.C, V2.K1
RUN = os.path.join(HERE, 'runs', 'door_wide_w2c_30_1')
Q_FINGER = 0.0016
LIM = {'base_m': 0.02, 'nonfinger_m': 0.005, 'finger_m': 0.005, 'self_m': 0.005, 'joint_margin_rad': 0.05,
       'mount_horiz_m': [0.12, 0.48], 'bearing_deg': 75.0}
D2 = K1.D2


def T_HG():
    t = json.load(open(os.path.join(RUN, 'task.json')))
    return np.array(t['grasp_basis']['T_HG'], float)


def disc_ok(x, y, q):
    for name, c, R, hs in G2.shapes(D2, q, K1.H):
        if float(G2.disc_clear(x, y, c[2], C.RB, c, R, hs)) < LIM['base_m']:
            return False
    return True


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '8'))) as pool:
        out = pool.map(one_q, list(range(0, 101, 5)))
    for rec in out:
        print(f"q={rec['q_deg']:3d}° disc {rec['n_disc_ok']:5d} reach {rec['n_reach_ok']:5d} ik {rec['n_ik_ok']:4d} feas {rec['n_feasible']:4d} "
              f"x_max {rec['x_max']} y_max {rec['y_max']} 門洞側 {rec['n_doorway_side']:3d} gap {rec['gap_tip_to_latch_m']:.3f}", flush=True)
    print(f'{time.time() - t0:.0f} s')
    res = {'schema': 'door_coopen_reach/1', 'limits': LIM, 'grid': {'x': [float(XS[0]), float(XS[-1]), 0.05], 'y': [float(YS[0]), float(YS[-1]), 0.05], 'yaw_deg_step': 30},
           'source_grasp': 'runs/door_wide_w2c_30_1 task.grasp_basis.T_HG', 'q_finger': Q_FINGER, 'per_q': out,
           'not_modeled': ['接觸力與夾持柔順', '取樣間碰撞', '底盤外凸物（圓盤近似）', '動態可行性（另由路徑核對）', '粗篩偏寬但仍可能漏掉極端構型']}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)


XS = np.round(np.arange(-1.0, 1.55, 0.05), 3)
YS = np.round(np.arange(-1.5, 0.75, 0.05), 3)


def one_q(qd):
    THG = T_HG()
    geo = GM.default_backend()
    xs, ys = XS, YS
    yaws = [math.radians(a) for a in range(0, 360, 30)]
    W = float(D2['frame_clear_width'])
    if True:
        q = math.radians(qd)
        Tg = KC._H_of(q) @ THG
        pg = Tg[:3, 3]
        feas = []
        n_disc = n_reach = n_ik = 0
        for x in xs:
            for y in ys:
                if not disc_ok(float(x), float(y), q):
                    continue
                n_disc += 1
                for yaw in yaws:
                    mx, my = x + 0.11 * math.cos(yaw), y + 0.11 * math.sin(yaw)
                    hd = math.hypot(mx - pg[0], my - pg[1])
                    brg = math.atan2(pg[1] - y, pg[0] - x)
                    if not (LIM['mount_horiz_m'][0] <= hd <= LIM['mount_horiz_m'][1]) or \
                            abs(math.degrees(math.atan2(math.sin(yaw - brg), math.cos(yaw - brg)))) > LIM['bearing_deg']:
                        continue
                    n_reach += 1
                    base = np.array([x, y, yaw], float)
                    qf9 = C.ik(base, Tg, None)
                    if qf9 is None:
                        continue
                    n_ik += 1
                    qa = np.asarray(qf9, float)[C.IDX]
                    jm = float(min(min(qa - V2.ELO + 0.05), min(V2.EHI + 0.05 - qa)))
                    if jm < LIM['joint_margin_rad']:
                        continue
                    qf = [float(v) for v in list(base) + list(qa)]
                    d = geo(qf, q, Q_FINGER)
                    if d['nonfinger'][0] < LIM['nonfinger_m'] or d['finger'][0] < LIM['finger_m'] or d['base'][0] < LIM['base_m'] or d['self'][0] < LIM['self_m']:
                        continue
                    # 門扇所在直線分側：門洞側＝背面法向 −n＝(sin q, cos q) 一側（以門扇中點為參考）
                    side = (x - 0.475 * math.cos(q)) * math.sin(q) + (y + 0.475 * math.sin(q)) * math.cos(q)
                    feas.append({'x': float(x), 'y': float(y), 'yaw_deg': round(math.degrees(yaw)), 'jm': jm,
                                 'clear': {k: float(d[k][0]) for k in ('nonfinger', 'finger', 'base', 'self')}, 'side_m': float(side)})
        tip = np.array([0.95 * math.cos(q), -0.95 * math.sin(q)])
        gap = float(np.linalg.norm(tip - np.array([W, 0.0])))
        rec = {'q_deg': qd, 'grasp_p': [float(v) for v in pg], 'n_disc_ok': n_disc, 'n_reach_ok': n_reach, 'n_ik_ok': n_ik, 'n_feasible': len(feas),
               'gap_tip_to_latch_m': gap,
               'x_max': max((f['x'] for f in feas), default=None), 'y_max': max((f['y'] for f in feas), default=None),
               'n_doorway_side': sum(1 for f in feas if f['side_m'] > 0.0),
               'best_doorway_side': max(feas, key=lambda f: f['side_m'], default=None),
               'feasible': feas}
        return rec


if __name__ == '__main__':
    main()
