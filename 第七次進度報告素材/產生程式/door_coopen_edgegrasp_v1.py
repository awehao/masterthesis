#!/usr/bin/env python3
"""連續協同通行：「繞門緣握把手」可行性 v1（Howard 2026-10-26 選定；無 Isaac、無接觸力；單一程序＋溫度暫停）。

問題：現抓姿由正面（工具 z＝−n）夾把手，持續抓持會把底盤帶到鉸鏈側（door_coopen_reach_v1：0–100° 無任何可行抓持底盤在門洞側）。
改抓法候選：夾爪繞把手桿軸（u）轉 φ 夾桿（φ＝0 為現抓姿；φ＝−90° 由上方、＋90° 由下方；|φ| ≤ 90°，再大會夾到桿與門面之間的 27 mm 縫），
可沿桿滑 s（桿心 0.80 m 起 −0.05／0／+0.05，桿在兩立柱 0.715–0.885 之間）、繞接近軸翻 180°（roll）。
TCP 位姿 T＝H(q)·Trans(u·(s−0.80))·Rx(φ)·Rz_tcp(roll)·T_HG（T_HG 取實跑 door_wide_w2c_30_1）。

逐門角 q（預設 30／45／60／75°）只在「門洞側」底盤位姿上找可行夾法（這是本問題的關鍵；門前側已知可行）：
  門洞側＝底盤中心在門扇所在直線的背面側（side_m > 0），或位於通行帶（x 0.30–0.75、y ≥ −0.40）；
  底盤格點 x −0.20–1.40、y −1.20–0.60 每 0.10 m；偏航取「底盤→抓取點」方位 ±45°（每 45°）；
  粗篩：安裝點到抓取點水平 0.12–0.50 m；工具接近方向與「安裝點→抓取點」夾角 ≤ 80°（夾爪須由手臂這一側接近）。
  可行＝底盤 ≥ 20 mm、IK（多種子）、關節餘裕 ≥ 0.05 rad、非手指 ≥ 5 mm、手指（不含手指—橫桿設計接觸）≥ 5 mm、自碰 ≥ 5 mm。
輸出：每個 q 的門洞側可行 (底盤, φ, s, roll) 與摘要（數量、最深入通行帶的 y、最小淨空）。若為 0，列出最接近可行的失敗原因分布。
**界線**：格點取樣、靜態、剛性夾持；繞過門緣的手臂連桿以點雲核淨空；不含夾持力、換抓過程與動態。

    python3 evaluation/door_coopen_edgegrasp_v1.py [q ...]   # 寫 results/door/door_coopen_edgegrasp_v1.json（不覆寫）
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
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k2_check as KC                             # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_edgegrasp_v1.json')
C, V2 = RV.C, RV.V2
THG = RV.T_HG()
PHIS = (-90, -60, -30, 0, 30, 60, 90)
SS = (0.75, 0.80, 0.85)
ROLLS = (0, 180)


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1.0]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def grasp_T(q, phi, s, roll):
    T = np.eye(4)
    T[0, 3] = s - 0.80
    return KC._H_of(q) @ T @ Rx(math.radians(phi)) @ THG @ Rz(math.radians(roll))


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    qs = [int(a) for a in sys.argv[1:]] or [30, 45, 60, 75]
    t0 = time.time()
    res = []
    for qd in qs:
        q = math.radians(qd)
        feas, why = [], {'disc': 0, 'filter': 0, 'ik': 0, 'joint': 0, 'nonfinger': 0, 'finger': 0, 'base': 0, 'self': 0}
        n_eval = 0
        for x in np.round(np.arange(-0.20, 1.401, 0.10), 2):
            for y in np.round(np.arange(-1.20, 0.601, 0.10), 2):
                side = (x - 0.475 * math.cos(q)) * math.sin(q) + (y + 0.475 * math.sin(q)) * math.cos(q)
                if not (side > 0.0 or (0.30 <= x <= 0.75 and y >= -0.40)):
                    continue
                if not RV.disc_ok(float(x), float(y), q):
                    why['disc'] += 1
                    continue
                for phi in PHIS:
                    for s in SS:
                        for roll in ROLLS:
                            Tg = grasp_T(q, phi, s, roll)
                            pg, az = Tg[:3, 3], Tg[:3, 2]
                            brg = math.atan2(pg[1] - y, pg[0] - x)
                            for dyaw in (-45, 0, 45):
                                yaw = brg + math.radians(dyaw)
                                m = np.array([x + 0.11 * math.cos(yaw), y + 0.11 * math.sin(yaw), 0.574])
                                v = pg - m
                                hd = math.hypot(v[0], v[1])
                                if not (0.12 <= hd <= 0.50) or float(az @ v) / (np.linalg.norm(v) + 1e-12) < math.cos(math.radians(80)):
                                    why['filter'] += 1
                                    continue
                                TJ.cool()
                                n_eval += 1
                                base = np.array([x, y, yaw], float)
                                qf9 = C.ik(base, Tg, None)
                                if qf9 is None:
                                    why['ik'] += 1
                                    continue
                                qa = np.asarray(qf9, float)[C.IDX]
                                jm = float(min(min(qa - V2.ELO + 0.05), min(V2.EHI + 0.05 - qa)))
                                if jm < RV.LIM['joint_margin_rad']:
                                    why['joint'] += 1
                                    continue
                                d = TJ.GEO([float(v_) for v_ in list(base) + list(qa)], q, RV.Q_FINGER)
                                bad = [k for k, lim in (('nonfinger', 0.005), ('finger', 0.005), ('base', 0.02), ('self', 0.005)) if d[k][0] < lim]
                                if bad:
                                    why[bad[0]] += 1
                                    continue
                                feas.append({'x': float(x), 'y': float(y), 'yaw_deg': round(math.degrees(yaw), 1), 'phi': phi, 's': s, 'roll': roll,
                                             'side_m': float(side), 'jm': jm, 'qa': [float(v_) for v_ in qa],
                                             'clear': {k: float(d[k][0]) for k in ('nonfinger', 'finger', 'base', 'self')}})
        band = [f for f in feas if 0.30 <= f['x'] <= 0.75 and f['y'] >= -0.40]
        rec = {'q_deg': qd, 'n_eval_ik': n_eval, 'n_feasible_doorway_side': len(feas), 'n_in_band': len(band),
               'band_y_max': max((f['y'] for f in band), default=None), 'phis_feasible': sorted({f['phi'] for f in feas}),
               'reject_counts': why, 'feasible': feas}
        res.append(rec)
        print(f"q={qd}°：門洞側可行 {len(feas)}（通行帶內 {len(band)}，y_max {rec['band_y_max']}），φ {rec['phis_feasible']}；IK 試 {n_eval}；"
              f"拒絕 {why}；{time.time() - t0:.0f} s", flush=True)
    json.dump({'schema': 'door_coopen_edgegrasp/1', 'phis': PHIS, 's': SS, 'rolls': ROLLS, 'per_q': res,
               'not_modeled': ['格點取樣與取樣間', '夾持力與換抓過程', '動態可行性', '底盤外凸物']},
              open(OUT, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
