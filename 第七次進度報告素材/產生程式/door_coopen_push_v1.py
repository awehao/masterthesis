#!/usr/bin/env python3
"""連續協同通行：離線可行性盤點之三——由門洞側推門扇背面時的底盤可行域（無 Isaac、無接觸力；候選接觸模式「換側推門」）。

門扇局部：把手座標 H(q) 的 x＝u（沿門扇）、y＝−n（指入門扇）、z 向上；把手面＝門扇機器人側面，背面在把手面 −n 方向 thickness 處。
推觸位姿 P(q, r_c)：門扇背面上距鉸鏈 r_c（0.85／0.90／0.93 m）、高 0.85 m 的點，TCP 停在背面外 STANDOFF＝15 mm（可接觸位置），
工具 z 軸＝+n（由門洞側朝門扇推，推動方向使門角增加），工具 x＝+u。
逐門角 q（30–95°，每 5°）在底盤格點（x 0.20–1.60、y −1.20–0.60，每 0.05 m；偏航每 30°，與「底盤→接觸點」方位差 ≤ 75°）檢查：
  底盤圓盤對門扇／立柱／牆 ≥ 20 mm；IK（多種子）；關節餘裕 ≥ 0.05 rad；全連桿淨空（同線上 default_backend）：
  非手指 ≥ 5 mm、自碰 ≥ 5 mm、底盤 ≥ 20 mm；手指對門件記錄（推觸件，預期 0–15 mm，≥ 1 mm 才記為可接觸未穿入）。
輸出每個 (q, r_c) 的可行底盤與摘要：通行帶（x ∈ [0.35, 0.65]）內可行底盤的最大 y、可行底盤是否在門扇門洞側、x 範圍。
**界線**：格點取樣、靜態、推觸只到「可接觸位置」不含接觸力／摩擦／門被推動的動力學；手臂由抓持到推觸的換位路徑另核。

    python3 evaluation/door_coopen_push_v1.py   # 寫 results/door/door_coopen_push_v1.json（不覆寫）
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
import door_k2_check as KC                             # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_push_v1.json')
C, V2, GM = RV.C, RV.V2, RV.GM
D2 = RV.D2
THICK = float(D2['leaf']['thickness'])
STANDOFF = 0.015
BAR_TO_FACE = float(D2['handle']['standoff'])
LIM = dict(RV.LIM, finger_min_m=0.001)
XS = np.round(np.arange(0.20, 1.601, 0.05), 3)
YS = np.round(np.arange(-1.20, 0.601, 0.05), 3)
RCS = (0.85, 0.90, 0.93)
BAND = (0.35, 0.65)


def push_pose(q, rc):
    H = KC._H_of(q)
    u, ym, z = H[:3, 0], H[:3, 1], H[:3, 2]                  # ym＝−n
    n = -ym
    from_hinge = float(D2['handle']['from_hinge'])
    p_bar = H[:3, 3]
    hinge_pt = p_bar - u * from_hinge - n * BAR_TO_FACE           # 把手面上鉸鏈線（同高）
    p = hinge_pt + u * rc - n * (THICK + STANDOFF)                # 背面外 STANDOFF
    T = np.eye(4)
    T[:3, 0], T[:3, 2] = u, n
    T[:3, 1] = np.cross(n, u)
    T[:3, 3] = p
    return T


def one(args):
    qd, rc = args
    q = math.radians(qd)
    geo = GM.default_backend()
    Tp = push_pose(q, rc)
    pc = Tp[:3, 3]
    feas, n_disc, n_reach, n_ik = [], 0, 0, 0
    for x in XS:
        for y in YS:
            if not RV.disc_ok(float(x), float(y), q):
                continue
            n_disc += 1
            for a in range(0, 360, 30):
                yaw = math.radians(a)
                mx, my = x + 0.11 * math.cos(yaw), y + 0.11 * math.sin(yaw)
                hd = math.hypot(mx - pc[0], my - pc[1])
                brg = math.atan2(pc[1] - y, pc[0] - x)
                if not (0.12 <= hd <= 0.48) or abs(math.degrees(math.atan2(math.sin(yaw - brg), math.cos(yaw - brg)))) > 75.0:
                    continue
                n_reach += 1
                base = np.array([x, y, yaw], float)
                qf9 = C.ik(base, Tp, None)
                if qf9 is None:
                    continue
                n_ik += 1
                qa = np.asarray(qf9, float)[C.IDX]
                jm = float(min(min(qa - V2.ELO + 0.05), min(V2.EHI + 0.05 - qa)))
                if jm < LIM['joint_margin_rad']:
                    continue
                qf = [float(v) for v in list(base) + list(qa)]
                d = geo(qf, q, RV.Q_FINGER)
                if d['nonfinger'][0] < LIM['nonfinger_m'] or d['base'][0] < LIM['base_m'] or d['self'][0] < LIM['self_m'] or d['finger'][0] < LIM['finger_min_m']:
                    continue
                side = (x - 0.475 * math.cos(q)) * math.sin(q) + (y + 0.475 * math.sin(q)) * math.cos(q)
                feas.append({'x': float(x), 'y': float(y), 'yaw_deg': a, 'jm': jm, 'qa': [float(v) for v in qa],
                             'clear': {k: float(d[k][0]) for k in ('nonfinger', 'finger', 'base', 'self')}, 'who': {k: d[k][1] for k in ('nonfinger', 'finger')},
                             'side_m': float(side)})
    band = [f for f in feas if BAND[0] <= f['x'] <= BAND[1]]
    return {'q_deg': qd, 'r_c': rc, 'contact_p': [float(v) for v in pc], 'n_disc_ok': n_disc, 'n_reach_ok': n_reach, 'n_ik_ok': n_ik, 'n_feasible': len(feas),
            'n_doorway_side': sum(1 for f in feas if f['side_m'] > 0), 'x_range': [min((f['x'] for f in feas), default=None), max((f['x'] for f in feas), default=None)],
            'band_y_max': max((f['y'] for f in band), default=None), 'band_n': len(band), 'feasible': feas}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    from multiprocessing import Pool
    jobs = [(qd, rc) for qd in range(30, 96, 5) for rc in RCS]
    t0 = time.time()
    with Pool(int(os.environ.get('NPROC', '8'))) as pool:
        out = pool.map(one, jobs)
    for r in out:
        print(f"q={r['q_deg']:3d}° r_c={r['r_c']:.2f} feas {r['n_feasible']:4d} 門洞側 {r['n_doorway_side']:4d} x {r['x_range']} 通行帶 n {r['band_n']:3d} y_max {r['band_y_max']}",
              flush=True)
    print(f'{time.time() - t0:.0f} s')
    json.dump({'schema': 'door_coopen_push/1', 'limits': LIM, 'standoff_m': STANDOFF, 'band_x': BAND, 'per_q_rc': out,
               'not_modeled': ['接觸力、摩擦與門被推動的動力學', '取樣間碰撞', '底盤外凸物', '手臂換位路徑（抓持→推觸）']},
              open(OUT, 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
