#!/usr/bin/env python3
"""連續協同通行：回程（推門方向）候選軌跡 v1（無 Isaac、無接觸力；單一程序；Codex 20261026_141054 §3-4 的替代候選）。

場景：機器人在門的另一側（+y，房間內），門向 −y 開（對機器人而言是推門）；門扇在 +y 側沒有把手，接觸＝推門扇背面
（door_coopen_push_v1.push_pose：工具 z＝+n，推動方向使門角增加）。機器人推著門扇、跟著門扇穿過門洞到 −y 側。
起點：門角 0，在 +y 側（x 0.40–0.70、y 0.45–0.90、偏航朝 −y ±60°）找可推觸背面（r_c 0.85／0.90／0.93）的底盤位姿。
搜尋：門角每 1° 一步（門由推觸帶動的幾何假設；無力學），底盤 Δx ∈ {−0.04…0.04}、Δy ∈ {−0.06…0.02}（0.02 步）、Δ偏航 ±4°，
接觸點沿門扇滑動（r_c ±0.03，0.70–0.93）；IK 以上一步為種子、相鄰步手臂 ≤ 0.10 rad；判準同 door_coopen_traj_v1（推觸件 ≥ 1 mm）。
目標：底盤沿 −y 穿過門洞（通行帶 x ∈ [0.35, 0.68]）；停止＝底盤中心 y ≤ −0.45（後緣離開牆面＋20 mm）。
報：開門—通行重疊（底盤 y 減少且門角增加的門角範圍）、通過時門角、最小淨空、關節餘裕、相鄰步最大變化。
**界線**：同 traj_v1；另：+y 側房間牆面只建模門洞兩側牆（extent_x −1.5–2.5），未建模房內其他物體。

    python3 evaluation/door_coopen_return_v1.py   # 寫 results/door/door_coopen_return_v1.json（不覆寫）
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
import door_coopen_push_v1 as PU                       # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_return_v1.json')


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    found = []
    for rc in (0.90, 0.85, 0.93):
        Tp = PU.push_pose(0.0, rc)
        pc = Tp[:3, 3]
        for x in np.arange(0.40, 0.71, 0.05):
            for y in np.arange(0.45, 0.91, 0.05):
                for a in range(-150, -29, 15):
                    yaw = math.radians(a)
                    brg = math.atan2(pc[1] - y, pc[0] - x)
                    mx, my = x + 0.11 * math.cos(yaw), y + 0.11 * math.sin(yaw)
                    if abs(math.degrees(math.atan2(math.sin(yaw - brg), math.cos(yaw - brg)))) > 60 or not (0.12 <= math.hypot(mx - pc[0], my - pc[1]) <= 0.48):
                        continue
                    TJ.cool()
                    r = TJ.check(np.array([x, y, yaw]), 0.0, Tp, None, push=True, full=True)
                    if r is not None:
                        found.append((-float(y), np.array([x, y, yaw]), r, rc))
        if found:
            break
    print(f'起點候選 {len(found)}（{time.time() - t0:.0f} s）', flush=True)
    if not found:
        json.dump({'schema': 'door_coopen_return/1', 'ok': False, 'why': 'no_start'}, open(OUT, 'w'), ensure_ascii=False, indent=1)
        return
    found.sort(key=lambda z: -z[0])
    _, b0, r0, rc0 = found[0]
    start = {'base': b0, 'qa': r0['qa'], 'extra': {'r_c': rc0}, 'jm': r0['jm'], 'clear': r0['clear']}

    def pT(q, ext):
        return PU.push_pose(q, ext['r_c'])

    def dR(st):
        out = []
        for drc in (0.0, -0.03, 0.03):
            rc = round(st['extra']['r_c'] + drc, 3)
            if not (0.70 <= rc <= 0.93):
                continue
            for dx in (-0.04, -0.02, 0.0, 0.02, 0.04):
                for dy in (-0.06, -0.04, -0.02, 0.0, 0.02):
                    for dyaw in (-4, 0, 4):
                        out.append((dx, dy, dyaw, {'r_c': rc}))
        return out

    def sR(b, ext, qd):
        return float(-b[1] - 2.0 * max(0.0, abs(float(b[0]) - 0.515) - 0.165))

    def stop(st, qd):
        return st['base'][1] <= -0.45

    layer, qend, why = TJ.beam(start, 0, 110, pT, sR, dR, push=True, width=5, per_parent=8, stop=stop)
    P = layer[0]['path']
    ys, qs = [p['base'][1] for p in P], [p['q'] for p in P]
    dec = [qs[i] for i in range(1, len(P)) if ys[i] < ys[i - 1] - 1e-9]
    js = [float(np.max(np.abs(np.array(b['qa']) - np.array(a['qa'])))) for a, b in zip(P, P[1:])]
    rec = {'schema': 'door_coopen_return/1', 'ok': why == 'stop_condition', 'why': why, 'q_end_deg': qend, 'start': {'base': [float(v) for v in b0], 'r_c': rc0},
           'end_base': [float(v) for v in P[-1]['base']], 'overlap_q_deg': [min(dec), max(dec)] if dec else None, 'n_forward_steps': len(dec),
           'q_at_wall_plane_deg': next((p['q'] for p in P if p['base'][1] <= 0.12 + 0.0), None),
           'joint_step_max_rad_per_deg': max(js) if js else None, 'joint_margin_min': min(p['jm'] for p in P if p.get('jm') is not None),
           'clear_min': {k: min(p['clear'][k] for p in P if p.get('clear')) for k in ('nonfinger', 'finger', 'base', 'self')},
           'trajectory': TJ.ser(P, 'R'), 'door_rate_assumed_deg_s': 1.0,
           'not_modeled': ['接觸力、摩擦、門被推動的動力學（假設門跟著推觸點）', '取樣間碰撞', '底盤外凸物', '房內其他物體', '控制器追蹤與延遲']}
    print(f"回程：{why}，門角 {qend}°，底盤 {np.round(P[-1]['base'], 3)}，重疊 {rec['overlap_q_deg']}（前進 {len(dec)} 步），"
          f"過牆面時門角 {rec['q_at_wall_plane_deg']}，最小淨空 {({k: round(v, 3) for k, v in rec['clear_min'].items()})}，{time.time() - t0:.0f} s", flush=True)
    json.dump(rec, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
