#!/usr/bin/env python3
"""朝前穿門拉關：內側抓法候選 φ 的有界截面核對（Codex 20261029_142310 核可：最多兩個有幾何依據的 φ；先查目前零可行的 32°，再核兩端）。
幾何依據（door_regrasp_fwd_probe r3 之後的計算，記於輸出 basis）：
  - 現行 φ30 的接近方向與「站在 +y 側、肩→把手」方向只差 1–12°；朝前時失敗主因是**可達**：底盤須離門扇／把手 ≥ 20 mm，
    手臂裝於底盤前方 +0.11 m，轉為朝 +y 後肩部離把手多約 0.22 m，加上高度差約 0.28 m，超過 Lite 6 工作半徑（約 0.44 m）。
  - 改為「車頭朝前側身經過把手」（把手在機器人右側＝底盤在把手 −x 側 0.30–0.40 m）：肩→把手 0.41–0.49 m；所需 φ 隨門關 125°（48°）→110°（32°）→95°（16°）→80°（2°）。
  ⇒ 候選 φ＝105°（48→2° 全程所需值的中位，兩端偏差約 ±25°，由底盤位置／偏航連續調整補償）、φ＝120°（偏向開門端與零可行的 32°）。
仍夾同一支內側橫桿；內側抓法 T_HG(φ)＝Rx(φ)·T_HG0（同 door_regrasp_ref_registry 的定義）；判準同 door_coopen_pass_close_inner_v1.check。
截面：底盤 x 0.25–0.75、y＝把手 y−0.20…+0.40（5 cm）、偏航 {60, 75, 90, 105, 120}°；可達預篩放寬為肩→抓取點 ≤ 0.60 m（Lite 6 工作半徑約 0.44 m，保守上界）。
階段：閉爪持握（P）＋同一底盤的預抓取（沿工具 −z 0.06 m、張爪，A）兩者皆可行才記為可行（接近與持握同一底盤可達）。
    python3 -u evaluation/door_regrasp_fwd_probe2.py 32 [48 2 16]   # 寫 results/door/door_regrasp_fwd_probe2_q<角>.json（不覆寫）
"""
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_pass_close_inner_v1 as PI           # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_geom_v3 as G3                              # noqa: E402

PHIS = (105.0, 120.0)
YAWS = (60.0, 75.0, 90.0, 105.0, 120.0)
XS = np.round(np.arange(0.25, 0.751, 0.05), 3)
REACH = 0.60


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    R = np.eye(4)
    R[1, 1], R[1, 2], R[2, 1], R[2, 2] = c, -s, s, c
    return R


def reachable(base, T):
    lb = np.asarray(PI.C.K.fk(list(base) + [0.0] * 6, 'link_base'))[:3, 3] + np.array([0.0, 0.0, 0.243])
    return float(np.linalg.norm(T[:3, 3] - lb)) <= REACH


def solve(base, T, qfi, stage, q):
    best, why_c = None, collections.Counter()
    for seed in [None] + [np.array(sd) for sd in PI.C.SEEDS]:
        q9 = PI.C.ik(np.asarray(base), T, seed, only_prev=seed is not None)
        if q9 is None:
            why_c['ik_fail'] += 1
            continue
        x = np.concatenate([base, np.asarray(q9)[PI.C.IDX]])
        ok, why, info = PI.check(x, q, qfi, stage)
        if ok:
            sc = min(info['jm'], 2.0) + 5.0 * min(info['nonfinger'][0], info['self'][0])
            if best is None or sc > best[0]:
                best = (sc, x, info)
        else:
            why_c[(why or '?').split(':')[0]] += 1
    return best, why_c


def main():
    qs = [float(v) for v in sys.argv[1:]] or [32.0]
    T0 = np.asarray(RV.T_HG(), float)
    for q in qs:
        out = os.path.join(HERE, 'results', 'door', f'door_regrasp_fwd_probe2_q{int(q)}.json')
        if os.path.exists(out):
            raise SystemExit(f'{out} 已存在，不覆寫')
        H = np.asarray(G3.handle_pose_inner(PI.D3, math.radians(q)), float)
        ys = np.round(np.arange(H[1, 3] - 0.20, H[1, 3] + 0.401, 0.05), 3)
        res = {'schema': 'door_regrasp_fwd_probe2/1', 'q_deg': q, 'phis': list(PHIS), 'yaws': list(YAWS), 'reach_prefilter_m': REACH,
               'handle_xyz': H[:3, 3].tolist(), 'per_phi': {}}
        for phi in PHIS:
            T = H @ Rx(math.radians(phi)) @ T0
            Tpre = PI.tz(T, -0.06)
            feas, fails, n_try = [], collections.Counter(), 0
            for yw in YAWS:
                for bx in XS:
                    for by in ys:
                        b = np.array([bx, by, math.radians(yw)])
                        if not reachable(b, T):
                            fails['prefilter_reach'] += 1
                            continue
                        n_try += 1
                        best, wc = solve(b, T, PI.CLOSED, 'P', q)
                        if best is None:
                            fails.update(wc)
                            continue
                        bp, wcp = solve(b, Tpre, PI.OPEN, 'A', q)
                        if bp is None:
                            fails.update({'pregrasp_' + k: v for k, v in wcp.items()})
                            continue
                        feas.append({'x': float(bx), 'y': float(by), 'yaw_deg': yw, 'score': float(best[0]), 'jm': best[2]['jm'],
                                     'nonfinger_mm': best[2]['nonfinger'][0] * 1e3, 'self_mm': best[2]['self'][0] * 1e3,
                                     'base_mm': best[2]['base'][0] * 1e3, 'qa_hold': best[1][3:].tolist(), 'qa_pre': bp[1][3:].tolist()})
            byy = collections.Counter(f['yaw_deg'] for f in feas)
            res['per_phi'][str(phi)] = {'n_try': n_try, 'n_feasible': len(feas), 'by_yaw': {str(k): v for k, v in sorted(byy.items())},
                                        'top': sorted(feas, key=lambda f: -f['score'])[:10], 'fail_reasons': dict(fails.most_common(8))}
            print(f'q {q}° φ {phi}°：求解 {n_try}、持握＋預抓取皆可行 {len(feas)}（偏航 {dict(sorted(byy.items()))}）；失敗主因 {dict(fails.most_common(4))}', flush=True)
        json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
        print('寫', out, flush=True)


if __name__ == '__main__':
    main()
