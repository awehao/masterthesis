#!/usr/bin/env python3
"""朝前穿門拉關可行性探測（Codex 20261029_130606：先做 mode 5→7→8 的一條正向連續候選，或報第一個具體阻點；不掃抓法、權重）。
固定：內側抓法＝ref pb1f 登錄 inner_grasp.T_HG（φ＋30°），門 door_local_v3，判準＝door_coopen_pass_close_inner_v1.check（關節餘裕 0.05 rad、
非手指／自碰 5 mm、底盤 20 mm、手指對非指定物件 5 mm；閉爪後只豁免內側橫桿）。
問題：底盤偏航維持行進方向（+y，90°；容許 ±20° 內插的三個值）時，門角 48°→2° 的每個門角，門洞附近是否有可抓構型，且底盤 y 隨門關閉單調前進。
做法：每個門角 q ∈ {48, 40, 32, 24, 16, 8, 2}°、偏航 ∈ {90, 70, 110}°、底盤 (x, y) 格點（x 0.30–0.90／y −0.80–0.60，5 cm）上
  以多種子 IK 解抓取位姿 H_in(q)·T_HG，check(stage='P'，手指閉合)；記每個門角的可行格點與最佳淨空，及「最小 y／最大 y」。
  另核接近姿態：q＝48° 的預抓取（沿工具 −z 0.06 m、手指張開，stage 'A'）。
輸出 results/door/door_regrasp_fwd_probe.json（不覆寫）：每門角可行格點、y 範圍；第一個無可行格點的門角與各格點最常見的失敗原因。
界線：離線運動學、靜態逐點可行性，不是連續路徑規劃；連續候選需另以全身追蹤串接。
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
import door_geom_v3 as G3                              # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_regrasp_fwd_probe.json')
REF = os.path.join(HERE, 'results', 'door', 'door_regrasp_ref_pb1f.json')
QS = (48.0, 40.0, 32.0, 24.0, 16.0, 8.0, 2.0)
YAWS = (90.0, 70.0, 110.0)
XS = np.round(np.arange(0.30, 0.901, 0.05), 3)
YS = np.round(np.arange(-0.80, 0.601, 0.05), 3)


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
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    pkg = json.load(open(REF))
    THG = np.array(pkg['inner_grasp']['T_HG'], float).reshape(4, 4)
    res = {'schema': 'door_regrasp_fwd_probe/1', 'ref': os.path.relpath(REF, HERE), 'phi_deg': pkg['inner_grasp'].get('phi_deg'),
           'grid': {'x': XS.tolist(), 'y': YS.tolist(), 'yaw_deg': list(YAWS)}, 'per_q': {}}
    first_block = None
    for q in QS:
        T = G3.handle_pose_inner(PI.D3, math.radians(q)) @ THG
        feas, fails = [], collections.Counter()
        for yw in YAWS:
            for bx in XS:
                for by in YS:
                    best, wc = solve(np.array([bx, by, math.radians(yw)]), T, PI.CLOSED, 'P', q)
                    if best is not None:
                        feas.append({'x': float(bx), 'y': float(by), 'yaw_deg': yw, 'score': float(best[0]),
                                     'nonfinger_mm': best[2]['nonfinger'][0] * 1e3, 'self_mm': best[2]['self'][0] * 1e3,
                                     'base_mm': best[2]['base'][0] * 1e3, 'jm': best[2]['jm'], 'qa': best[1][3:].tolist()})
                    else:
                        fails.update(wc)
        ys = [f['y'] for f in feas]
        res['per_q'][str(q)] = {'n_feasible': len(feas), 'y_min': min(ys) if ys else None, 'y_max': max(ys) if ys else None,
                                'by_yaw': {str(yw): sum(1 for f in feas if f['yaw_deg'] == yw) for yw in YAWS},
                                'top': sorted(feas, key=lambda f: -f['score'])[:8], 'fail_reasons': dict(fails.most_common(6))}
        print(f"q {q:4.1f}°：可行 {len(feas)}（偏航 {res['per_q'][str(q)]['by_yaw']}），y ∈ [{res['per_q'][str(q)]['y_min']}, {res['per_q'][str(q)]['y_max']}]；"
              f"失敗主因 {dict(fails.most_common(3))}", flush=True)
        if not feas and first_block is None:
            first_block = {'q_deg': q, 'fail_reasons': dict(fails.most_common(6))}
    # 接近：q＝48° 預抓取（手指張開）
    Tg = G3.handle_pose_inner(PI.D3, math.radians(48.0)) @ THG
    Tpre = PI.tz(Tg, -0.06)
    feas_a, fails_a = [], collections.Counter()
    for yw in YAWS:
        for bx in XS:
            for by in YS:
                best, wc = solve(np.array([bx, by, math.radians(yw)]), Tpre, PI.OPEN, 'A', 48.0)
                if best is not None:
                    feas_a.append({'x': float(bx), 'y': float(by), 'yaw_deg': yw, 'score': float(best[0])})
                else:
                    fails_a.update(wc)
    res['pregrasp_48'] = {'n_feasible': len(feas_a), 'top': sorted(feas_a, key=lambda f: -f['score'])[:8], 'fail_reasons': dict(fails_a.most_common(6))}
    print(f'預抓取 48°：可行 {len(feas_a)}；失敗主因 {dict(fails_a.most_common(3))}', flush=True)
    res['first_block'] = first_block
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('寫', OUT, flush=True)


if __name__ == '__main__':
    main()
