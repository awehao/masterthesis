#!/usr/bin/env python3
"""連續協同通行：繞緣持握 v3——持握末段底盤就沿通過路徑起始方向前進（離線靜態，無 Isaac；單一程序＋溫度暫停）。

**v2 揭露**：door_coopen_edgehold_v2 的 55° 候選在 48–55° 底盤只在 ±2 cm 內晃動（實質停住、只靠手臂開門約 7°），
door_coopen_pass_smooth_v1 報的「交界轉角 114°」是用這幾步的雜訊位移算的，無意義；真正的問題是放手前底盤停頓。
本版只重做 55° 終點的反向持握，其餘同 v2（夾法 φ 60°、s 0.85、roll 0；判準同 traj_v1＋底盤 ≥ 30 mm 規劃餘裕）：
  反向前 N_BLEND＝10 步（即正向 45→55°）評分＝沿 −v_out 的位移（v_out＝pass_smooth_v1 通過路徑前 10 cm 方向）
    − 2×垂直 v_out 的位移絕對值 − 0.5×停住（位移 < 5 mm）；其後同 v2（往門前側 (0.5, −1.0)）。
  底盤步長 Δx、Δy ∈ {−0.04…0.04}（0.02 步）、Δ偏航 ±4°（同 v1／v2）。
報：同 v2，另報正向最後 10 步底盤沿 v_out 的前進量與速度（門 1 °/s 計）、停住步數。

    python3 evaluation/door_coopen_edgehold_v3.py   # 寫 results/door/door_coopen_edgehold_v3.json（不覆寫）
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
import door_coopen_arc_v3 as A3                        # noqa: E402
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_edgehold_v1 as EH                   # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_edgehold_v2.json')
SRC_P = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_smooth_v1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_edgehold_v3.json')
Q_REL, BASE_PLAN, N_BLEND = 55, 0.030, 10
STEPS = EH.STEPS


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    h = json.load(open(SRC))
    pv = json.load(open(SRC_P))
    end = next(r for r in h['per_end'] if r['q_rel_deg'] == Q_REL)
    e = end['trajectory'][-1]
    g = h['grasp']
    P = np.array(pv['path'])
    v_out = (P[10] - P[0]) / np.linalg.norm(P[10] - P[0])
    perp = np.array([-v_out[1], v_out[0]])
    T_of = lambda q: EG.grasp_T(q, g['phi'], g['s'], g['roll'])       # noqa: E731
    orig = TJ.check

    def chk(b, q, T, seed):
        r = orig(b, q, T, seed)
        return r if (r is not None and r['clear']['base'] >= BASE_PLAN) else None
    st0 = {'base': np.array(e['base'], float), 'qa': np.array(e['qa'], float), 'jm': e['jm'], 'clear': e['clear']}
    layer = [dict(st0, path=[dict(st0, q=Q_REL)])]
    why, qend = 'end', 0
    for qd in range(Q_REL - 1, -1, -1):
        TJ.cool()
        q = math.radians(qd)
        blend = (Q_REL - qd) <= N_BLEND
        cand = []
        for st in layer:
            nb = []
            for dx in STEPS:
                for dy in STEPS:
                    for dyaw in (-4, 0, 4):
                        b = np.array([st['base'][0] + dx, st['base'][1] + dy, st['base'][2] + math.radians(dyaw)])
                        if blend:
                            dd = np.array([dx, dy])
                            sc = float(-dd @ v_out) - 2.0 * abs(float(dd @ perp)) - (0.5 if np.linalg.norm(dd) < 0.005 else 0.0)
                            sc += st.get('score_acc', 0.0)
                        else:
                            sc = -float(np.hypot(*(b[:2] - EH.FRONT)))
                        nb.append((sc, b))
            nb.sort(key=lambda z: -z[0])
            k = 0
            for sc, b in nb:
                if k >= 8:
                    break
                r = chk(b, q, T_of(q), st['qa'])
                if r is None:
                    continue
                k += 1
                ns = {'base': b, 'qa': r['qa'], 'score': sc, 'score_acc': sc if blend else 0.0, 'jm': r['jm'], 'clear': r['clear']}
                ns['path'] = st['path'] + [dict(ns, q=qd)]
                cand.append(ns)
        if not cand:
            why, qend = 'no_feasible_child', qd + 1
            break
        cand.sort(key=lambda z: -z['score'])
        seen, layer = set(), []
        for c_ in cand:
            key = (round(c_['base'][0], 2), round(c_['base'][1], 2), round(math.degrees(c_['base'][2]) / 3))
            if key not in seen:
                seen.add(key)
                layer.append(c_)
            if len(layer) >= 5:
                break
    P_ = layer[0]['path']
    rec = {'schema': 'door_coopen_edgehold/3', 'q_rel_deg': Q_REL, 'v_out': v_out.tolist(), 'n_blend': N_BLEND, 'reached_q_deg': qend, 'why': why}
    if qend == 0:
        fw = [{'q': float(p['q']), 'base': np.array(p['base'], float), 'qa': np.array(p['qa'], float), 'jm': p['jm'], 'clear': p['clear']} for p in P_[::-1]]
        mv = [b['base'][:2] - a['base'][:2] for a, b in zip(fw, fw[1:])]
        js = [float(np.max(np.abs(b['qa'] - a['qa']))) for a, b in zip(fw, fw[1:])]
        cm = {k: min(p['clear'][k] for p in fw) for k in ('nonfinger', 'finger', 'base', 'self')}
        tail = mv[-N_BLEND:]
        rec.update({'start_base_q0': fw[0]['base'].tolist(), 'clear_min': cm,
                    'margin_min_mm': {k: (cm[k] - {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020, 'self': 0.005}[k]) * 1000 for k in cm},
                    'joint_margin_min': min(p['jm'] for p in fw), 'joint_step_max_rad': max(js),
                    'blend_progress_along_vout_m': float(sum(m @ v_out for m in tail)), 'blend_lateral_abs_m': float(sum(abs(m @ perp) for m in tail)),
                    'blend_n_stopped': sum(1 for m in tail if np.linalg.norm(m) < 5e-3),
                    'n_stopped_total': sum(1 for m in mv if np.linalg.norm(m) < 1e-3), 'path_len_m': float(sum(np.linalg.norm(m) for m in mv)),
                    'duration_s': float(sum(A3.timing(fw))), 'end_j1': float(fw[-1]['qa'][0]),
                    'trajectory': [{'q_deg': p['q'], 'base': p['base'].tolist(), 'qa': p['qa'].tolist(), 'jm': p['jm'], 'clear': p['clear']} for p in fw]})
        print(f"握回 0°，起於 {np.round(fw[0]['base'][:2], 2)}；餘裕 {({k: round(v, 1) for k, v in rec['margin_min_mm'].items()})} mm；"
              f"末 {N_BLEND}° 沿通過方向 {rec['blend_progress_along_vout_m']:.3f} m、側向 {rec['blend_lateral_abs_m']:.3f} m、停 {rec['blend_n_stopped']} 步；"
              f"全程停 {rec['n_stopped_total']}；{rec['path_len_m']:.2f} m；{time.time() - t0:.0f} s", flush=True)
    else:
        print(f'{why}：只握回到 {qend}°；{time.time() - t0:.0f} s', flush=True)
    json.dump(rec, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
