#!/usr/bin/env python3
"""連續協同通行：滿足誤差預算的繞緣持握軌跡 v2（Codex 20261026_183611 §3-2；離線靜態，無 Isaac；單一程序＋溫度暫停）。

v1（door_coopen_edgehold_v1 → edgearc_v1 q45 roll 0）在 10 mm 底盤誤差下不成立（door_coopen_budget_v1）。本版：
  終點＝door_coopen_release_set_v1 的可持握放手狀態（放手後門角在 [q_rel − 2°, q_rel + 5°] 任意時，放手位姿淨空與到通過的最寬路徑瓶頸
    皆 ≥ 30 mm＝門檻 20＋誤差 10），**再限放手底盤位於門扇背面一側**（side_m ≥ 0.05 m：放手後往前即可，不是繞過自由端再切入）；
    每個 q_rel 取 min(F, 瓶頸) 最大的 2 個，至多 8 個終點。
  反向持握：自終點門角每 1° 往 0° 延伸（夾法固定 φ 60°、s 0.85、roll 0），判準同 v1，**另加規劃餘裕：持握各步底盤 ≥ 30 mm**；
    底盤 Δx、Δy ∈ {−0.04…0.04}（0.02 步）、Δ偏航 ±4°；評分＝往門前側 (0.5, −1.0) 靠近（同 v1 close 方向）。
  握回 0° 者倒序即開門軌跡。報：持握段各類最小淨空與餘裕、底盤往 +x／−x／後退／停住步數、時長（arc_v3.timing），
    以及起點（0°）底盤位姿。放手後路徑的重建與平滑在下一步（§3-3）做，本檔只帶出終點的 F 與瓶頸。
**界線**：同 edgehold_v1；門餘轉以區間包絡代替；圓盤近似。

    python3 evaluation/door_coopen_edgehold_v2.py   # 寫 results/door/door_coopen_edgehold_v2.json（不覆寫）
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

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_release_set_v1.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_edgehold_v2.json')
BASE_PLAN = 0.030
SIDE_MIN = 0.05
PER_Q, N_MAX = 2, 8


def side_m(x, y, q):
    return (x - 0.475 * math.cos(q)) * math.sin(q) + (y + 0.475 * math.sin(q)) * math.cos(q)


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    src = json.load(open(SRC))
    g = src['grasp']
    ends = []
    for r in src['per_q']:
        q = math.radians(r['q_rel_deg'])
        c = [f for f in r['feasible'] if side_m(f['x'], f['y'], q) >= SIDE_MIN]
        c.sort(key=lambda f: -min(f['F_m'], f['bottleneck_m']))
        ends += [(r['q_rel_deg'], f) for f in c[:PER_Q]]
    ends.sort(key=lambda e: -min(e[1]['F_m'], e[1]['bottleneck_m']))
    ends = ends[:N_MAX]
    print(f'門扇背面側終點 {len(ends)}：' + '；'.join(f"q{q} ({f['x']:.2f},{f['y']:.2f}) min(F,瓶頸) {min(f['F_m'], f['bottleneck_m']) * 1000:.0f} mm" for q, f in ends), flush=True)
    orig = TJ.check

    def check_plan(base, q, T, seed, push=False, full=False):
        r = orig(base, q, T, seed, push=push, full=full)
        return r if (r is not None and r['clear']['base'] >= BASE_PLAN) else None
    TJ.check = check_plan
    res = []
    try:
        for qr, f in ends:
            seed = {'x': f['x'], 'y': f['y'], 'yaw_deg': f['yaw_deg'], 'phi': g['phi'], 's': g['s'], 'roll': g['roll'],
                    'qa': f['qa'], 'jm': f['jm'], 'clear': f['clear']}
            layer, qend, why = EH.run(seed, qr, 0, EH.FRONT)
            P = layer[0]['path']
            rec = {'q_rel_deg': qr, 'end': {k: f[k] for k in ('x', 'y', 'yaw_deg', 'F_m', 'bottleneck_m')}, 'reached_q_deg': qend, 'why': why}
            if qend == 0:
                fw = [{'q': float(p['q']), 'base': np.array(p['base'], float), 'qa': np.array(p['qa'], float), 'jm': p['jm'], 'clear': p['clear']} for p in P[::-1]]
                mv = [b['base'][:2] - a['base'][:2] for a, b in zip(fw, fw[1:])]
                js = [float(np.max(np.abs(b['qa'] - a['qa']))) for a, b in zip(fw, fw[1:])]
                cm = {k: min(p['clear'][k] for p in fw) for k in ('nonfinger', 'finger', 'base', 'self')}
                rec.update({'start_base_q0': fw[0]['base'].tolist(), 'clear_min': cm,
                            'margin_min_mm': {k: (cm[k] - {'nonfinger': 0.005, 'finger': 0.005, 'base': 0.020, 'self': 0.005}[k]) * 1000 for k in cm},
                            'joint_margin_min': min(p['jm'] for p in fw), 'joint_step_max_rad': max(js),
                            'n_dx_pos': sum(1 for m in mv if m[0] > 1e-3), 'n_dx_neg': sum(1 for m in mv if m[0] < -1e-3),
                            'n_dy_neg': sum(1 for m in mv if m[1] < -1e-3), 'n_stopped': sum(1 for m in mv if np.linalg.norm(m) < 1e-3),
                            'path_len_m': float(sum(np.linalg.norm(m) for m in mv)), 'duration_s': float(sum(A3.timing(fw))),
                            'trajectory': [{'q_deg': p['q'], 'base': p['base'].tolist(), 'qa': p['qa'].tolist(), 'jm': p['jm'], 'clear': p['clear']} for p in fw]})
                print(f"終點 q{qr} ({f['x']:.2f},{f['y']:.2f})：握回 0°，起於 {np.round(fw[0]['base'][:2], 2)}；餘裕 {({k: round(v, 1) for k, v in rec['margin_min_mm'].items()})} mm；"
                      f"+x {rec['n_dx_pos']}／−x {rec['n_dx_neg']}／後 {rec['n_dy_neg']}／停 {rec['n_stopped']}；{rec['path_len_m']:.2f} m、{rec['duration_s']:.0f} s；{time.time() - t0:.0f} s", flush=True)
            else:
                print(f"終點 q{qr} ({f['x']:.2f},{f['y']:.2f})：{why}，只握回到 {qend}°；{time.time() - t0:.0f} s", flush=True)
            res.append(rec)
    finally:
        TJ.check = orig
    json.dump({'schema': 'door_coopen_edgehold/2', 'source': os.path.relpath(SRC, HERE), 'base_plan_m': BASE_PLAN, 'side_min_m': SIDE_MIN,
               'grasp': g, 'per_end': res, 'not_modeled': ['門餘轉動力學（區間包絡）', '圓盤近似', '取樣間碰撞', '夾持力與穩定性']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
