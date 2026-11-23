#!/usr/bin/env python3
"""連續協同通行：主候選（edgehold_v3）的 H0 起點接近與出口交還核對 v1（Codex 20261026_183611 §3-4；離線靜態，無 Isaac）。

H0（門 0°）：
  1. 底盤自 (x0, y0 − 0.80)（門前側空地）沿直線到起點 (x0, y0)，圓盤淨空（每 1 cm）；起點偏航直接取持握起點偏航（到位前轉好）；
  2. 手臂自 K2_READY（STOW）以關節線性內插 40 點展開到預抓取（TCP 沿工具 −z 退 60 mm，IK 以抓取解為種子），各類淨空；
  3. 開指接近 60 → 0 mm 與閉合首觸（沿用 door_coopen_grasp_contract_v1 的 approach／closure）。
出口交還：通過路徑（pass_smooth_v1）終點，手臂已收到 STOW、偏航＝放手偏航＋90°（pass_smooth_v3）：
  核整機（底盤圓盤＋手臂各連桿）對門扇／立柱／兩側牆淨空，門角 q_rel −2 … +5°；並報底盤中心距牆面（y＝0.12）與整機包絡是否已離開門洞（y_min ≥ 0.12）。
**界線**：靜態；導航接手後的路徑與 CBF 行為不在內；起點接近路徑為直線示意，非導航規劃結果。

    python3 evaluation/door_coopen_h0_exit_v1.py   # 寫 results/door/door_coopen_h0_exit_v1.json（不覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_coopen_edgegrasp_v1 as EG                  # noqa: E402
import door_coopen_grasp_contract_v1 as GC             # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_geom_v2 as G2                              # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_h0_exit_v1.json')
C, K1 = RV.C, RV.V2.K1


def disc_min(x, y, q):
    return min(float(G2.disc_clear(x, y, c[2], C.RB, c, R, hs)) for _, c, R, hs in G2.shapes(K1.D2, q, K1.H))


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    h3 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_coopen_edgehold_v3.json')))
    g = json.load(open(os.path.join(HERE, 'results', 'door', 'door_coopen_edgehold_v2.json')))['grasp']
    s3 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_coopen_pass_smooth_v3.json')))
    pv = json.load(open(os.path.join(HERE, 'results', 'door', 'door_coopen_pass_smooth_v1.json')))
    st = h3['trajectory'][0]
    base, qa = st['base'], st['qa']
    T0 = EG.grasp_T(0.0, g['phi'], g['s'], g['roll'])
    line = [disc_min(base[0], float(y), 0.0) for y in np.linspace(base[1] - 0.80, base[1], 81)]
    ap = GC.approach(base, T0, np.asarray(qa), 0.0)
    cl = GC.closure([float(v) for v in base] + [float(v) for v in qa], 0.0, np.array([0.0, -1.0, 0.0]))
    pre = next((r for r in ap if r['d_mm'] == 60 and r['ik']), None)
    h0 = GC.h0(base, pre['qf'][3:], 0.0) if pre else None
    ok = [r for r in ap if r['ik']]
    H0 = {'start_base': base, 'base_line_min_m': min(line), 'approach_n_ik_fail': sum(1 for r in ap if not r['ik']),
          'approach_min': {k: min(r['geo'][k][0] for r in ok) for k in ('nonfinger', 'finger', 'base', 'self')} if ok else None,
          'closure_first_contact': [None if f is None else f['q_finger'] for f in cl['first_contact']],
          'closure_normal_angle_deg': cl.get('normal_angle_deg'), 'unfold': h0}
    print(f"H0：起點 {np.round(base[:2], 3)} 偏航 {math.degrees(base[2]):.0f}°；底盤直線接近最小 {H0['base_line_min_m'] * 1000:.0f} mm；"
          f"開指接近 IK 失敗 {H0['approach_n_ik_fail']}、最小 {({k: round(v * 1000, 1) for k, v in (H0['approach_min'] or {}).items()})} mm；"
          f"首觸 {H0['closure_first_contact']}、法向夾角 {H0['closure_normal_angle_deg']}；"
          f"展開最小 {({k: round(v * 1000, 1) for k, v in h0['unfold_min'].items()}) if h0 else None} mm（{h0['unfold_min_who'] if h0 else None}）", flush=True)
    q_rel = s3['q_rel_deg']
    end = pv['path'][-1]
    yaw = s3['per_point'][-1]['yaw']
    qf = [end[0], end[1], yaw] + list(TJ.STOW)
    ex = {}
    for dq in (-2.0, 0.0, 5.0):
        d = TJ.GEO(qf, math.radians(q_rel + dq), C.FINGER_OPEN)
        ex[f'q{q_rel + dq:.0f}'] = {k: [float(d[k][0]), d[k][1]] for k in ('nonfinger', 'finger', 'base', 'self')}
    pts = []
    for n_ in C.ARM_LINKS:
        T = C.K.fk(np.asarray(qf, float), n_)
        pts.append((T[:3, :3] @ C.CL[n_].T).T + T[:3, 3])
    y_min_arm = float(np.min(np.concatenate(pts)[:, 1]))
    EXIT = {'base_end': end, 'yaw_deg': math.degrees(yaw), 'clear': ex, 'base_center_minus_wall_m': end[1] - 0.12,
            'envelope_y_min_m': min(end[1] - C.RB, y_min_arm), 'out_of_doorway': min(end[1] - C.RB, y_min_arm) >= 0.12}
    print(f"出口：底盤 {np.round(end, 3)}、偏航 {EXIT['yaw_deg']:.0f}°；整機包絡 y_min {EXIT['envelope_y_min_m']:.3f}（牆面 0.12）⇒ 離開門洞 {EXIT['out_of_doorway']}；"
          f"淨空 {({k: {c: round(v[0] * 1000, 1) for c, v in m.items()} for k, m in ex.items()})} mm", flush=True)
    json.dump({'schema': 'door_coopen_h0_exit/1', 'H0': H0, 'exit': EXIT,
               'not_modeled': ['導航接手後的路徑與 CBF 行為', '起點接近路徑為直線示意', '動力學與追蹤']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
