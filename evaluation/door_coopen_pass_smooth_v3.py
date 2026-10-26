#!/usr/bin/env python3
"""連續協同通行：放手後收臂 v3——底盤轉身讓出 j1，手臂繞回門扇自由端（離線靜態，無 Isaac）。

**v2 結果與診斷**（door_coopen_pass_smooth_v2.json）：張爪＋沿工具 −z 退 60 mm 成立；沿門扇外移 5 cm 內 IK 失敗。
持握 55° 候選的 j1 由 0° 時約 0.6 rad 隨開門一路增加到放手時 2.65 rad（退開後 2.82），上限 2.89：手臂在持握中被「繞」上門扇自由端，
放手後夾爪往自由端移需要 j1 再增加而頂到上限；底盤固定的關節空間 RRT-connect（door_coopen_unwrap_rrt_v1，4000 次）也未找到路徑。
探測（scratchpad，固定底盤位置、2 mm 步）：同一笛卡兒退法若底盤偏航同時 +45°／+90° 則全程成立（j1 降到 1.33／0.76），−45°／−90° 於 j1 2.93 失敗。
本版：沿用 v2 的 a 張爪、b 沿 −z 退 60 mm，接著
  c. 沿門扇方向 u 外移 0.30 m、d. 沿 −n 移 0.20 m（每 2 mm 一點，TCP 姿態固定），同時底盤偏航線性增加 DYAW（預設 +90°）、
     底盤位置沿 v1 通過路徑每點前進 BASE_PER_PT；
  e. 關節線性內插到 STOW（40 點），底盤續沿路徑前進；
  每點核 IK 連續（相鄰 ≤ 0.10 rad）、δ ∈ {−2…5}° 下非手指／手指／自碰 ≥ 5 mm、底盤 ≥ 30 mm，報 δ=0 是否成立與可容許範圍。
時間化（運動學下界）：每點所需時間＝max(TCP 位移／0.05 m/s, 手臂最大關節變化／1.0 rad/s, 偏航變化／0.20 rad/s, 底盤位移／0.035 m/s)。

    python3 evaluation/door_coopen_pass_smooth_v3.py [DYAW_deg] [BASE_PER_PT_m]   # 寫 results/door/door_coopen_pass_smooth_v3.json（不覆寫）
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
import door_coopen_pass_smooth_v2 as S2                # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_smooth_v3.json')
C = RV.C
V_TCP, ARM_RATE, W_ANG, V_BASE = 0.05, 1.0, 0.20, 0.035


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    dyaw = math.radians(float(sys.argv[1]) if len(sys.argv) > 1 else 90.0)
    bstep = float(sys.argv[2]) if len(sys.argv) > 2 else 0.001
    d2 = json.load(open(S2.OUT))
    pv = json.load(open(S2.SRC))
    h = json.load(open(S2.SRC_H))
    q_rel = d2['q_rel_deg']
    q = math.radians(q_rel)
    yaw0 = next(r for r in h['per_end'] if r['q_rel_deg'] == q_rel)['trajectory'][-1]['base'][2]
    P = np.array(pv['path'])
    g = h['grasp']
    Tg = EG.grasp_T(q, g['phi'], g['s'], g['roll'])
    u = np.array([math.cos(q), -math.sin(q), 0.0])
    n = np.array([-math.sin(q), -math.cos(q), 0.0])
    rows = [dict(p, yaw=yaw0) for p in d2['per_point'] if p['phase'] in ('open', 'retract')]
    last = rows[-1]
    s_m = last['s_m']
    seed = np.array(last['qa'])
    p = (Tg @ np.array([0, 0, -0.06, 1.0]))[:3]
    pts = []
    for ph, vec, L in (('along_u', u, 0.30), ('around_edge', -n, 0.20)):
        for _ in range(int(round(L / 0.002))):
            p = p + vec * 0.002
            pts.append((ph, p.copy()))
    fail = None
    for i, (ph, pp) in enumerate(pts):
        TJ.cool()
        s_m += bstep
        k = min(int(round(s_m / 0.01)), len(P) - 1)
        yaw = yaw0 + dyaw * (i + 1) / len(pts)
        base = np.array([P[k][0], P[k][1], yaw])
        T = np.eye(4)
        T[:3, :3] = Tg[:3, :3]
        T[:3, 3] = pp
        q9 = C.ik(base, T, seed, only_prev=True)
        if q9 is None or float(np.max(np.abs(np.asarray(q9)[C.IDX] - seed))) > TJ.LIM['joint_step_rad']:
            fail = {'phase': ph, 'i': i, 'why': 'ik' if q9 is None else 'joint_step', 'j1': float(seed[0])}
            break
        seed = np.asarray(q9)[C.IDX]
        ok0, pos, neg, g0 = S2.tol(list(base) + list(seed), q_rel, C.FINGER_OPEN)
        rows.append({'phase': ph, 's_m': s_m, 'yaw': float(yaw), 'tcp': pp.tolist(), 'qa': seed.tolist(), 'q_finger': C.FINGER_OPEN,
                     'ok_at_0': ok0, 'delta_plus_ok_deg': pos, 'delta_minus_ok_deg': neg, 'geo0': g0})
    if fail is None:
        qa0, yaw_end = seed.copy(), yaw0 + dyaw
        for kk in range(1, 41):
            TJ.cool()
            s_m += 0.0075
            k = min(int(round(s_m / 0.01)), len(P) - 1)
            base = np.array([P[k][0], P[k][1], yaw_end])
            qa = (1 - kk / 40) * qa0 + (kk / 40) * TJ.STOW
            ok0, pos, neg, g0 = S2.tol(list(base) + list(qa), q_rel, C.FINGER_OPEN)
            rows.append({'phase': 'stow', 's_m': s_m, 'yaw': float(yaw_end), 'qa': qa.tolist(), 'q_finger': C.FINGER_OPEN,
                         'ok_at_0': ok0, 'delta_plus_ok_deg': pos, 'delta_minus_ok_deg': neg, 'geo0': g0})
    dts = []
    for a, b in zip(rows, rows[1:]):
        ka, kb = min(int(round(a['s_m'] / 0.01)), len(P) - 1), min(int(round(b['s_m'] / 0.01)), len(P) - 1)
        dtcp = np.linalg.norm(np.subtract(b['tcp'], a['tcp'])) if ('tcp' in a and 'tcp' in b) else 0.0
        dts.append(max(dtcp / V_TCP, float(np.max(np.abs(np.subtract(b['qa'], a['qa'])))) / ARM_RATE,
                       abs(b['yaw'] - a['yaw']) / W_ANG, float(np.linalg.norm(P[kb] - P[ka])) / V_BASE))
    summ = {}
    for ph in ('open', 'retract', 'along_u', 'around_edge', 'stow'):
        r = [o for o in rows if o['phase'] == ph]
        if r:
            summ[ph] = {'n': len(r), 'all_ok_at_0': all(o['ok_at_0'] for o in r), 'n_bad_at_0': sum(1 for o in r if not o['ok_at_0']),
                        'delta_plus_ok_min_deg': min(o['delta_plus_ok_deg'] for o in r), 'delta_minus_ok_max_deg': max(o['delta_minus_ok_deg'] for o in r),
                        'min0_mm': {c: min(o['geo0'][c][0] for o in r) * 1000 for c in ('nonfinger', 'finger', 'self', 'base')}}
            print(f"{ph}：{len(r)} 點；δ=0 不過 {summ[ph]['n_bad_at_0']}；可容許 續轉 ≥ {summ[ph]['delta_plus_ok_min_deg']}°、回轉 ≤ {summ[ph]['delta_minus_ok_max_deg']}°；"
                  f"最小 {({c: round(v, 1) for c, v in summ[ph]['min0_mm'].items()})} mm", flush=True)
    print(f"失敗：{fail}；底盤沿路徑前進 {s_m:.3f} m、偏航 {math.degrees(dyaw):+.0f}°；收臂時長下界 {sum(dts):.1f} s", flush=True)
    json.dump({'schema': 'door_coopen_pass_smooth/3', 'q_rel_deg': q_rel, 'dyaw_deg': math.degrees(dyaw), 'base_per_pt_m': bstep,
               'fail': fail, 'per_point': rows, 'summary': summ, 'retract_time_lb_s': float(sum(dts)), 'base_advance_m': s_m,
               'rates': {'v_tcp': V_TCP, 'arm': ARM_RATE, 'yaw': W_ANG, 'base': V_BASE},
               'not_modeled': ['張爪時的接觸力與滑移', '門餘轉動力學（逐點容許範圍代替）', '取樣間碰撞', '控制器追蹤']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
