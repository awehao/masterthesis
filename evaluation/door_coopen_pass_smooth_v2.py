#!/usr/bin/env python3
"""連續協同通行：放手後收臂 v2——繞回門扇自由端的笛卡兒收臂（離線靜態，無 Isaac）。

**v1 缺陷揭露**（door_coopen_pass_smooth_v1.json 的收臂結果作廢，路徑／時間化部分仍有效）：
  1. 張爪時底盤前進而手臂關節固定 ⇒ 夾爪被底盤拖向橫桿（非手指 −0.9 mm）；應為 TCP 相對門固定、手臂補償底盤；
  2. 收到 STOW 用關節線性內插 ⇒ 繞緣夾法的手臂繞過門扇自由端，直接內插會穿過門扇（−20 mm）；應先退開、沿門扇外移、繞回自由端外再收攏。
本版收臂（門角 q_rel；TCP 目標在世界座標固定、底盤每取樣點沿 v1 通過路徑前進 0.5 cm、偏航維持放手值、IK 以前一點為種子）：
  a. 張爪：TCP＝抓取位姿，finger_joint 0.0016 → 0.0089（3 點）；
  b. 沿工具 −z 退 60 mm（每 1 cm）；
  c. 沿門扇方向 u（往自由端）外移 0.30 m（每 1 cm）——把手在 0.85 m，外移後 TCP 越過自由端（0.95 m）；
  d. 沿 −n（門扇背面方向，底盤所在側）移 0.20 m（每 1 cm）；
  e. 關節線性內插到 STOW（20 點）。
  b–d 的 TCP 姿態維持抓取姿態；IK 失敗即記錄並停止。
每點以門角 q_rel + δ（δ ∈ {−2, −1, −0.5, 0, 0.5, 1, 2, 3, 5}°）核非手指／手指（張爪開度）／自碰 ≥ 5 mm，報 δ = 0 是否成立與自 0 連續可容許的 δ 範圍；
另核底盤 ≥ 30 mm（δ 包絡同 v1）。

    python3 evaluation/door_coopen_pass_smooth_v2.py   # 寫 results/door/door_coopen_pass_smooth_v2.json（不覆寫）
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
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402

SRC = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_smooth_v1.json')
SRC_H = os.path.join(HERE, 'results', 'door', 'door_coopen_edgehold_v2.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_pass_smooth_v2.json')
C = RV.C
DS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
TH = {'nonfinger': 0.005, 'finger': 0.005, 'self': 0.005}
BASE_PLAN = 0.030
Q_CLOSED, OPEN = RV.Q_FINGER, C.FINGER_OPEN
BASE_STEP = 0.005


def tol(qf, q_rel, qfi):
    ok, g0 = {}, None
    for dq in DS:
        g = TJ.GEO(qf, math.radians(q_rel + dq), qfi)
        if dq == 0.0:
            g0 = g
        ok[dq] = all(g[c][0] >= TH[c] for c in TH) and g['base'][0] >= BASE_PLAN
    pos = 0.0
    for dq in sorted(x for x in DS if x > 0):
        if not ok[dq]:
            break
        pos = dq
    neg = 0.0
    for dq in sorted((x for x in DS if x < 0), reverse=True):
        if not ok[dq]:
            break
        neg = dq
    return ok[0.0], pos, neg, {c: (float(g0[c][0]), g0[c][1]) for c in ('nonfinger', 'finger', 'self', 'base')}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    d = json.load(open(SRC))
    h = json.load(open(SRC_H))
    q_rel = d['q_rel_deg']
    cand = next(r for r in h['per_end'] if r['q_rel_deg'] == q_rel)
    end = cand['trajectory'][-1]
    yaw = end['base'][2]
    P = np.array(d['path'])
    q = math.radians(q_rel)
    g = h['grasp']
    Tg = EG.grasp_T(q, g['phi'], g['s'], g['roll'])
    u = np.array([math.cos(q), -math.sin(q), 0.0])
    n = np.array([-math.sin(q), -math.cos(q), 0.0])
    plan = [('open', Tg, Q_CLOSED + (OPEN - Q_CLOSED) * k / 2) for k in range(3)]
    for k in range(1, 7):
        plan.append(('retract', Tg @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.01 * k], [0, 0, 0, 1.0]]), OPEN))
    T1 = plan[-1][1]
    for k in range(1, 31):
        T = T1.copy()
        T[:3, 3] += u * 0.01 * k
        plan.append(('along_u', T, OPEN))
    T2 = plan[-1][1]
    for k in range(1, 21):
        T = T2.copy()
        T[:3, 3] += -n * 0.01 * k
        plan.append(('around_edge', T, OPEN))
    seed = np.array(end['qa'])
    rows, s_m, fail = [], 0.0, None
    for ph, T, qfi in plan:
        TJ.cool()
        k = min(int(round(s_m / 0.01)), len(P) - 1)
        base = np.array([P[k][0], P[k][1], yaw])
        q9 = C.ik(base, T, seed, only_prev=True)
        if q9 is None:
            q9 = C.ik(base, T, seed)
            if q9 is not None and float(np.max(np.abs(np.asarray(q9)[C.IDX] - seed))) > TJ.LIM['joint_step_rad']:
                q9 = None
        if q9 is None:
            fail = {'phase': ph, 's_m': s_m}
            break
        seed = np.asarray(q9)[C.IDX]
        ok0, pos, neg, g0 = tol(list(base) + list(seed), q_rel, qfi)
        rows.append({'phase': ph, 's_m': s_m, 'qa': seed.tolist(), 'q_finger': qfi, 'ok_at_0': ok0, 'delta_plus_ok_deg': pos,
                     'delta_minus_ok_deg': neg, 'geo0': g0})
        s_m += BASE_STEP
    if fail is None:
        qa0 = seed.copy()
        for k in range(1, 21):
            TJ.cool()
            kk = min(int(round(s_m / 0.01)), len(P) - 1)
            base = np.array([P[kk][0], P[kk][1], yaw])
            qa = (1 - k / 20) * qa0 + (k / 20) * TJ.STOW
            ok0, pos, neg, g0 = tol(list(base) + list(qa), q_rel, OPEN)
            rows.append({'phase': 'stow', 's_m': s_m, 'qa': qa.tolist(), 'q_finger': OPEN, 'ok_at_0': ok0, 'delta_plus_ok_deg': pos,
                         'delta_minus_ok_deg': neg, 'geo0': g0})
            s_m += BASE_STEP
    summ = {}
    for ph in ('open', 'retract', 'along_u', 'around_edge', 'stow'):
        r = [o for o in rows if o['phase'] == ph]
        if not r:
            continue
        summ[ph] = {'n': len(r), 'all_ok_at_0': all(o['ok_at_0'] for o in r),
                    'delta_plus_ok_min_deg': min(o['delta_plus_ok_deg'] for o in r), 'delta_minus_ok_max_deg': max(o['delta_minus_ok_deg'] for o in r),
                    'min0': {c: min(o['geo0'][c][0] for o in r) for c in ('nonfinger', 'finger', 'self', 'base')}}
        bad = [o for o in r if not o['ok_at_0']]
        print(f"{ph}：{len(r)} 點；δ=0 全過 {summ[ph]['all_ok_at_0']}（不過 {len(bad)}）；可容許續轉最小 {summ[ph]['delta_plus_ok_min_deg']}°、回轉 {summ[ph]['delta_minus_ok_max_deg']}°；"
              f"δ=0 最小 {({c: round(v * 1000, 1) for c, v in summ[ph]['min0'].items()})} mm"
              + ('' if not bad else f"；首個不過 {bad[0]['s_m']:.3f} m {({c: (round(v[0] * 1000, 1), v[1]) for c, v in bad[0]['geo0'].items()})}"), flush=True)
    print(f"IK 失敗：{fail}；底盤共前進 {s_m:.2f} m", flush=True)
    json.dump({'schema': 'door_coopen_pass_smooth/2', 'source': [os.path.relpath(SRC, HERE), os.path.relpath(SRC_H, HERE)], 'q_rel_deg': q_rel,
               'deltas_deg': DS, 'base_step_m': BASE_STEP, 'ik_fail': fail, 'per_point': rows, 'summary': summ,
               'v1_defects': ['張爪時手臂固定而底盤前進', '收臂以關節內插穿過門扇'],
               'not_modeled': ['張爪時的接觸力與滑移', '門餘轉動力學（逐點容許範圍代替）', '取樣間碰撞', '時間化（下一版併入）']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
