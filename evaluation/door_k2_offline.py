#!/usr/bin/env python3
"""K2 前置離線核對（門資產 v2 實際牆幾何；不開模擬）：候選停位 (前距 0.50, 側向 −0.20) 的接近、夾持、保持、釋放、退離。

停車誤差敏感度：停位 ± {0, 0.02} m（x、y）× {−2°, 0, +2°}（27 組），模擬「實測停車」不會恰在名目停位。
每組：A 預抓取接近 s 0.10 → 0（手指張開）→ B 夾持 s=0、門角 0 / 1° / 2°（門被帶動的小角度，手指閉合）→
C 釋放（s=0 手指張開）→ D 退離 s 0 → 0.10（手指張開）。底盤全程固定在該組停位。
可行條件沿用 door_offline_check_v1（IK 1 mm／0.01 rad、關節 ≥ 0.05、手臂環境 ≥ 5 mm（手指—橫桿設計接觸另報）、底盤圓盤 ≥ 2 cm、自碰 ≥ 5 mm）；
牆用 door_geom_v2（鉸鏈側退讓）；抓取基準實錄 T_HG（6.8 mm）。點雲／膠囊體／圓盤近似、取樣點間未核——只支援規格撰寫，不是碰撞安全證明。

    python3 evaluation/door_k2_offline.py     # 輸出 results/door/door_k2_offline.json（已存在則拒絕覆寫）
"""
import hashlib
import json
import math
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_geom_v2 as G2                              # noqa: E402
import door_offline_check_v1 as C                      # noqa: E402  IK、FK、自碰、目標（把手幾何 v0=v2）

SPEC_P = os.path.join(WS, 'src/my_omnibot_description/config/door_local_v2.yaml')
OUT = os.path.join(HERE, 'results', 'door', 'door_k2_offline.json')
D2 = yaml.safe_load(open(SPEC_P))
H = float(D2['handle']['height'])
STAND = (0.50, -0.20)
for k in ('wall', 'frame_clear_width', 'leaf'):
    pass
assert all(D2['handle'][k] == C.D['handle'][k] for k in ('from_hinge', 'standoff', 'bar', 'posts')) and D2['leaf'] == C.D['leaf']


def evaluate(qf, q, q_finger):
    if qf is None:
        return False, {'why': ['ik']}
    m, j = C.jm(qf[C.IDX])
    W = C.world(qf, C.ARM_LINKS, q_finger)
    sh = G2.shapes(D2, q, H)
    env, ewho, shell, cam = np.inf, None, np.inf, np.inf
    for n, P in W.items():
        for name, c, R, hs in sh:
            d = float(G2.box_dist(P, c, R, hs).min())
            if d < env:
                env, ewho = d, f'{n}|{name}'
            if n == 'uflite_gripper_link':
                shell = min(shell, d)
            if n == 'link_eef':
                cam = min(cam, d)
        db = float(G2.bar_capsule_dist(D2, P, q, H).min())
        if n == 'uflite_gripper_link':
            shell = min(shell, db)
        if n == 'link_eef':
            cam = min(cam, db)
        if n not in C.FINGERS and db < env:
            env, ewho = db, f'{n}|bar'
    bc, bwho = min((G2.disc_clear(qf[0], qf[1], c[2], C.RB, c, R, hs), name) for name, c, R, hs in sh)
    sc, swho = C.self_clear(qf, q_finger)
    why = [w for w, bad in (('joint', m < C.LIM_M), ('env', env < C.ENV_M), ('base', bc < C.BASE_M), ('self', sc < C.SELF_M)) if bad]
    return not why, {'joint_margin': m, 'joint': j, 'env': env, 'env_who': ewho, 'shell_min': shell, 'camera_stand_min': cam,
                     'base_clear': bc, 'base_who': bwho, 'self_clear': sc, 'self_who': swho, 'why': why or None}


def run_one(dx, dy, dyaw):
    T0 = C.target(0.0, 0.0)
    base = np.array([T0[0, 3] + STAND[1] + dx, T0[1, 3] - STAND[0] + dy, math.pi / 2 + math.radians(dyaw)])
    seq = ([('A_approach', 0.0, s, C.FINGER_OPEN) for s in np.linspace(0.10, 0.0, 11)]
           + [('B_grasp', math.radians(qd), 0.0, 0.0) for qd in (0.0, 1.0, 2.0)]
           + [('C_release', 0.0, 0.0, C.FINGER_OPEN)]
           + [('D_retreat', 0.0, s, C.FINGER_OPEN) for s in np.linspace(0.0, 0.10, 11)])
    prev, rows, max_step = None, [], 0.0
    qf = C.ik(base, C.target(0.0, 0.10), None)
    if qf is not None:
        prev = qf[C.IDX]
    for seg, q, s, fing in seq:
        qf = C.ik(base, C.target(q, s), prev, only_prev=prev is not None)
        ok, info = evaluate(qf, q, fing)
        if qf is not None and prev is not None:
            max_step = max(max_step, float(np.abs(qf[C.IDX] - prev).max()))
        if qf is not None:
            prev = qf[C.IDX]
        rows.append({'seg': seg, 'q_deg': round(math.degrees(q), 2), 's': round(float(s), 3), 'ok': ok, **info})
    okall = all(r['ok'] for r in rows)
    mins = {k: min((r[k] for r in rows if k in r), default=None) for k in ('joint_margin', 'env', 'shell_min', 'camera_stand_min', 'base_clear', 'self_clear')}
    return {'dx': dx, 'dy': dy, 'dyaw_deg': dyaw, 'base': base.tolist(), 'all_ok': okall, 'min': mins, 'max_joint_step_rad': max_step,
            'fails': [{k: r.get(k) for k in ('seg', 'q_deg', 's', 'why', 'env_who', 'base_who', 'self_who')} for r in rows if not r['ok']], 'rows': rows}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()   # noqa: E731
    res = {'schema': 'door_k2_offline/1', 'spec': os.path.relpath(SPEC_P, WS), 'spec_sha256': sha(SPEC_P), 'stand': STAND, 'handle_height_m': H,
           'grasp_basis': {'source': os.path.relpath(C.TASK_P, WS), 'grasp_depth_m': C.DEPTH},
           'scope': 'v2 牆幾何；接近→夾持（門角 0/1/2°）→釋放→退離；底盤固定於各擾動停位；近似與取樣限制同 v1', 'cases': []}
    for dx in (-0.02, 0.0, 0.02):
        for dy in (-0.02, 0.0, 0.02):
            for dyaw in (-2.0, 0.0, 2.0):
                r = run_one(dx, dy, dyaw)
                res['cases'].append(r)
                m = r['min']
                print(f"dx {dx:+.2f} dy {dy:+.2f} dyaw {dyaw:+.0f}°：{'OK ' if r['all_ok'] else 'NG '} jm {m['joint_margin']:.3f} env {m['env']*1e3:.1f}mm "
                      f"殼 {m['shell_min']*1e3:.1f} 相機 {m['camera_stand_min']*1e3:.1f} 底盤 {m['base_clear']:.3f} 自碰 {m['self_clear']*1e3:.1f}mm 跳 {r['max_joint_step_rad']:.3f}"
                      + ('' if r['all_ok'] else f"  {r['fails'][:2]}"), flush=True)
    ok = [c for c in res['cases'] if c['all_ok']]
    res['summary'] = {'n': len(res['cases']), 'n_all_ok': len(ok),
                      'min_over_ok': {k: min(c['min'][k] for c in ok) for k in ok[0]['min']} if ok else None,
                      'max_joint_step_over_ok': max(c['max_joint_step_rad'] for c in ok) if ok else None}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('摘要', res['summary'])


if __name__ == '__main__':
    main()
