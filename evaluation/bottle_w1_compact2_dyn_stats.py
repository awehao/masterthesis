#!/usr/bin/env python3
"""緊湊 v2 移動物兩案 vs 前持同案（只讀紀錄）：抓後到 PARKED 的近停區間（平移 < 0.01 m/s 連續 ≥ 0.1 s）、底盤圓（r 0.30）／瓶身／夾爪格點到移動物圓柱面的水平最小距離、底盤偏航最大。
    cd evaluation && python3 bottle_w1_compact2_dyn_stats.py
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '../src/ammr_wholebody_mpc')
import bottle_s1_geom as G, bottle_w1_compact_pose as CP, bottle_w1_compactcarry2_ctrl as C2, bottle_w1_dyn_case as DC
q_side = np.array(json.load(open('results/bottle/w1_compact_pose_v2.json'))['reference']['q_side'])
GG, BP = C2.grip_grid_tcp(q_side), G.bottle_points(12, 5)
out = {}
for run, case in (('bottle_w1_dyn1fc1', DC.CASE), ('bottle_w1_dyn1cc2_1', DC.CASE), ('bottle_w1_dyn2fc1', DC.CASE2), ('bottle_w1_dyn2cc2_1', DC.CASE2)):
    D = os.path.join('runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t = c['t']; tg = r['t_grasp']
    pe = {e.get('to'): e['t'] for e in r['events'] if e['what'] == 'place_phase'}
    tpark = pe.get('PARKED', 1e9)
    v = np.r_[0, np.hypot(np.diff(c['base_x']), np.diff(c['base_y'])) / np.diff(t)]
    # 近停：抓後到 PARKED 前，平移 < 0.01 m/s 連續 ≥ 0.1 s
    m = (t >= tg) & (t < tpark) & (v < 0.01)
    stops, i = [], 0
    while i < len(t):
        if m[i]:
            j = i
            while j + 1 < len(t) and m[j + 1]:
                j += 1
            if t[j] - t[i] >= 0.1:
                stops.append((round(float(t[i]), 2), round(float(t[j]), 2)))
            i = j + 1
        else:
            i += 1
    R, H = case['radius'], case['height']
    dmin = dict(base=9.9, bottle=9.9, gripper=9.9)
    for k in range(0, len(t), 2):
        if t[k] < tg or t[k] > tpark or not np.isfinite(c['dyn_x'][k]):
            continue
        ox, oy = c['dyn_x'][k], c['dyn_y'][k]
        db = math.hypot(c['base_x'][k] - ox, c['base_y'][k] - oy) - R - 0.30
        q = np.array([c['base_x'][k], c['base_y'][k], c['base_yaw'][k]] + [c[f'q{j}'][k] for j in range(1, 7)])
        Tt = G.C.K.fk(q, 'link_tcp')
        Rb = CP.quat(*(c[n][k] for n in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz')))
        Pb = (Rb @ BP.T).T + np.array([c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]])
        Pg = (Tt[:3, :3] @ GG.T).T + Tt[:3, 3]
        dmin['base'] = min(dmin['base'], db)
        dmin['bottle'] = min(dmin['bottle'], float((np.hypot(Pb[:, 0] - ox, Pb[:, 1] - oy) - R).min()))
        dmin['gripper'] = min(dmin['gripper'], float((np.hypot(Pg[:, 0] - ox, Pg[:, 1] - oy) - R).min()))
    ev = {e['what']: e['t'] for e in r['events'] if e['what'] in ('compact_start', 'compact_reached', 'compact_return', 'compact_returned', 'front_carry_start', 'front_carry_reached', 'mission_done')}
    out[run] = dict(done=ev.get('mission_done'), parked=pe.get('PARKED'), ev=ev, stops=stops, dmin_mm={k: round(v_ * 1e3, 1) for k, v_ in dmin.items()},
                    yaw_max=round(float(np.degrees(np.abs(c['base_yaw'][(t >= tg) & (t <= tpark)]).max())), 2))
    print(run, json.dumps(out[run], ensure_ascii=False))
json.dump(out, open('results/bottle/w1_compact2_dyn_stats.json', 'w'), ensure_ascii=False, indent=1)
