#!/usr/bin/env python3
"""沿桿單軸參考重對齊的離線靜態幾何核對（規格 K3_bar_axis_realign_spec.md；只讀、近似、取樣間未核）。
與 door_k3au_offline_geom.py 同法，只把軸改為**把手 +x（桿軸）**＝H_frozen 旋轉第一欄；y／z／姿態不變。
δ ∈ {0, −0.25, …, −3} mm（依 door_k3a_ev_1 HOLD 實測：目標比 FK 多 +1.16 mm（把手 +x）⇒ δ≈−1.16）。
另報手指與夾爪殼對立柱（post0／post1）的最小距離（沿桿移動的主要風險）與實錄 HOLD 的沿桿投影。

    python3 evaluation/door_k3ax_offline_geom.py   # 寫 results/door/door_k3ax_offline_geom.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3au_offline_geom as GA                   # noqa: E402
import door_geom_v2 as G2                             # noqa: E402
import door_k2_check as KC                            # noqa: E402

C, K1, V2 = GA.C, GA.K1, GA.V2
OUT = os.path.join(HERE, 'results', 'door', 'door_k3ax_offline_geom.json')
DELTAS_MM = [0.0, -0.25, -0.5, -0.75, -1.16, -1.5, -2.0, -2.5, -3.0]


def post_dist(qf, q_door, q_finger, names):
    W = C.world(qf, names, q_finger)
    sh = {n: (c, R, hs) for n, c, R, hs in G2.shapes(K1.D2, q_door, K1.H) if n.startswith('post')}
    return {b: min(float(G2.box_dist(P, *sh[p]).min()) for p in sh) for b, P in W.items()}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t = json.load(open(os.path.join(GA.RUN, 'task.json')))
    r = json.load(open(os.path.join(GA.RUN, 'room_run.json')))
    gate = json.load(open(os.path.join(GA.RUN, 'park_gate.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    hold = [s for s in r['steps'] if ev['HOLD'] <= s[I['sim_t']] < ev['OPEN']]
    base = np.array(gate['gate']['anchor'], float)
    q_door = float(np.median([s[I['door_q_rad']] for s in hold]))
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    HF, THG = np.array(t['H_frozen']), np.array(t['grasp_basis']['T_HG'])
    T0 = HF @ THG
    a = HF[:3, 0] / np.linalg.norm(HF[:3, 0])
    q2 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k2_offline_v2.json')))['Q_PRE_DOOR']
    prev = C.ik(base, T0, np.array(q2))
    prev = None if prev is None else prev[C.IDX]
    rows = []
    for dmm in DELTAS_MM:
        T = T0.copy()
        T[:3, 3] = T0[:3, 3] + a * dmm * 1e-3
        qf = C.ik(base, T, prev, only_prev=prev is not None)
        if qf is None:
            rows.append({'delta_mm': dmm, 'ik': False})
            continue
        prev = qf[C.IDX]
        parts, bc = GA.parts_clearance(qf, q_door, q_finger)
        non_finger = {n: v for n, v in parts.items() if n not in GA.FINGERS}
        env_min = min(min(v['min_to_door_parts_m'], v['min_to_bar_m']) for v in non_finger.values())
        env_who = min(non_finger.items(), key=lambda kv: min(kv[1]['min_to_door_parts_m'], kv[1]['min_to_bar_m']))[0]
        fing_parts = min(parts[f]['min_to_door_parts_m'] for f in GA.FINGERS)
        posts = post_dist(qf, q_door, q_finger, ('uflite_finger1', 'uflite_finger2', 'uflite_gripper_link', 'link_eef'))
        rows.append({'delta_mm': dmm, 'ik': True, 'joint_margin_rad': C.jm(prev)[0], 'non_finger_env_min_m': env_min, 'non_finger_env_who': env_who,
                     'fingers_to_door_parts_min_m': fing_parts, 'to_posts_m': posts,
                     'fingers_to_bar_m': {f: parts[f]['min_to_bar_m'] for f in GA.FINGERS}, 'base_clear_m': bc,
                     'ok_target_compat': env_min >= GA.ENV_M and fing_parts >= GA.ENV_M and bc >= GA.BASE_M and min(posts.values()) >= GA.ENV_M})
    dl = [float(a @ (KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])[:3, 3] - T0[:3, 3])) for s in hold]
    res = {'schema': 'door_k3ax_offline_geom/1', 'source_run': 'door_k3a_ev_1', 'axis_a_world': a.tolist(), 'axis': '把手 +x（桿軸）＝H_frozen 旋轉第一欄',
           'q_door_hold_deg': math.degrees(q_door), 'q_finger_hold_m': q_finger, 'deltas_mm': DELTAS_MM, 'target_compat': rows,
           'actual_tracking_from_run': {'hold_bar_axis_projection_mm': {'median': float(np.median(dl)) * 1e3, 'min': min(dl) * 1e3, 'max': max(dl) * 1e3, 'n': len(dl)},
                                        'note': '實錄（未重對齊）。δ 至 −3 mm 的實際追蹤未實測。'},
           'note': '點雲／盒／膠囊近似、取樣間未核；只支援規格，不是碰撞安全證明。手指—橫桿為設計接觸，只列不判。'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for x in rows:
        print(f"δ {x['delta_mm']:+.2f} mm：非手指最小 {x['non_finger_env_min_m'] * 1e3:.1f} mm（{x['non_finger_env_who']}）、手指對門扇 "
              f"{x['fingers_to_door_parts_min_m'] * 1e3:.1f} mm、對立柱 { {k[-7:]: round(v * 1e3, 1) for k, v in x['to_posts_m'].items()} } mm、"
              f"關節 {x['joint_margin_rad']:.3f}、ok {x['ok_target_compat']}" if x['ik'] else f"δ {x['delta_mm']} IK 失敗")
    print('實錄 HOLD 沿桿投影：', res['actual_tracking_from_run']['hold_bar_axis_projection_mm'], 'axis', np.round(a, 4))


if __name__ == '__main__':
    main()
