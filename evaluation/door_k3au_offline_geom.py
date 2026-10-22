#!/usr/bin/env python3
"""K3-A-U 卸載參考的離線靜態幾何核對（規格修訂 1 §2；只讀、點雲／盒／膠囊近似、取樣間未核）。

**目標相容**：以實跑停車錨點（door_k3a_ev_1 park_gate anchor）與 HOLD 門角，對卸載參考 p_ref0＋a·δ（δ ∈ [−3, 0] mm；a＝T_ref0 工具 z）
  逐點求 IK（自 δ＝0 連續），手指取 HOLD 實測閉合位置；報各剛體對門扇各部（門板、立柱、牆）與橫桿膠囊的最小距離。
  手指—橫桿＝設計接觸（夾持中），另列不判；其餘部件（夾爪殼、link_eef、link6…）須 ≥ 5 mm、底盤 ≥ 2 cm。
**實際追蹤安全**（另報、不混入目標相容）：door_k3a_ev_1 的 HOLD 實測 TCP 相對參考的接近軸投影（即本趟若做卸載時的 δ），
  以及該趟 HOLD 的非預期接觸／牆接觸最大值（實錄）。δ 更大（至 −3 mm）時的實際追蹤只有目標相容的證據，未實測。

    python3 evaluation/door_k3au_offline_geom.py   # 寫 results/door/door_k3au_offline_geom.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_offline_v2 as V2                       # noqa: E402
import door_geom_v2 as G2                             # noqa: E402
import door_k2_check as KC                            # noqa: E402

C, K1 = V2.C, V2.K1
OUT = os.path.join(HERE, 'results', 'door', 'door_k3au_offline_geom.json')
RUN = os.path.join(HERE, 'runs', 'door_k3a_ev_1')
DELTAS_MM = [0.0, -0.25, -0.5, -0.75, -1.05, -1.5, -2.0, -2.5, -3.0]
FINGERS = ('uflite_finger1', 'uflite_finger2')
ENV_M, BASE_M = 0.005, 0.02


def parts_clearance(qf, q_door, q_finger):
    W = C.world(qf, C.ARM_LINKS, q_finger)
    sh = G2.shapes(K1.D2, q_door, K1.H)
    out = {}
    for n, P in W.items():
        best = (math.inf, None)
        for name, c, R, hs in sh:
            d = float(G2.box_dist(P, c, R, hs).min())
            if d < best[0]:
                best = (d, name)
        dbar = float(G2.bar_capsule_dist(K1.D2, P, q_door, K1.H).min())
        out[n] = {'min_to_door_parts_m': best[0], 'who': best[1], 'min_to_bar_m': dbar}
    bc = min(G2.disc_clear(qf[0], qf[1], c[2], C.RB, c, R, hs) for name, c, R, hs in sh)
    return out, float(bc)


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t = json.load(open(os.path.join(RUN, 'task.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    gate = json.load(open(os.path.join(RUN, 'park_gate.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    hold = [s for s in r['steps'] if ev['HOLD'] <= s[I['sim_t']] < ev['OPEN']]
    base = np.array(gate['gate']['anchor'], float)
    q_door = float(np.median([s[I['door_q_rad']] for s in hold]))
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    HF = np.array(t['H_frozen'])
    THG = np.array(t['grasp_basis']['T_HG'])
    T0 = HF @ THG
    a = T0[:3, 2] / np.linalg.norm(T0[:3, 2])
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
        parts, bc = parts_clearance(qf, q_door, q_finger)
        non_finger = {n: v for n, v in parts.items() if n not in FINGERS}
        env_min = min(min(v['min_to_door_parts_m'], v['min_to_bar_m']) for v in non_finger.values())
        env_who = min(non_finger.items(), key=lambda kv: min(kv[1]['min_to_door_parts_m'], kv[1]['min_to_bar_m']))[0]
        fing_parts = min(parts[f]['min_to_door_parts_m'] for f in FINGERS if f in parts)
        rows.append({'delta_mm': dmm, 'ik': True, 'joint_margin_rad': C.jm(prev)[0],
                     'non_finger_env_min_m': env_min, 'non_finger_env_who': env_who,
                     'fingers_to_door_parts_min_m': fing_parts,
                     'fingers_to_bar_m': {f: parts[f]['min_to_bar_m'] for f in FINGERS if f in parts},
                     'base_clear_m': bc, 'ok_target_compat': env_min >= ENV_M and fing_parts >= ENV_M and bc >= BASE_M,
                     'parts': parts})
    # 實際追蹤（實錄）
    G = np.array(t['G_T_H'])
    dl = []
    for s in hold:
        Tm = KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        dl.append(float(a @ (Tm[:3, 3] - T0[:3, 3])))
    unexp = max((max([v.get('panel', 0.0) + v.get('ambiguous', 0.0) + (0.0 if k in FINGERS else v.get('bar', 0.0))
                      for k, v in (s[I['leaf_contacts']] or {}).items()] or [0.0])) for s in hold)
    wall = max((max((s[I['wall_contacts']] or {}).values() or [0.0])) for s in hold)
    res = {'schema': 'door_k3au_offline_geom/1', 'source_run': 'door_k3a_ev_1', 'base_anchor': base.tolist(), 'q_door_hold_deg': math.degrees(q_door),
           'q_finger_hold_m': q_finger, 'axis_a_world': a.tolist(), 'deltas_mm': DELTAS_MM,
           'criteria': {'non_finger_env_m': ENV_M, 'fingers_to_door_parts_m': ENV_M, 'base_m': BASE_M,
                        'finger_bar': '設計接觸（夾持中），只列不判'},
           'target_compat': rows,
           'actual_tracking_from_run': {'hold_delta_projection_mm': {'median': float(np.median(dl)) * 1e3, 'min': min(dl) * 1e3, 'max': max(dl) * 1e3, 'n': len(dl)},
                                        'hold_unexpected_contact_max_n': unexp, 'hold_wall_contact_max_n': wall,
                                        'note': '實錄（未卸載）。δ 至 −3 mm 的實際追蹤未實測，只有目標相容證據。'},
           'note': '點雲／盒／膠囊近似、取樣間未核；只支援規格，不是碰撞安全證明。'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for x in rows:
        if x['ik']:
            print(f"δ {x['delta_mm']:+.2f} mm：非手指最小 {x['non_finger_env_min_m'] * 1e3:.1f} mm（{x['non_finger_env_who']}）"
                  f"、手指對門扇各部 {x['fingers_to_door_parts_min_m'] * 1e3:.1f} mm、手指—橫桿 "
                  f"{ {k: round(v * 1e3, 2) for k, v in x['fingers_to_bar_m'].items()} } mm、底盤 {x['base_clear_m']:.3f} m、"
                  f"關節餘裕 {x['joint_margin_rad']:.3f}、ok {x['ok_target_compat']}")
        else:
            print(f"δ {x['delta_mm']:+.2f} mm：IK 失敗")
    print('實際追蹤（實錄 HOLD）：', res['actual_tracking_from_run'])


if __name__ == '__main__':
    main()
