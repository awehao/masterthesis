#!/usr/bin/env python3
"""K3-A-B 開跑前的雙向離線靜態幾何核對（規格 K3_finger_balance_compliance_spec.md §3＋Codex 20261023 第 4 項；只讀、近似、取樣間未核）。

與 door_k3ax_offline_geom.py 同法（同一份點雲／盒／膠囊近似與門幾何），但取樣涵蓋：
  - 門角：登錄 OPEN 監看範圍 −1° … 9°（−1, 0, 1, 2, 4, 6, 8, 9°）；
  - 柔順位移：Δ ∈ {−1, −0.5, 0, +0.5, +1} mm，沿 e₁₂＝R_target·(0, −1, 0)（工具 −y）。
目標＝HOLD 參考 T0＝H_frozen·T_HG 繞鉸鏈（原點、−z）轉該門角，再沿 e₁₂ 平移 Δ；IK 由前一個樣本暖啟動。
另做 §3 第 1 項：HOLD 實測姿態下兩指點雲重心在把手座標的 z（確認 finger1 在 +z_h、finger2 在 −z_h，對應 e₁₂ 指向 finger2 側）。
手指—橫桿為設計接觸，只列不判。結果只支援規格，不是碰撞安全或力—位移關係的證明。

    python3 evaluation/door_k3ab_offline_geom.py   # 寫 results/door/door_k3ab_offline_geom.json（已存在則拒絕覆寫）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3au_offline_geom as GA                   # noqa: E402
import door_k3ax_offline_geom as GX                   # noqa: E402
import door_k2_check as KC                            # noqa: E402
from door_target import rotate_about_hinge            # noqa: E402

C = GA.C
OUT = os.path.join(HERE, 'results', 'door', 'door_k3ab_offline_geom.json')
Q_DEG = [-1.0, 0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 9.0]
D_MM = [-1.0, -0.5, 0.0, 0.5, 1.0]
E12 = np.array([0.0, -1.0, 0.0])
AX = np.array([0.0, 0.0, -1.0])


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
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    HF, THG = np.array(t['H_frozen']), np.array(t['grasp_basis']['T_HG'])
    T0 = HF @ THG
    q2 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k2_offline_v2.json')))['Q_PRE_DOOR']
    prev0 = C.ik(base, T0, np.array(q2))
    prev0 = None if prev0 is None else prev0[C.IDX]
    # §3-1：HOLD 姿態下兩指 link 原點在把手座標的 z
    s_mid = hold[len(hold) // 2]
    qf_hold = np.concatenate([np.asarray(s_mid[I['base_xyth']], float), np.asarray(s_mid[I['q_arm_meas']], float)])
    Hinv = np.linalg.inv(KC._H_of(s_mid[I['door_q_rad']]))
    side = {}
    for b_, P in C.world(qf_hold, ('uflite_finger1', 'uflite_finger2'), q_finger).items():
        Ph = (Hinv[:3, :3] @ np.asarray(P, float).T).T + Hinv[:3, 3]
        side[b_] = float(np.mean(Ph[:, 2]))                      # 手指點雲重心的把手 z
    rows = []
    prev = prev0
    for qd in Q_DEG:
        Tq = rotate_about_hinge(T0, np.zeros(3), AX, math.radians(qd))
        e12 = Tq[:3, :3] @ E12
        for dmm in D_MM:
            T = Tq.copy()
            T[:3, 3] = Tq[:3, 3] + e12 * dmm * 1e-3
            qf = C.ik(base, T, prev, only_prev=prev is not None)
            if qf is None:
                rows.append({'q_deg': qd, 'delta_mm': dmm, 'ik': False})
                continue
            prev = qf[C.IDX]
            qdoor = math.radians(qd)
            parts, bc = GA.parts_clearance(qf, qdoor, q_finger)
            nf = {n: v for n, v in parts.items() if n not in GA.FINGERS}
            env_min = min(min(v['min_to_door_parts_m'], v['min_to_bar_m']) for v in nf.values())
            env_who = min(nf.items(), key=lambda kv: min(kv[1]['min_to_door_parts_m'], kv[1]['min_to_bar_m']))[0]
            fing_parts = min(parts[f]['min_to_door_parts_m'] for f in GA.FINGERS)
            posts = GX.post_dist(qf, qdoor, q_finger, ('uflite_finger1', 'uflite_finger2', 'uflite_gripper_link', 'link_eef'))
            ok = env_min >= GA.ENV_M and fing_parts >= GA.ENV_M and bc >= GA.BASE_M and min(posts.values()) >= GA.ENV_M
            rows.append({'q_deg': qd, 'delta_mm': dmm, 'ik': True, 'joint_margin_rad': C.jm(prev)[0], 'non_finger_env_min_m': env_min,
                         'non_finger_env_who': env_who, 'fingers_to_door_parts_min_m': fing_parts, 'to_posts_m': posts,
                         'fingers_to_bar_m': {f: parts[f]['min_to_bar_m'] for f in GA.FINGERS}, 'base_clear_m': bc, 'ok_target_compat': ok})
        prev = None if not rows or not rows[-1].get('ik') else prev
    res = {'schema': 'door_k3ab_offline_geom/1', 'source_run': 'door_k3a_ev_1', 'q_deg': Q_DEG, 'delta_mm': D_MM,
           'e12_tcp': E12.tolist(), 'finger_link_z_handle_m': side, 'q_finger_hold_m': q_finger, 'samples': rows,
           'n_ok': sum(1 for x in rows if x.get('ok_target_compat')), 'n_samples': len(rows),
           'thresholds': {'env_m': GA.ENV_M, 'base_m': GA.BASE_M},
           'note': '點雲／盒／膠囊近似、只在取樣點核對（樣本之間未核）；只支援規格，不是碰撞安全證明，也不是力—位移關係。手指—橫桿為設計接觸，只列不判。'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('兩指點雲重心的把手 z（m）：', side)
    for x in rows:
        if not x['ik']:
            print(f"q {x['q_deg']:+.0f}° Δ {x['delta_mm']:+.1f} mm：IK 失敗")
            continue
        print(f"q {x['q_deg']:+.0f}° Δ {x['delta_mm']:+.1f} mm：非手指 {x['non_finger_env_min_m'] * 1e3:.1f} mm（{x['non_finger_env_who']}）、"
              f"手指對門扇 {x['fingers_to_door_parts_min_m'] * 1e3:.1f}、立柱 {min(x['to_posts_m'].values()) * 1e3:.1f}、底盤 {x['base_clear_m'] * 1e3:.0f}、"
              f"關節 {x['joint_margin_rad']:.3f}、ok {x['ok_target_compat']}")
    print(f"ok {res['n_ok']}/{res['n_samples']}")


if __name__ == '__main__':
    main()
