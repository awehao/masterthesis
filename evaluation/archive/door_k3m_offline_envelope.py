#!/usr/bin/env python3
"""K3-M 前置離線核對：底盤包絡 × 門角的靜態取樣（無 Isaac、無接觸；依 Codex 20261023_200048）。

目的：為「底盤可動」模式的底盤包絡監看界限選值（取代固定底盤的 20 mm／2° 漂移中止），不是碰撞安全證明。
取樣：底盤相對停車錨點（錨點座標：前進＝錨點 yaw 方向、側向＝其左方）
  前進 {−0.15, −0.10, −0.05, 0, +0.02} m × 側向 {−0.05, 0, +0.05, +0.10} m × 偏航 {−6, −3, 0, +3, +6}°
  × 門角 {−1, 0, 2, 4, 6, 8, 9}°（登錄 OPEN 範圍 −1…9°）。
每個樣本：夾爪目標＝HOLD 目標繞鉸鏈轉該門角（剛性夾持）；以該底盤位姿解手臂 IK（由同一底盤位姿前一門角暖啟動）；
  記 IK 是否成功、關節餘裕、非手指剛體（含相機支架、夾爪殼）對門扇／門框／橫桿、手指對門板、對立柱、底盤圓盤淨空。
門檻沿用既有：非手指 ≥ 5 mm、手指對門板與立柱 ≥ 5 mm、底盤 ≥ 20 mm；關節餘裕另報。
另以 door_k3m_offline_replay.json（free）的底盤軌跡標示其所在格位。
近似：點雲／盒／膠囊、取樣之間未核、TCP 精確到達目標的靜態假設。

r2：補前進 +0.02 m（錨點附近的正向起步偏移），輸出 door_k3m_offline_envelope_r2.json；r1 輸出 door_k3m_offline_envelope.json 保留。

    python3 evaluation/door_k3m_offline_envelope.py   # 寫 results/door/door_k3m_offline_envelope_r2.json（已存在則拒絕覆寫）
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k3au_offline_geom as GA                   # noqa: E402
import door_k3ax_offline_geom as GX                   # noqa: E402
from door_target import rotate_about_hinge            # noqa: E402

C = GA.C
OUT = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_envelope_r2.json')
RUN = os.path.join(HERE, 'runs', 'door_k3a_ev_1')
FWD = [-0.15, -0.10, -0.05, 0.0, 0.02]
LAT = [-0.05, 0.0, 0.05, 0.10]
YAW_DEG = [-6.0, -3.0, 0.0, 3.0, 6.0]
Q_DEG = [-1.0, 0.0, 2.0, 4.0, 6.0, 8.0, 9.0]
AX = np.array([0.0, 0.0, -1.0])


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t = json.load(open(os.path.join(RUN, 'task.json')))
    r = json.load(open(os.path.join(RUN, 'room_run.json')))
    gate = json.load(open(os.path.join(RUN, 'park_gate.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    ev = {e['to']: e['t'] for e in t['policy_events'] if 'to' in e}
    hold = [s for s in r['steps'] if ev['HOLD'] <= s[I['sim_t']] < ev['OPEN']]
    q_finger = float(np.median([np.mean(s[I['q_finger']]) for s in hold]))
    ax, ay, ath = [float(x) for x in gate['gate']['anchor']]
    T_hold = np.array(t['H_frozen']) @ np.array(t['grasp_basis']['T_HG'])
    q2 = np.array(json.load(open(os.path.join(HERE, 'results', 'door', 'door_k2_offline_v2.json')))['Q_PRE_DOOR'])
    ca, sa = math.cos(ath), math.sin(ath)
    rows = []
    for fwd, lat, yd in itertools.product(FWD, LAT, YAW_DEG):
        base = np.array([ax + ca * fwd - sa * lat, ay + sa * fwd + ca * lat, ath + math.radians(yd)])
        prev = None
        for qd in Q_DEG:
            T = rotate_about_hinge(T_hold, np.zeros(3), AX, math.radians(qd))
            qf = C.ik(base, T, prev if prev is not None else q2, only_prev=prev is not None)
            row = {'fwd_m': fwd, 'lat_m': lat, 'dyaw_deg': yd, 'q_deg': qd, 'ik': qf is not None}
            if qf is None:
                rows.append(row)
                continue
            prev = qf[C.IDX]
            qdoor = math.radians(qd)
            parts, bc = GA.parts_clearance(qf, qdoor, q_finger)
            nf = {n: v for n, v in parts.items() if n not in GA.FINGERS}
            env = min(min(v['min_to_door_parts_m'], v['min_to_bar_m']) for v in nf.values())
            fing = min(parts[f]['min_to_door_parts_m'] for f in GA.FINGERS)
            post = min(GX.post_dist(qf, qdoor, q_finger, ('uflite_finger1', 'uflite_finger2', 'uflite_gripper_link', 'link_eef')).values())
            row.update({'joint_margin_rad': float(C.jm(prev)[0]), 'non_finger_env_min_m': env,
                        'non_finger_env_who': min(nf.items(), key=lambda kv: min(kv[1]['min_to_door_parts_m'], kv[1]['min_to_bar_m']))[0],
                        'fingers_to_door_parts_min_m': fing, 'to_posts_min_m': post, 'base_clear_m': bc,
                        'ok': env >= GA.ENV_M and fing >= GA.ENV_M and post >= GA.ENV_M and bc >= GA.BASE_M})
            rows.append(row)
    cells = {}
    for x in rows:
        k = (x['fwd_m'], x['lat_m'], x['dyaw_deg'])
        cells.setdefault(k, []).append(x)
    cell_ok = {k: all(x.get('ok') for x in v) for k, v in cells.items()}
    rp = os.path.join(HERE, 'results', 'door', 'door_k3m_offline_replay.json')
    traj = None
    if os.path.exists(rp):
        fr = next(c for c in json.load(open(rp))['configs'] if c['name'] == 'free')['summary']
        traj = {'base_fwd_range_m': fr['base_fwd_range_m'], 'base_lat_range_m': fr['base_lat_range_m'], 'base_dyaw_absmax_deg': fr['base_dyaw_absmax_deg']}
    bad = [dict(fwd_m=k[0], lat_m=k[1], dyaw_deg=k[2]) for k, ok in cell_ok.items() if not ok]
    res = {'schema': 'door_k3m_offline_envelope/1', 'source_run': 'door_k3a_ev_1', 'anchor': [ax, ay, ath],
           'grid': {'fwd_m': FWD, 'lat_m': LAT, 'dyaw_deg': YAW_DEG, 'q_deg': Q_DEG}, 'thresholds': {'env_m': GA.ENV_M, 'base_m': GA.BASE_M},
           'n_samples': len(rows), 'n_ik_fail': sum(1 for x in rows if not x['ik']), 'n_ok': sum(1 for x in rows if x.get('ok')),
           'n_cells': len(cells), 'n_cells_ok': sum(cell_ok.values()), 'cells_not_ok': bad,
           'min_over_ok_samples': {k: min(x[k] for x in rows if x.get('ok')) for k in ('non_finger_env_min_m', 'fingers_to_door_parts_min_m',
                                                                                     'to_posts_min_m', 'base_clear_m', 'joint_margin_rad')},
           'free_replay_trajectory': traj, 'samples': rows,
           'note': '靜態取樣（TCP 精確到達、剛性夾持、無接觸）；樣本之間未核；只支援底盤包絡監看界限的選值，不是碰撞安全證明。'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"樣本 {res['n_samples']}、IK 失敗 {res['n_ik_fail']}、ok {res['n_ok']}；格位 ok {res['n_cells_ok']}/{res['n_cells']}")
    print('不 ok 格位：', bad[:40])
    print('ok 樣本最小值：', {k: round(v * (1e3 if not k.startswith('joint') else 1), 2) for k, v in res['min_over_ok_samples'].items()})
    worst = {}
    for x in rows:
        if x['ik'] and not x['ok']:
            k = x['non_finger_env_who'] if x['non_finger_env_min_m'] < GA.ENV_M else ('base' if x['base_clear_m'] < GA.BASE_M else 'finger/post')
            worst[k] = worst.get(k, 0) + 1
    print('不 ok 原因計數：', worst)
    print('free 重播軌跡：', traj)


if __name__ == '__main__':
    main()
