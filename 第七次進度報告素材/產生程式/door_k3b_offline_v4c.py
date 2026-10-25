#!/usr/bin/env python3
"""K3-B 離線核對 v4c（Codex 20261025_230812 修訂所需；v4、v4b 輸出不改）。

一、HELD 起點的手指—橫桿模型距離
  door_wide_w2c_30_1 HELD 實測組態；兩指各用自己的實測開度；只算兩個指定配對（finger1／finger2 對橫桿膠囊體）。
  Codex 以兩指平均開度重算為約 −0.25／−0.30 mm。本項逐指重算並另報平均開度版。
  用途：說明 t_open 時不能要求 d ≥ 0（夾持中為有界設計接觸）。
二、近接觸區細格門角帶（剛體擾動同 v4：ex ±3、ey ±1、ez ±Z(s)、轉角 ±0.5°；開度 95 %）
  - s 每 0.0025 m（0～0.05），之後每 0.01 m 到 0.10；Δ 細格。
  - 分類取最差：(a) 指定配對＝finger1／finger2 對橫桿；(b) 非手指＝link6、link_eef、uflite_gripper_link 對門扇／立柱／牆／橫桿，
    以及手指對門扇／立柱／牆（只有對橫桿可例外）。
  - 每個 s 報：非手指 ≥ 5 mm 的 Δ 上限；指定配對 ≥ 0、≥ 5 mm 的 Δ 上限。
  - 單調化：帶上界取「自本 s 到退離終點」的最小值（running min），消除越界陷阱，例如 s＝0.02 處 +0.75° → +0.1°。
三、收臂路線改用 K2_READY（實跑所用 --stow-q）、**無任何自碰豁免**
  - 路線：退離終點 → Q_PRE_DOOR（展開律）→ K2_READY（自 K2_READY 以同一律展開到 Q_PRE_DOOR 的軌跡反序）；
  - 門角 Δ ∈ −3～+5°；逐點 K1.evaluate（關節 50 mm、env 5 mm、底盤 20 mm、自碰 5 mm）。
近似同 v4，不是碰撞安全證明。

    python3 evaluation/door_k3b_offline_v4c.py   # 輸出 results/door/door_k3b_offline_v4c.json（已存在則拒絕覆寫）
"""
from __future__ import annotations

import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_geom_v2 as G2                              # noqa: E402
import door_k2_check as KC                             # noqa: E402
import door_k2_offline_v2 as V2                        # noqa: E402
import door_k3_offline_v3 as V3                        # noqa: E402

C, K1 = V2.C, V2.K1
V4 = os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4c.json')
K2_READY = np.array([0.0, 0.0, 0.035, 0.0, -1.5707963, 0.0])                # run_door_wide_*.sh 的 --stow-q
S_FINE = [round(0.0025 * k, 4) for k in range(21)] + [round(0.01 * k, 2) for k in range(6, 11)]
D_FINE = (-1.0, -0.5, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
D_STOW = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
FINGERS = ('uflite_finger1', 'uflite_finger2')
NONF = ('link6', 'link_eef', 'uflite_gripper_link')
Z_OF = (lambda s: 0.004 if s < 0.01 else 0.006 if s < 0.02 else 0.007)


def law_traj(q0, qg, j3_clear=False):
    U = V2.UNFOLD
    q, tr, cleared = q0.copy(), [q0.copy()], not j3_clear
    for _ in range(20000):
        if not cleared and q[2] < U['j3_clear_rad']:
            qd = np.zeros(6)
            qd[2] = min(U['qd_max'], max(0.05, U['gain'] * (U['j3_clear_rad'] - q[2])))
        else:
            cleared = True
            qd = np.clip(U['gain'] * (qg - q), -U['qd_max'], U['qd_max'])
        nxt = q + qd * U['dt']
        qd[(nxt < V2.ELO) | (nxt > V2.EHI)] = 0.0
        q = q + qd * U['dt']
        tr.append(q.copy())
        if cleared and float(np.abs(qg - q).max()) <= U['q_tol']:
            break
    return tr


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    v4 = json.load(open(V4))
    hs = v4['held_state']
    base, qr = np.array(hs['base']), hs['q_door']
    H, E, aH = KC._H_of(qr), np.array(v4['tool_in_handle_measured']), np.array(v4['a_H_handle'])
    r = json.load(open(os.path.join(HERE, 'runs', v4['source_run'], 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    row = min(r['steps'], key=lambda s: abs(s[I['sim_t']] - hs['t']))
    qfing = [float(x) for x in row[I['q_finger']]]
    qf = np.zeros(9)
    qf[:3], qf[C.IDX] = base, hs['q_arm']
    # 一、HELD 起點兩配對
    held = {}
    for label, ops in (('per_finger', qfing), ('mean', [sum(qfing) / 2] * 2)):
        o = {}
        for n, op in zip(FINGERS, ops):
            P = C.world(qf, (n,), op)[n]
            o[n] = float(G2.bar_capsule_dist(K1.D2, P, qr, K1.H).min())
        held[label] = o
    held['q_finger_measured_m'] = qfing
    # 二、細格門角帶
    T0 = H @ E
    local, owner = {}, {}
    W = C.world(qf, V3.RIGID, C.FINGER_OPEN * 0.95)
    local = np.vstack([(np.linalg.inv(T0) @ np.c_[P, np.ones(len(P))].T).T[:, :3] for P in W.values()])
    owner = np.concatenate([[n] * len(P) for n, P in W.items()])
    is_f = np.isin(owner, FINGERS)
    band = {}
    for s in S_FINE:
        Z = Z_OF(s)
        base_T = np.eye(4)
        base_T[:3, 3] = aH * s
        tool = base_T @ E
        poses = []
        for ez, ex, ey, (ax, ra) in itertools.product((-Z, -Z / 2, 0.0, Z / 2, Z), V4_EX, V4_EY, V3.RHO):
            P = tool.copy()
            if ax is not None:
                P[:3, :3] = P[:3, :3] @ V3.rot(ax, ra)
            P[:3, 3] += (ex, ey, ez)
            poses.append(H @ P)
        TT = np.array(poses)
        X = np.einsum('pij,nj->pni', TT[:, :3, :3], local) + TT[:, None, :3, 3]
        rows = {}
        for dd in D_FINE:
            qd = qr + math.radians(dd)
            sh = G2.shapes(K1.D2, qd, K1.H)
            flat = X.reshape(-1, 3)
            d_box = np.min(np.stack([G2.box_dist(flat, c, R, h) for _, c, R, h in sh]), axis=0).reshape(X.shape[:2])
            d_bar = G2.bar_capsule_dist(K1.D2, flat, qd, K1.H).reshape(X.shape[:2])
            pair = float(d_bar[:, is_f].min())                                  # 指定配對：手指—橫桿
            nonf = float(min(d_box.min(), d_bar[:, ~is_f].min()))               # 非手指全部；手指對門扇／立柱／牆
            rows[f'{dd:g}'] = {'pair_mm': pair * 1e3, 'nonfinger_mm': nonf * 1e3}
        band[f'{s:g}'] = rows

    def max_ok(s, key, th):
        ok = [float(d) for d, v in band[f'{s:g}'].items() if v[key] >= th]
        return max(ok) if ok else None
    summ = []
    for s in S_FINE:
        summ.append({'s': s, 'nonfinger_ge5_max': max_ok(s, 'nonfinger_mm', 5.0), 'pair_ge0_max': max_ok(s, 'pair_mm', 0.0),
                     'pair_ge5_max': max_ok(s, 'pair_mm', 5.0)})
    # 單調化（running min 自終點往回）
    for key in ('nonfinger_ge5_max', 'pair_ge0_max', 'pair_ge5_max'):
        run = math.inf
        for x in reversed(summ):
            v = x[key] if x[key] is not None else -math.inf
            run = min(run, v)
            x[key + '_monotone'] = run
    # 三、收臂（K2_READY、無豁免）
    prev = np.array(hs['q_arm'])
    for s in v4['s_path']:
        T = np.eye(4)
        T[:3, 3] = aH * s
        prev = C.ik(base, H @ T @ E, prev, only_prev=True)[C.IDX]
    q_end = prev
    qpre = np.array(json.load(open(os.path.join(HERE, 'results', 'door', 'door_k2_offline_v2.json')))['Q_PRE_DOOR'])
    segs = {'end_to_qpre': law_traj(q_end, qpre), 'qpre_to_k2ready': law_traj(K2_READY, qpre, j3_clear=True)[::-1]}
    stow = {}
    for dd in D_STOW:
        o = {}
        for name, tr in segs.items():
            pts = tr[::5] + [tr[-1]]
            fails, worst = [], None
            for i, qa in enumerate(pts):
                qx = np.zeros(9)
                qx[:3], qx[C.IDX] = base, qa
                ok, info = K1.evaluate(qx, qr + math.radians(dd), C.FINGER_OPEN)
                if not ok:
                    fails.append({'i': i, 'why': info['why'], 'env_who': info['env_who'], 'self_who': info['self_who'], 'base_who': info['base_who'],
                                  'env': info['env'], 'self_clear': info['self_clear']})
                m = {k: info[k] for k in ('joint_margin', 'env', 'base_clear', 'self_clear')}
                worst = m if worst is None else {k: min(worst[k], m[k]) for k in m}
            o[name] = {'n_points': len(pts), 'all_ok': not fails, 'fails': fails[:3], 'min': worst, 'duration_s': (len(tr) - 1) * V2.UNFOLD['dt']}
        stow[f'{dd:g}'] = o
    ok_d = [dd for dd in D_STOW if all(v['all_ok'] for v in stow[f'{dd:g}'].values())]
    res = {'schema': 'door_k3b_offline_v4c/1', 'source': 'door_k3b_offline_v4.json', 'q_rel_deg': math.degrees(qr),
           'held_pair_distance_m': held, 'band_fine': band, 'band_summary': summ, 's_fine': S_FINE, 'delta_fine_deg': list(D_FINE),
           'stow_target': 'K2_READY', 'K2_READY': K2_READY.tolist(), 'stow': stow, 'stow_ok_deltas_deg': ok_d,
           'thresholds': {'joint_m': C.LIM_M, 'env_m': C.ENV_M, 'base_m': C.BASE_M, 'self_m': C.SELF_M}, 'self_exemption': None}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('HELD 配對距離（mm）', {k: {n: round(v * 1e3, 3) for n, v in o.items()} for k, o in held.items() if k != 'q_finger_measured_m'})
    for x in summ:
        print(f"s {x['s']:<6g} 非手指≥5 {x['nonfinger_ge5_max']} (單調 {x['nonfinger_ge5_max_monotone']})  配對≥0 {x['pair_ge0_max']} "
              f"(單調 {x['pair_ge0_max_monotone']})  配對≥5 {x['pair_ge5_max']} (單調 {x['pair_ge5_max_monotone']})")
    for dd in D_STOW:
        o = stow[f'{dd:g}']
        print(f'收臂 Δ {dd:g}°', {k: (v['all_ok'], round(v['min']['env'] * 1e3, 2), round(v['min']['self_clear'] * 1e3, 2)) for k, v in o.items()},
              (o['end_to_qpre']['fails'][:1] or o['qpre_to_k2ready']['fails'][:1]))
    print('收臂可行 Δ', ok_d, '；寫出', OUT)


V4_EX, V4_EY = (-0.003, 0.0, 0.003), (-0.001, 0.0, 0.001)

if __name__ == '__main__':
    main()
