#!/usr/bin/env python3
"""K3-B 離線幾何核對 v4（Codex 20261025_225555：K3-B 判準規格＋既有資料／離線幾何核對；只離線、不模擬）。
door_k3_offline_v3.py（8°、K2 固定停位、名目解）不改；v4 改以 **door_wide_w2c_30_1 的實測 HELD 組態** 為起點：
  - 底盤＝HELD 實測位姿（解除與退離期間底盤保持不中斷 ⇒ 底盤固定在此位姿）；
  - 手臂＝HELD 實測關節；工具相對把手＝實測 E＝H(q)⁻¹·TCP（不是名目 T_HG）。
一、退離路徑（全連桿，K1.evaluate：關節餘裕、夾爪／殼／相機架對環境、底盤、自碰）
  - 沿接近軸反向（a_H，把手座標）由 s＝0 退到 0.10 m，每 0.005 m 一點；
  - IK 由實測關節起逐點承接（only_prev）；
  - 門角 q_rel＋Δ，Δ ∈ {−1, 0, +0.5, +1, +2, +3, +5}°（門獨立續轉、手臂不跟門）；
  - 手指開度 f ∈ {0.95, 1.0}；張爪後手指—橫桿不豁免（K1.evaluate 的 env 含橫桿）。
二、夾爪剛體群擾動表（v3 同法）：在實測偏移外再加 ex ±3 mm、ey ±1 mm、ez 依 v3 Z(s) 登錄（4／6／7 mm）、轉角 ±0.5°，
    對門扇／立柱／牆／橫桿取最差。Z(s) 來自 K2 0°、5 N 單趟；30°、12 N 預載下是否相同未知，明列。
三、收臂前置（s＝0.10 之後）：由退離終點以展開相位同一律反向走回 STOW（V2.unfold_traj 反序），
    每 0.1 s 一點；門角 q_rel 與 q_rel＋5°；手指全開。
近似：點雲／膠囊體／盒，取樣點間未核——只支援規格選值，不是碰撞安全證明。

    python3 evaluation/door_k3b_offline_v4.py    # 輸出 results/door/door_k3b_offline_v4.json（已存在則拒絕覆寫）
"""
from __future__ import annotations

import hashlib
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
RUN = 'door_wide_w2c_30_1'
OUT = os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4.json')
S_PATH = [round(x, 3) for x in np.arange(0.0, 0.1001, 0.005)]
D_PATH = (-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0)
FRAC = (0.95, 1.0)
S_RIG = [round(x, 3) for x in np.linspace(0.0, 0.10, 11)]
D_RIG = (-1.0, -0.5, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
EX, EY = (-0.003, 0.0, 0.003), (-0.001, 0.0, 0.001)
Z_OF = (lambda s: 0.004 if s < 0.01 else 0.006 if s < 0.02 else 0.007)        # v3 登錄 Z(s)（K2 0°、5 N）
RHO = V3.RHO


def held_state():
    r = json.load(open(os.path.join(HERE, 'runs', RUN, 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    s_ = json.load(open(os.path.join(HERE, 'runs', RUN, 'align_solver.json')))
    t_held = s_['stats']['k3m_pause']['supervisor_outcome']['transitions']['HELD']
    row = min(r['steps'], key=lambda s: abs(s[I['sim_t']] - t_held))
    return {'t': row[I['sim_t']], 'base': [float(x) for x in row[I['base_xyth']]], 'q_arm': [float(x) for x in row[I['q_arm_meas']]],
            'q_door': float(row[I['door_q_rad']])}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    hs = held_state()
    base = np.array(hs['base'])
    qr = hs['q_door']
    H = KC._H_of(qr)
    TCP = KC._tcp(hs['base'], hs['q_arm'])
    E = np.linalg.inv(H) @ TCP                                                    # 實測工具相對把手
    n0 = np.linalg.inv(V3.H_of(math.degrees(qr))) @ C.target(qr, 0.0)
    n1 = np.linalg.inv(V3.H_of(math.degrees(qr))) @ C.target(qr, 0.01)
    a_H = (n1[:3, 3] - n0[:3, 3]) / 0.01                                          # 接近軸（把手座標，退離為 +s）
    tool_in_H = {s: np.block([[np.eye(3), (s * a_H)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]) @ E for s in S_PATH}
    # 起點一致性：實測關節對實測工具位姿的 IK 殘差
    q0 = np.zeros(9)
    q0[:3], q0[C.IDX] = base, hs['q_arm']
    qf_ik = C.ik(base, H @ tool_in_H[0.0], np.array(hs['q_arm']), only_prev=True)
    ik0_dev = None if qf_ik is None else float(np.abs(qf_ik[C.IDX] - np.array(hs['q_arm'])).max())
    # 一、退離路徑
    path = {}
    prev = np.array(hs['q_arm'])
    sols = {}
    for s in S_PATH:
        qf = C.ik(base, H @ tool_in_H[s], prev, only_prev=True)
        sols[s] = qf
        if qf is not None:
            prev = qf[C.IDX]
    for dd, f in itertools.product(D_PATH, FRAC):
        rows = []
        for s in S_PATH:
            qf = sols[s]
            if qf is None:
                rows.append({'s': s, 'ok': False, 'why': 'ik_none'})
                continue
            ok, info = K1.evaluate(qf, qr + math.radians(dd), C.FINGER_OPEN * f)
            rows.append({'s': s, 'ok': ok, **{k: info.get(k) for k in ('joint_margin', 'joint', 'env', 'env_who', 'shell_min', 'camera_stand_min',
                                                                       'base_clear', 'self_clear', 'self_who', 'why')}})
        path[f'd{dd:g}_f{f:g}'] = rows
    joint_steps = []
    for a_, b_ in zip(S_PATH, S_PATH[1:]):
        if sols[a_] is not None and sols[b_] is not None:
            d = np.abs(sols[b_][C.IDX] - sols[a_][C.IDX])
            joint_steps.append({'s': [a_, b_], 'max_rad': float(d.max()), 'joint': int(d.argmax()) + 1})
    # 二、夾爪剛體群擾動表
    qf0 = sols[0.0] if sols[0.0] is not None else q0
    T0 = H @ tool_in_H[0.0]
    local = {}
    for f in FRAC:
        W = C.world(qf0, V3.RIGID, C.FINGER_OPEN * f)
        local[f] = np.vstack([(np.linalg.inv(T0) @ np.c_[P, np.ones(len(P))].T).T[:, :3] for P in W.values()])
    owner = np.concatenate([[n] * len(P) for n, P in C.world(qf0, V3.RIGID, C.FINGER_OPEN).items()])
    rig = {}
    for s in S_RIG:
        Z = Z_OF(s)
        ezs = (-Z, -Z / 2, 0.0, Z / 2, Z)
        poses = []
        for ez, ex, ey, (ax, ra) in itertools.product(ezs, EX, EY, RHO):
            P = tool_in_H[min(S_PATH, key=lambda x: abs(x - s))].copy()
            if ax is not None:
                P[:3, :3] = P[:3, :3] @ V3.rot(ax, ra)
            P[:3, 3] += (ex, ey, ez)
            poses.append(H @ P)
        TT = np.array(poses)
        for f in FRAC:
            for dd in D_RIG:
                qd = qr + math.radians(dd)
                sh = G2.shapes(K1.D2, qd, K1.H)
                X = np.einsum('pij,nj->pni', TT[:, :3, :3], local[f]) + TT[:, None, :3, 3]
                flat = X.reshape(-1, 3)
                best = np.full(flat.shape[0], np.inf)
                lab = np.zeros(flat.shape[0], int)
                for j, (name, c, R, hs_) in enumerate(sh):
                    d = G2.box_dist(flat, c, R, hs_)
                    m = d < best
                    best[m], lab[m] = d[m], j
                d = G2.bar_capsule_dist(K1.D2, flat, qd, K1.H)
                m = d < best
                best[m], lab[m] = d[m], len(sh)
                k = int(best.argmin())
                names = [x[0] for x in sh] + ['bar']
                npt = local[f].shape[0]
                rig.setdefault(f'{s:g}', {}).setdefault(f'{f:g}', {})[f'{dd:g}'] = [float(best[k]), f'{owner[k % npt]}|{names[lab[k]]}']
    # 三、收臂（退離終點 → STOW）
    q_end = sols[S_PATH[-1]]
    stow = {}
    if q_end is not None:
        traj = V2.unfold_traj(q_end[C.IDX])[::-1]
        for dd in (0.0, 5.0):
            worst, fails = None, []
            for i, qa in enumerate(traj[::5] + [traj[-1]]):
                qf = np.zeros(9)
                qf[:3], qf[C.IDX] = base, qa
                ok, info = K1.evaluate(qf, qr + math.radians(dd), C.FINGER_OPEN)
                if not ok:
                    fails.append({'i': i, 'why': info['why'], 'env_who': info['env_who'], 'self_who': info['self_who'], 'base_who': info['base_who']})
                m = {k: info[k] for k in ('joint_margin', 'env', 'base_clear', 'self_clear')}
                worst = m if worst is None else {k: min(worst[k], m[k]) for k in m}
            stow[f'd{dd:g}'] = {'n_points': len(traj[::5]) + 1, 'all_ok': not fails, 'fails': fails[:5], 'min': worst}
    res = {'schema': 'door_k3b_offline/4', 'source_run': RUN, 'held_state': hs, 'q_rel_deg': math.degrees(qr),
           'tool_in_handle_measured': E.tolist(), 'a_H_handle': a_H.tolist(), 'ik_start_dev_rad': ik0_dev,
           's_path': S_PATH, 'delta_path_deg': list(D_PATH), 'finger_frac': list(FRAC), 'path': path, 'path_joint_steps': joint_steps,
           'rigid_perturbed': rig, 'rigid_grid': {'s': S_RIG, 'delta_deg': list(D_RIG), 'ex_m': EX, 'ey_m': EY, 'Z_of_s_m': 'v3 登錄 4／6／7 mm',
                                                  'rho': [[a, r] for a, r in RHO]},
           'stow': stow, 'spec_sha256': hashlib.sha256(open(K1.SPEC_P, 'rb').read()).hexdigest(),
           'note': 'path[d<Δ>_f<f>]＝K1.evaluate 逐點；rigid_perturbed[s][f][Δ]＝[最小距離 m, 部件|對象]；底盤固定於 HELD 實測位姿'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('IK 起點殘差', ik0_dev, ' a_H', np.round(a_H, 4).tolist())
    for key in ('d0_f0.95', 'd1_f0.95', 'd2_f0.95', 'd5_f0.95'):
        rows = path[key]
        print(key, 'ok', all(r['ok'] for r in rows), ' env 最小', min((r.get('env') or 9) for r in rows), ' 首個不通過',
              next(((r['s'], r.get('why'), r.get('env_who')) for r in rows if not r['ok']), None))
    for s in ('0', '0.01', '0.02', '0.05', '0.1'):
        print('剛體擾動 s', s, {d: round(v[0] * 1e3, 2) for d, v in rig[s]['0.95'].items() if d in ('0', '0.5', '1', '2', '5')})
    print('收臂', {k: (v['all_ok'], v['min']) for k, v in stow.items()})
    print('寫出', OUT)


if __name__ == '__main__':
    main()
