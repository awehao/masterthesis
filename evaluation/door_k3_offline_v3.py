#!/usr/bin/env python3
"""K3 前置離線核對 v3：解除後「夾爪—門」淨空的**狀態範圍**圖（依 Codex 20261021_181050 審查）。
v2（door_k3_offline_v2*.json）只核名目 8° 工具姿態＋全開手指，改定位為名目診斷表；本檔把下列誤差納入：
  - q_rel ∈ {7.5, 8.0, 8.5}°（釋放條件 |q − θ_goal| ≤ 0.5°）；
  - 工具相對把手的偏移 e（把手座標：x 沿橫桿、y 離開門扇法向、z 鉛直＝夾合方向）與小轉角 ρ；
    K2 door_k2_1 實錄（本檔 k2_envelope() 重算）：夾持時 (−1.12, −0.91, +0.12) mm；張爪瞬間 z 跳 ≈ −2.3 mm；
    退離 s < 0.01 時 |z| ≤ 3.42、s ∈ [0.01, 0.02) 時 ≤ 5.48、全程 ≤ 6.21 mm；|x| ≤ 2.27 mm；轉角 ≤ 0.33°；
  - 手指開度 f ∈ {0.95, 1.0}（K3 提議解除確認門檻 ≥ 95%）；
  - 門角 q_rel + Δ，Δ ∈ [−1, +5]°（門獨立轉、手臂不跟門），張爪後手指—橫桿不豁免。
夾爪剛體群（link6、link_eef、uflite_gripper_link、兩指）由名目 8° 解的點雲轉到工具座標，再依擾動後工具位姿重放；
link_base～link5 用名目解另報（偏移數 mm 不影響其數十 mm 的淨空）。底盤不受影響（見 v2）。
近似：點雲／膠囊體（橫桿）／盒，取樣間未核——只支援規格選值，不是碰撞安全證明。

    python3 evaluation/door_k3_offline_v3.py    # 輸出 results/door/door_k3_offline_v3.json（已存在則拒絕覆寫）
"""
import hashlib
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_offline_v2 as V2                       # noqa: E402
import door_geom_v2 as G2                             # noqa: E402

C, K1 = V2.C, V2.K1
OUT = os.path.join(HERE, 'results', 'door', 'door_k3_offline_v3.json')
THETA = 8.0
Q_REL = (7.5, 8.0, 8.5)
S_GRID = [round(x, 3) for x in np.linspace(0.0, 0.10, 11)]
D_GRID = (-1.0, -0.5, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.75, 3.5, 4.0, 4.75, 5.0)
EX = (-0.003, 0.0, 0.003)
EY = (-0.001, 0.0, 0.001)
EZ = tuple(round(k * 0.001, 3) for k in range(-7, 8))
RHO = [(None, 0.0)] + [(ax, sg * math.radians(0.5)) for ax in range(3) for sg in (-1, 1)]
FRAC = (0.95, 1.0)
RIGID = ('link6', 'link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')
OTHER = ('link_base', 'link1', 'link2', 'link3', 'link4', 'link5')


def rot(ax, a):
    c, s = math.cos(a), math.sin(a)
    R = np.eye(3)
    i, j = [(1, 2), (2, 0), (0, 1)][ax]
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def H_of(qdeg):
    """把手位姿＝工具名目目標 × T_HG⁻¹（與 C.target 同一來源）。"""
    return C.target(math.radians(qdeg), 0.0) @ np.linalg.inv(C.T_HG())


def k2_envelope():
    import door_k2_check as KC
    t = json.load(open(os.path.join(HERE, 'runs', 'door_k2_1', 'task.json')))
    r = json.load(open(os.path.join(HERE, 'runs', 'door_k2_1', 'room_run.json')))
    I = {c: i for i, c in enumerate(r['steps_cols'])}
    THG, Hf = np.array(t['grasp_basis']['T_HG']), np.array(t['H_frozen'])
    t_rel = next(e['t'] for e in t['policy_events'] if e.get('to') == 'RELEASE')
    bins = {}
    for s in r['steps']:
        if s[I['sim_t']] < t_rel:
            continue
        E = np.linalg.inv(Hf) @ KC._tcp(s[I['base_xyth']], s[I['q_arm_meas']])
        d = E[:3, 3] - THG[:3, 3]
        R = E[:3, :3] @ THG[:3, :3].T
        b = f'{math.floor(max(-d[1], 0.0) * 100) / 100:.2f}'
        o = bins.setdefault(b, {'n': 0, 'x_mm': 0.0, 'z_min_mm': 0.0, 'z_max_mm': 0.0, 'rot_deg': 0.0})
        o['n'] += 1
        o['x_mm'] = max(o['x_mm'], abs(d[0]) * 1e3)
        o['z_min_mm'] = min(o['z_min_mm'], d[2] * 1e3)
        o['z_max_mm'] = max(o['z_max_mm'], d[2] * 1e3)
        o['rot_deg'] = max(o['rot_deg'], math.degrees(math.acos(min(1.0, (np.trace(R) - 1) / 2))))
    return {'source': 'runs/door_k2_1（門 0°、K2 退離 0.020 m/s）', 'by_s_bin': bins}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    base = V2.base_of(0.0, 0.0, 0.0)
    q2 = json.load(open(os.path.join(HERE, 'results', 'door', 'door_k2_offline_v2.json')))['Q_PRE_DOOR']
    prev = C.ik(base, C.target(0.0, 0.0), np.array(q2))[C.IDX]
    for qd in np.arange(0.1, THETA + 1e-9, 0.1):
        prev = C.ik(base, C.target(math.radians(qd), 0.0), prev, only_prev=True)[C.IDX]
    qf8 = C.ik(base, C.target(math.radians(THETA), 0.0), prev, only_prev=True)
    T8 = C.target(math.radians(THETA), 0.0)
    H8 = H_of(THETA)
    # 夾爪剛體群：工具座標下的點雲（每個開度一份）
    local = {}
    for f in FRAC:
        W = C.world(qf8, RIGID, C.FINGER_OPEN * f)
        local[f] = np.vstack([(np.linalg.inv(T8) @ np.c_[P, np.ones(len(P))].T).T[:, :3] for P in W.values()])
    owner = np.concatenate([[n] * len(P) for n, P in C.world(qf8, RIGID, C.FINGER_OPEN).items()])
    N = {s: np.linalg.inv(H8) @ C.target(math.radians(THETA), s) for s in S_GRID}     # 名目工具位姿（把手座標）
    # 其他連桿（名目解）：與門的最小淨空
    other = {}
    for s in (0.0, 0.05, 0.10):
        qfs = C.ik(base, C.target(math.radians(THETA), s), prev, only_prev=True)
        Wo = C.world(qfs, OTHER, 0.0)
        for dd in (0.0, 5.0):
            sh = G2.shapes(K1.D2, math.radians(THETA + dd), K1.H)
            other[f's{s}_d{dd}'] = min(min(float(G2.box_dist(P, c, R, hs).min()) for _, c, R, hs in sh) for P in Wo.values())
    res = {'schema': 'door_k3_offline/3', 'theta_goal_deg': THETA, 'q_rel_deg': Q_REL, 's_grid': S_GRID, 'delta_grid_deg': D_GRID,
           'ex_m': EX, 'ey_m': EY, 'ez_m': EZ, 'rho': [[a, r] for a, r in RHO], 'finger_frac': FRAC,
           'spec_sha256': hashlib.sha256(open(K1.SPEC_P, 'rb').read()).hexdigest(),
           'k2_envelope': k2_envelope(), 'other_links_min_env_m': other,
           'note': 'env[q_rel][s][Δ][f][ez] = 對 ex×ey×ρ 取最差的夾爪群—門扇／立柱／牆／橫桿最小距離（m）'}
    tab = {}
    for qr in Q_REL:
        Hq = H_of(qr)
        for s in S_GRID:
            poses = []
            for ez in EZ:
                for ex, ey, (ax, ra) in itertools.product(EX, EY, RHO):
                    P = N[s].copy()
                    if ax is not None:
                        P[:3, :3] = P[:3, :3] @ rot(ax, ra)
                    P[:3, 3] += (ex, ey, ez)
                    poses.append(Hq @ P)
            TT = np.array(poses)                                              # ez 為外層：reshape(len(EZ), -1) 對齊
            for f in FRAC:
                for dd in D_GRID:
                    qd = math.radians(qr + dd)
                    sh = G2.shapes(K1.D2, qd, K1.H)
                    X = np.einsum('pij,nj->pni', TT[:, :3, :3], local[f]) + TT[:, None, :3, 3]   # (姿態, 點, 3)
                    flat = X.reshape(-1, 3)
                    best = np.full(flat.shape[0], np.inf)
                    lab = np.zeros(flat.shape[0], int)
                    for j, (name, c, R, hs) in enumerate(sh):
                        d = G2.box_dist(flat, c, R, hs)
                        m = d < best
                        best[m], lab[m] = d[m], j
                    d = G2.bar_capsule_dist(K1.D2, flat, qd, K1.H)
                    m = d < best
                    best[m], lab[m] = d[m], len(sh)
                    best, lab = best.reshape(len(EZ), -1), lab.reshape(len(EZ), -1)
                    names = [x[0] for x in sh] + ['bar']
                    npt = local[f].shape[0]
                    row = []
                    for i in range(len(EZ)):
                        k = int(best[i].argmin())
                        row.append([float(best[i, k]), f'{owner[k % npt]}|{names[lab[i, k]]}'])
                    tab.setdefault(f'{qr:g}', {}).setdefault(f'{s:g}', {}).setdefault(f'{f:g}', {})[f'{dd:g}'] = row
            print(f'q_rel {qr:g} s {s:g}：s 名目（ez=0）f=0.95 Δ=0 淨空 {tab[f"{qr:g}"][f"{s:g}"]["0.95"]["0"][7][0] * 1e3:.2f} mm', flush=True)
    res['env'] = tab
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('寫出', OUT)


if __name__ == '__main__':
    main()
