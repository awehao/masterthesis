#!/usr/bin/env python3
"""連續協同通行：繞緣夾法抓取契約＋起點接近快速可行性 v1（Codex 20261026_183611 §3-1；離線靜態，無 Isaac、無接觸力）。

主候選：繞緣夾法 φ 60°、s 0.85、roll 0（door_coopen_edgearc_v1 q45 種子）。對照：實跑過的正面夾法 φ 0、s 0.80、roll 0（K2 停位）。
兩者以同一套點雲（door_offline_check_v1 CL、FINGER_AXIS）、門幾何（door_geom_v2）、幾何監看（door_k3m_geom_monitor）計算：

1. 開指接近（＝張爪後沿原軸退出，靜態幾何相同）：夾爪全開（finger_joint 0.0089），TCP 沿工具 −z 自抓取位姿後退 d ∈ {60, 50, …, 0} mm，
   底盤固定、IK 以前一點為種子；每點報 nonfinger（含夾爪殼—橫桿）、finger（不含手指—橫桿設計接觸）、self、base，
   以及兩指各自對橫桿的距離（door_k3b_pair_geom 同一算法）——接近途中手指不得先碰桿（> 0）。
2. 閉合：抓取位姿下 finger_joint 由 0.0089 每 0.0001 收到 0，報兩指各自首次接觸（對橫桿距離 ≤ 0）時的開度、兩指首觸開度差，
   首觸時接觸法向（手指點雲最近桿軸點 → 該點的徑向）與兩法向夾角（理想對向＝180°），以及法向與門扇法向 n 的夾角。
3. 門緣干涉：抓取位姿下全開與首觸開度時，nonfinger／finger 對門扇、立柱、牆的最小距離與最近物件。
4. 起點接近（H0 快速可行性）：手臂由 K2_READY（STOW）以關節線性內插 40 點展開到預抓取（d＝60 mm）解，底盤固定於起點，
   每點核 nonfinger／finger／self／base；底盤自 (x, y−0.60) 沿 +y 直線到起點（門 0°）的圓盤淨空。
門角：繞緣夾法於 0°（起點組態）與 46°（放手組態）各做一次；正面夾法於 0°。
**界線**：靜態幾何；不含接觸力、摩擦、夾持穩定性（力封閉）與柔順；關節內插不是規劃器軌跡；手指以點雲＋線性開合近似。

    python3 evaluation/door_coopen_grasp_contract_v1.py   # 寫 results/door/door_coopen_grasp_contract_v1.json（不覆寫）
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
import door_geom_v2 as G2                              # noqa: E402
import door_k2_check as KC                             # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_grasp_contract_v1.json')
C, V2 = RV.C, RV.V2
K1 = V2.K1
OPEN = C.FINGER_OPEN
DS_MM = (60, 50, 40, 30, 20, 15, 10, 5, 0)


def pair_and_normals(qf, q, qfi):
    """兩指對橫桿：距離、最近點徑向法向。"""
    T_H = G2.handle_pose(K1.D2, q, K1.H)
    u, c = T_H[:3, 0], T_H[:3, 3]
    L = K1.D2['handle']['bar']['length'] / 2
    out = []
    for n in C.FINGERS:
        T = C.K.fk(np.asarray(qf, float), n)
        P = C.CL[n] + C.FINGER_AXIS[n] * qfi
        W = (T[:3, :3] @ P.T).T + T[:3, 3]
        d = G2.bar_capsule_dist(K1.D2, W, q, K1.H)
        k = int(np.argmin(d))
        s = float(np.clip((W[k] - c) @ u, -L, L))
        a = c + s * u
        r = W[k] - a
        out.append({'d': float(d[k]), 'normal': (r / (np.linalg.norm(r) + 1e-12)).tolist(), 's_on_bar': s})
    return out


def geo(qf, q, qfi):
    d = TJ.GEO([float(v) for v in qf], q, qfi)
    return {k: [float(d[k][0]), d[k][1]] for k in ('nonfinger', 'finger', 'base', 'self')}


def approach(base, T, seed, q):
    rows, prev = [], seed
    for dmm in DS_MM:
        Td = T @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -dmm / 1000.0], [0, 0, 0, 1.0]])
        qf9 = C.ik(np.asarray(base, float), Td, prev, only_prev=prev is not None)
        if qf9 is None:
            qf9 = C.ik(np.asarray(base, float), Td, prev)
        if qf9 is None:
            rows.append({'d_mm': dmm, 'ik': False})
            continue
        qf = [float(v) for v in base] + [float(v) for v in np.asarray(qf9)[C.IDX]]
        prev = np.asarray(qf9)[C.IDX]
        rows.append({'d_mm': dmm, 'ik': True, 'qf': qf, 'geo': geo(qf, q, OPEN), 'pair': [p['d'] for p in pair_and_normals(qf, q, OPEN)]})
    return rows


def closure(qf, q, n_door):
    first = [None, None]
    for k in range(int(round(OPEN / 0.0001)), -1, -1):
        qfi = k * 0.0001
        pn = pair_and_normals(qf, q, qfi)
        for i in range(2):
            if first[i] is None and pn[i]['d'] <= 0.0:
                first[i] = {'q_finger': qfi, 'normal': pn[i]['normal'], 's_on_bar': pn[i]['s_on_bar']}
    rec = {'first_contact': first}
    if all(first):
        n1, n2 = np.array(first[0]['normal']), np.array(first[1]['normal'])
        rec['q_finger_diff'] = abs(first[0]['q_finger'] - first[1]['q_finger'])
        rec['normal_angle_deg'] = math.degrees(math.acos(float(np.clip(n1 @ n2, -1, 1))))
        rec['normal_vs_door_n_deg'] = [math.degrees(math.acos(float(np.clip(abs(np.array(f['normal']) @ n_door), -1, 1)))) for f in first]
        qc = max(first[0]['q_finger'], first[1]['q_finger'])
        rec['geo_at_first_contact'] = geo(qf, q, qc)
    return rec


def h0(base, qa_pre, q):
    rows = []
    for t in np.linspace(0.0, 1.0, 40):
        qa = (1 - t) * TJ.STOW + t * np.asarray(qa_pre)
        rows.append(geo(list(base) + list(qa), q, OPEN))
    mins = {k: min(r[k][0] for r in rows) for k in ('nonfinger', 'finger', 'base', 'self')}
    who = {k: min(rows, key=lambda r: r[k][0])[k][1] for k in mins}
    path = [min(float(G2.disc_clear(base[0], float(y), c[2], C.RB, c, R, hs)) for _, c, R, hs in G2.shapes(K1.D2, q, K1.H))
            for y in np.linspace(base[1] - 0.60, base[1], 31)]
    return {'unfold_min': mins, 'unfold_min_who': who, 'base_approach_min_m': min(path), 'stow': TJ.STOW.tolist()}


def case(name, base, qa, T, q):
    TJ.cool()
    qf = [float(v) for v in base] + [float(v) for v in qa]
    n_door = np.array([-math.sin(q), -math.cos(q), 0.0])
    ap = approach(base, T, np.asarray(qa), q)
    cl = closure(qf, q, n_door)
    rec = {'case': name, 'q_deg': math.degrees(q), 'base': [float(v) for v in base], 'qa': [float(v) for v in qa],
           'grasp_open': geo(qf, q, OPEN), 'pair_open': [p['d'] for p in pair_and_normals(qf, q, OPEN)],
           'approach': ap, 'closure': cl}
    ok = [r for r in ap if r['ik']]
    rec['approach_summary'] = {'n_ik_fail': sum(1 for r in ap if not r['ik']),
                               'min': {k: min(r['geo'][k][0] for r in ok) for k in ('nonfinger', 'finger', 'base', 'self')} if ok else None,
                               'pair_min_before_contact_m': min(min(r['pair']) for r in ok if r['d_mm'] > 0) if ok else None}
    pre = next((r for r in ap if r['d_mm'] == 60 and r['ik']), None)
    rec['h0'] = h0(base, pre['qf'][3:], q) if (pre and abs(q) < 1e-9) else None
    fc = cl['first_contact']
    print(f"{name}：接近 IK 失敗 {rec['approach_summary']['n_ik_fail']}，接近最小 {({k: round(v, 4) for k, v in (rec['approach_summary']['min'] or {}).items()})}，"
          f"接觸前指—桿最小 {rec['approach_summary']['pair_min_before_contact_m']}；首觸開度 {[None if f is None else f['q_finger'] for f in fc]}，"
          f"法向夾角 {cl.get('normal_angle_deg')}，對門法向 {cl.get('normal_vs_door_n_deg')}；"
          f"全開門緣 {({k: (round(v[0], 4), v[1]) for k, v in rec['grasp_open'].items() if k in ('nonfinger', 'finger')})}"
          + ('' if rec['h0'] is None else f"；H0 展開最小 {({k: round(v, 4) for k, v in rec['h0']['unfold_min'].items()})}（{rec['h0']['unfold_min_who']}）、底盤接近 {rec['h0']['base_approach_min_m']:.3f}"),
          flush=True)
    return rec


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    ea = json.load(open(os.path.join(HERE, 'results', 'door', 'door_coopen_edgearc_v1.json')))
    s = next(x for x in ea['per_seed'] if x['seed_q_deg'] == 45 and x['seed']['roll'] == 0)
    tr = s['best']['trajectory']
    sd = s['seed']
    res = []
    for p in (tr[0], tr[-1]):
        q = math.radians(p['q_deg'])
        res.append(case(f"edge_phi{sd['phi']}_q{p['q_deg']:.0f}", p['base'], p['qa'], EG.grasp_T(q, sd['phi'], sd['s'], sd['roll']), q))
    sw = json.load(open(os.path.join(HERE, 'results', 'door', 'door_wide_offline_sweep_90.json')))
    q0 = np.array(sw['trajectory'][0]['q'], float)
    r0 = TJ.check(q0[:3], 0.0, KC._H_of(0.0) @ TJ.THG, q0[3:])
    res.append(case('front_phi0_q0', q0[:3], r0['qa'], KC._H_of(0.0) @ TJ.THG, 0.0))
    json.dump({'schema': 'door_coopen_grasp_contract/1', 'finger_open': OPEN, 'approach_d_mm': DS_MM, 'cases': res,
               'not_modeled': ['接觸力、摩擦、力封閉與夾持穩定性', '柔順與滑移', '關節內插非規劃器軌跡', '膠囊近似低估端面外間距（保守）']},
              open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
