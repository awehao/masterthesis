#!/usr/bin/env python3
"""K3-B 收臂路線核對 v4b（door_k3b_offline_v4.py 的補充；v4 輸出不改）。
v4 第三項（退離終點直接以展開律反向回 STOW）在 STOW 附近 self 2.8 mm 不通過。
  - 這是 STOW 姿態本身的自碰淨空（K2 offline v2 的展開起點同為 2.8 mm，unfold_all_ok＝0 已記），不是路線新問題。
本檔核對路線「退離終點 → Q_PRE_DOOR（展開律同一律）→ STOW（已驗證展開路徑反序）」，
  - 門角 q_rel＋Δ，Δ ∈ 0～5°（每 0.5°）；
  - 逐點 K1.evaluate（手指全開）；
  - 報各 Δ 是否除 STOW 端點固有自碰外全通過、最差淨空與失敗部位。
  - 用於登錄「收臂前置：門角續轉 ≤ Δ_stow」。
近似同 v4，不是碰撞安全證明。

    python3 evaluation/door_k3b_offline_v4b_stow.py   # 輸出 results/door/door_k3b_offline_v4b_stow.json（已存在則拒絕覆寫）
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_k2_check as KC                             # noqa: E402
import door_k2_offline_v2 as V2                        # noqa: E402

C, K1 = V2.C, V2.K1
V4 = os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4b_stow.json')
D_GRID = [round(0.5 * k, 1) for k in range(11)]


def law(q0, qg):
    U = V2.UNFOLD
    q, tr = q0.copy(), [q0.copy()]
    for _ in range(20000):
        qd = np.clip(U['gain'] * (qg - q), -U['qd_max'], U['qd_max'])
        q = q + qd * U['dt']
        tr.append(q.copy())
        if float(np.abs(qg - q).max()) <= U['q_tol']:
            break
    return tr


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    v4 = json.load(open(V4))
    base, qr = np.array(v4['held_state']['base']), v4['held_state']['q_door']
    H, E, aH = KC._H_of(qr), np.array(v4['tool_in_handle_measured']), np.array(v4['a_H_handle'])
    prev = np.array(v4['held_state']['q_arm'])
    for s in v4['s_path']:
        T = np.eye(4)
        T[:3, 3] = aH * s
        prev = C.ik(base, H @ T @ E, prev, only_prev=True)[C.IDX]
    q_end = prev
    qpre = np.array(json.load(open(os.path.join(HERE, 'results', 'door', 'door_k2_offline_v2.json')))['Q_PRE_DOOR'])
    segs = {'end_to_qpre': law(q_end, qpre), 'qpre_to_stow': V2.unfold_traj(qpre)[::-1]}
    out = {}
    for dd in D_GRID:
        o = {}
        for name, tr in segs.items():
            pts = tr[::5] + [tr[-1]]
            fails, worst = [], None
            for i, qa in enumerate(pts):
                qf = np.zeros(9)
                qf[:3], qf[C.IDX] = base, qa
                ok, info = K1.evaluate(qf, qr + math.radians(dd), C.FINGER_OPEN)
                stow_end_self = (name == 'qpre_to_stow' and info['why'] == ['self'] and info['self_who'] == 'link1|link5'
                                 and float(np.abs(qa - V2.STOW).max()) <= 0.05)
                if not ok and not stow_end_self:
                    fails.append({'i': i, 'why': info['why'], 'env_who': info['env_who'], 'self_who': info['self_who'], 'env': info['env']})
                m = {k: info[k] for k in ('joint_margin', 'env', 'base_clear', 'self_clear')}
                worst = m if worst is None else {k: min(worst[k], m[k]) for k in m}
            o[name] = {'n_points': len(pts), 'ok_except_stow_inherent': not fails, 'fails': fails[:3], 'min': worst,
                       'duration_s': (len(tr) - 1) * V2.UNFOLD['dt']}
        out[f'{dd:g}'] = o
    ok_d = [dd for dd in D_GRID if all(v['ok_except_stow_inherent'] for v in out[f'{dd:g}'].values())]
    res = {'schema': 'door_k3b_offline_v4b/1', 'source': 'door_k3b_offline_v4.json', 'q_rel_deg': math.degrees(qr), 'q_end_arm': q_end.tolist(),
           'Q_PRE_DOOR': qpre.tolist(), 'by_delta_deg': out, 'max_delta_ok_deg': max(ok_d) if ok_d else None,
           'ok_deltas_deg': ok_d,
           'stow_inherent': 'STOW 姿態自碰淨空 link1|link5 2.8 mm（K2 offline v2 起點同值），STOW 端點 0.05 rad 內此項不計入失敗並另報'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for dd in D_GRID:
        o = out[f'{dd:g}']
        print(f'Δ {dd:g}°', {k: (v['ok_except_stow_inherent'], round(v['min']['env'] * 1e3, 2)) for k, v in o.items()},
              (o['end_to_qpre']['fails'][:1] or o['qpre_to_stow']['fails'][:1]))
    print('收臂可容許續轉上限（格點）', res['max_delta_ok_deg'], '；寫出', OUT)


if __name__ == '__main__':
    main()
