#!/usr/bin/env python3
"""K3-B 離線核對 v4d：門角帶的格點間覆蓋（Codex 20261026_010158：5 mm 追蹤落後須補「以 s_meas 查表」的幾何推導與格點間覆蓋）。
v4、v4b、v4c 輸出不改。

推導：
  - 線上帶以**實測** s_meas 查表（door_k3b_release2.band_upper_deg：取不大於 s_meas 的最大格點）。
    幾何前提只與「工具實際所在位置」與門角有關，與命令 s_cmd 無關 ⇒ 追蹤落後不改變帶的依據，只要帶在每個實際 s 都成立。
  - 本檔逐 0.5 mm 取 s（0～0.10 m，涵蓋 2.5 mm 與 10 mm 兩種格距內的所有中間點），
    以該 s 的查表上界 Δ＝band_upper_deg(s) 與下界 −1°，套同一剛體擾動驗證：
      ex ±3 mm、ey ±1 mm（對實測 s 的量測／模型誤差，不是格點量化）、ez ±Z(s)、轉角 ±0.5°、開度 95 %。
  - 準則：
    - 非手指（含夾爪殼—橫桿、手指對門扇／立柱／牆）≥ 5 mm；
    - 兩配對：s < 0.02（段 B，退出前）d ≥ 0，s ≥ 0.02（退出後）≥ 5 mm。
    - 任一點不成立 ⇒ 報告該點，帶需收緊（本檔不改帶，只報）。
  - 另報每點最差距離與部件，以及「若改用格點下界 s 查表」時的對照（同一函式，故相同）。
近似同 v4c，不是碰撞安全證明。

    python3 evaluation/door_k3b_offline_v4d.py   # 輸出 results/door/door_k3b_offline_v4d.json（已存在則拒絕覆寫）
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
import door_k3b_release2 as KR                         # noqa: E402

C, K1 = V2.C, V2.K1
V4 = os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4.json')
OUT = os.path.join(HERE, 'results', 'door', 'door_k3b_offline_v4d.json')
S_FINE = [round(0.0005 * k, 4) for k in range(201)]
FINGERS = ('uflite_finger1', 'uflite_finger2')
EX, EY = (-0.003, 0.0, 0.003), (-0.001, 0.0, 0.001)


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    v4 = json.load(open(V4))
    hs = v4['held_state']
    base, qr = np.array(hs['base']), hs['q_door']
    H, E = KC._H_of(qr), np.array(v4['tool_in_handle_measured'])
    qf = np.zeros(9)
    qf[:3], qf[C.IDX] = base, hs['q_arm']
    T0 = H @ E
    W = C.world(qf, V3.RIGID, C.FINGER_OPEN * 0.95)
    local = np.vstack([(np.linalg.inv(T0) @ np.c_[P, np.ones(len(P))].T).T[:, :3] for P in W.values()])
    owner = np.concatenate([[n] * len(P) for n, P in W.items()])
    is_f = np.isin(owner, FINGERS)
    rows, bad = [], []
    for s in S_FINE:
        Z = KR.z_max_m(s)
        Ts = np.eye(4)
        Ts[:3, 3] = KR.A_H * s
        tool = Ts @ E
        poses = []
        for ez, ex, ey, (ax, ra) in itertools.product((-Z, -Z / 2, 0.0, Z / 2, Z), EX, EY, V3.RHO):
            P = tool.copy()
            if ax is not None:
                P[:3, :3] = P[:3, :3] @ V3.rot(ax, ra)
            P[:3, 3] += (ex, ey, ez)
            poses.append(H @ P)
        TT = np.array(poses)
        X = np.einsum('pij,nj->pni', TT[:, :3, :3], local) + TT[:, None, :3, 3]
        flat = X.reshape(-1, 3)
        up = KR.band_upper_deg(s)
        rec = {'s': s, 'band_up_deg': up}
        for tag, dd in (('upper', up), ('lower', KR.REG['lower_deg'])):
            qd = qr + math.radians(dd)
            sh = G2.shapes(K1.D2, qd, K1.H)
            d_box = np.min(np.stack([G2.box_dist(flat, c, R, h) for _, c, R, h in sh]), axis=0).reshape(X.shape[:2])
            d_bar = G2.bar_capsule_dist(K1.D2, flat, qd, K1.H).reshape(X.shape[:2])
            pair = float(d_bar[:, is_f].min())
            nonf = float(min(d_box.min(), d_bar[:, ~is_f].min()))
            need_pair = 0.0 if s < KR.REG['near_s_m'] else KR.REG['pair_exit_m']
            ok = nonf >= KR.REG['nonfinger_m'] - 1e-9 and pair >= need_pair - 1e-9
            rec[tag] = {'delta_deg': dd, 'nonfinger_mm': nonf * 1e3, 'pair_mm': pair * 1e3, 'need_pair_mm': need_pair * 1e3, 'ok': ok}
            if not ok:
                bad.append((s, tag, round(nonf * 1e3, 3), round(pair * 1e3, 3)))
        rows.append(rec)
    res = {'schema': 'door_k3b_offline_v4d/1', 'source': 'door_k3b_offline_v4.json', 'q_rel_deg': math.degrees(qr), 's_step_m': 0.0005,
           'band': KR.BAND, 'rows': rows, 'n_bad': len(bad), 'bad': bad,
           'min_margin': {'nonfinger_mm': min(min(r['upper']['nonfinger_mm'], r['lower']['nonfinger_mm']) for r in rows) - 5.0,
                          'pair_near_mm': min(min(r['upper']['pair_mm'], r['lower']['pair_mm']) for r in rows if r['s'] < 0.02),
                          'pair_after_mm': min(min(r['upper']['pair_mm'], r['lower']['pair_mm']) for r in rows if r['s'] >= 0.02) - 5.0},
           'note': '每 0.5 mm 的 s 以線上同一查表函式取上界；剛體擾動取最差；不是碰撞安全證明'}
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('不成立點數', len(bad), bad[:10])
    print('最小餘量', res['min_margin'])
    worst = sorted(rows, key=lambda r: min(r['upper']['nonfinger_mm'] - 5.0, r['upper']['pair_mm'] - r['upper']['need_pair_mm']))[:6]
    for r in worst:
        print(f"s {r['s']:.4f} up {r['band_up_deg']}° 非手指 {r['upper']['nonfinger_mm']:.2f} 配對 {r['upper']['pair_mm']:.2f}（需 {r['upper']['need_pair_mm']:.0f}）")
    print('寫出', OUT)


if __name__ == '__main__':
    main()
