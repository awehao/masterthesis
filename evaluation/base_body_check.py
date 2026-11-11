#!/usr/bin/env python3
"""手臂對底盤車體（含光達座）的碰撞核對（Codex 20261111_054341 第 2 項：物理前補齊）。純離線、與底盤位姿無關（全在底盤座標）。
車體＝URDF base_link 碰撞圓柱（r 0.300、z 0.05–0.33）的點雲；光達（lidar.xacro：r 0.036、長 0.044，裝在 base_link z 0.2152）完全在其內 ⇒ 以車體點雲保守涵蓋。
手臂連桿：link2–link6、link_eef、夾爪殼、兩指（link_base 固定於車頂、link1 在其上旋轉，與車體設計接觸，排除）。
核對對象：
  1 觀看視角生成器輸出族：收攏 → V_HIGH 腕部樣板＋q1（−1.5…1.5，13 點）的 min-jerk 路徑（21 取樣）；
  2 TPA 整段軌跡（左側 nom 物件；起點＝實錄 8.80 s U 中、11.40 s H、合成 K2_READY；終點＝x_pre 的預抓分支）每 0.1 s 取樣；
  3 收攏姿態、V_HIGH 本身。門檻 ≥ 0.02 m（同對桌）。
    python3 evaluation/base_body_check.py OUT.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_plan_worker as PW  # noqa: E402
import bottle_r0_approach as R0  # noqa: E402
import bottle_ref as BR  # noqa: E402
import bottle_s0_ctrl as C0  # noqa: E402
import bottle_w1_tpa_ctrl as TPA  # noqa: E402
import tpa_offline_check as TO  # noqa: E402
from ammr_wholebody_mpc.arm_limits import LITE6_SAFE as L  # noqa: E402

LINKS = ('link2', 'link3', 'link4', 'link5', 'link6', 'link_eef', 'uflite_gripper_link', 'uflite_finger1', 'uflite_finger2')
TH = 0.02


def base_dist(qa):
    q9 = np.r_[0.0, 0.0, 0.0, qa]
    Tb = R0.K.fk(q9, 'base_link')
    Pb = (Tb[:3, :3] @ R0.CL['base_link'].T).T + Tb[:3, 3]
    best = (np.inf, None)
    for n in LINKS:
        T = R0.K.fk(q9, n)
        P = (T[:3, :3] @ R0.CL[n].T).T + T[:3, 3]
        d = float(np.min(np.linalg.norm(P[:, None, :] - Pb[None, ::2, :], axis=2)))
        if d < best[0]:
            best = (d, n)
    return best


def main():
    res = dict(threshold=TH, views=[], tpa=[], poses={})
    for nm, qa in (('收攏 K2_READY', R0.Q_TUCK), ('V_HIGH', R0.Q_VHIGH)):
        d = base_dist(qa)
        res['poses'][nm] = (round(d[0], 4), d[1])
    for q1 in np.linspace(-1.5, 1.5, 13):
        qv = R0.Q_VHIGH.copy()
        qv[0] = q1
        w = (np.inf, None)
        for s in np.linspace(0, 1, 21):
            m = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
            w = min(w, base_dist(R0.Q_TUCK + (qv - R0.Q_TUCK) * m), key=lambda z: z[0])
        res['views'].append(dict(q1=round(float(q1), 3), min=round(w[0], 4), link=w[1], ok=bool(w[0] >= TH)))
    r, c, O, fr, tl = TO.load()
    xb, yb = fr['xy']
    zs = fr['z_support']
    yl = float(tl['y_lane'])
    T_g = C0.grasp_target(xb, yb, zs + 0.10)
    T_pre = BR.offset_along_tool(T_g, -0.06)
    new = r['relwin']['new']
    x_pre, x_slow = float(new['x_pre']), float(new['x_slow'])
    starts = []
    for nm, t_ in (('實錄 8.80 s', 8.80), ('實錄 11.40 s', 11.40)):
        q, qd, base = TO.rec_state(c, O, t_)
        starts.append((nm, q, qd, base))
    starts.append(('合成 K2_READY', R0.Q_TUCK.copy(), np.zeros(6), TO.rec_state(c, O, 8.80)[2]))
    for nm, q0, qd0, base0 in starts:
        inp = dict(xy=[xb, yb], z_support=zs, band=0.10, pre_m=0.06, lift_m=0.05, win_rel=[-0.35, 0.15], x_nom=0.0, y_lane=yl,
                   q_seeds=[s.tolist() for s in [q0] + list(R0.RNG_SEEDS)], q_now=q0.tolist(), q_seed=None)
        ok, why, rs = PW.plan_grasp(inp)
        Rr, why2 = TPA.plan_tpa(R0.K, q0, qd0, np.array([x_pre, yl, 0.0]), T_pre, np.asarray(rs['q_pre_lo'], float), float(base0[0]), x_pre, x_slow, 0.35, 0.10, 0.35,
                                R0.IDX, L.lower, L.upper)
        P = Rr['P']
        w = (np.inf, None)
        for t in np.arange(0.0, P['T'] + 1e-9, 0.1):
            w = min(w, base_dist(TPA.traj_eval(P, t)), key=lambda z: z[0])
        res['tpa'].append(dict(start=nm, T=round(P['T'], 2), min=round(w[0], 4), link=w[1], ok=bool(w[0] >= TH)))
    res['all_ok'] = all(v['ok'] for v in res['views'] + res['tpa']) and all(v[0] >= TH for v in res['poses'].values())
    json.dump(res, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
    print('姿態', res['poses'])
    print('視角', [(v['q1'], v['min'], v['link']) for v in res['views']])
    print('TPA', res['tpa'])
    print('全部通過', res['all_ok'])


if __name__ == '__main__':
    main()
