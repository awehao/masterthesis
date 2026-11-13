#!/usr/bin/env python3
"""r6 相對速度單元核對（Codex 20261113_130916 §2）：MovingGraspCtrl.rel_motion（把手參考—TCP 相對變換差分）對兩個合成情境，並與 r1–r5 的 TCP 世界位姿差分對照。
  S1 門以 0.2 rad/s 轉、夾爪剛性跟著把手（T_tcp＝T_H(φ)·T_HE）⇒ 相對應為 0
  S2 TCP 世界不動、門以 0.2 rad/s 轉 ⇒ 相對線速度應≈把手點繞鉸鏈的速度 r·ω
    python3 evaluation/door_motm_relvel_check.py OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_motm_grasp_ctrl as G
W, DT = G.W, G.DT
P = np.load(os.path.join(HERE, 'runs', 'm8house_door_full_2', 'plan.npz'))
T_WH0, phi0, T_HE = P['T_WH_bind'], float(P['phi_bind']), P['T_HE']
Hof = lambda phi: W.Rz4(-(phi - phi0)) @ T_WH0                       # noqa: E731
om = 0.2
out = {}
for tag in ('S1_comoving', 'S2_tcp_static'):
    lins, angs, wlin = [], [], []
    Tc_fixed = Hof(phi0) @ T_HE
    prev = None
    for k in range(10):
        phi = phi0 + om * k * DT
        H = Hof(phi)
        Tc = H @ T_HE if tag == 'S1_comoving' else Tc_fixed
        Trel = np.linalg.inv(H @ T_HE) @ Tc                          # 同控制器：對 s＝0 抓取位（此處以 T_HE 代表）
        if prev is not None:
            l_, a_ = G.MovingGraspCtrl.rel_motion(prev[0], Trel)
            lins.append(l_); angs.append(a_)
            wlin.append(float(np.linalg.norm(Tc[:3, 3] - prev[1][:3, 3])) / DT)
        prev = (Trel, Tc)
    r_h = float(np.hypot(*T_WH0[:2, 3]))
    out[tag] = dict(rel_lin_mmps_max=round(max(lins) * 1e3, 6), rel_ang_rps_max=round(max(angs), 6), world_diff_lin_mmps_max=round(max(wlin) * 1e3, 3), handle_speed_mmps=round(r_h * om * 1e3, 3))
    print(tag, out[tag])
out['S1_ok'] = out['S1_comoving']['rel_lin_mmps_max'] < 1e-6 and out['S1_comoving']['rel_ang_rps_max'] < 1e-6
out['S2_ok'] = out['S2_tcp_static']['rel_lin_mmps_max'] > 5.0
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print(out['S1_ok'], out['S2_ok'])
