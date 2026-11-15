#!/usr/bin/env python3
"""B2 的回授符號與限幅核對（局部線性模型；只篩方向，不是力的安全保證）。
模型：手臂被剛性約束在 q_act（閉鏈），設定點 s＝q_eq0＋δ（q_eq0＝無外力平衡位置的名目設定）。位置驅動力矩 kp(s−q_act) 由約束承受：
  JᵀW_ext＝−kp·(s＋off−q_act) ⇒ W_ext＝J⁻ᵀ·kp·(q_act−s−off)。開門方向另加一個固定的切向負載 λ0（應被保留，不被釋放）。
案例：(a) 垂直 1.2 mm 偏差（第 7、8 趟靜止狀態量級）；(b) 垂直 2.9 mm；(c) 大偏差 12 mm（應飽和、不發散）；(d) 符號反向（把回授取負，應使約束反力變大——證明符號有鑑別力）。
    python3 evaluation/door_preload_release_check.py OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_open_exec_ctrl as XO
import door_fast_kin as DFK
import door_preload_release as PR
K = XO.K
RUN = os.path.join(HERE, 'runs', 'm8house_door_motm_8')
M8 = np.load(os.path.join(RUN, 'open_steps.npz'), allow_pickle=True)['meas']
q9 = M8[-1, 1:10].copy()
KP, DT = 600.0, 0.05
T0, J9 = DFK.FC.fk_jac(q9)
J = J9[:, 3:9]
p = T0[:3, 3]
out = {}
for name, dz, sign, lam0 in (('dz_1p2mm', 0.0012, 1.0, 6.0), ('dz_2p9mm', 0.0029, 1.0, 6.0), ('dz_12mm_saturate', 0.012, 1.0, 6.0), ('wrong_sign', 0.0012, -1.0, 6.0)):
    dq0 = np.linalg.solve(J, np.r_[0.0, 0.0, dz, 0.0, 0.0, 0.0])    # 設定點平衡位置比實際高 dz ⇒ 手臂往上頂
    delta = np.zeros(6)
    t_hat = np.array([-p[1], p[0], 0.0]) / math.hypot(p[0], p[1])
    hist = []
    for n in range(200):
        tau_pd = KP * (dq0 + delta)                                   # kp·(s＋off−q_act)
        W = -np.linalg.solve(J.T, tau_pd)                             # 外界對夾爪的旋量（TCP、世界）
        F, Mm = W[:3] + lam0 * t_hat, W[3:]
        Fc, Mc, lam = PR.split(F, Mm, p)
        new, info = PR.update(delta, sign * F, sign * Mm, p, J, KP, DT)
        hist.append((n * DT, float(np.linalg.norm(Fc)), float(np.linalg.norm(Mc)), lam, float(np.abs(delta).max()), float(np.abs(new - delta).max() / DT)))
        delta = new
    H = np.array(hist)
    out[name] = dict(Fc_start_n=round(H[0, 1], 2), Fc_end_n=round(H[-1, 1], 3), Fc_at_0p5s=round(H[10, 1], 2), Fc_at_1s=round(H[20, 1], 2), Fc_max_n=round(H[:, 1].max(), 2), monotone_nonincreasing=bool(np.all(np.diff(H[:, 1]) <= 1e-9)),
                     lambda_start_n=round(H[0, 3], 3), lambda_end_n=round(H[-1, 3], 3), delta_abs_max_rad=round(H[:, 4].max(), 5), rate_abs_max=round(H[:, 5].max(), 5),
                     within_limits=bool(H[:, 4].max() <= PR.DELTA_MAX + 1e-12 and H[:, 5].max() <= PR.RATE_MAX + 1e-9), saturated=bool(H[:, 4].max() >= PR.DELTA_MAX - 1e-9))
    print(name, out[name])
out['params'] = dict(KI=PR.KI, RATE_MAX=PR.RATE_MAX, DELTA_MAX=PR.DELTA_MAX, kp=KP, dt=DT, basis='m8house_door_motm_8 末拍構型')
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
