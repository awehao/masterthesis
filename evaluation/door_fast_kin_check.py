#!/usr/bin/env python3
"""G2 元件核對：(a) FastGravity 對 door_body_model.arm_gravity（逐構型最大絕對差）；(b) ik_closure_fc 對開門執行層原 _ik_closure（同種子：解、殘差、σ 的差）；
構型：full_2 密集參考全部取樣＋隨機擾動（手臂 ±0.3 rad、底盤 ±5 cm、偏航 ±5°）；IK 目標取擾動後構型的 TCP，種子取原構型（典型一拍內的變化量級另加 ±0.05 rad）。
    python3 evaluation/door_fast_kin_check.py OUT.json"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_open_exec_ctrl as XO
import door_fast_kin as DFK
import door_body_model as BM
K = XO.K
rng = np.random.default_rng(20261113)
P = np.load(os.path.join(HERE, 'runs', 'm8house_door_full_2', 'plan.npz'))
cfgs = [x[:9].copy() for x in P['Xd']]
for _ in range(1500):
    x = P['Xd'][rng.integers(len(P['Xd']))][:9].copy()
    x[:2] += rng.uniform(-0.05, 0.05, 2); x[2] += math.radians(rng.uniform(-5, 5)); x[3:] += rng.uniform(-0.3, 0.3, 6)
    cfgs.append(x)
bodies = BM.arm_bodies(K); FG = DFK.FastGravity(K, bodies)
gd, t_o, t_f = 0.0, [], []
for x in cfgs:
    t0 = time.perf_counter(); a = BM.arm_gravity(K, x, bodies); t1 = time.perf_counter(); b = FG(x); t2 = time.perf_counter()
    gd = max(gd, float(np.abs(a - b).max())); t_o.append((t1 - t0) * 1e3); t_f.append((t2 - t1) * 1e3)
# IK：原式（複製自 r7 _ik_closure，不經 g2）
def ik_orig(Tg, b, seed):
    q = np.asarray(seed, float).copy(); res = np.inf
    for _ in range(30):
        q9 = np.r_[b, q]; Tc = K.fk(q9, 'link_tcp'); dR = Tg[:3, :3] @ Tc[:3, :3].T
        e = np.r_[Tg[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
        res = float(np.linalg.norm(e))
        if res < 1e-10:
            break
        J = K.jacobian(q9, 'link_tcp')[:, 3:9]; q = q + np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
    J = K.jacobian(np.r_[b, q], 'link_tcp')[:, 3:9]
    return q, res, float(np.linalg.svd(J, compute_uv=False).min())
qd, rd, sd, nconv_o, nconv_f, ti_o, ti_f, br = 0.0, 0.0, 0.0, 0, 0, [], [], 0
for x in cfgs[::2]:
    Tg = K.fk(x, 'link_tcp'); seed = x[3:] + rng.uniform(-0.05, 0.05, 6)
    t0 = time.perf_counter(); a = ik_orig(Tg, x[:3], seed); t1 = time.perf_counter(); b = DFK.ik_closure_fc(Tg, x[:3], seed); t2 = time.perf_counter()
    ti_o.append((t1 - t0) * 1e3); ti_f.append((t2 - t1) * 1e3)
    nconv_o += a[1] < 1e-6; nconv_f += b[1] < 1e-6
    if a[1] < 1e-6 and b[1] < 1e-6:
        qd = max(qd, float(np.abs(a[0] - b[0]).max())); sd = max(sd, abs(a[2] - b[2]))
        br += int(float(np.abs(a[0] - b[0]).max()) > 1e-6)
    else:
        br += int((a[1] < 1e-6) != (b[1] < 1e-6))
out = dict(gravity=dict(n=len(cfgs), max_abs_diff_Nm=gd, orig_ms_p50=round(float(np.median(t_o)), 4), fast_ms_p50=round(float(np.median(t_f)), 4)),
           ik=dict(n=len(cfgs[::2]), converged_orig=int(nconv_o), converged_fc=int(nconv_f), q_max_abs_diff_rad=qd, sigma_max_abs_diff=sd, n_branch_or_conv_mismatch=int(br),
                   orig_ms_p50=round(float(np.median(ti_o)), 4), fc_ms_p50=round(float(np.median(ti_f)), 4)))
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1); print(out)
