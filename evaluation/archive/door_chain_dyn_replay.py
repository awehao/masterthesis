#!/usr/bin/env python3
"""第二趟命令原樣重播到 door_chain_dyn 局部動力學模型，與 Isaac 實錄比對（Codex 20261112_234431 工程驗收）。
固定時間對齊（不掃位移）：模型第 k 步用 ctrl 第 k 列的施加設定點與生效底盤命令，結果對 meas 第 k＋1 列。
比較 0.08–0.19 s：link6 座標、link6 原點的力／力矩向量增量（各扣第 1 列基線）；正規化 RMSE、峰值誤差（尺度＝量測動態增量峰值）≤25%；
起升時刻（增量首次 >1 N）誤差 ≤1 物理步；同時核關節、門角、底盤軌跡。
    python3 evaluation/door_chain_dyn_replay.py RUN_DIR OUT.json"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_chain_dyn as D                                           # noqa: E402

run = sys.argv[1]
Z = np.load(os.path.join(run, 'open_steps.npz'))
M, C = Z['meas'], Z['ctrl']
P = np.load(os.path.join(run, 'plan.npz'))
cm = D.ChainModel(P['T_WH_bind'], float(P['phi_bind']), P['T_HE'])
x = np.r_[M[0, 2:11], M[0, 11]]
v = np.zeros(10)
PRE_S = 0.5                                                          # 靜平衡初始化：以開門前設定點 s₀、底盤命令 0 先走 0.5 s（模型自身的靜態；不改參數）
x_meas0 = x.copy()
for _ in range(int(round(PRE_S / 0.01))):
    x, v, _ = cm.step(x, v, C[0, 32:38], np.zeros(3))
settle_shift = dict(arm_mrad=[round(float(z), 3) for z in (x[3:9] - x_meas0[3:9]) * 1e3], door_deg=round(math.degrees(float(x[9] - x_meas0[9])), 5), base_mm=[round(float(z), 3) for z in (x[:2] - x_meas0[:2]) * 1e3], yaw_mdeg=round(math.degrees(float(x[2] - x_meas0[2])) * 1e3, 3))
x_model0 = x.copy()
X, Wm, INFO = [x.copy()], [], []
for k in range(len(C)):
    x, v, info = cm.step(x, v, C[k, 32:38], C[k, 23:26])
    X.append(x.copy()); Wm.append(info['Wj_link6']); INFO.append(info)
X, Wm = np.array(X), np.array(Wm)
n = min(len(X), len(M))
t = M[1:n, 0]
ib = int(np.argmin(np.abs(t - 0.08)))                                # 共同基線：t＝0.08 s（第一筆命令開始作用的物理步之前）
meas = M[1:n, 13:19] - M[1 + ib, 13:19]
pred = Wm[:n - 1] - Wm[ib]
win = (t >= 0.08 - 1e-9) & (t <= 0.19 + 1e-9)


def cmp(a, b):
    sc = float(np.linalg.norm(b, axis=1).max())
    return dict(scale_meas_peak=round(sc, 3), nrmse=round(float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))) / sc), 3),
                peak_err=round(float(abs(np.linalg.norm(a, axis=1).max() - sc) / sc), 3), pass_=bool(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))) / sc <= 0.25 and abs(np.linalg.norm(a, axis=1).max() - sc) / sc <= 0.25))


def onset(a, thr=1.0):
    i = np.where(np.linalg.norm(a, axis=1) > thr)[0]
    return float(t[i[0]]) if len(i) else None
Fc, Tc = cmp(pred[win, :3], meas[win, :3]), cmp(pred[win, 3:], meas[win, 3:])
on_m, on_p = onset(meas[:, :3]), onset(pred[:, :3])
dX = X[1:n] - x_model0                                               # 軌跡比較用增量（模型扣自身靜平衡起點、量測扣實測起點）
dM = np.c_[M[1:n, 2:11], M[1:n, 11]] - x_meas0
traj = dict(settle_shift=settle_shift, arm_joint_inc_err_mrad_max=[round(float(x_), 3) for x_ in np.abs(dX[:, 3:9] - dM[:, 3:9]).max(axis=0) * 1e3],
            door_inc_err_deg_max=round(math.degrees(float(np.abs(dX[:, 9] - dM[:, 9]).max())), 4), base_inc_pos_err_mm_max=round(float(np.linalg.norm(dX[:, :2] - dM[:, :2], axis=1).max()) * 1e3, 3),
            base_inc_yaw_err_mdeg_max=round(math.degrees(float(np.abs(dX[:, 2] - dM[:, 2]).max())) * 1e3, 2),
            arm_joint_err_mrad_max=[round(float(x_), 3) for x_ in np.abs(X[1:n, 3:9] - M[1:n, 5:11]).max(axis=0) * 1e3],
            door_err_deg_max=round(math.degrees(float(np.abs(X[1:n, 9] - M[1:n, 11]).max())), 4), door_model_end_deg=round(math.degrees(float(X[n - 1, 9])), 4), door_meas_end_deg=round(math.degrees(float(M[n - 1, 11])), 4),
            base_pos_err_mm_max=round(float(np.linalg.norm(X[1:n, :2] - M[1:n, 2:4], axis=1).max()) * 1e3, 3), base_yaw_err_mdeg_max=round(math.degrees(float(np.abs(X[1:n, 2] - M[1:n, 4]).max())) * 1e3, 2),
            closure_resid_mm_max=round(float(max(np.linalg.norm(i_['e_pre'][:3]) for i_ in INFO)) * 1e3, 4))
out = dict(run=os.path.basename(run.rstrip('/')), model='door_chain_dyn（剛性閉鏈＋關節 PD 柔順；底盤步首重設速度、步內自由；門單自由度）', alignment='固定：模型步 k ↔ ctrl 列 k ↔ meas 列 k+1；不掃時間位移；增量基線＝t 0.08 s（兩者同一時刻）；模型先以 s₀ 靜平衡 0.5 s',
           bodies=len(D.BODIES), missing_bodies=D.MISSING, I_hinge=round(D.I_H, 3), door_damp=round(D.C_ANG * D.I_CZ, 3), tau_max=D.TAU_MAX.tolist(),
           window_s=[0.08, 0.19], force=Fc, torque=Tc, onset_s=dict(meas=on_m, model=on_p, err_steps=None if on_m is None or on_p is None else round(abs(on_m - on_p) / 0.01)), trajectories=traj,
           saturation_steps=int(sum(any(i_['sat']) for i_ in INFO)),
           rows=[dict(t=round(float(t[i]), 3), meas_dF=[round(float(z), 2) for z in meas[i, :3]], model_dF=[round(float(z), 2) for z in pred[i, :3]], meas_dT=[round(float(z), 3) for z in meas[i, 3:]], model_dT=[round(float(z), 3) for z in pred[i, 3:]],
                     door_meas_deg=round(math.degrees(float(M[i + 1, 11])), 4), door_model_deg=round(math.degrees(float(X[i + 1, 9])), 4), tau=[round(float(z), 2) for z in INFO[i]['tau']]) for i in range(n - 1)],
           unmodeled='關節摩擦（physxJoint:jointFriction＝1）、科氏／離心、輪子與地面、PhysX TGS 迭代細節、手指柔順')
ok = Fc['pass_'] and Tc['pass_'] and out['onset_s']['err_steps'] is not None and out['onset_s']['err_steps'] <= 1
out['pass'] = bool(ok)
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print({k_: v_ for k_, v_ in out.items() if k_ != 'rows'})
for r in out['rows'][6:]:
    print(r)
