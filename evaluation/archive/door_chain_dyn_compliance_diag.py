#!/usr/bin/env python3
"""door_chain_dyn 殘差診斷：Isaac 的 FixedJoint（夾爪殼↔門扇）在求解器下並非完全剛性。
1 由 log.npz 的夾爪殼／門扇原始位姿算綁定後相對位姿漂移（位置 mm、轉角 mrad）。
2 對照：把實測漂移當閉鏈偏移輸入模型（輸入＝實測資料，不是擬合係數），底盤兩種語意（步首重設＋步內自由／照實測軌跡）各跑一次，比較 0.08–0.19 s 力旋量增量。
這是**診斷**，不是模型通過；實測偏移不能用於預測新命令。
    python3 evaluation/door_chain_dyn_compliance_diag.py RUN_DIR OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_chain_dyn as D
import door_bind_logic as BL
run = sys.argv[1]
Z = np.load(os.path.join(run, 'open_steps.npz')); M, C = Z['meas'], Z['ctrl']; P = np.load(os.path.join(run, 'plan.npz'))
L = np.load(os.path.join(run, 'log.npz')); R = L['rows']; cols = list(L['cols']); ig, il = cols.index('gp_x'), cols.index('lp_x')
O, E = np.where(R[:, 1] == 5)[0], np.where(R[:, 1] == 4)[0]
rel = lambda r: np.linalg.inv(BL.pose_from_pq(r[il:il + 3], r[il + 3:il + 7])) @ BL.pose_from_pq(r[ig:ig + 3], r[ig + 3:ig + 7])   # noqa: E731
rel0 = rel(R[E[-1]])
Tgt = np.eye(4); Tgt[2, 3] = 0.0836                                  # 夾爪殼 → TCP（URDF joint_tcp）
drift, deltas = [], []
for k in O:
    d = np.linalg.inv(rel0) @ rel(R[k])
    w = 0.5 * np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
    drift.append(dict(t_rel=round(float(R[k, 0] - R[O[0], 0] + 0.01), 3), dp_mm=[round(float(z) * 1e3, 4) for z in d[:3, 3]], drot_mrad=[round(float(z) * 1e3, 4) for z in w], F_norm=round(float(np.linalg.norm(R[k, 10:13])), 2)))
    deltas.append(np.linalg.inv(Tgt) @ d @ Tgt)


def run_model(mode, use_delta):
    cm = D.ChainModel(P['T_WH_bind'], float(P['phi_bind']), P['T_HE'])
    x = np.r_[M[0, 2:11], M[0, 11]]; v = np.zeros(10)
    for _ in range(50):
        x, v, _ = cm.step(x, v, C[0, 32:38], np.zeros(3), base_prescribed=None if mode == 'reset' else np.zeros(3))
    Wm = []
    for k in range(len(C)):
        cm.delta = deltas[k] if use_delta else np.eye(4)
        vb = None if mode == 'reset' else (M[k + 1, 2:5] - M[k, 2:5]) / 0.01
        x, v, info = cm.step(x, v, C[k, 32:38], C[k, 23:26], base_prescribed=vb)
        Wm.append(info['Wj_link6'])
    Wm = np.array(Wm); t = M[1:, 0]; ib = int(np.argmin(np.abs(t - 0.08)))
    meas, pred = M[1:, 13:19] - M[1 + ib, 13:19], Wm - Wm[ib]
    w = (t >= 0.08 - 1e-9) & (t <= 0.19 + 1e-9)
    o = {}
    for nm, sl in (('force', slice(0, 3)), ('torque', slice(3, 6))):
        a, b = pred[w, sl], meas[w, sl]; sc = float(np.linalg.norm(b, axis=1).max())
        o[nm] = dict(nrmse=round(float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))) / sc), 3), peak_err=round(float(abs(np.linalg.norm(a, axis=1).max() - sc) / sc), 3), end_model=[round(float(z), 3) for z in pred[-1, sl]], end_meas=[round(float(z), 3) for z in meas[-1, sl]])
    return o
out = dict(run=os.path.basename(run.rstrip('/')), fixedjoint_drift=drift,
           variants={f'{m}_{"meas_offset" if u else "rigid"}': run_model(m, u) for m in ('reset', 'prescribed') for u in (False, True)},
           note='reset＝Isaac 語意（步首重設底盤速度、步內自由）；prescribed＝底盤照實測軌跡；meas_offset＝閉鏈加入實測夾爪–門扇相對漂移（診斷輸入，非預測）')
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out['variants'], ensure_ascii=False)); print(drift[-1])
