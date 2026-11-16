#!/usr/bin/env python3
"""K2（Codex 20261116）：手臂 設定點—實測 偏差的模型核對（只讀既有 Isaac 紀錄）。
模型＝保持段錨 off0＋(G(q)−G0)/kp（重力隨構型；FastGravity3、kp 600）−τ·q̇（一階動態落後，τ 取辨識檔）。
    python3 evaluation/far_dry/arm_offset_model_check.py OUT.json RUN_ID [RUN_ID ...]"""
import json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
import door_motm_grasp_ctrl as G
import door_fast_kin as DFK

BM, K = G.BM, G.K
grav = DFK.FastGravity3(K, BM.arm_bodies(K))
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
tau = 0.01 / np.asarray(ident['alpha'], float)
KP = 600.0
out = {}
for run in sys.argv[2:]:
    L = np.load(os.path.join(HERE, 'runs', run, 'log.npz'))
    A, c = L['L'], [str(x) for x in L['cols']]
    t, q, s, q9 = A[:, 0], A[:, 4:10], A[:, 10:16], A[:, 1:10]
    qd = np.gradient(q, 0.01, axis=0)
    mov = np.abs(A[:, c.index('ua0'):c.index('ua0') + 6]).max(axis=1) > 1e-6
    if not mov.any():
        out[run] = dict(note='手臂沒有動')
        continue
    k_first = int(np.argmax(mov))
    hold = slice(max(k_first - 300, 0), k_first - 5)
    off_meas = q - s
    kk = np.arange(len(q9))[::5]
    Gq = np.array([grav(x) for x in q9[kk]])
    G0 = Gq[(kk >= hold.start) & (kk < hold.stop)].mean(axis=0)
    off0 = off_meas[hold].mean(axis=0)
    pred_static = off0 + (Gq - G0) / KP
    r_static = off_meas[kk] - pred_static
    r_full = off_meas[kk] - (pred_static - tau * qd[kk])
    m = mov[kk]
    tr = m & (t[kk] > t[kk][m][-1] - 5.4)                           # 追蹤段（手臂運動的最後 5.4 s）
    f = lambda x: [round(float(v), 4) for v in x]                    # noqa: E731
    out[run] = dict(n_track=int(tr.sum()), off0_hold=f(off0), G0_Nm=f(G0), G_track_mean_Nm=f(Gq[tr].mean(axis=0)), gravity_term_change_abs_max=f(np.abs((Gq[tr] - G0) / KP).max(axis=0)),
                    off_abs_max=f(np.abs(off_meas[kk][tr]).max(axis=0)), resid_after_gravity_abs_max=f(np.abs(r_static[tr]).max(axis=0)),
                    resid_after_gravity_and_lag_abs_max=f(np.abs(r_full[tr]).max(axis=0)), resid_after_gravity_and_lag_mean=f(r_full[tr].mean(axis=0)), resid_after_gravity_and_lag_std=f(r_full[tr].std(axis=0)))
    print(run, json.dumps(out[run]))
out['note'] = 'off＝q−s（實測關節−設定點）。遠距執行器的手臂摩擦＝URDF 匯入預設（legacy jointFriction=1，未改）；門旁第 12 趟用 no_passive_arm_friction——兩者不同，殘差可能含摩擦'
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
