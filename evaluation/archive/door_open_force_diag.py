#!/usr/bin/env python3
"""開門正式趟手腕力診斷 r2（Codex 20261112_234431 更正）。
對齊：meas 第 k 列在 ex.step 之前取＝第 k−1 個物理步的結果；該步施加的設定點＝ctrl 第 k−1 列（更新後）⇒ 量測 k 配設定點 k−1（含中止那一列）。
預測：手臂 PD 不平衡 Δτ＝kp·[(s_{k−1}−q_k)−(s−q)_基線]−kd·q̇_k，ΔW_tcp＝J⁻ᵀΔτ（世界座標、TCP 點），轉到 link6 座標、移到 link6 原點（joint6 子連桿；link_eef／夾爪殼固定在同一原點），
與量測（metadata joint6＋1 列，子連桿座標，未校準）的變化量逐分量比較。未扣手臂慣性、重力隨構型變化、底盤加速項——**不能證明唯一力來源**。
門：資產門扇長 0.95 m、厚 0.04 m、8 kg；鉸鏈慣量 I_h＝m(l²+t²)/12＋m·[(l/2)²+(t/2)²]；PhysX 角阻尼 2.0/s 只作用於繞質心角速度 ⇒ 等效阻尼力矩 2.0·I_cz·φ̇。物理門角取 log.npz（每物理步），與最後控制拍分列。
    python3 evaluation/door_open_force_diag.py RUN_DIR OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_open_exec_ctrl as XO
K = XO.K
KP, KD = 600.0, 60.0
run = sys.argv[1]
Z = np.load(os.path.join(run, 'open_steps.npz')); M, C = Z['meas'], Z['ctrl']
res = json.load(open(os.path.join(run, 'result.json')))
nm = len(M)
ks = np.arange(1, nm)                                                # 量測列 k ↔ 設定點列 k−1
q9 = M[:, 2:11]
qd = np.gradient(q9[:, 3:], 0.01, axis=0)


def J6(q):
    T0 = K.fk(q, 'link_tcp'); J = np.zeros((6, 6))
    for j in range(6):
        dq = np.zeros(9); dq[3 + j] = 1e-6
        T1 = K.fk(q + dq, 'link_tcp')
        J[:3, j] = (T1[:3, 3] - T0[:3, 3]) / 1e-6
        dR = T1[:3, :3] @ T0[:3, :3].T
        J[3:, j] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2e-6
    return J


e0 = C[0, 32:38] - q9[1, 3:]                                         # 基線：第一個施加區間（命令未到、靜止）
pred, meas = [], []
for k in ks:
    dtau = KP * ((C[k - 1, 32:38] - q9[k, 3:]) - e0) - KD * qd[k]
    W = np.linalg.solve(J6(q9[k]).T, dtau)
    Tt, To = K.fk(q9[k], 'link_tcp'), K.fk(q9[k], 'link6')
    R = To[:3, :3]
    F_l = R.T @ W[:3]
    T_l = R.T @ (W[3:] + np.cross(Tt[:3, 3] - To[:3, 3], W[:3]))
    pred.append(np.r_[F_l, T_l])
    meas.append(M[k, 13:19] - M[1, 13:19])
pred, meas = np.array(pred), np.array(meas)
t = M[ks, 0]
win = (t >= 0.08 - 1e-9) & (t <= 0.19 + 1e-9)


def cmp(a, b):
    sc = float(np.linalg.norm(b, axis=1).max())
    return dict(scale_meas_peak=round(sc, 3), nrmse=round(float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))) / sc), 3), peak_err=round(float(abs(np.linalg.norm(a, axis=1).max() - sc) / sc), 3),
                cos_end=round(float(a[-1] @ b[-1] / (np.linalg.norm(a[-1]) * np.linalg.norm(b[-1]) + 1e-12)), 3))
Fcmp, Tcmp = cmp(pred[win, :3], meas[win, :3]), cmp(pred[win, 3:], meas[win, 3:])
# 門
L = np.load(os.path.join(run, 'log.npz'))['rows']
O = L[L[:, 1] == 5]
m_, l_, th_ = 8.0, 0.95, 0.04
I_cz = m_ * (l_ ** 2 + th_ ** 2) / 12.0
I_h = I_cz + m_ * ((l_ / 2) ** 2 + (th_ / 2) ** 2)
phi = O[:, 2]
dphi = np.gradient(phi, 0.01); ddphi = np.gradient(dphi, 0.01)
tau_door = I_h * ddphi + 2.0 * I_cz * dphi
r_h = math.hypot(0.80, 0.040)
out = dict(run=os.path.basename(run.rstrip('/')), kp=KP, kd=KD, alignment='量測 k ↔ 施加設定點 k−1；含中止樣本 t=%.2f' % M[-1, 0], frame='link6 座標、link6 原點（量測與預測同座標同作用點；各扣基線）',
           window_s=[0.08, 0.19], force=Fcmp, torque=Tcmp,
           rows=[dict(t=round(float(t[i]), 3), meas_F=[round(float(x), 3) for x in meas[i, :3]], pred_F=[round(float(x), 3) for x in pred[i, :3]], meas_T=[round(float(x), 3) for x in meas[i, 3:]], pred_T=[round(float(x), 3) for x in pred[i, 3:]], F_norm_raw=round(float(M[ks[i], 12]), 3)) for i in range(len(ks))],
           door=dict(I_hinge=round(I_h, 3), I_com_z=round(I_cz, 3), r_handle=round(r_h, 3), phi_phys_end_deg=round(math.degrees(float(phi[-1])), 4), phi_last_ctrl_tick_deg=res['O_open']['phi_end_deg'],
                     tau_max_nm=round(float(np.abs(tau_door).max()), 3), handle_force_equiv_max_n=round(float(np.abs(tau_door).max()) / r_h, 3), ddphi_max=round(float(np.abs(ddphi).max()), 3),
                     note='門慣量＋角阻尼所需的把手切向力（由物理門角二階差分反算；差分雜訊未濾）'),
           caveat='未扣手臂慣性、重力隨構型、底盤加速項；單條短暫單調上升序列的高相關不能證明唯一力來源；有效時間常數約 0.2 s 只列觀察')
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print({k: v for k, v in out.items() if k != 'rows'})
for r in out['rows'][6:]:
    print(r)
