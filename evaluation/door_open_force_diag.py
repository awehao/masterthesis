#!/usr/bin/env python3
"""開門正式趟的手腕力診斷（m8house_door_open_2 起有 open_steps.npz）：
以「手臂 PD 不平衡」預測 TCP 力旋量變化：Δτ＝kp·[(s−q)−(s−q)_0]−kd·q̇（相對綁定後開門起點；未扣手臂慣性 M·q̈、TCP 與 joint6 座標不同）⇒ ΔW＝J⁻ᵀΔτ，
與量測手腕六維（metadata joint6＋1 列，未校準）的變化量範數比較。
    python3 evaluation/door_open_force_diag.py RUN_DIR OUT.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_open_exec_ctrl as XO
K = XO.K
KP, KD = 600.0, 60.0                                                 # 執行器 --kp／--kd 預設（run.log 無覆寫）
run = sys.argv[1]
Z = np.load(os.path.join(run, 'open_steps.npz')); M, C = Z['meas'], Z['ctrl']
n = min(len(M), len(C))
q9, s, t = M[:n, 2:11], C[:n, 32:38], M[:n, 0]
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


e0, F0, T0 = s[0] - q9[0, 3:], M[0, 13:16], M[0, 16:19]
rows = []
for k in range(n):
    dtau = KP * ((s[k] - q9[k, 3:]) - e0) - KD * qd[k]
    W = np.linalg.solve(J6(q9[k]).T, dtau)
    rows.append(dict(t=round(float(t[k]), 3), F_meas=round(float(M[k, 12]), 3), dF_meas=round(float(np.linalg.norm(M[k, 13:16] - F0)), 3), dT_meas=round(float(np.linalg.norm(M[k, 16:19] - T0)), 3),
                     dF_pred=round(float(np.linalg.norm(W[:3])), 3), dT_pred=round(float(np.linalg.norm(W[3:])), 3), dtau=[round(float(x), 3) for x in dtau],
                     sp_minus_q_change_mrad=[round(float(x) * 1e3, 2) for x in (s[k] - q9[k, 3:]) - e0], qdot_mradps=[round(float(x) * 1e3, 1) for x in qd[k]]))
first = int(np.argmax(np.abs(C[:n, 23:32]).max(axis=1) > 0))
o = np.array([[r['dF_meas'], r['dF_pred'], r['dT_meas'], r['dT_pred']] for r in rows[first:]])
out = dict(run=os.path.basename(run.rstrip('/')), kp=KP, kd=KD, first_effective_cmd_t=float(t[first]), corr_dF=round(float(np.corrcoef(o[:, 0], o[:, 1])[0, 1]), 3), corr_dT=round(float(np.corrcoef(o[:, 2], o[:, 3])[0, 1]), 3),
           ratio_end_dF=round(float(o[-1, 1] / o[-1, 0]), 2), ratio_end_dT=round(float(o[-1, 3] / o[-1, 2]), 2), rows=rows,
           note='相關高＝力旋量來自手臂關節 PD 不平衡（手臂被閉鏈拉住、跟不上設定點）；幅值高估可能來自未扣慣性與座標差；量測未校準')
json.dump(out, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)
print({k: v for k, v in out.items() if k != 'rows'})
