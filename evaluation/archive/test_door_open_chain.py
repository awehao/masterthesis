#!/usr/bin/env python3
"""開門執行層 × door_chain_dyn 局部動力學模型的閉迴路比較（r5 'qp' vs A 'closure'）。
起點＝第二趟實錄（m8house_door_open_2）：同一份計畫、同一實測起始構型與門角、同一承接設定點與偏置；模型先以承接設定點靜平衡 0.5 s。
受控對象＝剛性閉鏈＋關節 PD 柔順（r1 未過驗收門檻：對第二趟 Tx／Fy 高估、Fx 低估），所以這裡只比較**趨勢**（原始幾何偏差、PD 不平衡、模型力旋量增量、限制），
不用來保證低於 40 N；控制器不接模型力（force_n＝None，不觸發超力中止）。
    python3 evaluation/test_door_open_chain.py MODE T_END OUT.json"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_chain_dyn as D                                           # noqa: E402
import door_open_exec_ctrl as X                                      # noqa: E402

mode, t_end, out_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]
RUN = os.path.join(HERE, 'runs', 'm8house_door_open_2')
res = json.load(open(os.path.join(RUN, 'result.json')))
P = np.load(os.path.join(RUN, 'plan.npz'))
Z = np.load(os.path.join(RUN, 'open_steps.npz'))
M0 = Z['meas'][0]
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
q0 = M0[2:11].copy(); phi0 = float(M0[11])
s0 = q0[3:] + np.array(res['handover']['setpoint_minus_measured'])
bias = np.array(res['handover']['bias'])
fq = float(res['plan']['finger_q_model'])
cm = D.ChainModel(P['T_WH_bind'], float(P['phi_bind']), P['T_HE'])
x = np.r_[q0, phi0]; v = np.zeros(10)
for _ in range(50):
    x, v, _ = cm.step(x, v, s0, np.zeros(3))
c = X.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], math.radians(85.0), dict(s=s0.copy(), pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=bias), fq, arm_mode=mode)
rows, W0 = [], None
t = 0.0
n = int(round(t_end / 0.01))
for k in range(n):
    out = c.step(t, x[:9], float(x[9]), None)
    if out['abort']:
        break
    x_prev = x.copy()
    x, v, info = cm.step(x, v, out['arm_target'], out['base_twist'])
    if W0 is None:
        W0 = info['Wj_link6'].copy()
    dW = info['Wj_link6'] - W0
    rows.append([t + 0.01, *dW, float(np.abs(info['tau']).max()), *info['tau'], math.degrees(float(x[9])), *x[:9], *out['arm_target']])
    t += 0.01
R = np.array(rows)
rep = c.report()
dF, dT = np.linalg.norm(R[:, 1:4], axis=1), np.linalg.norm(R[:, 4:7], axis=1)


def at(tt):
    i = int(np.argmin(np.abs(R[:, 0] - tt)))
    return dict(t=round(float(R[i, 0]), 2), dF=round(float(dF[i]), 2), dT=round(float(dT[i]), 3), door_deg=round(float(R[i, 14]), 3))
outd = dict(mode=mode, t_end=t_end, steps=len(R), aborted=c.abort, stop=rep['stop'], closure_stats=rep.get('closure_stats'),
            model_wrench_increment=dict(dF_max=round(float(dF.max()), 2), dF_t_at_max=round(float(R[np.argmax(dF), 0]), 2), dT_max=round(float(dT.max()), 3), dF_p95=round(float(np.percentile(dF, 95)), 2),
                                        samples=[at(tt) for tt in (0.1, 0.15, 0.19, 0.25, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0) if tt <= R[-1, 0] + 1e-9]),
            joint_torque_abs_max=round(float(R[:, 7].max()), 3), setpoint_model_deviation=rep['setpoint_model_deviation'], closure_measured_in_model=rep['closure_measured'],
            door_end_deg=round(float(R[-1, 14]), 3), timing_tick_ms=rep['timing']['whole_tick_ms'],
            note='受控對象＝door_chain_dyn r1（未過驗收門檻）；只比趨勢，不保證 40 N；開門前 0.5 s 模型靜平衡；增量基線＝第一步')
json.dump(outd, open(out_path, 'w'), ensure_ascii=False, indent=1, default=float)
np.save(out_path.replace('.json', '_rows.npy'), R)
print(json.dumps({k_: outd[k_] for k_ in ('mode', 'steps', 'aborted', 'model_wrench_increment', 'joint_torque_abs_max', 'door_end_deg', 'closure_stats')}, ensure_ascii=False, default=float))
