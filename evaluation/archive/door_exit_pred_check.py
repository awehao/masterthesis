#!/usr/bin/env python3
"""full_1 實錄核對（Codex 20261113_091401 第 1 點）：退出起點（只用退出前資料）建立新舊兩個預測模型，
1 重現舊模型（承接偏置）的「現在停」拒絕點，並列新模型（錨定偏置）的同一掃掠結果；
2 用退出後 0.29 s 的零命令實錄，比較兩模型對關節、TCP、兩指—橫桿間隙的預測誤差（門角用實錄）。不使用未來資料重新錨定。
    python3 evaluation/door_exit_pred_check.py OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_exit_ctrl as XE
K, S, C0 = XE.K, XE.S, XE.C0
RUN = os.path.join(HERE, 'runs', 'm8house_door_full_1')
r = json.load(open(os.path.join(RUN, 'result.json'))); L = np.load(os.path.join(RUN, 'log.npz'))['rows']; P = np.load(os.path.join(RUN, 'plan.npz'))
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))); al = np.asarray(ident['alpha'], float)
R7, RX = L[L[:, 1] == 7], L[L[:, 1] == 8]
q0 = np.r_[R7[-1, 5:8], R7[-1, 8:14]]; phi0 = float(R7[-1, 2]); om0 = float((R7[-1, 2] - R7[-11, 2]) / 0.1); fq = 0.5 * float(R7[-1, 3] + R7[-1, 4])
s0 = np.load(os.path.join(RUN, 'open_steps.npz'))['ctrl'][-1, 32:38].copy()
bias_old = np.asarray(C0.make_cfg(ident).arm_model.bias, float) + al * np.array(r['I_insert']['d_hat'])     # 承接：bias0＋α·d̂（插入觀測；首次接觸後凍結）
out = dict(equiv_offset_old_mrad=(bias_old / al * 1e3).round(3).tolist(), measured_q_minus_s_mrad=((q0[3:] - s0) * 1e3).round(3).tolist(), omega0_dps=round(math.degrees(om0), 4))
res = {}
for tag, anchor in (('old_inherited', False), ('new_anchored', True)):
    c = XE.DoorExitCtrl(ident, P['T_WH_bind'], float(P['phi_bind']), s0, [np.zeros(9)] * XE.D_STEPS, np.zeros(9), np.zeros(9), bias_old, fq, phi0, om0, np.zeros((1, 2)), q0[3:], scope_end='X2', anchor_pred_bias=anchor)
    o = c.step(0.0, q0.copy(), phi0)
    # 零命令預測 vs 實錄
    qp, sp = q0.copy(), s0.copy(); err_q, err_tcp, gap_p, gap_m = [], [], [], []
    for k in range(len(RX)):
        qp, sp = S.plant_phys_step(qp, sp, np.zeros(9), c.cfg, 1)
        qm = np.r_[RX[k, 5:8], RX[k, 8:14]]
        err_q.append(np.abs(qp[3:] - qm[3:]).max()); err_tcp.append(np.linalg.norm(K.fk(qp, 'link_tcp')[:3, 3] - K.fk(qm, 'link_tcp')[:3, 3]))
        gap_p.append(c.finger_bar(qp, float(RX[k, 2]))); gap_m.append(c.finger_bar(qm, float(RX[k, 2])))
    res[tag] = dict(first_tick_stop=c.stop, exemption_released=c.finger_release_t is not None, stop_sweep_min_margin_mm=round(c.mins['stop_sweep'] * 1e3, 3), pred_bias_equiv_mrad=c.anchor['equiv_offset_used_mrad'],
                    zero_cmd_0p29s=dict(n=len(RX), joint_err_max_mrad=round(float(max(err_q)) * 1e3, 3), tcp_err_max_mm=round(float(max(err_tcp)) * 1e3, 3), gap_pred_end_mm=round(gap_p[-1] * 1e3, 3), gap_meas_end_mm=round(gap_m[-1] * 1e3, 3),
                                        gap_err_max_mm=round(float(np.abs(np.array(gap_p) - np.array(gap_m)).max()) * 1e3, 3)))
out.update(res)
out['checks'] = dict(old_reproduces_rejection=bool(res['old_inherited']['first_tick_stop'] and res['old_inherited']['first_tick_stop']['why'] == 'stop_now_violation'), new_passes_first_tick=bool(res['new_anchored']['first_tick_stop'] is None),
                     new_pred_better=bool(res['new_anchored']['zero_cmd_0p29s']['joint_err_max_mrad'] < res['old_inherited']['zero_cmd_0p29s']['joint_err_max_mrad']))
out['pass'] = bool(all(out['checks'].values()))
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, default=float)[:2200])
