#!/usr/bin/env python3
"""第十一趟兩份移動續段請求（k=2 接縫 σ 1.55、k=3 σ 1.95）的離線取證（Codex：R2）。只讀紀錄＋同一規劃器同一輸入重算。
輸入照執行器：xs＝已執行參考在接縫節點的值；vs＝參考中央差分的底盤／門角速度＋closure_velocity 的手臂速度。
列：規劃結果、超限關節／時刻／需求值／上限（手臂速率 qd、設定點速率 |qd＋τ·qdd|）、接縫起始速度本身的大小、前後段在接縫的速度是否一致。
    python3 evaluation/door_p2_moving_req_forensic.py OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault('P2_TERM', '1'); os.environ.setdefault('P2_FINAL', '1')
import door_motm_p2 as P2
WB = P2.WB
import door_clear_door_fast as _F
WB.clear_door = _F.clear_door_fast
RUN = os.path.join(HERE, 'runs', 'm8house_door_motm_11')
FS = np.load(os.path.join(RUN, 'first_segment.npz')); Z = np.load(os.path.join(RUN, 'open_steps.npz'), allow_pickle=True)
res = json.load(open(os.path.join(RUN, 'result.json')))
T_WH_b, T_HE, phi_b = FS['T_WH_b'], FS['T_HE_act'], float(FS['phi_b'])
fq = float(np.mean([e for e in res['events'] if e['what'] == 'handover_attempt'][-1]['bind_values']['finger_q']))
Xd, h = Z['Xd_final'], float(Z['h'])
GOAL, H = math.radians(85.0), 2.0
WB.set_model(phi0=phi_b, finger_q=fq)
hlp = WB.DoorWBOpt(T_WH_b, np.eye(4), Xd[0], GOAL, N=2, dt=0.2, T_HE=T_HE, project_start=False)
V_ARM, TAU = WB.V_ARM, WB.TAU_SP
out = dict(run='m8house_door_motm_11', V_ARM=V_ARM, TAU_SP=TAU, h_ref=h, cases=[])
# 已執行首段在各時刻的手臂速率（參考本身）
qd_ref = np.diff(Xd[:, 3:9], axis=0) / h
out['first_segment_ref'] = dict(arm_rate_max=round(float(np.abs(qd_ref).max()), 4), at_sigma=round(float(np.argmax(np.abs(qd_ref).max(axis=1)) * h), 3), joint=int(np.argmax(np.abs(qd_ref).max(axis=0))) + 1)
for k_, sig_s in ((2, 1.55), (3, 1.95)):
    i_ = min(int(round(sig_s / h)), len(Xd) - 1)
    xs = Xd[i_].copy()
    vv = (Xd[min(i_ + 1, len(Xd) - 1)] - Xd[max(i_ - 1, 0)]) / (2 * h)
    WB.set_model(phi0=phi_b, finger_q=fq)
    va = hlp.closure_velocity(xs, vv[:3], vv[9])
    vs = np.r_[vv[:3], va, vv[9]]
    try:
        o, info, v, w = P2.plan_seg(T_WH_b, T_HE, xs, vs, H, GOAL, fq, phi_b)
    except ValueError as e:
        out['cases'].append(dict(k=k_, sigma_s=sig_s, error=repr(e)[:200])); continue
    Xn, hn, adj = o.dense_reference(4)
    dX = np.diff(Xn, axis=0) / hn
    qd = dX[:, 3:9]
    qdd = np.diff(np.vstack([vs[3:9], qd]), axis=0) / hn
    sr = np.abs(qd + TAU * qdd)
    i_q, j_q = np.unravel_index(np.argmax(np.abs(qd)), qd.shape)
    i_s, j_s = np.unravel_index(np.argmax(sr), sr.shape)
    over_q = [(round(float(i * hn), 3), int(j) + 1, round(float(abs(qd[i, j])), 4)) for i, j in zip(*np.where(np.abs(qd) > V_ARM + 1e-4))]
    over_s = [(round(float(i * hn), 3), int(j) + 1, round(float(sr[i, j]), 4)) for i, j in zip(*np.where(sr > V_ARM + 1e-3))]
    c = dict(k=k_, sigma_s=sig_s, plan=dict(why=info.get('why'), iters=info.get('iters'), feasible=v['feasible'], failed=v['failed'], wall_s=round(w, 2), proj_adjust_max_rad=round(float(adj), 5)),
             seam_v0=dict(arm_qd_from_closure_velocity=[round(float(x), 4) for x in va], arm_qd_abs_max=round(float(np.abs(va).max()), 4), ref_central_diff_arm_qd=[round(float(x), 4) for x in vv[3:9]], closure_vs_ref_diff_max=round(float(np.abs(va - vv[3:9]).max()), 4),
                          base_v=[round(float(x), 4) for x in vv[:3]], door_rate_deg_s=round(math.degrees(float(vv[9])), 3)),
             new_segment_first_dense_arm_qd=[round(float(x), 4) for x in qd[0]], first_dense_minus_seam_v0_max=round(float(np.abs(qd[0] - va).max()), 4),
             arm_rate=dict(max=round(float(abs(qd[i_q, j_q])), 4), at_t=round(float(i_q * hn), 3), joint=int(j_q) + 1, limit=V_ARM, n_over=len(over_q), over_first=over_q[:6], over_t_range=[over_q[0][0], over_q[-1][0]] if over_q else None),
             setpoint_rate=dict(max=round(float(sr[i_s, j_s]), 4), at_t=round(float(i_s * hn), 3), joint=int(j_s) + 1, limit=V_ARM, n_over=len(over_s), over_first=over_s[:6], over_t_range=[over_s[0][0], over_s[-1][0]] if over_s else None,
                                qd_part=round(float(qd[i_s, j_s]), 4), tau_qdd_part=round(float(TAU * qdd[i_s, j_s]), 4)),
             n_dense=len(Xn), phi_end_deg=v.get('phi_end_deg'))
    out['cases'].append(c)
    print(json.dumps(c, ensure_ascii=False))
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print(out['first_segment_ref'])
