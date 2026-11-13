#!/usr/bin/env python3
"""MotM 開門第一包（1）非零狀態交接契約（真入口：door_wbopt r5＋door_open_exec_ctrl r7）。
上游（移動中抓持保持）：底盤沿世界 −x 以 v 通過，手臂以「命令生效時刻預測底盤」的 IK 讓 TCP 停在綁定時的抓取位姿（門靜止於 φ_b），
  命令經 8 步管線＋拍內斜坡；穩態 1.2 s 後在拍內第 j0 物理步交接（j0∈{0,2,4}）。受控對象：底盤照管線命令積分；手臂一階模型，
  靜態偏置 d(q)＝D＋(G(q)−G(q_b))/kp（含重力隨構型）；門角由 TCP 反算（理想閉鏈幾何，同 test_door_open_exec.Fake）。
下游：規劃起點＝管線清空後的預測狀態（投影到閉鏈）、起始速度 v0＝（最後在途底盤命令的世界速度、閉鏈一致手臂速度、φ̇＝0）；
  執行層 r7：handover moving＝True、off_static＝上游靜態偏置估計（D＋重力項）。
核：交接前後設定點、實際送出命令、參考的連續性；起步閉鏈偏差、限制；對照組＝同一移動狀態但用舊假設（規劃從靜止起、錨定取 q−s、斜坡上一命令＝u_last）。
    python3 evaluation/test_door_moving_handover.py J0 V MODE OUT.json      MODE＝moving｜static_old"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_wbopt as WB
if os.environ.get('FAST_CLEAR', '1') == '1':
    import door_clear_door_fast as _F
    WB.clear_door = _F.clear_door_fast                                 # 等值快速版（逐構型核過）
import door_open_exec_ctrl as X
import door_body_model as BM
K, S, C0 = X.K, X.S, X.C0
PHYS, NPH, D_STEPS, DT = X.PHYS, X.NPH, X.D_STEPS, X.DT
j0, v, mode, out_path = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
RUN = os.path.join(HERE, 'runs', 'm8house_door_full_2')
P = np.load(os.path.join(RUN, 'plan.npz')); L = np.load(os.path.join(RUN, 'log.npz'))['rows']
T_WH_b, phi_b, T_HE = P['T_WH_bind'], float(P['phi_bind']), P['T_HE']
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json'))); al = np.asarray(ident['alpha'], float)
D = np.array(json.load(open(os.path.join(RUN, 'result.json')))['I_insert']['d_hat'])     # 名目靜態偏置（插入觀測）
E4 = L[L[:, 1] == 4][-1]; q_b = np.r_[E4[5:8], E4[8:14]]
FQ = 0.5 * (float(E4[3]) + float(E4[4]))
WB.set_model(phi0=phi_b, finger_q=FQ)
T_grasp = WB.handle_pose(T_WH_b, phi_b) @ T_HE
bodies = BM.arm_bodies(K); G_b = BM.arm_gravity(K, q_b, bodies); KP = 600.0
dq = lambda q9: D + (BM.arm_gravity(K, q9, bodies) - G_b) / KP       # noqa: E731
cfg = C0.make_cfg(ident); cfg.base_fixed = False; cfg.arm_model.bias = al * D
tau = PHYS / al
vw = np.array([-v, 0.0])                                             # 世界 −x


def ik(b, seed):
    q = seed.copy()
    for _ in range(40):
        q9 = np.r_[b, q]; Tc = K.fk(q9, 'link_tcp'); dR = T_grasp[:3, :3] @ Tc[:3, :3].T
        e = np.r_[T_grasp[:3, 3] - Tc[:3, 3], 0.5 * np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])]
        if np.linalg.norm(e) < 1e-11:
            break
        J = K.jacobian(q9, 'link_tcp')[:, 3:9]; q = q + np.linalg.solve(J.T @ J + 1e-12 * np.eye(6), J.T @ e)
    return q


# ---- 上游：移動中保持抓取位姿
T_pre = 1.2 + (j0 + 1) * PHYS
b0 = np.r_[q_b[:2] + np.array([0.03, 0.0]), q_b[2]]               # 上游自可抓窗口起點（綁定點 +x 0.03 m；door_motm_window_r1「沿門面 −x」[−0.03, 0.19]）出發
qa0 = ik(b0, q_b[3:])
q = np.r_[b0, qa0 + dq(np.r_[b0, qa0])]                              # 起點：手臂在靜態偏置平衡（未動）
s = qa0.copy(); pipe = [np.zeros(9) for _ in range(D_STEPS)]; u_last = np.zeros(9); u_prev = np.zeros(9); u_from = np.zeros(3); rj = NPH
q_ik_prev = None; log_up = []
n_total = int(round(T_pre / PHYS))
for k in range(n_total):
    if k % NPH == 0:
        qp, sp = q.copy(), s.copy()
        for u_in in pipe:
            qp, sp = S.plant_phys_step(qp, sp, u_in, cfg, 1)
        c_, s_ = math.cos(qp[2]), math.sin(qp[2])
        ub = np.array([c_ * vw[0] + s_ * vw[1], -s_ * vw[0] + c_ * vw[1], 0.0])
        a_ = min(1.0, k * PHYS / 0.3)                                    # 0.3 s 線性起步（上游自身的起步，不屬交接）
        ub = a_ * ub
        b_end = qp[:3] + np.r_[(a_ * vw) * DT, 0.0]
        q1 = ik(b_end, q_ik_prev if q_ik_prev is not None else qp[3:])
        q0_ = ik(qp[:3], q1)
        S1 = q1 + tau * (q1 - q0_) / DT - D
        u = np.r_[ub, (S1 - sp) / DT]
        u_prev, u_last = u_last.copy(), u
        q_ik_prev = q1
        u_from, rj = np.asarray(pipe[-1][:3], float).copy(), 0
    cmd = X.ramp_cmd(u_from, u_last, rj); rj += 1
    pipe.append(cmd); ua = pipe.pop(0)
    s = s + PHYS * ua[3:]
    cs, sn = math.cos(q[2]), math.sin(q[2])
    q[:3] += PHYS * np.array([cs * ua[0] - sn * ua[1], sn * ua[0] + cs * ua[1], ua[2]])
    q[3:] = q[3:] + al * (s - q[3:] + dq(q))
    log_up.append(np.r_[k * PHYS, q, s, ua])
    if k % NPH == 0:
        cc_ = WB.clear_door(np.r_[q, phi_b], T_WH_b)
        up_min = min(globals().get('up_min', np.inf), min(cc_[k_] - WB.THR[k_] for k_ in ('env', 'base', 'self')))
        up_sig = min(globals().get('up_sig', np.inf), WB.sigma_min(np.r_[q, phi_b]))
assert (n_total - 1) % NPH == j0 or True
phase_in_tick = (n_total) % NPH                                        # 交接時本拍已過的物理步數
Tc_h = K.fk(q, 'link_tcp')
up_err_mm = float(np.linalg.norm(Tc_h[:3, 3] - T_grasp[:3, 3])) * 1e3
up_rate_max = float(np.abs(np.diff(Lu_tmp := np.array(log_up)[-60:, 10:16], axis=0)).max() / PHYS)
if up_err_mm > 5.0 or up_rate_max > 1.0 + 1e-6:                        # 上游自己保持不住（例：該速度所需關節速率超過上限）⇒ 交接核對不適用
    json.dump(dict(j0=j0, v=v, mode=mode, upstream_infeasible=True, tcp_err_at_handover_mm=round(up_err_mm, 3), upstream_setpoint_rate_max=round(up_rate_max, 4)), open(out_path, 'w'), ensure_ascii=False, indent=1)
    print('upstream_infeasible', round(up_err_mm, 3), round(up_rate_max, 4)); sys.exit(0)
qp, sp = q.copy(), s.copy()
for u_in in pipe:
    qp, sp = S.plant_phys_step(qp, sp, u_in, cfg, 1)
c_, s_ = math.cos(qp[2]), math.sin(qp[2])
ulast_b = np.asarray(pipe[-1][:3], float)
vbase = np.array([c_ * ulast_b[0] - s_ * ulast_b[1], s_ * ulast_b[0] + c_ * ulast_b[1], ulast_b[2]])
o_tmp = WB.DoorWBOpt(T_WH_b, np.eye(4), np.r_[qp, phi_b], math.radians(85.0), N=10, dt=0.2, T_HE=T_HE, project_start=True)
x0 = o_tmp.x0.copy()
proj = float(np.abs(x0[3:9] - qp[3:]).max())
qd0 = o_tmp.closure_velocity(x0, vbase, 0.0)
v0 = np.r_[vbase, qd0, 0.0] if mode == 'moving' else None
G = math.radians(85.0); b = x0[:3]
Tr = 1.875 * (G - x0[9]) * (float(np.hypot(b[0], b[1])) + WB.L_W) / (0.9 * WB.W_LIM)
o = WB.DoorWBOpt(T_WH_b, np.eye(4), x0, G, N=int(math.ceil((Tr + 2.0) / 0.2)), dt=0.2, T_HE=T_HE, project_start=False, v0=v0)
o.init_corotate(Tr); info = o.solve(max_iter=45); ver = o.verify(); Xd, h, _ = o.dense_reference(4)
off_static = dq(q) if mode == 'moving' else None
ho = dict(s=s.copy(), pipe=[u_.copy() for u_ in pipe], u_last=u_last.copy(), u_prev=u_prev.copy(), bias=al * D, moving=(mode == 'moving'), off_static=off_static)
c = X.DoorOpenExec(ident, Xd, h, T_WH_b, phi_b, T_HE, G, ho, FQ, arm_mode='closure')
# ---- 下游：同一受控對象續跑
r_h = float(np.hypot(*(WB.handle_pose(T_WH_b, phi_b)[:2, 3]))); z_h = float(WB.handle_pose(T_WH_b, phi_b)[2, 3])
log_dn, t = [], 0.0
seq = []                                                             # 交接後每物理步：設定點、送出的底盤與手臂命令（對照上游最後幾步）
s_prev = s.copy(); ua_prev = log_up[-1][16:25].copy()
jumps = dict(setpoint=0.0, base_cmd=0.0, arm_cmd=0.0)
T_RUN = float(os.environ.get('T_RUN', '3.0'))
while t < T_RUN:
    phi, pH = c.phi_of(q)
    out = c.step(t, q.copy(), phi, None)
    if out.get('abort'):
        break
    ua = np.r_[out['base_twist'], (out['arm_target'] - s_prev) / PHYS]
    jumps['setpoint'] = max(jumps['setpoint'], float(np.abs(out['arm_target'] - s_prev).max()))
    jumps['base_cmd'] = max(jumps['base_cmd'], float(np.abs(ua[:3] - ua_prev[:3]).max()))
    jumps['arm_cmd'] = max(jumps['arm_cmd'], float(np.abs(ua[3:] - ua_prev[3:]).max()))
    s_prev, ua_prev = out['arm_target'].copy(), ua.copy()
    seq.append(np.r_[out['arm_target'], ua])
    cs, sn = math.cos(q[2]), math.sin(q[2])
    q[:3] += PHYS * np.array([cs * out['base_twist'][0] - sn * out['base_twist'][1], sn * out['base_twist'][0] + cs * out['base_twist'][1], out['base_twist'][2]])
    q[3:] = q[3:] + al * (out['arm_target'] - q[3:] + dq(q))
    t += PHYS
    phi2, pH2 = c.phi_of(q)
    ep, er = c.closure_at(q, phi2)
    log_dn.append([t, float(np.hypot(pH2[0], pH2[1])) - r_h, float(pH2[2]) - z_h, er, math.degrees(phi2), *out['base_twist']])
Lu, Ld = np.array(log_up), np.array(log_dn)
# 上游穩態的同類量（每步變化）作為「正常」尺度
up_tail = Lu[-60:]
norm_set = float(np.abs(np.diff(up_tail[:, 10:16], axis=0)).max())
norm_base = float(np.abs(np.diff(up_tail[:, 16:19], axis=0)).max())
rep = c.report()
# 交接邊界（上游最後 10 步＋下游前 20 步）的逐步變化：設定點變化率的二階差（＝命令跳動）、底盤命令跳動
Sq = np.vstack([np.c_[Lu[-10:, 10:16], Lu[-10:, 16:25]], np.array(seq[:20])])
ucmd = np.diff(Sq[:, :6], axis=0) / PHYS                              # 每物理步實際生效的手臂設定點速率（命令每拍分段常數）
dd_set = np.abs(np.diff(ucmd, axis=0)).max(axis=1) / DT                 # 相鄰命令差／DT（與 A_ARM 同語意：每拍一次換命令）
dbc = np.abs(np.diff(Sq[:, 6:9], axis=0)) / PHYS                     # 底盤命令逐物理步變化（本體 vx, vy, ω；拍內斜坡）
db_ratio = (dbc / np.asarray(X.B.A_BASE)[None]).max(axis=1)
db = dbc.max(axis=1)
up_dd = float(np.abs(np.diff(np.diff(Lu[-60:-10, 10:16], axis=0) / PHYS, axis=0)).max() / DT)
boundary = dict(arm_cmd_change_per_DT_max=round(float(dd_set.max()), 3), arm_cmd_change_at_index=int(dd_set.argmax()) - 8, arm_cmd_change_upstream_steady_max=round(up_dd, 3), arm_cmd_change_limit=C0.A_ARM,
                base_cmd_rate_max_at_boundary=round(float(db.max()), 4), base_cmd_rate_at_index=int(db.argmax()) - 9, base_acc_limit=list(X.B.A_BASE),
                base_acc_component_max=[round(float(x_), 4) for x_ in dbc.max(axis=0)], base_acc_ratio_max=round(float(db_ratio.max()), 4),
                note='索引 0＝交接後第一個物理步；設定點二階差反映手臂命令跳動')
tw = Ld[:, 5:8]
acc = float(np.abs(np.diff(tw, axis=0)).max() / PHYS)
res = dict(j0=j0, v=v, mode=mode, phase_in_tick_at_handover=phase_in_tick, upstream=dict(tcp_err_at_handover_mm=round(up_err_mm, 4), setpoint_rate_max=round(up_rate_max, 4), clear_margin_min_mm=round(float(up_min) * 1e3, 2), sigma_min=round(float(up_sig), 4), base_at_handover=q[:3].round(4).tolist(), base_vs_bind_m=round(float(np.linalg.norm(q[:2] - q_b[:2])), 4)), boundary=boundary, proj_to_closure_rad=proj, v0=None if v0 is None else np.round(v0, 5).tolist(),
           plan=dict(why=info['why'], wall_s=info['wall_s'], feasible=ver['feasible'], failed=ver['failed'], start_state_err=ver['start_state_err'], start_vel=ver['start_vel'], body_acc_max=ver['body_acc_max']),
           continuity=dict(setpoint_step_max_rad=round(jumps['setpoint'], 6), setpoint_step_upstream_max=round(norm_set, 6), base_cmd_step_max=round(jumps['base_cmd'], 5), base_cmd_step_upstream_max=round(norm_base, 5), arm_cmd_step_max=round(jumps['arm_cmd'], 4)),
           startup=dict(radial_max_mm_0p5s=round(float(np.abs(Ld[Ld[:, 0] <= 0.5, 1]).max()) * 1e3, 3), height_max_mm_0p5s=round(float(np.abs(Ld[Ld[:, 0] <= 0.5, 2]).max()) * 1e3, 3), rot_max_deg_0p5s=round(float(Ld[Ld[:, 0] <= 0.5, 3].max()), 4),
                        radial_max_mm_all=round(float(np.abs(Ld[:, 1]).max()) * 1e3, 3), phi_end_deg=round(float(Ld[-1, 4]), 3), base_acc_max=round(acc, 3)),
           exec=dict(stop=rep['stop'], abort=c.abort, closure_stats={k_: v_ for k_, v_ in rep['closure_stats'].items() if k_ != 'fails'}, anchor_source=(rep.get('closure_anchor') or {}).get('off_source')))
json.dump(res, open(out_path, 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k_: res[k_] for k_ in ('j0', 'v', 'mode', 'phase_in_tick_at_handover', 'boundary', 'startup', 'exec')}, ensure_ascii=False, default=float))
print(json.dumps(res['plan'], ensure_ascii=False, default=float))
