#!/usr/bin/env python3
"""真入口承接核對：DoorOpenExec（arm_mode='closure'，假世界）開門 → 正常停穩 → 解綁 → 張爪 → 以**開門執行層當下的**設定點、管線、上一命令、偏置建立 DoorExitCtrl
（不重置、不回填舊紀錄）→ X1→X1b→X2（首版物理範圍，到收臂為止）。
核：承接值逐值相同（設定點、管線、上一命令）；退出第一拍命令由承接狀態起算；整段淨空／限制／停止有效；中間無物理跳躍。
受控對象沿用 test_door_open_exec.Fake（開門段；理想閉鏈幾何、一階手臂）＋解綁後門自由阻尼（Fake 既有）；退出段手臂同一階模型、門＝Fake 的自由門角。
    python3 evaluation/test_door_open_exit_handover.py PLAN.npz PASS_JSON OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_door_open_exec as T
import door_exit_ctrl as XE
X = T.X
P = np.load(sys.argv[1]); pj = json.load(open(sys.argv[2]))
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
q0 = P['q_meas'].copy(); al = np.asarray(ident['alpha'], float)
c = X.DoorOpenExec(ident, P['Xd'], float(P['h']), P['T_WH_bind'], float(P['phi_bind']), P['T_HE'], T.PHI_GOAL, dict(s=q0[3:] - T.D_TRUE, pipe=[np.zeros(9)] * 8, u_last=np.zeros(9), u_prev=np.zeros(9), bias=al * T.D_TRUE), T.FINGER_Q, arm_mode='closure')
F = T.Fake(c, q0)
F.run(t_end=40.0)
rep_o = c.report()
assert rep_o['done'] and rep_o['stop_effect']['ok'], rep_o['stop']
# 解綁（執行器語意：停穩後解除、門自由；手指張開——Fake 視為立即）
phi_u, _ = F.door()
om = (phi_u - F.phi_hist[-1]) / T.PHYS if F.phi_hist else 0.0
F.bound, F.phi_free, F.omega_free = False, float(phi_u), float(om)
c.confirm_unbind(True, detail=dict(t=round(F.t, 3)))
# 張爪觀察 0.5 s：手臂保持（設定點不動）、底盤零、門自由
for _ in range(50):
    F.q[3:] = F.q[3:] + F.alpha * (c.s - F.q[3:] + T.D_TRUE)
    F.omega_free *= math.exp(-T.I_DAMP * T.PHYS)
    F.phi_free = min(max(F.phi_free + F.omega_free * T.PHYS, 0.0), math.radians(110.0))
    F.t += T.PHYS
ho = dict(s=c.s.copy(), pipe=[u.copy() for u in c.pipe], u_last=c.u_last.copy(), u_prev=c.u_prev.copy(), bias=np.asarray(c.cfg.arm_model.bias, float).copy())
# 退出段：承接開門執行層當下狀態
phi_now = F.phi_free
x = XE.DoorExitCtrl(ident, P['T_WH_bind'], float(P['phi_bind']), ho['s'], ho['pipe'], ho['u_last'], ho['u_prev'], ho['bias'], XE.FINGER_OPEN, phi_now, F.omega_free, np.array(pj['segments'][0]['path_xy'])[1:], np.array([-0.013, -0.052, 0.086, 0.012, -1.625, -0.002]))
same = dict(setpoint=bool(np.array_equal(x.s, ho['s'])), pipe=bool(all(np.array_equal(a, b) for a, b in zip(x.pipe, ho['pipe']))), u_last=bool(np.array_equal(x.u_last, ho['u_last'])), u_prev=bool(np.array_equal(x.u_prev, ho['u_prev'])))
q = F.q.copy(); t0 = F.t; t = 0.0; first = None; jump = 0.0; q_prev = q.copy()
while t < 30.0:
    o = x.step(t, q.copy(), F.phi_free)
    if first is None:
        first = dict(arm_target_minus_handover_s=float(np.abs(o['arm_target'] - ho['s']).max()), base_twist=o['base_twist'].tolist())
    if o.get('abort'):
        break
    cs, sn = math.cos(q[2]), math.sin(q[2])
    vx, vy, wz = o['base_twist']
    q[:3] += 0.01 * np.array([cs * vx - sn * vy, sn * vx + cs * vy, wz])
    q[3:] = q[3:] + F.alpha * (o['arm_target'] - q[3:] + T.D_TRUE)
    F.omega_free *= math.exp(-T.I_DAMP * T.PHYS)
    F.phi_free = min(max(F.phi_free + F.omega_free * T.PHYS, 0.0), math.radians(110.0))
    jump = max(jump, float(np.abs(q - q_prev).max())); q_prev = q.copy()
    t += 0.01
    if x.phase == 'X3X4' and x.stop is None:                         # 首版物理範圍到 X2：進入 X3X4 即外部停止（不跑運動學跟隨）
        x._stop('scope_end_after_X2')
    if x.stop is not None and x.stop_ok is not None:
        break
rx = x.report()
ck = dict(open_done=bool(rep_o['done']), handover_values_identical=all(same.values()), first_cmd_from_handover=bool(first['arm_target_minus_handover_s'] < 0.01 * 1.0 + 1e-9),
          reached_X2_end=bool(any(p[0] == 'X3X4' for p in rx['phases'])), stop_reason=bool(rx['stop'] and rx['stop']['why'] == 'scope_end_after_X2'), stop_effective=bool(rx['stop_effect'] and rx['stop_effect'].get('ok')),
          no_abort=rx['abort'] is None, clear_ok=bool(all(rx['min_clear'][k] >= XE.W.THR[k] for k in XE.KEYS)), no_phys_jump=bool(jump < 0.01))
out = dict(checks=ck, pass_=bool(all(ck.values())), handover_same=same, first_exit_step=first, open_report=dict(stop=rep_o['stop'], t_end=rep_o['t_end'], phi_end_deg=rep_o['phi_end_deg']), exit_report=rx,
           note='假世界；開門段沿用 Fake（理想閉鏈幾何、一階手臂、常數偏置 D_TRUE）；解綁與張爪以 Fake 語意（手指立即張開、門自由阻尼）；退出段同受控對象')
json.dump(out, open(sys.argv[3], 'w'), ensure_ascii=False, indent=1, default=float)
print('PASS' if out['pass_'] else 'FAIL', {k: v for k, v in ck.items() if not v}, same, first, rx['phases'], rx['min_clear'], rx['stop'])
