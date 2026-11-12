#!/usr/bin/env python3
"""開門後退出→收臂→繞門扇→穿門的假世界整段（door_exit_ctrl 真入口）。
受控對象：底盤照管線輸出的本體速度積分（理想底盤介面）；手臂一階設定點模型，靜態偏置隨構型＝起點實測 (q₀−s₀)＋URDF 重力變化／kp
（與控制器同一個重力實作——同模型比較，揭露）；門＝獨立被動狀態 φ(t)＝φ₀＋ω₀/c_eff·(1−e^{−c_eff t})（資產推得 c_eff）；兩指開度固定在實測張開值。
起點＝第三趟實錄張爪觀察末（log.npz）；開門段最後設定點＝open_steps.npz ctrl 末列；路徑＝door_pass_check r2 的 A* 路徑點（含登錄未知物真值，只是幾何／運動學串接，不是導航成果）。
案例：nominal；forced_X1／X1b／X2／X3turn／X4（各段中途外部停止）；door_push（門被外力推出帶外）。
    python3 evaluation/test_door_exit_fake.py CASE PASS_JSON OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_exit_ctrl as XE
import door_body_model as BM
case, pass_json, out = sys.argv[1], sys.argv[2], sys.argv[3]
RUN = os.path.join(HERE, 'runs', 'm8house_door_open_3')
L = np.load(os.path.join(RUN, 'log.npz')); R = L['rows']
Z = np.load(os.path.join(RUN, 'open_steps.npz'))
P = np.load(os.path.join(RUN, 'plan.npz'))
ident = json.load(open(os.path.join(HERE, 'results', 'wgmpc_arm_sp_ident_free4.json')))
pj = json.load(open(pass_json))
Rr = R[R[:, 1] == 7]
q0 = np.r_[Rr[-1, 24:27], Rr[-1, 16:22]]
phi0 = float(Rr[-1, 2])
omega0 = float((Rr[-1, 2] - Rr[-11, 2]) / 0.1)
fq = 0.5 * (float(Rr[-1, 3]) + float(Rr[-1, 4]))
s0 = Z['ctrl'][-1, 32:38].copy()
al = np.asarray(ident['alpha'], float)
d0 = q0[3:] - s0                                                     # 起點實測靜態偏置（q−s）
path = np.vstack([np.array(sg['path_xy']) for sg in pj['segments']])
path = np.vstack([path, np.array(pj['P2'])[None]])
Q_STOW = np.array([-0.013, -0.052, 0.086, 0.012, -1.625, -0.002])
c = XE.DoorExitCtrl(ident, P['T_WH_bind'], float(P['phi_bind']), s0, [np.zeros(9)] * XE.D_STEPS, np.zeros(9), np.zeros(9), al * d0, fq, phi0, omega0, path[1:], Q_STOW)
K = XE.K
bodies = BM.arm_bodies(K)
G0 = BM.arm_gravity(K, q0, bodies)
q = q0.copy(); t = 0.0
log = []
phi_push = 0.0
while t < 90.0:
    phi = phi0 + omega0 / XE.C_EFF * (1.0 - math.exp(-XE.C_EFF * t)) + phi_push
    if case == 'door_push' and c.phase == 'X1b' and t - c.t_phase > 0.5:
        phi_push = min(phi_push + math.radians(0.05) * 0.01 / 0.01, math.radians(0.6))   # 外力把門再推開（每步 0.05°，到 0.6° 為止）
    o = c.step(t, q.copy(), phi)
    tgt = {'forced_X1': 'X1', 'forced_X1b': 'X1b', 'forced_X2': 'X2', 'forced_X3turn': 'X3X4', 'forced_X4': 'X3X4'}.get(case)
    if tgt and c.stop is None and c.phase == tgt:
        el = t - c.t_phase
        if (case == 'forced_X3turn' and c.nav and c.nav['mode'] == 'turn' and el > 0.6) or (case == 'forced_X4' and c.nav and c.nav['mode'] == 'move' and q[1] > -0.4) or (case not in ('forced_X3turn', 'forced_X4') and el > 0.5):
            c._stop('forced_stop')
    cs, sn = math.cos(q[2]), math.sin(q[2])
    vx, vy, wz = o['base_twist']
    q[:3] += 0.01 * np.array([cs * vx - sn * vy, sn * vx + cs * vy, wz])
    d = d0 + (BM.arm_gravity(K, q, bodies) - G0) / 600.0
    q[3:] = q[3:] + al * (o['arm_target'] - q[3:] + d)
    t += 0.01
    if c.n_step % XE.NPH == 0:
        log.append([t, *q, phi])
    if c.stop is not None and c.stop_ok is not None:
        break
rep = c.report()
Lg = np.array(log)
expect = 'exit_pass_done' if case == 'nominal' else ('door_out_of_band' if case == 'door_push' else 'forced_stop')
P2 = np.array(pj['P2'])
ck = dict(stop_reason=bool(rep['stop'] and rep['stop']['why'] == expect), stop_effective=bool(rep['stop_effect'] and rep['stop_effect']['ok']),
          clear_ok=bool(all(rep['min_clear'][k] >= XE.W.THR[k] for k in XE.KEYS)), finger_after_release_ok=bool(rep['min_clear']['finger_bar_after_release'] >= XE.W.THR['env'] or rep['finger_release_t'] is None),
          limits_ok=bool(rep['limits']['arm_rate'] <= 1.0 + 1e-6 and rep['limits']['arm_acc'] <= XE.C0.A_ARM + 1e-6 and rep['limits']['wheel_v'] <= XE.LIM.W_LIM + 1e-6 and rep['limits']['wheel_a'] <= XE.LIM.WA_LIM + 1e-6))
if case == 'nominal':
    ck.update(done=bool(rep['done']), at_P2=bool(np.linalg.norm(q[:2] - P2) < 0.03), yaw_90=bool(abs(math.degrees(q[2]) - 90.0) < 2.0), arm_stowed=bool(np.abs(q[3:] - Q_STOW).max() < 0.03), finger_released=rep['finger_release_t'] is not None)
res = dict(case=case, expect=expect, checks=ck, pass_=bool(all(ck.values())), report=rep, final=dict(base=q[:3].round(4).tolist(), yaw_deg=round(math.degrees(q[2]), 2), arm=q[3:].round(4).tolist(), t=round(t, 2)),
           sources=dict(start='runs/m8house_door_open_3/log.npz 張爪觀察末列', setpoint='open_steps.npz ctrl 末列', handle='runs/m8house_door_open_3/plan.npz', path=os.path.basename(pass_json) + '（A* 理想跟隨；含登錄未知物真值，只是幾何／運動學串接）', stow='m8house_da_online_1 起點手臂'),
           omega0_dps=round(math.degrees(omega0), 4), phi0_deg=round(math.degrees(phi0), 4))
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
np.save(out.replace('.json', '_log.npy'), Lg)
print(case, 'PASS' if res['pass_'] else 'FAIL', {k: v for k, v in ck.items() if not v}, rep['stop'], rep['phases'], rep['min_clear'], rep['limits'], res['final'])
