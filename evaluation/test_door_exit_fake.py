#!/usr/bin/env python3
"""開門後退出→收臂→繞門扇→穿門的假世界整段（door_exit_ctrl 真入口）。
受控對象：底盤照管線輸出的本體速度積分（理想底盤介面）；手臂一階設定點模型，靜態偏置隨構型＝起點實測 (q₀−s₀)＋URDF 重力變化／kp
（與控制器同一個重力實作——同模型比較，揭露）；門＝獨立被動狀態 φ(t)＝φ₀＋ω₀/c_eff·(1−e^{−c_eff t})（資產推得 c_eff）；兩指開度固定在實測張開值。
起點＝第三趟實錄張爪觀察末（log.npz）；開門段最後設定點＝open_steps.npz ctrl 末列；路徑＝door_pass_check r2 的 A* 路徑點（含登錄未知物真值，只是幾何／運動學串接，不是導航成果）。
案例：nominal；forced_X1／X1b／X2／X3turn／X4（各段中途外部停止）；door_push（門被外力推出帶外 ⇒ 以當下狀態重建帶再核停止）；
      r2 反例：finger_pen_start（起點兩指已穿入橫桿 ⇒ 不送命令即停）、finger_slow（X1 中底盤被外力緩慢推向橫桿 ⇒ 間隙以至今最大為基準判惡化）、
      finger_reapproach（例外撤除後 X1b 中被推回橫桿 ⇒ 門檻違反）、push_during_stop（外部停止後再被推 ⇒ 停止中違反升級模擬中止）。
    python3 evaluation/test_door_exit_fake.py CASE PASS_JSON OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_exit_ctrl as XE
import door_body_model as BM
NPH_ = XE.NPH
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
if case == 'closed_X2':                                              # 閉門約 0°（冒煙幾何）：第三趟綁定靜置末的實測狀態、張爪、不綁；承接插入控制器設定點
    Re = R[R[:, 1] == 4]
    q0 = np.r_[Re[-1, 24:27], Re[-1, 16:22]]
    phi0, omega0, fq = float(Re[-1, 2]), 0.0, XE.FINGER_OPEN
    s0 = q0[3:] + np.array(json.load(open(os.path.join(RUN, 'result.json')))['handover']['setpoint_minus_measured'])
al = np.asarray(ident['alpha'], float)
d0 = q0[3:] - s0                                                     # 起點實測靜態偏置（q−s）
path = np.vstack([np.array(sg['path_xy']) for sg in pj['segments']])
path = np.vstack([path, np.array(pj['P2'])[None]])
Q_STOW = np.array([-0.013, -0.052, 0.086, 0.012, -1.625, -0.002])
c = XE.DoorExitCtrl(ident, P['T_WH_bind'], float(P['phi_bind']), s0, [np.zeros(9)] * XE.D_STEPS, np.zeros(9), np.zeros(9), al * d0, fq, phi0, omega0, path[1:], Q_STOW, scope_end=('X2' if case in ('closed_X2', 'open_X2') else None))
K = XE.K
bodies = BM.arm_bodies(K)
G0 = BM.arm_gravity(K, q0, bodies)
q = q0.copy(); t = 0.0
log = []
phi_push = 0.0
phi_off = 0.0
if case == 'finger_pen_start':                                       # 起點門角偏移使橫桿穿入一指（同一偏移也給控制器當起點門角：帶本身成立）
    for dd in np.linspace(0.05, 0.6, 12):
        for sg in (-1.0, 1.0):
            if c.finger_bar(q0, phi0 + sg * math.radians(dd)) < 0.0:
                phi_off = sg * math.radians(dd); break
        if phi_off:
            break
    c.phi0 = phi0 + phi_off
tool_z = XE.K.fk(q0, 'link_tcp')[:3, 2]                              # 工具接近軸（朝橫桿）
push = np.zeros(2)
n_phys_after_abort, aborted = 0, False
while t < 90.0:
    phi = phi0 + phi_off + omega0 / XE.C_EFF * (1.0 - math.exp(-XE.C_EFF * t)) + phi_push
    if case == 'door_push' and c.phase == 'X1b' and t - c.t_phase > 0.5:
        phi_push = min(phi_push + math.radians(0.05), math.radians(0.6))   # 外力把門再推開（每步 0.05°，到 0.6° 為止）
    if case == 'finger_slow' and c.stop is not None and c.stop['why'] == 'forced_stop' and c.finger_exempt:
        push = push + 1.5e-5 * tool_z[:2] / max(np.linalg.norm(tool_z[:2]), 1e-9)   # X1 例外期間外部停止後，每步 0.015 mm（每拍 0.075 mm，小於單拍容差 0.1 mm）推向橫桿：只有「對至今最大間隙」才抓得到

    if case == 'push_during_stop' and c.stop is not None and c.stop['why'] == 'forced_stop':
        push = push + 1.0e-3 * tool_z[:2] / max(np.linalg.norm(tool_z[:2]), 1e-9)   # 停止中每步 1 mm 推向橫桿
    t_before = t
    o = c.step(t, q.copy(), phi)
    if o.get('abort'):                                               # 模擬中止：不再步進物理
        aborted = True
        o2 = c.step(t, q.copy(), phi)
        n_phys_after_abort += int(not o2.get('abort'))
        break
    tgt = {'forced_X1': 'X1', 'forced_X1b': 'X1b', 'forced_X2': 'X2', 'forced_X3turn': 'X3X4', 'forced_X4': 'X3X4', 'push_during_stop': 'X1', 'finger_slow': 'X1'}.get(case)
    if tgt and c.stop is None and c.phase == tgt:
        el = t - c.t_phase
        if (case == 'forced_X3turn' and c.nav and c.nav['mode'] == 'turn' and el > 0.6) or (case == 'forced_X4' and c.nav and c.nav['mode'] == 'move' and q[1] > -0.4) or (case == 'finger_slow' and el >= 0.0) or (case not in ('forced_X3turn', 'forced_X4', 'finger_slow') and el > (0.8 if case == 'push_during_stop' else 0.5)):
            c._stop('forced_stop')
    if case == 'finger_reapproach' and c.phase == 'X1' and c._ref_done and c.finger_release_t is not None and c.stop is None:
        push = push + 2.0e-4 * tool_z[:2] / max(np.linalg.norm(tool_z[:2]), 1e-9)   # 例外撤除後、X1 參考走完停穩前，每步 0.2 mm 推回橫桿（控制器一停就不再推）
    cs, sn = math.cos(q[2]), math.sin(q[2])
    vx, vy, wz = o['base_twist']
    q[:3] += 0.01 * np.array([cs * vx - sn * vy, sn * vx + cs * vy, wz])
    if np.any(push):
        q[:2] += push; push = np.zeros(2) if case in ('finger_slow', 'finger_reapproach', 'push_during_stop') else push
    d = d0 + (BM.arm_gravity(K, q, bodies) - G0) / 600.0
    q[3:] = q[3:] + al * (o['arm_target'] - q[3:] + d)
    t += 0.01
    if c.n_step % XE.NPH == 0:
        log.append([t, *q, phi])
    if c.stop is not None and c.stop_ok is not None:
        break
rep = c.report()
Lg = np.array(log)
expect = dict(closed_X2='scope_end_after_X2', open_X2='scope_end_after_X2', nominal='exit_pass_done', door_push='door_out_of_band', finger_pen_start='finger_penetration', finger_slow='forced_stop', finger_reapproach='finger_bar', push_during_stop='forced_stop').get(case, 'forced_stop')
P2 = np.array(pj['P2'])
if case in ('push_during_stop', 'finger_slow'):
    want = ('clearance', 'finger_bar') if case == 'push_during_stop' else ('finger_gap_worsening',)
    ck = dict(stop_reason=bool(rep['stop'] and rep['stop']['why'] == expect), escalated_abort=bool(rep['abort'] is not None and rep['abort']['why'] == 'violation_during_stop' and any(w in rep['abort']['viol'] for w in want)),
              no_physics_after_abort=bool(aborted and n_phys_after_abort == 0), stop_not_success=bool(rep['stop_effect'] and rep['stop_effect'].get('ok') is None))
else:
  ck = dict(stop_reason=bool(rep['stop'] and rep['stop']['why'] == expect), stop_effective=bool(rep['stop_effect'] and rep['stop_effect']['ok']), not_aborted=rep['abort'] is None,
          clear_ok=bool(case in ('finger_reapproach', 'finger_pen_start') or all(rep['min_clear'][k] >= XE.W.THR[k] for k in XE.KEYS)), finger_after_release_ok=bool(case == 'finger_reapproach' or rep['min_clear']['finger_bar_after_release'] >= XE.W.THR['env'] or rep['finger_release_t'] is None),
          limits_ok=bool(rep['limits']['arm_rate'] <= 1.0 + 1e-6 and rep['limits']['arm_acc'] <= XE.C0.A_ARM + 1e-6 and rep['limits']['wheel_v'] <= XE.LIM.W_LIM + 1e-6 and rep['limits']['wheel_a'] <= XE.LIM.WA_LIM + 1e-6))
if case in ('closed_X2', 'open_X2'):
    ck.update(no_x3x4_cmd=bool(rep['x3x4_cmd_ticks'] == 0 and not any(ph == 'X3X4' or ph == 'STOP:X3X4' for _, ph, _u in c.cmd_log)), reached_X2_done=bool(any(e['what'] == 'phase_done' and e['phase'] == 'X2' for e in rep['events'])),
              arm_stowed=bool(np.abs(q[3:] - Q_STOW).max() < 0.03), finger_released=rep['finger_release_t'] is not None)
if case == 'nominal':
    st = c.sweep_timing
    def kd(tag, e):
        v = [x['k_decel'] for x in st if x['tag'] == tag and x['extra'] == e]
        return sorted(set(v))
    timing = dict(stop_now=kd('stop_now', 0), candidate_e1=kd('candidate', 1), candidate_e2=kd('candidate', 2))
    ck.update(sweep_timing_distinct=bool(timing['stop_now'] == [0] and timing['candidate_e1'] == [NPH_] and timing['candidate_e2'] == [2 * NPH_]))
    ck.update(done=bool(rep['done']), at_P2=bool(np.linalg.norm(q[:2] - P2) < 0.03), yaw_90=bool(abs(math.degrees(q[2]) - 90.0) < 2.0), arm_stowed=bool(np.abs(q[3:] - Q_STOW).max() < 0.03), finger_released=rep['finger_release_t'] is not None)
res = dict(case=case, expect=expect, checks=ck, pass_=bool(all(ck.values())), report=rep, final=dict(base=q[:3].round(4).tolist(), yaw_deg=round(math.degrees(q[2]), 2), arm=q[3:].round(4).tolist(), t=round(t, 2)),
           sources=dict(start='runs/m8house_door_open_3/log.npz 張爪觀察末列', setpoint='open_steps.npz ctrl 末列', handle='runs/m8house_door_open_3/plan.npz', path=os.path.basename(pass_json) + '（A* 理想跟隨；含登錄未知物真值，只是幾何／運動學串接）', stow='m8house_da_online_1 起點手臂'),
           omega0_dps=round(math.degrees(omega0), 4), phi0_deg=round(math.degrees(phi0), 4))
if case == 'nominal':
    res['sweep_timing'] = timing
if case == 'finger_pen_start':
    res['phi_offset_deg'] = round(math.degrees(phi_off), 3)
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1, default=float)
np.save(out.replace('.json', '_log.npy'), Lg)
print(case, 'PASS' if res['pass_'] else 'FAIL', {k: v for k, v in ck.items() if not v}, rep['stop'], rep['phases'], rep['min_clear'], rep['limits'], res['final'])
