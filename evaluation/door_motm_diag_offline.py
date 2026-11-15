#!/usr/bin/env python3
"""m8house_door_motm_diag_1 取證紀錄的離線分析（Codex 20261115_135740）。只讀紀錄。
  A. 根連桿：z、側滾、俯仰全程變化量；覆寫前（A）／覆寫後（B）／物理步後（C）的 z 向線速度與側滾／俯仰角速度。
  B. 旋量座標：API 說明寫明 measured_joint_forces 在「該連桿進入關節的子側關節座標」；USD 讀回 joint6 localPos1＝0、localRot1＝單位 ⇒ link6 座標、作用點 link6 原點。
  C. 靜力對照：measured_joint_efforts（求解器回報的關節軸向力矩）對 −(G＋JᵀW_ext)；W_ext 由 joint6 旋量換到 TCP（含力臂項）。
  D. 非 PD 項：measured_joint_efforts − [kp(目標−q) − kd·q̇]（kp／kd 取執行器設定 600／60；語意：若 measured＝驅動＋關節摩擦等全部軸向力矩，此差＝非驅動軸向力矩）。
    python3 evaluation/door_motm_diag_offline.py OUT.json"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import door_body_model as BM
import door_open_exec_ctrl as XO
import door_motm_force_offline as FO
K = XO.K
D = os.path.join(HERE, 'runs', 'm8house_door_motm_diag_1')
m = json.load(open(os.path.join(D, 'diag_meta.json')))
X = np.load(os.path.join(D, 'diag_steps.npz'))['D']
off, k = {}, 0
for name, n in m['layout']:
    off[name.split('(')[0]] = (k, k + n); k += n
g = lambda nm: X[:, off[nm][0]:off[nm][1]]
t, mode = g('t')[:, 0], g('mode')[:, 0]
A, B, C = g('rootA_pos3_quat4_lin3_ang3'), g('rootB'), g('rootC')


def rpy(qw):
    w, x, y, z = qw.T
    return np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)), np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


rC, pC, yC = rpy(C[:, 3:7])
ib = int(np.argmax(mode > 0))
out = dict(t_bind=round(float(t[ib]) - 0.01, 3), n_rows=int(len(X)), n_err=m['n_err'])
seg = dict(pre_bind=mode == 0, after_bind=mode > 0)
out['root'] = {}
for nm_, s in seg.items():
    out['root'][nm_] = dict(z_range_mm=round(float(np.ptp(C[s, 2])) * 1e3, 5), roll_range_mrad=round(float(np.ptp(rC[s])) * 1e3, 5), pitch_range_mrad=round(float(np.ptp(pC[s])) * 1e3, 5),
                            A_pre_override_abs_max=dict(lin_z=round(float(np.abs(A[s, 9]).max()), 6), ang_x=round(float(np.abs(A[s, 10]).max()), 6), ang_y=round(float(np.abs(A[s, 11]).max()), 6)),
                            B_post_override_abs_max=dict(lin_z=round(float(np.abs(B[s, 9]).max()), 6), ang_x=round(float(np.abs(B[s, 10]).max()), 6), ang_y=round(float(np.abs(B[s, 11]).max()), 6)),
                            C_after_step_abs_max=dict(lin_z=round(float(np.abs(C[s, 9]).max()), 6), ang_x=round(float(np.abs(C[s, 10]).max()), 6), ang_y=round(float(np.abs(C[s, 11]).max()), 6)),
                            override_changed_xy_yaw_abs_max=dict(lin_x=round(float(np.abs(A[s, 7] - B[s, 7]).max()), 5), lin_y=round(float(np.abs(A[s, 8] - B[s, 8]).max()), 5), ang_z=round(float(np.abs(A[s, 12] - B[s, 12]).max()), 5)))
out['frame'] = dict(api_doc_excerpt=m['api_doc']['get_measured_joint_forces'][:260], usd_joint6={k_: v_ for k_, v_ in m['usd_joints'].items() if k_.endswith('joint6')}, j6_row=m['j6_row'], joint6_index=m['joint_indices']['joint6'],
                    fixed_joint={k_: v_ for k_, v_ in m['usd_joints'].items() if 'ideal_grasp' in k_})
names = m['dof_names']; ja = [names.index(f'joint{i}') for i in range(1, 7)]
jp, jv, eff, tg, F6 = g('joint_pos')[:, ja], g('joint_vel')[:, ja], g('measured_joint_efforts')[:, ja], g('position_targets')[:, ja], g('measured_joint_forces[j6]')
KP, KD = 600.0, 60.0
bodies = BM.arm_bodies(K)


def at(i):
    q9 = np.r_[C[i, 0], C[i, 1], yC[i], jp[i]]
    Fg, Mg, T6 = FO.down_gravity(q9)
    Fw, Mw = T6[:3, :3] @ F6[i, :3], T6[:3, :3] @ F6[i, 3:]
    Fe, Mo = -(Fw + Fg), -(Mw + Mg)
    Tt = K.fk(q9, 'link_tcp')
    Mt = Mo - np.cross(Tt[:3, 3] - T6[:3, 3], Fe)
    J = K.jacobian(q9, 'link_tcp')
    Gq = BM.arm_gravity(K, q9, bodies)
    pd = KP * (tg[i] - jp[i]) - KD * jv[i]
    return dict(Fe=Fe, Mt=Mt, static_rhs=-(Gq + J[:, 3:9].T @ np.r_[Fe, Mt]), eff=eff[i], pd=pd, G=Gq)


def win(a, b):
    I = [i for i in range(len(t)) if a <= t[i] <= b]
    E = [at(i) for i in I]
    mean = lambda k_: np.mean([e[k_] for e in E], axis=0)
    return dict(t=[round(a, 2), round(b, 2)], n=len(I), F_ext_world=np.round(mean('Fe'), 2).tolist(), M_ext_at_tcp_world=np.round(mean('Mt'), 2).tolist(),
                measured_joint_efforts=np.round(mean('eff'), 2).tolist(), minus_G_minus_JTW=np.round(mean('static_rhs'), 2).tolist(), statics_resid=np.round(mean('eff') - mean('static_rhs'), 2).tolist(),
                pd_torque=np.round(mean('pd'), 2).tolist(), non_pd_axis_torque=np.round(mean('eff') - mean('pd'), 2).tolist(), joint_rate_abs_max=round(float(np.abs(jv[I]).max()), 4))


tb = out['t_bind']
out['windows'] = dict(free_hold_0p2_0p4=win(0.2, 0.4), pre_bind_last_0p1=win(tb - 0.10, tb - 0.01), after_bind_first_tick=win(tb + 0.01, tb + 0.05), door_start=win(tb + 0.10, tb + 0.25), static_tail_2s=win(float(t[-1]) - 2.0, float(t[-1])))
out['urdf_arm_joint_dynamics'] = [l.strip() for l in open(os.path.join(HERE, 'models', 'omni_bot_manip.urdf')) if '<dynamics' in l]
out['semantics_note'] = 'measured_joint_efforts 的語意依 API 說明＝求解器在自由度方向回報的力矩；本檔以「靜止時 measured 是否等於 −(G＋JᵀW_ext)」檢驗它是否為軸向總力矩；非 PD 項的來源（關節摩擦／限位／其他）本檔不判定'
json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out['root'], ensure_ascii=False))
for k_, v_ in out['windows'].items():
    print(k_, json.dumps(v_, ensure_ascii=False))
print(out['urdf_arm_joint_dynamics'])
