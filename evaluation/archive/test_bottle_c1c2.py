#!/usr/bin/env python3
"""C1（motm_b1_port）與 C2（bottle_models）測試：
  F1 參考前饋的線速度＝min-jerk 解析導數（INSERT 段，直線 Δp·ṡ(τ)）；F2 角速度為世界座標：R(t+h) ≈ exp([ω]h)·R(t)；
  F3 時間索引：ref_twist(t) 對應 target_at(t)（與 t±dt 不同）；
  V1 C1 速度框用 vbox：base_box_scale 2 時底盤可達 > 舊框（0.035255），且不超 vbox；
  V2 無誤差、只有前饋且限制不作用時，J_b·u ≈ v_ff（世界座標，未多轉一次）；偏航 90° 時亦成立；
  M1 C2 composed＝(1, 0, dt, 0)；step_sp 給手臂 q⁺＝q＋dt·u、s⁺＝s＋dt·u；M2 ideal_phys_step 5 步＝一個控制步；
  M3 C2 求解（未來參考版核心）可行，且與 C3 對同一輸入給出不同解（模型確實被換掉）。
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc import wgmpc_core_sp as S                     # noqa: E402
from ammr_wholebody_mpc import wgmpc_core_sp_fr as FR                 # noqa: E402
from ammr_wholebody_mpc.wgmpc_core import body_to_world              # noqa: E402
from scipy.spatial.transform import Rotation as _Rot                 # noqa: E402


def so3_exp(w):
    return _Rot.from_rotvec(np.asarray(w, float)).as_matrix()
import bottle_models as BM                                            # noqa: E402
import bottle_offline as BO                                           # noqa: E402
import bottle_ref as BR                                               # noqa: E402
import motm_b1_port as C1                                             # noqa: E402
import wg4b_b1_core as B1                                             # noqa: E402

K = BO.K
BAD, N_CHK = [], 0


def chk(name, cond, info=''):
    global N_CHK
    N_CHK += 1
    if not cond:
        BAD.append(f'{name} {info}')


task = BR.BottleTask(T_grasp=BR.grasp_pose(0.0, 0.42, 0.70))
q_ready, T_ready_rel = BO.ready_posture()
task.T_ready_rel = T_ready_rel
plan = (np.array([0.0, 100.0]), np.array([-1.0, -1.0 + 0.05 * 100.0]), 0.0)
st = BR.PhaseState(phase='INSERT', t_phase=10.0)
Tf = lambda t: BR.target_at(task, st, plan, t)                       # noqa: E731
for t in (10.2, 10.5, 10.8):
    tw = C1.ref_twist(Tf, t)
    s_ = (t - 10.0) / task.t_insert
    sdot = 30 * s_ ** 2 * (1 - s_) ** 2 / task.t_insert
    v_exp = (task.T_grasp[:3, 3] - task.T_pre[:3, 3]) * sdot
    chk(f'F1 INSERT 線速度＝解析 @ {t}', np.allclose(tw[:3], v_exp, atol=1e-6), f'{tw[:3]} vs {v_exp}')
st2 = BR.PhaseState(t_cruise0=0.0, t_pre=8.0)
Tc = lambda t: BR.target_at(task, st2, plan, t)                       # noqa: E731
for t in (2.0, 4.0, 6.0):
    tw = C1.ref_twist(Tc, t)
    h = 1e-3
    R_pred = so3_exp(tw[3:] * h) @ Tc(t)[:3, :3]
    chk(f'F2 角速度世界座標 @ {t}', np.allclose(R_pred, Tc(t + h)[:3, :3], atol=1e-7), str(np.abs(R_pred - Tc(t + h)[:3, :3]).max()))
    p_pred = Tc(t)[:3, 3] + tw[:3] * h
    chk(f'F3 時間索引 @ {t}', np.allclose(p_pred, Tc(t + h)[:3, 3], atol=1e-7) and not np.allclose(C1.ref_twist(Tc, t), C1.ref_twist(Tc, t + 0.5)))

# V1／V2
cfg = BO.make_cfg()
cfg.w_vref, cfg.base_vref = 0.0, None
p = B1.B1Params(q_pref=tuple(float(v) for v in q_ready))
q0 = np.r_[-0.3, 0.0, 0.0, q_ready]
T_far = K.fk(q0, 'link_tcp').copy()
T_far[0, 3] += 0.5
u_prev = np.r_[0.035, 0.0, 0.0, np.zeros(6)]
r = C1.solve_c1(K, q0, q_ready, u_prev, T_far, np.zeros(6), cfg, p)
chk('V1 可解', r.ok, r.reason)
if r.ok:
    chk('V1 底盤可超舊框且不超 vbox', abs(r.u0[0]) > 0.035255 + 1e-4 and np.all(np.abs(r.u0[:3]) <= cfg.vbox()[:3] + 1e-9), str(r.u0[:3]))
for th in (0.0, math.pi / 2):
    q1 = np.r_[0.0, 0.0, th, q_ready]
    T1 = K.fk(q1, 'link_tcp')
    v_ff = np.array([0.01, -0.005, 0.003, 0.0, 0.0, 0.02])
    r = C1.solve_c1(K, q1, q_ready, np.zeros(9), T1, v_ff, cfg, B1.B1Params(q_pref=tuple(q_ready), damping=1e-4, mu_post=0.0, base_weight=1.0))
    chk(f'V2 可解 θ={th:.2f}', r.ok, r.reason)
    if r.ok:
        Jb = K.jacobian(q1, 'link_tcp') @ body_to_world(th)
        chk(f'V2 J_b·u≈v_ff（世界）θ={th:.2f}', np.allclose(Jb @ r.u0, v_ff, atol=5e-4), str(Jb @ r.u0))

# M1／M2／M3
c2 = BM.WGMPCConfigSPIdeal(**{f: getattr(cfg, f) for f in cfg.__dataclass_fields__})
P, Q, G, h_, kp = c2.composed()
chk('M1 composed', np.all(P == 1) and np.all(Q == 0) and np.allclose(G, c2.dt) and np.all(h_ == 0) and kp == 5)
z = S.make_z(np.r_[0.1, 0.2, 0.3, q_ready], q_ready + 0.01)
u = np.r_[0.02, -0.01, 0.05, 0.1, -0.2, 0.05, 0.0, 0.3, -0.1]
z1 = S.step_sp(z, u, c2)
chk('M1 手臂 q⁺＝q＋dt·u', np.allclose(z1[3:9], z[3:9] + c2.dt * u[3:]))
chk('M1 設定點 s⁺＝s＋dt·u', np.allclose(z1[9:], z[9:] + c2.dt * u[3:]))
qa, sa = z[:9].copy(), z[9:].copy()
for _ in range(5):
    qa, sa = BM.ideal_phys_step(qa, sa, u, 0.01)
chk('M2 逐物理步 5 步＝控制步（手臂、設定點）', np.allclose(qa[3:], z1[3:9]) and np.allclose(sa, z1[9:]))
chk('M2 底盤與原核心逐物理步一致', np.allclose(qa[:3], S.plant_phys_step(z[:9], z[9:], u, cfg, 5)[0][:3]))
T_seq = np.tile(K.fk(z[:9], 'link_tcp'), (6, 1, 1))
T_seq[:, 0, 3] += 0.02
r2 = FR.solve_sp_fr(K, z, np.zeros(9), T_seq, c2)
r3 = FR.solve_sp_fr(K, z, np.zeros(9), T_seq, cfg)
chk('M3 C2 可解', r2.ok, r2.reason)
chk('M3 C2 與 C3 解不同（模型已換）', r2.ok and r3.ok and not np.allclose(r2.U, r3.U))
print(f'C1／C2 測試 {N_CHK} 項：' + ('全過' if not BAD else '失敗 ' + '；'.join(BAD[:6])))
sys.exit(0 if not BAD else 1)
