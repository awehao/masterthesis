"""快速 vs 省力的取捨曲線：把 200 mm 按比例拆給底盤與手臂。

底盤軌跡**給定**（按分擔比例線性走），手臂用 IK 補剩下的，記錄：
  歷時下界（底盤那一份除以底盤上限）
  手臂六軸累積行程（省力的代價）
  整段最差關節餘裕與最差奇異值（姿態品質）
"""
import sys
import numpy as np
sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics as WK
from ammr_wholebody_mpc.wgmpc_core import WGMPCConfig

K = WK.from_urdf_file('evaluation/models/omni_bot_wholebody_expanded.urdf')
LIM = np.array(K.joint_limits()); M = 0.05
ELO, EHI = LIM[0, 3:] + M, LIM[1, 3:] - M
R = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
V_BASE = WGMPCConfig().v_base_lin
BX, BYAW, BY0 = -0.136412, 1.297349, 0.56
Q_HOLD = np.array([-0.0321, 0.9177, 1.4853, -0.3580, -1.0325, -1.3813])
STROKE = 0.200


def ik(p, base, q_arm):
    q = np.concatenate([base, q_arm])
    for _ in range(500):
        T = K.fk(q, 'link_tcp'); ep = T[:3, 3] - p
        Rr = R.T @ T[:3, :3]
        ang = float(np.arccos(np.clip((np.trace(Rr)-1)/2, -1, 1)))
        wv = np.array([Rr[2,1]-Rr[1,2], Rr[0,2]-Rr[2,0], Rr[1,0]-Rr[0,1]])
        n = np.linalg.norm(wv)
        er = np.zeros(3) if n < 1e-12 else (T[:3,:3] @ (wv/n)) * ang
        e = np.concatenate([ep, er])
        if np.linalg.norm(ep) < 1e-6 and ang < 1e-5:
            break
        J = K.jacobian(q, 'link_tcp').copy(); J[:, :3] = 0.0
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-7*np.eye(6), -e)
        q = q + np.clip(dq, -0.05, 0.05); q[3:] = np.clip(q[3:], ELO, EHI)
    T = K.fk(q, 'link_tcp')
    ok = (np.linalg.norm(T[:3, 3] - p) < 1e-3)
    sl = float(min(np.minimum(q[3:]-ELO, EHI-q[3:])))
    sv = float(np.linalg.svd(K.jacobian(q, 'link_tcp')[:, 3:],
                             compute_uv=False)[-1])
    return q[3:].copy(), ok, sl, sv


print(f'底盤上限 {V_BASE*1e3:.1f} mm/s；停位 y={BY0}；行程 {STROKE*1e3:.0f} mm')
print(f'{"底盤分擔":>8}{"底盤走":>8}{"歷時下界":>9}{"手臂行程":>10}'
      f'{"最差餘裕":>9}{"最差奇異值":>11}{"全程可行":>9}')
rows = []
for f in (1.00, 0.90, 0.75, 0.50, 0.35, 0.20, 0.10, 0.00):
    qa = Q_HOLD.copy(); traj = [qa.copy()]
    ok_all = True; worst_sl = 9e9; worst_sv = 9e9
    for k in np.arange(0.005, 1.0001, 0.005):
        base = np.array([BX, BY0 - f*STROKE*k, BYAW])
        p = np.array([0.0, 1.1503 - STROKE*k, 0.55])
        qa, ok, sl, sv = ik(p, base, qa)
        traj.append(qa.copy())
        ok_all &= ok; worst_sl = min(worst_sl, sl); worst_sv = min(worst_sv, sv)
    Q = np.array(traj)
    travel = float(np.abs(np.diff(Q, axis=0)).sum())
    t = f*STROKE/V_BASE
    rows.append((f, t, travel, worst_sl, worst_sv, ok_all))
    print(f'{f*100:>7.0f}%{f*STROKE*1e3:>7.0f}mm{t:>8.2f}s'
          f'{travel:>9.4f}r{worst_sl:>9.4f}{worst_sv:>11.4f}'
          f'{"是" if ok_all else "**否**":>9}')
print(f'\n閘門「底盤沿軸走 ≥0.10 m」⇒ 分擔 ≥50% ⇒ '
      f'歷時下界 {0.10/V_BASE:.2f} s、手臂行程 '
      f'{[r[2] for r in rows if abs(r[0]-0.5)<1e-9][0]:.4f} rad')
print('對比全交底盤：歷時 5.67 s、手臂行程 '
      f'{[r[2] for r in rows if abs(r[0]-1.0)<1e-9][0]:.5f} rad')
