"""兩件事：(1) 全身解在 200 mm 行程裡各關節實際走多少；
(2) 固定底盤的對照趟次若從**同一夾持位姿**起步，能開多遠。"""
import sys
import numpy as np
sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics as WK

K = WK.from_urdf_file('evaluation/models/omni_bot_wholebody_expanded.urdf')
LIM = np.array(K.joint_limits()); M = 0.05
ELO, EHI = LIM[0, 3:] + M, LIM[1, 3:] - M
R_DES = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
Q0 = np.array([-0.137683, 0.372979, 1.297260,
               0.030503, 1.613495, 2.754491, -0.538007, -0.487081, -1.085142])
GRASP_Y = 1.165 - 0.0147
Z = 0.55


def err(q, p):
    T = K.fk(q, 'link_tcp')
    Rr = R_DES.T @ T[:3, :3]
    ang = np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))
    return T[:3, 3] - p, float(ang), T


def ik(p, q0, free_base, iters=400):
    q = q0.copy()
    for _ in range(iters):
        ep, ang, T = err(q, p)
        Rr = R_DES.T @ T[:3, :3]
        wv = np.array([Rr[2, 1]-Rr[1, 2], Rr[0, 2]-Rr[2, 0], Rr[1, 0]-Rr[0, 1]])
        s = np.linalg.norm(wv)
        er = np.zeros(3) if s < 1e-12 else (T[:3, :3] @ (wv/s)) * ang
        e = np.concatenate([ep, er])
        if np.linalg.norm(ep) < 1e-5 and ang < 1e-4:
            break
        J = K.jacobian(q, 'link_tcp')
        if not free_base:
            J = J.copy(); J[:, :3] = 0.0
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6*np.eye(6), -e)
        q = q + np.clip(dq, -0.08, 0.08)
        q[3:] = np.clip(q[3:], ELO, EHI)
    ep, ang, _ = err(q, p)
    return q, float(np.linalg.norm(ep)), ang, float(min(np.minimum(q[3:]-ELO, EHI-q[3:])))


# ---- (1) 全身解的關節行程 ----
q, *_ = ik(np.array([-0.001206, GRASP_Y, Z]), Q0, True)
Q_GRASP_WB = q.copy()
traj = [q.copy()]
for d in np.arange(0.005, 0.2001, 0.005):
    q, ep, ang, sl = ik(np.array([-0.001206, GRASP_Y-d, Z]), q, True)
    traj.append(q.copy())
T = np.array(traj)
print('=== 全身解在 0→200 mm 開啟行程裡的分配 ===')
print(f'  底盤  Δx {T[-1,0]-T[0,0]:+.4f}  Δy {T[-1,1]-T[0,1]:+.4f}  '
      f'Δyaw {T[-1,2]-T[0,2]:+.5f}  路徑長 '
      f'{np.hypot(*np.diff(T[:,:2],axis=0).T).sum():.4f} m')
tot = np.abs(np.diff(T[:, 3:], axis=0)).sum(axis=0)
print('  各軸累積行程（rad）：' +
      '  '.join(f'j{i+1} {v:.5f}' for i, v in enumerate(tot)))
print(f'  六軸合計 {tot.sum():.5f} rad；最大單軸 {tot.max():.5f} rad')
# 以 TCP 速度分解：底盤貢獻 vs 手臂貢獻
dT = np.diff(T, axis=0)
share_b, share_a = [], []
for i in range(len(dT)):
    Jm = K.jacobian(T[i], 'link_tcp')[:3]
    vb = Jm[:, :3] @ dT[i, :3]
    va = Jm[:, 3:] @ dT[i, 3:]
    share_b.append(np.linalg.norm(vb)); share_a.append(np.linalg.norm(va))
sb, sa = np.array(share_b), np.array(share_a)
print(f'  TCP 位移的來源（線性化逐步）：底盤 {sb.sum()*1e3:.1f} mm、'
      f'手臂 {sa.sum()*1e3:.1f} mm ⇒ 手臂佔 '
      f'{100*sa.sum()/(sa.sum()+sb.sum()):.1f}%')

# ---- (2) 固定底盤對照：從同一夾持位姿起步 ----
print('\n=== 固定底盤對照（底盤停在全身解的夾持位姿）===')
print(f'  底盤停於 ({Q_GRASP_WB[0]:+.4f}, {Q_GRASP_WB[1]:+.4f}) '
      f'yaw {Q_GRASP_WB[2]:+.4f}')
q = Q_GRASP_WB.copy()
last_ok = None
for d in np.arange(0.0, 0.2001, 0.005):
    q, ep, ang, sl = ik(np.array([-0.001206, GRASP_Y-d, Z]), q, False)
    ok = ep < 2e-3 and ang < 0.02 and sl > 0.0
    if ok:
        last_ok = (d, ep, ang, sl)
    else:
        print(f'  **在開度 {d*1e3:.1f} mm 失敗**：誤差 {ep*1e3:.3f} mm、'
              f'姿態 {ang:.5f} rad、j 餘裕 {sl:+.5f}')
        break
else:
    print('  全程 0→200 mm 可達（固定底盤）')
if last_ok:
    print(f'  固定底盤可達的最大開度：{last_ok[0]*1e3:.1f} mm'
          f'（誤差 {last_ok[1]*1e3:.3f} mm、j 餘裕 {last_ok[3]:+.5f}）')
