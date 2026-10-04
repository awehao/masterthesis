"""底盤該停多近：掃底盤 y，看手臂在整段行程裡「好不好用」。

好不好用用兩個量：
  有效限位最小餘裕（離關節限位多遠）
  手臂 Jacobian 最小奇異值（離奇異點多遠；越大越好）
行程：夾持 y=1.1503 → 開 200 mm y=0.9503（固定底盤，看手臂的工作區域品質）
"""
import sys
import numpy as np
sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics as WK

K = WK.from_urdf_file('evaluation/models/omni_bot_wholebody_expanded.urdf')
LIM = np.array(K.joint_limits()); M = 0.05
ELO, EHI = LIM[0, 3:] + M, LIM[1, 3:] - M
R = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
BX, BYAW = -0.136412, 1.297349        # 沿用階段 A 選到的橫向與朝向
QA0 = np.array([0.030291, 1.614035, 2.754512, -0.538549, -0.487832, -1.084764])


def ik(p, base, q_arm, seeds=4):
    rng = np.random.default_rng(3); best = None
    for s in range(seeds):
        q = np.concatenate([base, q_arm if s == 0 else
                            np.clip(q_arm + rng.normal(0, 0.3, 6), ELO, EHI)])
        for _ in range(400):
            T = K.fk(q, 'link_tcp'); ep = T[:3, 3] - p
            Rr = R.T @ T[:3, :3]
            ang = float(np.arccos(np.clip((np.trace(Rr)-1)/2, -1, 1)))
            wv = np.array([Rr[2,1]-Rr[1,2], Rr[0,2]-Rr[2,0], Rr[1,0]-Rr[0,1]])
            n = np.linalg.norm(wv)
            er = np.zeros(3) if n < 1e-12 else (T[:3,:3] @ (wv/n)) * ang
            e = np.concatenate([ep, er])
            if np.linalg.norm(ep) < 1e-5 and ang < 1e-4:
                break
            J = K.jacobian(q, 'link_tcp').copy(); J[:, :3] = 0.0
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6*np.eye(6), -e)
            q = q + np.clip(dq, -0.08, 0.08); q[3:] = np.clip(q[3:], ELO, EHI)
        T = K.fk(q, 'link_tcp')
        ep = float(np.linalg.norm(T[:3, 3] - p))
        Rr = R.T @ T[:3, :3]
        ang = float(np.arccos(np.clip((np.trace(Rr)-1)/2, -1, 1)))
        if ep < 5e-3 and ang < 0.02:
            sl = float(min(np.minimum(q[3:]-ELO, EHI-q[3:])))
            sv = float(np.linalg.svd(K.jacobian(q, 'link_tcp')[:, 3:],
                                     compute_uv=False)[-1])
            if sl > 0 and (best is None or sv > best[1]):
                best = (sl, sv, q.copy())
    return best


print('底盤橫向 x=-0.1364、朝向 yaw=1.2973（沿用階段 A）')
print(f'{"底盤 y":>7}{"夾持距離":>9}{"整段可行":>9}'
      f'{"最差 j 餘裕":>12}{"最差奇異值":>11}{"備註":>6}')
rows = []
for by in np.arange(0.36, 0.85, 0.04):
    base = np.array([BX, by, BYAW])
    q = QA0.copy(); ok = True; worst_sl = 9e9; worst_sv = 9e9
    for d in np.arange(0.0, 0.2001, 0.02):
        r = ik(np.array([0.0, 1.1503 - d, 0.55]), base, q)
        if r is None:
            ok = False; break
        worst_sl = min(worst_sl, r[0]); worst_sv = min(worst_sv, r[1])
        q = r[2][3:]
    D = 1.1503 - by
    note = ''
    if ok:
        rows.append((by, D, worst_sl, worst_sv))
        if worst_sl > 0.30 and worst_sv > 0.25:
            note = '好'
    print(f'{by:>7.2f}{D:>9.3f}{"是" if ok else "**否**":>9}'
          f'{worst_sl if ok else float("nan"):>12.4f}'
          f'{worst_sv if ok else float("nan"):>11.4f}{note:>6}')
if rows:
    best = max(rows, key=lambda r: min(r[2]/0.5, r[3]/0.5))
    print(f'\n綜合最好：底盤 y={best[0]:.2f}（夾持時手臂伸 {best[1]:.3f} m）'
          f'  最差 j 餘裕 {best[2]:+.4f} rad  最差奇異值 {best[3]:.4f}')
    print(f'對照：階段 A 停在 y=0.3775（伸 0.773 m），'
          f'夾持位本身就伸不到')
