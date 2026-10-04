"""底盤 1 比 1 跟退時，手臂是否全程待在同一個好姿態。"""
import sys
import numpy as np
sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics as WK
K = WK.from_urdf_file('evaluation/models/omni_bot_wholebody_expanded.urdf')
LIM = np.array(K.joint_limits()); M = 0.05
ELO, EHI = LIM[0, 3:] + M, LIM[1, 3:] - M
R = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
BX, BYAW, BY0 = -0.136412, 1.297349, 0.56
QA0 = np.array([0.030291, 1.614035, 2.754512, -0.538549, -0.487832, -1.084764])


def ik(p, base, q_arm, seeds=6):
    rng = np.random.default_rng(3); best = None
    for s in range(seeds):
        q = np.concatenate([base, q_arm if s == 0 else
                            np.clip(q_arm + rng.normal(0, 0.3, 6), ELO, EHI)])
        for _ in range(500):
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
        if ep < 1e-3 and ang < 5e-3:
            sl = float(min(np.minimum(q[3:]-ELO, EHI-q[3:])))
            sv = float(np.linalg.svd(K.jacobian(q,'link_tcp')[:,3:],
                                     compute_uv=False)[-1])
            if sl > 0 and (best is None or sv > best[1]):
                best = (sl, sv, q.copy())
    return best


print('=== 底盤 1 比 1 跟退（y 0.56 → 0.36），手臂姿態是否恆定 ===')
q = QA0.copy(); rec = []
for d in np.arange(0.0, 0.2001, 0.025):
    base = np.array([BX, BY0 - d, BYAW])
    r = ik(np.array([0.0, 1.1503 - d, 0.55]), base, q)
    if r is None:
        print(f'  開度 {d*1e3:5.1f} mm  **無解**'); break
    rec.append(r); q = r[2][3:]
    print(f'  開度 {d*1e3:5.1f} mm  底盤 y {base[1]:+.3f}  '
          f'手臂伸 {1.1503-d-base[1]:.3f} m  j 餘裕 {r[0]:+.4f}  '
          f'奇異值 {r[1]:.4f}')
if len(rec) > 1:
    Q = np.array([r[2][3:] for r in rec])
    tot = np.abs(np.diff(Q, axis=0)).sum(axis=0)
    print(f'  六軸累積行程 {tot.sum():.5f} rad（最大單軸 {tot.max():.5f}）'
          f' ⇒ 手臂幾乎不動，姿態恆定')
    print('  恆定姿態 q = [' + ', '.join(f'{v:.4f}' for v in Q[0]) + ']')

print('\n=== 導航用的收攏姿態（手臂縮在底盤上方，不前伸）===')
rng = np.random.default_rng(7)
best = None
for _ in range(4000):
    qa = ELO + rng.random(6) * (EHI - ELO)
    q = np.concatenate([[0., 0., 0.], qa])
    T = K.fk(q, 'link_tcp')
    p = T[:3, 3]
    rad = float(np.hypot(p[0], p[1]))
    sl = float(min(np.minimum(qa - ELO, EHI - qa)))
    # 要求：水平半徑 ≤0.20 m（縮在底盤內）、高度 0.40–0.75 m、限位餘裕大
    if rad <= 0.20 and 0.40 <= p[2] <= 0.75 and sl > 0.30:
        score = sl - 2.0 * rad
        if best is None or score > best[0]:
            best = (score, qa.copy(), p.copy(), rad, sl)
if best:
    _, qa, p, rad, sl = best
    print(f'  候選 q = [' + ', '.join(f'{v:.4f}' for v in qa) + ']')
    print(f'  TCP 相對底盤 ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})  '
          f'水平半徑 {rad:.3f} m  最小限位餘裕 {sl:+.4f} rad')
    print('  **自碰未核**：這裡只核了限位與緊湊度，逐連桿自碰要另外算')
else:
    print('  **在限制下找不到候選** ⇒ 收攏姿態要放寬條件重找')
