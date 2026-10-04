"""固定底盤時，TCP 沿世界 y、姿態鎖 R_DES、高度 0.55 的可達區間。

為什麼要算：固定底盤從階段 A 的接觸前位姿伸不到把手（差 94.4 mm），
但最小 j 餘裕仍有 +0.136 ⇒ 卡的不是關節限位，是**工作空間邊界**。
把邊界算出來，才能說清夾持段為何必然由底盤提供。
"""
import sys
import numpy as np
sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics as WK

K = WK.from_urdf_file('evaluation/models/omni_bot_wholebody_expanded.urdf')
LIM = np.array(K.joint_limits()); M = 0.05
ELO, EHI = LIM[0, 3:] + M, LIM[1, 3:] - M
R_DES = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
# **收斂後**的接觸前暫停位姿（保持窗結束，誤差 1.452 mm）。
# 不要用「首次進入容差」那一步：那一步誤差 4.87 mm，底盤還差 4.5 mm 沒走完，
# 足以讓邊界判斷翻面（量過：用那一步會算出接觸前暫停「不可達」）。
BASE = np.array([-0.136412, 0.377503, 1.297349])
QA = np.array([0.030291, 1.614035, 2.754512, -0.538549, -0.487832, -1.084764])


def solve(p, q_arm, seeds=8):
    best = (1e9, None)
    rng = np.random.default_rng(0)
    for s in range(seeds):
        q = np.concatenate([BASE, q_arm if s == 0 else
                            np.clip(q_arm + rng.normal(0, 0.25, 6), ELO, EHI)])
        for _ in range(500):
            T = K.fk(q, 'link_tcp')
            ep = T[:3, 3] - p
            Rr = R_DES.T @ T[:3, :3]
            ang = float(np.arccos(np.clip((np.trace(Rr)-1)/2, -1, 1)))
            wv = np.array([Rr[2,1]-Rr[1,2], Rr[0,2]-Rr[2,0], Rr[1,0]-Rr[0,1]])
            n = np.linalg.norm(wv)
            er = np.zeros(3) if n < 1e-12 else (T[:3,:3] @ (wv/n)) * ang
            e = np.concatenate([ep, er])
            if np.linalg.norm(ep) < 1e-5 and ang < 1e-4:
                break
            J = K.jacobian(q, 'link_tcp').copy(); J[:, :3] = 0.0
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6*np.eye(6), -e)
            q = q + np.clip(dq, -0.08, 0.08)
            q[3:] = np.clip(q[3:], ELO, EHI)
        T = K.fk(q, 'link_tcp')
        ep = float(np.linalg.norm(T[:3, 3] - p))
        Rr = R_DES.T @ T[:3, :3]
        ang = float(np.arccos(np.clip((np.trace(Rr)-1)/2, -1, 1)))
        sl = float(min(np.minimum(q[3:]-ELO, EHI-q[3:])))
        score = ep + 0.05*ang
        if score < best[0] and ep < 2e-3 and ang < 0.02 and sl > 0:
            best = (score, (ep, ang, sl, q.copy()))
    return best[1]


print(f'底盤固定於 ({BASE[0]:+.4f}, {BASE[1]:+.4f}) yaw {BASE[2]:+.4f}；'
      f'TCP 高 0.55、姿態鎖 R_DES')
ok_y = []
for y in np.arange(0.90, 1.21, 0.005):
    r = solve(np.array([-0.001206, y, 0.55]), QA)
    if r:
        ok_y.append((y, r[2]))
if ok_y:
    ys = [y for y, _ in ok_y]
    print(f'  可達 y 區間：{min(ys):.3f} … {max(ys):.3f} m '
          f'（共 {len(ys)}/{len(np.arange(0.90,1.21,0.005))} 個取樣點）')
    print(f'  對應底盤距離：{np.hypot(-0.001206-BASE[0], max(ys)-BASE[1]):.4f}'
          f' m（遠端）／{np.hypot(-0.001206-BASE[0], min(ys)-BASE[1]):.4f} m（近端）')
    for tag, y in (('接觸前暫停 y=1.0596', 1.0596),
                   ('夾持 y=1.1503', 1.1503),
                   ('開 100 mm 後 y=1.0503', 1.0503),
                   ('開 200 mm 後 y=0.9503', 0.9503)):
        hit = [s for yy, s in ok_y if abs(yy-y) < 0.003]
        print(f'  {tag}：{"可達" if hit else "**不可達**"}'
              + (f'（j 餘裕 {hit[0]:+.5f}）' if hit else ''))
else:
    print('  **整段皆不可達**')
