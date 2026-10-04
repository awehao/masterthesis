"""離線可行性核對：夾持位姿與 200 mm 開啟／關閉行程的運動學可達性。

**固定底盤**與**底盤參與**兩種情形各掃一次，並記下 j3 的有效餘裕 ——
階段 A 到達時 j3 已距有效上限僅 0.131 rad，夾持還要再往 +y 前伸 0.0906 m。
"""
import sys
import numpy as np
sys.path.insert(0, 'src/ammr_wholebody_mpc')
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics as WK

URDF = 'evaluation/models/omni_bot_wholebody_expanded.urdf'
K = WK.from_urdf_file(URDF)
LIM = np.array(K.joint_limits())
M = 0.05
ELO, EHI = LIM[0, 3:] + M, LIM[1, 3:] - M
R_DES = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])

# 階段 A 到達時的實測狀態（spawn 趟次）
Q0 = np.array([-0.137683, 0.372979, 1.297260,
               0.030503, 1.613495, 2.754491, -0.538007, -0.487081, -1.085142])
BAR_Y0 = 1.165          # 開度 0 時把手橫桿世界 y
TCP_OFF = 0.0147        # TCP 相對桿心沿工具 z 的偏移
GRASP_Y = BAR_Y0 - TCP_OFF
Z = 0.55


def pose_err(q, p_des):
    T = K.fk(q, 'link_tcp')
    ep = T[:3, 3] - p_des
    Rr = R_DES.T @ T[:3, :3]
    ang = np.arccos(np.clip((np.trace(Rr) - 1.0) / 2.0, -1.0, 1.0))
    return ep, float(ang)


def ik(p_des, q_init, free_base, iters=300):
    """阻尼最小平方 IK。free_base 決定底盤三維是否可動。"""
    q = q_init.copy()
    for _ in range(iters):
        ep, ang = pose_err(q, p_des)
        T = K.fk(q, 'link_tcp')
        Rr = R_DES.T @ T[:3, :3]
        # 旋轉誤差的軸角向量（世界框）
        wv = np.array([Rr[2, 1] - Rr[1, 2], Rr[0, 2] - Rr[2, 0],
                       Rr[1, 0] - Rr[0, 1]])
        s = np.linalg.norm(wv)
        er = np.zeros(3) if s < 1e-12 else (T[:3, :3] @ (wv / s)) * ang
        e = np.concatenate([ep, er])
        if np.linalg.norm(ep) < 1e-5 and ang < 1e-4:
            break
        J = K.jacobian(q, 'link_tcp')
        if not free_base:
            J = J.copy(); J[:, :3] = 0.0
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), -e)
        q = q + np.clip(dq, -0.08, 0.08)
        q[3:] = np.clip(q[3:], ELO, EHI)
    ep, ang = pose_err(q, p_des)
    slack = float(min(np.minimum(q[3:] - ELO, EHI - q[3:])))
    return q, float(np.linalg.norm(ep)), ang, slack


def sweep(free_base, label):
    print(f'\n=== {label} ===')
    q = Q0.copy()
    # 第一段：從接觸前位姿前伸到夾持位姿
    ok_grasp = None
    for d in np.linspace(0.0, GRASP_Y - 1.055145, 15):
        p = np.array([-0.001206, 1.055145 + d, Z])
        q, ep, ang, sl = ik(p, q, free_base)
        ok = ep < 2e-3 and ang < 0.02 and sl > 0.0
        if not ok and ok_grasp is None:
            ok_grasp = (p[1], ep, ang, sl)
    pg = np.array([-0.001206, GRASP_Y, Z])
    q, ep, ang, sl = ik(pg, q, free_base)
    print(f'  夾持位姿 y={GRASP_Y:.4f}：位置誤差 {ep*1e3:7.3f} mm  '
          f'姿態 {ang:7.5f} rad  j餘裕 {sl:+.5f}  '
          f'{"可達" if (ep<2e-3 and ang<0.02 and sl>0) else "**不可達**"}')
    if free_base:
        print(f'    底盤移到 ({q[0]:+.4f}, {q[1]:+.4f}) yaw {q[2]:+.4f}  '
              f'位移 {np.hypot(q[0]-Q0[0], q[1]-Q0[1]):.4f} m')
    # 第二段：開啟 0 → 200 mm（TCP 沿 −y）
    rows = []
    qq = q.copy()
    for d in np.arange(0.0, 0.2001, 0.010):
        p = np.array([-0.001206, GRASP_Y - d, Z])
        qq, ep, ang, sl = ik(p, qq, free_base)
        rows.append((d, ep, ang, sl, qq.copy()))
    first_bad = next((r for r in rows if not (r[1] < 2e-3 and r[2] < 0.02
                                              and r[3] > 0.0)), None)
    print('  開啟行程：')
    for d, ep, ang, sl, qv in rows[::4]:
        base = (f'  底盤 ({qv[0]:+.3f},{qv[1]:+.3f})' if free_base else '')
        print(f'    開度 {d*1e3:6.1f} mm  誤差 {ep*1e3:7.3f} mm  '
              f'姿態 {ang:7.5f}  j餘裕 {sl:+.5f}{base}')
    if first_bad is None:
        print(f'  ⇒ **全程 0→200 mm 可達**（最小 j 餘裕 '
              f'{min(r[3] for r in rows):+.5f} rad）')
    else:
        print(f'  ⇒ **在開度 {first_bad[0]*1e3:.1f} mm 失敗**'
              f'（誤差 {first_bad[1]*1e3:.3f} mm、姿態 {first_bad[2]:.5f}、'
              f'餘裕 {first_bad[3]:+.5f}）')
    if free_base:
        qe = rows[-1][4]
        print(f'  底盤全程位移 {np.hypot(qe[0]-q[0], qe[1]-q[1]):.4f} m'
              f'（沿世界 y {qe[1]-q[1]:+.4f} m）')
    return rows


sweep(False, '固定底盤（手臂單獨操作）')
sweep(True, '底盤參與（九維全身）')
