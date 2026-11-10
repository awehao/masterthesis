#!/usr/bin/env python3
"""門把手 RGB-D 擷取的視點登錄（Codex 20261110_225452 核可；規格 第八次進度報告素材/規格/門把手視覺幾何_資料缺口與最小擷取需求_r2.md）。擷取前產生並登錄，之後不改。
量的來源：只用門洞先驗（m8_scene_check.json door_prior：門洞寬 W、牆機器人側面 y＝0、門洞 x∈[0, W]）與運動學（URDF）；**不讀把手位置、資產遮罩或真值姿態**。
  底盤：到門面距離 d ∈ {1.5, 1.2, 0.9, 0.7} m（底盤中心 y＝−d）× 橫向 x ∈ {0.2, 0.5, 0.8} × 偏航偏移 δ ∈ {−15°, 0°, +15°}；
        偏航基準＝面向門面法向（世界 +y），世界偏航＝π/2＋δ（擷取前寫定，不依結果調整）。
  手臂：K2_READY（0,0,0.035,0,−1.5708,0）與觀測姿態 OBS——以手臂六軸數值 IK 讓相機光軸指向門洞中心 (W/2, 0, 0.90)，
        同時要求相機 x 軸水平（影像不滾轉）並以小權重靠近 K2_READY；0.90 m＝本包宣告的觀測中心高度先驗，不是視覺估出的把手高度，也不是已驗證的一般高度帶。
  可行性（登錄時就分類，不另挑視點補成功數）：IK 指向誤差 > 3° ⇒ ik_fail；關節超限 ⇒ limit_fail；手臂各連桿原點與相機 y > −0.05（穿入牆／門面側）⇒ config_collision。
分組：開發 d ∈ {1.5, 0.9}；保留 d ∈ {1.2, 0.7}（方法與門檻凍結後才開）。同視點 3 格只核量測一致性，不算三個獨立案例。
    python3 evaluation/m8_door_viewpoints.py OUT.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_wholebody_mpc'))
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics  # noqa: E402

OPT = 'camera_color_optical_frame'
K2_READY = np.array([0.0, 0.0, 0.035, 0.0, -1.5708, 0.0])
AIM_Z = 0.90
DISTS, LATS, YAWS_DEG = (1.5, 1.2, 0.9, 0.7), (0.2, 0.5, 0.8), (-15.0, 0.0, 15.0)
DEV_D = (1.5, 0.9)
LINKS = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', OPT]


def ik_point(K, base, aim, q0=K2_READY, w_reg=0.05, iters=300):
    lo, hi = (np.asarray(v)[3:] for v in K.joint_limits())
    q = q0.copy()

    def res(qa):
        T = K.fk(np.r_[base, qa], OPT)
        p, x, z = T[:3, 3], T[:3, 0], T[:3, 2]
        u = (aim - p) / max(np.linalg.norm(aim - p), 1e-9)
        return np.r_[np.cross(z, u), x[2], w_reg * (qa - q0)]
    for _ in range(iters):
        r = res(q)
        J = np.zeros((len(r), 6))
        for j in range(6):
            dq = np.zeros(6)
            dq[j] = 1e-5
            J[:, j] = (res(q + dq) - r) / 1e-5
        step = -np.linalg.solve(J.T @ J + 1e-4 * np.eye(6), J.T @ r)
        q = np.clip(q + np.clip(step, -0.2, 0.2), lo, hi)
        if np.linalg.norm(step) < 1e-7:
            break
    T = K.fk(np.r_[base, q], OPT)
    u = (aim - T[:3, 3]) / np.linalg.norm(aim - T[:3, 3])
    err = math.degrees(math.acos(float(np.clip(T[:3, 2] @ u, -1, 1))))
    return q, err, float(T[:3, 0][2])


def main():
    out = sys.argv[1]
    sc = json.load(open(os.path.join(WS, '第八次進度報告素材', '資料', 'm8_scene_check.json')))
    W = float(sc['door_prior']['W'])
    aim = np.array([W / 2.0, 0.0, AIM_Z])
    K = WholeBodyKinematics.from_urdf_file(os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf'))
    lo, hi = (np.asarray(v)[3:] for v in K.joint_limits())
    vps = []
    for d in DISTS:
        for x in LATS:
            for dy in YAWS_DEG:
                base = np.array([x, -d, math.pi / 2 + math.radians(dy)])
                for arm in ('K2_READY', 'OBS'):
                    if arm == 'K2_READY':
                        qa, err, xz = K2_READY.copy(), None, None
                    else:
                        qa, err, xz = ik_point(K, base, aim)
                    status = 'ok'
                    if arm == 'OBS' and err > 3.0:
                        status = 'ik_fail'
                    elif np.any(qa < lo - 1e-9) or np.any(qa > hi + 1e-9):
                        status = 'limit_fail'
                    else:
                        ys = [float(K.fk(np.r_[base, qa], ln)[1, 3]) for ln in LINKS]
                        if max(ys) > -0.05:
                            status = 'config_collision'
                    T = K.fk(np.r_[base, qa], OPT)
                    vps.append(dict(id=f"d{d:.1f}_x{x:.1f}_y{int(dy):+d}_{arm}", group='dev' if d in DEV_D else 'holdout', d=d, x=x, yaw_offset_deg=dy,
                                    base=[round(float(v), 6) for v in base], arm=arm, q_arm=[round(float(v), 6) for v in qa], status=status,
                                    ik_point_err_deg=None if err is None else round(err, 3), cam_x_axis_z=None if xz is None else round(xz, 4),
                                    cam_pos_fk=[round(float(v), 4) for v in T[:3, 3]], cam_z_fk=[round(float(v), 4) for v in T[:3, 2]],
                                    cam_to_aim_m=round(float(np.linalg.norm(aim - T[:3, 3])), 3)))
    reg = dict(spec='第八次進度報告素材/規格/門把手視覺幾何_資料缺口與最小擷取需求_r2.md', codex='20261110_225452', door_prior=sc['door_prior'], aim=aim.tolist(),
               aim_note='0.90 m＝本包宣告的觀測中心高度先驗（不是視覺估計、不是已驗證的一般高度帶）', yaw_reference='δ＝0 面向門面法向（世界 +y）；世界偏航＝π/2＋δ',
               frames_per_viewpoint=3, groups=dict(dev=list(DEV_D), holdout=[v for v in DISTS if v not in DEV_D]), viewpoints=vps,
               counts={s: sum(1 for v in vps if v['status'] == s) for s in ('ok', 'ik_fail', 'limit_fail', 'config_collision')},
               note='量只來自門洞先驗與 URDF；不讀把手位置、遮罩或真值；可行性失敗照登錄保留、不另挑視點')
    json.dump(reg, open(out, 'w'), ensure_ascii=False, indent=1)
    print(reg['counts'], 'OBS IK 誤差最大', max(v['ik_point_err_deg'] for v in vps if v['arm'] == 'OBS'))
    for v in vps[:4] + vps[-4:]:
        print(v['id'], v['status'], v['q_arm'], v['ik_point_err_deg'], v['cam_pos_fk'], v['cam_to_aim_m'])


if __name__ == '__main__':
    main()
