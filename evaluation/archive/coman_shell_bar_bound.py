"""夾爪殼—橫桿：**更緊的幾何距離下界**（局部加密，離線，不開模擬器）。

為什麼需要：距離節點在保持位姿報 `d = 8.69 mm`、覆蓋半徑 `rho = 13.35 mm`。
`d − rho = −4.65 mm` **只表示這組取樣證不出正間距**，
**不代表實際相交**，也不能據此新增接觸例外。

本檔只對**這一對**做分支與界限：把夾爪殼的碰撞三角形局部加密，
對每個三角形取形心到橫桿的**精確**帶號距離（沿用濾波器同一組
`_closest_local_batch` / `_inside`），配上該三角形的外接半徑：

    距離函數對點位置是 1-Lipschitz（到固定集合的距離）
    ⇒ 三角形內任一點 x：  d(x) >= d(形心) − r_c，  r_c = max|頂點 − 形心|

下界 = min over 三角形 (d(形心) − r_c)；上界 = min over 已取樣點 d。
只細分「下界仍低於目前上界」的三角形，直到上下界收斂。

**不全場重掃、不直接把 rho 改小。**
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(WS, 'src/ammr_wholebody_mpc'))
from ammr_wholebody_mpc.arm_link_distance import parse_obstacles          # noqa
from ammr_wholebody_mpc.arm_link_geometry import (                        # noqa
    link_collision_tris, obstacle_distances, sample_links_certified,
    certified_covering_radius)
from ammr_wholebody_mpc.wholebody_kinematics import WholeBodyKinematics   # noqa
from coman_zero_cmd_feasible import hold_state                            # noqa

URDF = os.path.join(HERE, 'models', 'omni_bot_wholebody_expanded.urdf')
LINK = 'uflite_gripper_link'
OBST = 'handle_bar'


def subdivide(T):
    """一個三角形 → 四個（邊中點）。"""
    a, b, c = T
    ab, bc, ca = (a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0
    return [np.array([a, ab, ca]), np.array([ab, b, bc]),
            np.array([ca, bc, c]), np.array([ab, bc, ca])]


def bound(tris, obs, tol=5.0e-5, max_tris=400000):
    """分支與界限。回傳 (下界, 上界, 迭代次數, 三角形數, 最近點)。"""
    work = [np.asarray(t, float) for t in tris]
    ub, ub_pt = np.inf, None
    it = 0
    while True:
        it += 1
        cent = np.array([t.mean(axis=0) for t in work])
        rc = np.array([max(np.linalg.norm(t - t.mean(axis=0), axis=1))
                       for t in work])
        verts = np.vstack([t for t in work])
        d_c, _, _ = obstacle_distances(cent, obs)
        d_v, _, _ = obstacle_distances(verts, obs)
        k = int(np.argmin(d_v))
        if float(d_v[k]) < ub:
            ub, ub_pt = float(d_v[k]), verts[k].copy()
        ub = min(ub, float(d_c.min()))
        lb_per = d_c - rc
        lb = float(lb_per.min())
        if ub - lb <= tol or len(work) > max_tris:
            return lb, ub, it, len(work), ub_pt
        keep = [work[i] for i in range(len(work)) if lb_per[i] < ub]
        if not keep:
            return lb, ub, it, len(work), ub_pt
        nxt = []
        for t in keep:
            nxt.extend(subdivide(t))
        work = nxt


def main() -> int:
    st = hold_state()
    K = WholeBodyKinematics.from_urdf_file(URDF)
    q = np.zeros(len(K.dof_names))
    q[0], q[1], q[2] = st['park']
    q[3:9] = st['q_arm']

    import subprocess
    specs = subprocess.run([sys.executable,
                            os.path.join(HERE, 'coman_obstacle_specs.py'),
                            str(st['unit'][0]), str(st['unit'][1])],
                           capture_output=True, text=True, check=True)
    obs_all = parse_obstacles([x for x in specs.stdout.split() if x.strip()])
    T_db = np.eye(4)
    T_db[0, 3] = st['unit'][0]
    T_db[1, 3] = st['drawer_y0'] - st['opening']
    for o in obs_all:
        if o.model == 'drawer_body':
            o.T_world_link = T_db
    bar = [o for o in obs_all if o.name == OBST]
    if not bar:
        raise SystemExit(f'找不到障礙物 {OBST}')
    b = bar[0]
    print(f'保持位姿：既有趟次 hold 相位，開度 {st["opening"]*1000:.2f} mm'
          f'（**未重跑、未開模擬器**）')
    print(f'橫桿：圓柱 r = {b.radius*1000:.2f} mm、長 {b.height*1000:.1f} mm；'
          f'世界位姿取自 drawer_body y = {T_db[1,3]:.5f}')

    xml = open(URDF, encoding='utf-8').read()
    tris_local = link_collision_tris(xml, links=[LINK])[LINK]
    T_wl = K.fk(q, LINK)
    tris = (tris_local.reshape(-1, 3) @ T_wl[:3, :3].T + T_wl[:3, 3]
            ).reshape(-1, 3, 3)
    print(f'{LINK} 碰撞三角形 {len(tris)} 個（與屏障同一組碰撞幾何）')

    # 節點目前的取樣（8 點級）給出的數字，作為對照
    S = sample_links_certified(xml, rho_target=0.015, tol=0.001)[LINK]
    W = (S.points @ T_wl[:3, :3].T) + T_wl[:3, 3]
    d_s, _, _ = obstacle_distances(W, [b])
    print(f'\n節點現行取樣：{len(S.points)} 點、rho {S.rho*1000:.2f} mm、'
          f'最近取樣距離 {float(d_s.min())*1000:.2f} mm'
          f' ⇒ 下界 {float(d_s.min() - S.rho)*1000:+.2f} mm（**證不出正間距**）')

    lb, ub, it, nt, pt = bound(tris, [b])
    print(f'\n局部加密（分支與界限，只針對 {LINK}—{OBST}）：'
          f'{it} 輪、最終 {nt} 個三角形')
    print(f'  真實最小距離下界 {lb*1000:+.4f} mm')
    print(f'  真實最小距離上界 {ub*1000:+.4f} mm  （差 {1000*(ub-lb):.4f} mm）')
    if pt is not None:
        print(f'  最近點（世界）{np.round(pt, 5).tolist()}')

    print()
    if lb > 0.0:
        print(f'**確認有間隙**：夾爪殼與橫桿的真實最小距離 ≥ {lb*1000:.3f} mm。')
        err = float(d_s.min()) - ub
        print(f'此位姿此配對的**實際取樣誤差**只有 {err*1000:.2f} mm'
              f'（取樣 {float(d_s.min())*1000:.2f} − 真實 {ub*1000:.2f}）；'
              f'rho {S.rho*1000:.2f} mm 是**整條連桿的最壞情況**覆蓋半徑，'
              f'不是此處的誤差。')
        print('先前「殼包住橫桿」的說法**不成立**，是取樣下界過保守造成的。')
        print(f'此配對仍為**非接觸配對**；現行取樣的 rho {S.rho*1000:.2f} mm '
              f'大於實際間距 {ub*1000:.2f} mm，屏障因此無法在此位姿給出零命令。')
        print('**不在此處改門檻、不新增接觸例外**；近接配置留待一次決定。')
        return 0
    if ub < 0.0:
        print(f'**確認相交**：真實最大侵入 ≥ {-ub*1000:.3f} mm。')
        print('下一步應核對碰撞模型與抓取位姿，**不得因既有軌跡經過該處就追認為設計接觸**。')
        return 2
    print('上下界仍跨過 0，**尚未判定**；需提高加密上限或收緊容差。')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
