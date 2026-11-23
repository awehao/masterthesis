#!/usr/bin/env python3
"""A 包（Howard 10/27「門兩邊都有門把」）：通過途中抓**內側把手**（門扇 −n 面）拉門關的可行域（離線運動學；門規格 door_local_v3、幾何 door_geom_v3）。
抓取：正面夾法基準 T_HG（door_coopen_reach_v1.T_HG＝實跑 door_wide_w2c_30_1 的 grasp_basis；外側把手上用過的同一套）套到內側把手 prim 座標：
  TCP＝H_in(q)·T_HG；H_in 為外側把手的鏡像（離開門面方向指向楔形側）。拉向楔形區（−n）⇒ 鉸鏈力矩 +z（關門）。
淨空後端（雙把手）：door_k3m_geom_monitor.clearance 的同一套近似，形狀換成 door_geom_v3.shapes（含內側立柱）、橫桿距離取兩支膠囊的較小者；
  自碰沿用既有後端（與門無關）；底盤圓盤對 v3 形狀 ≥ 20 mm。手指類照原後端不含橫桿（指定接觸＝手指—內側橫桿）。
格點：門角 {47.9, 40, 30, 20, 12, 6}°；底盤 x 0.30–1.30、y −0.90–0.50（0.10 m）、偏航 45° 一格；多種子 IK＋關節餘裕 ≥ 0.05＋非手指 ≥ 5 mm（含兩支橫桿）＋手指 ≥ 5 mm
  （對門扇／立柱／牆）＋自碰 ≥ 5 mm＋底盤 ≥ 20 mm。報：可行數、門洞內（0.30 ≤ x ≤ 0.75、y ≥ −0.30）可行數、y 最大的可行狀態、失敗原因分布。
界線：抓取可達性（靜態），不含接近／閉爪路徑、通過中的連續路徑與接觸力；Isaac 門資產尚未加入內側把手。

    python3 -u evaluation/door_coopen_inner_grasp_v1.py   # 寫 results/door/door_coopen_inner_grasp_v1.json（每門角另寫 .partial.json；不覆寫）
"""
import json
import math
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import door_geom_v3 as G3                              # noqa: E402
import door_coopen_reach_v1 as RV                      # noqa: E402
import door_coopen_traj_v1 as TJ                       # noqa: E402
import door_k3m_geom_monitor as GM                     # noqa: E402

OUT = os.path.join(HERE, 'results', 'door', 'door_coopen_inner_grasp_v1.json')
D3 = yaml.safe_load(open(os.path.join(WS, 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml')))
C = RV.C
H = float(D3['handle']['height'])
THG = RV.T_HG()
QS = (47.9, 40.0, 30.0, 20.0, 12.0, 6.0)
Q_CLOSED = RV.Q_FINGER


def world_fn(qf, names, q_finger):
    out = {}
    for n in names:
        T = C.K.fk(qf, n)
        P = C.CL[n] + (C.FINGER_AXIS[n] * q_finger if n in C.FINGER_AXIS else 0.0)
        out[n] = (T[:3, :3] @ P.T).T + T[:3, 3]
    return out


def geo3(qf, q, q_finger):
    sh = G3.shapes(D3, q, H)
    shapes = [(name, (lambda P, c=c, R=R, hs=hs: G3.box_dist(P, c, R, hs))) for name, c, R, hs in sh]
    bar = lambda P: G3.bar_capsule_dist_both(D3, P, q, H)          # noqa: E731

    def disc(qf_):
        best = (math.inf, None)
        for name, c, R, hs in sh:
            d = float(G3.disc_clear(qf_[0], qf_[1], c[2], C.RB, c, R, hs))
            if d < best[0]:
                best = (d, name)
        return best
    out = GM.clearance(qf, q, q_finger, world_fn, shapes, bar, disc, C.ARM_LINKS)
    out['self'] = TJ.GEO(qf, q, q_finger)['self']
    return out


def main():
    if os.path.exists(OUT):
        raise SystemExit(f'{OUT} 已存在，不覆寫')
    t0 = time.time()
    res = {'schema': 'door_coopen_inner_grasp/1', 'door_spec': 'door_local_v3.yaml', 'grasp': 'T_HG（w2c_30_1 正面夾法）套內側把手鏡像座標', 'per_q': []}
    for qd in QS:
        q = math.radians(qd)
        T = G3.handle_pose_inner(D3, q) @ THG
        feas, why = [], {}
        for x in np.round(np.arange(0.30, 1.301, 0.10), 3):
            TJ.cool()
            for y in np.round(np.arange(-0.90, 0.501, 0.10), 3):
                d0 = geo3([float(x), float(y), 0.0] + [0.0, 0.0, 0.035, 0.0, -1.5707963, 0.0], q, Q_CLOSED)['base']
                if d0[0] < 0.02:
                    why['base_disc'] = why.get('base_disc', 0) + 8
                    continue
                for yd in range(0, 360, 45):
                    b = [float(x), float(y), math.radians(yd)]
                    qf9 = C.ik(np.asarray(b), T, None, only_prev=False)
                    if qf9 is None:
                        why['ik'] = why.get('ik', 0) + 1
                        continue
                    qa = np.asarray(qf9)[C.IDX]
                    jm = float(min(min(qa - TJ.V2.ELO + 0.05), min(TJ.V2.EHI + 0.05 - qa)))
                    if jm < 0.05:
                        why['joint_margin'] = why.get('joint_margin', 0) + 1
                        continue
                    g = geo3(b + list(qa), q, Q_CLOSED)
                    bad = next((f'{k}:{g[k][1]}' for k, lim in (('nonfinger', 0.005), ('finger', 0.005), ('self', 0.005), ('base', 0.02)) if g[k][0] < lim), None)
                    if bad is not None:
                        why[bad] = why.get(bad, 0) + 1
                        continue
                    feas.append({'x': float(x), 'y': float(y), 'yaw_deg': yd, 'qa': [float(v) for v in qa], 'jm': jm,
                                 'clear': {k: float(g[k][0]) for k in ('nonfinger', 'finger', 'self', 'base')}})
        indoor = [f for f in feas if 0.30 <= f['x'] <= 0.75 and f['y'] >= -0.30]
        top = sorted(feas, key=lambda f: -f['y'])[:5]
        res['per_q'].append({'q_deg': qd, 'handle_inner': G3.handle_pose_inner(D3, q)[:3, 3].tolist(), 'n_feasible': len(feas), 'n_in_doorway': len(indoor),
                             'y_max': top[0]['y'] if top else None, 'top_by_y': top, 'why': dict(sorted(why.items(), key=lambda z: -z[1])[:8]),
                             'cells': [[f['x'], f['y'], f['yaw_deg'], round(min(f['clear'].values()) * 1e3, 1)] for f in feas]})
        print(f"門 {qd}°：可行 {len(feas)}、門洞內 {len(indoor)}；最大 y {res['per_q'][-1]['y_max']}；"
              f"例 {[(f['x'], f['y'], f['yaw_deg'], round(min(f['clear'].values()) * 1e3, 1)) for f in top[:3]]}；原因 {res['per_q'][-1]['why']}", flush=True)
        json.dump(res, open(OUT.replace('.json', '.partial.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    res['wall_s'] = time.time() - t0
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    main()
