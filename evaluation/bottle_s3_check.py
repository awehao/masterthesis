#!/usr/bin/env python3
"""底盤受限案例：側牆限制底盤側移，懸空方塊要求持瓶後退，手臂須改姿態。先證明手臂保持原姿態、只靠底盤過不去，再跑耦合控制名目一次。假受控對象，非物理；牆進掃描、地圖與 CBF，方塊取真值。
    python3 evaluation/bottle_s3_check.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_s1_groups_check as CK                     # noqa: E402
import bottle_s2_check as S2                            # noqa: E402
import test_bottle_s0b as T                             # noqa: E402

WALL = (np.array([0.65, -0.58, 0.25]), np.array([0.90, 0.06, 0.50]))     # x 0.20–1.10，內表面 y −0.55，高 0.50 m：掃描面看得到
LINKS = ('link3', 'link4', 'link5')
Y_FACE, D_TARGET, WALL_M = 0.33, 0.05, 0.47 + 0.05                       # 方塊 −y 面、距離目標、底盤對牆保持距離（靜態邊距＋點半徑）


def per_link(f, box):
    per = {}
    for h in f.hist[::5]:
        for n, P in G.link_world(h['q'], 0.0038).items():
            per[n] = min(per.get(n, 9.9), float(G.box_dist(P, *box).min()))
    return {k: round(v * 1e3, 1) for k, v in sorted(per.items(), key=lambda x: x[1])}


def base_wall(f):
    return round(min(float(WALL[0][1] + WALL[1][1] / 2 - 0.0) * -1 + h['q'][1] for h in f.hist if h['q'][0] > 0.1) * 1e3 - 390.0, 1) if False else round((min(h['q'][1] for h in f.hist if h['q'][0] > 0.15) + 0.55 - 0.39) * 1e3, 1)


def main():
    # 只靠底盤：手臂保持持瓶姿態，瓶最遠端 y 需 ≤ 面－距離目標；底盤 y 下限由牆決定
    q_carry = None
    y_limit = -0.55 + WALL_M
    c = SC.make_lidar(extra_boxes=[WALL])
    f = CK.fake(c, 50.0)
    r0 = CK.summary('無三維約束（側牆在場）', c, f)
    g0 = S2.box_min(f, S2.S2_BOX)
    car = [h for h in f.hist if h['phase'] == 'CARRY']
    Tt = T.K.fk(np.r_[0.0, 0.0, 0.0, car[0]['q'][3:]], 'link_tcp')
    y_rel = float(Tt[1, 3])
    y_need = Y_FACE - D_TARGET - 0.015
    print(f'  幾何：TCP 相對底盤 y {y_rel:.3f}；持瓶須 y ≤ {y_need:.3f}（面 {Y_FACE}−目標 {D_TARGET}−半徑 0.015）；底盤須 y ≤ {y_need - y_rel:.3f}，但牆使 y ≥ {y_limit:.3f} ⇒ 只靠底盤缺 {(y_limit - (y_need - y_rel)) * 1e3:.0f} mm')
    print(f"  無約束實測：{g0}；底盤 y 最低 {min(h['q'][1] for h in car):.3f}")
    c = SC.make_lidar(extra_boxes=[WALL], body_boxes=[S2.S2_BOX], body_links=LINKS)
    f = CK.fake(c, 50.0)
    r = CK.summary('耦合控制（手臂須讓路）', c, f)
    g = S2.box_min(f, S2.S2_BOX)
    car = [h for h in f.hist if h['phase'] == 'CARRY'] or f.hist[-20:]
    rel0 = T.K.fk(np.r_[0.0, 0.0, 0.0, car[0]['q'][3:]], 'link_tcp')[:3, 3]
    rel = np.array([T.K.fk(np.r_[0.0, 0.0, 0.0, h['q'][3:]], 'link_tcp')[:3, 3] - rel0 for h in car[::10]])
    Q = np.array([h['q'][3:] for h in car])
    print(f"  對方塊 {g}；分連桿 {per_link(f, S2.S2_BOX)}")
    print(f"  底盤 y 最低 {min(h['q'][1] for h in car):.3f}；持瓶相對底盤位移 y 峰 {rel[:, 1].min() * 1e3:.1f} mm、合位移峰 {np.linalg.norm(rel, axis=1).max() * 1e3:.1f} mm；關節變化峰 {np.round(np.ptp(Q, axis=0), 3).tolist()}；底盤↔牆（0.39 圓盤）{base_wall(f)} mm")
    print(f"  約束 {r['body_rows_by']}，列峰 {r['body_rows_max']}，分隔面 {r['body_planes']}；{r['qp_counts']}；GI 未採用 {r['gi_fail']}；Shield {r['nav_chain']['shield']}")
    json.dump(dict(no_constraint=g0, coupled=g, per_link=per_link(f, S2.S2_BOX), success=r['success'], stop=r['stop'], rows=r['body_rows_by'], timing=r['timing']['solve']), open(os.path.join(HERE, 'results', 'bottle', 's3_check.json'), 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    sys.exit(main())
