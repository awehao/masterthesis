#!/usr/bin/env python3
"""S2 最小案例離線驗證：底盤可過、持瓶會撞的三維障礙。先確認 G 有幾何衝突，再加預測距離約束跑名目一次。假受控對象，非物理；障礙取模擬真值，LiDAR 看不到。
    python3 evaluation/bottle_s2_check.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_ctrl as S1                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import test_bottle_s0b as T                             # noqa: E402
import bottle_s1_groups_check as CK                     # noqa: E402

S2_BOX = (np.array([0.62, 0.465, 0.775]), np.array([0.12, 0.27, 0.45]))   # 懸空方塊 z 0.55–1.00，y 0.33–0.60：掃描面 0.265 m 之上，底盤圓盤在 y≈0 不受影響


def box_min(f, box):
    """真值評估：全部手臂點雲與持瓶對方塊的最小距離。"""
    rel, out = None, dict(arm=9.9, bottle=9.9, base=9.9)
    BP = G.bottle_points()
    for h in f.hist[::5]:
        q = h['q']
        for n, P in G.link_world(q, 0.0038).items():
            out['arm'] = min(out['arm'], float(G.box_dist(P, *box).min()))
        if h['phase'] in ('LIFT', 'CARRY'):
            Tt = T.K.fk(q, 'link_tcp')
            if rel is None:
                T0 = np.eye(4)
                T0[:3, 3] = [T.BX, T.BY, T.TOP]
                rel = np.linalg.inv(Tt) @ T0
            Tb = Tt @ rel
            Pw = (Tb[:3, :3] @ BP.T).T + Tb[:3, 3]
            out['bottle'] = min(out['bottle'], float(G.box_dist(Pw, *box).min()))
        dy = max(abs(q[1] - box[0][1]) - box[1][1] / 2, 0.0)
        dx = max(abs(q[0] - box[0][0]) - box[1][0] / 2, 0.0)
        out['base'] = min(out['base'], float(np.hypot(dx, dy)) - 0.30)
    return {k: round(v * 1e3, 1) for k, v in out.items()}


def main():
    res = {}
    for name, kw in (('G 無約束（確認衝突）', {}), ('S2 預測距離約束', dict(body_boxes=[S2_BOX]))):
        c = SC.make_lidar(**kw)
        f = CK.fake(c, 50.0)
        r = CK.summary(name, c, f)
        g = box_min(f, S2_BOX)
        q = np.array([h['q'][:3] for h in f.hist])
        car = [h for h in f.hist if h['phase'] == 'CARRY']
        drift = max(float(np.linalg.norm(T.K.fk(np.r_[0.0, 0.0, 0.0, h['q'][3:]], 'link_tcp')[:3, 3] - T.K.fk(np.r_[0.0, 0.0, 0.0, car[0]['q'][3:]], 'link_tcp')[:3, 3])) for h in car[::10]) if car else None
        print(f"  對三維障礙最小距離 mm {g}；約束列峰 {r['body_rows_max']}；底盤 y 範圍 {q[:, 1].min():.3f}…{q[:, 1].max():.3f}；持瓶相對底盤位移峰 {None if drift is None else round(drift * 1e3, 1)} mm；{r['qp_counts']}；GI 未採用 {r['gi_fail']}", flush=True)
        res[name] = dict(success=r['success'], stop=(r['stop'] or {}).get('why'), geom=g, n_fallback=r['n_fallback'], ms=r['timing']['solve'], t_exit=r['t_exit_in'])
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 's2_check.json'), 'w'), ensure_ascii=False, indent=1, default=float)


if __name__ == '__main__':
    sys.exit(main())
