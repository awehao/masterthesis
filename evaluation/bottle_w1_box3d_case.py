#!/usr/bin/env python3
"""W1 整合趟的懸空三維障礙（封存，2026-11-01；Codex 20261101_142546）。由名目紀錄 bottle_w1_sr_nominal1 依事前規則算出，之後不再移動：
  參考時刻 τ＝名目 CARRY_NAV 期間，瓶心平面速度方向首次落在 +x ±20° 內且速率 ≥ 0.1 m/s 的時刻（持瓶段轉入近直線、行進方向與 S2 相同）；
  盒尺寸與高度沿用 S2（x 厚 0.12、y 寬 0.27、z 0.55–1.00 m，底面在 LiDAR 掃描面 0.265 m 之上）；x 置中於 τ 時瓶心；
  y 相對瓶心同 S2（名目瓶心在盒 −y 面內 0.09 m、離 +y 面 0.18 m）；
  另加條件：盒的 xy 投影不得與任一桌面（第一桌、第二桌，各外擴 0.10 m）重疊（第一版只用方向條件時取到 22.0 s、盒落在第一桌上方，已排除）。
障礙由模擬真值提供給控制器（LiDAR 看不到），不是視覺避障。
    python3 evaluation/bottle_w1_box3d_case.py   # 重算並核對與封存值一致
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF_RUN = 'bottle_w1_sr_nominal1'
SIZE = (0.12, 0.27, 0.45)
Z_C = 0.775
Y_IN = 0.09
ANG_DEG, V_MIN = 20.0, 0.1
TABLE_PAD = 0.10
BOX = ((1.142, -0.435, 0.775), (0.12, 0.27, 0.45))   # 封存值（2026-11-01 由規則算出：τ 28.0 s，名目瓶心 (1.142, −0.480)）


def tables():
    sys.path.insert(0, HERE)
    import bottle_s1_geom as G
    import test_bottle_place as TP
    return [(b[1][:2], b[2][:2]) for b in G.table_boxes() if b[0] == 'table_top'] + [(c_[:2], sz[:2]) for n, c_, sz in TP.table2_boxes() if n == 'table_top']


def overlap(c, sz, tc, ts):
    return abs(c[0] - tc[0]) < (sz[0] + ts[0]) / 2 + TABLE_PAD and abs(c[1] - tc[1]) < (sz[1] + ts[1]) / 2 + TABLE_PAD


def derive():
    r = json.load(open(os.path.join(HERE, 'runs', REF_RUN, 's1_report.json')))
    t0 = next(e['t'] for e in r['events'] if e['what'] == 'place_phase' and e.get('to') == 'CARRY_NAV')
    t1 = next(e['t'] for e in r['events'] if e['what'] == 'place_phase' and e.get('to') == 'PLACE_APPROACH')
    Z = np.load(os.path.join(HERE, 'runs', REF_RUN, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t, x, y = c['t'], c['bottle_base_x'], c['bottle_base_y']
    k = np.flatnonzero(np.abs(t * 10 - np.round(t * 10)) < 1e-6)
    for i in range(1, len(k)):
        a, b = k[i - 1], k[i]
        if not (t0 <= t[b] <= t1):
            continue
        v = np.array([x[b] - x[a], y[b] - y[a]]) / (t[b] - t[a])
        if np.hypot(*v) >= V_MIN and abs(np.degrees(np.arctan2(v[1], v[0]))) <= ANG_DEG:
            tau, xb, yb = float(t[b]), float(x[b]), float(y[b])
            y_min = yb - Y_IN
            if any(overlap((xb, y_min + SIZE[1] / 2), SIZE[:2], tc, ts) for tc, ts in tables()):
                continue
            return dict(tau=tau, bottle=[xb, yb], center=[round(xb, 3), round(y_min + SIZE[1] / 2, 3), Z_C], size=list(SIZE),
                        v_dir_deg=float(np.degrees(np.arctan2(v[1], v[0]))), t_carry_nav=t0)
    raise RuntimeError('no reference time found')


def main():
    d = derive()
    print(json.dumps(d, ensure_ascii=False))
    if BOX is not None:
        ok = np.allclose(np.r_[BOX[0], BOX[1]], np.r_[d['center'], d['size']], atol=1e-9)
        print('封存值一致' if ok else f'封存值不一致：{BOX}')
        return 0 if ok else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
