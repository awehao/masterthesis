#!/usr/bin/env python3
"""F1 返回段單一移動物的案例生成與幾何前提（Codex 20261103_044351；只讀 bottle_f1_2 紀錄，不跑 Isaac；生成法同 bottle_dyn_case_region，事前固定規則，不依成敗調整）。
交會：返回段 NAV 量測軌跡弧長分率 f 處的時刻 t_meet、位置 P、前後 0.5 s 位移方向 d。候選順序：f＝0.5、0.4、0.6、0.3、0.7、0.2、0.8；每個 f 先右側再左側（行進方向的法向 0.80 m）。取第一個前提全過者。
障礙：沿用 arena 交通 dyn_obs_0（0.30 m/s、半徑 0.25 m、高 1.0 m）。沿 −d 反向等速（迎面錯身）；起點 S＝P＋2.2 d＋0.80 n、終點 E＝P−2.5 d＋0.80 n；t_start＝t_meet−2.2／0.30（任務時刻；之前靜止在 S，到 E 後靜止在 E）。世界軌跡與排程固定，不追機器人。
前提：
  1 障礙軌跡（含兩端靜止點）離靜態物（兩桌桌面與桌腳、原障礙方塊）≥ 半徑＋0.05。
  2 障礙中心離返回路徑（該趟實測 NAV 軌跡）≥ 0.74 m（不進代價圖內切格，不觸發 path_blocked）。
  3 最接近時刻落在 NAV 區間，底盤中心到障礙表面 < 0.69（進入動態 CBF 影響範圍）。
  4 不反應時連桿點到障礙表面 > 0。
  5 靜止點與軌跡不擋其他段：S、E 與整條軌跡離第一輪接近、第一輪持瓶、第二輪接近、第二輪持瓶的底盤軌跡 ≥ 0.74 m（這些段走的時候障礙在靜止點，或依時刻在軌跡上）；並核任務各段實際時刻的底盤到障礙表面距離，非 NAV 段 ≥ 0.69（只有返回 NAV 段形成交互）。
    python3 evaluation/bottle_f1_dyn_case.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402

NOMINAL = 'bottle_f1_2'
OUT = os.path.join(HERE, 'experiments', '20261103_f1_dyn_case.json')
SPEED, RADIUS, HEIGHT, LATERAL, AHEAD, BEHIND = 0.30, 0.25, 1.0, 0.80, 2.2, 2.5
F_ORDER = (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8)
D = os.path.join(HERE, 'runs', NOMINAL)
R0 = json.load(open(os.path.join(D, 's1_report_r0.json')))
Z = np.load(os.path.join(D, 's1_steps.npz'))
c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
t, x, y = c['t'], c['base_x'], c['base_y']
rs = {e['to']: e['t'] for e in R0['events'] if e['what'] == 'return_state'}
LO, HI = rs['NAV'], rs['ROT1']


def generate(frac, side):
    idx = np.flatnonzero((t >= LO) & (t <= HI))
    s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x[idx]), np.diff(y[idx])))]
    k = idx[int(np.argmin(np.abs(s - s[-1] * frac)))]
    tm = float(t[k])
    a_, b_ = int(np.argmin(np.abs(t - (tm - 0.5)))), int(np.argmin(np.abs(t - (tm + 0.5))))
    d = np.array([x[b_] - x[a_], y[b_] - y[a_]])
    d /= np.linalg.norm(d)
    n = np.array([d[1], -d[0]]) if side == 'right' else np.array([-d[1], d[0]])
    P = np.array([x[k], y[k]])
    S, E = P + AHEAD * d + LATERAL * n, P - BEHIND * d + LATERAL * n
    return dict(name='f1_return_pass1', source='arena_traffic dyn_obs_0（同 w1_dyn_pass1 生成法）', nominal=NOMINAL, speed=SPEED, radius=RADIUS, height=HEIGHT, start=tuple(round(float(v), 4) for v in S), end=tuple(round(float(v), 4) for v in E),
                t_start=round(tm - AHEAD / SPEED, 3), t_meet=round(tm, 3), lateral=LATERAL, side=side, frac=frac, meet_robot=tuple(round(float(v), 4) for v in P), robot_dir=tuple(round(float(v), 4) for v in d), nav=[LO, HI], frame='world',
                rule='t_meet＝返回 NAV 實測軌跡弧長分率 frac 處；法向 0.80 m（先右後左）反向等速；起點前 2.2 m、終點後 2.5 m')


def geometry(CA):
    z1 = float(json.load(open(os.path.join(D, 's1_report.json')))['table1_z'])
    t1 = [('t1:' + n, c_, s_) for n, c_, s_ in G.table_boxes(dict(G.S1_TABLE, top_z=z1))]
    t2 = [('t2:' + n, c_ + np.array([2.4, -1.0, 0.0]), s_) for n, c_, s_ in G.table_boxes(dict(G.S1_TABLE))]
    box = [('box', np.array([-2.0, 0.0, 0.25]), np.array([0.40, 0.40, 0.50]))]
    S, E = np.array(CA['start']), np.array(CA['end'])
    line = S + np.linspace(0, 1, 2000)[:, None] * (E - S)

    def seg_box(cc, sz):
        return float(np.hypot(np.maximum(np.abs(line[:, 0] - cc[0]) - sz[0] / 2, 0), np.maximum(np.abs(line[:, 1] - cc[1]) - sz[1] / 2, 0)).min())
    dmin = {n: round(seg_box(cc, sz) - CA['radius'], 4) for n, cc, sz in t1 + t2 + box}
    worst = min(dmin, key=dmin.get)
    ob = np.array([DC.state(tt, CA)[0] for tt in t])
    db = np.hypot(x - ob[:, 0], y - ob[:, 1]) - CA['radius']
    nav = (t >= LO) & (t <= HI)
    i = int(np.argmin(np.where(nav, db, 9e9)))
    dpath = float(np.min(np.linalg.norm(np.c_[x[nav], y[nav]][::20][:, None, :] - line[None, ::10, :], axis=2)))
    best = (9.9, None)
    for k in np.flatnonzero((t >= t[i] - 4.0) & (t <= t[i] + 4.0))[::5]:
        q = np.array([c[n_][k] for n_ in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        P = np.vstack(list(G.link_world(q, float(c['finger1_q'][k])).values()))
        P = P[P[:, 2] <= CA['height']]
        dd = float((np.hypot(P[:, 0] - ob[k, 0], P[:, 1] - ob[k, 1]) - CA['radius']).min())
        if dd < best[0]:
            best = (round(dd, 4), round(float(t[k]), 2))
    other = ~nav
    dother = float(db[other].min())
    i_o = int(np.flatnonzero(other)[int(np.argmin(db[other]))])
    Po = np.c_[x[other], y[other]][::20]
    drest = min(float(np.linalg.norm(Po - S, axis=1).min()), float(np.linalg.norm(Po - E, axis=1).min()))
    thr = CA['radius'] + 0.39 + 0.05 + 0.05
    return {'1_static': dict(ok=dmin[worst] >= 0.05, worst=worst, min=dmin[worst]), '2_path': dict(ok=dpath >= thr, min=round(dpath, 4), thr=thr), '3_closest_in_nav': dict(ok=bool(db[i] < 0.69), t=round(float(t[i]), 2), base_surface=round(float(db[i]), 4)),
            '4_no_reaction_links': dict(ok=best[0] > 0.0, min=best[0], t=best[1]), '5_other_segments': dict(ok=bool(dother >= 0.69 and drest >= thr), base_surface_min=round(dother, 4), at_t=round(float(t[i_o]), 2), rest_to_other_path=round(drest, 4), thr=thr)}


def main():
    tried = []
    CA, geo, ok = None, None, False
    for f in F_ORDER:
        for side in ('right', 'left'):
            CA = generate(f, side)
            geo = geometry(CA)
            ok = all(v['ok'] for v in geo.values())
            tried.append(dict(frac=f, side=side, ok=ok, failed=[k for k, v in geo.items() if not v['ok']]))
            if ok:
                break
        if ok:
            break
    out = dict(case=CA, geometry=geo, all_ok=ok, tried=tried, nominal_nav=[LO, HI], generated_before_runs=True, ruling='reviews/20261103_044351_reply.md')
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    print('嘗試', [(x_['frac'], x_['side'], x_['failed']) for x_ in tried])
    print(json.dumps(CA, ensure_ascii=False))
    for k, v in geo.items():
        print(('✓' if v['ok'] else '✗'), k, {kk: vv for kk, vv in v.items() if kk != 'ok'})
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
