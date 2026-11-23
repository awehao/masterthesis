#!/usr/bin/env python3
"""連續交接的單一移動物案例（Codex 20261103_133750；只讀假世界 cont 軌跡，不跑 Isaac；規則同 bottle_f1_dyn_case，事前固定，不依成敗調整）。
範圍：前往第二瓶途中、走廊外的東房段（返回 NAV 起點到進走廊）。不放走廊內對向，也不挑戰接觸中動態閃避。
交會：該段實測軌跡弧長分率 f 處的時刻 t_meet、位置 P、前後 0.5 s 位移方向 d。候選順序 f＝0.5、0.4、0.6、0.3、0.7、0.2、0.8；每個 f 先右側再左側。
障礙：同 arena 交通 dyn_obs_0（0.30 m/s、半徑 0.25 m、高 1.0 m），沿 −d 反向等速；S＝P＋2.2 d＋0.80 n、E＝P−2.5 d＋0.80 n；t_start＝t_meet−2.2／0.30。世界軌跡與排程固定，不追機器人。
前提：
  1 障礙軌跡離靜態物（兩桌桌腳、方塊、四面牆）≥ 半徑＋0.05。
  2 障礙中心離整條 NAV 軌跡 ≥ 0.74 m（不進代價圖內切格、不觸發 path_blocked）。
  3 最接近時刻落在該段，底盤中心到障礙表面 < 0.69（進入動態 CBF 影響範圍）。
  4 其他段不被擋：依時刻算全程底盤中心到障礙表面，該段以外 ≥ 0.49（底盤圓盤 0.39＋0.10）。
    python3 evaluation/bottle_f1c_dyn_case.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_f1_lchan_scene as LS                      # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402

SCN = LS.load()
Z = np.load(os.path.join(HERE, 'runs', 'lchan_fake_' + SCN['name'], 'traj_cont.npz'), allow_pickle=True)
t, Q = Z['t'], Z['q']
ret = {str(b): float(a) for a, b in Z['ret']}
x, y = Q[:, 0], Q[:, 1]
SPEED, RADIUS, HEIGHT, LATERAL, AHEAD, BEHIND = 0.30, 0.25, 1.0, 0.80, 2.2, 2.5
F_ORDER = (0.5, 0.4, 0.6, 0.3, 0.7, 0.2, 0.8)
t_sw = float(Z['t_switch'])
ic = LS.in_corridor(Q[:, :2])
LO = ret['NAV']
HI = float(t[np.flatnonzero((t >= LO) & ic)[0]])          # 進走廊前
BOXES = [('t1:' + n, c_, s_) for n, c_, s_ in G.table_boxes() if n != 'table_top'] + [('t2:' + n, c_ + np.array([SCN['table2_x'], -1.0, 0.0]), s_) for n, c_, s_ in G.table_boxes() if n != 'table_top'] + [('box', np.array([-2.0, 0.0, 0.25]), np.array([0.4, 0.4, 0.5]))] + [('wall:' + n, c_, s_) for n, c_, s_ in SCN['walls']]


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
    rd = lambda v_: tuple(round(float(u_), 4) for u_ in v_)   # noqa: E731
    return dict(name='f1c_transit_pass1', source='arena_traffic dyn_obs_0（同 w1_dyn_pass1 生成法）', nominal='假世界 cont 軌跡（場景 v2）', speed=SPEED, radius=RADIUS, height=HEIGHT, start=rd(S), end=rd(E), t_start=round(tm - AHEAD / SPEED, 3), t_meet=round(tm, 3), lateral=LATERAL, side=side, frac=frac, meet_robot=rd(P), robot_dir=rd(d), nav=[LO, HI],
                rule='t_meet＝前往第二瓶途中東房段實測軌跡弧長分率 frac 處；法向 0.80 m（先右後左）反向等速；起點前 2.2 m、終點後 2.5 m')


def geometry(CA):
    S, E = np.array(CA['start']), np.array(CA['end'])
    line = S + np.linspace(0, 1, 2000)[:, None] * (E - S)
    dst = {n: round(float(np.hypot(np.maximum(np.abs(line[:, 0] - c_[0]) - s_[0] / 2, 0), np.maximum(np.abs(line[:, 1] - c_[1]) - s_[1] / 2, 0)).min()) - CA['radius'], 4) for n, c_, s_ in BOXES}
    worst = min(dst, key=dst.get)
    ob = np.array([DC.state(tt, CA)[0] for tt in t[::5]])
    tt_, xx, yy = t[::5], x[::5], y[::5]
    db = np.hypot(xx - ob[:, 0], yy - ob[:, 1]) - CA['radius']
    win = (tt_ >= LO) & (tt_ <= HI)
    i = int(np.argmin(np.where(win, db, 9e9)))
    other = float(np.where(win, 9e9, db).min())
    nav = (t >= LO) & (t <= float(ret.get('NAV', LO)) + 60.0) & (t <= t_sw)
    d_path = float(np.min(np.linalg.norm(line[::20, None, :] - np.column_stack([x[nav][::10], y[nav][::10]])[None, :, :], axis=2)))
    chk = {'1 離靜態物': dst[worst] >= 0.05, '2 中心離路徑': d_path >= 0.74, '3 交會在段內且進 CBF 範圍': bool(db[i] < 0.69 and LO <= tt_[i] <= HI), '4 其他段不被擋': other >= 0.49}
    return dict(ok=all(chk.values()), checks=chk, static_min=dst[worst], static_worst=worst, d_center_path=round(d_path, 3), base_center_surface=round(float(db[i]), 3), at_t=round(float(tt_[i]), 2), other_min=round(other, 3))


log = []
pick = None
for f in F_ORDER:
    for side in ('right', 'left'):
        CA = generate(f, side)
        g = geometry(CA)
        log.append(dict(frac=f, side=side, **{k: g[k] for k in ('ok', 'static_min', 'static_worst', 'd_center_path', 'base_center_surface', 'at_t', 'other_min')}))
        print(f, side, g['ok'], g['static_worst'], g['static_min'], g['d_center_path'], g['base_center_surface'], g['other_min'], CA['start'], CA['end'], flush=True)
        if g['ok'] and pick is None:
            pick = (CA, g)
if pick is None:
    print('沒有候選符合規則；不放寬，交回裁定')
    sys.exit(1)
json.dump(dict(case=pick[0], geometry=pick[1], candidates=log, window=dict(nav=LO, corridor_in=HI, t_switch=t_sw)), open(os.path.join(HERE, 'experiments', '20261103_f1c_dyn_case.json'), 'w'), ensure_ascii=False, indent=1, default=float)
print('採用', pick[0]['frac'], pick[0]['side'], pick[1])
