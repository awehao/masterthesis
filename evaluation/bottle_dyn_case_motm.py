#!/usr/bin/env python3
"""動態移動夾取 A 案：接近段橫越移動物（Codex 20261110_233102；規則在名目趟之前寫定，不依成敗調整）。
全部世界座標；障礙真值只供 Isaac 場景驅動、評估與安全核對，不送進控制器（控制器只經掃描→追蹤器）。
生成規則（只讀名目趟 NOMINAL 的量測，名目趟與 A 案同瓶位、同控制版本、同速度）：
  t_acc   ＝名目趟 tlane_accept 時刻（目標專屬車道已由視覺量測生成）。
  t_entry ＝名目趟底盤首次進入目標專屬車道入口（區域 x ≥ x_entry 且 |y − y_lane| < 0.10）的時刻。
  交會時刻 t_c＝t_entry − δ，δ 依序取 1.0、1.5、0.5、2.0 s（離 1.0 近者優先），且須 t_c ≥ t_acc＋1.0（交互發生在依視覺生成的接近上）。
  P_c＝t_c 時名目底盤位置；d＝前後 0.5 s 位移方向；n_open＝與 d 垂直、指離第一桌中心的一側（空曠側）。
  障礙：0.20 m/s、半徑 0.25 m、高 1.0 m；起點 S＝P_c＋L_in·n_open（L_in＝2.0 m，若起步時刻 < 1.0 s 則縮短到 0.20·(t_c−1.0)），
        終點 E＝P_c − L_out·n_open（L_out 依序 1.0、0.8、0.6 m），等速直線、t_c 時中心恰在 P_c（不反應就撞上）；之前靜止在 S、到 E 後靜止。
前提（全過才採用；候選依上列順序，取第一個全過者）：
  1 靜態：障礙軌跡對第一桌（區域平移後）、第二桌、原障礙方塊的距離 ≥ 半徑＋0.05，且不經過地圖佔據格（膨脹半徑＋0.05）。
  2 須反應：名目底盤（半徑 0.30）與障礙在同一時刻的最近表面距離 < 0（不反應就碰撞）。
  3 可觀測：t_c − 2.0 s 時名目底盤到障礙中心的視線不穿過靜態方塊，距離 ≤ 5.0 m。
  4 交互在接近段：最近時刻落在 [t_acc, t_entry]（取得目標之後、進入車道之前）。
    python3 evaluation/bottle_dyn_case_motm.py NOMINAL_RUN [--dry]   （--dry 只印、不寫 experiments）
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'experiments', '20261111_motm_dyn_case.json')
SPEED, RADIUS, HEIGHT = 0.20, 0.25, 1.0
DELTAS, L_OUTS, L_IN = (1.0, 1.5, 0.5, 2.0), (1.0, 0.8, 0.6), 2.0


def load(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'), allow_pickle=True)
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    plan = json.load(open(os.path.join(D, 'plan.json')))['plan']
    O = np.array(plan['start_world']) - np.array(plan['start'])
    return D, r, c, O


def statics(r, O):
    sys.path.insert(0, HERE)
    import bottle_s1_geom as G
    z1 = float(r['table1_z'])
    t1 = [('t1:' + n, c_ + np.r_[O, 0.0], s_) for n, c_, s_ in G.table_boxes(dict(G.S1_TABLE, top_z=z1))]
    t2 = [('t2:' + n, c_ + np.array([2.4, -1.0, 0.0]), s_) for n, c_, s_ in G.table_boxes(dict(G.S1_TABLE))]
    box = [('box', np.array([-2.0, 0.0, 0.25]), np.array([0.40, 0.40, 0.50]))]
    return t1, t2 + box


def state(t, CA):
    S, E = np.array(CA['start'], float), np.array(CA['end'], float)
    L = float(np.linalg.norm(E - S))
    s = min(max((float(t) - CA['t_start']) * CA['speed'], 0.0), L)
    return S + (E - S) / L * s


def seg_box(P, cc, sz):
    dx = np.maximum(np.abs(P[:, 0] - cc[0]) - sz[0] / 2, 0)
    dy = np.maximum(np.abs(P[:, 1] - cc[1]) - sz[1] / 2, 0)
    return float(np.hypot(dx, dy).min())


def map_clear(D, P, r):
    import yaml
    from PIL import Image
    y = yaml.safe_load(open(os.path.join(D, 'map.yaml')))
    img = np.array(Image.open(os.path.join(D, y['image'])))[::-1]
    res, ox, oy = float(y['resolution']), float(y['origin'][0]), float(y['origin'][1])
    occ = np.argwhere(img < 128)
    if not len(occ):
        return 9.9
    W = np.c_[ox + (occ[:, 1] + 0.5) * res, oy + (occ[:, 0] + 0.5) * res]
    d = np.min(np.linalg.norm(P[::10, None, :] - W[None, :, :], axis=2))
    return float(d) - r


def generate(run):
    D, r, c, O = load(run)
    ev = r['events']
    acc = next(e for e in ev if e['what'] == 'tlane_accept')
    t_acc, x_entry, y_lane = float(acc['t']), float(acc['x_entry']), float(acc['y_lane'])
    t, bx, by = c['t'], c['base_x'], c['base_y']
    rx, ry = bx - O[0], by - O[1]
    k_in = np.flatnonzero((rx >= x_entry) & (np.abs(ry - y_lane) < 0.10))
    t_entry = float(t[k_in[0]])
    t1, others = statics(r, O)
    tc1 = np.mean([cc[:2] for n, cc, sz in t1 if 'top' in n] or [cc[:2] for n, cc, sz in t1], axis=0)
    tried = []
    for dl in DELTAS:
        t_c = t_entry - dl
        if t_c < t_acc + 1.0:
            tried.append(dict(delta=dl, why='before_lane_accept'))
            continue
        k = int(np.argmin(np.abs(t - t_c)))
        a, b = int(np.argmin(np.abs(t - (t_c - 0.5)))), int(np.argmin(np.abs(t - (t_c + 0.5))))
        d = np.array([bx[b] - bx[a], by[b] - by[a]])
        d /= max(np.linalg.norm(d), 1e-9)
        P = np.array([bx[k], by[k]])
        n = np.array([-d[1], d[0]])
        if float(n @ (tc1 - P)) > 0:
            n = -n
        l_in = min(L_IN, SPEED * (t_c - 1.0))
        for lo in L_OUTS:
            S, E = P + l_in * n, P - lo * n
            CA = dict(name='motm_A_cross', source='bottle_dyn_case_motm（接近段橫越）', nominal=run, speed=SPEED, radius=RADIUS, height=HEIGHT,
                      start=tuple(round(float(v), 4) for v in S), end=tuple(round(float(v), 4) for v in E), t_start=round(t_c - l_in / SPEED, 3), t_meet=round(t_c, 3),
                      meet_robot=tuple(round(float(v), 4) for v in P), robot_dir=tuple(round(float(v), 4) for v in d), open_side=tuple(round(float(v), 4) for v in n),
                      delta=dl, l_in=round(l_in, 3), l_out=lo, t_acc=t_acc, t_entry=t_entry, frame='world')
            geo = check(CA, D, r, c, O, t1 + others)
            ok = all(v['ok'] for v in geo.values())
            tried.append(dict(delta=dl, l_out=lo, ok=ok, failed=[kk for kk, v in geo.items() if not v['ok']]))
            if ok:
                return CA, geo, tried
    return None, None, tried


def check(CA, D, r, c, O, boxes):
    S, E = np.array(CA['start']), np.array(CA['end'])
    line = S + np.linspace(0, 1, 800)[:, None] * (E - S)
    dmin = {n: round(seg_box(line, cc, sz) - CA['radius'], 4) for n, cc, sz in boxes}
    worst = min(dmin, key=dmin.get)
    dm = map_clear(D, line, CA['radius'])
    t = c['t']
    ob = np.array([state(tt, CA) for tt in t])
    db = np.hypot(c['base_x'] - ob[:, 0], c['base_y'] - ob[:, 1]) - CA['radius'] - 0.30
    i = int(np.argmin(db))
    k2 = int(np.argmin(np.abs(t - (CA['t_meet'] - 2.0))))
    pb, po = np.array([c['base_x'][k2], c['base_y'][k2]]), ob[k2]
    sight = pb + np.linspace(0, 1, 200)[:, None] * (po - pb)
    blocked = [n for n, cc, sz in boxes if seg_box(sight, cc, sz) <= 0.0]
    checks = {'1_static': (dmin[worst] >= 0.05 and dm >= 0.05, dict(worst=worst, min=dmin[worst], map_min=round(dm, 4))),
              '2_needs_reaction': (db[i] < 0.0, dict(min_surface=round(float(db[i]), 4), t=round(float(t[i]), 2))),
              '3_observable': (not blocked and float(np.linalg.norm(po - pb)) <= 5.0, dict(dist=round(float(np.linalg.norm(po - pb)), 3), blocked=blocked)),
              '4_in_approach': (CA['t_acc'] <= t[i] <= CA['t_entry'], dict(t_closest=round(float(t[i]), 2), window=[CA['t_acc'], CA['t_entry']]))}
    return {k: dict(ok=bool(v[0]), **v[1]) for k, v in checks.items()}


def cases():
    j = json.load(open(OUT))
    ca = dict(j['case'])
    for k in ('start', 'end', 'meet_robot', 'robot_dir', 'open_side'):
        ca[k] = tuple(ca[k])
    return {ca['name']: ca}


def main():
    run = sys.argv[1]
    dry = '--dry' in sys.argv
    CA, geo, tried = generate(run)
    print('嘗試', tried)
    if CA is None:
        print('無候選全過 ⇒ 不生成 A 案（交第一阻點）')
        return 1
    print(json.dumps(CA, ensure_ascii=False))
    for k, v in geo.items():
        print(('✓' if v['ok'] else '✗'), k, {kk: vv for kk, vv in v.items() if kk != 'ok'})
    if not dry:
        json.dump(dict(case=CA, geometry=geo, tried=tried, generated_before_runs=True, ruling='reviews/20261110_233102_reply.md'), open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
        print('寫出', OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
