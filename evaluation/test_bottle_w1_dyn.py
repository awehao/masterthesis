#!/usr/bin/env python3
"""第二批移動物案例的最少幾何與一次離線回歸（Codex 20261101_090533；假受控對象＋離線射線，非物理）。
幾何（用凍結 G 名目 bottle_w1_1 的量測軌跡，假設機器人完全不反應）：
  1 障礙軌跡離所有靜態物（兩桌桌腳、桌面——圓柱高 1.0 m 會碰到 0.585 m 桌面——、方塊）≥ 半徑＋0.05 m；
  2 障礙中心離持瓶規劃路徑（w1_1 report 的 carry_path）≥ 0.74 m（不進代價圖內切格 ⇒ 不觸發 path_blocked）；
  3 最接近時刻落在 w1_1 的 CARRY_NAV 區間；底盤中心到障礙表面最小距離 < 0.69（進入動態 CBF 影響範圍）；
  4 不反應時所有連桿點雲與持瓶表面到障礙仍 > 0（不靠 CBF 也不會撞，反應距離足夠）。
閉迴路（假受控對象；掃描＝靜態盒＋移動圓柱）：凍結 G 與瞬時組（bottle_w1_nopred_ctrl）各一次，記完成、最小距離、CBF 動態列、減速與時間。
    python3 evaluation/test_bottle_w1_dyn.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_ctrl as W1                             # noqa: E402
import bottle_w1_dyn_case as DC                         # noqa: E402
import bottle_w1_nopred_ctrl as NPD                     # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import test_bottle_w1 as T                               # noqa: E402

BAD = []
CA = DC.CASE


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def seg_box_dist(p0, p1, c, sz, n=400):
    P = p0 + np.linspace(0, 1, n)[:, None] * (p1 - p0)
    dx = np.maximum(np.abs(P[:, 0] - c[0]) - sz[0] / 2, 0)
    dy = np.maximum(np.abs(P[:, 1] - c[1]) - sz[1] / 2, 0)
    return float(np.hypot(dx, dy).min())


def geometry():
    S, E = np.array(CA['start']), np.array(CA['end'])
    statics = [(n, c, sz) for n, c, sz in G.table_boxes()] + [('box', SC.OBST['c'], SC.OBST['sz'])] + [('t2_' + n, c, sz) for n, c, sz in TP.table2_boxes()]
    dmin = {n: seg_box_dist(S, E, c, sz) - CA['radius'] for n, c, sz in statics}
    worst = min(dmin, key=dmin.get)
    chk(f'1 障礙軌跡離靜態物 ≥ 0.05 m（最近 {worst} {dmin[worst]:.3f} m）', dmin[worst] >= 0.05, dmin)
    r = json.load(open(os.path.join(HERE, 'runs', 'bottle_w1_1', 's1_report.json')))
    cp = np.array(r['place']['carry_path'], float)[:, :2]
    line = S + np.linspace(0, 1, 2000)[:, None] * (E - S)
    dpath = float(np.min(np.linalg.norm(cp[:, None, :] - line[None, :, :], axis=2)))
    chk(f'2 障礙中心離持瓶規劃路徑 ≥ 0.74 m（{dpath:.3f}）', dpath >= 0.74, dpath)
    Z = np.load(os.path.join(HERE, 'runs', 'bottle_w1_1', 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t = c['t']
    lo = next(e['t'] for e in r['events'] if e.get('what') == 'place_phase' and e.get('to') == 'CARRY_NAV')
    hi = next(e['t'] for e in r['events'] if e.get('what') == 'place_phase' and e.get('to') == 'PLACE_APPROACH')
    ob = np.array([DC.state(tt)[0] for tt in t])
    db = np.hypot(c['base_x'] - ob[:, 0], c['base_y'] - ob[:, 1]) - CA['radius']
    i = int(np.argmin(db))
    chk(f'3 最接近 {t[i]:.2f} s 在持瓶區間 [{lo}, {hi}]，底盤中心到障礙表面 {db[i]:.3f} m < 0.69', lo <= t[i] <= hi and db[i] < 0.69, (t[i], db[i]))
    BP = G.bottle_points()
    best = (9.9, None, None)
    for k in range(0, len(t), 5):
        if abs(t[k] - t[i]) > 4.0:
            continue
        q = np.array([c[x][k] for x in ('base_x', 'base_y', 'base_yaw', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6')])
        pts = [P for n_, P in G.link_world(q, float(c['finger1_q'][k])).items()]
        w, x, y, z = (c[n_][k] for n_ in ('bottle_qw', 'bottle_qx', 'bottle_qy', 'bottle_qz'))
        Rb = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        pts.append((Rb @ BP.T).T + np.array([c['bottle_base_x'][k], c['bottle_base_y'][k], c['bottle_base_z'][k]]))
        P = np.vstack(pts)
        P = P[P[:, 2] <= CA['height']]
        d = float((np.hypot(P[:, 0] - ob[k, 0], P[:, 1] - ob[k, 1]) - CA['radius']).min())
        if d < best[0]:
            best = (d, float(t[k]), None)
    chk(f'4 不反應時連桿與持瓶到障礙 > 0（最小 {best[0]:.3f} m @ {best[1]:.2f} s）', best[0] > 0.0, best)
    return dict(static_min=dmin, path_min=dpath, closest_t=float(t[i]), base_surface_min=float(db[i]), carry=(lo, hi), body_min=best[0], body_min_t=best[1])


def run(cls):
    legs = [(b[1], b[2]) for b in G.table_boxes() if b[0] != 'table_top']
    scene = SC.scene_boxes() + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']

    def scan_fn(t, q):
        r = NC.ray_boxes(q[:2], float(q[2]), scene, G.LIDAR_Z)
        oc, _ = DC.state(t)
        return np.minimum(r, DC.ray_cyl(q[:2], NC.scan_angles() + float(q[2]), oc, CA['radius'], NC.RANGE_MAX))
    del legs
    orig, orig_ml = T.W1.W1PlaceController, SC.make_lidar
    T.W1.W1PlaceController = cls
    SC.make_lidar = lambda **kw: orig_ml(scan_fn=scan_fn, **kw)
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController, SC.make_lidar = orig, orig_ml
    r, _ = T.run(c, f)
    t = np.array([h['t'] for h in f.hist])
    Q = np.array([h['q'] for h in f.hist])
    ob = np.array([DC.state(tt)[0] for tt in t])
    db = np.hypot(Q[:, 0] - ob[:, 0], Q[:, 1] - ob[:, 1]) - CA['radius']
    k = np.flatnonzero(np.abs(t * 10 - np.round(t * 10)) < 1e-6)
    sp = np.hypot(np.diff(Q[k, 0]), np.diff(Q[k, 1])) / 0.1
    acc = np.abs(np.diff(sp)) / 0.1
    win = (t[k][1:-1] >= CA['t_meet'] - 3) & (t[k][1:-1] <= CA['t_meet'] + 3)
    ev = {e['what']: e['t'] for e in r['events']}
    nd = [len([o for o in (cc.get('obs') or []) if not o.get('static')]) for cc in c.compact] if getattr(c, 'compact', None) else []
    out = dict(done=bool(r['mission_done']), stop=r['stop'], t_done=ev.get('mission_done'), base_surface_min=float(db.min()), base_surface_min_t=float(t[int(np.argmin(db))]),
               dec_max_win=float(acc[win].max()) if win.any() else None, v_min_win=float(sp[1:][win].min()) if win.any() else None,
               n_dyn_cycles=int(sum(1 for x in nd if x > 0)), n_zeroed=getattr(c, 'n_zeroed', None), finite=bool(T.finite_cmds(f)))
    return out


def main():
    res = dict(case=CA, geometry=geometry())
    for name, cls in (('G_full', W1.W1PlaceController), ('G_nopred', NPD.W1PlaceController)):
        o = run(cls)
        res[name] = o
        print(f"   {name}：完成 {o['done']} 阻點 {o['stop'] and o['stop'].get('why')} DONE {o['t_done']} 底盤到障礙表面最小 {o['base_surface_min']:.3f} m @ {o['base_surface_min_t']:.2f} "
              f"錯身 ±3 s 內最低車速 {o['v_min_win']} 最大速度變化率 {o['dec_max_win']} 動態障礙進入控制週期 {o['n_dyn_cycles']} 置零 {o['n_zeroed']}", flush=True)
        chk(f'{name}：完成或受控停止、命令有限', (o['done'] or o['stop'] is not None) and o['finite'], o['stop'])
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_dyn_offline.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
