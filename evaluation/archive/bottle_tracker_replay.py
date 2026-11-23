#!/usr/bin/env python3
"""追蹤器誤判定位（Codex 20261101_080232 (c)；不改演算法、不開新實跑）：沿既有實跑的量測底盤軌跡，離線重建 10 Hz 掃描（軸對齊盒射線，同 bottle_s1_cand.make_lidar），
餵原始 scan_obstacle_tracker（bottle_nav_chain.NavChain，同程序），比較兩種地圖（只有第一桌腳＝bottle_w1c_1 的錯誤地圖；含第二桌腳＝正確地圖），
並和該趟實際記錄的障礙清單（s1_cbf.npz obs_json）對照。離線射線不是 PhysX，時序也不是獨立節點；只用來看誤判能否由幾何與追蹤邏輯重現。
    python3 evaluation/bottle_tracker_replay.py [bottle_w1c_1]
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
import test_bottle_place as TP                          # noqa: E402


def recorded(run):
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_cbf.npz'), allow_pickle=True)
    out = []
    for t, o in zip(Z['t'], Z['obs_json']):
        dyn = [x for x in json.loads(str(o)) if not x.get('static')]
        out.append((float(t), dyn))
    return out


def replay(run, map_boxes, scene, t_max=None, trace_x=None):
    """trace_x：記錄 x > trace_x 區域的掃描命中點與追蹤器內部軌跡（位置、速度、年齡、歷史長度、是否移動物、拒絕碼）。"""
    Z = np.load(os.path.join(HERE, 'runs', run, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    ch = NC.NavChain(map_boxes=map_boxes)
    rows, nxt = [], 0.0
    for i, t in enumerate(c['t']):
        if t_max is not None and t > t_max:
            break
        if t + 1e-9 < nxt:
            continue
        nxt = t + 1.0 / NC.SCAN_HZ
        q = (float(c['base_x'][i]), float(c['base_y'][i]), float(c['base_yaw'][i]))
        r = NC.ray_boxes(np.array(q[:2]), q[2], scene, G.LIDAR_Z)
        ch.on_scan(float(t), r, q)
        tr = None
        if trace_x is not None:
            a = NC.scan_angles() + q[2]
            ok = np.isfinite(r)
            P = np.c_[q[0] + r[ok] * np.cos(a[ok]), q[1] + r[ok] * np.sin(a[ok])]
            tr = dict(hits=np.round(P[P[:, 0] > trace_x], 3).tolist(),
                      tracks=[dict(x=round(float(k.kf.position[0]), 3), y=round(float(k.kf.position[1]), 3), vx=round(float(k.kf.velocity[0]), 3), vy=round(float(k.kf.velocity[1]), 3),
                                   age=int(k.age), hist=len(k.hist), mover=bool(getattr(k, 'is_mover', False)), reject=getattr(k, 'reject', None)) for k in ch.tr._tracks if k.kf.position[0] > trace_x])
        rows.append((float(t), q, [dict(d) for d in ch.dyn], len(ch.stat), tr))
    return rows


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else 'bottle_w1c_1'
    legs1 = [(b[1], b[2]) for b in G.table_boxes() if b[0] != 'table_top']
    legs2 = [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']
    scene = legs1 + [(SC.OBST['c'], SC.OBST['sz'])] + legs2
    rec = recorded(run)
    res = {}
    for tag, mb in (('map_table1_only', legs1), ('map_with_table2', legs1 + legs2)):
        R = replay(run, mb, scene, t_max=9.0, trace_x=1.5)
        ev = [(t, q, d) for t, q, d, _, _ in R if d]
        tw = [dict(t=round(t, 2), robot=[round(v, 3) for v in q], **tr) for t, q, d, _, tr in R if any(abs(t - e[0]) <= 0.55 for e in ev)]
        res[tag] = dict(n_scan=len(R), n_dyn_scans=len(ev), trace=tw, dyn=[dict(t=round(t, 2), robot=[round(v, 3) for v in q], dyn=[{k: round(float(v), 3) for k, v in x.items()} for x in d]) for t, q, d in ev])
        print(f'[{tag}] 掃描 {len(R)}，含動態障礙的掃描 {len(ev)}')
        for t, q, d in ev[:12]:
            print(f'   {t:5.2f} 車 ({q[0]:+.2f},{q[1]:+.2f},{np.degrees(q[2]):+.1f}°) 動態 {[(round(x["x"], 2), round(x["y"], 2), round(x["vx"], 2), round(x["vy"], 2)) for x in d]}')
    rd = [(t, d) for t, d in rec if d and t <= 9.0]
    print(f'[實錄 {run}] 含動態障礙的控制週期 {len(rd)}；時段 {rd[0][0] if rd else None}–{rd[-1][0] if rd else None}')
    res['recorded'] = [dict(t=round(t, 2), dyn=[{k: round(float(v), 3) for k, v in x.items() if k in ('x', 'y', 'vx', 'vy')} for x in d]) for t, d in rd]
    json.dump(dict(run=run, note='離線射線＋原始追蹤器同程序重播；非 PhysX、非獨立節點時序', **res), open(os.path.join(HERE, 'results', 'bottle', f'tracker_replay_{run}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
