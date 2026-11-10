#!/usr/bin/env python3
"""第八次第一包：不動作接線檢查的核對（WIRING=1 那趟；只讀紀錄，真值只用來核對）。
1 行人照世界軌跡走：實際位置對目標位置的誤差（t ≥ 1 s）；機器人保持不動。
2 原始掃描打到每個行人與箱體：各物體「至少一點落在表面 0.10 m 內」的掃描比例（看不到的物體會是 0，照列，不推論遮擋）。
3 /scan_filtered：行人附近的點應被拿掉、箱體附近的點應保留。
4 追蹤器：/gmpc/tracks_debug 中，行人 0.5 m 內有 confirmed 航跡的比例與速度估計；箱體 0.5 m 內 confirmed 航跡（應少）。
  /gmpc/obstacles（動態 CBF）與 /gmpc/static_obstacles（靜態 CBF）點落在各物體附近的比例。
5 代價圖：最後一張 /global_costmap/costmap 在各箱體覆蓋格的最大代價（≥ 99＝內切以上）。
6 話題頻率。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_wiring_check.py RUN_ID
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402

NEAR = 0.10


def rate(msgs):
    if len(msgs) < 2:
        return None
    t = np.array([x[0] for x in msgs])
    return round((len(t) - 1) / (t[-1] - t[0]), 2)


def flat_pts(m, k):
    d = np.asarray(m.data, float)
    return d.reshape(-1, k)[:, :2] if len(d) else np.zeros((0, 2))


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    tr, static, walkers = MB.load_truth(D)
    B = MB.read_bag(os.path.join(D, 'bag'))
    od = MB.odom_series(B['/odom'])
    rec2sim = (np.array([x[0] for x in B['/odom']]), od[:, 0])                  # 無 stamp 的話題：bag 接收時刻 → 模擬時間（模擬與牆鐘速率不同，不能用固定偏移）

    def to_sim(t_rec):
        return float(np.interp(t_rec, rec2sim[0], rec2sim[1]))
    out = dict(run=run, n_msgs={k: len(v) for k, v in B.items()}, rates={k: rate(v) for k, v in B.items()})
    k1 = tr['t'] >= 1.0
    out['walker_follow'] = {w['name']: round(float(np.hypot(tr[f"{w['name']}_x"][k1] - tr[f"{w['name']}_tx"][k1], tr[f"{w['name']}_y"][k1] - tr[f"{w['name']}_ty"][k1]).max()), 4) for w in walkers}
    out['robot_disp_max_m'] = round(float(np.hypot(od[:, 1] - od[0, 1], od[:, 2] - od[0, 2]).max()), 4)
    polys = {u['name']: MB.box_poly(u) for u in static}

    def near_counts(topic, getpts):
        res = {n: 0 for n in list(polys) + [w['name'] for w in walkers]}
        n = 0
        for t_rec, m in B.get(topic, []):
            ts = MB.stamp(m) if hasattr(m, 'header') else to_sim(t_rec)
            P = getpts(m, ts)
            if P is None:
                continue
            n += 1
            for nm, V in polys.items():
                if len(P) and (MB.poly_dist(P, V) <= NEAR).any():
                    res[nm] += 1
            for w in walkers:
                c = MB.walker_at(tr, w['name'], ts)
                if len(P) and (np.linalg.norm(P - c, axis=1) <= w['radius'] + NEAR).any():
                    res[w['name']] += 1
        return dict(n=n, frac={k: round(v / max(n, 1), 3) for k, v in res.items()})

    def scan_pts(m, ts):
        return MB.scan_points(m, MB.pose_at(od, ts))[0]
    out['scan_raw'] = near_counts('/scan', scan_pts)
    out['scan_filtered'] = near_counts('/scan_filtered', scan_pts)

    def dyn_pts(m, ts):
        return flat_pts(m, 5)

    def stat_pts(m, ts):
        d = np.asarray(m.data, float)
        return d.reshape(-1, 3)[:, :2] if len(d) and len(d) % 3 == 0 else (flat_pts(m, 5) if len(d) % 5 == 0 else None)
    out['cbf_dynamic_pts'] = near_counts('/gmpc/obstacles', lambda m, ts: dyn_pts(m, ts))
    out['cbf_static_pts'] = near_counts('/gmpc/static_obstacles', lambda m, ts: stat_pts(m, ts))
    # 追蹤航跡
    trk = {w['name']: dict(n=0, hit=0, speed_est=[], speed_true=w['speed_mps']) for w in walkers}
    box_conf = {n: 0 for n in polys}
    box_speed = {n: [] for n in polys}
    n_tr = 0
    for t_rec, m in B.get('/gmpc/tracks_debug', []):
        T = MB.tracks(m)
        n_tr += 1
        ts = to_sim(t_rec)
        for w in walkers:
            c = MB.walker_at(tr, w['name'], ts)
            near = [x for x in T if x['confirmed'] and math.hypot(x['x'] - c[0], x['y'] - c[1]) <= 0.5]
            trk[w['name']]['n'] += 1
            if near:
                trk[w['name']]['hit'] += 1
                trk[w['name']]['speed_est'].append(math.hypot(near[0]['vx'], near[0]['vy']))
        for nm, V in polys.items():
            wc = [MB.walker_at(tr, w['name'], ts) for w in walkers]
            nb = [x for x in T if x['confirmed'] and MB.poly_dist(np.array([[x['x'], x['y']]]), V)[0] <= 0.5
                  and all(math.hypot(x['x'] - c[0], x['y'] - c[1]) > w['radius'] + 0.3 for c, w in zip(wc, walkers))]   # 靠近行人的航跡歸給行人
            if nb:
                box_conf[nm] += 1
                box_speed[nm].append(max(math.hypot(x['vx'], x['vy']) for x in nb))
    out['tracks'] = dict(n=n_tr, walkers={k: dict(frac_confirmed_near=round(v['hit'] / max(v['n'], 1), 3), speed_est_med=None if not v['speed_est'] else round(float(np.median(v['speed_est'])), 3),
                                                     speed_true=v['speed_true']) for k, v in trk.items()},
                         boxes_confirmed_frac={k: round(v / max(n_tr, 1), 3) for k, v in box_conf.items()},
                         boxes_confirmed_speed={k: None if not v else dict(med=round(float(np.median(v)), 3), max=round(float(np.max(v)), 3)) for k, v in box_speed.items()})
    # 代價圖
    cm = B.get('/global_costmap/costmap', [])
    if cm:
        m = cm[-1][1]
        i = m.info
        A = np.asarray(m.data, np.int16).reshape(i.height, i.width)
        res = {}
        for nm, V in polys.items():
            xs = np.arange(V[:, 0].min(), V[:, 0].max(), i.resolution / 2)
            ys = np.arange(V[:, 1].min(), V[:, 1].max(), i.resolution / 2)
            X, Y = np.meshgrid(xs, ys)
            P = np.c_[X.ravel(), Y.ravel()]
            P = P[MB.poly_dist(P, V) == 0.0]
            ix = np.floor((P[:, 0] - i.origin.position.x) / i.resolution).astype(int)
            iy = np.floor((P[:, 1] - i.origin.position.y) / i.resolution).astype(int)
            c = A[iy, ix]
            res[nm] = dict(max=int(c.max()), frac_ge99=round(float((c >= 99).mean()), 3))
        wres = {}
        ts = MB.stamp(m)
        for w in walkers:
            cxy = MB.walker_at(tr, w['name'], ts)
            ix = int((cxy[0] - i.origin.position.x) / i.resolution)
            iy = int((cxy[1] - i.origin.position.y) / i.resolution)
            wres[w['name']] = int(A[iy, ix]) if 0 <= ix < i.width and 0 <= iy < i.height else None
        out['costmap_last'] = dict(t=ts, boxes=res, walker_center_cost=wres)
    json.dump(out, open(os.path.join(D, 'm8_wiring_check.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
