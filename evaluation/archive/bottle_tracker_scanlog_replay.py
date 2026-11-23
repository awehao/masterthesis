#!/usr/bin/env python3
"""追蹤器回放（實際掃描）：把物理趟記錄的每幀原始掃描（scan_log.npz：任務時刻、掃描時底盤位姿、360 筆距離）依序餵給追蹤器（bottle_nav_chain.NavChain，同程序），
地圖＝第一桌腳＋第二桌腳（同跑法 MAP_ARGS --table2 2.4,-1.0）。逐幀記錄發布的動態障礙、所屬軌跡與速度，對照讀回真值（s1_steps dyn_* 欄）。
同程序回放與獨立節點時序不同（沒有傳輸延遲），只看追蹤邏輯本身能否重現。
    python3 evaluation/bottle_tracker_scanlog_replay.py RUN [--cls 模組.類別]
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import test_bottle_place as TP                          # noqa: E402


def map_boxes():
    return [(b[1], b[2]) for b in G.table_boxes() if b[0] != 'table_top'] + [(c_, sz) for n, c_, sz in TP.table2_boxes() if n != 'table_top']


def replay(run, tracker_cls=None, radius=None):
    D = os.path.join(HERE, 'runs', run)
    S = np.load(os.path.join(D, 'scan_log.npz'))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    if tracker_cls is not None:
        orig = NC.ScanObstacleTracker
        NC.ScanObstacleTracker = tracker_cls
    try:
        ch = NC.NavChain(map_boxes=map_boxes())
    finally:
        if tracker_cls is not None:
            NC.ScanObstacleTracker = orig
    tr = ch.tr
    rows = []
    for t, pose, rr in zip(S['t'], S['pose'], S['ranges']):
        if t <= 0.0 and rows:
            continue
        ch.on_scan(float(t), rr, tuple(float(v) for v in pose))
        k = int(np.argmin(np.abs(c['t'] - t)))
        truth = None
        if 'dyn_x' in c and np.isfinite(c['dyn_x'][k]):
            truth = dict(x=float(c['dyn_x'][k]), y=float(c['dyn_y'][k]), vx=float(c['dyn_vx'][k]), vy=float(c['dyn_vy'][k]))
        tracks = [dict(tid=int(k_.tid), x=float(k_.kf.position[0]), y=float(k_.kf.position[1]), vx=float(k_.kf.velocity[0]), vy=float(k_.kf.velocity[1]), age=int(k_.age),
                       misses=int(k_.misses), mover=bool(k_.is_mover), reject=int(k_.reject), innov=float(k_.innov), n_pts=int(len(k_.pts))) for k_ in tr._tracks]
        rows.append(dict(t=float(t), pose=[float(v) for v in pose], dyn=[dict(d) for d in ch.dyn], stat=[dict(d) for d in ch.stat], tracks=tracks, truth=truth, ranges=rr))
    return rows


def summarize(rows, radius):
    """每幀發布的動態項：速度誤差（對讀回真值）、位置（表面點到真值圓心距 − 半徑）；真值不存在時（名目）全部算虛假。"""
    out = dict(n_scan=len(rows), n_dyn_scan=0, n_false=0, speed_err=[], false_fast=[], first_pub=None, surf_err=[])
    for r in rows:
        if not r['dyn']:
            continue
        out['n_dyn_scan'] += 1
        T = r['truth']
        for d in r['dyn']:
            sp = float(np.hypot(d['vx'], d['vy']))
            if T is None or float(np.hypot(d['x'] - T['x'], d['y'] - T['y'])) > radius + 0.30:
                out['n_false'] += 1
                if sp > 0.6:
                    out['false_fast'].append((round(r['t'], 2), round(d['x'], 3), round(d['y'], 3), round(sp, 3)))
                continue
            if out['first_pub'] is None:
                out['first_pub'] = round(r['t'], 2)
            out['speed_err'].append((round(r['t'], 2), round(sp, 3), round(float(np.hypot(T['vx'], T['vy'])), 3), round(float(np.hypot(d['vx'] - T['vx'], d['vy'] - T['vy'])), 3)))
            out['surf_err'].append(round(float(np.hypot(d['x'] - T['x'], d['y'] - T['y'])) - radius, 3))
    E = np.array([e[3] for e in out['speed_err']]) if out['speed_err'] else np.zeros(0)
    out['vel_err_p50'] = float(np.median(E)) if len(E) else None
    out['vel_err_max'] = float(E.max()) if len(E) else None
    out['n_vel_err_gt_0p5'] = int((E > 0.5).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--cls', default=None)
    ap.add_argument('--radius', type=float, default=None)
    a = ap.parse_args()
    cls = None
    if a.cls:
        m, k = a.cls.rsplit('.', 1)
        cls = getattr(importlib.import_module(m), k)
    rows = replay(a.run, cls)
    rad = a.radius
    if rad is None:
        rep = json.load(open(os.path.join(HERE, 'runs', a.run, 's1_report.json')))
        rad = float((rep.get('dyn') or {}).get('case', {}).get('radius', 0.0) or 0.0) if (rep.get('dyn') or {}).get('case') else 0.0
    s = summarize(rows, rad)
    print(json.dumps({k: (v[:8] if isinstance(v, list) else v) for k, v in s.items()}, ensure_ascii=False, default=float))
    return 0


if __name__ == '__main__':
    sys.exit(main())
