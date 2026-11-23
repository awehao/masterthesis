#!/usr/bin/env python3
"""B 分析（Codex 20261110_215340）：r5 決策切換時，造成切換的航跡是哪一類。只讀紀錄；真值只用來把航跡對到物體。
對每次 STOP_BLOCKED／EXIT／WAIT／COMMIT 進入時刻：取距機器人 4 m 內的確認航跡，記 tid、速度、方向、距該 tid 首次出現的時間、
1 s 前的速度與方向變化，以及最近的真值物體（行人或箱體；> 0.6 m 記 none）。分類：new（< 1.5 s）、speed_change（1 s 內 > 0.15 m/s）、
slow（< 0.15 m/s）、dir_change（1 s 內 > 30°）、static_misdetect（最近物體是箱體）。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_switch_tracks.py RUN_ID
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    nav = json.load(open(os.path.join(D, 'm8nav.json')))
    B = MB.read_bag(os.path.join(D, 'bag'), ['/odom', '/gmpc/tracks_debug'])
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])
    T = [(float(np.interp(r, rec, od[:, 0])), MB.tracks(m)) for r, m in B['/gmpc/tracks_debug']]
    first, hist = {}, {}
    for t, tr in T:
        for x in tr:
            if x['confirmed']:
                first.setdefault(x['tid'], t)
                hist.setdefault(x['tid'], []).append((t, x['vx'], x['vy']))
    with np.load(os.path.join(D, 'm8fw_truth.npz')) as Z:
        c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
        walkers, static = json.loads(str(Z['walkers'])), json.loads(str(Z['static']))
    rows, cls_count = [], {}
    for e in (nav.get('conflict_layer_log') or []):
        if e.get('mode') not in ('STOP_BLOCKED', 'EXIT', 'WAIT', 'COMMIT') or e.get('event'):
            continue
        t = e['t']
        j = int(np.argmin([abs(tt - t) for tt, _ in T]))
        pr = MB.pose_at(od, t)
        k = int(np.argmin(np.abs(c['t'] - t)))
        objs = [(w['name'], np.array([c[f"{w['name']}_x"][k], c[f"{w['name']}_y"][k]])) for w in walkers] + [(u['name'], np.array([u['x'], u['y']])) for u in static]
        for x in T[j][1]:
            if not x['confirmed'] or math.hypot(x['x'] - pr[1], x['y'] - pr[2]) > 4.0:
                continue
            sp = math.hypot(x['vx'], x['vy'])
            h = [hh for hh in hist.get(x['tid'], []) if t - 1.2 <= hh[0] <= t - 0.8]
            dsp = None if not h else sp - math.hypot(h[0][1], h[0][2])
            ddir = None if not h or sp < 0.05 or math.hypot(h[0][1], h[0][2]) < 0.05 else math.degrees(abs(math.atan2(math.sin(math.atan2(x['vy'], x['vx']) - math.atan2(h[0][2], h[0][1])), math.cos(math.atan2(x['vy'], x['vx']) - math.atan2(h[0][2], h[0][1])))))
            near = min(objs, key=lambda o: float(np.linalg.norm(o[1] - np.array([x['x'], x['y']]))))
            dn = float(np.linalg.norm(near[1] - np.array([x['x'], x['y']])))
            obj = near[0] if dn < 0.6 else 'none'
            cl = []
            if t - first.get(x['tid'], t) < 1.5:
                cl.append('new')
            if dsp is not None and abs(dsp) > 0.15:
                cl.append('speed_change')
            if sp < 0.15:
                cl.append('slow')
            if ddir is not None and ddir > 30:
                cl.append('dir_change')
            if obj.startswith('unknown'):
                cl.append('static_misdetect')
            for q in cl or ['stable']:
                cls_count[q] = cls_count.get(q, 0) + 1
            rows.append(dict(t=round(t, 2), mode=e['mode'], tid=x['tid'], speed=round(sp, 3), d_speed_1s=None if dsp is None else round(dsp, 3), d_dir_1s=None if ddir is None else round(ddir, 1),
                             age=round(t - first.get(x['tid'], t), 2), obj=obj, cls=cl or ['stable']))
    out = dict(run=run, n_decisions=sum(1 for e in (nav.get('conflict_layer_log') or []) if e.get('mode') in ('STOP_BLOCKED', 'EXIT', 'WAIT', 'COMMIT') and not e.get('event')),
               class_counts=cls_count, rows=rows, note='真值只用來把航跡對到物體；一個決策時刻可有多條航跡')
    json.dump(out, open(os.path.join(D, 'm8_switch_tracks.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(dict(n_decisions=out['n_decisions'], class_counts=cls_count, sample=rows[:8]), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
