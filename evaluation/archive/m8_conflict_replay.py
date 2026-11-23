#!/usr/bin/env python3
"""第八次第一包候選的離線重播（開迴路）：把 m8nav_1 錄到的 /plan、/odom、/gmpc/tracks_debug、/global_costmap/costmap 依時間餵給 m8_conflict.ConflictLayer。
機器人仍照當趟實際軌跡走，所以只回答「衝突層會在何時做出何種決定、停點或離開點在哪」，不回答「照做之後是否避免碰撞」（那要閉迴路）。
對照：短窗定位的首次包絡侵入時刻與事後最後可避讓時刻（m8_short_window.json）。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_conflict_replay.py RUN_ID
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402
import m8_conflict as CF  # noqa: E402


def main():
    run = sys.argv[1]
    D = os.path.join(HERE, 'runs', run)
    sw = json.load(open(os.path.join(D, 'm8_short_window.json')))['summary']
    B = MB.read_bag(os.path.join(D, 'bag'), ['/odom', '/plan', '/gmpc/tracks_debug', '/global_costmap/costmap'])
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])

    def to_sim(r):
        return float(np.interp(r, rec, od[:, 0]))
    plan = np.array([[p.pose.position.x, p.pose.position.y] for p in B['/plan'][0][1].poses])
    cms = sorted((MB.stamp(m), m) for _, m in B['/global_costmap/costmap'])
    cur = {'m': None}

    def blocked(p):
        m = cur['m']
        if m is None:
            return True                                         # 代價圖缺失 ⇒ 擋住（失效關閉）
        i = m.info
        ix, iy = int((p[0] - i.origin.position.x) / i.resolution), int((p[1] - i.origin.position.y) / i.resolution)
        if not (0 <= ix < i.width and 0 <= iy < i.height):
            return True
        c = m.data[iy * i.width + ix]
        return c >= 99 or c < 0
    L = CF.ConflictLayer(blocked=blocked)
    k_proj = 0
    rows = []
    for r_, m in B['/gmpc/tracks_debug']:
        t = to_sim(r_)
        if t < 18.7 or t > sw['t_first_intrusion'] + 1.0:
            continue
        cur['m'] = next((x[1] for x in reversed(cms) if x[0] <= t), None)
        pr = MB.pose_at(od, t)
        j = int(np.searchsorted(od[:, 0], t))
        j0 = int(np.searchsorted(od[:, 0], t - 0.2))
        j = min(j, len(od) - 1)
        vel = ((od[j, 1] - od[j0, 1]) / max(od[j, 0] - od[j0, 0], 1e-6), (od[j, 2] - od[j0, 2]) / max(od[j, 0] - od[j0, 0], 1e-6))
        d = np.linalg.norm(plan[k_proj:k_proj + 40] - pr[1:3], axis=1)
        k_proj += int(np.argmin(d))
        res = L.update(t, (pr[1], pr[2], pr[3]), vel, plan[k_proj:], MB.tracks(m), tracks_age=0.0)
        rows.append(dict(t=round(t, 2), mode=res['mode'], n_movers=res.get('n_movers'), robot=[round(pr[1], 3), round(pr[2], 3)]))
    first = {}
    for r in rows:
        first.setdefault(r['mode'], r['t'])
    out = dict(run=run, compare=dict(t_first_intrusion=sw['t_first_intrusion'], last_escape_exists=sw['last_escape_exists']), first_time_by_mode=first,
               log=L.log, params=dict(V_MOVER=CF.V_MOVER, T_H=CF.T_H, T_LONG=CF.T_LONG, DT_WIN=CF.DT_WIN, R_BODY=CF.R_BODY, R_BASE=CF.R_BASE, MARGIN=CF.MARGIN, SIG0=CF.SIG0,
                                      SIG_V=CF.SIG_V, V_NOM=CF.V_NOM, V_EXIT=CF.V_EXIT, T_DELAY=CF.T_DELAY, A_BRAKE=CF.A_BRAKE, note='開發假設'),
               note='開迴路重播：機器人照當趟實際軌跡，只看決策時機與停點；不代表照做後能避免碰撞')
    json.dump(out, open(os.path.join(D, 'm8_conflict_replay_r2.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(json.dumps(dict(compare=out['compare'], first=first, log=L.log[:8]), ensure_ascii=False, indent=1, default=float))


if __name__ == '__main__':
    main()
