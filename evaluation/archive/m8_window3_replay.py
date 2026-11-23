#!/usr/bin/env python3
"""r6 決策狀態機的開迴路重放（Codex 20261110_215340「先重放 25.83 案 49.13–51.73 s」）。
把 r5 該趟錄到的 /odom、/gmpc/tracks_debug、/global_costmap/costmap 與當時節點用的巡航路徑（/plan 中終點＝門前觀察區的 Smac 路徑）依時間餵給
WindowLayer2（r5 原狀態機）與 WindowLayer3（r6），同窗口比較：模式切換次數、保持型模式（STOP_BLOCKED／HOLD_STALE／WAIT stop_now）進出次數、
隱含重規劃（EXIT／STOP 回 CRUISE／COMMIT）次數、EXIT 目標重建次數。機器人照 r5 實際軌跡走（開迴路），只驗決策邏輯、不代表安全。
    source /opt/ros/jazzy/setup.bash; python3 evaluation/m8_window3_replay.py RUN_ID T_A T_B
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_bag as MB  # noqa: E402
import m8_window2 as W2  # noqa: E402
import m8_window3 as W3  # noqa: E402

GOAL = (0.5, -1.543)
HOLDISH = ('STOP_BLOCKED', 'HOLD_STALE')


def main():
    run, ta, tb = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    D = os.path.join(HERE, 'runs', run)
    B = MB.read_bag(os.path.join(D, 'bag'), ['/odom', '/gmpc/tracks_debug', '/global_costmap/costmap', '/plan'])
    od = MB.odom_series(B['/odom'])
    rec = np.array([x[0] for x in B['/odom']])

    def to_sim(r):
        return float(np.interp(r, rec, od[:, 0]))
    plans = []
    for r, m in B['/plan']:
        if len(m.poses) < 10:
            continue
        P = np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses])
        if math.hypot(P[-1, 0] - GOAL[0], P[-1, 1] - GOAL[1]) < 0.05:
            plans.append((to_sim(r), P))
    cms = sorted(((MB.stamp(m), m) for _, m in B['/global_costmap/costmap']), key=lambda z: z[0])
    cur = {'m': None}

    def blocked(p):
        m = cur['m']
        if m is None:
            return True
        i = m.info
        ix, iy = int((p[0] - i.origin.position.x) / i.resolution), int((p[1] - i.origin.position.y) / i.resolution)
        if not (0 <= ix < i.width and 0 <= iy < i.height):
            return True
        c = m.data[iy * i.width + ix]
        return c >= 99 or c < 0
    out = {}
    for name, cls in (('v2_r5', W2.WindowLayer2), ('v3_r6', W3.WindowLayer3)):
        L = cls(blocked=blocked)
        modes, prev, k_proj, P_cur = [], None, 0, None
        hold_toggles, replans, exit_rebuilds, last_goal = 0, 0, 0, None
        for r, m in B['/gmpc/tracks_debug']:
            t = to_sim(r)
            if t < ta - 3.0 or t > tb:                                   # 前 3 s 暖機航跡歷史
                continue
            cur['m'] = next((x[1] for x in reversed(cms) if x[0] <= t), None)
            pp = [x for x in plans if x[0] <= t]
            if not pp:
                continue
            if P_cur is None or pp[-1][1] is not P_cur:
                P_cur, k_proj = pp[-1][1], 0
            pr = MB.pose_at(od, t)
            j = int(np.searchsorted(od[:, 0], t))
            j = min(j, len(od) - 1)
            j0 = int(np.searchsorted(od[:, 0], t - 0.2))
            vel = ((od[j, 1] - od[j0, 1]) / max(od[j, 0] - od[j0, 0], 1e-6), (od[j, 2] - od[j0, 2]) / max(od[j, 0] - od[j0, 0], 1e-6))
            d = np.linalg.norm(P_cur[k_proj:k_proj + 40] - pr[1:3], axis=1)
            if len(d):
                k_proj += int(np.argmin(d))
            res = L.update(t, (pr[1], pr[2], pr[3]), vel, P_cur[k_proj:], MB.tracks(m), tracks_age=0.0)
            mode = res['mode'] if res['mode'] != 'WAIT' else ('WAIT_stop_now' if (res.get('hold') or {}).get('kind') == 'stop_now' else 'WAIT')
            if t < ta:
                prev = mode
                continue
            if mode == 'EXIT':
                g = tuple((res.get('hold') or {}).get('goal') or [])
                if last_goal is not None and g != last_goal:
                    exit_rebuilds += 1
                last_goal = g
            if prev is not None and mode != prev:
                modes.append((round(t, 2), prev, mode))
                ih_p, ih_n = (prev in HOLDISH or prev == 'WAIT_stop_now'), (mode in HOLDISH or mode == 'WAIT_stop_now')
                if ih_p != ih_n:
                    hold_toggles += 1
                if prev in ('EXIT',) + HOLDISH and mode in ('CRUISE', 'COMMIT'):
                    replans += 1
            prev = mode
        out[name] = dict(n_mode_changes=len(modes), hold_toggles=hold_toggles, implied_replans=replans, exit_goal_rebuilds=exit_rebuilds,
                         kept=getattr(L, 'kept', None), changes=modes[:30])
    res = dict(run=run, window=[ta, tb], note='開迴路重放：機器人照 r5 實際軌跡；只驗決策邏輯，不代表安全', **out)
    json.dump(res, open(os.path.join(D, f'm8_window3_replay_{ta:.2f}_{tb:.2f}.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    for k in ('v2_r5', 'v3_r6'):
        print(k, {kk: out[k][kk] for kk in ('n_mode_changes', 'hold_toggles', 'implied_replans', 'exit_goal_rebuilds', 'kept')})
        print('   ', out[k]['changes'][:12])


if __name__ == '__main__':
    main()
