#!/usr/bin/env python3
"""瓶子 N2 路徑純邏輯驗收（無 ROS）：弧長進度連續、直線段相切、投影不跳段、核對與監看的拒絕條件。
    python3 evaluation/test_bottle_n2_path.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_n2_path as NP                             # noqa: E402

BAD, N = [], [0]


def chk(name, cond, info=''):
    N[0] += 1
    if not cond:
        BAD.append(f'{name} {info}')


def detour(n=200):
    x = np.linspace(-3.6, -0.9, n)
    y = -0.75 * np.sin(np.pi * (x + 3.6) / 2.7) ** 2
    y[-1] = 0.0
    return np.column_stack([x, y])


def grid(boxes=(), stamp=10.0):
    res, ox, oy, w, h = 0.05, -5.0, -3.0, 170, 120
    a = np.zeros((h, w), np.int16)
    xs, ys = ox + (np.arange(w) + 0.5) * res, oy + (np.arange(h) + 0.5) * res
    for (cx, cy, sx, sy, v) in boxes:
        a[np.ix_(np.abs(ys - cy) <= sy / 2, np.abs(xs - cx) <= sx / 2)] = v
    return NP.Grid(a.ravel(), res, ox, oy, w, h, stamp)


def t_path():
    P = NP.join_lane(detour(), -0.9)
    pp = NP.PlannedPath(P, -0.9)
    chk('終點在直線段入口', np.linalg.norm(P[-1] - [-0.9, 0.0]) < 1e-9, P[-1])
    sig = np.arange(pp.sig0, 0.5, 0.005)
    X = pp.xy(sig)
    step = np.linalg.norm(np.diff(X, axis=0), axis=1)
    chk('位置連續：σ 每 5 mm 位移 ≤ 5 mm', step.max() <= 0.005 + 1e-6, step.max())
    chk('σ 是弧長：位移≈σ 增量', abs(step.sum() - (sig[-1] - sig[0])) < 0.01, (step.sum(), sig[-1] - sig[0]))
    T = pp.tangent(sig)
    ang = np.degrees(np.arctan2(T[:, 1], T[:, 0]))
    j = int(np.searchsorted(sig, -0.9))
    chk('入口前後切線連續（≤ 3°）', abs(ang[j - 1] - ang[j]) <= 3.0, (ang[j - 1], ang[j]))
    chk('直線段 y＝0、切線 +x', np.allclose(pp.xy([-0.5, 0.0, 0.8])[:, 1], 0.0) and np.allclose(pp.tangent([0.0]), [[1.0, 0.0]]))
    Ts = pp.T                                            # 切線是每 20 mm 一段的分段常數：逐段比
    turn = np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(Ts[:, 1], Ts[:, 0])))))
    chk('規劃段相鄰 20 mm 段轉角 ≤ 3°（含 Bézier 接點，無折角）', turn.max() <= 3.0, turn.max())
    chk('起點不動', np.linalg.norm(pp.xy(pp.sig0)[0] - detour()[0]) < 1e-9)
    chk('dy 恆 0（σ 是弧長，速度不換算）', pp.dy(-2.0) == 0.0 and pp.dy(0.2) == 0.0)


def t_project():
    pp = NP.PlannedPath(NP.join_lane(detour(), -0.9), -0.9)
    s = pp.sig0
    for sig in np.arange(pp.sig0, -0.5, 0.05):
        p = pp.xy(sig)[0] + np.array([0.0, 0.03])
        s = pp.project(p, s)
    chk('沿路徑投影一路跟上', abs(s - (-0.5 - 0.05 + 0.05)) < 0.06 or abs(s - np.arange(pp.sig0, -0.5, 0.05)[-1]) < 0.03, s)
    s0 = pp.sig0
    for _ in range(5):
        s0 = pp.project(pp.xy(pp.sig0)[0], s0)
    chk('站在起點時投影停在 σ0，不往回退', abs(s0 - pp.sig0) < 1e-9, s0 - pp.sig0)
    h = pp.sig0 + 0.5
    s2 = pp.project(pp.xy(pp.sig0 + 2.5)[0], h)
    chk('只在提示附近找，不跳到遠段', s2 <= h + 0.6 + 1e-9, (s2, h))


def t_validate():
    P = detour()
    g = grid()
    ok, why = NP.validate_plan('map', P, P[0], P[-1], g)
    chk('正常路徑通過', ok, why)
    chk('frame 不對拒絕', not NP.validate_plan('odom', P, P[0], P[-1], g)[0])
    chk('起點偏離拒絕', not NP.validate_plan('map', P, P[0] + [0.3, 0.0], P[-1], g)[0])
    chk('終點偏離拒絕', not NP.validate_plan('map', P, P[0], P[-1] + [0.0, 0.2], g)[0])
    chk('取樣間距過大拒絕', not NP.validate_plan('map', P[::40], P[0], P[-1], g)[0])
    gb = grid([(-2.25, -0.75, 0.3, 0.3, 100)])
    chk('路徑上有致命格拒絕', not NP.validate_plan('map', P, P[0], P[-1], gb)[0])
    chk('點數不足拒絕', not NP.validate_plan('map', P[:1], P[0], P[-1], g)[0])
    Pn = P.copy()
    Pn[5, 0] = np.nan
    chk('非有限拒絕', not NP.validate_plan('map', Pn, P[0], P[-1], g)[0])
    chk('地圖外（未知）當擋住', NP.blocked(g.cost(np.array([[9.0, 9.0]]))).all())


def t_monitor():
    pp = NP.PlannedPath(NP.join_lane(detour(), -0.9), -0.9)
    m = NP.PathMonitor(pp)
    m.on_grid(grid(stamp=10.0), pp.sig0)
    chk('乾淨代價圖不失效', m.why is None and m.n_check == 1)
    chk('代價圖新鮮不失效', m.check_age(11.5, pp.sig0) is None)
    p = pp.xy(pp.sig0 + 1.5)[0]
    m.on_grid(grid([(p[0], p[1], 0.2, 0.2, 99)], stamp=12.0), pp.sig0 + 0.5)
    chk('剩餘段被佔到內切 ⇒ path_blocked', m.why is not None and m.why['why'] == 'path_blocked', m.why)
    m2 = NP.PathMonitor(pp)
    m2.on_grid(grid([(p[0], p[1], 0.2, 0.2, 99)], stamp=12.0), pp.sig0 + 1.8)
    chk('已走過的障礙不算', m2.why is None, m2.why)
    m3 = NP.PathMonitor(pp)
    m3.on_grid(grid(stamp=10.0), pp.sig0)
    chk('代價圖過期 ⇒ costmap_stale', (m3.check_age(12.5, pp.sig0) or {}).get('why') == 'costmap_stale')
    m4 = NP.PathMonitor(pp)
    m4.on_grid(grid(stamp=10.0), 0.0)
    chk('進直線段後不再要求代價圖', m4.check_age(30.0, 0.0) is None)


def main():
    for f in (t_path, t_project, t_validate, t_monitor):
        f()
    print(f'{N[0] - len(BAD)}／{N[0]} 通過')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
