#!/usr/bin/env python3
"""m8_cross（r7 候選 B）單元測試：python3 evaluation/test_m8_cross.py"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m8_cross as X  # noqa: E402
import m8_window2 as W2  # noqa: E402

FREE = (lambda p: False)
OK = []


def check(name, cond):
    OK.append(bool(cond))
    print(('PASS ' if cond else 'FAIL ') + name)


def diag_path():
    P = W2.densify(np.array([-1.0, -2.5]), np.array([1.5, 2.5]))[1:]
    yaw = np.full(len(P), math.atan2(5.0, 2.5))
    return P, yaw


def walker(x, y, vx, vy, tid=4):
    return dict(tid=tid, x=x, y=y, vx=vx, vy=vy, confirmed=True)


def main():
    P, Y = diag_path()
    rp = np.array([-1.0, -2.5])
    w = walker(3.0, 0.0, 0.6, 0.0)                                 # 中心線 y=0，往東
    c, why = X.build_cross(rp, math.pi / 2, P, Y, [(w, 'stable')], FREE, (w, 'stable'))
    check('建立候選', c is not None and why is None)
    hb, db = X.band_half(), W2.d_brake(W2.V_UB_R)
    check('交點在原路與中心線交處 (0.25,0)', np.allclose(c['xc'], [0.25, 0.0], atol=0.03))
    check('等待點在近側 hb+db', np.allclose(c['W'], [0.25, -(hb + db)], atol=1e-6))
    check('穿越段垂直於航跡方向', abs(float((c['Wp'] - c['W']) @ np.array([1.0, 0.0]))) < 1e-9)
    check('穿越長度 2(hb+db)', abs(c['len']['cross'] - 2 * (hb + db)) < 1e-3)
    check('接近與穿越維持目前朝向', np.allclose(c['yaw'][:c['i_Wp'] + 1], math.pi / 2))
    c2, why2 = X.build_cross(rp, math.pi / 2, P, Y, [(w, 'unstable')], FREE, (w, 'unstable'))
    check('方向不穩不建立', c2 is None and why2['why'] == 'direction_unstable')
    blk = (lambda p: abs(p[0] - 0.25) < 0.05 and abs(p[1]) < 0.3)
    c3, why3 = X.build_cross(rp, math.pi / 2, P, Y, [(w, 'stable')], blk, (w, 'stable'))
    check('穿越段有靜態障礙即拒絕', c3 is None and why3 == dict(why='static', seg='cross'))
    other = walker(0.25, -(hb + db), 0.0, 0.0, tid=9)                 # 等待點上有不穩航跡
    c4, why4 = X.build_cross(rp, math.pi / 2, P, Y, [(w, 'stable'), (other, 'unstable')], FREE, (w, 'stable'))
    check('等待點在帶內即拒絕', c4 is None and why4['why'] == 'wait_point_or_sweep_in_band')
    L = W2.densify(np.array([0.0, 0.0]), np.array([1.0, 0.0]))[1:]
    L2 = np.vstack([L, W2.densify(np.array([1.0, 0.0]), np.array([1.0, 1.0]))[1:]])
    s, lb, ub, parts = X.window_m(L2, np.zeros(len(L2)), np.zeros(2), 0.0, np.zeros(2))
    check('朝向不變時本體轉向為 0', parts['rot_s'] == 0.0)
    check('90° 平移轉角 = 2·V_UB_R·sin45°/A_UB', abs(parts['corner_s'] - 2 * W2.V_UB_R * math.sin(math.pi / 4) / W2.A_UB) < 1e-3)
    s2, lb2, ub2, parts2 = X.window_m(L2, np.r_[np.zeros(len(L)), np.full(len(L2) - len(L), math.pi / 2)], np.zeros(2), 0.0, np.zeros(2))
    check('計畫朝向轉 90° 計入本體轉向', abs(parts2['rot_s'] - (math.pi / 2) / W2.W_TURN) < 1e-3 and ub2[-1] > ub[-1])
    away = walker(1.5, 0.0, 0.6, 0.0)                               # 已過交點往東遠離：不衝突
    sP, lbP, ubP, _ = X.window_m(c['P'], c['yaw'], rp, math.pi / 2, np.zeros(2))
    conf, cov, first = X.conflicts(c['P'], lbP, ubP, [(away, 'stable')], W2.DT_MARGIN)
    check('行人遠離時穿越無衝突', not conf.any())
    come = walker(4.0, 0.0, -0.6, 0.0)
    conf2, cov2, first2 = X.conflicts(c['P'], lbP, ubP, [(come, 'stable')], W2.DT_MARGIN)
    check('行人迎面而來時穿越有衝突並給出細節', conf2.any() and first2['tid'] == 4)
    i0, i1 = X.band_span(c['P'], c['line'])
    check('幾何帶入出索引落在穿越段', i0 is not None and i1 is not None and c['i_W'] < i0 <= c['i_Wp'] and i1 <= c['i_Wp'] + 1)
    print(f'{sum(OK)}/{len(OK)}')
    return 0 if all(OK) else 1


if __name__ == '__main__':
    sys.exit(main())
