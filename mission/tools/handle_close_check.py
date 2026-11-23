#!/usr/bin/env python3
"""抽屜把手閉爪診斷：閉爪前後 TCP 相對橫桿真值（夾爪座標：x 接近方向、y 兩指開合方向、z）、兩指開度與受力。用法：handle_close_check.py <run_dir> <t_close>"""
import json, math, os, sys
os.environ.setdefault('M8_SCENE_REG', '20261121_m8_house_r2w3_scene.json')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'evaluation'))
import numpy as np
import door_exit_ctrl as XE
import drawer_scene as DRS
K = XE.K
RD, t0 = sys.argv[1], float(sys.argv[2])
r = json.load(open(RD + '/result.json'))
DRS.set_pose(*DRS.offset_pose(0.04, -0.02, 2.0))
bar = DRS.SPEC['drawer']['handle']['bar']
z = np.load(RD + '/log.npz'); c = [str(x) for x in z['cols']]; L = z['L']; t = L[:, c.index('t')]
DL = np.array(r['drawer']['log'], float); MK = np.array(r['marker']['log'], float); mc = r['marker']['log_cols']
for dt in (-1.0, -0.5, -0.2, 0.0, 0.1, 0.2, 0.3):
    i = int(np.searchsorted(t, t0 + dt)); q = L[i, [c.index('q%d' % k) for k in range(9)]]
    T = K.fk(q, 'link_tcp'); op = float(np.interp(t[i], DL[:, 0], DL[:, 1]))
    hb = np.asarray(DRS.to_world(bar['center'], op, True), float)
    d = T[:3, :3].T @ (hb - T[:3, 3])
    j = int(np.clip(np.searchsorted(MK[:, 0], t[i]), 0, len(MK) - 1))
    print('t %.2f 偏差(夾爪座標) x %+6.1f y %+6.1f z %+6.1f mm  開度 %.4f/%.4f  受力 %.1f/%.1f N  抽屜開度 %.4f' % (
        t[i], d[0] * 1e3, d[1] * 1e3, d[2] * 1e3, L[i, c.index('finger1')], L[i, c.index('finger2')], MK[j, mc.index('f1_n')], MK[j, mc.index('f2_n')], op))
