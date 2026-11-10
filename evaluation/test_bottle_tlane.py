#!/usr/bin/env python3
"""bottle_tlane（目標專屬入口／車道）離線核對（Codex 20261110_233102 第 2 項三核對＋幾何）：python3 evaluation/test_bottle_tlane.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_n2_path as NP  # noqa: E402
import bottle_plan_worker as PW  # noqa: E402
import bottle_tlane as TL  # noqa: E402

OK = []


def check(name, cond):
    OK.append(bool(cond))
    print(('PASS ' if cond else 'FAIL ') + name)


class FreeGrid:
    def cost(self, P):
        return np.zeros(len(P), int)


class WallGrid:
    def __init__(self, y_wall):
        self.y = y_wall

    def cost(self, P):
        return np.where(np.asarray(P)[:, 1] > self.y, 100, 0)


def base_path():
    raw = np.column_stack([np.linspace(-3.6, -1.4, 111), np.linspace(0.4, 0.05, 111)])
    return NP.PlannedPath(NP.join_lane(raw, -0.9), -0.9)


def main():
    pp = base_path()
    sig_r, d_stop, win_lo, fresh = -2.06, 0.20, -0.35, 0.5
    r0 = TL.decide(None, 8.61, fresh, pp, sig_r, 0.35, d_stop, win_lo, FreeGrid(), 8.5)
    check('1 無觀測不建立目標專屬車道', not r0['ok'] and r0['why'] == 'no_observation' and 'path_new' not in r0)
    r1 = TL.decide(dict(t=7.9, xy=[-0.03, 0.44]), 8.61, fresh, pp, sig_r, 0.35, d_stop, win_lo, FreeGrid(), 8.5)
    check('2 舊觀測（0.71 s > 0.5 s）拒收', not r1['ok'] and r1['why'] == 'observation_stale')
    rf = TL.decide(dict(t=8.70, xy=[-0.03, 0.44]), 8.61, fresh, pp, sig_r, 0.35, d_stop, win_lo, FreeGrid(), 8.5)
    check('2b 未來時間戳觀測拒收', not rf['ok'] and rf['why'] == 'observation_stale')
    ra = TL.decide(dict(t=8.50, xy=[-0.07, 0.40]), 8.61, fresh, pp, sig_r, 0.35, d_stop, win_lo, FreeGrid(), 8.5)
    rb = TL.decide(dict(t=8.50, xy=[0.06, 0.46]), 8.61, fresh, pp, sig_r, 0.35, d_stop, win_lo, FreeGrid(), 8.5)
    check('3a 兩個量測位置都建立車道', ra['ok'] and rb['ok'])
    check('3b 入口隨量測 x 改變（差 0.13 m）', abs((rb['x_entry'] - ra['x_entry']) - 0.13) < 1e-9)
    check('3c 車道 y 隨量測 y 改變（差 0.06 m）', abs((rb['y_lane'] - ra['y_lane']) - 0.06) < 1e-9)
    A, Bp = ra['path_new'], rb['path_new']
    xs = np.linspace(-0.5, 0.2, 8)
    check('3d 車道段參考不同（y＝y_lane）', np.allclose(A.xy(xs)[:, 1], ra['y_lane']) and np.allclose(Bp.xy(xs)[:, 1], rb['y_lane']))
    dmax = float(np.max(np.linalg.norm(A.xy(np.linspace(-1.6, -1.0, 13) + ra['dsig0'] * 0) - Bp.xy(np.linspace(-1.6, -1.0, 13)), axis=1)))
    check('3e 接近段幾何不同（入口前最大差 > 1 cm）', dmax > 0.01)
    pre_old = pp.xy(np.linspace(pp.sig0, ra['sig_join'], 20))
    pre_new = A.xy(np.linspace(pp.sig0, ra['sig_join'], 20) + ra['dsig0'])
    check('4 重接點前與原路徑幾何一致（σ 平移 Δσ0 後 ≤ 2 cm）', float(np.max(np.linalg.norm(pre_old - pre_new, axis=1))) <= 0.02)
    T = A.tangent(np.array([ra['x_entry'] - 1e-3, ra['x_entry'] + 0.1]))
    check('5 入口處切線 +x 連續', np.allclose(T[1], [1.0, 0.0]) and float(T[0] @ np.array([1.0, 0.0])) > 0.99)
    rl = TL.decide(dict(t=12.5, xy=[-0.03, 0.44]), 12.6, fresh, pp, -0.95, 0.35, d_stop, win_lo, FreeGrid(), 12.5)
    check('6 機器人已近入口 ⇒ too_late_to_rejoin（不退回固定車道）', not rl['ok'] and rl['why'] == 'too_late_to_rejoin')
    rw = TL.decide(dict(t=8.50, xy=[-0.03, 0.47]), 8.61, fresh, pp, sig_r, 0.35, d_stop, win_lo, WallGrid(0.03), 8.5)
    check('7 新車道進代價圖內切 ⇒ 拒絕', not rw['ok'] and str(rw['why']).startswith('blocked'))
    rs = TL.decide(dict(t=8.50, xy=[-0.03, 0.44]), 11.0, 3.0, pp, sig_r, 0.35, d_stop, win_lo, FreeGrid(), 8.5)
    check('8 代價圖過期 ⇒ 拒絕', not rs['ok'] and rs['why'] == 'costmap_stale')
    rt = TL.decide(dict(t=8.50, xy=[-0.03, 0.60]), 8.61, fresh, pp, sig_r, 0.35, d_stop, win_lo, FreeGrid(), 8.5, y_tube=(-0.04, 0.10))
    check('10 車道落在放行側向管外 ⇒ 拒絕（不放寬）', not rt['ok'] and rt['why'] == 'lane_outside_release_tube')
    inp = dict(xy=[-0.03, 0.44], z_support=0.58, band=0.10, pre_m=0.05, lift_m=0.05, win_rel=[-0.35, 0.15], x_nom=0.0,
               q_seed=[0.0, 0.3, 0.8, 0.0, 0.5, 0.0])
    ok0, why0, r_y0 = PW.plan_grasp(dict(inp))
    ok1, why1, r_y1 = PW.plan_grasp(dict(inp, y_lane=0.04))
    print('   工作者 y_lane 0：', ok0, why0, r_y0.get('margin'), '；0.04：', ok1, why1, r_y1.get('margin'))
    check('9 工作者沿 y_lane 掃描（兩車道的可達餘裕不同）', (r_y0.get('margin') != r_y1.get('margin')) or (ok0 != ok1))
    print(f'{sum(OK)}/{len(OK)}')
    return 0 if all(OK) else 1


if __name__ == '__main__':
    sys.exit(main())
