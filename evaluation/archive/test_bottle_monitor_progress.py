#!/usr/bin/env python3
"""導航段監看獨立進度的最小驗收（Codex 20261101_093009）。用 bottle_w1_dyn2_g1 實際的 Smac 導航路徑與假代價圖（指定點周圍 0.45 m 內為內切）。
  1 機器人在導航段、前方被佔 ⇒ path_blocked；
  2 參考先過入口、機器人尚未抵達（舊 sig_robot 凍結的情形），前方入口附近被佔 ⇒ 仍 path_blocked；
  3 機器人沿末段越過入口且在側向管內 ⇒ 監看結束，身後殘段（−0.926, 0）被佔不停止；
  4 投影過入口但側向在管外 ⇒ 不結束，入口前仍檢查；
  5 量測非有限 ⇒ 不推進也不結束。
    python3 evaluation/test_bottle_monitor_progress.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_monitor_progress as MP                    # noqa: E402
import bottle_n2_path as NP                             # noqa: E402
import bottle_s0b_ctrl as B                             # noqa: E402

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


class Grid:
    def __init__(self, blocked_xy, stamp=0.0):
        self.b, self.stamp = np.array(blocked_xy, float), stamp

    def cost(self, P):
        P = np.asarray(P, float)
        return np.where(np.hypot(P[:, 0] - self.b[0], P[:, 1] - self.b[1]) <= 0.45, 99, 0)


def fresh(pp):
    return NP.PathMonitor(pp), MP.MonitorProgress(pp, B.Y_TUBE)


def walk(pp, mp, sig_to, dy=0.0):
    """沿路徑每 1 cm 走到 σ＝sig_to（實跑每步只移動幾 mm，搜尋窗 0.6 m 足夠），dy：最後一點的側向偏移。"""
    for s_ in np.arange(pp.sig0, sig_to + 1e-9, 0.01):
        mp.update(np.r_[pp.xy(s_)[0], 0.0])
    p = pp.xy(sig_to)[0] + np.array([0.0, dy])
    return mp.update(np.r_[p, 0.0])


def main():
    plan = json.load(open(os.path.join(HERE, 'runs', 'bottle_w1_dyn2_g1', 'plan.json')))['plan']
    pp = NP.PlannedPath(np.array(plan['joined']), -0.9)
    mid = pp.xy(0.5 * (pp.sig0 + pp.x_lane))[0]
    mon, mp = fresh(pp)
    q = np.r_[pp.xy(pp.sig0 + 0.3)[0], 0.0]
    mp.update(q)
    mon.on_grid(Grid(mid), mp.sig)
    chk('1 導航段前方被佔 ⇒ path_blocked', mon.why is not None and mon.why['why'] == 'path_blocked', mon.why)
    mon, mp = fresh(pp)
    walk(pp, mp, pp.x_lane - 0.4)                                       # 機器人在入口前 0.4 m（參考已過入口的情形：舊 sig_robot 會凍結）
    mon.on_grid(Grid(pp.xy(pp.x_lane - 0.1)[0]), mp.sig)
    chk(f'2 機器人未抵達入口（σ {mp.sig:.3f}）、前方入口附近被佔 ⇒ 仍 path_blocked', mon.why is not None, (mp.sig, mon.why))
    mon, mp = fresh(pp)
    walk(pp, mp, -0.5)                                                  # 沿路徑走過入口到直線段 x −0.5
    mon.on_grid(Grid((-0.926, -0.003)), mp.sig)
    chk(f'3 越過入口且在側向管內 ⇒ 監看結束（done {mp.done}），身後殘段被佔不停止', mp.done and mon.why is None, (mp.report(), mon.why))
    mon, mp = fresh(pp)
    walk(pp, mp, pp.x_lane - 0.01)                                      # 沿路徑走到入口前 1 cm（仍未結束）
    mp.update(np.array([pp.x_lane + 0.05, 0.30, 0.0]))                  # 投影過入口，但 y 0.30 在側向管外
    mon.on_grid(Grid(pp.xy(pp.x_lane - 0.05)[0]), mp.sig)
    chk(f'4 側向在管外 ⇒ 不結束（done {mp.done}、σ {mp.sig:.4f}），入口前仍檢查', (not mp.done) and mp.sig < pp.x_lane and mon.why is not None, (mp.report(), mon.why))
    mon, mp = fresh(pp)
    s0 = mp.update(np.r_[pp.xy(pp.sig0 + 0.2)[0], 0.0])
    s1 = mp.update(np.array([np.nan, 0.0, 0.0]))
    chk('5 量測非有限 ⇒ 不推進不結束', s1 == s0 and not mp.done and mp.n_bad == 1, mp.report())
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for b in BAD:
        print('  失敗', b)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
