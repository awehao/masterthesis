#!/usr/bin/env python3
"""m8_window3 r6b 反例（Codex 20261110_220329）：離開目標在 GMPC 到達容差內（約 0.19 m）時，(a) 已清出 ⇒ 經 T_CLEAR 回 CRUISE、不卡在 EXIT；
(b) 仍在衝突區 ⇒ 不放行 CRUISE、也不繼續等同一個舊目標。
    python3 evaluation/test_m8_window3.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_window3 as W3  # noqa: E402

RES = []


def check(name, cond, info=None):
    RES.append(dict(name=name, ok=bool(cond), info=info))
    print(('PASS ' if cond else 'FAIL ') + name + ('' if info is None else f'  {info}'))


def straight(a, b, ds=0.05):
    n = int(math.ceil(np.linalg.norm(np.subtract(b, a)) / ds)) + 1
    return np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)]


def wk(x, y, vx, vy, tid=1):
    return dict(tid=tid, x=x, y=y, vx=vx, vy=vy, confirmed=True)


def main():
    tol = 0.20
    path = straight((0, -2), (0, 4))
    G = (0.0, -1.5)
    # (a) 已清出：行人在遠處背向走開；機器人距離開目標 0.19 m
    L = W3.WindowLayer3(blocked=lambda p: False, exit_reached=tol)
    L.mode, L.hold, L.side = 'EXIT', dict(kind='exit', goal=list(G), side=1.0), 1.0
    modes = []
    for k in range(30):
        t = 0.1 * k
        r = L.update(t, (0.0, -1.5 - 0.19, 1.57), (0.0, 0.0), path[path[:, 1] >= -1.69], [wk(-6.0 - 0.6 * t, 0.0, -0.6, 0.0)], 0.1)
        modes.append(r['mode'])
    check('(a) 已清出、距目標 0.19 m ⇒ 1 s 後回 CRUISE（不卡 EXIT）', modes[-1] == 'CRUISE' and 'CRUISE' not in modes[:9], modes[::3])
    check('(a) 記錄離開執行到位', any(e.get('event') == 'exit_executed' for e in L.log), [e for e in L.log if e.get('event')][:2])
    # (b) 仍在衝突區：行人朝機器人所在路線走來；機器人距離開目標 0.19 m、仍在行人前方路線帶上
    L = W3.WindowLayer3(blocked=lambda p: False, exit_reached=tol)
    L.mode, L.hold, L.side = 'EXIT', dict(kind='exit', goal=[0.0, 0.19], side=-1.0), -1.0
    modes, goals = [], []
    for k in range(30):
        t = 0.1 * k
        r = L.update(t, (0.0, 0.0, 1.57), (0.0, 0.0), path[path[:, 1] >= 0.0], [wk(5.0 - 0.6 * t, 0.0, -0.6, 0.0)], 0.1)
        modes.append(r['mode'])
        goals.append(tuple((r.get('hold') or {}).get('goal') or ()))
    check('(b) 仍在衝突區 ⇒ 不放行 CRUISE', 'CRUISE' not in modes, sorted(set(modes)))
    check('(b) 不繼續等同一個舊目標', all(g != (0.0, 0.19) for g in goals[1:]), sorted(set(goals))[:3])
    out = dict(n=len(RES), n_fail=sum(1 for x in RES if not x['ok']), results=RES)
    json.dump(out, open(os.path.join(HERE, 'results', 'm8_window3_tests.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    print(f"{out['n'] - out['n_fail']}／{out['n']} 通過")
    return 0 if out['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
