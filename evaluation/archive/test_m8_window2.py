#!/usr/bin/env python3
"""m8_window2 純函式反例（Codex 20261110_194328）。
    python3 evaluation/test_m8_window2.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_window2 as W  # noqa: E402

RES = []
FREE = (lambda p: False)


def check(name, cond, info=None):
    RES.append(dict(name=name, ok=bool(cond), info=info))
    print(('PASS ' if cond else 'FAIL ') + name + ('' if info is None else f'  {info}'))


def straight(a, b, ds=0.05):
    n = int(math.ceil(np.linalg.norm(np.subtract(b, a)) / ds)) + 1
    return np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)]


def rem(path, y):
    return path[path[:, 1] >= y - 1e-9]


def wk(x, y, vx, vy, tid=1):
    return dict(tid=tid, x=x, y=y, vx=vx, vy=vy, confirmed=True)


def warm(L, t_end, pose, vel, path, make, dt=0.1):
    """以同一 tid 連續餵航跡讓它變穩定，回傳最後一次結果。"""
    r = None
    for t in np.arange(0.0, t_end + 1e-9, dt):
        r = L.update(float(t), pose, vel, path, [make(float(t))], 0.1)
    return r


def main():
    path = straight((0, -4), (0, 4))
    L = W.WindowLayer2(blocked=FREE)
    check('航跡過期 ⇒ HOLD_STALE', L.update(0, (0, -4, 1.57), (0, 0), path, [], 0.9)['mode'] == 'HOLD_STALE')
    # 新出現（不穩）、低估速度 0.3 的行人：v1 會判時窗足夠，v2 以 0.8 上界外推 ⇒ WAIT
    L = W.WindowLayer2(blocked=FREE)
    r = L.update(0, (0, -2.0, 1.57), (0, 0.3), rem(path, -2.0), [wk(6, 0, -0.3, 0)], 0.1)
    check('新出現的行人用速度上界 ⇒ WAIT（不放行）', r['mode'] == 'WAIT', r.get('mode'))
    # 停下的確認航跡（速度 0）在路徑旁 ⇒ 四周擴張圓 ⇒ WAIT（風險沒有消失）
    L = W.WindowLayer2(blocked=FREE)
    r = warm(L, 2.0, (0, -3.0, 1.57), (0, 0.3), rem(path, -3.0), lambda t: wk(2.0, 0, 0.0, 0.0))
    check('停下的確認航跡 ⇒ 不當作無風險（WAIT）', r['mode'] == 'WAIT', r.get('mode'))
    # 穩定、遠、慢、背向 ⇒ CRUISE
    L = W.WindowLayer2(blocked=FREE)
    r = warm(L, 3.0, (0, -3.0, 1.57), (0, 0.0), rem(path, -3.0), lambda t: wk(-6 - 0.5 * t, 0, -0.5, 0))
    check('穩定且背向走遠 ⇒ CRUISE', r['mode'] == 'CRUISE', r.get('mode'))
    # 在帶內、穩定行人遠 ⇒ COMMIT
    L = W.WindowLayer2(blocked=FREE)
    r = warm(L, 2.0, (0, -0.2, 1.57), (0, 0.22), rem(path, -0.2), lambda t: wk(9.2 - 0.6 * t, 0, -0.6, 0))
    check('在帶內、穩定行人遠 ⇒ COMMIT', r['mode'] == 'COMMIT', r)
    # 在帶內、行人近 ⇒ 不 COMMIT
    L = W.WindowLayer2(blocked=FREE)
    r = warm(L, 2.0, (0, 0.0, 1.57), (0, 0.0), rem(path, 0.0), lambda t: wk(4.5 - 0.6 * t, 0, -0.6, 0))
    check('在帶內、行人近 ⇒ 不 COMMIT（EXIT 或 STOP_BLOCKED）', r['mode'] in ('EXIT', 'STOP_BLOCKED'), r.get('mode'))
    # COMMIT 整段核對：在第一行人的帶內，第二行人會掃過剩餘路徑後段 ⇒ 不 COMMIT
    L = W.WindowLayer2(blocked=FREE)
    for t in np.arange(0.0, 2.01, 0.1):
        r = L.update(float(t), (0, -0.2, 1.57), (0, 0.22), rem(path, -0.2), [wk(12.0 - 0.6 * t, 0, -0.6, 0, tid=1), wk(5.0 - 0.6 * t, 1.5, -0.6, 0, tid=2)], 0.1)
    check('剩餘路徑後段會被另一行人掃到 ⇒ 不 COMMIT', r['mode'] != 'COMMIT', r.get('mode'))
    # 等待逾時
    L = W.WindowLayer2(blocked=FREE)
    p6 = straight((0, -6), (0, 4))
    r0 = L.update(0, (0, -6, 1.57), (0, 0.0), p6, [wk(0, 0, 0.0, 0.0)], 0.1)
    r = L.update(61.0, (0, -6, 1.57), (0, 0.0), p6, [wk(0, 0, 0.0, 0.0)], 0.1)
    check('路徑前方有停下的航跡、等待超過 60 s ⇒ WAIT_TIMEOUT', r0['mode'] == 'WAIT' and r['mode'] == 'WAIT_TIMEOUT', [r0['mode'], r['mode']])
    # 時間界：t_ub ≥ t_lb，且轉彎路徑 t_ub 較大
    p2 = np.vstack([straight((0, 0), (0, 1)), straight((0, 1), (1, 1))[1:]])
    s, lb, ub = W.robot_window(p2, np.array([0.0, 0.0]), 0.0)
    s1, lb1, ub1 = W.robot_window(straight((0, 0), (0, 2)), np.array([0.0, 0.0]), 0.0)
    check('通過時間上界 ≥ 下界、轉彎加時', bool((ub >= lb).all()) and ub[-1] > ub1[-1], dict(ub_turn=round(float(ub[-1]), 2), ub_straight=round(float(ub1[-1]), 2)))
    out = dict(n=len(RES), n_fail=sum(1 for x in RES if not x['ok']), results=RES)
    json.dump(out, open(os.path.join(HERE, 'results', 'm8_window2_tests.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"{out['n'] - out['n_fail']}／{out['n']} 通過")
    return 0 if out['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
