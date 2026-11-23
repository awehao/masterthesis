#!/usr/bin/env python3
"""m8_window 純函式反例（Codex 20261110_190030）。
    python3 evaluation/test_m8_window.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_window as W  # noqa: E402

RES = []
FREE = (lambda p: False)


def check(name, cond, info=None):
    RES.append(dict(name=name, ok=bool(cond), info=info))
    print(('PASS ' if cond else 'FAIL ') + name + ('' if info is None else f'  {info}'))


def straight(a, b, ds=0.05):
    n = int(math.ceil(np.linalg.norm(np.subtract(b, a)) / ds)) + 1
    return np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)]


def rem(path, y):
    """剩餘路徑（節點傳的是投影點之後的路徑）。"""
    return path[path[:, 1] >= y - 1e-9]


def wk(x, y, vx, vy):
    return dict(tid=1, x=x, y=y, vx=vx, vy=vy, confirmed=True)


def main():
    path = straight((0, -4), (0, 4))
    L = W.WindowLayer(blocked=FREE)
    check('航跡過期 ⇒ HOLD_STALE', L.update(0, (0, -4, 1.57), (0, 0), path, [], 0.9)['mode'] == 'HOLD_STALE')
    check('沒有航跡訊息 ⇒ HOLD_STALE', L.update(0, (0, -4, 1.57), (0, 0), path, [], None)['mode'] == 'HOLD_STALE')
    # 行人將經過前方路徑、機器人在帶外 ⇒ WAIT，停點在衝突區前
    L = W.WindowLayer(blocked=FREE)
    r = L.update(0, (0, -4, 1.57), (0, 0.3), path, [wk(4, 0, -0.6, 0)], 0.1)
    ok = r['mode'] == 'WAIT' and r['hold']['kind'] == 'wait'
    check('前方衝突、在帶外 ⇒ WAIT', ok, r.get('hold'))
    if ok:
        st = np.array(r['hold']['stop'])
        check('停點不在任何帶內', W.first_arrival(st, wk(4, 0, -0.6, 0)) is None, st.tolist())
        check('停點在衝突區入口之前', r['hold']['i_stop'] < r['hold']['i_a'], (r['hold']['i_stop'], r['hold']['i_a']))
    # 行人已走遠（背向）⇒ T_CLEAR 後 CRUISE
    away = [wk(-3, 0, -0.6, 0)]
    ra = L.update(10.0, (0, -1.3, 1.57), (0, 0), path, away, 0.1)
    rb = L.update(11.1, (0, -1.3, 1.57), (0, 0), path, away, 0.1)
    check('行人背向走遠：滿 1 s 才回 CRUISE', ra['mode'] == 'WAIT' and rb['mode'] == 'CRUISE', [ra['mode'], rb['mode']])
    # 時窗足夠（行人遠、慢）⇒ 不等
    L = W.WindowLayer(blocked=FREE)
    r = L.update(0, (0, -1.5, 1.57), (0, 0.3), path, [wk(12, 0, -0.3, 0)], 0.1)
    check('時窗足夠 ⇒ CRUISE（不等）', r['mode'] == 'CRUISE', r['mode'])
    # 已在帶內、行人遠 ⇒ COMMIT
    L = W.WindowLayer(blocked=FREE)
    r = L.update(0, (0, -0.3, 1.57), (0, 0.3), rem(path, -0.3), [wk(6.5, 0, -0.6, 0)], 0.1)
    check('已在帶內、行人遠 ⇒ COMMIT', r['mode'] == 'COMMIT', r)
    # 已在帶內、行人近 ⇒ EXIT（側移）；兩側都擋 ⇒ STOP_BLOCKED
    L = W.WindowLayer(blocked=FREE)
    r = L.update(0, (0, 0.0, 1.57), (0, 0), rem(path, 0.0), [wk(3.2, 0, -0.6, 0)], 0.1)
    check('已在帶內、行人近 ⇒ EXIT 或 STOP_BLOCKED（不 COMMIT）', r['mode'] in ('EXIT', 'STOP_BLOCKED'), r)
    L = W.WindowLayer(blocked=FREE)
    r = L.update(0, (0, -0.3, 1.57), (0, 0), rem(path, -0.3), [wk(6.0, 0, -0.6, 0)], 0.1)
    check('已在帶內、行人中距、往前來不及 ⇒ EXIT（側移來得及）或 COMMIT', r['mode'] in ('EXIT', 'COMMIT') and (r['mode'] == 'COMMIT' or abs(r['hold']['goal'][0]) < 1e-6), r)
    L = W.WindowLayer(blocked=lambda p: abs(p[1]) > 0.0)
    r = L.update(0, (0, 0.0, 1.57), (0, 0), rem(path, 0.0), [wk(3.2, 0, -0.6, 0)], 0.1)
    check('兩側都擋 ⇒ STOP_BLOCKED', r['mode'] == 'STOP_BLOCKED', r['mode'])
    # 等待逾時 ⇒ WAIT_TIMEOUT（不強行穿越）
    L = W.WindowLayer(blocked=FREE)
    L.update(0, (0, -4, 1.57), (0, 0.3), path, [wk(4, 0, -0.6, 0)], 0.1)
    r = L.update(61.0, (0, -4, 1.57), (0, 0.3), path, [wk(4, 0, -0.6, 0)], 0.1)
    check('等待超過 60 s ⇒ WAIT_TIMEOUT', r['mode'] == 'WAIT_TIMEOUT', r['mode'])
    # 代價圖缺失 ⇒ 找不到停點也不產生離開線
    L = W.WindowLayer()
    r = L.update(0, (0, -4, 1.57), (0, 0.3), path, [wk(4, 0, -0.6, 0)], 0.1)
    check('代價圖缺失 ⇒ 不給停點路徑（stop_now 或 STOP_BLOCKED）', r['mode'] == 'STOP_BLOCKED' or (r['mode'] == 'WAIT' and r['hold']['kind'] == 'stop_now'), r.get('mode'))
    out = dict(n=len(RES), n_fail=sum(1 for x in RES if not x['ok']), results=RES)
    json.dump(out, open(os.path.join(HERE, 'results', 'm8_window_tests.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"{out['n'] - out['n_fail']}／{out['n']} 通過")
    return 0 if out['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
