#!/usr/bin/env python3
"""m8_conflict r2 純函式反例（Codex 20261110_183705 第 3 項）。不跑 ROS、不跑模擬。
    python3 evaluation/test_m8_conflict.py
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m8_conflict as CF  # noqa: E402

RES = []


def check(name, cond, info=None):
    RES.append(dict(name=name, ok=bool(cond), info=info))
    print(('PASS ' if cond else 'FAIL ') + name + ('' if info is None else f'  {info}'))


def straight(a, b, ds=0.05):
    n = int(math.ceil(np.linalg.norm(np.subtract(b, a)) / ds)) + 1
    return np.c_[np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)]


def walker(x, y, vx, vy):
    return dict(tid=1, x=x, y=y, vx=vx, vy=vy, confirmed=True)


FREE = (lambda p: False)


def main():
    # 1 航跡過期 ⇒ 不放行
    L = CF.ConflictLayer(blocked=FREE)
    r = L.update(0.0, (0, 0, 0), (0, 0), straight((0, 0), (3, 0)), [], tracks_age=0.8)
    check('航跡過期 ⇒ HOLD_STALE', r['mode'] == 'HOLD_STALE', r['mode'])
    r = L.update(0.0, (0, 0, 0), (0, 0), straight((0, 0), (3, 0)), [], tracks_age=None)
    check('沒有航跡訊息 ⇒ HOLD_STALE', r['mode'] == 'HOLD_STALE', r['mode'])
    # 2 無移動物 ⇒ CRUISE
    L = CF.ConflictLayer(blocked=FREE)
    r = L.update(0.0, (0, 0, 0), (0.3, 0), straight((0, 0), (3, 0)), [walker(2, 2, 0.05, 0)], tracks_age=0.1)
    check('只有慢速物 ⇒ CRUISE', r['mode'] == 'CRUISE', r['mode'])
    # 3 站在行人路線正中、路徑往北 ⇒ EXIT 往北（路徑延續側）
    L = CF.ConflictLayer(blocked=FREE)
    path = straight((0, 0), (0, 3))
    r = L.update(0.0, (0, 0, math.pi / 2), (0, 0), path, [walker(3, 0, -0.6, 0)], tracks_age=0.1)
    check('路線正中 ⇒ EXIT', r['mode'] == 'EXIT', r['mode'])
    check('EXIT 往路徑延續側（北）', r['mode'] == 'EXIT' and r['hold']['goal'][1] > 0.5, r.get('hold'))
    # 4 優先側（北）被擋 ⇒ 換南；兩側都擋 ⇒ STOP_BLOCKED
    L = CF.ConflictLayer(blocked=lambda p: p[1] > 0.3)
    r = L.update(0.0, (0, 0, math.pi / 2), (0, 0), path, [walker(3, 0, -0.6, 0)], tracks_age=0.1)
    check('北側被擋 ⇒ EXIT 改往南', r['mode'] == 'EXIT' and r['hold']['goal'][1] < -0.5, r.get('hold'))
    L = CF.ConflictLayer(blocked=lambda p: abs(p[1]) > 0.3)
    r = L.update(0.0, (0, 0, math.pi / 2), (0, 0), path, [walker(3, 0, -0.6, 0)], tracks_age=0.1)
    check('兩側都擋 ⇒ STOP_BLOCKED', r['mode'] == 'STOP_BLOCKED', r['mode'])
    # 5 代價圖缺失（blocked 一律 True）⇒ 選不出離開線
    L = CF.ConflictLayer()
    r = L.update(0.0, (0, 0, math.pi / 2), (0, 0), path, [walker(3, 0, -0.6, 0)], tracks_age=0.1)
    check('代價圖缺失 ⇒ 不產生離開線（STOP_BLOCKED）', r['mode'] == 'STOP_BLOCKED', r['mode'])
    # 6 前方衝突、機器人在帶外 ⇒ WAIT；停點煞得住、不在帶內
    L = CF.ConflictLayer(blocked=FREE)
    path = straight((0, -3), (0, 3))
    r = L.update(0.0, (0, -3, math.pi / 2), (0, 0.3), path, [walker(4, 0, -0.6, 0)], tracks_age=0.1)
    ok = r['mode'] == 'WAIT' and r['hold']['kind'] == 'wait'
    check('前方衝突、在帶外 ⇒ WAIT', ok, r.get('hold'))
    if ok:
        st = np.array(r['hold']['stop'])
        check('停點弧長 ≥ 煞停距離', r['hold']['s_stop'] >= CF.d_brake(0.3) - 1e-9, dict(s=r['hold']['s_stop'], db=round(CF.d_brake(0.3), 3)))
        check('停點不在任何前方路線帶', not CF.in_band(st, walker(4, 0, -0.6, 0))[0], st.tolist())
    # 7 半徑隨預測時間增大
    m = walker(0, 0, 1.0, 0)
    p_far = np.array([10.0, float(CF.rc(5.0)) + 0.1])
    check('遠處點：rc(5) 外但 rc(τ≈10) 內 ⇒ in_band', CF.in_band(p_far, m)[0], dict(rc5=round(float(CF.rc(5.0)), 3), rc10=round(float(CF.rc(10.0)), 3)))
    # 8 EXIT 進行中出現新移動物擋住離開線 ⇒ 不沿用舊目標
    L = CF.ConflictLayer(blocked=FREE)
    path = straight((0, 0), (0, 3))
    r1 = L.update(0.0, (0, 0, math.pi / 2), (0, 0), path, [walker(3, 0, -0.6, 0)], tracks_age=0.1)
    g1 = r1['hold']['goal']
    r2 = L.update(0.1, (0, 0.03, math.pi / 2), (0, 0.25), path[1:], [walker(2.94, 0, -0.6, 0), dict(tid=2, x=-3.0, y=1.0, vx=0.8, vy=0.0, confirmed=True)], tracks_age=0.1)
    check('新移動物擋住原離開線 ⇒ 目標改變或改停', r2['mode'] in ('STOP_BLOCKED',) or (r2['mode'] == 'EXIT' and r2['hold']['goal'] != g1), dict(r1=g1, r2=r2.get('hold', {}).get('goal'), mode=r2['mode']))
    # 9 行人走過 ⇒ T_CLEAR 後回 CRUISE
    L = CF.ConflictLayer(blocked=FREE)
    path = straight((0, -3), (0, 3))
    L.update(0.0, (0, -3, math.pi / 2), (0, 0), path, [walker(4, 0, -0.6, 0)], tracks_age=0.1)
    gone = [walker(-6, 0, -0.6, 0)]
    r_a = L.update(5.0, (0, -3, math.pi / 2), (0, 0), path, gone, tracks_age=0.1)
    r_b = L.update(6.1, (0, -3, math.pi / 2), (0, 0), path, gone, tracks_age=0.1)
    check('行人走過：未滿 1 s 不回 CRUISE、滿 1 s 回 CRUISE', r_a['mode'] != 'CRUISE' and r_b['mode'] == 'CRUISE', [r_a['mode'], r_b['mode']])
    out = dict(n=len(RES), n_fail=sum(1 for x in RES if not x['ok']), results=RES)
    json.dump(out, open(os.path.join(HERE, 'results', 'm8_conflict_tests.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"{out['n'] - out['n_fail']}／{out['n']} 通過")
    return 0 if out['n_fail'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
