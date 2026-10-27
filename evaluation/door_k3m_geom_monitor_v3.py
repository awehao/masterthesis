#!/usr/bin/env python3
"""K3-M 幾何距離監看的雙把手後端（Codex r48 §3「新流程的保護都須看見內側把手」；門規格 door_local_v3、幾何 door_geom_v3）。
door_k3m_geom_monitor（含 GeometryMonitor 判定、default_backend）不改；本模組只提供 v3 距離後端，交給同一個 GeometryMonitor。

與 v2 後端（door_k3m_geom_monitor.default_backend）的差異：
  - 門形狀＝door_geom_v3.shapes（v2 形狀＋內側立柱 post_in0／post_in1）；
  - nonfinger：手臂各連桿、link_eef、夾爪殼 對 全部形狀 **及兩支橫桿**（外側、內側膠囊各自計，取較小）；
  - finger：兩指 對 全部形狀（含四根立柱、門扇、兩面牆）＋**非目標橫桿**；只排除「手指—目標橫桿」設計接觸（與 v2 排除手指—橫桿同義）。
    目標橫桿由呼叫端依相位設定：fn.set_target('out'｜'in'｜None)；None（自由移動、收臂、移動）⇒ 兩支橫桿都計入手指。
    目標橫桿上的意外接觸不由本監看處理，沿用既有接觸力保護（閉爪前碰桿保護；r49 §3 首版不允許「輕觸」）。
    相位切換目標時手指類距離可能驟降、接近率變大 ⇒ 較保守（可能停止）；相位機須在手指離開目標橫桿且留有餘裕後才切換。
  - base：底盤圓盤 對 v3 全部形狀；
  - self：與 v2 後端同一份程式與點雲（逐字搬入；test_door_k3m_geom_monitor_v3 以 v2 後端對照逐值相等）。
另回傳（不參與判定，供紀錄）：'finger_bar_out'、'finger_bar_in'＝兩指對各橫桿最小距離、'target_bar'＝當輪使用的目標。
"""
from __future__ import annotations

import math

import numpy as np

from door_k3m_geom_monitor import FINGERS, SELF_NEAR_M, self_pairs

TARGETS = ('out', 'in', None)


def backend_v3(clouds=None, self_clouds=None, self_mode='sphere', D3=None):
    import door_k2_offline_v2 as V2
    import door_geom_v3 as G3
    import yaml
    import os
    from scipy.spatial import cKDTree
    C, K1 = V2.C, V2.K1
    if D3 is None:
        D3 = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                              'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml')))
    H = float(D3['handle']['height'])
    if abs(H - float(K1.H)) > 1e-12:
        raise ValueError(f'v3 把手高 {H} 與離線點雲設定 {K1.H} 不符')
    CL = clouds if clouds is not None else C.CL
    pairs = self_pairs()
    s_names = sorted({n for p in pairs for n in p})
    SC = self_clouds if self_clouds is not None else {n: C.CL[n] for n in s_names}
    cen = {n: SC[n].mean(axis=0) for n in s_names}
    rad = {n: float(np.linalg.norm(SC[n] - cen[n], axis=1).max()) for n in s_names}
    stats = {'n_exact': 0, 'n_calls': 0}
    state = {'target': 'out'}

    def world_fn(qf, names, q_finger):
        out = {}
        for n in names:
            T = C.K.fk(qf, n)
            P = CL[n] + (C.FINGER_AXIS[n] * q_finger if n in C.FINGER_AXIS else 0.0)
            out[n] = (T[:3, :3] @ P.T).T + T[:3, 3]
        return out

    def self_dist(qf, q_finger):                       # 逐字同 door_k3m_geom_monitor.default_backend.self_dist
        stats['n_calls'] += 1
        T = {n: C.K.fk(qf, n) for n in s_names}
        best = (math.inf, None)
        todo = []
        for x, y in pairs:
            if self_mode == 'full':
                todo.append((x, y))
                continue
            cx = T[x][:3, :3] @ (cen[x] + (C.FINGER_AXIS[x] * q_finger if x in C.FINGER_AXIS else 0.0)) + T[x][:3, 3]
            cy = T[y][:3, :3] @ (cen[y] + (C.FINGER_AXIS[y] * q_finger if y in C.FINGER_AXIS else 0.0)) + T[y][:3, 3]
            lb = float(np.linalg.norm(cx - cy)) - rad[x] - rad[y]
            if lb < SELF_NEAR_M:
                todo.append((x, y))
            elif lb < best[0]:
                best = (lb, f'{x}|{y}:lb')
        need = sorted({n for p in todo for n in p})
        W = {}
        for n in need:
            P = SC[n] + (C.FINGER_AXIS[n] * q_finger if n in C.FINGER_AXIS else 0.0)
            W[n] = (T[n][:3, :3] @ P.T).T + T[n][:3, 3]
        trees = {}
        for x, y in todo:
            if x not in trees:
                trees[x] = cKDTree(W[x])
            d = float(trees[x].query(W[y], k=1)[0].min())
            stats['n_exact'] += 1
            if d < best[0]:
                best = (d, f'{x}|{y}')
        return best

    def fn(qf, q_door, q_finger):
        tgt = state['target']
        sh = G3.shapes(D3, q_door, H)
        W = world_fn(qf, C.ARM_LINKS, q_finger)
        out = {'nonfinger': (math.inf, None), 'finger': (math.inf, None), 'finger_bar_out': (math.inf, None), 'finger_bar_in': (math.inf, None)}
        for n, P in W.items():
            d_parts = min((float(np.min(G3.box_dist(P, c, R, hs))), name) for name, c, R, hs in sh)
            d_out = float(np.min(G3._bar_outer(D3, P, q_door, H)))
            d_in = float(np.min(G3.bar_capsule_dist_inner(D3, P, q_door)))
            if n in FINGERS:
                if d_out < out['finger_bar_out'][0]:
                    out['finger_bar_out'] = (d_out, n)
                if d_in < out['finger_bar_in'][0]:
                    out['finger_bar_in'] = (d_in, n)
                cand = [d_parts] + ([(d_out, 'bar_out')] if tgt != 'out' else []) + ([(d_in, 'bar_in')] if tgt != 'in' else [])
                d, who = min(cand)
                if d < out['finger'][0]:
                    out['finger'] = (d, f'{n}:{who}')
            else:
                d, who = min([d_parts, (d_out, 'bar_out'), (d_in, 'bar_in')])
                if d < out['nonfinger'][0]:
                    out['nonfinger'] = (d, f'{n}:{who}')
        best = (math.inf, None)
        for name, c, R, hs in sh:
            d = float(G3.disc_clear(qf[0], qf[1], c[2], C.RB, c, R, hs))
            if d < best[0]:
                best = (d, name)
        out['base'] = best
        out['self'] = self_dist(qf, q_finger)
        out['target_bar'] = (float('nan'), tgt)
        fn.last = out                                     # 供呼叫端讀 finger_bar_out／finger_bar_in（不參與判定）
        return out

    def set_target(t):
        if t not in TARGETS:
            raise ValueError(f'目標橫桿須為 {TARGETS}')
        state['target'] = t

    fn.last = None
    fn.set_target = set_target
    fn.get_target = lambda: state['target']
    fn.pairs = pairs
    fn.self_stats = stats
    fn.self_mode = self_mode
    fn.spec = D3.get('name')
    return fn
