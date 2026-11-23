#!/usr/bin/env python3
"""連續展臂候選：手腕相機在鎖定前的可見區間（Codex 20261101_190727 交接第 1 點；離線、非渲染）。
基線 SHROW 與候選（bottle_w1_continuous_approach_ctrl）在同一假受控對象閉迴路（test_bottle_w1）各跑一趟，沿模型軌跡逐 0.05 s（PREGRASP 之後逐 0.01 s）
用 bottle_w1_wrist_visibility 的同一套幾何（原相機內外參 CAMS['wrist']、hfov 69°、640×480、半像素中心、Isaac 近裁切 0.05 m、瓶背面剔除、桌腳／方塊／第二桌遮擋、
夾爪與兩指網格點深度緩衝）算瓶的可見比例與側面弧欄數；並套估計器距離閘 0.10–0.30 m（同 test_bottle_w1.msg 的距離定義：相機到瓶頂上 0.10 m）。
「可見」＝vis_frac ≥ 0.3、弧欄 ≥ 10（同模組的粗門檻，只看區間，不是估計器判準）且在距離閘內。手指在閉爪前視為全開（G.Q_OPEN）。
核對：候選在展臂開始到鎖定間的可見區間，與基線同段比較；候選在鎖定前至少有連續 ≥ 1.0 s 的可見區間（≥ 5 Hz 時足以形成 3 筆量測），且不晚於基線 0.5 s 以上開始。
這只是模型軌跡上的幾何可見度，不證明真實影像估計成功。
    python3 evaluation/test_bottle_w1_continuous_visibility.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_cand as SC                             # noqa: E402
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_continuous_approach_ctrl as CA         # noqa: E402
import bottle_w1_shrow_ctrl as SR                       # noqa: E402
import bottle_w1_wrist_visibility as WV                 # noqa: E402
import test_bottle_place as TP                          # noqa: E402
import test_bottle_w1 as T                              # noqa: E402

BAD = []


def chk(name, cond, info=''):
    if not cond:
        BAD.append(f'{name} {info}')
    print(('  ✓ ' if cond else '  ✗ ') + name + ('' if cond else f' {info}'))


def run(cls):
    orig = T.W1.W1PlaceController
    T.W1.W1PlaceController = cls
    try:
        c, f = T.build()
    finally:
        T.W1.W1PlaceController = orig
    r, _ = T.run(c, f)
    return c, f, r


def visibility(f, ev):
    boxes = [(cc, sz) for _, cc, sz in G.table_boxes() if _ != 'table_top'] + [(SC.OBST['c'], SC.OBST['sz'])] + [(cc, sz) for _, cc, sz in TP.table2_boxes()]
    top = [(cc, sz) for n_, cc, sz in G.table_boxes() if n_ == 'table_top']
    sp = WV.spacing()
    b0, ax = np.array([T.T.BX, T.T.BY, T.T.TOP]), np.array([0.0, 0.0, 1.0])
    P, N = WV.bottle_pts(b0, ax)
    t0, t1 = ev['unfold_start'], ev.get('wrist_lock', ev.get('close_request', 1e9))
    out = []
    for k, h in enumerate(f.hist):
        t = h['t']
        if t < t0 - 1e-9 or t > t1 + 1e-9:
            continue
        if h['phase'] == 'CRUISE' and k % 5:
            continue
        q = h['q']
        Tc = G.C.K.fk(q, WV.CAM['parent']) @ WV.CAM['T']
        Ti = np.linalg.inv(Tc)
        front = np.einsum('ij,ij->i', N, Tc[:3, 3] - P) > 0
        u, v, zc = WV.project(Ti, P)
        inimg = front & (zc >= WV.NEAR_SIM) & (zc <= WV.CAM['far']) & (u >= -0.5) & (u < WV.W_ - 0.5) & (v >= -0.5) & (v < WV.H_ - 0.5)
        scene = ~WV.seg_hits_boxes(Tc[:3, 3], P, boxes + top)
        zb = WV.zbuffer(Ti, G.link_world(q, G.Q_OPEN), sp, WV.NEAR_SIM)
        iu, iv = np.clip(np.round(u).astype(int), 0, WV.W_ - 1), np.clip(np.round(v).astype(int), 0, WV.H_ - 1)
        vis = inimg & scene & (zc < zb[iv, iu] - 1e-3)
        side = vis[:2700].reshape(30, 90)
        dist = float(np.linalg.norm(np.array([T.T.BX, T.T.BY, T.T.TOP + 0.1]) - Tc[:3, 3]))
        vf, arc = float(vis.sum() / max(front.sum(), 1)), int(side.any(axis=0).sum())
        out.append(dict(t=round(float(t), 3), phase=h['phase'], vis_frac=vf, arc=arc, dist=dist, ok=bool(vf >= 0.3 and arc >= 10 and 0.10 <= dist <= 0.30)))
    return out


def intervals(rows):
    iv, cur = [], None
    for r in rows:
        if r['ok'] and cur is None:
            cur = [r['t'], r['t']]
        elif r['ok']:
            cur[1] = r['t']
        elif cur is not None:
            iv.append(cur)
            cur = None
    if cur is not None:
        iv.append(cur)
    return iv


def main():
    res = {}
    for nm, cls in (('baseline_shrow', SR.W1PlaceController), ('continuous', CA.W1PlaceController)):
        c, f, r = run(cls)
        ev = {}
        for e in r['events']:
            ev.setdefault(e['what'], e['t'])
        rows = visibility(f, ev)
        iv = intervals(rows)
        first = iv[0][0] if iv else None
        longest = max((b - a for a, b in iv), default=0.0)
        ca = r.get('continuous_approach') or {}
        res[nm] = dict(events={k: ev.get(k) for k in ('unfold_start', 'blend_start', 'wrist_fix', 'wrist_lock', 'close_request', 'mission_done')}, done=bool(r['mission_done']), stop=r['stop'],
                       intervals=iv, first_visible=first, longest_s=longest, n_rows=len(rows),
                       dist_range_before_lock=[round(min(x['dist'] for x in rows), 3), round(max(x['dist'] for x in rows), 3)] if rows else None,
                       unfold_end=next((x[0] for x in ca.get('progress_log', []) if x[2] >= 1.0), None))
        print(f"   [{nm}] 展臂 {ev.get('unfold_start')}→完成 {res[nm]['unfold_end']}；朝瓶 {ev.get('blend_start')}；手腕形成量測 {ev.get('wrist_fix')}；鎖定 {ev.get('wrist_lock')}；閉爪 {ev.get('close_request')}；"
              f"完成 {r['mission_done']}；可見區間 {[[round(a, 2), round(b, 2)] for a, b in iv]}；最長 {longest:.2f} s")
    b, cnd = res['baseline_shrow'], res['continuous']
    chk('候選鎖定前有連續 ≥ 1.0 s 的可見區間', cnd['longest_s'] >= 1.0 - 1e-9, cnd['intervals'])
    chk('候選首次可見不晚於基線 0.5 s 以上', cnd['first_visible'] is not None and b['first_visible'] is not None and cnd['first_visible'] <= b['first_visible'] + 0.5 + 1e-9, (b['first_visible'], cnd['first_visible']))
    chk('候選在合成閉迴路中形成手腕量測並鎖定、任務完成', cnd['events']['wrist_fix'] is not None and cnd['events']['wrist_lock'] is not None and cnd['done'], cnd['events'])
    json.dump(res, open(os.path.join(HERE, 'results', 'bottle', 'w1_continuous_visibility.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n{"全過" if not BAD else "未過 " + str(len(BAD))}')
    for x in BAD:
        print('  失敗', x)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
