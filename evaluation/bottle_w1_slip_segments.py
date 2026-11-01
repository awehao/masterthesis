#!/usr/bin/env python3
"""持瓶滑移分段定位（只讀紀錄；Codex 20261101_204716 第 3 項）。滑移 d(t)＝瓶對 TCP 相對位姿與抓持成立時刻（t_grasp）基準的位置差
（與 bottle_s0_logic 同定義；TCP 用實測關節 FK，瓶用讀回位姿），角度差同。
分段：抓持→LIFT、LIFT→離桌判定、側持運送（→收近）、收近過渡、緊湊穩態、轉回升高、轉回外伸、轉回完成→PARKED、PARKED→RELEASE（含下降與接觸）。
每段報：段首、段末、段內最大、段增量（末−首）、段內最大增量（段內 max − 段首）；另報段內瓶底最低、瓶點到車體頂（半徑 0.32 m 內、頂 z 0.38）最小垂直淨空。
    python3 evaluation/bottle_w1_slip_segments.py RUN_ID [RUN_ID ...]
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src', 'ammr_wholebody_mpc'))
import bottle_s1_geom as G                              # noqa: E402
import bottle_w1_compact_pose as CP                     # noqa: E402


def rel(Tt, Tb):
    return np.linalg.inv(Tt) @ Tb


def one(run):
    D = os.path.join(HERE, 'runs', run)
    r = json.load(open(os.path.join(D, 's1_report.json')))
    Z = np.load(os.path.join(D, 's1_steps.npz'))
    c = {str(k): Z['rows'][:, i] for i, k in enumerate(Z['cols'])}
    t = c['t']
    ev = {}
    for e in r['events']:
        k = e['what'] if e['what'] not in ('place_phase', 'phase') else e['what'][0] + '_' + e.get('to', '')
        ev.setdefault(k, e['t'])
    tg = r['t_grasp']
    BP = G.bottle_points(12, 5)
    D_, A_, BOT, DECK = [], [], [], []
    T0 = None
    for i in range(len(t)):
        q = np.array([c['base_x'][i], c['base_y'][i], c['base_yaw'][i]] + [c[f'q{j}'][i] for j in range(1, 7)])
        Tt = G.C.K.fk(q, 'link_tcp')
        Tb = np.eye(4)
        Tb[:3, :3] = CP.quat(c['bottle_qw'][i], c['bottle_qx'][i], c['bottle_qy'][i], c['bottle_qz'][i])
        Tb[:3, 3] = [c['bottle_base_x'][i], c['bottle_base_y'][i], c['bottle_base_z'][i]]
        R_ = rel(Tt, Tb)
        if T0 is None and t[i] >= tg - 1e-9:
            T0 = R_
        if T0 is None:
            D_.append(np.nan); A_.append(np.nan); BOT.append(np.nan); DECK.append(np.nan)
            continue
        D_.append(float(np.linalg.norm(R_[:3, 3] - T0[:3, 3])))
        dR = T0[:3, :3].T @ R_[:3, :3]
        A_.append(math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))))
        Pw = (Tb[:3, :3] @ BP.T).T + Tb[:3, 3]
        BOT.append(float(Pw[:, 2].min()))
        dx, dy = Pw[:, 0] - q[0], Pw[:, 1] - q[1]
        inr = np.hypot(dx, dy) < 0.32
        DECK.append(float((Pw[inr, 2] - 0.38).min()) if inr.any() else np.inf)
    D_, A_, BOT, DECK = (np.array(x) for x in (D_, A_, BOT, DECK))
    bnd = [('抓持→LIFT', tg, ev.get('p_LIFT')), ('LIFT→離桌判定', ev.get('p_LIFT'), r['t_lift_ok']), ('側持運送（→收近）', r['t_lift_ok'], ev.get('compact_start')),
           ('收近過渡', ev.get('compact_start'), ev.get('compact_reached')), ('緊湊穩態', ev.get('compact_reached'), ev.get('compact_return')),
           ('轉回升高', ev.get('compact_return'), ev.get('compact_raised')), ('轉回外伸', ev.get('compact_raised'), ev.get('compact_returned')),
           ('轉回完成→PARKED', ev.get('compact_returned'), ev.get('p_PARKED')), ('PARKED→RELEASE（下降、接觸）', ev.get('p_PARKED'), ev.get('p_RELEASE'))]
    segs = []
    for nm, a, b in bnd:
        if a is None or b is None or b <= a:
            segs.append(dict(seg=nm, t=[a, b], n=0))
            continue
        m = (t >= a - 1e-9) & (t <= b + 1e-9) & np.isfinite(D_)
        k0, k1 = np.nonzero(m)[0][[0, -1]]
        segs.append(dict(seg=nm, t=[round(a, 2), round(b, 2)], d0_mm=round(D_[k0] * 1e3, 2), d1_mm=round(D_[k1] * 1e3, 2), dmax_mm=round(float(D_[m].max()) * 1e3, 2),
                         inc_mm=round((D_[k1] - D_[k0]) * 1e3, 2), inc_max_mm=round((float(D_[m].max()) - D_[k0]) * 1e3, 2), amax_deg=round(float(A_[m].max()), 2),
                         bottom_min=round(float(BOT[m].min()), 3), deck_min=round(float(DECK[m].min()), 3) if np.isfinite(DECK[m]).any() else None))
    kmax = int(np.nanargmax(D_[t <= (ev.get('p_RELEASE') or 1e9)]))
    return dict(run=run, judge_slip_mm=round(r['slip_max_mm_deg'][0], 2), fk_slip_max_mm=round(float(D_[kmax]) * 1e3, 2), t_max=round(float(t[kmax]), 2), segments=segs)


def main():
    out = {}
    for run in sys.argv[1:]:
        o = out[run] = one(run)
        print(f"{run}：判定滑移 {o['judge_slip_mm']} mm；FK 重算最大 {o['fk_slip_max_mm']} mm @ {o['t_max']} s")
        for s in o['segments']:
            if s.get('n') == 0:
                print(f"   {s['seg']:<22} —")
                continue
            print(f"   {s['seg']:<22} {s['t'][0]:6.2f}–{s['t'][1]:6.2f}  首 {s['d0_mm']:5.2f} 末 {s['d1_mm']:5.2f} 最大 {s['dmax_mm']:5.2f}  增量 {s['inc_mm']:+5.2f}（段內最大 {s['inc_max_mm']:+5.2f}）mm  角 {s['amax_deg']:.2f}°  瓶底 {s['bottom_min']:.3f}  車體頂 {s['deck_min']}")
    json.dump(out, open(os.path.join(HERE, 'results', 'bottle', 'w1_slip_segments.json'), 'w'), ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == '__main__':
    sys.exit(main())
