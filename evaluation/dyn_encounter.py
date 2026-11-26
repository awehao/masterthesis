#!/usr/bin/env python3
"""動態遭遇指標（11/26 導航動態能力比較）：只看機器人與「動態障礙物」的關係，牆與靜態物另計（analyze.py）。

機器人位姿＝raw /odom（gz 無漂移真值里程，同 analyze.py 的碰撞判定）；障礙物＝/model/dyn_obs_*/pose 真值。
表面距離 d＝中心距 − 障礙物半徑 − 機器人半徑 0.30。區間＝第一筆非零 /cmd_vel 到最後一筆 odom（到達或到時）。
指標：
  dmin_dyn_m        對動態障礙物的最小表面距離（<0 ＝ 幾何重疊＝碰撞）
  n_contact_dyn     d<0 的獨立事件數（間隔 >1 s 算不同次）
  t_d05_s / t_d10_s d<0.5 m／<1.0 m 的累積時間
  n_enc10           d<1.0 m 的遭遇次數（同一障礙物間隔 >2 s 算不同次）
  v_enc / v_free    遭遇中（d<1.0）／其餘時間的機器人平均速率；v_enc_ratio＝兩者比
  ttc_min_s         等速外插的最小碰撞時間（只計靠近中且 d<2 m 的時刻）
用法：dyn_encounter.py --traj <情境 yaml> <bag_dir> [<bag_dir> ...] [--csv out.csv]
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A                                                   # noqa: E402

R_ROBOT = 0.30


def episodes(mask, t, gap):
    n, last = 0, None
    for on, tt in zip(mask, t):
        if on:
            if last is None or tt - last > gap:
                n += 1
            last = tt
    return n


def one(bag, radii):
    M = A.read_bag(bag)
    to, xyth = A.extract_odom(M)
    cmd = M.get('/cmd_vel', [])
    t_mv = next((t_ns / 1e9 for t_ns, m in cmd if math.hypot(m.linear.x, m.linear.y) > 1e-3 or abs(m.angular.z) > 1e-3), None)
    if t_mv is None or len(to) < 10:
        return dict(bag=os.path.basename(bag), note='no motion')
    sel = to >= t_mv
    t, P = to[sel], xyth[sel, :2]
    dt = np.gradient(t)
    v = np.hypot(*np.gradient(P, t, axis=0).T)
    tracks = [(ts, xy, nm) for ts, xy, nm in A.extract_gt_obstacles(M) if nm.startswith('dyn_obs')]
    D = np.full((len(t), len(tracks)), np.inf)
    VREL = np.zeros((len(t), len(tracks), 2))
    RREL = np.zeros((len(t), len(tracks), 2))
    for j, (ts, xy, nm) in enumerate(tracks):
        ox, oy = np.interp(t, ts, xy[:, 0]), np.interp(t, ts, xy[:, 1])
        D[:, j] = np.hypot(P[:, 0] - ox, P[:, 1] - oy) - radii.get(nm, 0.25) - R_ROBOT
        ov = np.gradient(np.c_[ox, oy], t, axis=0)
        RREL[:, j] = np.c_[ox, oy] - P
        VREL[:, j] = ov - np.gradient(P, t, axis=0)
    dmin = D.min(axis=1)
    jmin = D.argmin(axis=1)
    enc = dmin < 1.0
    n_enc = sum(episodes(D[:, j] < 1.0, t, 2.0) for j in range(D.shape[1]))
    # TTC：相對位置 r、相對速度 w（障礙物−機器人），靠近中（r·w<0）且 d<2 m
    ttc = np.inf
    for i in range(len(t)):
        j = jmin[i]
        if dmin[i] < 2.0:
            r, w = RREL[i, j], VREL[i, j]
            rw = float(r @ w)
            if rw < 0:
                closing = -rw / max(np.linalg.norm(r), 1e-9)
                ttc = min(ttc, max(dmin[i], 0.0) / max(closing, 1e-6))
    return dict(bag=os.path.basename(bag.rstrip('/')), dmin_dyn_m=round(float(dmin.min()), 3),
                n_contact_dyn=episodes(dmin < 0.0, t, 1.0),
                t_d05_s=round(float(dt[dmin < 0.5].sum()), 2), t_d10_s=round(float(dt[enc].sum()), 2), n_enc10=n_enc,
                v_enc=round(float(v[enc].mean()), 3) if enc.any() else None, v_free=round(float(v[~enc].mean()), 3),
                v_enc_ratio=round(float(v[enc].mean() / v[~enc].mean()), 3) if enc.any() else None,
                ttc_min_s=round(float(ttc), 2) if np.isfinite(ttc) else None, dur_s=round(float(t[-1] - t[0]), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj', required=True)
    ap.add_argument('--csv', default='')
    ap.add_argument('bags', nargs='+')
    a = ap.parse_args()
    radii = {d['name']: float(d['radius']) for d in yaml.safe_load(open(a.traj))['dynamic_obstacles']}
    rows = [one(b, radii) for b in a.bags]
    for r in rows:
        print(r)
    if a.csv:
        new = not os.path.exists(a.csv)
        with open(a.csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            if new:
                w.writeheader()
            for r in rows:
                w.writerow(r)


if __name__ == '__main__':
    main()
