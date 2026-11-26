#!/usr/bin/env python3
"""動態遭遇指標（11/26 導航動態能力比較）：只看機器人與「動態障礙物」的關係，牆與靜態物另計（analyze.py）。

機器人位姿＝raw /odom（gz 無漂移真值里程，同 analyze.py 的碰撞判定）；障礙物＝/model/dyn_obs_*/pose 真值。
表面距離 d＝機器人中心到障礙物「真實形狀」表面的距離 − 機器人半徑 0.30；形狀取自世界檔（analyze.DYN_GEOM：圓柱或軸對齊方盒，
移動體保持 spawn 方向）。場景 yaml 的 radius 對方盒只是包覆半徑，不能拿來算距離（11/26 第一版誤用 ⇒ 高估碰撞）。需設 BIGARENA=1 等同跑法的環境。區間＝第一條 /plan 到「到達」（地圖位姿進入目標 0.30 m 內，同 analyze.py 與目標檢查器）；
未到達則到最後一筆 odom。到達後機器人停在終點、不再列入（快速行人可能撞上停著的車）。
時間戳間隔 <0.5×中位數的假筆（DDS 時序）先去掉，避免速度尖峰。
碰撞分類（接觸開始那一刻）：機器人朝障礙物的速度分量 >0.05 m/s ＝主動（n_contact_active）；
其餘（靜止、遠離或橫移時被行人追撞／側撞）＝被動（n_contact_passive）。行人不讓路且最快達機器人上限 4 倍，被動碰撞未必可避免。
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


def one(bag, radii, goal):
    M = A.read_bag(bag)
    to, xyth = A.extract_odom(M)
    plans = M.get('/plan', [])
    if not plans or len(to) < 10:
        return dict(bag=os.path.basename(bag), note='no plan')
    keep = np.r_[True, np.diff(to) >= 0.5 * np.median(np.diff(to))]   # 去 DDS 假筆
    to, xyth = to[keep], xyth[keep]
    t0 = plans[0][0] / 1e9
    mt, mx = A.extract_robot_map_pose(M)[:2]                          # 到達判定同 analyze.py／目標檢查器：地圖位姿
    if len(mt) < 2:
        mt, mx = to, xyth
    near = np.where((mt >= t0) & (np.hypot(mx[:, 0] - goal[0], mx[:, 1] - goal[1]) <= 0.30))[0]
    t1 = mt[near[0]] if len(near) else to[-1]
    sel = (to >= t0) & (to <= t1)
    t, P = to[sel], xyth[sel, :2]
    dt = np.gradient(t)
    v = np.hypot(*np.gradient(P, t, axis=0).T)
    tracks = [(ts, xy, nm) for ts, xy, nm in A.extract_gt_obstacles(M) if nm.startswith('dyn_obs')]
    D = np.full((len(t), len(tracks)), np.inf)
    VREL = np.zeros((len(t), len(tracks), 2))
    RREL = np.zeros((len(t), len(tracks), 2))
    geom = getattr(A, 'DYN_GEOM', {})
    for j, (ts, xy, nm) in enumerate(tracks):
        ox, oy = np.interp(t, ts, xy[:, 0]), np.interp(t, ts, xy[:, 1])
        g = geom.get(nm)
        if g and g[0] == 'box':
            D[:, j] = np.hypot(np.maximum(np.abs(ox - P[:, 0]) - g[1] / 2, 0.0), np.maximum(np.abs(oy - P[:, 1]) - g[2] / 2, 0.0)) - R_ROBOT
        else:
            D[:, j] = np.hypot(P[:, 0] - ox, P[:, 1] - oy) - (g[1] if g else radii.get(nm, 0.25)) - R_ROBOT
        ov = np.gradient(np.c_[ox, oy], t, axis=0)
        RREL[:, j] = np.c_[ox, oy] - P
        VREL[:, j] = ov - np.gradient(P, t, axis=0)
    dmin = D.min(axis=1)
    jmin = D.argmin(axis=1)
    VR = np.gradient(P, t, axis=0)
    n_act = n_pas = 0
    prev = False
    for i in range(len(t)):
        on = dmin[i] < 0.0
        if on and not prev:
            j = jmin[i]
            r = RREL[i, j] / max(np.linalg.norm(RREL[i, j]), 1e-9)
            if float(VR[i] @ r) > 0.05:
                n_act += 1
            else:
                n_pas += 1
        prev = on
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
    i_m = int(np.argmin(dmin))
    d_goal = np.hypot(P[:, 0] - goal[0], P[:, 1] - goal[1])
    return dict(bag=os.path.basename(bag.rstrip('/')), dmin_dyn_m=round(float(dmin.min()), 3),
                goal_dist_at_dmin_m=round(float(d_goal[i_m]), 2),
                t_d10_near_goal_s=round(float(dt[enc & (d_goal < 2.5)].sum()), 2),
                n_contact_dyn=episodes(dmin < 0.0, t, 1.0), n_contact_active=n_act, n_contact_passive=n_pas,
                arrived=bool(len(near)),
                t_d05_s=round(float(dt[dmin < 0.5].sum()), 2), t_d10_s=round(float(dt[enc].sum()), 2), n_enc10=n_enc,
                v_enc=round(float(v[enc].mean()), 3) if enc.any() else None, v_free=round(float(v[~enc].mean()), 3),
                v_enc_ratio=round(float(v[enc].mean() / v[~enc].mean()), 3) if enc.any() else None,
                ttc_min_s=round(float(ttc), 2) if np.isfinite(ttc) else None, dur_s=round(float(t[-1] - t[0]), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj', required=True)
    ap.add_argument('--csv', default='')
    ap.add_argument('--goal', required=True, help='x,y（同 POSES_CSV 該路線）')
    ap.add_argument('bags', nargs='+')
    a = ap.parse_args()
    radii = {d['name']: float(d['radius']) for d in yaml.safe_load(open(a.traj))['dynamic_obstacles']}
    goal = [float(v) for v in a.goal.split(',')]
    if A.load_static_clearance() is None or not getattr(A, 'DYN_GEOM', None):
        sys.exit('讀不到世界檔形狀（BIGARENA／SCENE 環境變數要和跑法一致）')
    rows = [one(b, radii, goal) for b in a.bags]
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
