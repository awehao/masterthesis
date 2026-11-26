#!/usr/bin/env python3
"""N1 情境產生器（11/27 Howard：複雜場景、行人不能太快、要有未知靜態障礙、每次隨機起終點）。

一個（布局, 種子）產生同一組：起點／終點、7 個未知靜態障礙位置；再依條件輸出兩份情境檔：
  static    只有已知＋未知靜態障礙（行人停在場外、速度 0）
  crossing  再加 10 位行人：長距離來回（Howard 原本 bigarena_traffic 的方式），速度 0.10–0.25 m/s（不超過車速 0.26），
            位置、方向、相位全部隨機（一半順著走廊／走道方向）；遭遇由時機自然產生。所有行人路段離起點／終點至少 ZONE（表面距離）。
            11/27 Howard「障礙物設計可以不用那麼刻意」⇒ 預設不安排穿越者（--n-cross 0）、未知障礙不刻意放路線旁（--n-route-unk 0）；
            兩個選項保留（>0 時：穿越者路段穿過名目路徑 20–80% 弧長、在機器人以 V_NOM 到達時刻 ±JITTER 通過）。
同一種子的 static／crossing 起終點與未知障礙完全相同（成對）；四個方法讀同一份情境檔（成對）。
起終點：空地（離已知障礙 ≥0.8 m）、直線距離 ≥10 m、名目路徑 12–26 m 且至少經過一處窄處（門口／走道，淨空 <1.0 m）。
未知障礙：中心離已知障礙 ≥ 包覆半徑＋0.75 m（至少一側過得去）、離起終點 ≥1.5 m＋半徑、彼此 ≥1.0 m；
          預設全部隨機（--n-route-unk >0 時其中幾個放在名目路徑中段 2.5 m 內）；
          放完後在真實佔據（已知＋未知）上仍須連通，且繞行後路長 ≤ 1.3 倍（不整個堵死門口）。
輸出：<out>/n1_<layout>_<cond>_k<seed>.yaml＋_meta.json；並更新 evaluation/results/n1_poses_<layout>.csv（seed 列）。
用法：make_n1_scenario.py --layout n1_office --seed 3 --out-dir DIR"""
import argparse
import csv
import heapq
import json
import math
import os
import random
import re
import zlib

import numpy as np
import yaml
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
WS = os.path.dirname(os.path.dirname(PKG))

ap = argparse.ArgumentParser()
ap.add_argument('--layout', required=True, choices=['n1_bigarena', 'n1_office', 'n1_warehouse'])
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--out-dir', required=True)
ap.add_argument('--v-lo', type=float, default=0.10)
ap.add_argument('--v-hi', type=float, default=0.25)
ap.add_argument('--n-cross', type=int, default=0)   # 11/27 Howard「障礙物設計可以不用那麼刻意」：預設全部隨機，不安排穿越
ap.add_argument('--n-route-unk', type=int, default=0)   # 同上：未知障礙全隨機
ap.add_argument('--v-nom', type=float, default=0.22, help='估機器人到穿越點時刻的名目速度')
ap.add_argument('--jitter', type=float, default=6.0)
ap.add_argument('--zone', type=float, default=2.5)
ap.add_argument('--infl', type=float, default=0.45)
ap.add_argument('--poses-csv', default='')
a = ap.parse_args()

drv = open(os.path.join(PKG, 'ammr_bringup', 'dynamic_obstacle_driver.py')).read()
ns = {'math': math}
exec(drv[drv.index('def profile_arclength'):drv.index('class DynamicObstacleDriver')], ns)   # noqa: S102
schedule_state = ns['schedule_state']

# ---- 地圖（已知）與世界檔（未知箱、行人尺寸）----
my = yaml.safe_load(open(os.path.join(PKG, 'maps', a.layout + '.yaml')))
res, ox, oy = my['resolution'], my['origin'][0], my['origin'][1]
with open(os.path.join(PKG, 'maps', my['image']), 'rb') as f:
    assert f.readline().strip() == b'P5'
    line = f.readline()
    while line.startswith(b'#'):
        line = f.readline()
    w, h = map(int, line.split()); int(f.readline())
    img = np.frombuffer(f.read(), np.uint8).reshape(h, w)
OCC_K = ((255 - img.astype(float)) / 255.0) > my['occupied_thresh']
sdf = re.sub(r'<!--.*?-->', '', open(os.path.join(PKG, 'worlds', a.layout + '.sdf')).read(), flags=re.S)
UNK = []
for m in re.finditer(r'<model name="(unknown_obs_\d+)">(.*?)</model>', sdf, re.S):
    sx, sy = [float(v) for v in re.search(r'<box><size>([\d.]+) ([\d.]+)', m.group(2)).groups()]
    UNK.append((m.group(1), sx, sy))
SHAPE = {}
for m in re.finditer(r'<model name="(dyn_obs_\d+)">(.*?)</model>', sdf, re.S):
    b = re.findall(r'<box><size>([\d.]+) ([\d.]+)', m.group(2)); c = re.search(r'<cylinder><radius>([\d.]+)', m.group(2))
    if c:
        SHAPE[m.group(1)] = float(c.group(1))
    else:
        xs = [float(u) for u, _ in b]; ys = [float(v) for _, v in b]
        SHAPE[m.group(1)] = 0.5 * math.hypot(max(xs + ys), max(xs + ys)) if len(b) > 1 else 0.5 * math.hypot(xs[0], ys[0])
YY, XX = np.mgrid[0:h, 0:w]
XC, YC = ox + (XX + 0.5) * res, oy + (h - 1 - YY + 0.5) * res
rng = random.Random(zlib.crc32(f'n1|{a.layout}|{a.seed}'.encode()))


def cell(p):
    return int(h - 1 - (p[1] - oy) / res), int((p[0] - ox) / res)


def dist_of(occ):
    return ndimage.distance_transform_edt(~occ) * res


def at(D, p):
    r, c = cell(p)
    return float(D[r, c]) if (0 <= r < h and 0 <= c < w) else 0.0


def astar(D, s, g, infl):
    step = 2
    free = D[::step, ::step] > infl
    H2, W2 = free.shape
    s2 = (cell(s)[0] // step, cell(s)[1] // step); g2 = (cell(g)[0] // step, cell(g)[1] // step)
    free[s2] = free[g2] = True
    nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    gs, par, pq = {s2: 0.0}, {}, [(0.0, s2)]
    while pq:
        _, u = heapq.heappop(pq)
        if u == g2:
            break
        for dr, dc, cst in nb:
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < H2 and 0 <= v[1] < W2 and free[v] and gs[u] + cst < gs.get(v, 1e18):
                gs[v] = gs[u] + cst; par[v] = u
                heapq.heappush(pq, (gs[v] + math.hypot(v[0] - g2[0], v[1] - g2[1]), v))
    if g2 not in par:
        return None
    cells = [g2]
    while cells[-1] != s2:
        cells.append(par[cells[-1]])
    P = np.array([[ox + (c * step + 1) * res, oy + (h - 1 - r * step - 1) * res] for r, c in cells[::-1]])
    P[0], P[-1] = s, g
    P = ndimage.uniform_filter1d(P, 7, axis=0, mode='nearest'); P[0], P[-1] = s, g
    return P


def plen(P):
    return float(np.sum(np.hypot(*np.diff(P, axis=0).T)))


def box_mask(x, y, sx, sy, yaw):
    cs, sn = math.cos(-yaw), math.sin(-yaw)
    lx, ly = cs * (XC - x) - sn * (YC - y), sn * (XC - x) + cs * (YC - y)
    return (np.abs(lx) <= sx / 2) & (np.abs(ly) <= sy / 2)


D_K = dist_of(OCC_K)

# ---- 起終點 ----
for _ in range(20000):
    s = np.array([rng.uniform(ox + 1, ox + w * res - 1), rng.uniform(oy + 1, oy + h * res - 1)])
    g = np.array([rng.uniform(ox + 1, ox + w * res - 1), rng.uniform(oy + 1, oy + h * res - 1)])
    if at(D_K, s) < 0.8 or at(D_K, g) < 0.8 or np.hypot(*(s - g)) < 10.0:
        continue
    P = astar(D_K, s, g, a.infl)
    if P is None or not (12.0 <= plen(P) <= 26.0):
        continue
    pc = np.array([at(D_K, p) for p in P])
    if not (pc < 1.0).any():
        continue
    break
else:
    raise SystemExit('!! 找不到起終點')
cum = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; SLEN = float(cum[-1])
mid = P[(cum >= 0.2 * SLEN) & (cum <= 0.8 * SLEN)]

# ---- 未知靜態障礙 ----
for attempt in range(200):
    occ_t = OCC_K.copy(); placed = []
    order = list(range(len(UNK))); rng.shuffle(order)
    ok = True
    for j, k in enumerate(order):
        nm, sx, sy = UNK[k]
        rad = 0.5 * math.hypot(sx, sy)
        near_route = j < a.n_route_unk
        for _t in range(6000):
            if near_route and _t >= 3000:
                near_route = False                                     # 路線旁放不下就改隨機（meta 記實際值）
            if near_route:
                q = mid[rng.randrange(len(mid))]; ang = rng.uniform(0, 2 * math.pi); r_ = rng.uniform(0.6, 2.5)
                x, y = q[0] + r_ * math.cos(ang), q[1] + r_ * math.sin(ang)
            else:
                x, y = rng.uniform(ox + 1, ox + w * res - 1), rng.uniform(oy + 1, oy + h * res - 1)
            if at(D_K, (x, y)) < rad + 0.75:
                continue
            if min(math.hypot(x - s[0], y - s[1]), math.hypot(x - g[0], y - g[1])) < rad + 1.5:
                continue
            if any(math.hypot(x - u, y - v) < rad + rr + 1.0 for u, v, rr in [(p['x'], p['y'], p['rad']) for p in placed]):
                continue
            yaw = rng.choice([0.0, math.pi / 2])
            placed.append(dict(name=nm, x=round(float(x), 3), y=round(float(y), 3), yaw=round(float(yaw), 4), sx=sx, sy=sy, rad=round(float(rad), 3), near_route=bool(near_route)))
            occ_t |= box_mask(x, y, sx, sy, yaw)
            break
        else:
            ok = False; break
    if not ok:
        continue
    D_T = dist_of(occ_t)
    P2 = astar(D_T, s, g, a.infl)
    if P2 is not None and plen(P2) <= 1.3 * SLEN:
        break
else:
    raise SystemExit('!! 未知障礙放不下')


# ---- 行人（crossing）----
MARGIN = [0.2]                                                        # 行人路段沿線淨空 ≥ R＋MARGIN


def extent(c, u, R, lim):
    d = 0.0
    while d < lim:
        x = c + u * (d + 0.05)
        if at(D_T, x) < R + MARGIN[0] or min(np.hypot(*(x - s)), np.hypot(*(x - g))) < a.zone + R:
            break
        d += 0.05
    return d


def seg_ok(p, q, R):
    n = max(2, int(np.hypot(*(q - p)) / 0.05))
    return all(at(D_T, p + (q - p) * i / n) >= R + MARGIN[0] and min(np.hypot(*(p + (q - p) * i / n - s)), np.hypot(*(p + (q - p) * i / n - g))) >= a.zone + R
               for i in range(n + 1))


def cross_point(p, q):
    """路段 pq 與名目路徑在 20–80% 弧長內的第一個交點 → (弧長, 點)。"""
    for i in range(len(P) - 1):
        if not (0.2 * SLEN <= cum[i] <= 0.8 * SLEN):
            continue
        a1, a2 = P[i], P[i + 1]
        d1, d2 = q - p, a2 - a1
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-9:
            continue
        t = ((a1[0] - p[0]) * d2[1] - (a1[1] - p[1]) * d2[0]) / den
        u = ((a1[0] - p[0]) * d1[1] - (a1[1] - p[1]) * d1[0]) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return cum[i] + u * np.hypot(*d2), p + t * d1, t * np.hypot(*d1)
    return None


names = sorted(SHAPE); rng.shuffle(names)
names.sort(key=lambda n: SHAPE[n])                                    # 小的先當穿越者（大推車常塞不進走廊、門口）
movers, mmeta = [], []
n_cross = 0
USED_S = []


def sample(nm, want_cross, min_len, tries):
    R = SHAPE[nm]
    for _t in range(tries):
        if want_cross:
            q0 = mid[rng.randrange(len(mid))]; c = q0 + np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)])
        else:
            c = np.array([rng.uniform(ox + 1, ox + w * res - 1), rng.uniform(oy + 1, oy + h * res - 1)])
        if at(D_T, c) < R + 0.3:
            continue
        th = rng.uniform(0, 2 * math.pi)
        if rng.random() < 0.5:                                         # 一半沿走廊／走道方向（倉儲、辦公室的人流多半順著通道）
            th = rng.randrange(4) * math.pi / 2 + math.radians(rng.uniform(-5, 5))
        u = np.array([math.cos(th), math.sin(th)])
        e1, e2 = extent(c, -u, R, rng.uniform(3, 7)), extent(c, u, R, rng.uniform(3, 7))
        if e1 + e2 < min_len:
            continue
        p, q = c - u * e1, c + u * e2
        if not seg_ok(p, q, R):
            continue
        cp = cross_point(p, q)
        if want_cross and (cp is None or any(abs(cp[0] - u) < 2.0 for u in USED_S)):
            continue                                                   # 穿越點彼此沿路徑 ≥2 m，不擠在同一處
        return p, q, cp
    return None


for nm in names:
    R = SHAPE[nm]
    v = rng.uniform(a.v_lo, a.v_hi)
    best, role = None, 'background'
    if n_cross < a.n_cross:
        best = sample(nm, True, 4.0, 4000)
        role = 'cross' if best else 'background'
    if best is None:
        for ml in (5.0, 4.0, 3.0):
            best = sample(nm, False, ml, 3000)
            if best:
                break
    if best is None:                                                  # 最後退路：大推車在窄布局（倉儲）只要求淨空 R+0.1、長 2.5 m
        _m = MARGIN[0]; MARGIN[0] = 0.1
        best = sample(nm, False, 2.5, 10000)
        MARGIN[0] = _m
        if best:
            role = 'background_tight'
    if best is None:
        raise SystemExit(f'!! {nm} 放不下')
    p, q, cp = best
    L = float(np.hypot(*(q - p)))
    if role == 'cross':
        s_c, xy_c, d_c = cp
        T = s_c / a.v_nom + rng.uniform(-a.jitter, a.jitter)
        s_tw = d_c if rng.random() < 0.5 else 2 * L - d_c
        s0 = (s_tw - v * T) % (2 * L)
        n_cross += 1; USED_S.append(float(s_c))
        info = dict(s_cross_m=round(float(s_c), 2), frac=round(float(s_c / SLEN), 3), cross_xy=[round(float(xy_c[0]), 3), round(float(xy_c[1]), 3)],
                    t_pass_s=round(float(T), 1))
    else:
        s0 = rng.uniform(0, 2 * L); info = {}
    movers.append(dict(name=nm, start=[round(float(p[0]), 3), round(float(p[1]), 3)], end=[round(float(q[0]), 3), round(float(q[1]), 3)],
                       speed=round(v, 4), radius=round(R, 3), height=1.0, phase0_m=round(float(s0), 4), direction=1.0))
    mmeta.append(dict(name=nm, role=role, v=round(v, 4), len_m=round(L, 2), **info))

# 開場 4 s 內行人不得掃到起點（表面距離 ≥0.5 m）
for o in movers:
    ob = dict(pts=[o['start'], o['end']], cum=[0.0, math.hypot(o['end'][0] - o['start'][0], o['end'][1] - o['start'][1])], speed=o['speed'],
              phase0_m=o['phase0_m'], direction=1.0, profile=None)
    dmin = min(math.hypot(schedule_state(ob, t * 0.25)[0][0] - s[0], schedule_state(ob, t * 0.25)[0][1] - s[1]) for t in range(17)) - o['radius'] - 0.30
    if dmin < 0.5:
        raise SystemExit(f'!! {o["name"]} 開場太近起點（{dmin:.2f} m）')

# ---- 輸出 ----
os.makedirs(a.out_dir, exist_ok=True)
unk_out = [dict(name=p['name'], x=p['x'], y=p['y'], yaw=p['yaw']) for p in placed]
static_movers = [dict(name=nm, start=[30.0 + 2.0 * j, 30.0], end=[30.0 + 2.0 * j, 31.0], speed=0.0, radius=round(SHAPE[nm], 3), height=1.0,
                      phase0_m=0.0, direction=1.0) for j, nm in enumerate(sorted(SHAPE))]
common = dict(layout=a.layout, seed=a.seed, start=[round(float(s[0]), 3), round(float(s[1]), 3)], goal=[round(float(g[0]), 3), round(float(g[1]), 3)],
              path_len_m=round(SLEN, 2), path_len_with_unknown_m=round(plen(P2), 2),
              path=[[round(float(x), 3), round(float(y), 3)] for x, y in P[::4]], unknown=placed)
for cond, dyn, meta_m in (('static', static_movers, []), ('crossing', movers, mmeta)):
    base = f'{a.layout}_{cond}_k{a.seed}'
    with open(os.path.join(a.out_dir, base + '.yaml'), 'w') as f:
        f.write(f'# N1 {a.layout} 條件 {cond} 種子 {a.seed}；make_n1_scenario.py 產生；必須搭配 AMMR_OBSTACLE_MODE=scheduled＋SCENE={a.layout}\n')
        yaml.safe_dump(dict(dynamic_obstacles=dyn, unknown_static=unk_out), f, allow_unicode=True, sort_keys=False)
    json.dump(dict(common, cond=cond, movers=meta_m,
                   params=dict(v=[a.v_lo, a.v_hi], n_cross=a.n_cross, n_route_unk=a.n_route_unk, v_nom=a.v_nom, jitter=a.jitter, zone=a.zone)),
              open(os.path.join(a.out_dir, base + '_meta.json'), 'w'), ensure_ascii=False, indent=1)
pc_path = a.poses_csv or os.path.join(WS, 'evaluation', 'results', f'n1_poses_{a.layout}.csv')
rows = {}
if os.path.exists(pc_path):
    rows = {int(r['seed']): r for r in csv.DictReader(open(pc_path))}
rows[a.seed] = dict(seed=a.seed, start_x=round(float(s[0]), 3), start_y=round(float(s[1]), 3), goal_x=round(float(g[0]), 3), goal_y=round(float(g[1]), 3),
                    straight_m=round(float(np.hypot(*(s - g))), 2))
with open(pc_path, 'w', newline='') as f:
    wr = csv.DictWriter(f, fieldnames=['seed', 'start_x', 'start_y', 'goal_x', 'goal_y', 'straight_m'])
    wr.writeheader()
    for k in sorted(rows):
        wr.writerow(rows[k])
print(f'{a.layout} k{a.seed}: 路長 {SLEN:.1f} m（含未知 {plen(P2):.1f}）、未知 {len(placed)}（路線旁 {sum(p["near_route"] for p in placed)}）、行人 {len(movers)}（穿越 {n_cross}）')
