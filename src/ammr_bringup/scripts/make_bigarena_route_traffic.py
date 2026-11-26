#!/usr/bin/env python3
"""bigarena 路途穿越交通（11/26，Howard「終點附近不要放，路上多點」）。

每個案例（機器人路線 × 相位種子）：
  1. 由地圖算機器人名目路徑：佔據格膨脹 INFL（同 costmap 0.45 m）後 8 鄰格 A*，再抽稀成折線。
  2. 沿路徑弧長 CROSS_FRAC 區間（預設 20–80%）抽至多 n_cross 個穿越點（彼此沿弧長 ≥ CROSS_GAP），起點／終點 ZONE m 內不取；
     只放在開闊處（穿越點靜態淨空 ≥ OPEN_CLR），離路徑窄處（門口，淨空 < NARROW_CLR）沿弧長至少 NARROW_GAP。
     （第一版沒有這條：route 1 先導三個方法都卡在門口——行人來回穿門，每半個週期就擋一次，測到的是擠門不是動態閃避。）
  3. 每個穿越點配一個移動體，做長距離來回（ping-pong）：
       穿越角度（相對路徑切向）：垂直 70–110°（50%）、斜向 35–70°／110–145°（30%）、
       反向偏側 155–180° 且整段側移 0.6–1.2 m（20%，在路旁逆向走過，不正對機器人）。
       路段兩端各 2.5–5 m，沿線靜態淨空不足就截短；長度 <MIN_LEN 或整段離起點／終點不到 ZONE_SEG 就換角度重抽。
     時機：機器人以 V_NOM 走到穿越點的時刻 ＋ U(−JITTER, JITTER) s，移動體在那時通過穿越點（方向隨機）——
     有的在機器人前面過、有的在後面過、有的交會；不是每個都正對著撞。
  4. 其餘移動體當背景交通：自由空間中隨機長路段，同樣離起點／終點 ZONE_SEG 以上。
移動體的名稱與形狀固定（bigarena.sdf 的 kinematic 模型：5 圓柱、5 方盒）；淨空檢查用包覆半徑。
速度等級與型態同 make_bigarena_dynamic.py（平均速率跨型態一致、輪廓時間偏移不同步）。
必須搭配 AMMR_OBSTACLE_MODE=scheduled。輸出 DIR/bigarena_route_<speed>_<pattern>_r<route>_k<seed>.yaml＋_meta.json。
"""
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
SPEEDS = dict(slow=(0.10, 0.15), walk=(0.30, 0.40), fast=(0.50, 0.60))

ap = argparse.ArgumentParser()
ap.add_argument('--speed', choices=sorted(SPEEDS), required=True)
ap.add_argument('--pattern', choices=('const', 'vary', 'stopgo'), required=True)
ap.add_argument('--route', type=int, required=True)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--out-dir', required=True)
ap.add_argument('--n-cross', type=int, default=7)
ap.add_argument('--v-nom', type=float, default=0.25, help='估機器人到穿越點時刻用的名目速度')
ap.add_argument('--jitter', type=float, default=4.0)
ap.add_argument('--zone', type=float, default=2.5, help='起點／終點此半徑內不取穿越點')
ap.add_argument('--zone-seg', type=float, default=2.0, help='移動體整段路線離起點／終點的最小表面距離')
ap.add_argument('--min-len', type=float, default=3.5)
ap.add_argument('--infl', type=float, default=0.45)
ap.add_argument('--cross-frac', default='0.2,0.8')
ap.add_argument('--open-clr', type=float, default=0.6, help='穿越點處路徑兩側最小靜態淨空（開闊處才放）')
ap.add_argument('--narrow-clr', type=float, default=0.6, help='路徑淨空低於此值＝窄處（門口）')
ap.add_argument('--narrow-gap', type=float, default=1.0, help='穿越點離路徑窄處（沿弧長）至少此距離')
ap.add_argument('--cross-gap', type=float, default=1.5, help='穿越點之間沿弧長最小間距')
ap.add_argument('--seg-margin', type=float, default=0.2, help='移動體路段沿線靜態淨空 ≥ 包覆半徑＋此值（不穿門、不貼牆）')
ap.add_argument('--world', default=os.path.join(PKG, 'worlds', 'bigarena.sdf'))
ap.add_argument('--map', default=os.path.join(PKG, 'maps', 'bigarena'))
ap.add_argument('--poses', default=os.path.join(WS, 'evaluation', 'results', 'bigarena_poses.csv'))
a = ap.parse_args()
f0, f1 = [float(v) for v in a.cross_frac.split(',')]

drv = open(os.path.join(PKG, 'ammr_bringup', 'dynamic_obstacle_driver.py')).read()
ns = {'math': math}
exec(drv[drv.index('def profile_arclength'):drv.index('class DynamicObstacleDriver')], ns)   # noqa: S102
profile_arclength = ns['profile_arclength']

row = next(r for r in csv.DictReader(open(a.poses)) if int(r['seed']) == a.route)
S0 = np.array([float(row['start_x']), float(row['start_y'])])
G0 = np.array([float(row['goal_x']), float(row['goal_y'])])

# ---- 地圖：佔據格＋未知靜態柱（同 analyze.py 的靜態幾何）----
my = yaml.safe_load(open(a.map + '.yaml'))
res, ox, oy = my['resolution'], my['origin'][0], my['origin'][1]
with open(a.map + '.pgm', 'rb') as f:
    assert f.readline().strip() == b'P5'
    line = f.readline()
    while line.startswith(b'#'):
        line = f.readline()
    w, h = map(int, line.split()); int(f.readline())
    img = np.frombuffer(f.read(), np.uint8).reshape(h, w)
occ = ((255 - img.astype(float)) / 255.0) > my['occupied_thresh']
sdf = re.sub(r'<!--.*?-->', '', open(a.world).read(), flags=re.S)
yy, xx = np.mgrid[0:h, 0:w]
XC, YC = ox + (xx + 0.5) * res, oy + (h - 1 - yy + 0.5) * res
for m_ in re.finditer(r'<model name="(unknown_obs_\d+)">(.*?)</model>', sdf, re.S):   # 逐模型解析：bigarena 的未知障礙全是方盒（舊的單一 regex 只認圓柱 ⇒ 漏掉，行人被卡在方盒旁）
    b_ = m_.group(2); pz = [float(v) for v in re.search(r'<pose>([-\d. e]+)</pose>', b_).group(1).split()]
    bx = re.search(r'<box><size>([\d.]+) ([\d.]+)', b_); cy = re.search(r'<cylinder><radius>([\d.]+)', b_)
    if bx:
        th = pz[5] if len(pz) > 5 else 0.0; cs, sn = math.cos(-th), math.sin(-th)
        lx, ly = cs * (XC - pz[0]) - sn * (YC - pz[1]), sn * (XC - pz[0]) + cs * (YC - pz[1])
        occ |= (np.abs(lx) <= float(bx.group(1)) / 2) & (np.abs(ly) <= float(bx.group(2)) / 2)
    elif cy:
        occ |= np.hypot(XC - pz[0], YC - pz[1]) <= float(cy.group(1))
DIST = ndimage.distance_transform_edt(~occ) * res                   # 到最近靜態表面（m）


def cell(p):
    return int(h - 1 - (p[1] - oy) / res), int((p[0] - ox) / res)


def clr(p):
    r, c = cell(p)
    return float(DIST[r, c]) if (0 <= r < h and 0 <= c < w) else 0.0


SHAPE = {}
for m in re.finditer(r'<model name="(dyn_obs_\d+)">(.*?)</model>', sdf, re.S):
    bx = re.search(r'<box><size>([\d.]+) ([\d.]+)', m.group(2)); cy = re.search(r'<cylinder><radius>([\d.]+)', m.group(2))
    SHAPE[m.group(1)] = ('box', float(bx.group(1)), float(bx.group(2))) if bx else ('cyl', float(cy.group(1)), float(cy.group(1)))
RCOV = {k: (math.hypot(v[1], v[2]) / 2 if v[0] == 'box' else v[1]) for k, v in SHAPE.items()}

# ---- 名目路徑：膨脹後 8 鄰格 A*（0.1 m 格）----
step = 2
free = DIST[::step, ::step] > a.infl
H2, W2 = free.shape


def to2(p):
    r, c = cell(p)
    return r // step, c // step


def from2(rc):
    return np.array([ox + (rc[1] * step + 1) * res, oy + (h - 1 - rc[0] * step - 1) * res])


s2, g2 = to2(S0), to2(G0)
for q in (s2, g2):                                                   # 起終點若落在膨脹區，放寬該格
    free[q] = True
nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
gs = {s2: 0.0}; par = {}; pq = [(0.0, s2)]
while pq:
    _, u = heapq.heappop(pq)
    if u == g2:
        break
    for dr, dc, cst in nb:
        v = (u[0] + dr, u[1] + dc)
        if 0 <= v[0] < H2 and 0 <= v[1] < W2 and free[v]:
            ng = gs[u] + cst
            if ng < gs.get(v, 1e18):
                gs[v] = ng; par[v] = u
                heapq.heappush(pq, (ng + math.hypot(v[0] - g2[0], v[1] - g2[1]), v))
if g2 not in par:
    raise SystemExit('!! A* 找不到路徑')
cells = [g2]
while cells[-1] != s2:
    cells.append(par[cells[-1]])
path = np.array([from2(c) for c in cells[::-1]])
path[0], path[-1] = S0, G0
path = ndimage.uniform_filter1d(path, 9, axis=0, mode='nearest')     # 稍微平滑格點鋸齒
path[0], path[-1] = S0, G0
seg = np.hypot(*np.diff(path, axis=0).T); cum = np.r_[0.0, np.cumsum(seg)]; SLEN = float(cum[-1])
PCLR = np.array([clr(p) for p in path])
NARROW_S = cum[PCLR < a.narrow_clr]                                  # 路徑窄處（門口）弧長


def at(s):
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
    f = (s - cum[i]) / max(seg[i], 1e-9)
    p = path[i] + f * (path[i + 1] - path[i])
    t = (path[i + 1] - path[i]) / max(seg[i], 1e-9)
    return p, t


rng = random.Random(zlib.crc32(f'route|{a.speed}|{a.pattern}|{a.route}|{a.seed}'.encode()))


def seg_ok(p, q, R):
    """整段靜態淨空 ≥ R＋0.1、離起終點 ≥ zone_seg＋R。"""
    n = max(2, int(np.hypot(*(q - p)) / 0.05))
    for k in range(n + 1):
        x = p + (q - p) * k / n
        if clr(x) < R + a.seg_margin:
            return False
        if min(np.hypot(*(x - S0)), np.hypot(*(x - G0))) < a.zone_seg + R:
            return False
    return True


def extent(c, u, R, lim):
    """從 c 沿 u 走到淨空不足或進入起終點區為止的距離（上限 lim）。"""
    d = 0.0
    while d < lim:
        x = c + u * (d + 0.05)
        if clr(x) < R + a.seg_margin or min(np.hypot(*(x - S0)), np.hypot(*(x - G0))) < a.zone_seg + R:
            break
        d += 0.05
    return d


names = sorted(SHAPE)
rng.shuffle(names)
names.sort(key=lambda n: RCOV[n])                                    # 小的先試穿越（大推車在多牆的 bigarena 常放不進開闊處）；同半徑依種子順序
n_cross = min(a.n_cross, len(names))
n_placed = 0
lo_s, hi_s = f0 * SLEN, f1 * SLEN
used_s = []
out_obs, meta = [], []
for k, nm in enumerate(names):
    R = RCOV[nm]
    lo, hi = SPEEDS[a.speed]
    v_mean = rng.uniform(lo, hi)
    prof, v0 = None, v_mean
    if a.pattern == 'vary':
        prof = dict(type='sine', amp=0.5, period_s=round(rng.uniform(6.0, 10.0), 2)); prof['offset_s'] = round(rng.uniform(0.0, prof['period_s']), 2)
    elif a.pattern == 'stopgo':
        go, st, rp = round(rng.uniform(4.0, 7.0), 2), round(rng.uniform(1.5, 3.0), 2), 0.8
        prof = dict(type='stopgo', go_s=go, stop_s=st, ramp_s=rp, offset_s=round(rng.uniform(0.0, go + st), 2))
        v0 = v_mean * (go + st) / (go - rp)
    placed = None
    if n_placed < n_cross:
        for _ in range(600):
            s_c = rng.uniform(lo_s, hi_s)                              # 整段中段取點，與已放穿越點沿弧長 ≥ cross_gap
            if any(abs(s_c - q) < a.cross_gap for q in used_s):
                continue
            c, tau = at(s_c)
            if min(np.hypot(*(c - S0)), np.hypot(*(c - G0))) < a.zone:
                continue
            if clr(c) < a.open_clr or (len(NARROW_S) and float(np.min(np.abs(NARROW_S - s_c))) < a.narrow_gap):
                continue
            r_ = rng.random()
            if r_ < 0.5:
                kind, th, off = 'cross', math.radians(rng.uniform(70, 110)), 0.0
            elif r_ < 0.8:
                kind, th, off = 'oblique', math.radians(rng.choice([rng.uniform(35, 70), rng.uniform(110, 145)])), 0.0
            else:
                kind, th, off = 'counter', math.radians(rng.uniform(155, 180)), rng.choice([-1, 1]) * rng.uniform(0.6, 1.2)
            th *= rng.choice([-1, 1])
            u = np.array([math.cos(th) * tau[0] - math.sin(th) * tau[1], math.sin(th) * tau[0] + math.cos(th) * tau[1]])
            nrm = np.array([-tau[1], tau[0]])
            cc = c + off * nrm
            if clr(cc) < R + a.seg_margin:
                continue
            e_neg, e_pos = extent(cc, -u, R, rng.uniform(2.5, 5.0)), extent(cc, u, R, rng.uniform(2.5, 5.0))
            if e_neg + e_pos < a.min_len:
                continue
            p, q = cc - u * e_neg, cc + u * e_pos
            if not seg_ok(p, q, R):
                continue
            L = float(np.hypot(*(q - p)))
            T = s_c / a.v_nom + rng.uniform(-a.jitter, a.jitter)
            d_c = e_neg
            s_tw = d_c if rng.random() < 0.5 else 2 * L - d_c                # 通過穿越點時的行進方向
            dist_T, _ = profile_arclength(prof, v0, T)
            used_s.append(s_c); n_placed += 1
            placed = dict(p=p, q=q, L=L, s0=(s_tw - dist_T) % (2 * L),
                          info=dict(role='cross', kind=kind, angle_deg=round(math.degrees(th), 1), lateral_off_m=round(off, 2),
                                    s_cross_m=round(s_c, 2), frac=round(s_c / SLEN, 3), cross_xy=[round(float(c[0]), 3), round(float(c[1]), 3)],
                                    eta_nom_s=round(s_c / a.v_nom, 1), t_pass_s=round(T, 1), len_m=round(L, 2)))
            break
    if placed is None:                                                # 背景交通（或穿越點找不到）
        # 前 2000 次最短 5 m（原規則；亂數序列不變 ⇒ 原本放得下的案例產生結果完全相同），
        # 放不下才放寬到 4 m、3 m 再各試 2000 次（11/26 dyn1126 fast const r2 的 dyn_obs_5 大推車放不下 ⇒ 整案例被跳過）
        for it in range(6000):
            len_min = 5.0 if it < 2000 else (4.0 if it < 4000 else 3.0)
            c = np.array([rng.uniform(ox, ox + w * res), rng.uniform(oy, oy + h * res)])
            if clr(c) < R + 0.3:
                continue
            th = rng.uniform(0, 2 * math.pi); u = np.array([math.cos(th), math.sin(th)])
            e_neg, e_pos = extent(c, -u, R, rng.uniform(3, 5)), extent(c, u, R, rng.uniform(3, 5))
            if e_neg + e_pos < len_min:
                continue
            p, q = c - u * e_neg, c + u * e_pos
            if not seg_ok(p, q, R):
                continue
            L = float(np.hypot(*(q - p)))
            placed = dict(p=p, q=q, L=L, s0=rng.uniform(0, 2 * L), info=dict(role='background', len_m=round(L, 2)))
            if len_min < 5.0:
                placed['info']['len_min_relaxed'] = len_min
            break
    if placed is None:
        raise SystemExit(f'!! {nm} 放不下')
    o = dict(name=nm, start=[round(float(placed['p'][0]), 3), round(float(placed['p'][1]), 3)],
             end=[round(float(placed['q'][0]), 3), round(float(placed['q'][1]), 3)], speed=round(v0, 4), radius=round(R, 3),
             height=1.0, phase0_m=round(float(placed['s0']), 4), direction=1.0)
    if prof:
        o['speed_profile'] = prof
    out_obs.append(o)
    meta.append(dict(name=nm, shape=SHAPE[nm], v_mean=round(v_mean, 4), v_cruise=round(v0, 4), **placed['info']))

os.makedirs(a.out_dir, exist_ok=True)
base = f'bigarena_route_{a.speed}_{a.pattern}_r{a.route}_k{a.seed}'
with open(os.path.join(a.out_dir, base + '.yaml'), 'w') as f:
    f.write(f'# bigarena 路途穿越交通：速度 {a.speed} {SPEEDS[a.speed]} m/s（平均）、型態 {a.pattern}、路線 {a.route}、相位種子 {a.seed}\n'
            f'# 由 src/ammr_bringup/scripts/make_bigarena_route_traffic.py 產生；radius＝包覆半徑（方盒形狀以世界檔為準）；必須搭配 AMMR_OBSTACLE_MODE=scheduled。\n')
    yaml.safe_dump(dict(dynamic_obstacles=out_obs), f, allow_unicode=True, sort_keys=False)
json.dump(dict(speed=a.speed, pattern=a.pattern, route=a.route, seed=a.seed, start=S0.tolist(), goal=G0.tolist(), path_len_m=round(SLEN, 2),
               path=[[round(float(x), 3), round(float(y), 3)] for x, y in path[::5]], obstacles=meta,
               params=dict(n_cross=n_cross, v_nom=a.v_nom, jitter=a.jitter, zone=a.zone, zone_seg=a.zone_seg, cross_frac=[f0, f1],
                           open_clr=a.open_clr, narrow_clr=a.narrow_clr, narrow_gap=a.narrow_gap, seg_margin=a.seg_margin, cross_gap=a.cross_gap),
               narrow_s=[round(float(x), 2) for x in NARROW_S[::5]]),
          open(os.path.join(a.out_dir, base + '_meta.json'), 'w'), ensure_ascii=False, indent=1)
print(os.path.join(a.out_dir, base + '.yaml'))
