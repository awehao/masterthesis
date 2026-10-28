#!/usr/bin/env python3
"""mission_room 的未知靜態障礙物與移動障礙物（Howard 10/28：要有一些未知靜態障礙物跟移動障礙物）。
原則（feedback_scenario_design）：不手擺在路線上、不瞄準機器人；遭遇由時間產生，趟次間會變，須多趟讀。
  未知靜態：不寫入 mission_room.sdf（導航靜態地圖看不到），A 房 N_A、B 房 N_B 個箱體／椅子，拒絕取樣（固定種子）：
    離牆與已知家具 ≥ 0.6 m、彼此 ≥ 1.2 m、離關鍵停位（S、D1、B1、K、R）≥ 1.2 m、不進門操作區（門洞前後 x∈[−1.0, 2.0]、y∈[−1.9, 1.6]）；
    取樣後以 0.05 m 格點、機器人半徑 0.35 m＋0.10 m 核連通：S→D1、門洞內側→K、K→R、R(外側出門點)→S 皆須有路，任一不通即重抽。
  移動：行人（圓柱 r 0.25、高 1.7 m）沿長直線來回（端點停 1–3 s），速度 0.4–0.8 m/s，路線不經門洞與門操作區，不指向任何停位；
    A 房 2 人（東西向橫越、南北向東側）、B 房 1 人（東西向）。相位由種子定。
r2：r1 行人線 b1（y 3.9）距抽屜停位 K 0.26 m（等於穿過停位）、a1 起點撞西牆矮櫃、a1 線穿未知箱 a3 ⇒ 改線並加核：
  行人線對已知家具 ≥ 半徑＋0.15 m、對停位 ≥ 0.9 m；未知靜態離行人線 ≥ 半徑＋外接半徑＋0.15 m。r1 輸出改名 *_r1_walker_conflict.json 保留。
輸出 results/mission/mission_obstacles_seed<k>.json（不覆寫）。

    python3 evaluation/mission_room_obstacles.py 1
"""
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
SDF = os.path.join(WS, 'src/ammr_bringup/worlds/mission_room.sdf')
POSES = os.path.join(HERE, 'results', 'mission', 'mission_room_poses.json')
ROOM = (-4.0, 5.0, -5.45, 5.45)                     # 外牆內面
DOOR_ZONE = (-1.0, 2.0, -1.9, 1.6)
DOOR_WALL = (-1.5, 2.5, 0.0, 0.12)                   # 門資產牆（門洞 x∈[0,1] 另挖空）
CABINET = (2.4 - 0.3, 2.4 + 0.3, 5.05 - 0.26, 5.05 + 0.23)
N_A, N_B = 3, 2
SHAPES = [('box', [0.5, 0.4, 0.6]), ('box', [0.6, 0.6, 0.45]), ('chair', [0.45, 0.45, 0.9]), ('box', [0.4, 0.4, 0.8])]
R_ROBOT = 0.45
WALKERS = [  # 名稱、端點、房間
    ('walker_a1', (-2.9, -2.6), (3.8, -2.6)),          # r3：東端縮到 3.8（r2 與 a2 線在東南角交會 0.15 m）
    ('walker_a2', (4.55, -4.9), (4.55, -1.8)),
    ('walker_b1', (-2.2, 3.25), (4.4, 3.25)),
]


def seg_dist(x, y, p0, p1):
    a, b, c = np.array(p0, float), np.array(p1, float), np.array([x, y], float)
    t = float(np.clip((c - a) @ (b - a) / max(1e-12, (b - a) @ (b - a)), 0.0, 1.0))
    return float(np.linalg.norm(c - (a + t * (b - a))))


def walker_ok(kb, keys):
    """r2：行人線段對已知家具（外接矩形，沿線 0.05 m 取樣）≥ 半徑＋0.15 m、對停位 ≥ 0.9 m（不穿過停位）。"""
    bad = []
    for n, p0, p1 in WALKERS:
        for t in np.linspace(0, 1, int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 0.05) + 1):
            x, y = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
            if min(x - ROOM[0], ROOM[1] - x, y - ROOM[2], ROOM[3] - y) < 0.4 or any(rect_dist(x, y, r) < 0.4 for r in kb):
                bad.append((n, 'furniture', round(x, 2), round(y, 2)))
                break
        if any(seg_dist(k[0], k[1], p0, p1) < 0.9 for k in keys):
            bad.append((n, 'pose'))
    return bad


def known_boxes():
    raw = re.sub(r'<!--.*?-->', '', open(SDF, encoding='utf-8').read(), flags=re.S)
    out = []
    for m in ET.fromstring(raw).find('world').findall('model'):
        b = m.find('.//collision/geometry/box')
        if b is None:
            continue
        p = [float(v) for v in m.findtext('pose').split()][:2]
        s = [float(v) for v in b.findtext('size').split()][:2]
        out.append((p[0] - s[0] / 2, p[0] + s[0] / 2, p[1] - s[1] / 2, p[1] + s[1] / 2))
    out += [(DOOR_WALL[0], 0.0, DOOR_WALL[2], DOOR_WALL[3]), (1.0, DOOR_WALL[1], DOOR_WALL[2], DOOR_WALL[3]), CABINET]
    return out


def rect_dist(x, y, r):
    dx = max(r[0] - x, 0.0, x - r[1])
    dy = max(r[2] - y, 0.0, y - r[3])
    return math.hypot(dx, dy)


def free_grid(rects, res=0.05):
    xs = np.arange(ROOM[0], ROOM[1] + 1e-9, res)
    ys = np.arange(ROOM[2], ROOM[3] + 1e-9, res)
    F = np.ones((len(xs), len(ys)), bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if min(x - ROOM[0], ROOM[1] - x, y - ROOM[2], ROOM[3] - y) < R_ROBOT or any(rect_dist(x, y, r) < R_ROBOT for r in rects):
                F[i, j] = False
    return xs, ys, F


def connected(xs, ys, F, a, b, res=0.05):
    ia = (int(round((a[0] - xs[0]) / res)), int(round((a[1] - ys[0]) / res)))
    ib = (int(round((b[0] - xs[0]) / res)), int(round((b[1] - ys[0]) / res)))
    if not (F[ia] and F[ib]):
        return False
    seen, q = {ia}, deque([ia])
    while q:
        c = q.popleft()
        if c == ib:
            return True
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if 0 <= n[0] < F.shape[0] and 0 <= n[1] < F.shape[1] and F[n] and n not in seen:
                seen.add(n)
                q.append(n)
    return False


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = os.path.join(HERE, 'results', 'mission', f'mission_obstacles_seed{seed}.json')
    if os.path.exists(out):
        raise SystemExit(f'{out} 已存在，不覆寫')
    rng = np.random.default_rng(seed)
    P = json.load(open(POSES))['poses']
    keys = [(v['x'], v['y']) for v in P.values()]
    kb = known_boxes()
    # 連通核對點：門洞外側／內側各取門洞中心前後 0.9 m（門關著時門洞本身不通，分房核）
    pairs = [((P['S']['x'], P['S']['y']), (P['D1']['x'], P['D1']['y'])), ((0.5, 0.9), (P['K']['x'], P['K']['y'])),
             ((P['K']['x'], P['K']['y']), (P['R']['x'], P['R']['y'])), ((0.5, -0.9), (P['S']['x'], P['S']['y']))]
    wb = walker_ok(kb, keys)
    if wb:
        raise SystemExit(f'行人路線不合格：{wb}')
    for attempt in range(200):
        obs = []
        for room, n in (('A', N_A), ('B', N_B)):
            tries = 0
            while sum(1 for o in obs if o['room'] == room) < n and tries < 5000:
                tries += 1
                x = rng.uniform(ROOM[0] + 0.6, ROOM[1] - 0.6)
                y = rng.uniform(ROOM[2] + 0.6, -0.6) if room == 'A' else rng.uniform(0.7, ROOM[3] - 0.6)
                kind, size = SHAPES[int(rng.integers(len(SHAPES)))]
                yaw = float(rng.uniform(0, math.pi))
                h = 0.5 * math.hypot(size[0], size[1])
                if any(rect_dist(x, y, r) < 0.6 + h for r in kb):
                    continue
                if DOOR_ZONE[0] - h <= x <= DOOR_ZONE[1] + h and DOOR_ZONE[2] - h <= y <= DOOR_ZONE[3] + h:
                    continue
                if any(math.hypot(x - k[0], y - k[1]) < 1.2 + h for k in keys):
                    continue
                if any(math.hypot(x - o['x'], y - o['y']) < 1.2 + h + o['r_bound'] for o in obs):
                    continue
                if any(seg_dist(x, y, w[1], w[2]) < 0.25 + h + 0.15 for w in WALKERS):   # r2：不放在行人線上
                    continue
                obs.append({'name': f'unknown_{room.lower()}{sum(1 for o in obs if o["room"] == room) + 1}', 'room': room, 'kind': kind,
                            'x': float(x), 'y': float(y), 'yaw_deg': math.degrees(yaw), 'size': size, 'r_bound': h})
        rects = kb + [(o['x'] - o['r_bound'], o['x'] + o['r_bound'], o['y'] - o['r_bound'], o['y'] + o['r_bound']) for o in obs]   # 外接方形（保守）
        xs, ys, F = free_grid(rects)
        conn = [connected(xs, ys, F, a, b) for a, b in pairs]
        if all(conn):
            break
        print(f'attempt {attempt}: 連通 {conn} ⇒ 重抽', flush=True)
    else:
        raise SystemExit('200 次內找不到連通配置')
    walkers = []
    for name, p0, p1 in WALKERS:
        walkers.append({'name': name, 'p0': list(p0), 'p1': list(p1), 'speed_mps': float(rng.uniform(0.4, 0.8)),
                        'dwell_s': float(rng.uniform(1.0, 3.0)), 'phase_s': float(rng.uniform(0.0, 20.0)), 'radius': 0.25, 'height': 1.7})
    doc = {'schema': 'mission_obstacles/1', 'seed': seed, 'map': os.path.relpath(SDF, WS), 'poses': os.path.relpath(POSES, HERE),
           'rule': __doc__.split('輸出')[0].strip(), 'attempts': attempt + 1, 'connectivity_pairs': pairs, 'static_unknown': obs, 'walkers': walkers}
    json.dump(doc, open(out, 'w'), ensure_ascii=False, indent=1)
    print('寫', out)
    for o in obs:
        print(f"  {o['name']} {o['kind']} ({o['x']:.2f}, {o['y']:.2f}) {o['size']}")
    for w in walkers:
        print(f"  {w['name']} {w['p0']}↔{w['p1']} {w['speed_mps']:.2f} m/s 停 {w['dwell_s']:.1f} s 相位 {w['phase_s']:.1f} s")


if __name__ == '__main__':
    main()
