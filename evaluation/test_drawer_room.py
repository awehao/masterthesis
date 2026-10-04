#!/usr/bin/env python3
"""drawer_room.sdf 的幾何核對。**不開模擬器**，純幾何。

要核的事，每一件都算，不靠看圖：
  A 櫃體與全開抽屜的**保留區真的留空** —— 場景裡沒有任何碰撞體侵入
  B 起點以導航足跡半徑淨空
  C 停位以導航足跡半徑淨空，且不侵入保留區
  D 交棒區（停位周圍 0.30 m）淨空
  E 房間封閉（四面牆圍成閉環，lidar 不會看穿）
  F 起點到交棒區存在導航足跡通得過的路徑（粗格 BFS）
  G 傢俱不在起點到櫃體的直線上形成漏斗

座標來源：櫃體 (0, 1.45) 是凍結值；保留區由 drawer_unit.yaml 推得。
導航足跡半徑取**實際配置值**：costmap 0.28、shield 0.30、GMPC 0.33，
核對用最大的 0.33。
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
WORLD = os.path.join(WS, 'src/ammr_bringup/worlds/drawer_room.sdf')
ASSET = os.path.join(WS, 'src/my_omnibot_description/config/drawer_unit.yaml')

# 導航足跡半徑（實際配置值；核對取最大者）
R_COSTMAP, R_SHIELD, R_GMPC = 0.28, 0.30, 0.33
R_NAV = max(R_COSTMAP, R_SHIELD, R_GMPC)

CABINET_XY = (0.0, 1.45)          # **凍結值**
START = (-2.80, -3.20)
PARK = (-0.136412, 0.56)
HANDOVER_ZONE_M = 0.30

N = 0
BAD = []


def chk(name, cond, extra=''):
    global N
    N += 1
    print(f'  {"ok  " if cond else "FAIL"}  {name}'
          + (f'  — {extra}' if extra else ''))
    if not cond:
        BAD.append(name)


# ---------------------------------------------------------------- 場景幾何
def world_boxes():
    """回傳 [(name, xlo, xhi, ylo, yhi)]，只取**碰撞**幾何，忽略地板。"""
    out = []
    w = ET.parse(WORLD).getroot().find('world')
    for m in w.findall('model'):
        nm = m.get('name')
        if nm == 'ground_plane':
            continue
        pose = m.find('pose')
        px, py = (0.0, 0.0)
        if pose is not None:
            v = [float(x) for x in pose.text.split()]
            px, py = v[0], v[1]
        for link in m.findall('link'):
            for col in link.findall('collision'):
                g = col.find('geometry')
                b = g.find('box')
                c = g.find('cylinder')
                if b is not None:
                    sx, sy, _ = [float(x) for x in b.find('size').text.split()]
                    out.append((nm, px - sx/2, px + sx/2, py - sy/2, py + sy/2))
                elif c is not None:
                    r = float(c.find('radius').text)
                    out.append((nm, px - r, px + r, py - r, py + r))
    return out


def reserved_zone():
    d = yaml.safe_load(open(ASSET))
    xs, ys = [], []
    for _, c, s in d['cabinet'] + d['drawer']['body'] \
            + d['drawer']['handle']['posts']:
        xs += [c[0] - s[0]/2, c[0] + s[0]/2]
        ys += [c[1] - s[1]/2, c[1] + s[1]/2]
    bar = d['drawer']['handle']['bar']
    xs += [bar['center'][0] - bar['length']/2,
           bar['center'][0] + bar['length']/2]
    ys += [bar['center'][1] - bar['radius'], bar['center'][1] + bar['radius']]
    upper = float(d['drawer']['joint']['upper'])
    return (CABINET_XY[0] + min(xs), CABINET_XY[0] + max(xs),
            CABINET_XY[1] + min(ys) - upper, CABINET_XY[1] + max(ys))


def box_dist(p, bx):
    """點到軸對齊矩形的距離（在矩形內回 0）。"""
    _, xlo, xhi, ylo, yhi = bx
    dx = max(xlo - p[0], 0.0, p[0] - xhi)
    dy = max(ylo - p[1], 0.0, p[1] - yhi)
    return float(np.hypot(dx, dy))


BOXES = world_boxes()
RX0, RX1, RY0, RY1 = reserved_zone()
print(f'場景碰撞體 {len(BOXES)} 個；導航足跡半徑取 {R_NAV}'
      f'（costmap {R_COSTMAP}／shield {R_SHIELD}／GMPC {R_GMPC}）')
print(f'保留區（世界）x [{RX0:+.3f}, {RX1:+.3f}]  y [{RY0:+.3f}, {RY1:+.3f}]')

# ---------------------------------------------------------------- A 保留區留空
print('\nA 櫃體與全開抽屜的保留區')
hit = [nm for nm, xlo, xhi, ylo, yhi in BOXES
       if not (xhi <= RX0 or xlo >= RX1 or yhi <= RY0 or ylo >= RY1)]
chk('A1 沒有碰撞體侵入保留區', not hit, f'侵入者 {hit}' if hit else '')
# 北牆應**貼近但不侵入**：內面與櫃體背面的間隙
north = [b for b in BOXES if b[0] == 'wall_north'][0]
gap_n = north[3] - RY1
chk('A2 北牆在保留區之外', gap_n >= 0.0, f'間隙 {gap_n*1e3:.1f} mm')
chk('A3 北牆貼得夠近（間隙 ≤ 0.10 m，櫃體算靠牆）', 0.0 <= gap_n <= 0.10,
    f'{gap_n*1e3:.1f} mm')

# ---------------------------------------------------------------- B 起點
print('\nB 起點')
dmin = min((box_dist(START, b), b[0]) for b in BOXES)
chk('B1 起點以導航足跡淨空', dmin[0] > R_NAV,
    f'最近 {dmin[1]} 距 {dmin[0]:.3f} m（需 > {R_NAV}）')
chk('B2 起點在房間內',
    -4.0 < START[0] < 4.0 and -5.0 < START[1] < 1.70)
d_res = box_dist(START, ('reserved', RX0, RX1, RY0, RY1))
chk('B3 起點離保留區夠遠', d_res > 1.0, f'{d_res:.3f} m')

# ---------------------------------------------------------------- C 停位
print('\nC 停位')
dmin = min((box_dist(PARK, b), b[0]) for b in BOXES)
chk('C1 停位以導航足跡淨空（對場景物件）', dmin[0] > R_NAV,
    f'最近 {dmin[1]} 距 {dmin[0]:.3f} m')
d_res = box_dist(PARK, ('reserved', RX0, RX1, RY0, RY1))
chk('C2 停位的導航足跡**不侵入**保留區', d_res > R_NAV,
    f'距保留區 {d_res:.3f} m，餘裕 {(d_res-R_NAV)*1e3:+.1f} mm')
# 開啟段底盤會退到 y = 0.36；那時抽屜也跟著出來，相對距離不變，一併核
PARK_OPEN = (PARK[0], PARK[1] - 0.200)
RES_OPEN = ('reserved_open', RX0, RX1, RY0, RY1 - 0.0)   # 保留區已含全開
d_open = box_dist(PARK_OPEN, RES_OPEN)
chk('C3 全開時底盤仍不侵入保留區', d_open > R_NAV,
    f'距 {d_open:.3f} m，餘裕 {(d_open-R_NAV)*1e3:+.1f} mm')

# ---------------------------------------------------------------- D 交棒區
print('\nD 交棒區')
ok_zone = True
worst = (9e9, None)
for th in np.linspace(0, 2*np.pi, 73)[:-1]:
    p = (PARK[0] + HANDOVER_ZONE_M*np.cos(th),
         PARK[1] + HANDOVER_ZONE_M*np.sin(th))
    for b in BOXES:
        d = box_dist(p, b)
        if d < worst[0]:
            worst = (d, b[0])
        if d <= R_NAV:
            ok_zone = False
chk('D1 交棒區邊界全周以導航足跡淨空', ok_zone,
    f'最近 {worst[1]} 距 {worst[0]:.3f} m')

# ---------------------------------------------------------------- E 房間封閉
print('\nE 房間封閉')
names = {b[0] for b in BOXES}
chk('E1 四面牆都在', {'wall_north', 'wall_south', 'wall_east',
                      'wall_west'} <= names, str(sorted(names)))
W = {b[0]: b for b in BOXES if b[0].startswith('wall_')}
# 牆要在四角互相搭到：東西牆的 y 範圍要覆蓋南北牆的 y 位置
cover_n = (W['wall_east'][3] <= W['wall_north'][4]
           and W['wall_east'][4] >= W['wall_north'][3])
cover_s = (W['wall_east'][3] <= W['wall_south'][4]
           and W['wall_east'][4] >= W['wall_south'][3])
chk('E2 東牆與南北牆在轉角重疊', cover_n and cover_s,
    f'東牆 y [{W["wall_east"][3]:+.2f}, {W["wall_east"][4]:+.2f}]')
span_x = (W['wall_north'][1] <= W['wall_west'][2]
          and W['wall_north'][2] >= W['wall_east'][1])
chk('E3 南北牆橫跨到東西牆', span_x,
    f'北牆 x [{W["wall_north"][1]:+.2f}, {W["wall_north"][2]:+.2f}]')

# ---------------------------------------------------------------- F 路徑連通
print('\nF 起點到交棒區的連通性（粗格 BFS，足跡已內縮）')
GS = 0.05
xlo, xhi, ylo, yhi = -4.0, 4.0, -5.0, 1.70
nx = int((xhi - xlo) / GS) + 1
ny = int((yhi - ylo) / GS) + 1
free = np.ones((nx, ny), bool)
for i in range(nx):
    for j in range(ny):
        p = (xlo + i*GS, ylo + j*GS)
        if any(box_dist(p, b) <= R_NAV for b in BOXES):
            free[i, j] = False
            continue
        if box_dist(p, ('res', RX0, RX1, RY0, RY1)) <= R_NAV:
            free[i, j] = False


def cell(p):
    return (int(round((p[0]-xlo)/GS)), int(round((p[1]-ylo)/GS)))


s, g = cell(START), cell(PARK)
chk('F1 起點格可行', free[s], str(s))
# 停位本身在交棒區中心；目標設為交棒區內任一可行格
goal_cells = set()
for i in range(nx):
    for j in range(ny):
        if free[i, j] and np.hypot(xlo+i*GS-PARK[0],
                                   ylo+j*GS-PARK[1]) <= HANDOVER_ZONE_M:
            goal_cells.add((i, j))
chk('F2 交棒區內有可行格', len(goal_cells) > 0, f'{len(goal_cells)} 格')
seen = {s}
q = deque([s])
reached = False
while q:
    c = q.popleft()
    if c in goal_cells:
        reached = True
        break
    for dd in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nxt = (c[0]+dd[0], c[1]+dd[1])
        if (0 <= nxt[0] < nx and 0 <= nxt[1] < ny and free[nxt]
                and nxt not in seen):
            seen.add(nxt)
            q.append(nxt)
chk('F3 起點可達交棒區', reached, f'探索 {len(seen)} 格')
chk('F4 可行區佔比合理（房間不是被塞滿）',
    free.sum() / free.size > 0.5, f'{100*free.sum()/free.size:.1f}%')

# ---------------------------------------------------------------- G 不是刻意佈的路障
print('\nG 傢俱不形成漏斗')
seg = np.array(PARK) - np.array(START)
furn = [b for b in BOXES if not b[0].startswith('wall_')]
off = []
for b in furn:
    # 傢俱中心到起點—停位直線的垂距
    c = np.array([(b[1]+b[2])/2, (b[3]+b[4])/2])
    t = float(np.clip((c - START) @ seg / (seg @ seg), 0.0, 1.0))
    d = float(np.linalg.norm(c - (np.array(START) + t*seg)))
    off.append((b[0], d))
chk('G1 沒有傢俱落在起點—停位直線 0.6 m 內',
    all(d > 0.6 for _, d in off),
    '；'.join(f'{nm} {d:.2f} m' for nm, d in off))
chk('G2 傢俱數量少（不是迷宮）', len(furn) <= 5, f'{len(furn)} 件')

print(f'\n{N - len(BAD)}/{N} 通過')
for b in BAD:
    print('  **' + b + '**')
raise SystemExit(1 if BAD else 0)
