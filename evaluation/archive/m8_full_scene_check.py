#!/usr/bin/env python3
"""第八次完整任務場景的離線核對（Howard reviews/20261111_howard_complete_mission_scene.md 交付條件 1–5；不跑模擬）。讀 experiments/20261111_m8_full_scene.json 登錄的同一份檔案。
1 牆體封閉：隔牆帶（y∈[0, 0.12]）沿 x 的覆蓋區間（SDF 牆＋門資產牆＋關閉門扇）⇒ 列出未覆蓋區間（只該剩門扇與門框的縫）；
  0.02 m 格、不膨脹的洪水：外部不能進到房內；關門時 A 房（S）不能到 B 房（抽屜前）；開門時可以。
2 同一套幾何：靜態地圖由同一 SDF 重產（逐格比對結果記在登錄）；未知箱體覆蓋格在地圖上全為空。
3 可通行：門洞寬對底盤包絡（0.39）與收攏包絡；以 0.39、0.47（靜態 CBF 邊距）膨脹靜態物＋未知箱體＋開啟門扇（向 A 房 90°），0.05 m 格 8 鄰接 BFS：S → 門前 → B 房抽屜前。
4 操作空間：門扇掃掠扇形（鉸鏈原點、半徑 門扇長＋把手外凸、A 房側）內無未知箱體與行人路線；抽屜全拉出（0.22 m）的伸出區＋前方 1.2×1.0 m 操作區內無未知箱體，行人路線距離另報。
5 代理物：在抽屜內部邊界內、抽屜全拉出時整支在櫃口外、上方淨空到上層板、直徑在夾爪指片範圍內。
輸出 第八次進度報告素材/資料/m8_full_scene_check.json；有任一必要項不成立 ⇒ 離開碼 1。
    python3 evaluation/m8_full_scene_check.py
"""
import json
import math
import os
import sys
from collections import deque

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, 'src', 'ammr_bringup', 'scripts'))
import generate_mission_map as GM                       # noqa: E402

REG = json.load(open(os.path.join(HERE, 'experiments/20261111_m8_full_scene.json')))
OUT = os.path.join(WS, '第八次進度報告素材/資料/m8_full_scene_check.json')
ROOM = (-4.0, 5.0, -5.45, 5.45)


def rects_static():
    GM.SDF = os.path.join(WS, REG['world']['path'])
    return [(n, xa, xb, ya, yb) for n, xa, xb, ya, yb in GM.boxes()]


def raster(rects, res, x0=-4.5, x1=5.5, y0=-6.0, y1=6.0, inflate=0.0):
    nx, ny = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    occ = np.zeros((ny, nx), bool)
    xs, ys = x0 + (np.arange(nx) + 0.5) * res, y0 + (np.arange(ny) + 0.5) * res
    for _, xa, xb, ya, yb in rects:
        ix = (xs >= xa - inflate) & (xs <= xb + inflate)
        iy = (ys >= ya - inflate) & (ys <= yb + inflate)
        if inflate > 0:                                                  # 圓角膨脹：角落用距離
            sub = np.ix_(iy, ix)
            X, Y = np.meshgrid(xs[ix], ys[iy])
            dx = np.maximum(np.maximum(xa - X, X - xb), 0.0)
            dy = np.maximum(np.maximum(ya - Y, Y - yb), 0.0)
            occ[sub] |= np.hypot(dx, dy) <= inflate
        else:
            occ[np.ix_(iy, ix)] = True
    return occ, xs, ys, res


def cell(g, p):
    occ, xs, ys, res = g
    return int(np.clip(round((p[1] - ys[0]) / res), 0, len(ys) - 1)), int(np.clip(round((p[0] - xs[0]) / res), 0, len(xs) - 1))


def reach(g, a, b):
    occ = g[0]
    s, t = cell(g, a), cell(g, b)
    if occ[s] or occ[t]:
        return dict(ok=False, why='start_or_goal_occupied', start_occ=bool(occ[s]), goal_occ=bool(occ[t]))
    seen = np.zeros_like(occ)
    q = deque([s])
    seen[s] = True
    while q:
        i, j = q.popleft()
        if (i, j) == t:
            return dict(ok=True)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                k = (i + di, j + dj)
                if 0 <= k[0] < occ.shape[0] and 0 <= k[1] < occ.shape[1] and not seen[k] and not occ[k]:
                    seen[k] = True
                    q.append(k)
    return dict(ok=False, why='disconnected')


def seg_point_dist(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * (b - a))))


def rect_seg_dist(r, a, b, n=60):
    _, xa, xb, ya, yb = r
    pts = [np.array([x, y]) for x in np.linspace(xa, xb, 8) for y in np.linspace(ya, yb, 8)]
    return min(seg_point_dist(p, a, b) for p in pts)


def main():
    D = yaml.safe_load(open(os.path.join(WS, REG['door']['asset'])))
    L, th, W = float(D['leaf']['length']), float(D['leaf']['thickness']), float(D['frame_clear_width'])
    hp = D.get('handle', {})
    stand = float(hp.get('standoff', 0.06)) + float((hp.get('bar') or {}).get('radius', 0.013)) if isinstance(hp, dict) else 0.07
    ob = json.load(open(os.path.join(WS, REG['obstacles']['path'])))
    S = REG['start']['pose']
    Sxy = (S['x'], S['y'])
    dx, dy = REG['drawer']['pose'][:2]
    front_goal = (dx, dy - 0.236 - 0.22 - 0.55)                         # 抽屜全拉出後面板前 0.55 m（只作可達目標點，不是停位答案）
    door_front_A, door_front_B = (0.55, -1.6), (0.55, 0.8)                # 門外觀察區（A 房，開啟門扇掃掠外）與 B 房側門前
    static = rects_static()
    leaf_closed = ('door_leaf_closed', 0.0, L, 0.0, th)
    leaf_open = ('door_leaf_open90', 0.0, th, -L, 0.0)                    # 向 A 房（y<0）開 90°
    unk = [(o['name'], o['x'] - o['r_bound'], o['x'] + o['r_bound'], o['y'] - o['r_bound'], o['y'] + o['r_bound']) for o in ob['static_unknown']]
    res = {}
    # 1 隔牆帶覆蓋
    band = [r for r in static + [leaf_closed] if r[3] < 0.12 - 1e-6 and r[4] > 0.0 + 1e-6]
    xs = np.arange(ROOM[0], ROOM[1], 0.005) + 0.0025
    cov = np.zeros_like(xs, bool)
    for _, xa, xb, ya, yb in band:
        cov |= (xs >= xa) & (xs <= xb)
    gaps, i = [], 0
    while i < len(xs):
        if not cov[i]:
            j = i
            while j < len(xs) and not cov[j]:
                j += 1
            gaps.append([round(xs[i] - 0.0025, 3), round(xs[j - 1] + 0.0025, 3), round((j - i) * 0.005, 3)])
            i = j
        else:
            i += 1
    res['1_partition_gaps_door_closed'] = dict(gaps=gaps, expected='只有門扇末端與門框的縫（門扇長 %.2f、門框淨寬 %.2f）' % (L, W),
                                              ok=all(g[2] <= 0.06 for g in gaps))
    g0 = raster(static + [leaf_closed], 0.02)
    outside, inside_A, inside_B = (-4.4, 0.0), Sxy, front_goal
    res['1_outer_closed'] = dict(outside_to_A=reach(g0, outside, inside_A)['ok'], outside_to_B=reach(g0, outside, inside_B)['ok'])
    res['1_outer_closed']['ok'] = not res['1_outer_closed']['outside_to_A'] and not res['1_outer_closed']['outside_to_B']
    gc = raster(static + [leaf_closed], 0.02, inflate=0.03)              # 門縫 5 cm：以 3 cm 膨脹（任何寬於 6 cm 的縫才會留通）
    go = raster(static + [leaf_open], 0.02)
    res['1_rooms'] = dict(closed_A_to_B=reach(gc, inside_A, inside_B)['ok'], open_A_to_B=reach(go, inside_A, inside_B)['ok'])
    res['1_rooms']['ok'] = (not res['1_rooms']['closed_A_to_B']) and res['1_rooms']['open_A_to_B']
    res['1_rooms']['note'] = '關門不通：格 0.02 m、障礙膨脹 0.03 m，因此 5 cm 門縫被封住（寬於 6 cm 的縫才算通）；離散平面連通核對'
    # 2 未知箱體不在靜態地圖
    import yaml as Y
    mp = os.path.join(WS, REG['static_map']['path'])
    my = Y.safe_load(open(mp))
    b = open(os.path.join(os.path.dirname(mp), my['image']), 'rb').read()
    parts = b.split(b'\n', 3)
    w, h = map(int, parts[1].split())
    img = np.frombuffer(parts[3], np.uint8).reshape(h, w)[::-1]
    r_ = float(my['resolution'])
    ox, oy = my['origin'][:2]
    occ_cells = []
    for o in ob['static_unknown']:
        i0, i1 = int((o['x'] - o['size'][0] / 2 - ox) / r_), int((o['x'] + o['size'][0] / 2 - ox) / r_)
        j0, j1 = int((o['y'] - o['size'][1] / 2 - oy) / r_), int((o['y'] + o['size'][1] / 2 - oy) / r_)
        occ_cells.append(int((img[j0:j1 + 1, i0:i1 + 1] < 128).sum()))
    res['2_unknown_not_in_map'] = dict(occupied_cells_per_box=occ_cells, ok=all(c == 0 for c in occ_cells), map_identical_to_mission_room=REG['static_map']['identical_to_mission_room'])
    # 3 可通行
    env_base = 0.39
    res['3_door_width'] = dict(clear_width=W, open_leaf_intrusion=th, usable=round(W - th, 3), base_envelope_diam=2 * env_base,
                               margin_each_side=round((W - th - 2 * env_base) / 2, 3), ok=(W - th) > 2 * env_base)
    path_res = {}
    for infl in (0.39, 0.47):
        gi = raster(static + [leaf_open] + unk, 0.05, inflate=infl)
        path_res[str(infl)] = dict(S_to_doorA=reach(gi, Sxy, door_front_A), doorA_to_doorB=reach(gi, door_front_A, door_front_B),
                                   doorB_to_drawer=reach(gi, door_front_B, front_goal), S_to_drawer=reach(gi, Sxy, front_goal))
    res['3_passable'] = dict(by_inflation=path_res, front_goal=[round(v, 3) for v in front_goal], ok=all(v['ok'] for pr in path_res.values() for v in pr.values()),
                             note='0.39 與 0.47 兩種膨脹全部納入判定；離散平面連通核對（格 0.05 m），不是全身通過保證')
    # 4 操作空間
    R_sw = L + stand
    sector_pts = [np.array([R * math.cos(a), R * math.sin(a)]) for R in np.linspace(0.05, R_sw, 12) for a in np.linspace(-math.pi / 2, 0.0, 19)]
    def rect_pt_dist(r, p):
        _, xa, xb, ya, yb = r
        return math.hypot(max(xa - p[0], 0, p[0] - xb), max(ya - p[1], 0, p[1] - yb))
    sw = dict(radius=round(R_sw, 3), min_dist_unknown={u[0]: round(min(rect_pt_dist(u, p) for p in sector_pts), 3) for u in unk},
              min_dist_walker_route={wk['name']: round(min(seg_point_dist(p, wk['p0'], wk['p1']) for p in sector_pts) - wk['radius'], 3) for wk in ob['walkers']},
              min_dist_static={r[0]: round(min(rect_pt_dist(r, p) for p in sector_pts), 3) for r in static if r[4] < 0.0 or r[3] < 0.0})
    sw['ok'] = all(v > 0.3 for v in sw['min_dist_unknown'].values()) and all(v > 0.3 for v in sw['min_dist_walker_route'].values())
    res['4_door_sweep'] = sw
    ext = ('drawer_extended', dx - 0.2875, dx + 0.2875, dy - 0.236 - 0.009 - 0.22 - 0.05, dy - 0.225)
    opz = ('drawer_operation_zone', dx - 0.6, dx + 0.6, ext[3] - 1.0, ext[3])
    ws_ = dict(extended_region=[round(v, 3) for v in ext[1:]], operation_zone=[round(v, 3) for v in opz[1:]],
               unknown_overlap={u[0]: bool(not (u[2] < opz[1] or u[1] > opz[2] or u[4] < opz[3] or u[3] > opz[4])) for u in unk},
               static_overlap={r[0]: bool(not (r[2] < opz[1] or r[1] > opz[2] or r[4] < opz[3] or r[3] > opz[4])) for r in static if r[0] != 'drawer_cabinet'},
               walker_route_dist_to_zone={wk['name']: round(rect_seg_dist(opz, wk['p0'], wk['p1']) - wk['radius'], 3) for wk in ob['walkers']})
    ws_['ok'] = not any(ws_['unknown_overlap'].values()) and not any(ws_['static_overlap'].values())
    res['4_drawer_space'] = ws_
    # 5 代理物
    M = yaml.safe_load(open(os.path.join(WS, REG['marker']['asset'])))
    DR = yaml.safe_load(open(os.path.join(WS, REG['drawer']['asset'])))
    c = M['placement']['center']
    ry, r, Lm = M['half_extent_y'], M['half_extent_z'], M['length']        # 六角柱平放：y 向半寬＝外接圓半徑、z 向半高＝邊心距
    sides = [b_ for b_ in DR['drawer']['body'] if b_[0] in ('side_l', 'side_r')]
    x_in = min(abs(b_[1][0]) - b_[2][0] / 2 for b_ in sides)
    back = [b_ for b_ in DR['drawer']['body'] if b_[0] == 'back'][0]
    y_back_in = back[1][1] - back[2][1] / 2
    y_front_in = M['drawer_interior']['front_inner_y']
    shelf = [b_ for b_ in DR['cabinet'] if b_[0] == 'shelf_above'][0]
    z_shelf = shelf[1][2] - shelf[2][2] / 2
    cab_front = -0.225
    travel = float(DR['drawer']['joint']['upper'])
    mk = dict(x_ok=abs(c[0]) + Lm / 2 <= x_in, y_ok=(c[1] - ry >= y_front_in) and (c[1] + ry <= y_back_in), z_on_bottom=round(c[2] - r - M['drawer_interior']['bottom_top_z'], 4),
              outside_cabinet_when_open=(c[1] + ry - travel) < cab_front, clearance_to_shelf_above=round(z_shelf - (c[2] + r), 4),
              gripper_contact_joint=M['gripper_basis']['contact_finger_joint_m'], gripper_open_margin=M['gripper_basis']['open_margin_each_side_m'])
    mk['ok'] = mk['x_ok'] and mk['y_ok'] and mk['outside_cabinet_when_open'] and 0 <= mk['z_on_bottom'] <= 0.002 and mk['gripper_contact_joint'] > 0 and mk['gripper_open_margin'] > 0.005
    res['5_marker'] = mk
    walkers = {wk['name']: dict(speed=wk['speed_mps'], d_to_S=round(seg_point_dist(Sxy, wk['p0'], wk['p1']) - wk['radius'], 3),
                                d_to_door_front_A=round(seg_point_dist(door_front_A, wk['p0'], wk['p1']) - wk['radius'], 3)) for wk in ob['walkers']}
    res['traffic'] = walkers
    must = ['1_partition_gaps_door_closed', '1_outer_closed', '1_rooms', '2_unknown_not_in_map', '3_door_width', '3_passable', '4_door_sweep', '4_drawer_space', '5_marker']
    res['verdict'] = dict(ok=all(res[k]['ok'] for k in must), failed=[k for k in must if not res[k]['ok']])
    res['registration'] = REG['name']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
    for k in must + ['traffic']:
        print(k, json.dumps(res[k], ensure_ascii=False, default=float)[:420], flush=True)
    print('判定', res['verdict'], flush=True)
    return 0 if res['verdict']['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
