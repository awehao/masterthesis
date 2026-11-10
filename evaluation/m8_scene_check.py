#!/usr/bin/env python3
"""第八次第一包（場景＋掃描驅動導航）的離線核對，只讀場景檔，不跑模擬（Codex 20261110_180004）。
1 門前觀察區：只由門的先驗幾何（door_local_v3：鉸鏈在原點、門扇長 0.95、向 −y 拉開、門洞 x∈[0, 1.0]、把手外凸）＋實際底盤半徑 0.39 算，不用舊 D1 停位。
2 連通性：用實際控制器的靜態避障邊距（gmpc static_cbf_safe_margin 0.47；比底盤半徑 0.39 保守）把牆、已知家具、關閉的門扇、未知箱體膨脹，
  0.05 m 格點 8 鄰接 BFS 核 S → 觀察區是否有路；另報 0.39（只看底盤半徑）的結果。舊生成器的 0.35＋0.10 假設不沿用。
3 未知箱體不在靜態地圖：mission_room.pgm 在每個箱體覆蓋格須全部為空（非佔據）。
4 行人：軌跡對觀察區與起點的最近距離；軌跡端點停留位置對觀察區的距離（行人不該站在觀察區上）。
輸出 第八次進度報告素材/資料/m8_scene_check.json。
    python3 evaluation/m8_scene_check.py
"""
import json
import math
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import mission_room_obstacles as MO  # noqa: E402

DOOR = os.path.join(WS, 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml')
OBS = os.path.join(HERE, 'results', 'mission', 'mission_obstacles_seed1.json')
POSES = os.path.join(HERE, 'results', 'mission', 'mission_room_poses.json')
MAP = os.path.join(WS, 'src', 'ammr_bringup', 'maps', 'mission_room.yaml')
OUT = os.path.join(WS, '第八次進度報告素材', '資料', 'm8_scene_check.json')
R_BASE, M_STATIC, RES = 0.39, 0.47, 0.05
OBS_EXTRA = 0.15                                        # 觀察區離門扇掃掠外緣的額外餘裕


def door_prior():
    import yaml
    d = yaml.safe_load(open(DOOR))
    return dict(W=float(d['frame_clear_width']), L=float(d['leaf']['length']), t=float(d['leaf']['thickness']),
                h_out=float(d['handle']['standoff']) + float(d['handle']['bar']['radius']))


def obs_zone(dp):
    """門洞中心前方、門扇掃掠（半徑＝門扇長＋把手外凸）之外再留底盤半徑與餘裕；車頭朝門（+y）。"""
    reach = dp['L'] + dp['h_out']
    y = -(reach + R_BASE + OBS_EXTRA)
    return dict(x=dp['W'] / 2.0, y=round(y, 4), yaw=math.pi / 2, reach=reach, note='由門洞寬、門扇長、把手外凸、底盤半徑與餘裕 0.15 算；不讀 D1')


def poly_box(x, y, yaw_deg, sx, sy):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([(x + c * px - s * py, y + s * px + c * py) for px, py in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2))])


def seg_pt_dist(P, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def poly_dist(P, V):
    """點到凸多邊形（含內部＝0）的距離。"""
    d = np.min([seg_pt_dist(P, V[i], V[(i + 1) % len(V)]) for i in range(len(V))], axis=0)
    inside = np.ones(len(P), bool)
    for i in range(len(V)):
        e = V[(i + 1) % len(V)] - V[i]
        inside &= (e[0] * (P[:, 1] - V[i][1]) - e[1] * (P[:, 0] - V[i][0])) >= 0
    d[inside] = 0.0
    return d


def obstacles(obs, dp):
    out = []
    for r in MO.known_boxes():
        out.append(('known', np.array([(r[0], r[2]), (r[1], r[2]), (r[1], r[3]), (r[0], r[3])])))
    x0 = 0.0
    out.append(('door_leaf_closed', np.array([(x0, 0.0), (x0 + dp['L'], 0.0), (x0 + dp['L'], dp['t']), (x0, dp['t'])])))
    for u in obs['static_unknown']:
        out.append(('unknown:' + u['name'], poly_box(u['x'], u['y'], u['yaw_deg'], u['size'][0], u['size'][1])))
    return out


def connectivity(obst, start, goal, margin):
    x0, x1, y0, y1 = MO.ROOM
    xs, ys = np.arange(x0, x1 + 1e-9, RES), np.arange(y0, y1 + 1e-9, RES)
    X, Y = np.meshgrid(xs, ys)
    P = np.c_[X.ravel(), Y.ravel()]
    D = np.full(len(P), np.inf)
    for _, V in obst:
        D = np.minimum(D, poly_dist(P, V))
    D = np.minimum(D, np.minimum.reduce([P[:, 0] - x0, x1 - P[:, 0], P[:, 1] - y0, y1 - P[:, 1]]))
    F = (D >= margin).reshape(X.shape)
    Dg = D.reshape(X.shape)

    def idx(p):
        return int(round((p[1] - y0) / RES)), int(round((p[0] - x0) / RES))
    s, g = idx(start), idx(goal)
    out = dict(margin=margin, start_clear=round(float(Dg[s]), 3), goal_clear=round(float(Dg[g]), 3), start_free=bool(F[s]), goal_free=bool(F[g]))
    if not (F[s] and F[g]):
        out['connected'] = False
        return out
    prev = {s: None}
    q = deque([s])
    while q:
        c = q.popleft()
        if c == g:
            break
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                n = (c[0] + di, c[1] + dj)
                if (di or dj) and 0 <= n[0] < F.shape[0] and 0 <= n[1] < F.shape[1] and F[n] and n not in prev:
                    prev[n] = c
                    q.append(n)
    out['connected'] = g in prev
    if out['connected']:
        path, c = [], g
        while c is not None:
            path.append(c)
            c = prev[c]
        Pth = np.array([(xs[j], ys[i]) for i, j in path[::-1]])
        out['grid_path_len_m'] = round(float(np.sum(np.linalg.norm(np.diff(Pth, axis=0), axis=1))), 2)
        out['grid_path_min_clear'] = round(float(min(Dg[i, j] for i, j in path)), 3)
    return out


def map_check(obs):
    import yaml
    from PIL import Image
    m = yaml.safe_load(open(MAP))
    img = np.asarray(Image.open(os.path.join(os.path.dirname(MAP), m['image'])), float) / 255.0
    occ = (1.0 - img) > float(m['occupied_thresh'])                       # negate 0：暗＝佔據
    H = img.shape[0]
    res, ox, oy = float(m['resolution']), float(m['origin'][0]), float(m['origin'][1])
    rows = []
    for u in obs['static_unknown']:
        V = poly_box(u['x'], u['y'], u['yaw_deg'], u['size'][0], u['size'][1])
        xs = np.arange(V[:, 0].min(), V[:, 0].max(), res / 2)
        ys = np.arange(V[:, 1].min(), V[:, 1].max(), res / 2)
        X, Y = np.meshgrid(xs, ys)
        P = np.c_[X.ravel(), Y.ravel()]
        P = P[poly_dist(P, V) == 0.0]
        ix = np.floor((P[:, 0] - ox) / res).astype(int)
        iy = H - 1 - np.floor((P[:, 1] - oy) / res).astype(int)
        rows.append(dict(name=u['name'], n_cells=int(len(P)), n_occupied=int(occ[iy, ix].sum())))
    return rows


def main():
    dp = door_prior()
    z = obs_zone(dp)
    obs = json.load(open(OBS))
    S = json.load(open(POSES))['poses']['S']
    obst = obstacles(obs, dp)
    goal = (z['x'], z['y'])
    start = (S['x'], S['y'])
    conn = [connectivity(obst, start, goal, m) for m in (M_STATIC, R_BASE)]
    walkers = []
    for w in obs['walkers']:
        Pz = np.array([goal, start])
        dz = seg_pt_dist(Pz, w['p0'], w['p1'])
        walkers.append(dict(name=w['name'], d_line_to_obs_zone=round(float(dz[0]), 3), d_line_to_start=round(float(dz[1]), 3),
                            d_endpoints_to_obs_zone=[round(math.hypot(p[0] - goal[0], p[1] - goal[1]), 3) for p in (w['p0'], w['p1'])],
                            clear_to_zone_minus_radii=round(float(dz[0]) - w['radius'] - R_BASE, 3)))
    out = dict(door_prior=dp, obs_zone=z, start=dict(start=S, note='研究目標情境起點 S（位姿檔）；mission_rt 系列實跑起點是 (−0.6, −2.8)，不同'),
               connectivity=conn, map_unknown_absent=map_check(obs), walkers=walkers,
               obstacles_file=os.path.relpath(OBS, WS), r_base=R_BASE, static_margin=M_STATIC)
    out['ok'] = bool(conn[0]['connected'] and all(r['n_occupied'] == 0 for r in out['map_unknown_absent']) and all(w['clear_to_zone_minus_radii'] > 0.0 for w in walkers))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(dict(ok=out['ok'], obs_zone=z, connectivity=conn, map=out['map_unknown_absent'], walkers=walkers), ensure_ascii=False, indent=0))


if __name__ == '__main__':
    main()
