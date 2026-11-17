#!/usr/bin/env python3
"""茶几圓柱體隨機放：離線可達性地圖（Claude 自行裁定 20261117_2315 第一步；不動控制、不跑 Isaac）。
桌面（茶几中心 (1.35, 2.3)、0.6×1.1 m、高 0.45 m）上離邊 ≥0.05 m 的 5 cm 網格，每個圓柱位置：
  候選底盤位姿＝西／南／北三側（東側到沙發只剩約 0.45 m，不列），底盤中心離桌邊 d∈D_SIDE、沿邊橫移 lat∈LAT、朝向＝正對桌邊；
  抓取 TCP＝瓶子線側向水平抓取定義（TCP z＝水平接近方向＝底盤到圓柱、TCP y＝水平開合、TCP 在圓柱中段 z＝0.55 m），預抓＝沿接近軸後退 0.06 m；
  可達＝抓取與預抓兩個位姿 IK 殘差 ≤1e-4、在 LITE6_SAFE 內（留 0.05 rad）、σ_min ≥ 0.02；
  淨空（粗核）：底盤圓（半徑 0.30 m）不碰家具箱、手臂各連桿原點在桌面投影內時高度 ≥ 0.50 m（抓取構型）。
輸出：每格最佳側、最佳候選、可達與否；整體可達比例；PNG 圖＋JSON。
    python3 evaluation/cyl_reach_map.py OUT_PREFIX"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import door_motm_grasp_ctrl as G                                     # noqa: E402
import door_fast_kin as DFK                                           # noqa: E402

K, XO = G.K, G.XO
LIM = XO.LITE6_SAFE
OUT = sys.argv[1]
scene = json.load(open(os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json')))
FURN = {f['name']: f for f in scene['furniture']}
TAB = FURN['coffee_table']
TX, TY, _ = TAB['c']; SX, SY, SZ = TAB['s']
TOP = SZ
X0, X1, Y0, Y1 = TX - SX / 2, TX + SX / 2, TY - SY / 2, TY + SY / 2
H_GRASP, PRE = TOP + 0.10, 0.06
RB = 0.30
D_SIDE = (0.33, 0.38, 0.43, 0.48)
LAT = tuple(np.round(np.arange(-0.30, 0.301, 0.05), 3))
SEEDS = [np.array([-0.0128, -0.0519, 0.086, 0.0116, -1.6249, -0.0018]), np.array([-0.354, -0.088, 1.435, -1.421, -0.358, -0.157]),
         np.array([0.0, 0.3, 1.2, 0.0, -0.9, 0.0]), np.array([0.0, 0.6, 1.6, 0.0, -1.0, 0.0])]
LINKS = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link_eef', 'link_tcp']
BOXES = [(np.array(o['c']), np.array(o['s'])) for o in scene['walls'] + scene['furniture']]


def grasp_T(c_xy, a, flip=False):
    sg = -1.0 if flip else 1.0                                        # 兩指對稱：開合軸正反兩向是同一種抓法（瓶子線 grasp_pose 的 flip）
    R = np.zeros((3, 3)); R[:, 2] = [a[0], a[1], 0.0]; R[:, 1] = [-sg * a[1], sg * a[0], 0.0]; R[:, 0] = np.cross(R[:, 1], R[:, 2])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [c_xy[0], c_xy[1], H_GRASP]
    return T


def base_free(b):
    for c, s in BOXES:
        dx = max(abs(b[0] - c[0]) - s[0] / 2, 0.0); dy = max(abs(b[1] - c[1]) - s[1] / 2, 0.0)
        if math.hypot(dx, dy) < RB:
            return False
    return True


def ik(T, b):
    best = None
    for sd in SEEDS:
        q, r, sm = DFK.ik_closure_fc(T, b, sd)
        q = (np.asarray(q, float) + math.pi) % (2 * math.pi) - math.pi   # 角度正規化到 (−π, π]
        if r <= 1e-4 and np.all(q >= LIM.lower + 0.05) and np.all(q <= LIM.upper - 0.05) and sm >= 0.02:
            if best is None or sm > best[2]:
                best = (q, r, sm)
    return best


def arm_over_table_ok(b, q):
    x9 = np.r_[b, q]
    for n in LINKS:
        p = K.fk(x9, n)[:3, 3]
        if X0 - 0.02 <= p[0] <= X1 + 0.02 and Y0 - 0.02 <= p[1] <= Y1 + 0.02 and p[2] < TOP + 0.05:
            return False
    return True


def candidates(c):
    out = []
    for side in ('west', 'south', 'north'):
        for d in D_SIDE:
            for lat in LAT:
                if side == 'west':
                    b = np.array([X0 - d, c[1] + lat, 0.0])
                elif side == 'south':
                    b = np.array([c[0] + lat, Y0 - d, math.pi / 2])
                else:
                    b = np.array([c[0] + lat, Y1 + d, -math.pi / 2])
                out.append((side, d, lat, b))
    return out


def main():
    xs = np.round(np.arange(X0 + 0.05, X1 - 0.05 + 1e-9, 0.05), 3)
    ys = np.round(np.arange(Y0 + 0.05, Y1 - 0.05 + 1e-9, 0.05), 3)
    cells = []
    for cx in xs:
        for cy in ys:
            c = np.array([cx, cy]); best = None; n_try = 0
            for side, d, lat, b in candidates(c):
                if not base_free(b):
                    continue
                v = c - b[:2]; dist = float(np.linalg.norm(v))
                if dist < 0.25 or dist > 0.80:
                    continue
                a = v / dist
                n_try += 1
                g, fl = None, False
                for fl in (False, True):
                    g = ik(grasp_T(c, a, fl), b)
                    if g is not None and arm_over_table_ok(b, g[0]):
                        break
                    g = None
                if g is None:
                    continue
                Tp = grasp_T(c, a, fl); Tp[:3, 3] -= PRE * np.r_[a, 0.0]
                p = ik(Tp, b)
                if p is None:
                    continue
                score = float(g[2])
                if best is None or score > best['sigma_min']:
                    best = dict(side=side, flip=bool(fl), d=float(d), lat=float(lat), base=[round(float(x), 4) for x in b], sigma_min=round(score, 4), q=[round(float(x), 4) for x in g[0]])
            cells.append(dict(c=[float(cx), float(cy)], reachable=best is not None, best=best, n_try=n_try))
            print('%.2f %.2f %s %s' % (cx, cy, 'OK ' if best else '-- ', '' if best is None else '%s d=%.2f lat=%.2f σ=%.3f' % (best['side'], best['d'], best['lat'], best['sigma_min'])), flush=True)
    n_ok = sum(1 for c_ in cells if c_['reachable'])
    summ = dict(table=dict(center=[TX, TY], size=[SX, SY], top=TOP), grid=dict(xs=xs.tolist(), ys=ys.tolist(), step=0.05, margin=0.05),
                grasp=dict(h=H_GRASP, pre=PRE, def_='TCP z＝水平接近方向、TCP y＝水平開合（瓶子線 grasp_pose 定義推廣）'),
                base=dict(sides=['west', 'south', 'north'], d=list(D_SIDE), lat=list(LAT), rb=RB, note='東側到沙發約 0.45 m，不列'),
                reach_rule='抓取＋預抓 IK 殘差 ≤1e-4、LITE6_SAFE 內留 0.05 rad、σ_min ≥ 0.02；底盤圓不碰家具；手臂連桿原點在桌面投影內高度 ≥ 桌面＋0.05',
                n_cells=len(cells), n_reachable=n_ok, frac_reachable=round(n_ok / max(len(cells), 1), 4), cells=cells)
    json.dump(summ, open(OUT + '.json', 'w'), ensure_ascii=False, indent=1)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.add_patch(plt.Rectangle((X0, Y0), SX, SY, fill=False, lw=2))
    for c, s in BOXES:
        if abs(c[0] - TX) < 2.5 and abs(c[1] - TY) < 2.5:
            ax.add_patch(plt.Rectangle((c[0] - s[0] / 2, c[1] - s[1] / 2), s[0], s[1], color='0.85', zorder=0))
    col = {'west': 'tab:blue', 'south': 'tab:orange', 'north': 'tab:green'}
    for c_ in cells:
        if c_['reachable']:
            ax.plot(*c_['c'], 'o', color=col[c_['best']['side']], ms=6)
        else:
            ax.plot(*c_['c'], 'x', color='tab:red', ms=6)
    ax.set_xlim(X0 - 0.8, X1 + 0.6); ax.set_ylim(Y0 - 0.8, Y1 + 0.8); ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title('reachable %d/%d (o=best side: blue W, orange S, green N; x=unreachable)' % (n_ok, len(cells)), fontsize=9)
    fig.savefig(OUT + '.png', dpi=110, bbox_inches='tight')
    print('reachable', n_ok, '/', len(cells))


if __name__ == '__main__':
    main()
