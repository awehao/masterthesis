#!/usr/bin/env python3
"""住宅格局場景 m8_house_r1 的離線幾何核對＋平面圖（不跑模擬）。全部由登錄檔 experiments/20261112_m8_house_scene.json 讀幾何。
核對（離散平面格 0.02 m 連通；不是全身通過保證）：
  1 外部封閉；2 關門時臥室↔客廳不通（障礙膨脹 0.03 m 封住 5 cm 門縫）、開門時通；3 客廳↔取物區經內牆開口相通；
  4 未知物在靜態地圖上佔 0 格；5 門洞／開口淨寬對底盤包絡；6 膨脹 0.39／0.47（含未知物、開門門扇）下 起點→門前→門後→開口→抽屜前 與返回皆可達；
  7 門扇掃掠區對家具／未知物淨空（行人路線距離另列）；8 抽屜操作區對家具／未知物淨空（行人另列）；9 代理物在抽屜內、拉出後在櫃口外。
平面圖：簡報方向（左→右＝+y、上→下＝+x），與簡報第 3 頁並排。    python3 evaluation/m8_house_scene_check.py"""
import json
import math
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
REG = json.load(open(os.path.join(HERE, 'experiments', '20261112_m8_house_scene.json')))
OB = json.load(open(os.path.join(WS, REG['obstacles']['path'])))
OUT_J = os.path.join(WS, '第八次進度報告素材/資料/m8_house_scene_check.json')
OUT_D = os.path.join(WS, '第八次進度報告素材/圖表/m8_house_scene')
RES, X0, Y0, X1, Y1 = 0.02, -3.8, -5.0, 3.8, 9.6
R_BASE = 0.39
print('#EARLY', REG['name'], flush=True)


def rect(o):
    return (o['c'][0] - o['s'][0] / 2, o['c'][0] + o['s'][0] / 2, o['c'][1] - o['s'][1] / 2, o['c'][1] + o['s'][1] / 2)


KNOWN = [(o['name'],) + rect(o) for o in REG['walls'] + REG['furniture']] + [('door_wall_hinge', -1.5, 0.0, 0.0, 0.12), ('door_wall_latch', 1.0, 2.5, 0.0, 0.12), ('drawer_cabinet', *REG['drawer']['footprint'])]
LEAF_CLOSED = ('leaf_closed', 0.0, 0.95, -0.04, 0.0)
LEAF_OPEN = ('leaf_open', -0.04, 0.0, -0.95, 0.0)
UNK = []
for o in OB['static_unknown']:
    r = 0.5 * math.hypot(o['size'][0], o['size'][1])                # 轉角方盒以外接圓保守代表
    UNK.append((o['name'], o['x'] - r, o['x'] + r, o['y'] - r, o['y'] + r))


def raster(rects, infl=0.0):
    W, H = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))
    g = np.zeros((W, H), bool)
    for _, xa, xb, ya, yb in rects:
        i0, i1 = int(np.floor((xa - infl - X0) / RES)), int(np.ceil((xb + infl - X0) / RES))
        j0, j1 = int(np.floor((ya - infl - Y0) / RES)), int(np.ceil((yb + infl - Y0) / RES))
        g[max(i0, 0):min(i1, W), max(j0, 0):min(j1, H)] = True
    return g


def reach(g, a, b):
    ia, ib = (int((a[0] - X0) / RES), int((a[1] - Y0) / RES)), (int((b[0] - X0) / RES), int((b[1] - Y0) / RES))
    if g[ia] or g[ib]:
        return dict(ok=False, why='endpoint_blocked', a_blocked=bool(g[ia]), b_blocked=bool(g[ib]))
    seen = np.zeros_like(g)
    seen[ia] = True
    q = deque([ia])
    while q:
        i, j = q.popleft()
        if (i, j) == ib:
            return dict(ok=True)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = (i + di, j + dj)
            if 0 <= k[0] < g.shape[0] and 0 <= k[1] < g.shape[1] and not seen[k] and not g[k]:
                seen[k] = True
                q.append(k)
    return dict(ok=False, why='disconnected')


def rect_dist(r, p):
    dx = max(r[1] - p[0], 0.0, p[0] - r[2])
    dy = max(r[3] - p[1], 0.0, p[1] - r[4])
    return math.hypot(dx, dy)


def seg_rect_dist(z, p0, p1):
    return min(rect_dist(('z',) + tuple(z), (p0[0] + u * (p1[0] - p0[0]), p0[1] + u * (p1[1] - p0[1]))) for u in np.linspace(0, 1, 101))


def main():
    S = REG['start']['pose']
    P = dict(S=(S['x'], S['y']), outside=(X0 + 0.1, Y0 + 0.1), door_front=(0.6, -1.6), door_back=(0.5, 1.25), living=(0.0, 3.0), right=(0.0, 6.5),
             drawer_front=(REG['drawer']['footprint'][1] + REG['drawer']['travel'] + 0.80, REG['drawer']['pose'][1]))
    res = dict(registration=REG['name'], points=P)
    gc = raster(KNOWN + [LEAF_CLOSED], infl=0.03)
    go = raster(KNOWN + [LEAF_OPEN])
    res['1_outer_closed'] = dict(outside_to_bedroom=reach(gc, P['outside'], P['S'])['ok'], outside_to_living=reach(gc, P['outside'], P['living'])['ok'], outside_to_right=reach(gc, P['outside'], P['right'])['ok'])
    res['1_outer_closed']['ok'] = not any(v for k, v in res['1_outer_closed'].items())
    res['2_door'] = dict(closed_bedroom_to_living=reach(gc, P['S'], P['living'])['ok'], open_bedroom_to_living=reach(go, P['S'], P['living'])['ok'], note='關門核對：格 0.02、膨脹 0.03 m 封住門扇末端 5 cm 門縫')
    res['2_door']['ok'] = (not res['2_door']['closed_bedroom_to_living']) and res['2_door']['open_bedroom_to_living']
    res['3_living_right'] = dict(connected=reach(gc, P['living'], P['right'])['ok'])
    res['3_living_right']['ok'] = res['3_living_right']['connected']
    # 4 未知物不入地圖
    pgm = open(os.path.join(WS, REG['static_map']['pgm']), 'rb').read().split(b'\n', 3)
    w, h = map(int, pgm[1].split())
    img = np.frombuffer(pgm[3], np.uint8).reshape(h, w)[::-1]
    mo = yaml_origin = None
    import yaml
    my = yaml.safe_load(open(os.path.join(WS, REG['static_map']['yaml'])))
    ox, oy, mr = my['origin'][0], my['origin'][1], my['resolution']
    occ = []
    for o in OB['static_unknown']:
        i, j = int((o['x'] - ox) / mr), int((o['y'] - oy) / mr)
        occ.append(int((img[j - 2:j + 3, i - 2:i + 3] < 128).sum()))
    res['4_unknown_not_in_map'] = dict(occupied_cells_near_each=occ, ok=all(c == 0 for c in occ))
    # 5 寬度
    res['5_widths'] = dict(door_clear=1.0, door_usable_with_open_leaf=0.96, opening2=REG['dims']['opening2_x'][1] - REG['dims']['opening2_x'][0], base_envelope=2 * R_BASE)
    res['5_widths']['ok'] = res['5_widths']['door_usable_with_open_leaf'] > 2 * R_BASE and res['5_widths']['opening2'] > 2 * R_BASE
    # 6 可通行
    legs = [('S', 'door_front'), ('door_front', 'door_back'), ('door_back', 'right'), ('right', 'drawer_front'), ('drawer_front', 'S')]
    pas = {}
    for infl in (0.39, 0.47):
        gi = raster(KNOWN + [LEAF_OPEN] + UNK, infl=infl)
        pas[str(infl)] = {f'{a}_to_{b}': reach(gi, P[a], P[b]) for a, b in legs}
    res['6_passable'] = dict(by_inflation=pas, ok=all(v['ok'] for d in pas.values() for v in d.values()), note='兩種膨脹全部納入；離散平面連通，不是全身通過保證；未知物以外接圓保守代表')
    # 7 門扇掃掠（鉸鏈原點、半徑 0.95＋0.05，掃向臥室 x∈[0,1]、y∈[−1,0]）
    sweep = (0.0, 1.0, -1.0, 0.0)
    stat = {n: round(rect_dist((n, xa, xb, ya, yb), (0.5, -0.5)) - 0.5 * math.sqrt(2), 3) for n, xa, xb, ya, yb in KNOWN if n not in ('door_wall_hinge', 'door_wall_latch') and not n.startswith('partition_door')}
    near = {k: v for k, v in sorted(stat.items(), key=lambda z: z[1])[:3]}
    unk = {n: round(rect_dist((n, xa, xb, ya, yb), (0.5, -0.5)) - 0.5 * math.sqrt(2), 3) for n, xa, xb, ya, yb in UNK}
    wk = {w_['name']: round(seg_rect_dist(sweep, w_['p0'], w_['p1']) - w_['radius'], 3) for w_ in OB['walkers']}
    res['7_door_sweep'] = dict(zone=sweep, nearest_static=near, unknown=unk, walker_route=wk, ok=min(list(stat.values()) + list(unk.values())) > 0.0)
    # 8 抽屜操作區
    fp = REG['drawer']['footprint']
    zone = (fp[1], fp[1] + REG['drawer']['travel'] + 1.20, REG['drawer']['pose'][1] - 0.60, REG['drawer']['pose'][1] + 0.60)

    def overlap(r):
        return not (r[2] <= zone[0] or r[1] >= zone[1] or r[4] <= zone[2] or r[3] >= zone[3])
    so = {n: overlap((n, xa, xb, ya, yb)) for n, xa, xb, ya, yb in KNOWN if n != 'drawer_cabinet'}
    uo = {n: overlap((n, xa, xb, ya, yb)) for n, xa, xb, ya, yb in UNK}
    wk2 = {w_['name']: round(seg_rect_dist(zone, w_['p0'], w_['p1']) - w_['radius'], 3) for w_ in OB['walkers']}
    res['8_drawer_zone'] = dict(zone=[round(v, 3) for v in zone], static_overlap=[k for k, v in so.items() if v], unknown_overlap=[k for k, v in uo.items() if v], walker_route_dist=wk2, ok=not any(so.values()) and not any(uo.values()))
    # 9 代理物
    mk = REG['marker']
    half_w = 0.5 * 0.03002
    inside = fp[0] < mk['center_world'][0] - half_w and mk['center_world'][0] + half_w < fp[1] and fp[2] < mk['center_world'][1] - mk['length'] / 2 and mk['center_world'][1] + mk['length'] / 2 < fp[3]
    out_when_open = (mk['center_world'][0] - half_w + REG['drawer']['travel']) > fp[1] - 0.011
    res['9_marker'] = dict(inside_cabinet_closed=bool(inside), outside_cabinet_when_open=bool(out_when_open), center=mk['center_world'], ok=bool(inside and out_when_open))
    must = [k for k in res if k[0].isdigit()]
    res['verdict'] = dict(ok=all(res[k]['ok'] for k in must), failed=[k for k in must if not res[k]['ok']], scope='場景幾何與離散平面連通；不代表任務完成或全身通過')
    os.makedirs(os.path.dirname(OUT_J), exist_ok=True)
    json.dump(res, open(OUT_J, 'w'), ensure_ascii=False, indent=1)
    for k in must:
        print(k, json.dumps({kk: vv for kk, vv in res[k].items() if kk != 'by_inflation'}, ensure_ascii=False)[:260])
    print('判定', res['verdict'])
    plot(P)
    return 0 if res['verdict']['ok'] else 1


def plot(P):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Rectangle, Circle, Arc
    FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
    font_manager.fontManager.addfont(FONT)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
    INK, MUTED = '#1d2229', '#5a6270'
    os.makedirs(OUT_D, exist_ok=True)

    def draw(ax):
        # 圖座標：橫＝世界 y、縱＝世界 x（向下為正）
        for o in REG['walls']:
            xa, xb, ya, yb = rect(o)
            ax.add_patch(Rectangle((ya, xa), yb - ya, xb - xa, fc='#3b4250', ec='none', zorder=3))
        for n, xa, xb, ya, yb in (('h', -1.5, 0.0, 0.0, 0.12), ('l', 1.0, 2.5, 0.0, 0.12)):
            ax.add_patch(Rectangle((ya, xa), yb - ya, xb - xa, fc='#3b4250', ec='none', zorder=3))
        for o in REG['furniture']:
            xa, xb, ya, yb = rect(o)
            ax.add_patch(Rectangle((ya, xa), yb - ya, xb - xa, fc=o['rgb'], ec='#30343c', lw=0.8, zorder=2))
            if o['name'] != 'tv':
                ax.text((ya + yb) / 2, (xa + xb) / 2, o['zh'], ha='center', va='center', fontsize=8, color='white' if sum(o['rgb']) < 1.2 else INK, zorder=4)
        fp = REG['drawer']['footprint']
        ax.add_patch(Rectangle((fp[2], fp[0]), fp[3] - fp[2], fp[1] - fp[0], fc='#1f5fbf', ec=INK, lw=1.0, zorder=4))
        ax.annotate('抽屜櫃（可拉出）\n內有粗馬克筆代理物', xy=((fp[2] + fp[3]) / 2, fp[1]), xytext=(6.2, -1.3), fontsize=9, color=INK, ha='center', arrowprops=dict(arrowstyle='-', color='#1f5fbf', lw=1.2), zorder=6,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#1f5fbf', lw=1.2))
        ax.add_patch(Rectangle((fp[2] + 0.05, fp[1]), fp[3] - fp[2] - 0.1, REG['drawer']['travel'], fc='none', ec='#1f5fbf', ls=(0, (3, 2)), lw=1.0, zorder=4))
        ax.plot([-0.04, 0.0], [0.0, 0.95], color='#c0392b', lw=3, zorder=5)                      # 關閉的門扇（沿 x）
        ax.add_patch(Arc((0.0, 0.0), 1.9, 1.9, theta1=90, theta2=180, color='#c0392b', ls=(0, (3, 2)), lw=1.0, zorder=5))
        ax.annotate('門（關閉、可操作）\n向臥室側拉開', xy=(-0.02, 0.5), xytext=(-1.2, -1.75), fontsize=9, color=INK, ha='center', arrowprops=dict(arrowstyle='-', color='#c0392b', lw=1.2), zorder=6,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#c0392b', lw=1.2))
        for o in OB['static_unknown']:
            ax.add_patch(Rectangle((o['y'] - o['size'][1] / 2, o['x'] - o['size'][0] / 2), o['size'][1], o['size'][0], angle=-o['yaw_deg'], rotation_point='center', fc='#d9a15a', ec='#7a4a12', lw=0.8, zorder=4))
        for w_ in OB['walkers']:
            ax.plot([w_['p0'][1], w_['p1'][1]], [w_['p0'][0], w_['p1'][0]], color='#c0392b', ls=(0, (5, 4)), lw=1.3, alpha=0.8, zorder=4)
            ax.add_patch(Circle((w_['p0'][1], w_['p0'][0]), w_['radius'], fc='#e9967a', ec='#8c2f1f', lw=0.8, zorder=5))
        s = REG['start']['pose']
        ax.add_patch(Circle((s['y'], s['x']), 0.39, fc='#3aa0e8', ec='#16507c', lw=1.0, zorder=5))
        ax.text(s['y'], s['x'] + 0.62, '起點 S', ha='center', va='center', fontsize=9, color=INK, zorder=6)
        ax.plot([s['y'], -1.4, 1.25, 3.2, 5.06, 6.4, 7.0], [s['x'], 0.5, 0.5, 0.2, 0.0, -1.0, P['drawer_front'][0]], color='#1e8449', ls=(0, (2, 3)), lw=1.6, alpha=0.9, zorder=4)
        for txt, yy, xx in (('臥室（起始房間）', -3.25, -1.72), ('中央客廳／通行區', 2.55, -0.6), ('右側取物區', 6.95, 2.72)):
            ax.text(yy, xx, txt, ha='center', va='center', fontsize=11, color=MUTED, fontweight='bold', zorder=1)
        ax.text(5.06, 0.0, '開口', ha='center', va='center', fontsize=8, color=MUTED, rotation=90, zorder=6)
        ax.set_xlim(-4.7, 9.3)
        ax.set_ylim(3.5, -3.5)
        ax.set_aspect('equal')
        ax.set_xlabel('世界 y（m）', fontsize=9, color=MUTED)
        ax.set_ylabel('世界 x（m）', fontsize=9, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#c8ccd2')
    fig, ax = plt.subplots(figsize=(12.6, 6.6), dpi=130)
    draw(ax)
    ax.set_title('住宅格局場景 m8_house_r1 平面配置（工程設計尺寸；綠虛線是任務意圖，不是控制軌跡）', fontsize=11, color=INK, loc='left')
    ax.plot([], [], color='#c0392b', ls=(0, (5, 4)), label='行人往返路線（0.20 m/s，不進地圖）')
    ax.add_patch(Rectangle((0, 0), 0, 0, fc='#d9a15a', ec='#7a4a12', label='未知箱／椅（不進地圖）'))
    ax.legend(loc='lower left', fontsize=8, frameon=True, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_D, 'm8_house_plan.png'))
    plt.close(fig)
    ref = '/tmp/claude-1000/-home-howardchen-masterthesis/2a47f71f-1a91-4ff0-9dbf-1892d8540042/scratchpad/goalpage-03.png'
    if os.path.exists(ref):
        im = plt.imread(ref)
        crop = im[395:790, 110:940]
        fig, axs = plt.subplots(1, 2, figsize=(16, 5.6), dpi=120, gridspec_kw=dict(width_ratios=[1.0, 1.25]))
        axs[0].imshow(crop)
        axs[0].axis('off')
        axs[0].set_title('參考：第六次進度報告第 3 頁「研究目標」', fontsize=11, color=INK, loc='left')
        draw(axs[1])
        axs[1].set_title('本版場景 m8_house_r1（平面配置）', fontsize=11, color=INK, loc='left')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_D, 'm8_house_plan_vs_slide.png'))
        plt.close(fig)
    print('圖', OUT_D)


if __name__ == '__main__':
    sys.exit(main())
