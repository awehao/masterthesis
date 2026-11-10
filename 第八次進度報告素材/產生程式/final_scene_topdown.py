#!/usr/bin/env python3
"""第八次最終場景俯視圖（拿抽屜中的筆）：只讀場景檔，不讀實跑結果。
來源：牆與已知家具＝src/ammr_bringup/worlds/mission_room.sdf（經 evaluation/mission_room_obstacles.known_boxes）；
起點與舊固定停位＝evaluation/results/mission/mission_room_poses.json；未知箱體與行人＝evaluation/results/mission/mission_obstacles_seed1.json（r2）。
舊固定停位（D1、B1、K、R）以灰色空心標出：第八次要改成由量測目標與當下狀態在線計算，這裡只標出舊做法的位置。
    python3 第八次進度報告素材/產生程式/final_scene_topdown.py
"""
import json
import math
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EV = os.path.join(ROOT, 'evaluation')
sys.path.insert(0, EV)
import mission_room_obstacles as MO  # noqa: E402

OUT = os.path.join(ROOT, '第八次進度報告素材', '圖表', '01_最終場景俯視_拿抽屜中的筆.png')
INK, MUTED, WALL, FURN = '#1f2328', '#6b7280', '#3b3f45', '#c9ced6'
UNKNOWN, WALKER, START, TARGET = '#b7791f', '#c2410c', '#15803d', '#1d4ed8'


def font():
    for f in font_manager.findSystemFonts():
        if 'NotoSansCJK-Regular' in f:
            font_manager.fontManager.addfont(f)
            return font_manager.FontProperties(fname=f).get_name()
    return None


def rect_patch(r, **kw):
    return Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2], **kw)


def oriented_box(x, y, yaw_deg, sx, sy):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    pts = [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]
    return [(x + c * px - s * py, y + s * px + c * py) for px, py in pts]


def main():
    fam = font()
    if fam:
        plt.rcParams['font.family'] = fam
    plt.rcParams['axes.unicode_minus'] = False
    poses = json.load(open(os.path.join(EV, 'results', 'mission', 'mission_room_poses.json')))['poses']
    obs = json.load(open(os.path.join(EV, 'results', 'mission', 'mission_obstacles_seed1.json')))
    fig, ax = plt.subplots(figsize=(8.6, 9.4), dpi=160)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f7f8fa')
    x0, x1, y0, y1 = MO.ROOM
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=WALL, lw=2.0))
    for r in MO.known_boxes():
        if r == MO.CABINET:
            continue
        is_wall = (r[1] - r[0] < 0.3 or r[3] - r[2] < 0.3) and max(r[1] - r[0], r[3] - r[2]) > 1.0
        ax.add_patch(rect_patch(r, fc=WALL if is_wall else FURN, ec='none'))
    ax.add_patch(rect_patch(MO.CABINET, fc='#e0e7ff', ec=TARGET, lw=1.6))
    ax.text(MO.CABINET[1] + 0.15, (MO.CABINET[2] + MO.CABINET[3]) / 2, '櫃子＋抽屜\n（筆在抽屜內：待新增）', ha='left', va='center', fontsize=9, color=TARGET)
    ax.plot([0.0, 1.0], [0.06, 0.06], color=TARGET, lw=2.6, solid_capstyle='butt')          # 門扇（關閉）
    th = [math.radians(-a) for a in range(0, 91, 3)]
    ax.plot([math.cos(t) for t in th], [0.06 + math.sin(t) for t in th], color=TARGET, lw=0.8, linestyle=(0, (3, 3)))   # 開門掃掠（向 A 房）
    ax.annotate('門（關閉，向 A 房開）', (0.25, 0.06), xytext=(-0.25, -0.55), ha='right', fontsize=9, color=TARGET, arrowprops=dict(arrowstyle='-', color=TARGET, lw=0.8))
    for u in obs['static_unknown']:
        ax.add_patch(Polygon(oriented_box(u['x'], u['y'], u['yaw_deg'], u['size'][0], u['size'][1]), closed=True, fc=UNKNOWN, ec='white', lw=1.0, alpha=0.9))
    for w in obs['walkers']:
        (ax_, ay_), (bx_, by_) = w['p0'], w['p1']
        ax.add_patch(FancyArrowPatch((ax_, ay_), (bx_, by_), arrowstyle='<->', mutation_scale=10, color=WALKER, lw=1.4, linestyle=(0, (5, 3))))
        mx, my = (ax_ + bx_) / 2, (ay_ + by_) / 2
        ax.add_patch(Circle((mx, my), w['radius'], fc=WALKER, ec='white', lw=1.0, alpha=0.85))
        if abs(by_ - ay_) < 1e-6:
            ax.text(min(ax_, bx_) + 0.05, ay_ + 0.18, f"行人 {w['speed_mps']:.2f} m/s", ha='left', fontsize=8, color=WALKER)
        else:
            ax.text(ax_ - 0.2, max(ay_, by_) - 0.1, f"行人\n{w['speed_mps']:.2f} m/s", ha='right', va='top', fontsize=8, color=WALKER)
    s = poses['S']
    ax.add_patch(Circle((s['x'], s['y']), 0.39, fc=START, ec='white', lw=1.2))
    ax.text(s['x'], s['y'] - 0.68, '起點／終點 S', ha='center', fontsize=9.5, color=START)
    for k in ('D1', 'B1', 'K', 'R'):
        p = poses[k]
        ax.add_patch(Circle((p['x'], p['y']), 0.12, fc='white', ec=MUTED, lw=1.2))
        ax.text(p['x'] + 0.2, p['y'], k, fontsize=8, color=MUTED, va='center')
    ax.text(-3.85, 0.45, 'B 房（書房）', fontsize=11, color=INK)
    ax.text(-3.85, -0.45, 'A 房', fontsize=11, color=INK, va='top')
    ax.set_xlim(x0 - 0.4, x1 + 0.4)
    ax.set_ylim(y0 - 0.4, y1 + 0.4)
    ax.set_aspect('equal')
    ax.set_xlabel('x（m，門局部座標）', color=MUTED)
    ax.set_ylabel('y（m）', color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title('第八次最終場景：拿抽屜中的筆（mission_room＋障礙 seed1 r2）', fontsize=12, color=INK, loc='left')
    handles = [Rectangle((0, 0), 1, 1, fc=UNKNOWN), Circle((0, 0), 1, fc=WALKER), Circle((0, 0), 1, fc='white', ec=MUTED)]
    ax.legend(handles, ['未知靜態箱體（不在地圖，只能靠 LiDAR）', '行人（沿虛線來回，不經門洞）', '舊固定停位（第八次改在線算）'], loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8, frameon=False)
    fig.text(0.01, 0.005, '來源：mission_room.sdf、mission_room_poses.json、mission_obstacles_seed1.json；不含實跑結果。', fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT, facecolor='white')
    print(OUT)


if __name__ == '__main__':
    main()
