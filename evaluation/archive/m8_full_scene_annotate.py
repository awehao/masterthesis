#!/usr/bin/env python3
"""第八次完整場景預覽圖標註（不跑模擬）：讀 Isaac 預覽的相機中繼資料（eye、R_wc、hfov、解析度），把場景登錄中的物件世界座標投影到俯視／總覽圖上加標籤。
標籤位置全由登錄檔與 SDF 算出（不手點像素）；行人畫 t＝0 位置與往返路線（虛線）。
    python3 evaluation/m8_full_scene_annotate.py 第八次進度報告素材/圖表/m8_full_scene/r4
"""
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from matplotlib import font_manager                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import mission_walkers as MW                                     # noqa: E402

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()

INK, MUTED = '#1d2229', '#5a6270'
CAT = dict(task='#1f5fbf', unknown='#b5651d', walker='#c0392b', known='#5a6270', start='#1e8449')
KNOWN_ZH = dict(table_a='桌 A', shelf_a='櫃 A', box_a='箱 A', sofa_b='沙發 B', table_b='桌 B', box_b='箱 B')


def read_world(path):
    raw = re.sub(r'<!--.*?-->', '', open(path, encoding='utf-8').read(), flags=re.S)
    out = []
    for m in ET.fromstring(raw).find('world').findall('model'):
        pose = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
        b = m.find('.//collision/geometry/box')
        if b is None:
            continue
        out.append((m.get('name'), pose[:3], [float(v) for v in b.findtext('size').split()]))
    return out


def projector(cam):
    eye, R = np.asarray(cam['eye'], float), np.asarray(cam['R_wc'], float)
    W, H = cam['res']
    fx = (W / 2.0) / math.tan(cam['hfov'] / 2.0)

    def P(p):
        c = R.T @ (np.asarray(p, float) - eye)
        return np.array([fx * c[0] / c[2] + W / 2.0, fx * c[1] / c[2] + H / 2.0])
    return P


def items(reg, obs, walls):
    S = reg['start']['pose']
    dx, dy, _ = reg['drawer']['pose']
    mk = reg['marker']['center_drawer_frame']
    it = [('start', 'S 起點（機器人）', [S['x'], S['y'], 0.5]),
          ('task', '門（可動門扇＋把手）\n開口 1.00 m、門扇 0.95 m', [0.5, 0.06, 2.0]),
          ('task', '抽屜櫃（可動抽屜＋把手）\n內有粗馬克筆代理物', [dx, dy, 0.9]),
          ('task', f'馬克筆代理物 六角柱 對邊{reg["marker"]["across_flats"] * 1000:.0f}×{reg["marker"]["length"] * 1000:.0f} mm、{reg["marker"]["mass_kg"] * 1000:.0f} g', [dx + mk[0], dy + mk[1], mk[2]])]
    for o in obs['static_unknown']:
        zh = '未知椅' if o['kind'] == 'chair' else '未知箱'
        it.append(('unknown', f"{zh} {o['name'].split('_')[1]}", [o['x'], o['y'], o['size'][2]]))
    for w in obs['walkers']:
        x, y = MW.position(w, 0.0)
        it.append(('walker', f"行人 {w['name'].split('_')[1]}（{w['speed_mps']:.2f} m/s）", [x, y, w['height']]))
    for n, p, s in walls:
        if n in KNOWN_ZH:
            it.append(('known', KNOWN_ZH[n], [p[0], p[1], s[2]]))
    return it


def draw(img_path, cam, it, obs, out_path, offsets, title, room_labels=True, only=None, routes=True):
    img = plt.imread(img_path)
    H, W = img.shape[:2]
    P = projector(cam)
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    for w in (obs['walkers'] if routes else []):
        a, b = P([*w['p0'], 0.02]), P([*w['p1'], 0.02])
        ax.plot([a[0], b[0]], [a[1], b[1]], ls=(0, (6, 4)), lw=2, color=CAT['walker'], alpha=0.75)
    fs = max(11, int(W / 95))
    for k, (cat, txt, p) in enumerate(it):
        u = P(p)
        if not (0 <= u[0] < W and 0 <= u[1] < H):
            continue
        key = txt.split('（')[0].split('\n')[0].split(' 六角柱')[0]
        if only is not None and key not in only:
            continue
        off = offsets.get(key, (40, -40))
        ax.plot(u[0], u[1], 'o', ms=7, mfc=CAT[cat], mec='white', mew=2, zorder=5)
        ax.annotate(txt, xy=u, xytext=(u[0] + off[0], u[1] + off[1]), fontsize=fs, color=INK, zorder=6,
                    ha='left' if off[0] >= 0 else 'right', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=CAT[cat], lw=1.6, alpha=0.92),
                    arrowprops=dict(arrowstyle='-', color=CAT[cat], lw=1.4))
    if room_labels:
        for txt, p in (('A 房', [0.2, -1.9, 0.0]), ('B 房', [3.6, 1.0, 0.0])):
            u = P(p)
            ax.text(u[0], u[1], txt, fontsize=fs + 8, color=MUTED, ha='center', va='center', alpha=0.85, fontweight='bold')
    ax.text(12, H - 12, title, fontsize=fs - 1, color=INK, ha='left', va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#c8ccd2', alpha=0.92))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    d = sys.argv[1]
    meta = json.load(open(os.path.join(d, 'm8_scene_preview_meta.json')))
    reg = json.load(open(os.path.join(HERE, 'experiments', '20261111_m8_full_scene.json')))
    obs = json.load(open(os.path.join(WS, reg['obstacles']['path'])))
    walls = read_world(os.path.join(WS, reg['world']['path']))
    it = items(reg, obs, walls)
    top_off = {'S 起點': (60, 40), '門': (60, 70), '抽屜櫃': (-60, 70), '馬克筆代理物': (-60, 135),
               '未知箱 a1': (60, 40), '未知箱 a2': (-50, -45), '未知椅 a3': (40, -70), '未知箱 b1': (50, -40), '未知箱 b2': (60, 40),
               '行人 a1': (60, -40), '行人 a2': (-60, 60), '行人 b1': (50, 40),
               '桌 A': (40, 60), '櫃 A': (-20, -60), '箱 A': (40, 60), '沙發 B': (30, 70), '桌 B': (40, 60), '箱 B': (-60, 50)}
    note = '預覽：未步進物理；行人為 t＝0 位置，虛線為往返路線；未知箱／椅與行人不進靜態地圖'
    draw(os.path.join(d, meta['fixed']['top']['file']), meta['fixed']['top'], it, obs, os.path.join(d, 'm8_scene_top_annot.png'), top_off,
         '完整任務場景（俯視，上＝+y）　' + note)
    ov_off = {'門': (-70, 60), '抽屜櫃': (100, -70), '行人 b1': (-80, -60), '未知箱 b2': (80, -50), '行人 a2': (20, -90), '未知椅 a3': (40, -80),
              '行人 a1': (60, -70), '未知箱 a2': (-80, -60), '未知箱 a1': (60, -50), '未知箱 b1': (-60, -60)}
    draw(os.path.join(d, meta['fixed']['overview']['file']), meta['fixed']['overview'], it, obs, os.path.join(d, 'm8_scene_overview_annot.png'), ov_off,
         '完整任務場景（總覽，由 −y 側斜俯；起點 S 被南牆遮住，見俯視圖）', room_labels=False, only=set(ov_off), routes=False)
    print('ok', [x[1].split('\n')[0] for x in it])


if __name__ == '__main__':
    main()
