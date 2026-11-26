#!/usr/bin/env python3
"""N1 導航外部基線的三種複雜布局（11/27 Howard：「要像前面兩種的複雜場景、動態障礙物不能太快、要有未知靜態障礙物」）。

產生 worlds/<name>.sdf 與 maps/<name>.pgm/.yaml，三張都是 20×20 m、座標框與 bigarena 相同（原點 -1.5,-1.5）、
gz world 名稱沿用 "bigarena"（run_omnibot_dynamic.sh 在 BIGARENA=1 時用這個名字呼叫 set_pose）：
  n1_bigarena   L1 多房間：bigarena 原本的牆與 18 個已知家具（地圖相同）
  n1_office     L2 辦公室走廊：十字走廊 2.4 m、八間辦公室、門 1.6 m、辦公桌
  n1_warehouse  L3 倉儲貨架：左右兩區貨架（中間有橫向通道）、中央主通道 3 m、右側暫存區棧板
三張共同：
  * 未知靜態障礙 unknown_obs_0..6：尺寸同 bigarena 的 7 個未知箱；kinematic、有碰撞體、不在地圖；
    平常停在場外，每個種子由情境檔（unknown_static）以 gz set_pose 放到位（make_n1_scenario.py）。
  * 行人 dyn_obs_0..9：形狀同 bigarena；只有外觀、沒有碰撞體（同 bigarena_ghost：不推擠機器人與彼此，雷射看得到）。
  * 地圖只含牆與已知家具。
用法：python3 src/ammr_bringup/scripts/generate_n1_layouts.py [--only n1_office]"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_bigarena as GB                                       # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LO, HI, T = GB.LO, GB.HI, GB.WALL_T
UNK_SIZES = [(0.998, 0.973), (1.259, 0.756), (0.704, 0.883), (0.850, 0.859), (0.847, 0.881), (0.804, 0.862), (0.744, 0.905)]
PARK = [(40.0 + 3.0 * i, 40.0) for i in range(len(UNK_SIZES))]      # 場外停放處（未放置時）


def outer():
    h, mid, S = T / 2, (LO + HI) / 2, HI - LO
    return [(mid, LO + h, S, T), (mid, HI - h, S, T), (LO + h, mid, T, S), (HI - h, mid, T, S)]


def run(axis, coord, a, b, gaps=()):
    """一道牆從 a 到 b（沿牆方向），gaps 為開口；回傳實心段 (cx,cy,sx,sy)。"""
    edges = [a] + [v for g in sorted(gaps) for v in g] + [b]
    out = []
    for i in range(0, len(edges) - 1, 2):
        s, e = edges[i], edges[i + 1]
        if e - s > 1e-6:
            out.append(((s + e) / 2, coord, e - s, T) if axis == 'h' else (coord, (s + e) / 2, T, e - s))
    return out


def office():
    W = outer()
    y1, y2, x1, x2 = 7.6, 10.0, 7.6, 10.0                              # 十字走廊 2.4 m
    W += run('h', y1, LO, HI, [(0.2, 1.8), (4.4, 6.0), (x1, x2), (11.0, 12.6), (15.6, 17.2)])
    W += run('h', y2, LO, HI, [(0.6, 2.2), (4.8, 6.4), (x1, x2), (11.4, 13.0), (15.2, 16.8)])
    W += run('v', x1, LO, y1, [(2.4, 4.0)]) + run('v', x1, y2, HI, [(13.0, 14.6)])
    W += run('v', x2, LO, y1, [(4.4, 6.0)]) + run('v', x2, y2, HI, [(14.6, 16.2)])
    W += run('v', 3.0, LO, y1, [(2.0, 3.6)]) + run('v', 3.0, y2, HI, [(14.2, 15.8)])          # 房間隔間（房間之間也有門）
    W += run('v', 14.5, LO, y1, [(3.8, 5.4)]) + run('v', 14.5, y2, HI, [(12.4, 14.0)])
    rooms = [(LO + 0.1, 3.0, LO + 0.1, y1), (3.0, x1, LO + 0.1, y1), (x2, 14.5, LO + 0.1, y1), (14.5, HI - 0.1, LO + 0.1, y1),
             (LO + 0.1, 3.0, y2, HI - 0.1), (3.0, x1, y2, HI - 0.1), (x2, 14.5, y2, HI - 0.1), (14.5, HI - 0.1, y2, HI - 0.1)]
    doors = [((a + b) / 2, y1) for a, b in [(0.2, 1.8), (4.4, 6.0), (11.0, 12.6), (15.6, 17.2)]] + \
            [((a + b) / 2, y2) for a, b in [(0.6, 2.2), (4.8, 6.4), (11.4, 13.0), (15.2, 16.8)]] + \
            [(x1, 3.2), (x1, 13.8), (x2, 5.2), (x2, 15.4), (3.0, 2.8), (3.0, 15.0), (14.5, 4.6), (14.5, 13.2)]
    rng = np.random.default_rng(27)
    known = []
    for (a, b, c, d) in rooms:                                         # 每間 2 張桌子，離牆與門有空間
        for _ in range(2):
            for _t in range(500):
                sx, sy = (1.2, 0.6) if rng.random() < 0.6 else (0.8, 0.8)
                if rng.random() < 0.5:
                    sx, sy = sy, sx
                x, y = rng.uniform(a + 0.9 + sx / 2, b - 0.9 - sx / 2), rng.uniform(c + 0.9 + sy / 2, d - 0.9 - sy / 2)
                rad = 0.5 * np.hypot(sx, sy)
                if min(np.hypot(x - p, y - q) for p, q in doors) < rad + 1.4:
                    continue
                if any(np.hypot(x - p, y - q) < rad + 0.5 * np.hypot(u, v) + 1.0 for p, q, u, v in known):
                    continue
                known.append((x, y, sx, sy))
                break
    return W, known


def warehouse():
    W = outer()
    known = []
    for i in range(6):                                                 # 貨架 0.8 m 深，走道 1.8 m；每區中間留 1.6 m 橫向通道
        y = 0.9 + 2.6 * i
        for a, b in [(-0.6, 2.4), (4.0, 6.4), (9.6, 12.2), (13.8, 15.6)]:
            known.append(((a + b) / 2, y, b - a, 0.8))
    for (x, y) in [(17.1, 1.0), (17.1, 4.2), (17.1, 7.8), (17.1, 11.2)]:   # 右側暫存區棧板
        known.append((x, y, 1.0, 1.0))
    for (x, y, sx, sy) in [(1.0, 16.9, 2.4, 0.8), (11.0, 17.2, 2.0, 0.8), (6.0, 16.6, 1.0, 1.0)]:   # 上方出貨區
        known.append((x, y, sx, sy))
    return W, known


def bigarena_layout():
    raw = re.sub(r'<!--.*?-->', '', (ROOT / 'worlds' / 'bigarena.sdf').read_text(), flags=re.S)
    W, known = [], []
    for m in re.finditer(r'<model name="((?:wall|known_obs)_\d+)">(.*?)</model>', raw, re.S):
        p = [float(v) for v in re.search(r'<pose>([-\d. e]+)</pose>', m.group(2)).group(1).split()]
        sx, sy = [float(v) for v in re.search(r'<box><size>([\d.]+) ([\d.]+)', m.group(2)).groups()]
        (W if m.group(1).startswith('wall') else known).append((p[0], p[1], sx, sy))
    return W, known


def unknown_sdf(i, sx, sy):
    x, y = PARK[i]
    return f"""
    <model name="unknown_obs_{i}">
      <pose>{x:.3f} {y:.3f} 0.500 0 0 0</pose>
      <link name="link">
        {GB.BODY_HEAD}
        <collision name="c"><geometry><box><size>{sx:.3f} {sy:.3f} 1.000</size></box></geometry></collision>
        <visual name="v"><geometry><box><size>{sx:.3f} {sy:.3f} 1.000</size></box></geometry>
          <material><ambient>0.9 0.7 0.1 1</ambient><diffuse>0.9 0.7 0.1 1</diffuse></material></visual>
      </link>
    </model>"""


def ghost(sdf):
    return re.sub(r'\s*<collision name="c\d?">.*?</collision>', '', sdf, flags=re.S)


def write(name, W, known):
    occ = GB.rasterise(W + known)
    img = np.where(occ, 0, 254).astype(np.uint8)
    (ROOT / 'maps' / f'{name}.pgm').write_bytes(f'P5\n{GB.N} {GB.N}\n255\n'.encode() + img.tobytes())
    yaml.safe_dump({'image': f'{name}.pgm', 'resolution': GB.RES, 'origin': [GB.ORIGIN[0], GB.ORIGIN[1], 0.0],
                    'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196}, open(ROOT / 'maps' / f'{name}.yaml', 'w'))
    parts = [GB.box_sdf(f'wall_{i}', *w, '0.4 0.4 0.4 1') for i, w in enumerate(W)]
    parts += [GB.box_sdf(f'known_obs_{i}', *k, '0.45 0.45 0.5 1', 1.0) for i, k in enumerate(known)]
    parts += [unknown_sdf(i, *s) for i, s in enumerate(UNK_SIZES)]
    parts += [ghost(GB.mover_sdf(nm, 30.0 + 2.0 * j, 30.0, shape, size)) for j, (nm, shape, size) in enumerate(GB.DYN_SHAPES)]
    world = GB.WORLD.replace('<world name="bigarena">', f'<world name="bigarena">\n    <!-- {name}：generate_n1_layouts.py 產生；world 名稱沿用 bigarena 供 set_pose -->')
    (ROOT / 'worlds' / f'{name}.sdf').write_text(world % ''.join(parts))
    print(f'{name}: {len(W)} 牆段、{len(known)} 已知家具、{len(UNK_SIZES)} 未知箱（場外待放）、{len(GB.DYN_SHAPES)} 行人（無碰撞體）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    for name, fn in (('n1_bigarena', bigarena_layout), ('n1_office', office), ('n1_warehouse', warehouse)):
        if a.only and a.only != name:
            continue
        write(name, *fn())


if __name__ == '__main__':
    main()
