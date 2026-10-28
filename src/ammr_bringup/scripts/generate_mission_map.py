#!/usr/bin/env python3
"""mission_room 已知靜態地圖（Nav2／AMCL／map-static CBF 用）：只含 mission_room.sdf 的已知物體＋門資產牆＋抽屜櫃外框。
不含：未知靜態障礙物、行人（只能由雷射看到）、門扇（會動；門洞在地圖上留空，門扇由雷射看到）。
門資產牆：door_local_v3 wall.extent_x [−1.5, 2.5]、y ∈ [0, 0.12]，門洞 x ∈ [0, 1.0] 挖空（鉸鏈側凹入 0.015 不另畫，取保守仍算牆）。
抽屜櫃：drawer_unit_bar26 櫃體外框（中心 --drawer-pose，x ±0.30、y −0.236（面板）～ +0.225）。

    python3 src/ammr_bringup/scripts/generate_mission_map.py   # 寫 maps/mission_room.pgm／.yaml（不覆寫）
"""
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SDF = os.path.join(PKG, 'worlds', 'mission_room.sdf')
OUT = os.path.join(PKG, 'maps', 'mission_room')
RES = 0.05
X0, Y0, X1, Y1 = -4.5, -6.0, 5.5, 6.0
DRAWER = (2.4, 5.05)


def boxes():
    raw = re.sub(r'<!--.*?-->', '', open(SDF, encoding='utf-8').read(), flags=re.S)
    out = []
    for m in ET.fromstring(raw).find('world').findall('model'):
        b = m.find('.//collision/geometry/box')
        if b is None:
            continue
        p = [float(v) for v in m.findtext('pose').split()][:2]
        s = [float(v) for v in b.findtext('size').split()][:2]
        out.append((m.get('name'), p[0] - s[0] / 2, p[0] + s[0] / 2, p[1] - s[1] / 2, p[1] + s[1] / 2))
    out += [('door_wall_hinge', -1.5, 0.0, 0.0, 0.12), ('door_wall_latch', 1.0, 2.5, 0.0, 0.12),
            ('drawer_cabinet', DRAWER[0] - 0.30, DRAWER[0] + 0.30, DRAWER[1] - 0.236, DRAWER[1] + 0.225)]
    return out


def main():
    if os.path.exists(OUT + '.pgm') or os.path.exists(OUT + '.yaml'):
        raise SystemExit(f'{OUT}.pgm/.yaml 已存在，不覆寫')
    W, H = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))
    img = np.full((H, W), 254, np.uint8)
    for _, xa, xb, ya, yb in boxes():
        i0, i1 = int(np.floor((xa - X0) / RES)), int(np.ceil((xb - X0) / RES))
        j0, j1 = int(np.floor((ya - Y0) / RES)), int(np.ceil((yb - Y0) / RES))
        img[H - j1:H - j0, max(0, i0):min(W, i1)] = 0           # 影像列 0＝最上（y 大）
    with open(OUT + '.pgm', 'wb') as f:
        f.write(f'P5\n{W} {H}\n255\n'.encode())
        f.write(img.tobytes())
    yaml.safe_dump({'image': 'mission_room.pgm', 'resolution': RES, 'origin': [X0, Y0, 0.0], 'negate': 0,
                    'occupied_thresh': 0.65, 'free_thresh': 0.196}, open(OUT + '.yaml', 'w'), sort_keys=False)
    print(f'寫 {OUT}.pgm（{W}×{H}，{RES} m）', [b[0] for b in boxes()])


if __name__ == '__main__':
    main()
