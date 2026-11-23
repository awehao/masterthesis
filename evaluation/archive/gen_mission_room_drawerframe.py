#!/usr/bin/env python3
"""mission_room（第七次整體任務地圖）轉成抽屜房座標，供抽屜流程在完整地圖上跑（Howard 10/30：先用完整地圖跑抽屜流程，之後與門串接）。
平移 (−2.4, −3.6)：抽屜櫃由 mission 的 (2.4, 5.05) 落回抽屜房的 (0, 1.45)，K 停位 → 抽屜房停位 (−0.136, 0.56)，下游節點幾何不變。
場景：mission_room.sdf 全部模型平移；另補門兩側牆（door_local_v3：鉸鏈側 x∈[−1.5, −0.016]、閂側 x∈[1.0, 2.5]，y∈[0, 0.12]，高 2.4）——
  mission_room.sdf 只有隔牆，門洞區原由門資產補；本趟**不放門扇**（地圖 mission_room.pgm 門洞為空，與地圖一致；本趟不經過門）。
地圖：同一張 mission_room.pgm，原點 (−4.5, −6.0) → (−6.9, −9.6)。
    python3 evaluation/gen_mission_room_drawerframe.py
"""
import os
import xml.etree.ElementTree as ET

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(WS, 'src/ammr_bringup/worlds/mission_room.sdf')
OUT = os.path.join(WS, 'src/ammr_bringup/worlds/mission_room_drawerframe.sdf')
MAP_OUT = os.path.join(WS, 'src/ammr_bringup/maps/mission_room_drawerframe.yaml')
DX, DY = -2.4, -3.6
DOOR_WALLS = {'door_wall_hinge': (-0.758, 0.06, 1.2, 1.484, 0.12, 2.4), 'door_wall_latch': (1.75, 0.06, 1.2, 1.5, 0.12, 2.4)}


def main():
    for p in (OUT, MAP_OUT):
        if os.path.exists(p):
            raise SystemExit(f'{p} 已存在，不覆寫')
    tree = ET.parse(SRC)
    root = tree.getroot()
    world = root.find('world')
    for m in world.iter('model'):
        p = m.find('pose')
        if p is None:
            continue
        v = [float(x) for x in p.text.split()]
        v[0] += DX
        v[1] += DY
        p.text = ' '.join(f'{x:.4f}' for x in v)
    for name, (x, y, z, sx, sy, sz) in DOOR_WALLS.items():
        m = ET.SubElement(world, 'model', name=name)
        ET.SubElement(m, 'static').text = 'true'
        ET.SubElement(m, 'pose').text = f'{x + DX:.4f} {y + DY:.4f} {z:.4f} 0 0 0'
        lk = ET.SubElement(m, 'link', name='link')
        for tag in ('collision', 'visual'):
            e = ET.SubElement(lk, tag, name=tag)
            g = ET.SubElement(ET.SubElement(e, 'geometry'), 'box')
            ET.SubElement(g, 'size').text = f'{sx} {sy} {sz}'
    tree.write(OUT, encoding='unicode')
    open(MAP_OUT, 'w').write('image: mission_room.pgm\nresolution: 0.05\norigin:\n- -6.9\n- -9.6\n- 0.0\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n'
                             '# mission_room.yaml 平移 (−2.4, −3.6)（gen_mission_room_drawerframe.py）\n')
    print('寫', OUT, MAP_OUT)


if __name__ == '__main__':
    main()
