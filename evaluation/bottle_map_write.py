#!/usr/bin/env python3
"""瓶子 N1 地圖：把 bottle_nav_chain.occupancy 同一份已知家具（桌腳、側牆）寫成 map_server 可讀的 pgm／yaml。障礙方塊是未知物，不進地圖。
    python3 evaluation/bottle_map_write.py <輸出基名> [--wall-box cx,cy,cz,sx,sy,sz] [--table2 dx,dy] [--region-offset ox,oy]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bottle_nav_chain as NC                           # noqa: E402
import bottle_s1_geom as G                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--wall-box', default='')
    ap.add_argument('--walls-file', default='', help='場景 JSON 的 boxes（牆體同步進地圖；只進地圖，場景與掃描由各自入口讀同一檔）')
    ap.add_argument('--table2', default='', help='第二桌位移 dx,dy（同一桌子資產；只放桌腳，桌面在掃描面之上）')
    ap.add_argument('--region-offset', default='0,0', help='第一桌（抓取區域）世界平移 ox,oy（R-B1；預設 0,0＝原場景；第二桌、側牆不隨之移動）')
    a, _ = ap.parse_known_args()
    ro = np.array([float(v) for v in a.region_offset.split(',')] + [0.0])
    boxes = [(b[1] + ro, b[2]) for b in G.table_boxes() if b[0] != 'table_top']
    if a.wall_box:
        v = [float(x) for x in a.wall_box.split(',')]
        boxes.append((np.array(v[:3]), np.array(v[3:])))
    if a.walls_file:
        for v in json.load(open(a.walls_file))['boxes'].values():
            boxes.append((np.array(v[:3], float), np.array(v[3:], float)))
    if a.table2:
        dx, dy = (float(v) for v in a.table2.split(','))
        boxes += [(c + np.array([dx, dy, 0.0]), sz) for n, c, sz in G.table_boxes() if n != 'table_top']
    g = NC.occupancy(boxes)
    nx, ny = g.info.width, g.info.height
    occ = np.array(g.data, np.int8).reshape(ny, nx)
    img = np.where(occ > 50, 0, 254).astype(np.uint8)[::-1]        # pgm 第一列是最大 y
    with open(a.out + '.pgm', 'wb') as f:
        f.write(f'P5\n{nx} {ny}\n255\n'.encode())
        f.write(img.tobytes())
    o = g.info.origin.position
    with open(a.out + '.yaml', 'w') as f:
        f.write(f'image: {os.path.basename(a.out)}.pgm\nmode: trinary\nresolution: {g.info.resolution}\norigin: [{o.x}, {o.y}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    print(f'地圖 {nx}×{ny}、佔據格 {int((occ > 50).sum())}、家具 {len(boxes)} 件')


if __name__ == '__main__':
    sys.exit(main())
