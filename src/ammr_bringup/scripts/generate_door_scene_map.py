#!/usr/bin/env python3
"""Isaac 門場景（door_local_v3）已知靜態地圖：NAV1（GMPC＋導航 CBF、掃描追蹤器的地圖扣除）用。Codex 20261029_081047：地圖須對應新場景，不借舊房間圖。
只含：門資產隔牆 wall.extent_x [−1.5, 2.5]、y ∈ [0, 0.12]，門洞 x ∈ [0, frame_clear_width] 挖空（鉸鏈側凹入 0.015 不另畫，保守仍算牆）。
不含：門扇與把手（會動；由雷射看到）。A 房（y < 0）與 B 房（y > 0.12）在 Isaac 場景沒有外牆，地圖範圍外視為未知。
座標＝門局部座標＝Isaac 世界座標（門根變換為單位，ev9r8 RG8 啟動核）。

    python3 src/ammr_bringup/scripts/generate_door_scene_map.py   # 寫 maps/door_scene_v3.pgm／.yaml（不覆寫）
"""
import os

import numpy as np
import yaml

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(os.path.dirname(os.path.dirname(PKG)), 'src', 'my_omnibot_description', 'config', 'door_local_v3.yaml')
OUT = os.path.join(PKG, 'maps', 'door_scene_v3')
RES = 0.05
X0, Y0, X1, Y1 = -3.5, -4.5, 4.0, 3.0


def boxes(d):
    w = d['wall']
    x_lo, x_hi = (float(v) for v in w['extent_x'])
    t, W = float(w['thickness']), float(d['frame_clear_width'])
    return [('wall_hinge_side', x_lo, 0.0, 0.0, t), ('wall_latch_side', W, x_hi, 0.0, t)]


def main():
    if os.path.exists(OUT + '.pgm') or os.path.exists(OUT + '.yaml'):
        raise SystemExit(f'{OUT}.pgm／.yaml 已存在，不覆寫')
    d = yaml.safe_load(open(SPEC))
    nx, ny = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))
    img = np.full((ny, nx), 254, np.uint8)
    for _, xa, xb, ya, yb in boxes(d):
        i0, i1 = int(np.floor((xa - X0) / RES)), int(np.ceil((xb - X0) / RES))
        j0, j1 = int(np.floor((ya - Y0) / RES)), int(np.ceil((yb - Y0) / RES))
        img[ny - j1:ny - j0, max(0, i0):min(nx, i1)] = 0
    with open(OUT + '.pgm', 'wb') as f:
        f.write(f'P5\n{nx} {ny}\n255\n'.encode())
        f.write(img.tobytes())
    yaml.safe_dump({'image': os.path.basename(OUT) + '.pgm', 'mode': 'trinary', 'resolution': RES, 'origin': [X0, Y0, 0.0],
                    'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.25}, open(OUT + '.yaml', 'w'), sort_keys=False)
    print(f'寫 {OUT}.pgm（{nx}×{ny}，{RES} m）／.yaml；牆 {boxes(d)}')


if __name__ == '__main__':
    main()
