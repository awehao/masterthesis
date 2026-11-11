#!/usr/bin/env python3
"""第八次住宅格局場景 m8_house_r1（Howard 2026-11-12：「希望能像我簡報中研究目標那一頁的房間一樣的結構」；reviews/20261112_howard_research_goal_room_layout.md）。
新版本，不覆寫舊兩房場景。**一份場景定義**（本檔 SCENE）產生：SDF（牆、家具）、靜態地圖、障礙檔、登錄檔；Isaac 預覽、離線核對、平面圖都讀登錄檔。
參考：第六次進度報告.pdf 第 3 頁。三個相連空間：左側起始臥室 →（可操作的關閉門）→ 中央客廳／通行區 →（內牆開口）→ 右側取物區（抽屜櫃＋粗馬克筆代理物）。
座標＝門局部座標（door_local_v3 不動）：鉸鏈在原點、門牆沿 x 佔 y∈[0, 0.12]、門洞 x∈[0, 1.0]、門向 y<0（臥室）側開。
  與簡報的對應：簡報「左→右」＝世界 +y；簡報「上→下」＝世界 +x（俯視圖以此方向呈現）。
尺寸是工程設計值（簡報沒有公尺尺寸）：依機器人包絡直徑 0.78 m、門洞 1.00 m、抽屜櫃 0.60×0.46 m 推導；保留空間相鄰、門與櫃的相對位置、家具語意，不宣稱逐像素或尺寸還原。
與簡報的已知差異（照實列）：門扇向臥室側開（既有門資產與開門流程的方向；簡報畫成向客廳開）；抽屜櫃資產轉 90° 靠取物區上牆（櫃面朝 +x）。
    python3 evaluation/m8_house_scene_build.py"""
import hashlib
import json
import math
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
DOOR = 'src/my_omnibot_description/config/door_local_v3.yaml'
DRAWER = 'src/my_omnibot_description/config/drawer_unit_bar26.yaml'
MARKER = 'src/my_omnibot_description/config/marker_proxy_m8_hex.yaml'
WORLD = 'src/ammr_bringup/worlds/mission_house_m8.sdf'
MAP = 'src/ammr_bringup/maps/mission_house_m8'
OBS = 'evaluation/results/mission/mission_obstacles_m8_house_r1.json'
REG = 'evaluation/experiments/20261112_m8_house_scene.json'

XI = (-3.0, 3.0)            # 室內 x 範圍（簡報上下）
Y_BED, Y_LIV, Y_RIGHT = (-4.2, 0.0), (0.12, 5.0), (5.12, 8.8)     # 臥室／客廳／取物區的室內 y 範圍
WT, WH = 0.2, 2.4           # 外牆厚、牆高（同門資產牆高）
IT = 0.12                   # 內牆厚（同門資產牆）
OPEN2 = (-0.8, 0.8)         # 客廳↔取物區的開口 x 範圍（淨寬 1.6 m）
DRAWER_POSE = (-2.755, 7.0, 90.0)                                   # 抽屜櫃中心 x, y, 偏航（度）：櫃背貼取物區上牆（x＝−3），櫃面朝 +x，抽屜沿 +x 拉出
START = dict(x=1.5, y=-2.8, yaw_deg=90.0, note='臥室起點／終點（面向門牆 +y）')


def wall(name, x0, x1, y0, y1, rgb=(0.9, 0.9, 0.9)):
    return dict(name=name, kind='wall', c=[(x0 + x1) / 2, (y0 + y1) / 2, WH / 2], s=[x1 - x0, y1 - y0, WH], rgb=list(rgb))


def furn(name, zh, cx, cy, sx, sy, sz, rgb, room):
    return dict(name=name, kind='furniture', zh=zh, room=room, c=[cx, cy, sz / 2], s=[sx, sy, sz], rgb=list(rgb))


def scene():
    xo0, xo1, yo0, yo1 = XI[0] - WT, XI[1] + WT, Y_BED[0] - WT, Y_RIGHT[1] + WT
    W = [wall('wall_top', xo0, XI[0], yo0, yo1), wall('wall_bottom', XI[1], xo1, yo0, yo1), wall('wall_left', XI[0], XI[1], yo0, Y_BED[0]), wall('wall_right', XI[0], XI[1], Y_RIGHT[1], yo1),
         wall('partition_door_a', XI[0], -1.5, 0.0, IT), wall('partition_door_b', 2.5, XI[1], 0.0, IT),        # 門牆延伸（門資產牆 x∈[−1.5, 2.5] 由門資產建）
         wall('partition_right_a', XI[0], OPEN2[0], Y_LIV[1], Y_RIGHT[0]), wall('partition_right_b', OPEN2[1], XI[1], Y_LIV[1], Y_RIGHT[0])]
    wood, dark, blue, pale = (0.80, 0.66, 0.45), (0.22, 0.22, 0.26), (0.30, 0.45, 0.70), (0.86, 0.80, 0.68)
    F = [furn('bed', '床', -0.3, -3.15, 1.5, 2.0, 0.5, blue, 'bedroom'), furn('desk', '書桌', -2.65, -2.7, 0.6, 1.4, 0.75, wood, 'bedroom'),
         furn('dresser', '矮櫃', -2.72, -1.0, 0.5, 0.9, 0.9, wood, 'bedroom'), furn('nightstand', '床頭櫃', 1.05, -3.9, 0.45, 0.45, 0.5, wood, 'bedroom'),
         furn('side_table_bed', '小桌', 2.6, -1.6, 0.6, 1.0, 0.6, pale, 'bedroom'),
         furn('tv_stand', '電視櫃', -2.72, 2.3, 0.5, 1.8, 0.5, wood, 'living'), furn('tv', '電視', -2.85, 2.3, 0.08, 1.1, 1.15, dark, 'living'), furn('media_cabinet', '邊櫃', -2.72, 3.8, 0.5, 0.6, 0.6, wood, 'living'),
         furn('loveseat', '雙人沙發', -1.3, 0.62, 1.5, 0.9, 0.8, dark, 'living'), furn('coffee_table', '茶几', 1.35, 2.3, 0.6, 1.1, 0.45, pale, 'living'),
         furn('sofa', '沙發', 2.55, 2.3, 0.9, 2.0, 0.8, dark, 'living'), furn('end_table_a', '邊桌', 2.7, 0.85, 0.5, 0.5, 0.5, wood, 'living'), furn('end_table_b', '邊桌', 2.7, 3.75, 0.5, 0.5, 0.5, wood, 'living'),
         furn('cabinet_a', '收納櫃', -2.72, 5.72, 0.55, 1.1, 1.0, wood, 'right'), furn('cabinet_b', '收納櫃', -2.72, 8.15, 0.55, 1.1, 1.0, wood, 'right'),
         furn('shelf', '層架', 0.9, 8.57, 2.2, 0.45, 1.2, wood, 'right')]
    return W, F


def drawer_footprint():
    """抽屜櫃外框（櫃體本地 x∈±0.30、y∈[−0.236, 0.225]）轉偏航後的世界軸對齊包圍盒。"""
    cx, cy, yaw = DRAWER_POSE
    c_, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    P = np.array([[x, y] for x in (-0.30, 0.30) for y in (-0.236, 0.225)])
    Wp = np.c_[cx + c_ * P[:, 0] - s_ * P[:, 1], cy + s_ * P[:, 0] + c_ * P[:, 1]]
    return [float(Wp[:, 0].min()), float(Wp[:, 0].max()), float(Wp[:, 1].min()), float(Wp[:, 1].max())]


def unknowns():
    """未知物與行人：事前明示設計值（不抽樣、不依機器人結果調整；r1 首輪離線核對發現膨脹後客廳不通與收納櫃侵入抽屜操作區，據幾何核對調整過一次，之後固定）。不進靜態地圖；真值只給場景與評估。行人低速開發設定 0.20 m/s。"""
    st = [dict(name='unknown_l1', kind='box', x=-0.9, y=3.0, size=[0.40, 0.30, 0.40], yaw_deg=30.0), dict(name='unknown_l2', kind='box', x=1.1, y=4.2, size=[0.35, 0.35, 0.30], yaw_deg=-20.0),
          dict(name='unknown_l3', kind='box', x=-1.5, y=1.9, size=[0.30, 0.40, 0.30], yaw_deg=10.0), dict(name='unknown_l4', kind='chair', x=1.7, y=4.3, size=[0.45, 0.45, 0.90], yaw_deg=45.0),
          dict(name='unknown_r1', kind='box', x=1.4, y=6.5, size=[0.35, 0.35, 0.30], yaw_deg=15.0)]
    wk = [dict(name='walker_l1', p0=[-1.9, 1.55], p1=[1.8, 1.55], speed_mps=0.20, dwell_s=1.0, phase_s=6.0, radius=0.25, height=1.7),
          dict(name='walker_l2', p0=[0.2, 1.3], p1=[0.2, 4.5], speed_mps=0.20, dwell_s=1.5, phase_s=14.0, radius=0.25, height=1.7),
          dict(name='walker_r1', p0=[0.3, 5.9], p1=[0.3, 8.2], speed_mps=0.20, dwell_s=1.2, phase_s=3.0, radius=0.25, height=1.7)]
    return st, wk


def write_sdf(W, F):
    L = ['<?xml version="1.0"?>', '<!-- 第八次住宅格局場景 m8_house_r1（Howard 2026-11-12）：由 evaluation/m8_house_scene_build.py 產生，勿手改。',
         '     門局部座標（door_local_v3）：門牆 y∈[0,0.12]、門洞 x∈[0,1]；門與門資產牆（x∈[-1.5,2.5]）由 door_asset_v3 建；抽屜櫃與代理物由各自資產建。',
         '     牆高 2.4 m、不含天花板。未知物與行人不在本檔。 -->', '<sdf version="1.9">', '  <world name="mission_house_m8">',
         '    <model name="ground_plane"><static>true</static><link name="link"><collision name="collision"><geometry><plane><normal>0 0 1</normal><size>30 30</size></plane></geometry></collision></link></model>']
    for o in W + F:
        c, s, rgb = o['c'], o['s'], o['rgb']
        col = f"{rgb[0]} {rgb[1]} {rgb[2]} 1"
        L += [f'    <model name="{o["name"]}">', '      <static>true</static>', f'      <pose>{c[0]:.4f} {c[1]:.4f} {c[2]:.4f} 0 0 0</pose>', '      <link name="link">',
              f'        <collision name="c"><geometry><box><size>{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}</size></box></geometry></collision>',
              f'        <visual name="v"><geometry><box><size>{s[0]:.4f} {s[1]:.4f} {s[2]:.4f}</size></box></geometry><material><ambient>{col}</ambient><diffuse>{col}</diffuse></material></visual>',
              '      </link>', '    </model>']
    L += ['  </world>', '</sdf>', '']
    open(os.path.join(WS, WORLD), 'w', encoding='utf-8').write('\n'.join(L))


def write_map(W, F, res=0.05):
    x0, y0, x1, y1 = -3.6, -4.8, 3.6, 9.4
    Wn, Hn = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    img = np.full((Hn, Wn), 254, np.uint8)
    rects = [(o['name'], o['c'][0] - o['s'][0] / 2, o['c'][0] + o['s'][0] / 2, o['c'][1] - o['s'][1] / 2, o['c'][1] + o['s'][1] / 2) for o in W + F]
    rects += [('door_wall_hinge', -1.5, 0.0, 0.0, 0.12), ('door_wall_latch', 1.0, 2.5, 0.0, 0.12), ('drawer_cabinet', *drawer_footprint())]
    for _, xa, xb, ya, yb in rects:
        i0, i1 = int(np.floor((xa - x0) / res)), int(np.ceil((xb - x0) / res))
        j0, j1 = int(np.floor((ya - y0) / res)), int(np.ceil((yb - y0) / res))
        img[Hn - j1:Hn - j0, max(0, i0):min(Wn, i1)] = 0
    with open(os.path.join(WS, MAP + '.pgm'), 'wb') as f:
        f.write(f'P5\n{Wn} {Hn}\n255\n'.encode())
        f.write(img.tobytes())
    yaml.safe_dump({'image': os.path.basename(MAP) + '.pgm', 'resolution': res, 'origin': [x0, y0, 0.0], 'negate': 0, 'occupied_thresh': 0.65, 'free_thresh': 0.196}, open(os.path.join(WS, MAP + '.yaml'), 'w'), sort_keys=False)
    return dict(shape=[Hn, Wn], occupied=int((img < 128).sum()), known=[r[0] for r in rects], excludes='未知箱體、行人、門扇、抽屜本體以外的可動物、代理物')


def sha(p):
    return hashlib.sha256(open(os.path.join(WS, p), 'rb').read()).hexdigest()


def main():
    W, F = scene()
    write_sdf(W, F)
    mp = write_map(W, F)
    st, wk = unknowns()
    json.dump(dict(schema='mission_obstacles/1', map=WORLD, rule='事前明示設計值；不抽樣、不依機器人結果調整；不進靜態地圖；真值不進感知或在線控制', static_unknown=st, walkers=wk), open(os.path.join(WS, OBS), 'w'), ensure_ascii=False, indent=1)
    mk = yaml.safe_load(open(os.path.join(WS, MARKER)))
    cx, cy, yaw = DRAWER_POSE
    c_, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    m = mk['placement']['center']
    reg = dict(name='m8_house_r1', created='2026-11-12（Howard reviews/20261112_howard_research_goal_room_layout.md）', reference='第六次進度報告.pdf 第 3 頁「研究目標」',
               frame='門局部座標（door_local_v3）；簡報左→右＝世界 +y、簡報上→下＝世界 +x', dims=dict(interior_x=list(XI), bedroom_y=list(Y_BED), living_y=list(Y_LIV), right_y=list(Y_RIGHT), wall_t=WT, inner_t=IT, wall_h=WH, opening2_x=list(OPEN2)),
               rooms=dict(bedroom='左側起始房（y<0）：起點、床、書桌、矮櫃、床頭櫃、小桌', living='中央客廳／通行區：電視櫃、沙發、茶几；行人與未知物', right='右側取物區：內牆開口相連；抽屜櫃＋代理物、收納櫃、層架'),
               world=dict(path=WORLD, sha256=sha(WORLD)), door=dict(asset=DOOR, sha256=sha(DOOR), builder='evaluation/door_asset_v3.py', opening_x=[0.0, 1.0], opens_toward='臥室（y<0）'),
               drawer=dict(asset=DRAWER, sha256=sha(DRAWER), pose=list(DRAWER_POSE), builder='evaluation/house_assets.py（drawer_asset.build_usd＋偏航）', footprint=drawer_footprint(), pull_dir=[round(s_, 6), round(-c_, 6)], travel=0.22,
                           note='偏航 90° 版抽屜的物理（滑動、限位）只做過靜態預覽，尚未做拉動驗證'),
               marker=dict(asset=MARKER, sha256=sha(MARKER), center_world=[round(cx + c_ * m[0] - s_ * m[1], 4), round(cy + s_ * m[0] + c_ * m[1], 4), m[2]], axis_yaw_deg=yaw, across_flats=mk['across_flats'], length=mk['length'], mass_kg=mk['mass_kg']),
               obstacles=dict(path=OBS, sha256=sha(OBS), n_static_unknown=len(st), n_walkers=len(wk), walker_speed_mps=0.20), start=dict(pose=START),
               static_map=dict(yaml=MAP + '.yaml', yaml_sha256=sha(MAP + '.yaml'), pgm=MAP + '.pgm', pgm_sha256=sha(MAP + '.pgm'), **mp),
               walls=W, furniture=F, deviations=['門扇向臥室側開（簡報畫成向客廳開）：沿用既有門資產與開門流程方向', '尺寸為工程設計值，非簡報比例還原', '家具為方盒代理（無細節模型）'],
               not_included='天花板；任何任務實跑；夾持可行性；偏航抽屜的拉動驗證')
    json.dump(reg, open(os.path.join(WS, REG), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(dict(map=mp['shape'], drawer=reg['drawer']['footprint'], pull=reg['drawer']['pull_dir'], marker=reg['marker']['center_world'], n_walls=len(W), n_furn=len(F)), ensure_ascii=False))


if __name__ == '__main__':
    main()
