#!/usr/bin/env python3
"""
隨機地圖生成器 - 同時產生 PGM/YAML（Nav2 用）和 SDF（Gazebo 用）
兩邊使用完全相同的障礙物資料，保證同步。

Usage:
    python3 generate_map.py
"""
import numpy as np
import random
import yaml
from pathlib import Path

# ===== 設定 =====
SEED        = 20
RESOLUTION  = 0.05   # m/pixel
W, H        = 400 , 400  # pixels → 20m x 20m（400*400
WALL_PX     = 4         # 外牆厚度 (pixels)
N_OBSTACLES = 45        # 障礙物數量
WALL_H      = 1.2       # Gazebo 牆高度 (m)
ORIGIN_X    = 0.0   # bottom-left corner = (0, 0)
ORIGIN_Y    = 0.0   # bottom-left corner = (0, 0)
SPAWN_X     = 1.5   # robot spawn world X
SPAWN_Y     = 1.5   # robot spawn world Y


def px_to_world(px, py):
    """Pixel → Gazebo world coords（Y 軸翻轉）"""
    wx = px * RESOLUTION + ORIGIN_X
    wy = (H - py) * RESOLUTION + ORIGIN_Y
    return wx, wy


def generate():
    random.seed(SEED)
    np.random.seed(SEED)

    img = np.full((H, W), 254, dtype=np.uint8)  # 全白（自由空間）

    # 外牆
    img[0:WALL_PX, :]   = 0
    img[H-WALL_PX:, :]  = 0
    img[:, 0:WALL_PX]   = 0
    img[:, W-WALL_PX:]  = 0

    obstacles = []  # (cx_world, cy_world, w_world, h_world)

    margin = 20  # pixels from wall
    for _ in range(N_OBSTACLES):
        for attempt in range(50):  # 嘗試不重疊放置
            pw = random.randint(12, 50)
            ph = random.randint(12, 50)
            px = random.randint(margin, W - pw - margin)
            py = random.randint(margin, H - ph - margin)

            # 確保不會蓋到起點（左下角附近）
            cx_px = px + pw // 2
            cy_px = py + ph // 2
            spawn_px = int(SPAWN_X / RESOLUTION)          # 30
            spawn_py = int(H - SPAWN_Y / RESOLUTION)      # 370
            if abs(cx_px - spawn_px) < 40 and abs(cy_px - spawn_py) < 40:
                continue

            img[py:py+ph, px:px+pw] = 0

            # 轉換到世界座標（取中心點）
            cx_w, cy_w = px_to_world(px + pw / 2, py + ph / 2)
            w_w = pw * RESOLUTION
            h_w = ph * RESOLUTION
            obstacles.append((cx_w, cy_w, w_w, h_w))
            break

    return img, obstacles


def save_pgm(img, path):
    Hh, Ww = img.shape
    with open(path, 'wb') as f:
        f.write(f'P5\n{Ww} {Hh}\n255\n'.encode())
        f.write(img.tobytes())
    print(f'  PGM: {path}')


def save_yaml(path, pgm_name):
    cfg = {
        'image': pgm_name,
        'resolution': float(RESOLUTION),
        'origin': [float(ORIGIN_X), float(ORIGIN_Y), 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
    }
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f'  YAML: {path}')


def box_sdf(name, x, y, z, sx, sy, sz, ambient, diffuse):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
      <link name="link">
        <collision name="c">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
        </collision>
        <visual name="v">
          <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
          <material>
            <ambient>{ambient}</ambient>
            <diffuse>{diffuse}</diffuse>
          </material>
        </visual>
      </link>
    </model>"""


def save_sdf(obstacles, path):
    room = W * RESOLUTION   # 20.0
    wt   = WALL_PX * RESOLUTION  # wall thickness in meters

    models = ''

    # 地板
    models += f"""
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="c">
          <geometry><plane><normal>0 0 1</normal><size>{room} {room}</size></plane></geometry>
        </collision>
        <visual name="v">
          <geometry><plane><normal>0 0 1</normal><size>{room} {room}</size></plane></geometry>
          <material><ambient>0.7 0.7 0.7 1</ambient><diffuse>0.7 0.7 0.7 1</diffuse></material>
        </visual>
      </link>
    </model>"""

    # 外牆（座標原點在左下角，範圍 0..20）
    wc = '1.0 1.0 1.0 1'
    half = room / 2
    models += box_sdf('wall_north',  half,  room,  WALL_H/2, room, wt,   WALL_H, wc, wc)
    models += box_sdf('wall_south',  half,  0,     WALL_H/2, room, wt,   WALL_H, wc, wc)
    models += box_sdf('wall_east',   room,  half,  WALL_H/2, wt,   room, WALL_H, wc, wc)
    models += box_sdf('wall_west',   0,     half,  WALL_H/2, wt,   room, WALL_H, wc, wc)

    # 障礙物（紅色）
    oc = '0.8 0.2 0.2 1'
    for i, (cx, cy, sw, sh) in enumerate(obstacles):
        models += box_sdf(f'obs_{i}', cx, cy, WALL_H/2, sw, sh, WALL_H, oc, oc)

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.8">
  <world name="random_room">
    <physics name="10ms" type="ignored">
      <max_step_size>0.01</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system"          name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"    name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system"          name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
{models}
  </world>
</sdf>
"""
    with open(path, 'w') as f:
        f.write(sdf)
    print(f'  SDF:  {path}')


if __name__ == '__main__':
    base = Path(__file__).parent.parent
    maps_dir   = base / 'maps'
    worlds_dir = base / 'worlds'
    maps_dir.mkdir(exist_ok=True)
    worlds_dir.mkdir(exist_ok=True)

    print(f'Generating {N_OBSTACLES} obstacles, {W*RESOLUTION:.0f}x{H*RESOLUTION:.0f}m room...')
    img, obstacles = generate()

    save_pgm(img,  maps_dir   / 'random_room.pgm')
    save_yaml(     maps_dir   / 'random_room.yaml', 'random_room.pgm')
    save_sdf(obstacles, worlds_dir / 'random_room.sdf')
    print(f'Done. {len(obstacles)} obstacles placed.')
