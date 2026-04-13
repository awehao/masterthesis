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
SEED        = 42
RESOLUTION  = 0.05   # m/pixel
W, H        = 400, 400  # pixels → 20m x 20m
WALL_PX     = 4         # 外牆厚度 (pixels)
N_OBSTACLES = 35        # 障礙物數量
WALL_H      = 1.2       # Gazebo 牆高度 (m)
ORIGIN_X    = -W * RESOLUTION / 2  # -10.0
ORIGIN_Y    = -H * RESOLUTION / 2  # -10.0


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

            # 確保不會蓋到起點（中心附近）
            cx_px = px + pw // 2
            cy_px = py + ph // 2
            center_px = W // 2
            center_py = H // 2
            if abs(cx_px - center_px) < 40 and abs(cy_px - center_py) < 40:
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

    # 外牆
    wc = '1.0 1.0 1.0 1'
    half = room / 2
    models += box_sdf('wall_north',  0,     half,  WALL_H/2, room, wt,   WALL_H, wc, wc)
    models += box_sdf('wall_south',  0,    -half,  WALL_H/2, room, wt,   WALL_H, wc, wc)
    models += box_sdf('wall_east',   half,  0,     WALL_H/2, wt,   room, WALL_H, wc, wc)
    models += box_sdf('wall_west',  -half,  0,     WALL_H/2, wt,   room, WALL_H, wc, wc)

    # 障礙物（紅色）
    oc = '0.8 0.2 0.2 1'
    for i, (cx, cy, sw, sh) in enumerate(obstacles):
        models += box_sdf(f'obs_{i}', cx, cy, WALL_H/2, sw, sh, WALL_H, oc, oc)

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.8">
  <world name="random_room">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
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
    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui>
          <property type="bool" key="showTitleBar">false</property>
          <property type="string" key="state">docked</property>
        </gz-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.4 0.4 0.4</ambient_light>
        <background_color>0.3 0.3 0.3</background_color>
        <camera_pose>0 0 18 0 1.5 1.5708</camera_pose>
      </plugin>
      <plugin filename="GzSceneManager" name="Scene Manager">
        <gz-gui>
          <property key="resizable" type="bool">false</property>
          <property key="state" type="string">floating</property>
          <property key="visible" type="bool">false</property>
        </gz-gui>
      </plugin>
    </gui>
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
