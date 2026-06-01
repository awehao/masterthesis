#!/usr/bin/env python3
"""
隨機地圖生成器 - 同時產生 PGM/YAML（Nav2 用）和 SDF（Gazebo 用）
兩邊使用完全相同的障礙物資料，保證同步。

輸出：
  maps/random_room.pgm/.yaml   - Nav2 靜態地圖（只含靜態障礙物 + 牆）
  worlds/random_room.sdf       - Gazebo 靜態世界（baseline 比較用）
  worlds/random_room_dynamic.sdf       - Gazebo 動態世界（靜態 + 移動障礙物）
  config/dynamic_trajectories.yaml     - 移動障礙物軌跡 (start/end/speed)

Usage:
    python3 generate_map.py
"""
import numpy as np
import random
import yaml
from pathlib import Path

# ===== 靜態設定 =====
SEED        = 20
RESOLUTION  = 0.05   # m/pixel
W, H        = 400 , 400  # pixels → 20m x 20m（400*400
WALL_PX     = 4         # 外牆厚度 (pixels)
N_OBSTACLES = 45        # 障礙物數量
WALL_H      = 1.2       # Gazebo 牆高度 (m)
ORIGIN_X    = -1.5  # world(-1.5,-1.5) = map image bottom-left corner
ORIGIN_Y    = -1.5  # spawn at (0,0) = map frame = 1.5m inside room
SPAWN_X     = 0.0
SPAWN_Y     = 0.0

# ===== 動態障礙物設定 =====
N_DYNAMIC      = 3       # 動態障礙物數量
DYN_RADIUS     = 0.25    # 圓柱半徑 (m)
DYN_HEIGHT     = 1.0     # 圓柱高度 (m)
DYN_SPEED      = 0.4     # 移動速度 (m/s)
DYN_CLEAR_M    = 0.8     # 起點與終點需保留的淨空 (m)
DYN_MIN_TRAJ_M = 2.5     # 最短軌跡長度 (m)
DYN_MAX_TRAJ_M = 6.0     # 最長軌跡長度 (m)
DYN_AVOID_SPAWN_M = 2.0  # 不可放置在距離 spawn 多近的範圍


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
            spawn_px = int((SPAWN_X - ORIGIN_X) / RESOLUTION)
            spawn_py = int(H - (SPAWN_Y - ORIGIN_Y) / RESOLUTION)
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


def is_clear(img, px, py, clear_px):
    """檢查以 (px, py) 為中心、半徑 clear_px 的方形範圍是否全為自由空間（254）。"""
    h, w = img.shape
    if px < 0 or py < 0 or px >= w or py >= h:
        return False
    x0 = max(0, px - clear_px)
    x1 = min(w, px + clear_px + 1)
    y0 = max(0, py - clear_px)
    y1 = min(h, py + clear_px + 1)
    return int(img[y0:y1, x0:x1].min()) >= 200  # 254 = free, 0 = obstacle


def place_dynamic_obstacles(img, n):
    """在地圖自由區掃描 n 個合法起點與直線軌跡。

    每個動態障礙物在 8 個方向中選最長無碰撞直線作為軌跡 (ping-pong)。
    """
    clear_px    = int(DYN_CLEAR_M    / RESOLUTION)
    min_traj_px = int(DYN_MIN_TRAJ_M / RESOLUTION)
    max_traj_px = int(DYN_MAX_TRAJ_M / RESOLUTION)
    h, w = img.shape
    margin = max(clear_px + 2, 25)

    spawn_px = int((SPAWN_X - ORIGIN_X) / RESOLUTION)
    spawn_py = int(h - (SPAWN_Y - ORIGIN_Y) / RESOLUTION)
    avoid_spawn_px = int(DYN_AVOID_SPAWN_M / RESOLUTION)

    dynamics = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (-1, 1), (1, -1), (-1, -1)]

    for _ in range(500):
        if len(dynamics) >= n:
            break
        px = random.randint(margin, w - margin - 1)
        py = random.randint(margin, h - margin - 1)

        # 避開機器人 spawn 周圍
        if (px - spawn_px) ** 2 + (py - spawn_py) ** 2 < avoid_spawn_px ** 2:
            continue
        # 起點需有淨空
        if not is_clear(img, px, py, clear_px):
            continue
        # 避開已放置的動態起點 (世界座標距離 > 2 m)
        sx_w, sy_w = px_to_world(px, py)
        too_close = any(
            (sx_w - d['start'][0]) ** 2 + (sy_w - d['start'][1]) ** 2 < 2.0 ** 2
            for d in dynamics
        )
        if too_close:
            continue

        # 8 方向找最長無碰撞直線
        best_len_px = 0
        best_end_px = (px, py)
        for dpx, dpy in directions:
            tx, ty = px, py
            step = 4
            while True:
                tx_n = tx + dpx * step
                ty_n = ty + dpy * step
                if not is_clear(img, tx_n, ty_n, clear_px):
                    break
                if tx_n < margin or tx_n >= w - margin:
                    break
                if ty_n < margin or ty_n >= h - margin:
                    break
                tx, ty = tx_n, ty_n
                cur = abs(tx - px) + abs(ty - py)
                if cur >= max_traj_px:
                    break
            cur_len = abs(tx - px) + abs(ty - py)
            if cur_len > best_len_px:
                best_len_px = cur_len
                best_end_px = (tx, ty)

        if best_len_px < min_traj_px:
            continue

        ex_w, ey_w = px_to_world(*best_end_px)
        dynamics.append({
            'name'  : f'dyn_obs_{len(dynamics)}',
            'start' : [round(sx_w, 4), round(sy_w, 4)],
            'end'   : [round(ex_w, 4), round(ey_w, 4)],
            'speed' : DYN_SPEED,
            'radius': DYN_RADIUS,
            'height': DYN_HEIGHT,
        })

    return dynamics


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


def dyn_model_sdf(name, x, y, radius, height):
    """產生動態圓柱障礙物 SDF。

    設計要點：
      - <kinematic>true</kinematic>：link 不參與動力學積分，外力推不動、不會被
        撞翻 → 由 VelocityControl 直接設定速度，pose 嚴格按指令演進
      - <gravity>false</gravity>：保險再加一層，移除重力
      - 仍保留 collision → LiDAR 看得到，機器人視為實心障礙
      - VelocityControl 在 world frame 套用 linear velocity
      - PosePublisher 發布 model pose（GZ → ROS bridge → /model/<name>/pose）
    """
    return f"""
    <model name="{name}">
      <pose>{x:.4f} {y:.4f} {height/2:.4f} 0 0 0</pose>
      <link name="link">
        <kinematic>true</kinematic>
        <gravity>false</gravity>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
          </inertia>
        </inertial>
        <collision name="c">
          <geometry><cylinder><radius>{radius}</radius><length>{height}</length></cylinder></geometry>
        </collision>
        <visual name="v">
          <geometry><cylinder><radius>{radius}</radius><length>{height}</length></cylinder></geometry>
          <material>
            <ambient>0.2 0.4 0.9 1</ambient>
            <diffuse>0.2 0.4 0.9 1</diffuse>
          </material>
        </visual>
      </link>
      <plugin filename="gz-sim-velocity-control-system"
              name="gz::sim::systems::VelocityControl"/>
      <plugin filename="gz-sim-pose-publisher-system"
              name="gz::sim::systems::PosePublisher">
        <publish_link_pose>false</publish_link_pose>
        <publish_model_pose>true</publish_model_pose>
        <use_pose_vector_msg>false</use_pose_vector_msg>
        <update_frequency>20</update_frequency>
      </plugin>
    </model>"""


def save_sdf(obstacles, path, dynamics=None):
    room = W * RESOLUTION   # 20.0
    wt   = WALL_PX * RESOLUTION  # wall thickness in meters

    models = ''

    # 地板（pose 對齊房間中心）
    gx = ORIGIN_X + room / 2
    gy = ORIGIN_Y + room / 2
    models += f"""
    <model name="ground_plane">
      <static>true</static>
      <pose>{gx:.4f} {gy:.4f} 0 0 0 0</pose>
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

    # 外牆（位置由 ORIGIN_X/Y 決定，自動對齊地圖）
    wc = '1.0 1.0 1.0 1'
    cx = ORIGIN_X + room / 2
    cy = ORIGIN_Y + room / 2
    models += box_sdf('wall_north',  cx,              ORIGIN_Y + room, WALL_H/2, room, wt,   WALL_H, wc, wc)
    models += box_sdf('wall_south',  cx,              ORIGIN_Y,        WALL_H/2, room, wt,   WALL_H, wc, wc)
    models += box_sdf('wall_east',   ORIGIN_X + room, cy,              WALL_H/2, wt,   room, WALL_H, wc, wc)
    models += box_sdf('wall_west',   ORIGIN_X,        cy,              WALL_H/2, wt,   room, WALL_H, wc, wc)

    # 障礙物（紅色）
    oc = '0.8 0.2 0.2 1'
    for i, (cx, cy, sw, sh) in enumerate(obstacles):
        models += box_sdf(f'obs_{i}', cx, cy, WALL_H/2, sw, sh, WALL_H, oc, oc)

    # 動態障礙物（藍色圓柱，VelocityControl + PosePublisher）
    if dynamics:
        for d in dynamics:
            models += dyn_model_sdf(
                d['name'],
                d['start'][0], d['start'][1],
                d['radius'], d['height'],
            )

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


def save_trajectories_yaml(dynamics, path):
    cfg = {'dynamic_obstacles': dynamics}
    with open(path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f'  YAML: {path}')


if __name__ == '__main__':
    base = Path(__file__).parent.parent
    maps_dir   = base / 'maps'
    worlds_dir = base / 'worlds'
    config_dir = base / 'config'
    maps_dir.mkdir(exist_ok=True)
    worlds_dir.mkdir(exist_ok=True)
    config_dir.mkdir(exist_ok=True)

    print(f'Generating {N_OBSTACLES} obstacles, {W*RESOLUTION:.0f}x{H*RESOLUTION:.0f}m room...')
    img, obstacles = generate()

    save_pgm(img,  maps_dir   / 'random_room.pgm')
    save_yaml(     maps_dir   / 'random_room.yaml', 'random_room.pgm')
    save_sdf(obstacles, worlds_dir / 'random_room.sdf')
    print(f'  Static obstacles: {len(obstacles)}')

    # 動態障礙物：在自由區放 N_DYNAMIC 個圓柱 + 直線軌跡
    print(f'Placing {N_DYNAMIC} dynamic obstacles...')
    dynamics = place_dynamic_obstacles(img, N_DYNAMIC)
    if len(dynamics) < N_DYNAMIC:
        print(f'  WARN: only placed {len(dynamics)}/{N_DYNAMIC} '
              f'(map too cluttered? lower N_OBSTACLES or N_DYNAMIC)')

    save_sdf(obstacles, worlds_dir / 'random_room_dynamic.sdf', dynamics=dynamics)
    save_trajectories_yaml(dynamics, config_dir / 'dynamic_trajectories.yaml')
    print(f'Done. Static: {len(obstacles)}  Dynamic: {len(dynamics)}')
