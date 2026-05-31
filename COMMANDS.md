# AMMR 常用指令

## 每次開始前

```bash
cd ~/masterthesis
source install/setup.bash
```

---

## Foxglove 連線

```bash
# 安裝（只需要一次）
sudo apt install ros-jazzy-foxglove-bridge

# 每次啟動（另開 terminal）
source /opt/ros/jazzy/setup.bash
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```

Foxglove → Open Connection → Foxglove WebSocket → `ws://localhost:8765`

---

## Build

```bash
# 指定套件 build
colcon build --packages-select ammr_bringup ammr_navigation ammr_simulation
source install/setup.bash

# 全部 build
colcon build
source install/setup.bash
```

---

## 啟動順序

### Stage 0 — NavFn + RPP Baseline（原始）

```bash
# Terminal 1：一鍵完整啟動（Gazebo → Nav2 → RViz）
cd ~/masterthesis && source install/setup.bash
ros2 launch ammr_bringup bringup.launch.py
```

### Stage 2 — SmacPlanner2D + MPPI Omni（修正版）

```bash
# Terminal 1：Gazebo
cd ~/masterthesis && source install/setup.bash
ros2 launch ammr_bringup gazebo.launch.py

# Terminal 2：Nav2（等 Gazebo 機器人 spawn 後）
cd ~/masterthesis && source install/setup.bash
ros2 launch ammr_navigation nav2_omni_mppi.launch.py
```

---

## 送 Goal 觸發 Planner

```bash
# 用 topic pub 送一個 map frame goal
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map', stamp: {sec: 0}}, \
    pose: {position: {x: 2.0, y: 2.0, z: 0.0}, \
           orientation: {w: 1.0}}}" --once
```

---

## Foxglove 驗證

| Topic | 用途 |
|-------|------|
| `/plan` | Global path（有 goal 後才出現） |
| `/trajectories` | MPPI 取樣軌跡 |
| `/global_costmap/costmap` | 全局 costmap |
| `/local_costmap/costmap` | 局部 costmap |
| `/cmd_vel` | 底盤速度指令 |

---

## 驗證指令

```bash
# 確認 /plan 有資料
ros2 topic echo /plan --once

# 確認 MPPI 使用 Omni model
ros2 param get /controller_server FollowPath.motion_model

# 確認 cmd_vel 有 linear.y（側移速度）
ros2 topic echo /cmd_vel

# 監看 Nav2 實際輸出（velocity_smoother 前）
ros2 topic echo /cmd_vel_nav

# Nav2 lifecycle 狀態
ros2 lifecycle list

# 所有 topic
ros2 topic list
```

---

## 手動測試底盤運動

```bash
# 前進
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {z: 0.0}}" --rate 10

# 側移（測試 Omni vy）
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.15, z: 0.0}, angular: {z: 0.0}}" --rate 10

# 斜向移動（前進 + 側移）
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.15, z: 0.0}, angular: {z: 0.0}}" --rate 10

# 原地旋轉
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {z: 0.5}}" --rate 10
```

---

## 鍵盤控制

```bash
ros2 run ammr_bringup teleop
```

```
W/↑  前進    S/↓  後退
A/←  左轉    D/→  右轉
```

---

## Debug

```bash
ros2 topic echo /odom               # odom 資料
ros2 topic echo /scan --no-arr      # laser scan
ros2 topic hz /cmd_vel              # cmd_vel 頻率
ros2 topic hz /odom                 # odom 頻率
ros2 param dump /controller_server  # 所有 controller 參數
ros2 param dump /velocity_smoother  # 所有 velocity_smoother 參數
```

---

## 重新生成地圖

1. 編輯 `src/ammr_bringup/scripts/generate_map.py`
   - `SEED = 42` — 改數字換布局
   - `N_OBSTACLES = 35` — 障礙物數量
   - `W, H = 400, 400` — 地圖大小（1px = 0.05m）
2. 執行生成腳本：
   ```bash
   cd ~/masterthesis/src/ammr_bringup && python3 scripts/generate_map.py
   ```
3. Build 並重啟：
   ```bash
   cd ~/masterthesis
   colcon build --packages-select ammr_bringup
   source install/setup.bash
   ```

---

## 設定檔位置

| 用途 | 路徑 |
|------|------|
| Stage 0 Nav2 params（NavFn + RPP） | `src/ammr_navigation/config/nav2_params.yaml` |
| Stage 2 Nav2 params（SmacPlanner2D + MPPI Omni） | `src/ammr_navigation/config/nav2_params_mppi.yaml` |
| MPPI controller 設定 | `src/ammr_navigation/config/controllers/mppi.yaml` |
| RPP controller 設定 | `src/ammr_navigation/config/controllers/rpp.yaml` |
| Stage 2 launch | `src/ammr_navigation/launch/nav2_omni_mppi.launch.py` |
| Omni wheel 運動學 | `src/ammr_bringup/ammr_bringup/omni_drive_controller.py` |
