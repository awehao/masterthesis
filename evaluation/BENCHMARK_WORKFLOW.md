# Benchmark 完整流程 — RPP vs MPPI Omni vs GMPC

> 3 個 baseline × 1 個 goal × 1 次(階段 1)。確認 pipeline 全通後可以擴成 ×3 次 / 多 goals。

## 0. 一次性準備

```bash
cd ~/masterthesis
source install/setup.bash
mkdir -p evaluation/bags evaluation/results

# 清空舊測試資料(可選)
rm -rf evaluation/results/figs/SAMPLE evaluation/results/SAMPLE_runs.csv
```

確認沒殭屍 process:

```bash
ros2 node list   # 必須空,如果有就 pkill 殺光再開
```

---

## 1. Baseline A:NavFn + RPP

### T1 — Gazebo
```bash
cd ~/masterthesis && source install/setup.bash
ros2 launch ammr_bringup gazebo.launch.py
```
等機器人 spawn 在 (0, 0)。

### T2 — RPP nav2 stack
```bash
cd ~/masterthesis && source install/setup.bash
ros2 launch ammr_navigation nav2.launch.py
```
等看到 `Managed nodes are active`(~5 秒)。

### T3 — 錄製(120 秒上限)
```bash
cd ~/masterthesis/evaluation
./record.sh rpp corner_goal 150
```
等 `Recording for 120 s into ...` 出現後**再等 3 秒**確保 record 真的啟動。

### T4 — 送 goal
```bash
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 17.0, y: 17.0}, orientation: {w: 1.0}}}" 
```

### 等待 + 清場
等機器人到達 / 或 120 秒到 → T3 自動結束。
**全部 Ctrl+C**:T3 → T2 → T1。

---

## 2. Baseline B:Smac + MPPI Omni

完全 fresh 重啟。

### T1
```bash
ros2 launch ammr_bringup gazebo.launch.py
```

### T2 — MPPI nav2 stack
```bash
ros2 launch ammr_navigation nav2_omni_mppi.launch.py
```

### T3
```bash
cd ~/masterthesis/evaluation
./record.sh mppi corner_goal 120
```

### T4 — **同一個 goal**
```bash
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 17.0, y: 17.0}, orientation: {w: 1.0}}}" --once
```

清場、Ctrl+C。

---

## 3. Baseline C:SE(2) GMPC(我們的)

完全 fresh 重啟。

### T1
```bash
ros2 launch ammr_bringup gazebo.launch.py
```

### T2 — GMPC stack
```bash
ros2 launch ammr_wholebody_mpc gmpc_nav2.launch.py
```

### T3
```bash
cd ~/masterthesis/evaluation
./record.sh gmpc corner_goal 120
```

### T4 — **同一個 goal**
```bash
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 17.0, y: 17.0}, orientation: {w: 1.0}}}" --once
```

清場、Ctrl+C。

---

## 4. 分析 3 個 bag

```bash
cd ~/masterthesis/evaluation
python3 analyze.py bags/rpp__corner_goal  --method rpp  --run corner_goal
python3 analyze.py bags/mppi__corner_goal --method mppi --run corner_goal
python3 analyze.py bags/gmpc__corner_goal --method gmpc --run corner_goal
```

每次會印指標摘要,並 append 一行到 `results/runs.csv`。

## 5. 出圖出表

```bash
python3 plot.py
```

產出:
- `results/figs/summary.png` ← **報告主圖**(9 metrics × 3 methods bar chart)
- `results/figs/<metric>.png` × 9 ← 個別 metric
- stdout 印 mean ± std 對比表

---

## Baseline 對應表

| Baseline | T1(都一樣) | T2 launch | Params 檔 | Planner | Controller |
|----------|-----------|-----------|-----------|---------|-----------|
| A: RPP | `gazebo.launch.py` | `ammr_navigation/nav2.launch.py` | `nav2_params.yaml` | **NavFn** | RPP(輸出 vx+wz) |
| B: MPPI | `gazebo.launch.py` | `ammr_navigation/nav2_omni_mppi.launch.py` | `nav2_params_mppi.yaml` | **SmacPlanner2D** | MPPI Omni |
| C: GMPC | `gazebo.launch.py` | `ammr_wholebody_mpc/gmpc_nav2.launch.py` | `nav2_params_mppi.yaml`(planner) + `gmpc_params.yaml`(controller) | **SmacPlanner2D** | **我們的 GMPC node** |

**對比的解讀**:
- **A → B**:legacy stack → modern stack(planner + controller 都換)
- **B → C**:**乾淨的 controller 對比**(同 SmacPlanner2D,只差 controller)← 論文重點

---

## 常見問題自救

**Q: 殭屍 process 卡住怎辦**
```bash
pkill -9 -f gmpc_node
pkill -9 -f goal_to_plan_relay
pkill -9 -f test_path_publisher
pkill -9 -f lifecycle_manager_navigation
pkill -9 -f planner_server
pkill -9 -f amcl
pkill -9 -f map_server
pkill -9 -f static_transform_publisher
pkill -9 -f ros_gz_bridge
pkill -9 -f "gz sim"
pkill -9 -f gazebo
sleep 3
ros2 daemon stop && ros2 daemon start
sleep 2
ros2 node list   # 應該空
```

**Q: T3 record 還沒開始 T4 就送 goal 了**
→ 重來。record 開始後**先等 3-5 秒**再發 goal。

**Q: 機器人沒動 / TF map→base_footprint 卡死**
→ AMCL 沒 converge。診斷:
```bash
ros2 run tf2_ros tf2_echo map base_footprint
```
跑 5 秒看 Translation 數字有沒有變。卡住 → Ctrl+C 全部殺光重來。

**Q: planner 回 "Goal occupied" / "no valid path found"**
→ (17, 17) 在隨機地圖被障礙物擋。改 (16, 16)、(17, 15)、或用 Foxglove 點 free space。**3 個 baseline 必須用同一個新 goal**(否則公平性破壞)。

**Q: 跑到一半想停**
→ T3 Ctrl+C 停 record,T2/T1 也 Ctrl+C。bag 已存,可以直接 analyze。

**Q: AMCL 不用每次拉 initialpose 嗎?**
→ 不用。`nav2_params_mppi.yaml` 裡 `set_initial_pose: true, initial_pose: {x:0, y:0, z:0, yaw:0}`,啟動會自動 init。**前提是 Gazebo robot 也剛 spawn 在 (0, 0)** — 所以每個 baseline 都要 T1 fresh 重啟。

---

## 跑完一輪要回報的資訊

每個 baseline 跑完,確認:
1. T2 是否有 `Path published: NN poses`(planner 成功?)
2. Gazebo 機器人是否真的開到 (17, 17) 附近
3. bag 是否有錄到資料:
   ```bash
   ls -lh evaluation/bags/<method>__corner_goal/
   ```
   應該有 `metadata.yaml` 和一個或多個 `.db3` 檔(總共幾 MB)

---

## 擴大規模(階段 2 / 3)

確認階段 1 全通後可以擴:

| 階段 | 跑什麼 | 工時 |
|------|--------|------|
| 1 | 1 goal × 3 methods × 1 run = 3 bags | ~30 分鐘 |
| 2 | 1 goal × 3 methods × 3 runs = 9 bags | ~90 分鐘 |
| 3 | 3 goals × 3 methods × 3 runs = 27 bags | 半天 |

階段 2 才會有 error bar(σ),階段 3 才能說「在 3 個場景下皆 X」。
報告強度 1 < 2 < 3,但 1 已經能說「我有 baseline 對比框架」。

---

## 一鍵 cheatsheet(複製貼上版)

```bash
# === RPP ===
# T1: ros2 launch ammr_bringup gazebo.launch.py
# T2: ros2 launch ammr_navigation nav2.launch.py
# T3:
cd ~/masterthesis/evaluation && ./record.sh rpp corner_goal 120
# T4:
ros2 topic pub /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: 17.0, y: 17.0}, orientation: {w: 1.0}}}" --once

# (清場 Ctrl+C T3 → T2 → T1)

# === MPPI ===
# T1: ros2 launch ammr_bringup gazebo.launch.py
# T2: ros2 launch ammr_navigation nav2_omni_mppi.launch.py
# T3:
cd ~/masterthesis/evaluation && ./record.sh mppi corner_goal 120
# T4: 同上 goal

# (清場)

# === GMPC ===
# T1: ros2 launch ammr_bringup gazebo.launch.py
# T2: ros2 launch ammr_wholebody_mpc gmpc_nav2.launch.py
# T3:
cd ~/masterthesis/evaluation && ./record.sh gmpc corner_goal 120
# T4: 同上 goal

# === 分析 ===
cd ~/masterthesis/evaluation
python3 analyze.py bags/rpp__corner_goal  --method rpp  --run corner_goal
python3 analyze.py bags/mppi__corner_goal --method mppi --run corner_goal
python3 analyze.py bags/gmpc__corner_goal --method gmpc --run corner_goal
python3 plot.py
```
