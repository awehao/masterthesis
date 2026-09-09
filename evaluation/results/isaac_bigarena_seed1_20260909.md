# Isaac Sim：bigarena ＋ 帶臂 omni_bot，seed 1 完整單案例

日期：2026-09-09　n=1。**這是帶臂平台在 Isaac 的單案例完成驗證，
不是第四次進度報告純底盤結果的重現，也尚未與 Gazebo 對照。**

## 一、凍結的設定

| 項目 | 值 |
| --- | --- |
| 模擬器 | Isaac Sim 6.0.1，**headless**，`limit_cpu_threads=8` |
| 世界／地圖 | `bigarena.sdf` / `bigarena.yaml`（52 模型：16 牆、18 known_obs、7 unknown_obs、10 動態） |
| 機器人 | `omni_bot.urdf.xacro` 展開，`use_arm:=true add_gripper:=true add_arm_camera:=true` |
| 手臂姿態 | `arm_initial_pose.yaml` 頂層六軸全 0，位置驅動 kp=1e5、kd=1e4 保持 |
| 方法 | `gmpc_scan`（`obstacle_source:=scan`，真實感知） |
| 障礙軌跡 | `TRAJ=bigarena_traffic`（10 個移動體，**legacy 回授式 ping-pong**，相位未固定） |
| 起訖 | `POSES_CSV=evaluation/results/bigarena_poses.csv`，**seed 1**，(17.10, 14.80) → (8.12, 0.48)，直線 16.90 m |
| 導航鏈 | `omni_bot_dynamic.launch.py` ＋ `NO_GZ=1`（只跳過 gz sim、ros_gz bridge、model spawn） |
| 命令鏈 | `gmpc → /cmd_vel_nav → velocity_smoother → /cmd_vel`（`SHIELD` 未開） |
| 位姿來源 | 控制器 `pose_source=tf`（`map→base_footprint`）；EKF 融合 `/odom`＋`/amcl_pose` |
| 雷射 | PhysX raycast，排除**兩個**形狀（見第四節） |
| 物理／即時率 | 10 ms 步長，節流目標 RTF 1.0 |
| 溫度中止線 | 88 °C（k10temp），獨立程序牆鐘每秒取樣 |

## 二、結果

| 項目 | 值 |
| --- | --- |
| **停止原因** | **`goal_reached_truth`** |
| **任務時間** | **60.3 s 模擬時間 / 79 s 牆鐘**（自 Isaac 收到 `/goal_pose` 起算） |
| **RTF（實測）** | **0.811 —— 未維持即時速度**，故模擬與牆鐘時間必須分別列出 |
| 真值終點誤差 | **0.249 m**（觸發當下；容差 0.25 m，餘裕僅 1 mm） |
| 真值路徑長 | 18.29 m（直線 16.90 m），位移 16.73 m |
| `/plan` | 64 筆；非零命令取樣 1189/2610（45.6%） |
| `\|truth − odom\|` | 中位 **0.0 mm**，p95 3.6 mm，最大 5.4 mm |
| `\|truth − amcl\|` | 中位 **39.5 mm**，p95 107.8 mm，最大 119.9 mm |
| 手臂保持誤差 | 最大 **0.737 mrad = 0.0422°**（joint2；各軸中位≈p95≈最大，是穩態靜差） |
| 底盤姿態 | \|roll\|max 0.00000°、\|pitch\|max 0.00005° |
| 溫度 | 平均 **77.4 °C**、峰值 **80 °C**，未觸線（520 筆牆鐘取樣） |

## 三、三處必須分清的界定

### 1. `goal_reached_truth` 是**測試器**的完成判定，不是 GMPC 自行判定完成

停止觸發前六個控制週期，控制器以自己的位姿算出的距目標是
0.345 → 0.339 → 0.326 → 0.312 → 0.297 → **0.290 m**，單調縮短。
**控制器全程最小距目標 0.290 m，從未進入自身的 `goal_tolerance_xy = 0.20 m`。**
測試器在真值進入 0.25 m 時停止，此時控制器仍在朝目標移動、未宣告完成。

停止當下控制器狀態：位姿來源 TF（診斷欄位 20 = 0），`|TF位姿 − EKF位姿|` = 0.0 mm。

### 2. 0.400 m 是與**移動體**的**表面淨距**，不是無碰撞保證

該值為真值位置到各 `dyn_obs` 真值位置的距離**減去該移動體半徑**，
即機器人**中心**到移動體**表面**的淨距，而非機器人表面到障礙表面。
底盤碰撞半徑 0.300 m，故對應的表面間距約 0.100 m。

**這個數字只涵蓋 10 個移動體**，不包含 16 面牆、18 個 known_obs、7 個 unknown_obs。
本輪**沒有量測對牆面與靜物的間距**，不能延伸為全場無碰撞。

### 3. RTF 0.811：未維持即時速度

60.3 s 是**模擬時間**；對應牆鐘 79 s。控制器在牆鐘上執行，模擬比即時慢約 19%，
等於每個模擬秒獲得的計算時間多於即時情況。與 Gazebo 對照時必須處理這項差異。

## 四、雷射近似（沿用並記錄）

PhysX raycast 在查詢時排除**兩個**形狀，物理碰撞不變：

1. `base_link/cylinder` —— 底盤碰撞圓柱（r=0.300、z 0.050–0.330 的保守近似，把雷射包住）
2. `lidar_link/cylinder_1` —— 雷射自身外殼（r=0.036，光束自其內部起跑，PhysX 回報距離 0）

命中被排除形狀的光束**不會變成 inf**，而是以 `raycast_all` 取整條射線的完整命中集合、
挑最近的非排除物件，因此不會漏掉其後方的環境（已用 0.02 m 厚薄板於 0.800 m 驗證，量到 0.7900 m）。

**目標是遮罩後 `/scan` 可比；`/scan_raw` 並不等價**——gz 的 `/scan_raw` 在四扇區有 34 條
0.2449–0.2633 m 的真實柱子回波，Isaac 這邊穿過去了。四扇區遮罩由既有 `scan_relay` 原樣套用
（中心 45/135/225/315°、半寬 10°）。手臂遮擋能力**未實測**（收納姿態下手臂最低件在雷射上方 0.24 m）。

## 五、與前一趟的差距：是比較錯誤，不是效能改善

前一趟（同 seed、同設定）我報「140.7 s 仍距目標 0.27 m」，這是**錯的比較**：
140.7 s 是 bag 的**總模擬時間**，含目標發布前約 77 s 待機，不是任務時間。

| | 前一趟 | 本趟 |
| --- | --- | --- |
| 任務時間（模擬） | 約 63 s（由 RTF 0.782 重建，**bag 已刪除**） | **60.3 s**（直接量測） |
| 任務時間（牆鐘） | 81 s | 79 s |
| 結束原因 | 錄製 180 s 牆鐘逾時觸發 cleanup | 真值抵達 |

**兩趟任務時間幾乎相同。** 前一趟是在距目標 0.27 m、仍在逼近時被切斷，離抵達僅一兩秒。
差異來自計時方式（錄製牆鐘 vs 任務模擬時間），不是控制器表現改變。

## 六、本輪修正的測試流程缺陷

- **計時拆分**：錄製先開始並涵蓋啟動與就緒檢查；任務時限改為**模擬時間**、自 Isaac 收到
  `/goal_pose` 起算；牆鐘逾時另設，只處理程序卡住
- **`/goal_pose` 重複發布不重設起點**：本趟收到 5 則，只採第一則（模擬時間 70.16 s）。
  訊息時間戳為 0.00（`ros2 topic pub` 未填 header），故**起點是 Isaac 收到時的模擬時間**
- **停止時保存兩筆樣本**：`at_trigger`（觸發當下，**到達判定與完成時間用這筆**）與
  `after_stop`（歸零並步進 20 步後，僅描述停機）。本趟兩者相同（0.249 m）
- **停機順序**：先歸零並步進生效 → 保存 → 才停止錄製與清理。前一趟 JSON 的
  `stop_reason` 是 `None`，就是被 cleanup 截斷所致

## 七、原始資料

- `evaluation/bags/isaac_bigarena_gmpc_scan__seed1`
- `evaluation/results/isaac_bigarena_gmpc_scan__seed1.json`（含 `at_trigger`／`after_stop`）
- `evaluation/results/isaac_bigarena_gmpc_scan__seed1_thermal.csv`（牆鐘每秒溫度）
- `evaluation/results/isaac_bigarena_seed1_platform.json`（前一趟的手臂／姿態離線補算）
- 程式：`isaac_bigarena_sim.py`、`isaac_common.py`、`tf_ready_check.py`、
  `thermal_sampler.sh`、`run_bigarena_isaac.sh`、`omni_bot_dynamic.launch.py`（`NO_GZ` 開關）

## 八、下一步的對照條件

Gazebo 同案例對照要沿用：**同一帶臂模型與展開參數、同一手臂姿態、同一 seed 與起訖點、
同一完成判準（真值 0.25 m，並同時記錄控制器自身判定）**。兩項差異需先處理：

1. **雷射近似**：Isaac 排除兩個形狀後穿過四根柱子，Gazebo 會真的看到它們（34 條回波）。
   遮罩後的 `/scan` 可比，`/scan_raw` 不可比
2. **計算時間條件**：Isaac 本趟 RTF 0.811，Gazebo 需量測其 RTF；若不同，控制器每模擬秒
   取得的計算時間不同，須列為差異或設法對齊
