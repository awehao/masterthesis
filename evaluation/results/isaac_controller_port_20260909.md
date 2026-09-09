# Isaac Sim 接上第一階段控制器：端到端流程確認

日期：2026-09-09　Isaac Sim 6.0.1，headless，GPU 39–41 °C

## 一、移植邊界：只換模擬器

`evaluation/isaac_bench_sim.py` 取代 `gz sim` 與 `ros_gz_bridge`，其餘一律不動：
Nav2、AMCL、GMPC 控制器、`omni_drive_controller`、`scan_relay`、`dynamic_obstacle_driver`
都是第四次進度報告用的同一份節點。`src/ammr_bringup/launch/isaac_dynamic.launch.py`
是 `gazebo_dynamic.launch.py` 去掉 gz 三個節點（`gz sim`、`parameter_bridge`、`create`）後的版本，
其餘節點與參數逐行相同。`evaluation/run_one_trial_isaac.sh` 由 `run_one_trial.sh` 複製而來，
只改第 1 步。

Isaac 端供應的介面，與原本 ros_gz bridge 暴露的相同：

| Topic | 方向 | 說明 |
| --- | --- | --- |
| `/clock` | 出 | 模擬時間，100 Hz（物理步 10 ms） |
| `/cmd_vel` | 入 | 車體 twist → 底盤根部速度狀態，**每個物理步重寫**（Gate 1 的 `hold`） |
| `/scan_raw` | 出 | 360 條 PhysX raycast，10 Hz |
| `/model/<dyn>/cmd_vel` | 入 | kinematic 障礙物 twist |
| `/model/<dyn>/pose` | 出 | 障礙物真值位姿，20 Hz |
| `/model/ammr_base/pose` | 出 | **機器人真值位姿，20 Hz——gz 從來沒有發布過這個** |

## 二、明列的設定

| 項目 | 值 | 依據 |
| --- | --- | --- |
| 物理步長 | 10 ms | 世界檔 `<max_step_size>0.01</max_step_size>`，照抄非自選 |
| 即時率 | **節流至 RTF 1.0** | 世界檔 `<real_time_factor>1.0</real_time_factor>`。不節流可跑到 4.5×，但那樣控制器每個模擬秒拿到的 CPU 與 Gazebo 不同，就不是同條件比較 |
| 控制頻率 | GMPC 20 Hz（控制器自身參數，未改） | `gmpc_node` `control_frequency` |
| 速度施加 | 每物理步重寫（`hold`） | 對應 `gz-sim-velocity-control-system` 的行為 |
| 座標轉換 | `/cmd_vel` 是車體 twist，依當前 yaw 轉成世界軸再設定速度狀態 | 與 gz VelocityControl 一致 |
| 零摩擦 | 4 個輪 + 地面綁定 mu=0 | URDF 的 `<gazebo><mu1>0.0</mu1>` 不會被匯入器讀取；gz 端本來就是零摩擦，動力來自 VelocityControl 而非輪力矩 |
| 命令逾時 | **關閉（0）** | 對應 gz VelocityControl 無逾時。加 watchdog 是功能變更，不併入移植對照 |
| 掃描參數 | 360 條、±3.14159、增量 0.017501894、0.12–10.0 m、10 Hz | 增量取自實錄 gz 掃描（`gmpc_cbf__scan_seed1`），非推算 |

已知且**刻意保留**的既有不一致：launch 的 static TF 寫 `base_link→lidar_link` 0.19，
URDF 的 `lidar_joint` 是 0.11。兩邊完全相同，不會偏袒任一模擬器，故照原樣保留，僅記錄。

## 三、致動核對（ammr_base，與 Gate 1 的 omni_bot 不同車）

`ammr_base` 有 4 個可動輪關節（`omni_bot` 全 fixed），故單獨量測：

| 量 | 命令 | 實測 | 追蹤率 |
| --- | ---: | ---: | ---: |
| 平移 | 0.1500 m/s | 0.1499 | **99.9%** |
| 旋轉（速度狀態） | 0.3000 rad/s | 0.2999 | **100.0%** |
| 旋轉（yaw 差分） | 0.3000 rad/s | 0.2997 | **99.9%** |

自由空間直線測試另見：命令 0.200 m/s 時實測 0.1993–0.2023，直到 x≈1.30 前緣接觸 `obs_44`
（該箱體 x 起於 1.495）後被擋下——是真實碰撞，不是追蹤失效。

## 四、端到端流程：通過

`gmpc_cbf` seed 0，150 s：機器人由 (0,0) 導航至 (14.63, 13.79)，路徑 23.83 m，
截止時距目標 3.99 m。錄到的 topic 齊全（`/scan` 1468、`/plan` 41、`/amcl_pose` 187、
`/gmpc/*` 各 2144、4 個 `/model/dyn_obs_*/pose` 各 2933）。
**RTF 1.000**，27000 個物理步中只有 11 步落後。

## 五、兩項必須先講清楚的發現

### 1. `/odom` 不是狀態資料，是命令推算

`omni_drive_controller` 的 `/odom` 是**由 `/cmd_vel` dead-reckoning 得來**，它只訂閱 `/cmd_vel`。
Gazebo 沒有機器人真值可對照，所以這件事一直沒有被量到。Isaac 現在有真值，實測：

| 試驗 | 真值路徑 | `/odom` 路徑 | 差 |
| --- | ---: | ---: | ---: |
| 150 s（流程確認） | — | — | 位置差 中位 119.7 mm、p95 185.7 mm、最大 256.3 mm |
| 250 s | 45.72 m | 49.22 m | **`/odom` 高估 +7.6%** |

因此 §10.19 把 `path_length_m`、`tracking_rmse_m` 記為「取自 `/odom`（實測）」**需要更正**：
`/odom` 在這套系統裡是命令推算量。依照「任務結果與實際路徑使用狀態資料」的要求，
Isaac 的任務結果應改用 `/model/ammr_base/pose` 真值；命令平滑度仍取自 `/cmd_vel`。

### 2. 250 s 的原始案例與 Gazebo 基線結果不同，原因尚未確定

同一組設定跑 250 s 時，t≈170–180 s 後真值與 `/odom`／AMCL 大幅分歧：
真值最終停在 (10.38, 0.29)，AMCL 卻報 (15.79, 16.44)（接近目標）。真值在 t=180 後**倒退**
（14.10 → 9.13 → 6.82）。

已排除的原因：**不是底盤執行問題**。全程真值/命令速度比中位 0.997
（t 90–160 為 0.997、160–200 為 1.000、200–292 為 0.994），旋轉與平移追蹤率皆 ≈100%。
p05 為 0.35，代表期間確有被擋住的接觸事件。

尚未確定的是：這是本來就存在的定位發散（Gazebo 同案例也會發生）、
還是 Isaac 與 Gazebo 的接觸行為差異所引發。**在與 Gazebo 同案例對照之前，
不能說移植重現了基線，也不能說 Isaac 表現較差。**

## 六、原始資料

- `evaluation/bags/isaac_gmpc_cbf__seed0`（250 s）、`isaac_gmpc_cbf__pipecheck150`（150 s）
- `evaluation/results/isaac_bench_isaac_gmpc_cbf__seed0.json`（真值軌跡、掃描統計、RTF）
- `evaluation/results/isaac_bench_pipecheck150.json`
- 程式：`evaluation/isaac_bench_sim.py`、`isaac_common.py`、`run_one_trial_isaac.sh`、
  `src/ammr_bringup/launch/isaac_dynamic.launch.py`
