# Isaac 模擬時間基準：缺陷、修正與驗收（2026-09-10）

## 缺陷

`isaac_bigarena_sim.py` 主迴圈以 `t = k * physics_dt` 記帳模擬時間，`/clock`、
所有感測器時戳、任務計時與真值 log 都由這個值產生。世界建立時
`rendering_dt = physics_dt * 4`，而**帶 `render=True` 的 `world.step()` 會推進
`rendering_dt / physics_dt` 個物理子步，主迴圈卻只記帳一步**。

因此模擬時間跑快了，倍率 = `(re_ − 1 + m) / re_`，其中 `re_` 是每幾步 render 一次、
`m = rendering_dt / physics_dt`。這不只影響報表：控制器、Nav2、逾時判斷與
加速度限制全部跑在這份 `/clock` 上。

## 倍率並非固定

以每趟自己的資料量測（真值位移 ÷ 命令積分）：

| 趟次類型 | render_hz | 量測倍率 |
|---|---|---|
| `cam1600_jpeg` | 10 | 1.300 |
| bigarena seed 1、朝向三趟 | 12 | 1.368–1.378 |
| `cam_on_seed1`（相機 tick 額外觸發 render） | 12 + cam 10 | 1.600 |
| `view30_1600` | 30 | 1.999 |

相機 tick 會強制 render（`_did_render = k % re_ == 0 or k % ce == 0`），所以開相機的
趟次倍率更高。**不可用單一倍率換算所有舊資料。**

## 一次錯誤的中間結論

第一版驗收在同一個行程內掃描 `rendering_dt`，得到「mult=1 與 mult=4 結果相同，
所以 `rendering_dt` 不是原因」。**該結論無效**：日誌顯示
`SimulationContext is already initialized. Constructor parameters are ignored on
subsequent calls`，第二個設定根本沒生效。改為每個設定獨立行程、並回讀
`get_physics_dt()` / `get_rendering_dt()` 確認後，結論相反。

## 驗收一：獨立無接觸場景（`evaluation/isaac_time_audit.py`）

單一剛體、重力關閉、無接觸，每步覆寫速度，所以位移 ÷ 命令 就是實際物理時間。
12 個設定各自獨立行程，參數皆回讀確認。命令 vx=0.20 m/s、wz=0.50 rad/s、200 步。

| rendering_dt | render_hz=0（不渲染） | 5 | 12 | 30 |
|---|---|---|---|---|
| `= 4 × physics_dt` | 1.0000 | 1.1500 | 1.3750 | 2.0050 |
| `= 2 × physics_dt` | 1.0000 | 1.0500 | 1.1250 | 1.3350 |
| `= physics_dt` | 1.0000 | **1.0000** | **1.0000** | **1.0000** |

量測倍率與 `(re_ − 1 + m) / re_` 在所有設定下吻合。`world.current_time` 在全部
12 個設定中都與位移反推的物理時間完全相同，**是可信的時間權威**。

**命令生效步**：只在 `k=5` 施加 vx=0.20，其餘為零 → 移動只發生在 `k=5`，
位移 2.0000 mm = 0.20 × 0.01，恰好一個物理步。命令在**當次 `world.step()` 生效**，
不跨步、不延遲。

## 修正

1. `rendering_dt = physics_dt`，並在啟動時回讀比對，不符即以退出碼 4 中止。
2. 模擬時間一律 `world.current_time` 讀取，不再由迴圈次數推算；`/clock`
   在步進**之後**以該值發布（原本發布的是 `t + dt`，迴圈自己的猜測）。
3. scan / pose / odom / render / camera 的節奏改以**模擬時間**到期判定，
   不再用 `k % N`；`--publish-s` 前置路徑同樣處理。
4. 記錄 `physics_dt`、`rendering_dt`、`time_skew_max_s`（迴圈計數時鐘與物理時鐘的
   最大偏差）——這三項先前完全沒有被記錄。

## 驗收二：修正後的真實模擬器

`--empty-world true --self-drive-goal 0.20`，三個 render 頻率：

| render_hz | rendering_dt 回讀 | 迴圈/物理時鐘偏差 | 位移倍率 |
|---|---|---|---|
| 5 | 0.01 | 3.0e-7 s | 0.99947 |
| 12 | 0.01 | 3.0e-7 s | 0.99947 |
| 30 | 0.01 | 3.0e-7 s | 0.99947 |

（修正前同樣三個頻率為 1.150 / 1.375 / 2.005。）

## 驗收三：`/clock` 本身

`evaluation/clock_probe.py` 直接訂閱 `/clock` 與 `/model/omni_bot/pose` 並自寫 CSV
（rosbag2 被中止時留下 0 位元組的 mcap、不寫 metadata，不可靠）。

- render_hz=5：`/clock` 前進 1.5000 s，位姿時戳前進 1.5000 s，**兩者差 0.000000 s**；
  同一則位姿上 `|/clock − 位姿時戳|` 最大 0.000 ms
- render_hz=30：同樣 **0.000000 s**、最大 0.000 ms
- 依 `/clock` 算出的速度 0.19989 / 0.19990 m/s（命令 0.20），比值 **0.99947 / 0.99948**

## 三個時刻分開記錄

新增欄位，互不取代：

| 欄位 | 意義 |
|---|---|
| `goal_stamp` | 目標訊息自身的 header 時戳 |
| `goal_sim_t` | 模擬器 callback 實際執行時的模擬時間 |
| `first_plan_sim_t` | 目標後第一個 `/plan`；**`arrival_time_s` 自此起算** |
| `motion_start_sim_t` | 首次非零 `/cmd_vel`，即「運動開始時間」 |
| `goal_cb_lag_s` | 回呼延遲，保留而非折抵 |
| `motion_elapsed_s` | 運動開始 → 停止觸發 |

「運動開始時間」**不是**任務起點，只是另一個獨立記錄的時刻。

## 尚未完成

修正後尚未重跑任何導航實驗。舊資料一律標記「時間基準待核正」
（各趟資料目錄下 `TIME_BASIS_PENDING.md`，各紀錄文件開頭橫幅）。
