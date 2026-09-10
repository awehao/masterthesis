# 計畫：修正時間基準下的 Isaac seed 1 heading OFF／ON 配對

狀態：**規劃，尚未執行**。不調算法、不開批次。舊 Gazebo 與 Isaac 資料全數保留，不覆寫。

## 0. 先講結論：現有 ON 那趟不能當配對資料

`omni_bot_dynamic.launch.py` 只傳 `use_sim_time` 與 `trajectories_file` 給
`dynamic_obstacle_driver`，`mode` 因此取預設值 **`legacy`**。該模式在程式自身的
註解中明寫：

> legacy — 依回授式 ping-pong，**相位由節點啟動時刻與回授時序決定，沒有種子、
> 不可重現**；實測同一模擬時間下兩趟的障礙物位置差達 **2.43 m**。

`run_bigarena_isaac.sh` 也從未發布 `/case_start`。因此
`…seed1__tfix2_seed1_164722` 的障礙相位不可重現，**保留為功能驗證，不作為配對資料**
（§10.33 的定位不變）。

## 1. 必須先補的四件事

既有的 `scheduled` 模式已經實作完整（三角波弧長、`/case_start` 為相位零點、
另發布排程目標供事後核對），但 bigarena／Isaac 這條路徑沒有接上。

| # | 缺口 | 補法 |
|---|---|---|
| 1 | launch 未傳 `mode` | `omni_bot_dynamic.launch.py` 讀 `AMMR_OBSTACLE_MODE`（與 `isaac_dynamic.launch.py`／`gazebo_dynamic.launch.py` **同名環境變數**，避免兩個模擬器跑不同情境），預設仍 `legacy` 以免動到舊路徑 |
| 2 | 軌跡檔無相位欄位 | `dynamic_trajectories_bigarena_traffic.yaml` 沒有 `phase0_m`／`direction`，預設全為 0／+1 → **10 個障礙物在相位零點同時位於各自 start**。這是可重現但退化的初始條件，會系統性改變遭遇樣態。建議另存 **v3 資產**（新檔名），以固定種子產生每個障礙物的 `phase0_m` 並**寫死在檔案裡**，保持「遭遇由時序決定」而非全部同步出發。舊檔不動。 |
| 3 | 未發布 `/case_start` | 移植 `run_one_trial_gz.sh` 已驗證的協定：等訂閱者 → **只發一次**（連發會把相位重設多次，有效零點變成最後一則）→ 讀 `/dynamic_obstacles/phase_epoch` 確認驅動確實採用 → 未採用就中止 |
| 4 | 未錄製排程主題 | bag 加入 `/dynamic_obstacles/target`、`/dynamic_obstacles/phase_epoch`（`/model/dyn_obs_*/pose` 本輪已加） |

相位零點建議設在**目標發布的同一時刻**，使情境從任務起點看起來與啟動時序無關。

## 2. 固定設定與唯一變因

| 項目 | 值 | 兩趟 |
|---|---|---|
| 模型 | 帶臂 omni_bot（`use_arm:=true`，收納姿態） | 相同 |
| 相機 / GUI | `CAMERA=false`、headless、`RENDER_HZ=12`、1600×900 | 相同 |
| 時間基準 | `physics_dt = rendering_dt = 0.010 s`（啟動回讀比對） | 相同 |
| 控制參數 | `gmpc_params.yaml` 全部，含 `wheel_coupling=True` | 相同 |
| 到達判準 | `--arrive-tol 0.30`，`dist_goal <= arrive_tol` | 相同 |
| 起終點 | seed 1：(17.10, 14.80) → (8.12, 0.48) | 相同 |
| 情境 | v3 排程軌跡，`AMMR_OBSTACLE_MODE=scheduled`，`/case_start` 對齊目標發布 | 相同 |
| 任務時限 | 180 s 模擬時間 | 相同 |
| CPU | 8 執行緒、92 °C 中止線 | 相同 |
| **`HEADING`** | **0（OFF）／ 1（ON）** | **唯一變因** |

ON 趟的 `heading_weight 2.0`、`heading_lookahead_m 1.2`、`heading_rate_max 1.0`
沿用現值，不調整。

## 3. 配對有效性的驗收（跑完就檢查，未過就不進入比較）

**不只確認公式相同，要確認實際軌跡真的對齊。**

1. 兩趟的 `phase_epoch` 皆存在，且與各自的目標發布時刻差 < 0.1 s。
2. 以 `phase_epoch` 對齊、取**共同時間區間**，逐一比對 10 個
   `/model/dyn_obs_*/pose`：位置差的**中位 < 10 mm、最大 < 100 mm**
   （沿用情境 v2 在 Gazebo 的既有量級：中位 1.5–2.9 mm、最大 60.2 mm）。
   門檻**事前設定**，但 **100 mm 不是天然可忽略的誤差**：須逐一報告十個障礙物的
   中位／p95／最大值；若最大誤差出現在機器人與該障礙物近距遭遇的時段，**另外標記**。
3. 比對 `/model/dyn_obs_*/pose` 與 `/dynamic_obstacles/target`，確認實際位置有跟上
   排程，而不是兩趟都同樣地偏離排程。
4. 任一項不過 → 保留資料並標 `ABORTED.txt`，**先查原因**，不自動反覆重跑直到通過。

**OFF 單趟通過只能確認它自己符合排程**；兩趟是否構成有效配對，要等 ON 完成、
以共同 epoch 比對實際軌跡之後才能判定。

## 4. 計時與評估定義（沿用已定稿者）

- `arrival_time_s`：自**該目標第一個終點相符的 `/plan`** 起算（容差 0.35 m）。
- 「運動開始時間」＝首次非零 `/cmd_vel`，**另記，不取代任務起點**。
- 目標發布、模擬器收到、第一個 `/plan`、運動開始 **四個時刻分開記錄**。
- 重送同一目標不重設任務計時。

### 比較指標（不只看前進比例）

| 面向 | 指標 |
|---|---|
| 是否完成 | `stop_reason`、觸發當下 `dist_goal`（完整精度） |
| 時間 | `arrival_time_s`、`motion_elapsed_s` |
| 路徑代價 | **真值實際路徑長**（20 Hz 折線，會略低估）、實際／直線比 |
| 安全 | 動態與靜態間距的 min / p5 / 中位（中心到表面，未扣 0.300） |
| 命令平滑度 | `|wz_cmd|` p95/max、`|dwz/dt|`、**`|dvx/dt|` 與 `|dvy/dt|`**（全向底盤不可漏側向）、輪速需求飽和比例 |
| 轉動 | 累計絕對轉角、淨轉角、角速度變號次數（門檻 `|wz|>0.02`） |
| 朝向功能 | **\|車頭 − 行進方位\|** 的中位／p90／`<10°` 比例（分穩態與過渡段） |
| 運動組成 | 前進／倒退／側移主導（**不互斥**，門檻 0.05 m/s） |

「朝前走」的代價要一起看：路徑長、任務時間、命令平滑度、間距是否變差。

## 5. 已知殘餘變因與不確定性（須寫進結論）

1. **AMCL 的隨機性尚未控制或確認**：Nav2 AMCL 用 1000–5000 個粒子，設定檔中未見
   seed 參數，本輪也**未實測**兩趟的定位是否重現。這是待確認項，**不宣稱
   「定位雜訊必然不同」**。
2. **路徑差異的歸屬需分辨**：heading 開啟後若 Nav2 給出不同路徑，那**可能是方法的
   作用結果**（車頭朝向改變 → 感測與代價分布改變），不能一律當成干擾排除。
   §10.34 已證實路徑改道會直接驅動朝向與轉動量，兩個方向的因果都要考慮。
3. **RTF 與執行緒排程**不完全一致（本輪實測 0.73–0.94）。

因此**單一配對只能作為對照流程試跑**，不足以下效能優劣結論。

## 6. 執行順序

1. 補完第 1 節四件事（改 launch、產生 v3 資產、移植 `/case_start` 協定、加錄主題）。
2. 先跑 **OFF 一趟**，通過第 3 節驗收。
3. 再跑 **ON 一趟**，通過第 3 節驗收，並與 OFF 比對障礙軌跡。
4. 兩趟都有效 → 出配對報告，明列第 5 節殘餘變因。
5. **確認流程有效後**才談重複與多 seed。
6. Gazebo／Isaac 對照**另案處理**——那回答「跨模擬器移植差異」，與 heading ON／OFF
   回答的問題不同，不與本項同時進行，避免同時更換模擬器與控制方法。

## 7. 本計畫不做的事

不調 heading 參數、不加末段凍結／切線替換／權重衰減、不改 Nav2、不重跑 Gazebo
舊實驗、不覆寫任何既有資料目錄。


## 8. 實作完成紀錄（2026-09-10）

### v3 情境資產

`src/ammr_bringup/config/dynamic_trajectories_bigarena_traffic_v3.yaml`
（sha256 `2e32069f3786b10dd1bb0764…`），由
`src/ammr_bringup/scripts/make_bigarena_traffic_v3.py` 產生，
中繼資料 `..._v3_meta.json`。

**v3 是新情境，不是恢復舊 seed 1 的遭遇樣態。** 路線與速度沿用原軌跡檔，
只加入 `phase0_m` 與 `direction`。相位種子 **20260910**。

scheduled 與 legacy 的行為差異（已寫入資產檔頭）：
- **折返位置**：legacy 距端點 `REACH_TOL = 0.20 m` 前就換向且換向瞬間由回授時序
  決定；scheduled 走到端點才折返，**每端多掃 0.20 m**。
- **速度變化**：scheduled 在端點瞬間反向，速度跳變 2v。
- 掃掠區域與遭遇時序都不同，**v3 與 legacy 的結果不可互相比較**。

起始碰撞檢查（機器人半徑 0.300 + 餘裕 0.10 m，機器人起點 (17.10, 14.80)）：
十個障礙物皆一次抽中、無重疊；離機器人最近 2.832 m，離靜態幾何最近 0.128 m，
彼此不重疊。逐一相位與位置見 `_meta.json`。

### 相位位置在任務開始前就放好

`isaac_bigarena_sim.py` 新增 `--mover-phase-yaml`：**生成時**就把移動體放到相位 0
的位置（公式與 `dynamic_obstacle_driver._schedule` 的 t = 0 情形相同）。
驅動節點在 `/case_start` 之前送零速度，因此障礙物在任務開始前已定位且靜止，
**不會在發目標當下被追蹤增益猛烈搬移**。

### 接線

| 檔案 | 改動 |
|---|---|
| `omni_bot_dynamic.launch.py` | 讀 `AMMR_OBSTACLE_MODE` 與 `AMMR_TRAJ_FILE`（與另兩支 launch 同名），**預設仍 legacy／場景預設軌跡** |
| `run_bigarena_isaac.sh` | scheduled 時：相位 0 定位檢查 → **只發一次** `/case_start` → 讀 `phase_epoch` 確認採用（未採用重試 3 次後中止）→ 移動確認 → 發目標；並寫出 `phase_alignment.json` |
| `run_bigarena_isaac.sh` | 靜止檢查在 scheduled 下**不加** `--traffic`（此時障礙物本來就該靜止） |
| bag 主題 | 加錄 `/case_start`、`/dynamic_obstacles/{target,phase_epoch,ground_truth}` |
| `evaluation/case_start_check.py` | 新增，三段檢查各自可獨立執行並留 JSON |

### 相位與目標的對齊方式

現有驅動實作的 epoch **是收到 `/case_start` 當下的模擬時間**，
**不支援指定未來時刻**，因此不採用「共同未來時間作零點」。
流程為：確認定位 → 發 `/case_start` → 確認 epoch → 確認移動 → 發目標，
並記錄 `phase_epoch`、`first_goal_publish_sim_t` 與兩者差值
（`phase_alignment.json`）。**發目標前流逝的相位長度必須在兩趟間比對**，
不能只寫「設計上同時」。
