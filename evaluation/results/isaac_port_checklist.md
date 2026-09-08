# Isaac Sim 移植準備：介面清單與驗收表

日期：2026-09-09。**Isaac Sim 尚未安裝於本機**（`omni` 匯入失敗、無常見安裝路徑），本檔為
不需要 Isaac 的前置整理。介面資料**由執行中的 Gazebo 系統實測**，非憑文件抄寫。

## 一、要移植的東西 vs 不動的東西

| 層 | 檔案／模組 | 移植時是否更動 |
| --- | --- | --- |
| 幾何模型 | `urdf/omni_bot.urdf.xacro`、`chassis_body.xacro`、`lite6_*.xacro` | 需轉為 USD；**幾何數值不得改** |
| 場景 | `worlds/arm_barrier_test.sdf`（`obs_0` 目標、`obs_1` 底盤障礙） | 需重建為 USD 場景，**位姿與尺寸照抄** |
| 安全層 | `ammr_wholebody_mpc`（距離節點、屏障濾波器） | **不動**，只要 topic 對齊 |
| 控制器 | `evaluation/wholebody_pregrasp.py` | **不動** |
| 命令橋接 | `arm_vel_adapter.py`、`arm_vel_gate.py`、`base_tf_bridge.py` | 需改為 Isaac 的命令／里程計介面 |
| 啟動 | `barrier_stack.py`、`arm_barrier_test.launch.py` | 需新版本 |

## 二、介面對照（實測）

### Topic

| Topic | 型別 | 方向 | 備註 |
| --- | --- | --- | --- |
| `/wholebody_safety/cmd_in` | `std_msgs/Float64MultiArray`(9) | 控制器 → 安全層 | 底盤 3 為**世界座標** |
| `/wholebody_safety/cmd_out` | `std_msgs/Float64MultiArray`(9) | 安全層 → 轉接器 | 同上 |
| `/wb_vel_cmd` | `std_msgs/Float64MultiArray`(9) | 轉接器 → 閘門 | 底盤 3 已轉為**車體座標** |
| `/lite6_vel_controller/commands` | `std_msgs/Float64MultiArray`(6) | 閘門 → 手臂 | 唯一發布端＝閘門 |
| `/cmd_vel` | `geometry_msgs/Twist` | 閘門 → 底盤 | 唯一發布端＝閘門 |
| `/joint_states` | `sensor_msgs/JointState` | 模擬 → 全體 | 實測 ~143 Hz |
| `/odom` | `nav_msgs/Odometry` | 模擬 → TF 橋接 | Gazebo 為**真值導出** |
| `/arm_link_distance/points` | `sensor_msgs/PointCloud2` | 距離節點 → 安全層 | 15 欄，見 `arm_link_distance.FIELDS` |
| `/wholebody_safety/diag` | `std_msgs/Float32MultiArray`(19) | 診斷 | 欄位 1 = reason |

### 關節（控制器自報的順序，**必須逐一核對，不可假設**）

`['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']`

### Frame（實測 world z）

| frame | world z | 意義 |
| --- | ---: | --- |
| `world` | 0.0000 | 慣性 |
| `base_footprint` | **0.0000** | **著地面**，模型根 |
| `base_link` | +0.0500 | ＝ `wheel_r` |
| `link_base` | +0.3300 | 手臂底座 |
| `link_tcp` | +0.5498 | 全零姿態下的工具 |

**`MODEL_Z = 0.0`**：運動學模型的根與世界原點重合。移植後**必須重新量測**——若 Isaac 的生成
高度不同，這個常數、生成 z、`base_tf_bridge --z` 三處要一起改（Gazebo 上這三處曾各帶一個
互相吻合的 0.05，錯誤因此隱形）。

## 三、移植驗收表

按順序做，前一項不過不進下一項。

### 第 1 關：關節命令

| 檢查 | 判準 |
| --- | --- |
| 關節名稱與順序取自控制器本身 | 與上表逐一相符 |
| 單關節正向命令 | 實際位移方向與符號相符，量值在 5% 內 |
| 單關節反向命令 | 同上 |
| **關節能否離開限位** | 推到 `hi − 0.02` 後反向命令**必須能退出**（Gazebo 上觸限即卡死，見 §10.18） |
| 六軸同時命令 | 無串擾，非命令軸位移 < 10 mrad |
| `/joint_states` 頻率與間隔 | 記錄 p50／p95／max，作為 `max_data_age` 的依據（Gazebo 的 0.5 s 不可沿用） |

### 第 2 關：底盤命令

| 檢查 | 判準 |
| --- | --- |
| `/cmd_vel` 三分量方向 | vx／vy／ωz 的正負與實際運動相符 |
| 車體座標 vs 世界座標 | 轉接器的旋轉在非零 yaw 下仍正確 |
| 里程計來源 | 記錄是真值導出或輪速推算；**Gazebo 用真值，Isaac 若用推算則本測試涉及定位** |
| 底盤速度上限 | 實測是否受輪系限制；目前模型是盒約束（§11 已知限制） |
| **輪地接觸模型** | Isaac 若為真實輪地接觸，**打滑／牽引力首次進入迴路**，須另立測試 |

### 第 3 關：停止行為

| 檢查 | 判準 |
| --- | --- |
| 閘門逾時歸零 | 命令歸零時間、**實際停止時間、期間位移**三者分開量 |
| **停止距離** | **Gazebo 的 0.00 mm 不可沿用**（`VelocityControl` 直接寫速度、無減速動力學） |
| reason 7（距離過期） | 命令歸零、reason 正確、恢復後回復 |
| reason 8（帶截斷） | 命令全程為零 |
| JointState／TF／odom 分別停止 | 各自的歸零時間與 reason |
| 減速度是否受限 | 取樣率須足以分辨（Gazebo 上 143 Hz 無法分辨 19.98 rad/s²） |

### 第 4 關：同案例閉迴路

以凍結基線（B, μ = 0.03）重跑同一案例，逐項對照
`evaluation/results/baseline_B_frozen_20260909.md`：

| 量 | Gazebo 基線（3 次） |
| --- | --- |
| 完成 | 3 / 3 |
| 完成時間 | 12.134 ± 0.023 s |
| 位置終值 | 2.293 ± 0.008 mm |
| 末端姿態終值 | 0.055 ± 0.001° |
| 最小安全間距 | 88.816 ± 0.000 mm |
| 發布間隔 p50 / p95 / max | 50.0 / 52.6 / 61.8 ms |
| QP p50 / p95 / max | 2.1 / 6.1 / 16.0 ms |

**差異超過上表標準差一個量級時，先查介面與幾何，不要先調控制器參數。**

## 四、移植前必須知道的已知差異

- Gazebo 的輪與滾輪是**純視覺**，接地是四顆零摩擦球，底盤由 `VelocityControl` 直接驅動。
  Isaac 若用真實輪地接觸，這是**模型層的改變**，不是移植誤差。
- Gazebo 的 `/odom` 是真值導出，**本案例不涉定位**。
- Gazebo 上 `max_data_age = 0.5 s` 是依該模擬器的停頓分布量出來的（曾量到 332 ms 停頓），
  **必須在 Isaac 上重新量測**。
- 關節觸及 URDF 上限後無法退出，是 Gazebo 上的觀察，**底層原因未定位**；Isaac 上要重新確認，
  目前以 `joint_limit_margin = 0.02 rad` 迴避。
