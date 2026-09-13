# Isaac 全身執行端移植：第一輪嘗試（2026-09-13）

規格：`evaluation/results/specs/isaac_wholebody_port_v2.md`（sha `083909d346bf4956`）

**定位**：把已在 Gazebo 驗收過的全身同動能力接到 Isaac 執行端，**不重新研究控制器**。
既有 `isaac_bigarena_sim.py`、`isaac_manip_sim.py`、`isaac_drawer_sim.py` **一律未動**。

## 1. 新增的檔案

| 檔案 | 用途 |
|---|---|
| `evaluation/isaac_wholebody_sim.py` | Isaac 執行端；只訂閱 `/wb_vel_cmd`（9 維） |
| `evaluation/wb_cmd_chain.py` | 命令鏈：有效性、快照、限制、整體失效處置 |
| `evaluation/test_wb_cmd_chain.py` | **不開模擬器**的純邏輯測試，43 項 |
| `evaluation/wb_bounded_cmd.py` | 有界測試命令源 → `/wholebody_safety/cmd_in` |
| `evaluation/wb_preflight.py` | 起動前檢查：內容 / 新鮮度 / 端點身分 |
| `evaluation/run_wb_base.sh` | base 介面測試的 runner |
| `evaluation/analyze_wb_base.py` | 三項證據的分析 |

## 2. 九次嘗試：前八次啟動／整合失敗，第九次完整跑完

前八次**全部是啟動／整合失敗，未進入運動測試** —— **不是全身控制方法失敗**。
其中**七次有趟次目錄**（各標 `FAILED.md`），**另一次在建立目錄前就失敗**
（`set -u` 與 ROS `setup.bash` 衝突，runner 在 `mkdir` 之前就退出）。

| 趟次 | 失敗原因 |
|---|---|
| `wb_base_123224` | `sys.path` 缺 `src/ammr_wholebody_mpc` |
| `wb_base_123350` | 用了批次視圖的複數形 API；`SingleArticulation` 只有單數形 |
| `wb_base_123600` | 缺 TF `odom → base_link`；連帶距離節點無輸出 |
| `wb_base_124716` | `tf2_ros` / `TransformStamped` 未匯入 |
| `wb_base_124900` | `loop()` 未取得 `fp` |
| `wb_base_124952` | runner 未 source workspace，`ammr_wholebody_mpc` 找不到 |
| （無目錄） | `set -u` 與 ROS `setup.bash` 衝突，在建立目錄前退出 |
| `wb_base_125312` | **adapter 需要 ros2_control 控制器才能核對關節順序** |
| `wb_base_130232` | **完整跑完**（見 §7） |

## 3. frame 接線：已修正並驗證

原本打算直接發 `odom → base_link`，**那是錯的**：

* 模型鏈是 `odom → base_footprint → base_link`，
  `base_footprint → base_link` 是 URDF 的固定 **+0.05 m**。
  另發 `odom → base_link` 會讓同一個 child 有兩個父節點。
* **兩份 URDF 用途不同，不可互換**：
  `omni_bot_manip.urdf`（根 `base_footprint`）給 robot_state_publisher 發 TF；
  `omni_bot_wholebody_expanded.urdf`（根 `world → virtual_base → base_x/y/theta`，
  底盤是**真實關節**）只當距離／安全節點的 FK **參數**。
  我一開始把全身版拿去發 TF，根本結構就不同。

修正後實測（`wb_base_125312`）：

| 項目 | 值 |
|---|---|
| TF `odom → base_footprint`（Isaac 直接發布） | age 0.0000 s |
| TF `odom → base_link`（**TF 鏈組合**） | 平移 (0, 0, **0.0500**)、四元數模長 1.000000 |
| articulation 根 vs `base_footprint` 位置差 | **0.0000 mm**（量測核對，非假設） |

另兩項一併修正：位姿改讀 `base_footprint` prim 的**實際世界變換**（保留完整姿態，
不再用 yaw 重建四元數）；`/odom.twist` 改以 **child_frame 表達**（世界速度旋轉而來）。
**速度的參考點未移動**，此差異列為明文限制。

## 4. 起動前檢查：15 項通過 14 項

`wb_base_125312`：`/clock` 前進、`/joint_states`（含 velocity 有效性）、`/odom`、
`/robot_description`（latched QoS）、**距離資料 11 點且必要欄位齊全**、
兩條 TF —— 內容、新鮮度（**以 `header.stamp` 判定**）與端點身分全部通過。

未通過：`/wb_vel_cmd` **發布者 []**。

## 5. 已解決：adapter 與 Isaac 鏈的架構不相容

`evaluation/arm_vel_adapter.py` 原本把消費端寫死為 `/lite6_vel_controller`
（Gazebo 鏈的 ros2_control 控制器）；**Isaac 鏈沒有控制器**，執行端直接吃
`/wb_vel_cmd`，所以 adapter 查不到參數服務就退出，`/wb_vel_cmd` 無發布者。

**修法（採用選項 a）**：adapter 新增 `--consumer-node`，**預設仍為
`/lite6_vel_controller`**（Gazebo 行為不變）；Isaac runner 明確指定
`/isaac_wholebody_sim`。核對對象因此是**真的會執行這些數字的那一端**。
查不到、型別不對或順序不同，仍**拒絕轉送**。

Isaac 端的 `joints` **不是另宣告一份讓檢查通過**：`CMD_JOINT_ORDER` 一份供
三處共用 —— `joints` 參數回報、六個速度分量的積分、articulation 的 DOF 索引
與套用。啟動時記錄映射並檢查無缺漏、無重複，執行中不改：

```
v[3]  joint1    DOF 0
v[4]  joint2    DOF 1
v[5]  joint3    DOF 2
v[6]  joint4    DOF 3
v[7]  joint5    DOF 4
v[8]  joint6    DOF 5
```

純邏輯測試由 43 項擴為 **59 項**，新增涵蓋：正確順序通過、交換兩個關節拒絕、
服務缺失／回應無效拒絕、型別不對拒絕、集合不同拒絕，以及**預設 Gazebo 查詢
路徑的回歸檢查**。

## 5b. 原始問題描述（保留）

`evaluation/arm_vel_adapter.py` 建構時查詢 `/lite6_vel_controller/get_parameters`
以核對關節順序，查不到就退出。那是 **Gazebo 鏈**的 ros2_control 控制器；
**Isaac 鏈沒有控制器**，執行端直接吃 `/wb_vel_cmd`。

這個檢查是**實質的安全性質**，不能只是跳過 —— adapter 自己的註解說明了理由：
九個數字到達不能證明每個屬於哪個關節；順序錯了，下游殘差看起來仍正常。

待裁決的兩個選項見對話紀錄；**未自行決定**。

## 6. 目前狀態

* 純邏輯測試 43 項通過 —— 但那是**純邏輯**，不等於執行期正確
* 回授、TF、距離資料、安全層資料**已實測在線並通過內容與新鮮度檢查**
* **尚未取得任何底盤運動證據**；命令源從未被放行

## 7. `wb_base_130232`：第一趟完整跑完的 base 介面測試

起動前檢查**全部通過**；命令源發出 181 則；三個結果檔齊全；退出碼 0。

### 7.1 完整鏈路是否傳到執行端 —— **是**

| 證據 | 值 |
|---|---|
| 端點身分 | `/joint_states`、`/odom` 各恰一個 `/isaac_wholebody_sim`；`/wb_vel_cmd` 恰一個 `/arm_vel_adapter` |
| 關節順序核對 | adapter 對 `/isaac_wholebody_sim` 核對通過 |
| 命令源 → 執行端 | 發出 181 則 → callback **1369** 則 |
| 接收／拒收／失效 | 1369 / **0** / **None** |
| 實際套用 | **6860 / 6999** 個物理步，`recv_seq 1–1368` |

callback 多於發出則數，是因為安全層以自己的節奏持續輸出（含命令源停止後的零命令）。

### 7.2 底盤是否按命令運動 —— ~~否~~ **本趟量不出來（見 §9 更正）**

> **2026-09-13 更正（依 `wb_base_diag_160757`）**
>
> 本節兩項結論不成立，原文保留以便對照：
>
> 1. **「實際位移 +0.0000 m」是讀不到，不是量到零。** 本趟的
>    `SingleArticulation` 指向 `/World/omni_bot`（URDF 匯入的根 Xform），
>    **不是 articulation root**（`/World/omni_bot/Geometry/base_footprint`）。
>    該 Xform 不隨關節運動，`base_x` 6999 筆**相異值只有 1 個、恰為 0.0**。
>    底盤本趟動沒動，**這份資料分不出來**。
>    同理 `base_lin_meas` 峰值 0.0101 m/s 也出自同一包裝，
>    不能用來支持「只追蹤 33 %」或「屬雜訊」。
>    當時的自檢「articulation 根 vs base_footprint 位置差 0.0000 mm」
>    是在 t=0、兩者都在原點時做的，**無鑑別力**。
> 2. **「積分恰等於名目值 ⇒ 安全層完全沒有修改命令」不成立。**
>    積分相同不等於逐筆相同；本趟未逐筆比對。

| 量 | 值 |
|---|---|
| 對**最終套用命令**積分 | Δx **+0.1200 m** |
| **實際位移** | Δx **+0.0000 m** |
| 差 | **120.03 mm** |
| 底盤實測速度峰值 | 0.0101 m/s（命令 0.03 m/s） |

套用命令的積分恰等於名目 0.12 m ⇒ **安全層完全沒有修改命令**。
但底盤實際未位移；0.0101 m/s 的峰值與停止後 0.001–0.009 m/s 的持續跳動同量級，
屬雜訊而非受控運動。

**手臂漂移**（base 模式零速度命令下）：joint2 最大 **+1.565 mrad**、
速度峰值 0.008214 rad/s；其餘 ≤ 0.58 mrad。

### 7.3 上游停止後

命令源最後一則 sim 27.70；其後 `vx_cmd` 全程 0.0000，
底盤實測速度仍在 0.001–0.009 m/s 之間跳動，未收斂到零；手臂速度 ≤ 0.0012 rad/s。
停止原因 `sim_limit`（**程序停止原因**）；凍結步數 14。

**不宣稱接收端逾時已驗證**：命令源停止後安全層仍持續輸出零、adapter 繼續發，
上述只說明上游失效處置與實際停止行為。
命令源**先歸零再停止發布**，也**不算「運動中突然斷訊」測試**。

### 7.4 尚未區分的成因

底盤未移動，至少兩個候選、現有資料分不開：

1. `set_linear_velocity` 每步設定後被物理求解器歸零（浮動底座與地面的接觸／摩擦）
2. 施加方式與導航版 `isaac_bigarena_sim.py` 的底盤速度改寫不同

**尚未比對** `isaac_bigarena_sim.py` 的實際施加方式。

## 8. 目前狀態（已被 §9 取代）

* 命令鏈、frame 接線、關節順序、安全層資料**已實測打通**
* ~~**base 單動介面測試未通過**：底盤未按命令移動~~ → 見 §9
* 全身協同**尚未開始**；本輪只到 base 單動

## 9. `wb_base_diag_160757`：短程低速三路同步診斷

目的：分清「讀錯位置」與「底盤確實沒有按命令走」。
命令剖面不變（report frame +x，零 2.0s → 斜升 1.0s → 保持 3.0s 於 0.03 m/s
→ 斜降 1.0s → 零 2.0s → 停止發布）。`wz_cmd` 全程 0。

### 9.1 同步記錄的三路

| 欄位 | 來源 |
|---|---|
| `sent_prev_vwx/vwy/wz` | **上一步實際傳進** `set_linear_velocity`／`set_angular_velocity` 的世界速度 |
| `phys_x/y`、`phys_vx/vy` | 下一物理步後 PhysX 經 articulation 回報的位置／速度 |
| `usd_x/y` | 同步的 USD `base_footprint` 位姿 |

迴圈順序是「物理步進 → 讀量測 → 套用新命令」，所以同一列的量測是**上一列命令**
的結果；`sent_prev_*` 在套用新命令**之前**先存下，就是為了讓 log 逐列對得上。

### 9.2 「讀錯位置」—— **證實存在，且已修正**

`SingleArticulation` 的 `prim_path` 從寫死的 `/World/omni_bot` 改為
**搜尋到的 articulation root** `/World/omni_bot/Geometry/base_footprint`
（與導航版 `isaac_bigarena_sim.py` 一致，用 `isaac_common.physics_parts`）。

本趟三路最大差值：

| 比較 | 最大差 |
|---|---|
| 物理端 vs USD `base_footprint` | **0.000000000 m** |
| USD vs 送交 TF 的 `base_x/y` | **0.000000000 m** |

三個來源指向同一個剛體，本趟不存在「讀錯位置」的空間。

### 9.3 「底盤沒按命令走」—— **不成立；本趟按命令走了**

| 量 | 值 |
|---|---|
| 命令 x 積分 | +0.11999 m |
| 實送 API x 積分 | +0.11999 m |
| 物理速度 x 積分 | +0.12013 m |
| **實際位移**（命令起點→終點） | **+0.11951 m** |
| 保持段（22.0–24.7s）物理 vx 平均 | **0.030002 m/s**（命令 0.03） |
| 保持段逐筆誤差 RMS | **1.0e-5 m/s** |
| 零命令段（10–20s）`phys_vx` RMS 底噪 | 9e-6 m/s |

保持段的逐筆誤差 RMS 與零命令底噪同量級 ⇒ 追蹤誤差已落到本趟量測底噪內。

### 9.4 歸因**尚未分離**

本版**同時**改了兩件事，所以「接觸設定漏接」**還沒有被確認**：

1. articulation root 搜尋（§9.2）
2. 零摩擦材質綁定：支撐球 4 + 地面 2 = **6 個綁定**
   （URDF 的 `<gazebo><mu1>` importer 不讀；雙方都綁才有效，
   預設混合規則取平均）

由於 §7.2 那趟的位姿欄位不可讀，**無法**回頭判斷當時底盤是否已在移動。
要分離只需一趟 `--no-frictionless`（保留 root 修正、只拿掉材質綁定）；
旗標已就位，尚未執行。

### 9.5 零命令下仍存在的兩項運動（未解釋）

| 項目 | 值 |
|---|---|
| 命令歸零後 x 反向漂移（26–70 s） | −3.893 mm，速率 **−0.0885 mm/s** |
| yaw 漂移（`wz_cmd` 全程 0），閒置段 10–20 s | **+0.01346 °/s** |
| yaw 漂移，命令結束後 26–70 s | **+0.01350 °/s** |
| 70 s 累積 yaw | +0.930° |

yaw 漂移率在「有下過命令前」與「命令結束後」幾乎相同 ⇒ 與本趟命令無關的常態漂移，
不是命令殘留。兩者量級對本趟 0.12 m 行程可忽略，但**對後續保持／pregrasp 的驗收會進入門檻量級**，
先記錄不處理。

### 9.6 本趟其他

* `cmd_chain`：received 1373、rejected 0、fail 無；`cmd_age` 中位 0.030 s、max 0.060 s
* 停止原因 `sim_limit`（**程序停止原因**，非任務成功）
* CPU 起 75.0 °C、峰值 **85.375 °C**（上限 92.0，未觸發）

## 10. 目前狀態

* 命令鏈、frame 接線、關節順序、安全層資料**已實測打通**
* **base 單動已實測會動且追蹤到底噪內**（§9.3）
* **未確認**零摩擦綁定是否為必要條件（§9.4）；需一趟 `--no-frictionless` 對照
* 零命令下的殘留平移／偏航漂移**未解釋**（§9.5）
* 全身協同**尚未開始**；pregrasp 仍由程式常數無條件禁止（輪級限制未實作）
