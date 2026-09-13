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

## 2. 八趟嘗試：**全部是啟動／整合失敗，未進入運動測試**

**不是全身控制方法失敗。**

| 趟次 | 失敗原因 |
|---|---|
| `wb_base_123224` | `sys.path` 缺 `src/ammr_wholebody_mpc` |
| `wb_base_123350` | 用了批次視圖的複數形 API；`SingleArticulation` 只有單數形 |
| `wb_base_123600` | 缺 TF `odom → base_link`；連帶距離節點無輸出 |
| `wb_base_124716` | `tf2_ros` / `TransformStamped` 未匯入 |
| `wb_base_124900` | `loop()` 未取得 `fp` |
| `wb_base_124952` | runner 未 source workspace，`ammr_wholebody_mpc` 找不到 |
| （第七趟） | `set -u` 與 ROS `setup.bash` 衝突，未建立資料目錄 |
| `wb_base_125312` | **adapter 需要 ros2_control 控制器才能核對關節順序** |

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

## 5. 未解決：adapter 與 Isaac 鏈的架構不相容

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
