# coman_drawer20_retry4：**命令鏈缺輸入，任務未開始**

**不是操作失敗** —— 求解節點起來了但**一筆命令都沒下**，相位未推進。
三題**都沒有證據**，不得以本趟填任何一題。

| 項目 | 值 |
|---|---|
| 時間 | 2026-10-01 00:34 起 |
| 起動前檢查 | 0 項失敗（含新增的 source 守衛） |
| **F1** 固定關節守衛 | 通過 |
| **F2** 目標開度 | 通過（0.020 m） |
| **F3** `machine_wired` | 通過 |
| **F4** `joints` 宣告 | **通過** —— `關節順序已對 /isaac_drawer_sim 核對通過：['joint1'…'joint6']` |
| **F5** 工作區 source | **通過** —— 四個節點全部啟動（dist／safety／adapter／diagrec） |
| R1.1 十五組間距 | **兩端載入一致**（求解端與安全節點 log 逐項相同） |
| `stop_reason` | `sim_limit` |
| 命令 | **未下**（`資料不齊（arm=True, base=False, rows=False）：不下命令。`） |
| CPU | 起 69.9 °C，峰值 **86.875 °C** @ sim 23.3 s（限 92 °C，餘裕 **5.1 °C**） |
| 殘留程序 | 無 |

## 缺口（第六個）：執行端不發 `/odom` 與 `odom → base_link` TF

| 執行端 | 發布 |
|---|---|
| **協同版**（本趟） | `/clock`、`/joint_states`、`/coman/task_state`、`/manip/*` |
| **e2 版**（既有可用） | `/clock`、`/joint_states`、**`/odom`** ＋ `TransformBroadcaster` |

連鎖後果：

* 求解節點取不到底盤狀態 ⇒ `base=False`
* 距離節點 `report_frame:=odom` 查不到 TF ⇒ 整片距離列 NODATA ⇒ `rows=False`

協同執行端由**固定底座**版衍生，底盤不動時不需要 odom；
接上開放底盤與全身 QP 時沒有補上。

**修法參考**：`isaac_wholebody_sim_e2.py:193-229`（`Odometry` 發布器 ＋
`TransformBroadcaster`）。該處註解已標出兩個易錯點：
`/odom.twist` 以 **child_frame_id 的本體座標**表達；
`odom → base_link` **不得重複發**，否則 `base_link` 會有兩個父節點。

**本輪未實作** —— 這是新增執行端對外發布（約 30 行，牽涉座標系與 TF 樹結構），
弄錯會安靜地產生錯誤的底盤狀態,而底盤狀態正是第二題的量測基礎。

## 熱觀察（新）

| 趟次 | 起 | 峰值 | 負載 |
|---|---|---|---|
| retry2 | 66.9 °C | 79.25 °C | Isaac 空轉（命令鏈未起） |
| **retry4** | 69.9 °C | **86.875 °C** | **命令鏈整條運作**（距離節點 1100+ 列、安全、求解、錄製） |

峰值在 sim 23.3 s 就出現,離 92 °C 中止線僅 **5.1 °C**。
真正下命令的趟次負載只會**更高**,連續重跑前應讓機器降溫。
