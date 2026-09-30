# coman_drawer20_retry2：**環境未 source ＋ 接線缺口，任務未開始**

**不是操作失敗** —— 執行端跑滿 120 s 但**一筆命令都沒收到**，相位始終 `idle`。
三題**都沒有證據**，不得以本趟填任何一題。

| 項目 | 值 |
|---|---|
| 時間 | 2026-10-01 00:28 起 |
| GPU | **正常**（RTX 5060、驅動 595.91.07、CUDA 13.2、`cudaErrorNoDevice` **0 次**） |
| 起動前檢查 | 0 項失敗 |
| **F1** 固定關節守衛 | **通過**（`0 個（模式 free_base 期望 0 個）`） |
| **F2** 目標開度 | **通過**（`0.020 m … **本趟覆寫，案例值 0.200 m**`） |
| **F3** `machine_wired` | **通過**（`machine_wired=True`，無 NameError） |
| `stop_reason` | `sim_limit`（跑滿 120.61 s sim／137.39 s wall，RTF 0.878） |
| `chain9.received` / `rejected` | **0 / 0** —— 命令從未送達 |
| 相位 | 只有 `idle`；最終開度 **0.000 mm** |
| `base_fixation.mode` | `free_base` |
| CPU | 起 66.9 °C，峰值 **79.25 °C** @ sim 91.3 s（限 92 °C，未觸發） |
| 殘留程序 | 無 |

## 兩個原因

### 1 工作區未 source（決定性）

```
Package 'ammr_wholebody_mpc' not found
package 'my_omnibot_description' not found, searching: ['/opt/ros/jazzy']
```

只有基礎 ROS 在路徑上。後果：

* `ros2 run ammr_wholebody_mpc arm_link_distance` 與 `wholebody_safety` **根本沒啟動**
* 求解節點的 `xacro` 找不到描述套件 ⇒ `CalledProcessError`
* 腳本仍一路跑到 `sim_limit` 才停 —— **沒有任何一處擋下來**

`run_coman_drawer20.sh` 不自己 source,依賴呼叫端的環境。

### 2 執行端未宣告 `joints` 參數

```
[arm_vel_adapter] 關節順序核對未通過（/isaac_drawer_sim）：消費端未回報 joints
[arm_vel_adapter] refusing to forward commands
```

adapter 會查消費端的 `joints` 參數核對關節順序（九個數字本身不帶關節身分,
順序錯就是每個關節拿到別人的速度,而下游殘差看起來完全正常）。
協同執行端**沒有宣告**這個參數,所以 adapter 拒絕轉發 —— **守衛正常運作,
缺的是本端沒提供資訊**。`isaac_wholebody_sim_e2.py:185` 早有同樣的宣告。

這是 F1／F2／F3 之後**第四、第五個**只存在於此路徑的缺口。

## 本趟唯一可記的觀察

`free_base` 模式在 Isaac 中連續跑了 **120 s 物理**、無命令、無監看失效、
底盤未觸發 `base_drift` 之類的中止。**這只說明站得住,不說明任何任務能力。**

## 已修

* **F4**：`isaac_coman_drawer_sim.py` 的 `DrawerNode.__init__` 加
  `declare_parameter('joints', list(ARM))`。
* **F5**：`run_coman_drawer20.sh` 新增起動前守衛 —— 工作區未 source 即
  **具名失敗並 `exit 65`**,不再浪費整趟。已實測：未 source 時擋下並印出
  `source .../install/setup.bash`。

## 重試

本目錄保留原樣；重試另開 `coman_drawer20_retry3`,並在呼叫端 source 工作區。
