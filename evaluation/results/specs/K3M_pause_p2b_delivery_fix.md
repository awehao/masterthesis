# K3-M 暫停 p2b：SETTLE 窗口資料送達修正（決策紀錄＋實作＋測試＋凍結 p2b r1）

## 0. 決策來源
- 2026-10-24 下午，Codex 審查管道無法使用：reviews/20261024_155402 已排入但沒有回覆。
- Howard 指示「codex不見了跑不出來 先自行決策」。本版由 Claude 裁定。
- 裁定時沿用既有約束：
  - 不調參、不放寬保護或判準、不改窗口參數；
  - 單趟；92 °C 熱中止；
  - 失敗照實保留；交 side／close 實錄重播；
  - 不回頭改判既有趟次。
- Codex 恢復後，本版與其結果一併送審。

## 1. 要修的缺陷（door_k3m_pause_p2_1）
- p2 任務節點的 SETTLE 窗口掛在既有 `/door/sync_state` 訂閱上，佇列深度 1。主迴圈每輪開頭才處理待收訊息（≤ 64 次 spin_once），兩輪之間到達的多筆只留最新一筆。
- 窗口實收約每 0.08 s 一筆，所以 gap 與 long_window 恆成立，SETTLE 完成訊號全程無效。

## 2. 裁定：方案 A（只改送達，不動判準）
- 不選方案 B（放寬 gap／long 參數去配合約 12.5 Hz 的送達）。理由：
  - B 會改變已核可的判準細節（0.1 s 窗口、0.03 s 缺步界）；
  - 而且會把「沒收到樣本」當成「低速證據」的機會放大。
- **p2b 改動**：`door_k3m_task_node_p2b.py`（p2 複本）。
  - 另開一個**只餵窗口**的 `/door/sync_state` 訂閱，深度 64。
  - 原深度 1 的訂閱、SyncTracker、量測路徑完全不變。
  - 窗口參數不變：0.1 s／0.03 s／0.13 s／0.2 s。
  - 策略 `door_k3m_policy_p2.py`、判定器 `door_k3m_check_p2.py` 不變。
- **語意差異**（比 p2 保守）：窗口訂閱不經 SyncTracker。
  - 重送由窗口拒收（duplicate）；
  - 倒退由窗口清空，之後須重新累積滿 0.1 s 才可能 valid。
- **新增紀錄**：
  - trace 每列 `settle_rx`：兩輪之間窗口訂閱實收筆數；
  - task.json `settle_delivery`：變體、深度、累計實收、長度錯誤數。
- **跑法** `run_door_k3m_pause_p2b.sh`：只換任務節點與凍結。

## 3. 測試（全部通過；ROS 測試連跑 3 次穩定）
| 測試 | 項目 |
|---|---|
| `test_door_k3m_task_node_p2b_ros.py` B1 | **缺陷情境**：兩輪之間連發 8 筆、中間不 spin，再照主迴圈做 64 次 spin_once，共 4 輪 32 筆。p2b 窗口實收 32、保留範圍內無缺、值 valid、rate 正確；同情境 p2 節點只留 4 筆，結果 `long_window`（重現缺陷） |
| 同檔 B2 | 排空後窗口最新步號＝量測路徑 `nd.sync` 步號，也就是窗口值的 step1 |
| 同檔 S | 重送 ⇒ 窗口拒收；倒退 ⇒ 窗口清空，0.08 s 時 `short_window`，補滿後 valid；量測路徑的 SyncTracker 照舊記到倒退 |
| 同檔 暫停路由 | p 版複本，照舊 |
| 回歸 | settle_p2 11、check_p2 8、task_node_p2_ros 2、policy_p 5、check_p 4、pause_sup 13、pause_mode 4、applied_src 5、task_node_p_ros 1、k3m_policy 12、k3a_policy 18 |

## 4. 凍結
`evaluation/results/door/freeze_door_k3m_pause_p2b_dev_r1.sha256`：
- p2 r1 的 172 檔原樣保留；
- 新增 p2b 任務節點、p2b 跑法、p2b ROS 測試。

## 5. 單趟 `door_k3m_pause_p2b_1`（Claude 依授權核可）
- **跑法**：`RUN_ID=door_k3m_pause_p2b_1 ROS_DOMAIN_ID=97 RUNNER=evaluation/run_door_k3m_pause_p2b.sh bash evaluation/run_guarded_v2.sh`
- **保護**：既有閂鎖與監看、獨立觀察器、92 °C。走廊偏航若再越界，照實失敗，不延長等待、不補跑。
- **分報**：開門、SETTLE（含每輪實收筆數）、命令停止、實際停止、保護完整性；附 side／close 該趟實錄重播（新命名規則）。

## 6. 補記：p2b_1 啟動競態 ⇒ 不改程式重試一次（Claude 依授權裁定）
- door_k3m_pause_p2b_1 在 WAIT_HANDOVER 逾時中止，沒有進入受測流程。
  - 時序觀察（不推定根因）：重開機後模擬冷啟動 20 s，交棒請求落在模擬 1.64 s，0.5 s 窗口內命令鏈未收到全身命令。
  - 詳見該趟 NOTE。
- 送達修正在該趟資料中已確認：實收與同步間隔數相同；step_mismatch 1.8%，屬保守重設。
- 核可 `door_k3m_pause_p2b_2`：同凍結、同跑法，只換 RUN_ID，只重試一次。再發生同一取消就停，另立交棒前置條件版本。
