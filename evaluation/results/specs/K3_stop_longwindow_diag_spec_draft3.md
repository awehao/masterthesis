# K3 停止後長窗隔離診斷 draft-3（四項接線必修；凍結 lw r2；**未核可實跑**）

依 Codex 20261023_231357。draft-2 的差異表、分析定義、門檻與控制參數都不變；本文件只列四項修正。draft-1／draft-2 保留。

## 1. 接觸窗口重送與非法值（`door_k3ab_lw_guard.py`）
- **合法性檢查**：必要欄位為 step、n_invalid、sim_t、unexpected／wall／wrist。
  - 實數須有限，bool 不算實數；step 與 n_invalid 須是 0…2^53 的整數。
  - 不合法 ⇒ contacts_invalid，**不拋例外**：∞、1.5、−1、10^30、字串、NaN、True、非 dict 都納入反例。
- **新鮮度**：只有合法、且來源步號前進的新窗口才刷新年齡。
  - 同一步號、同內容的重送只計數（n_ct_dup），不刷新年齡。
  - 同步號內容衝突、步號或時間倒退 ⇒ 觸發 contacts_invalid（detail 註明）。
  - 狀態訊息比照：同步號衝突 ⇒ state_invalid。
- **arm 前**的異常只記入 anomalies，不觸發。
- 觀察器回呼內的任何例外 ⇒ observer_exception（觸發保護），程序不崩潰。

## 2. 觀察器健康（觀察器心跳＋任務節點監看）
- **觀察器**：主迴圈每 20 ms 發 `/door/lw_observer/heartbeat`（seq 遞增、armed、trip、closing），第一筆前印「就緒」。
- **任務節點**（`door_k3ab_lw_task_node.py`，插入）：
  - 收到第一筆合法心跳前**不推進**相位機。`--lw-ready-timeout-s`（60 s 牆鐘）內未就緒 ⇒ 記 lw_observer_not_ready，有界結束。
  - 就緒後，心跳年齡 > `--lw-heartbeat-max-age-s`（1.0 s；觀察器退出或卡住）⇒ 記 lw_observer_unhealthy，有界結束：照原 finally 送 /wgmpc/stop 與 /room/stop，並保存紀錄。
  - 心跳 seq 必須嚴格遞增；重複或倒退不刷新。
- **跑法**：
  - 保留起觀察器 5 s 後的存活檢查；
  - 收尾在執行端行程結束後才 TERM 觀察器；
  - 新增機讀列 `LW_HEALTH sim=… sim_rc=… obs=… obs_rc=… task=… task_rc=…`；
  - task.json 記有失效時，在 run.log 標出。

## 3. closing 與 closed 分開
- **他方 /room/stop**（結束請求）⇒ `closing`：只停用停更告警，硬界與無效資料照判、照發；觸發記 `during_closing: true`。
- **運行器在執行端結束後送 TERM** ⇒ `finish` ⇒ `closed`（記 after_closing），之後才不再判定。
- 結束請求不直接當成正常收尾。

## 4. 分析器交叉核對（`door_k3ab_lw_analysis.py`）
- **保護觸發來源**（任一成立 ⇒ observation_terminated_early，停止判定未驗證；`guard_sources` 列出各來源）：
  - task.lw_guard；
  - 觀察器 trip；
  - room.stop_reason 含 experiment_guard；
  - task.lw_observer_unhealthy 或 lw_observer_not_ready。
- **保護覆蓋不完整**（⇒ `unverified_protection_incomplete`；另列 `stop_result_if_protection_ok` 供參考）：
  - 觀察器缺報告；
  - ended_by ≠ TERM；
  - 無 closed；
  - 主迴圈例外；
  - 無心跳；
  - 閂鎖卻未 arm；
  - 有 anomalies；
  - 運行器 LW_HEALTH 缺失，或觀察器／執行端不是 exited，或觀察器 rc ≠ 0。

## 5. 測試
- `test_door_k3ab_lw.py` 20 項通過。新增：
  - 重送 1 s 不掩蓋停更（Codex 反例）；
  - 11 種非法窗口不拋例外；
  - 同步號衝突與倒退（接觸、狀態）；
  - closing 後硬界照判、finish 後不判；
  - arm 前異常只記錄；
  - 心跳就緒、停止、重複與倒退；
  - room.stop_reason 記保護而 task 未記 ⇒ 提前終止（Codex 反例）；
  - 觀察器 trip 與 task 記失效；
  - 8 種覆蓋不完整。
- `test_door_k3ab_lw_observer_ros.py` 9 情境通過（隔離 ROS domain）：
  - O1 停更、O2 短脈衝、O4 未中止，同前；
  - O3：結束請求後在途 1 N ⇒ 觸發（during_closing），TERM ⇒ closed；
  - O3b：結束請求後只停更 ⇒ 不觸發，TERM ⇒ closed；
  - O5：觀察器就緒後退出 ⇒ 任務節點 1.005 s 後記 unhealthy 並結束；
  - O6：觀察器 SIGSTOP 卡住 ⇒ 1.018 s 後記 unhealthy；
  - O7：沒有觀察器 ⇒ 不推進，8 s 後記 not_ready。
- **煙霧**：以 door_k3ab_ev_1 跑分析器 ⇒ `unverified_protection_incomplete`（observer_report_missing、runner_health_missing）。舊趟本來就沒有觀察器，結果符合預期。

## 6. 凍結
- `freeze_door_k3ab_lw_dev_r2.sha256`：lw r1 清單加本文件；修改過的檔案重算雜湊。
- 跑法 FRZ 指向 r2。
