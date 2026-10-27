# 連續協同通行：核心段 — 規格 draft-10（台架初態啟動 r2，bench2 版）

draft-9（bench r1）保留、不跑。本版依 Codex r21 三項必修版本化修正；**目標、範圍、登錄 r9、初值、求解端 ADM17、其餘參數與保護全部同 draft-9**，只列差異。

**成果口徑**同 draft-9：指定初態（台架啟動）、經實際夾持取證後的持握開門與出口繞行核心驗證。
- `chain_pass` 不等於邊開邊繞成功；夾持、實測開門、開門同期出口進展分項報告。

## 1. 釋放綁定真正進鏈的首筆命令

### `door_bench_start2.py`（r1 保留）
- 宣告須 kind＝solution，並核本趟 run_id、正整數 seq、sim_t＝t0。
- `release_intake`：凍結期釋放當輪，把宣告 seq 的封裝交給真正入口 EnvIntake.receive 與命令鏈。條件：
  - 須回 fed；
  - 鏈未新增拒收、鏈未失效；
  - 同輪未中止；
  - 封裝 seq＝宣告。
- `verify_first_step`：第 1 物理步寫入 API 之前核對：
  - 控制者 wholebody，套用種類 wholebody_applied；
  - 套用來源 seq＝宣告，鏈模式 0；
  - 九軸有限；
  - 設定點＝s_init＋dt·u_arm（1e-9）。

### 執行端 ev9r2（ev9 保留）
- 釋放當輪改走 `release_intake`；失敗即不釋放。
- 成功後清除最新封裝；第 1 物理步不再收新封裝（首筆固定）。
- 第 1 步核對通過才發 `/bench/state released`；不符即 K2 閂鎖中止、不寫入 API、不前進物理。
- room_run bench 另記 first_intake、first_step。

### 測試（真 EnvIntake ＋ 真 CmdChainE2 ＋ 真 DualSourceExecutor）
- 錯趟 ⇒ wrong_run；非正常 kind ⇒ wrong_kind；非衍生 ⇒ not_derived；閂鎖過濾 ⇒ filtered。
- 釋放同輪中止、seq 非宣告 ⇒ 拒。
- 首筆被替換（第 1 步前又進一筆）⇒ applied_source。
- 種子未承接 s_init ⇒ setpoint_carry。

## 2. 物理前進後不得跳過監看

### `door_coopen_bench_grasp2.py`（r1 保留）
- 只有確認仍凍結（步號 0 且時間 ≤ t0）才跳過監看。
- 物理前進後每輪先監看：ACTIVE_MONITOR 暫時擴為原集合＋BENCH_START，try/finally 還原。
- 已收到經驗證的 released 才轉入 INIT_CHECK；這一輪本身已監看。
- 自第一筆前進樣本起超過 stale_s 0.2 s 仍無釋放證明 ⇒ 中止 bench_release_unconfirmed。

### 任務節點 v8（v7 保留）
- `/bench/state` 經 StateFilter 驗證：本趟 run_id、t0 不變、次序 frozen → released／aborted 不回退。未通過不採用，記入 rejects。

### 測試（實際 K3MP2 組合）
- 通知延遲：延遲期間每輪監看。
- 通知遺失：0.2 s 內中止。
- 第一筆前進資料已有手指—橫桿接觸：於 BENCH_START 中止。
- 凍結期注入：不中止，計入跳過輪數。

## 3. bench_anchor 完整承接舊判準

### 判定器 r10（r9 保留）
- **交叉核對錨點**：
  - room_run bench.anchor、hold_start(bench) 事件錨點、求解端 bench.anchor 三者相等（1e-9）；
  - 與 start_pose 差 ≤ 0.005 m／0.01 rad。
- 一致時，給鏈判定的 gate 副本設 gate.anchor＝bench_anchor（passed 不改，記 anchor_source）。**不再豁免任何鏈理由**（chain_exempt 為空）。
- 不一致時，鏈照原 gate 判，並記 bench_anchor。
- J7 另核 first_intake 與 first_step。

### 真鏈串接測試（door_wide_w1f_60_3 實錄 ＋ door_k3m_check_w1f2.judge；該趟原判通過）

| 情境 | 結果 |
|---|---|
| 原 gate | 通過 |
| gate.anchor＝None | k3m_room_evidence_invalid（Codex 重現） |
| 補上與原錨點相同的 bench 錨點，經 anchor_view | 與原判**完全相同**（通過、理由空） |
| bench 錨點整體平移 50 mm（三處一致） | 鏈的底盤核對照算並不通過（base_drift_before_switch） |

- 最外層 r10 `judge()` 跑在同資料上：chain_exempt 空；無 k3m_room_evidence_invalid、無 park_anchor_missing。
- 其餘理由只有 coopen_params_missing、coopen_gate_params_missing、iso_reg_invalid、bench_reg_invalid：該趟不是 coopen 台架趟，沒有這些登錄。
- **揭露**：目前沒有任何 coopen 台架實錄可作完整通過例。

## 4. 測試（全過）

| 測試檔 | 項數 |
|---|---|
| test_door_bench_start2 | 4 |
| test_door_coopen_bench_grasp2 | 5 |
| test_door_coopen_check_core10 | 5（含 r9 的 19 反例以 r10 重跑、真鏈串接） |
| test_door_coopen_core_bench2_static | 3 |
| 回歸：bench_start 7、check_core9 4、bench_static 4、check_core8 11、init_grasp2 10、gate2 13、k3a_policy 18、settle_p2 11 | — |

## 5. 實跑（待 Codex 短複核）
- 單趟 `door_coopen_core_bench_1`，跑法 `run_door_coopen_core_bench2.sh`，凍結 `freeze_door_coopen_core_bench_dev_r2`。
- 乾淨環境、92 °C 守護、跑前／跑後核對；失敗保留、不補跑、不新增影片。
