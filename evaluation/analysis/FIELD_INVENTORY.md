# 資料欄位盤點（唯讀，2026-09-30）

來源：現有程式與既有紀錄。**未修改執行端。** 缺項在 §9 集中列出。

時間基準只有兩種，**不得混用**：

| 標記 | 意義 | 來源 |
|---|---|---|
| `sim` | 模擬時間（秒） | Isaac `world` 時間；ROS 節點以 `use_sim_time` 取同一時鐘 |
| `wall` | 牆鐘（秒） | `time.perf_counter()` / `time.time()`，僅用於耗時與程序監看 |

座標系：`world` = Isaac 世界座標；`body` = 底盤本體座標（adapter 轉換後）；
`tool` = 夾爪座標；`odom` 僅出現在距離／安全節點的 `report_frame`。

取樣：執行端主迴圈**每個物理步**寫一列，`physics_dt` 預設 0.01 s ⇒ 100 Hz。
ROS 節點各自以 20 Hz（控制週期 0.05 s）或其 `publish_rate` 發布。

---

## 1. 抽屜開度、目標開度、操作相位

| 項目 | 來源檔案／欄位 | 單位 | 座標系 | 時間 | 實測？ | 缺值處理 |
|---|---|---|---|---|---|---|
| 抽屜開度 | `sim/drawer_run.json` → `log[*]` 欄 `opening`（`log_cols`） | m | 抽屜滑軸 | `sim`（欄 `t`） | **實測**（抽屜關節） | 每步都有；無缺值機制 |
| 開度速率 | 同上 `opening_v` | m/s | 同上 | `sim` | **實測**（關節速度） | 同上 |
| 本趟目標開度 | 同檔 `target_opening_used_m` | m | — | — | 設定值 | 單一純量 |
| 案例目標開度 | 同檔 `target_opening_case_m` | m | — | — | 設定值 | 200 mm，**非本趟目標** |
| 相位（執行端） | `log[*]` 欄 `phase` | 字串 | — | `sim` | 執行端狀態 | 每步都有 |
| 相位（求解端） | `/coman/contact_phase`（JSON：`phase`／`sim_t`／`raw`）；求解端 log 欄 `phase` | 字串 | — | `sim`（**來源時間**） | 政策輸出 | 過期即視為未知 |
| 正式到位判準 | `wb_coman_drawer20_criteria_v1.yaml`：`profile.target_stroke_m`、`P_pull.P1_final_opening_err_m_max`、`P1_hold_s` | m／m／s | — | `sim` | 規格 | — |
| **舊**案例容差判定 | `legacy_case_tol_arrived_sim_t`、`legacy_case_tol_opening_m`、`legacy_case_tol_hold_s` | s／m／s | — | `sim` | 舊判定 | **±10 mm，非 v1 正式資格** |

## 2. 連接、釋放請求、解除確認、退出完成

| 項目 | 來源 | 單位 | 時間 | 實測？ | 備註 |
|---|---|---|---|---|---|
| 連接（engage） | `events[]` 內 `{'event': 'engage_by_handover', 'sim_t': …}` | — | `sim` | 事件 | 由交接通過觸發 |
| 連接狀態（逐步） | `coman_machine_log` 欄 `handover_pass`；`log[*]` 的 `slip`／`f_*` 僅間接 | 布林 | `sim` | 旗標 | `coman_machine_cols` |
| 釋放資格 | `coman_machine_log` 欄 `normal_release_allowed` | 布林 | `sim` | 狀態機 | v1 資格 |
| 釋放握手 | 同上 `release_handshake` | 布林／碼 | `sim` | 事件 | 資格＋請求同週期 |
| 解除確認 | `coman_couple_link`（`record()`）：request／execute／confirm 各自的步號與時間 | step／s | `sim` | **讀回確認** | 執行 ≠ 確認 |
| 緊急解除 | `coman_machine_log` 欄 `emergency_decouple`；`coman_inject_info` | 布林 | `sim` | 閂鎖 | 獨立於正常請求 |
| 退出完成 | **見 §9 缺項 G1** | — | — | — | 政策端 `stamps['retreat_done']` **未輸出** |

## 3. 實測 TCP 與目標位姿

| 項目 | 來源 | 單位 | 座標系 | 時間 | 實測？ |
|---|---|---|---|---|---|
| 夾爪位姿（同一物理步） | `coman_pose_log` 欄 `grip_px…grip_qz`（`coman_pose_cols` = `coman_pose_reader.LOG_COLS`） | m／四元數 wxyz | `world` | `sim` ＋ `physics_step_id` | **實測**（stage 位姿） |
| 抽屜本體位姿 | 同上 `draw_px…draw_qz` | 同上 | `world` | 同上 | **實測** |
| 把手位姿 | 同上 `hand_px…hand_qz` | 同上 | `world` | 同上 | 抽屜位姿 × 固定變換 |
| 夾爪↔把手相對 | 同上 `rel_tx…rel_r22` | m／旋轉矩陣 | 相對 | 同上 | **實測** |
| 同一步讀取旗標 | 同上 `same_step_read`、`valid`、`invalid_reason` | 布林／字串 | — | — | **逐筆實際判定** |
| TCP（另一來源） | `log[*]` 的 `e_par`／`e_perp`（對理想路徑的分量誤差）、`stage_vs_fk_err` | m | `world` | `sim` | 實測 |
| TCP 發布 | `/…/tcp_pose`（PoseStamped） | m | `world` | `sim` | 實測，未落盤 |
| **目標**位姿 | **見 §9 缺項 G2** | — | — | — | 求解端只存 `ep`／`er` 純量 |

## 4. 底盤與手臂的實測運動

| 項目 | 來源 | 單位 | 座標系 | 時間 | 實測？ | 備註 |
|---|---|---|---|---|---|---|
| 底盤位移量 | `log[*]` 欄 `base_drift` | m | `world` | `sim` | 實測 | **是「距起點距離」的純量**，非位置 |
| 底盤偏擺量 | 同上 `base_dyaw_deg` | deg | `world` | `sim` | 實測 | **取絕對值** |
| 底盤位置／速度 | **見 §9 缺項 G3** | — | — | — | — | **未記錄** |
| 手臂關節角 | `log[*]` 欄 `joint1…joint6` | rad | 關節 | `sim` | **實測** | 100 Hz |
| 手臂關節速度 | **見 §9 缺項 G4** | — | — | — | — | 未記錄；可由關節角差分 |
| 手臂設定點 | `coman_applied_cmd_log` 欄 `q1…q6` ＋ `source` | rad | 關節 | `sim` ＋ step | **命令**（非實測） | `wb9`／`frozen` |
| 逐關節命令誤差 | `coman_joint_err_log` 欄 `dq1…dq6` | rad | 關節 | `sim` | 實測−命令 | **僅記錄，未用於任何門檻** |
| 手指關節 | `log[*]` 欄 `fj_req`／`fj_sent`／`fj1_act`／`fj2_act` | rad | 關節 | `sim` | 請求／送出／**實測** 分開 | 不可互相代替 |

## 5. 關節限位與餘裕

| 項目 | 來源 | 單位 | 時間 | 實測？ |
|---|---|---|---|---|
| 最小限位餘裕 | `log[*]` 欄 `limit_margin` = `min_k min(q_k − lower_k, upper_k − q_k)` | rad | `sim` | **實測關節角**算出 |
| 限位來源 | `LITE6_SAFE.lower/upper`（程式常數） | rad | — | 設定 |
| 逐關節餘裕 | 可由 `joint1…joint6` ＋ 限位重算（分析端） | rad | `sim` | 實測 |
| 案例中止門檻 | `manipulation_cases.yaml` → `tolerance.joint_limit_margin_rad` = 0.05 | rad | — | 設定 |

## 6. 耗時與逾時

| 項目 | 來源 | 單位 | 時間基準 | 備註 |
|---|---|---|---|---|
| 求解耗時 | 求解端 `--out` JSON → `log[*].solve_ms`；`/coman/cmd_meta` 第 3 欄 | ms | **wall** | 逐週期 |
| 安全節點：投影耗時 | `/wholebody_safety/diag` 欄 `filter_ms` | ms | **wall** | 按名稱索引 |
| 安全節點：整個回呼 | 同上 `node_ms` | ms | **wall** | 本輪新增 |
| 距離節點：整個週期 | `/arm_link_distance/diag` 欄 `node_cycle_ms` | ms | **wall** | 本輪新增 |
| 距離節點：G2 下界 | 同上 `tight_<pair>_elapsed_s`、`_lb`、`_ub`、`_tol_met` | s／m | **wall**／m | 逐配對 |
| 重複列移除數 | 同上 `dup_dropped` | 個 | — | 等價改善的計數 |
| 命令逾時／拒收 | `coman_chain9`（`chain9.summary()`）：`received`、`rejected`、`last_reject`、`frozen_steps`、`timeout_decel_steps`、`stop_unverified_steps`、`modified_steps`、`events` | 個 | `sim`（事件） | **`recv_seq`／`recv_sim_t` 是接收端編號與時間** |
| 安全層命令過期 | `/wholebody_safety/diag` 欄 `reason`（2 = stale-cmd） | 碼 | — | 逐週期 |
| 執行端命令年齡 | `coman_machine_log` 欄 `cmd_age_endpoint_s`、`safety_diag_age_s` | s | `sim` | 執行端自身 |
| 趟次總時間 | `sim_time_s`、`wall_s` | s | 兩者分開 | RTF 需自行計算 |
| 診斷錄製 | `runs/<id>/diag_record.json`：`cols`（**來自上游 `~/diag_fields`**）、`counts`、`data` | — | `sim` 戳 | 欄名不硬編碼 |

## 7. 依值配對的延遲估計

| 項目 | 來源 | 單位 | 時間 | 證據強度 |
|---|---|---|---|---|
| 延遲估計 | `coman_e2e_log` 欄 `e2e_age_s_value_paired_estimate` | s | `sim`（套用 − 來源） | **依值配對的估計，不是已證明同一筆命令** |
| 來源序號／時間 | 同上 `src_seq`、`src_sim_t` | —／s | `sim` | 由 `/coman/cmd_meta` 轉發 |
| 候選數 | 同上 `n_candidates` | 個 | — | 唯一相符才採用 |
| 配對狀態 | 同上 `pairing` ∈ {`value_paired_estimate`, `unpaired`} | — | — | `unpaired` 時年齡為 NaN |
| 配對鍵 | 手臂六分量（adapter 只旋轉底盤三分量） | — | — | 重複命令／遺失／延遲都可能誤配 |
| 說明欄 | `coman_e2e_note` | — | — | 已寫明不得單獨宣告時效通過 |

## 8. 趟次層級的來源與版本

| 項目 | 來源 |
|---|---|
| 資產與設定 sha | `sim/drawer_run.json` → `sha256_16`（`urdf`／`spec`／`cases`／`poses`） |
| 底盤模式 | 同檔 `base_fixation.mode`（`importer_fix_base`／free base） |
| 抓取模型 | 同檔 `grasp_model`（`fixed_attachment`／`friction`） |
| 停止原因 | 同檔 `stop_reason`；`monitor_failure` |
| 起動前檢查 | `runs/<id>/preflight.json` |
| 規格鏈 | v1／C1／R1.1／S1 的 sha 記於 S1；S1 另存 `checkers_prev_sha256_16` |
| 溫度 | `temp_start_c`、`temp_max_c`、`temp_max_sim_t`、`last_temp_c`、`cpu_limit_c` |

---

## 9. **缺項**（先回報，未修改執行端）

| 代號 | 缺什麼 | 影響 | 目前可否替代 |
|---|---|---|---|
| **G1** | 政策端的相位／事件時間戳（`approach_reached`／`attached`／`pull_done`／`hold_pass`／`decouple_confirmed`／`retreat_done`）只存在 `PullTaskPolicy.stamps`，**未輸出到任何檔案** | 「退出完成」與各相位邊界沒有權威時間戳 | **部分可替代**：求解端 log 的 `phase` 逐週期序列可反推相位邊界（20 Hz）；`coman_couple_link` 有解除確認；**退出完成只能由 `phase` 轉 `DONE` 推定** |
| **G2** | 求解端只記 `ep`／`er` 純量，**未記目標位姿**（`current_target()` 的 4×4） | 無法按相位算「實測 vs 目標」的位置／完整姿態誤差，只能拿到已算好的純量 | **不可完全替代**：`ep`／`er` 可用，但無法重算分量、也無法驗證誤差定義 |
| **G3** | **底盤實測位置與速度未記錄**。只有 `base_drift`（距起點距離純量）與 `base_dyaw_deg`（絕對值） | **直接影響第二題（拉動期間同動）**：距離純量的差分不是速度（方向資訊已丟失，往返會相消），絕對值偏擺也無法定向 | **不足**：可由 `coman_pose_log` 的 `grip_*` 反推？不行 —— 那是夾爪不是底盤。`/tf` 未落盤 |
| **G4** | 手臂關節**速度**未記錄 | 同動判定需要關節速率 | **可替代**：`joint1…joint6` 為 100 Hz 實測，差分可得速率；需說明差分與噪音 |
| **G5** | `events[]` 的結構不統一（`engage_by_handover` 是 dict，其他處直接 append `info`） | 事件解析需容錯 | 分析端以型別判斷處理 |
| **G6** | `diag_record.json` 的 `cols` 在未收到上游 `~/diag_fields` 時為 `null` | 無法按名稱取值 | **必須拒絕計算**，不得猜位置 |
| **G7** | **同動門檻不在任何已核准規格中**。v1 有 `profile`／`P_pull`／`B_base_tracking`／`continuity`，但**沒有** `base_lin_min_mps`／`joint_rate_min_rps`／`min_continuous_s` | 第二題（拉動期間同動）**無門檻可依** | **不可替代**：先前 0.005 m/s／0.005 rad/s 出自影片 README 的分析，**不是核准規格**。分析程式因此輸出 `cannot_determine`，不硬編 |

**G3 與 G7 會讓第二題無法回答**：G3 是資料缺、G7 是門檻缺，兩者都要補才判定得了。 G1／G2／G4 可用替代來源，但須在報告中說明口徑。
**本輪不修改執行端** —— 最小修改方案見 `evaluation/analysis/FIXED_BASE_GAP.md` §4。
