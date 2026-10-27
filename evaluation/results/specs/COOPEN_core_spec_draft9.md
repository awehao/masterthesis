# 連續協同通行：核心段 — 規格 draft-9（台架初態啟動，bench 版）

draft-1～8 保留。本版依 Codex r19（核可初始設定點預補償候選）與 r20（核可台架專用「偏移估計登錄初值＝d_init」）。

**成果口徑**：「指定初態（台架啟動：預補償設定點＋偏移估計初值）、經實際夾持取證後的持握開門與出口繞行核心驗證」。
- 跳過自主接近；不代表接近已解決；不稱完整自主開門通行。
- `chain_pass` 不等於邊開邊繞成功；夾持、實測開門、開門同期出口進展分項報告。

## 1. 啟動契約（`door_bench_start.py`；各節點旗標預設關閉 ⇒ 原行為）

| 項目 | 內容 |
|---|---|
| 名目初態 | 關節生成於 q_nominal（＝qa_grasp）。執行端 `--stow-q`、登錄、實測 t0 分開記錄；不符即拒跑 |
| 寫入設定點 | 手臂致動器初始目標 s_init ＝ q_nominal − d_init。只改初始設定點；不改重力、增益、碰撞、實測狀態，也不持續重設姿態 |
| 控制權 | 初始控制者為 wholebody；不經 nav→wholebody 交棒，不偽稱交棒通過 |
| 命令鏈種子 | 以 (u＝0, 設定點＝s_init, source＝bench_start) 建立。真 E2 鏈測試確認首筆命令後設定點＝s_init＋r·dt，不被實測覆寫 |
| 停車保持 | 監看自 t0 起，錨點＝生成位姿（bench_anchor）。靜止閘門不使用，`passed` 維持 False；`/park/gate` 另帶 bench、bench_anchor |
| 凍結前置 | 釋放前不呼叫 `world.step`（模擬時間停在 t0）。只重發 t0 的時鐘、關節、里程、同步狀態、把手、門角、控制權狀態、設定點回報（步號 0、ready 1、api 1）、套用回報（步號 0、零命令、api 0）、停車閘門 |
| 釋放 | 求解端 `/bench/first_solution` 宣告第一筆**正常求解**的 source_seq，且執行端最新全身封裝的 source_seq 與之相同。握手零命令不算解。牆鐘逾時 120 s ⇒ 不釋放、中止 |
| 求解端初值 | IntervalOffsetObserver 既有 d0 介面，d0＝d_init（六軸有限、≤ 0.05，超界拒跑不截斷）。第一次預測與求解即用 bias＝α⊙d_init；n_upd 仍 0；後續照真實有效區間更新；OPEN 凍結規則不變 |
| 預熱 | 凍結期時鐘不前進。冷啟動首解若超過 `--max-input-age` 0.2 s，會被既有規則丟棄（ROS 測試實測 728 ms），因此第一次正式求解前以同一初始快照先解一次。結果丟棄、不發布、不寫入暖啟動或估計狀態；丟棄規則不放寬 |

## 2. 初值來源（待驗證的初始化估計）
- d_init ＝ mean(q_arm_meas − arm_sp)：g3_1 第三次暫停 [25.28, 26.40) s，112 步。
  - 有效資料規則：證據 |fn|＝0、聚合 0、無 contacts_err、sep ≥ 3.8 mm。
- 不確定度：(0.0002, 0.0014, 0.0016, 0.0025, 0.0002, 0.0024) rad。
- 這組值含動態追蹤落後，不是已辨識的靜態偏移；不確定度也不是已證明的上界。
- 來源檔：`door_coopen_bench_release_offline_r2.json`（sha 記於登錄 r9）。

## 3. 版本化實作（只插入；差異經靜態測試限定）

| 檔案 | 版本關係 | 刪改的非註解行 | 主要內容 |
|---|---|---|---|
| `isaac_door_k2_sim_ev9.py` | ev5 版本化 | 4 行，各有旗標關閉回退 | B1–B5 |
| `dl3_wgmpc_wg2_adm17_node.py` | ADM9 版本化 | 1 行（觀測器建構帶 d0；無 bench 時 None） | C1–C5 |
| `door_coopen_core_task_node7.py` | v6 版本化 | 1 行（策略建構） | E1–E5；策略外層加 `door_coopen_bench_grasp`：起始 BENCH_START → 已釋放且時間前進 → INIT_CHECK（init_grasp2 r2 連續確認與監看擴充不變） |
| `run_door_coopen_core_bench.sh` | iso2 版本化 | — | ev9、ADM17（無 `--wait-for-drawer-handover`）、v7、**不啟動展開節點**；程序索引重排 |

- **揭露**：BENCH_START（凍結期）不呼叫監看，因為物理不演化、執行端不發接觸流；以 `bench_stats.monitor_skipped_cycles` 記錄。釋放後 INIT_CHECK 起全部監看照常。

## 4. 判定器 r9（`door_coopen_check_core9.py`）＝ r4 鏈（J1–J4）＋ J6 r3 ＋ J7 台架

### 鏈窗口
- door_k3a_check 以 UNFOLD 事件為窗口起點；沒有 UNFOLD 時記 `no_unfold_event` 並提前返回，後續鏈檢查會全部被跳過。
- r9 只在呼叫鏈判定期間，把 `phase_windows` 加上 UNFOLD＝BENCH_START 的別名。紀錄本身不改；結果記 `chain_window_origin`。
- `park_anchor_missing` 在台架流程必然出現，改由 J7 bench_anchor 取代，列入 `chain_exempt`。其他鏈理由照算。

### J6 r3
- 沿用 r8（|fn| 判接觸、聚合對帳、確認窗逐列核對）。
- 以下兩項由 J7 取代，列入 `iso_replaced`：
  - `iso_phase_sequence`
  - `iso_setpoint_mismatch`：台架刻意寫入 s_init，與實測相差 d_init。

### J7 理由
- `bench_reg_invalid`
- `bench_phase_sequence`：首相位須為 BENCH_START，須含 INIT_CHECK → ENGAGE；不得出現 WAIT_HANDOVER／UNFOLD／ALIGN／APPROACH。
- `bench_setpoint_written`
- `bench_initial_state`
- `bench_release`：執行端釋放 seq ＝ 宣告 ＝ 求解端 announced；kind solution；sim_t＝t0。
- `bench_offset_init`：d0＝d_init、n_upd 0、預熱未發布。
- `bench_anchor`
- `bench_evidence_missing`

## 5. 測試（全過）

| 測試檔 | 項數 | 內容 |
|---|---|---|
| test_door_bench_start | 7 | 見下 |
| test_door_bench_start_ros | 3 情境 | 真 ADM17 ＋ 台架替身，見下 |
| test_door_coopen_check_core9 | 4 | 正常；19 個反例；窗口別名期間生效、結束後還原；豁免與替換清單限定 |
| test_door_coopen_core_bench_static | 4 | ev9／ADM17／v7／跑法差異限定 |
| 回歸 | 85 | check_core8 11、init_grasp2 10、gate2 13、k3a_policy 18、settle_p2 11、bench 系列 |

test_door_bench_start：
- BenchConfig 的超界、不一致、非六軸、NaN、bool、CLI 不符；
- BenchRelease 的 run_id、seq 非法、sim_t≠t0、宣告前封裝、seq 不符、逾時、衝突；
- 真 E2 鏈種子：不被實測覆寫；重複種子拒絕；旗標關閉時原行為；
- 任務端起始相位（K3MP2 組合）：凍結期發目標、不監看；釋放後才 INIT_CHECK → DONE；釋放後碰桿即中止；台架中止即 bench_release_failed；
- 觀測器 d0。

test_door_bench_start_ros：
- R1 一致：t0 首解宣告；同 seq 釋放；握手零命令不當首解；預熱未發布；d0、首列 bias＝α⊙d_init；釋放後 1 s 偏離 0.00000 rad。
- R2 壓力（受控體偏差＝d_init＋不確定度，即受控體偏差≠登錄初值）：釋放正常、無失敗，1 s 偏離最大 0.0019 rad。
- R3 旗標關閉回歸：無 bench、偏移由 0 起、有發布。

## 6. 已知限制
- 替身執行端沒有物理與接觸；Isaac 端 ev9 的凍結／釋放只經靜態差異與替身協定測試，未在 Isaac 跑過。
- 手指—橫桿在 qa_grasp 開指時只有 5.15 mm（離線膠囊模型）。理想化離線下，d̂ 一致初始化時釋放為靜態；實際 Isaac 仍可能因模型差而接觸。若接觸，由 J6／J7 照實判失敗。
- 單趟申請：`door_coopen_core_bench_1`，凍結 `freeze_door_coopen_core_bench_dev_r1`；乾淨環境、92 °C 守護、跑前／跑後核對、失敗不補跑、不新增影片。
