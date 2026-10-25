# W2b：ev6 執行端接線＋任務端 HELD_WAIT／請求＋權威 r3（**不實跑**）

依 Codex 20261025_214144：
- r2 已補足大部分缺口，可進入下一包「ev6＋任務端接線實作與假世界測試」，仍**不核可 Isaac 實跑**。
- 同包須修三項：
  1. 選定時先核請求期限，再以原始 `stamp_sim_t` 核對新鮮度（一般流一致）；
  2. 套用確認要核實際 `source_seq／mode_seq／chain_mode／九軸套用值`，提交步缺報或不符 ⇒ 有界失敗、不回切；
  3. 接線測試用真正的 E2 命令鏈。
- 其餘裁定照做：
  - adapter 沿用求解端時間戳；接收牆鐘另存；
  - 雙流只適用於測試模式；
  - 不張爪、不退離、不增加角度、不改門檻；
  - K3-B 在真實受控釋放前定版；
  - 不新增影片。

## 1. 差異表
| 檔案 | 來源 | 內容 |
|---|---|---|
| `door_w2_mode3.py` | 新（`door_w2_mode.py` 不改，子類別） | ModeAuthority3：select 次序與 stamp 新鮮度、逐步確認 confirm_step、送鏈序號不減 |
| `door_w2_exec.py` | 新 | W2Exec：on_env3（接收牆鐘另存）／pre_step（select → EnvIntake 進 E2 鏈）／post_step（applied_source＋限幅後九軸 → confirm_step）。ev6 與假世界共用 |
| `door_w2_check_mode3.py` | 新（`door_w2_check_mode.py` 不改） | 先跑 r2 的 C0–C7，再加 C8–C11 |
| `door_k3m_policy_w2.py` | 新（p2 策略子類別） | 任務端 HELD_WAIT → 請求 → 提交確認 → 測試模式保持 → DONE；各失敗路徑有界中止 |
| `isaac_door_k2_sim_ev6.py` | ev5 只插入（1687 → 1779；`results/door/mk_ev6_patch.py` 核對） | 見 §3 |
| `door_k3m_task_node_w2.py` | w1b2 只插入（569 → 606；`results/door/mk_task_w2_patch.py` 核對） | 見 §4 |
| `test_door_w2_mode3.py`／`test_door_w2_exec_chain.py`／`test_door_w2_check_mode3.py`／`test_door_k3m_policy_w2.py`／`test_w2b_wiring_static.py` | 新 | §5 |
| `test_w2_held_mode_ros3.py` | `test_w2_held_mode_ros.py` 的版本化複本 | 假世界執行端改為 W2Exec＋真 E2 鏈；新增 M9 |

## 2. 權威 r3 語意（door_w2_mode3）
**select（每物理步一次）**
- 次序：
  1. 已中止 ⇒ 不選定。
  2. 上一步尚未確認 ⇒ 中止 `confirm_missing`。
  3. pending：
     - 先核請求期限：t − 請求登記時刻 > 1.0 s ⇒ `candidate_timeout`，同步有新候選也不提交；
     - 候選以原始 stamp 核新鮮：t − stamp ≤ 0.2 s 且 stamp ≤ t＋0.011；不新鮮 ⇒ 丟棄該筆，等下一筆；
     - 候選序號 ≤ 已送鏈最大序號 ⇒ 丟棄（避免 EnvIntake 判 out_of_order）；
     - 驗證：有限、保持形、相對上一步**確認過的實際套用九軸**的加速度框，框以物理步 dt 計。通過 ⇒ 原子提交。
  4. 提交後：目前流最新 stamp 老化 > 0.25 s ⇒ 中止 `post_commit_source_lost`，不回切。
  5. 一般流同樣以 stamp 判新鮮；不新鮮 ⇒ 本步不選定（鏈自行以來源時間逾時）。
- Codex 兩個重現改為必拒：
  - stamp 0／接收 0.19／選定 0.30 ⇒ 不提交；
  - 請求 0／候選 1.01 ⇒ 逾時、不提交。

**confirm_step（每次 select 之後、下一次 select 之前恰一次）**
- 回報內容：`{source_seq, mode_seq, chain_mode, u（九軸限幅後實際值）}`。
  - 由 W2Exec 從鏈讀出：applied_source；mode_seq 取自進鏈時記下的對照。
  - 呼叫端不能指定來源。
- 步號不符 ⇒ 中止 `confirm_order`。
- 提交步：須同時滿足下列條件，否則 `applied_confirmed=False`、中止 `commit_not_applied:<why>`，不回切：
  - source_seq＝first_source_seq；
  - mode_seq＝1；
  - chain_mode＝0；
  - 九軸有限且保持形：底盤 ≤ 0.005 m/s／0.02 rad/s，手臂 ≤ 1e-6。
- 其他步：
  - 有來源但 mode_seq ≠ 本步權威序號 ⇒ 中止 `applied_mode_mismatch`；
  - 提交後有來源但非保持形 ⇒ 中止 `applied_not_mode_shaped`；
  - 提交後無來源或鏈逾時 ⇒ 只計數，由 0.25 s 有界中止。
- `confirm_applied`（r2 只帶序號的介面）在 r3 呼叫即拋例外，避免誤用。

**鏈接收時刻＝來源 stamp**
- W2Exec 進鏈時，以封裝的 `stamp_sim_t` 當 E2 鏈的接收時刻。
  - 鏈內 0.2 s 逾時因此從來源時間起算，與權威一致；不在進鏈時重新起算（Codex 第 1 項「一般流也須一致」）。
- **這比 ev5 嚴格**：ev5 從到達時刻起算，延遲多少就多算多少。
  - 開啟 W2 後，整趟（含開門段）都用這個時序。
  - 來源延遲大時較早進逾時減速；不放寬任何門檻。

## 3. ev6（只插入；`--w2-mode-run-id` 預設空 ⇒ 與 EV5 同行為）
- 需 `--cmd-env` 與 `--run-id`，缺一 ⇒ exit 41。
- **唯一入口 v3**：
  - 訂 `/wb_vel_cmd_env3`；回呼只解碼排入有界佇列（512，滿則丟棄計數），另存接收牆鐘；
  - v2 回呼在 W2 開啟時先 return，只計數。
- **請求轉送**：
  - 訂任務端 `/door/w2_mode_req_task`，交 on_request；
  - 結果為 pending 才發 `/door/w2_mode_req`（latched）給求解端。
  - 這樣避免候選比權威登記先到（否則判 candidate_without_request 違規）。
- **每物理步次序**（`test_w2b_wiring_static` S2 固定）：
  1. 佇列交 on_env3；
  2. 權威入口違規 ⇒ `_ev4_violation`，即既有 K2 閂鎖中止；
  3. 請求；
  4. pre_step 進鏈：K2 過濾照舊，EnvIntake 違規走同一路徑；
  5. ex.step；
  6. post_step 確認；
  7. 權威中止 ⇒ `_ev4_violation(w2_mode:<why>)`：鏈過期、逾時減速到停、夾爪凍結，不張爪、不回切；
  8. 每 2 步發 `/door/w2_mode_state`。
- `/door/applied_src` 照 EV4 發布，值即鏈的實際套用。
- `room_run.json` 另記 w2：權威統計、逐步列、權威確認列、接收紀錄（含接收牆鐘）。
- **權威入口違規一律閂鎖中止，從嚴。**
  - 包含「請求被拒後求解端尚未看到狀態、多送一筆候選」的競態。
  - 這只會發生在已失敗的請求之後，任務端也已中止，所以不影響正常路徑。

## 4. 任務節點 w2（只插入；`--w2-mode-run-id` 預設空 ⇒ 與 w1b2 同行為）
- 策略 K3MW2Policy：PAUSE 期間父類監看、夾持檢查、握手與 pause_total_s 25 s 全部照舊；只改「何時結束」。
- 流程：
  1. 求解端 HELD，且 observation_passed、physical_stop_confirmed ⇒ held_wait。此時**不結束、不送 /room/stop**。
  2. held_wait 滿 1.0 s ⇒ 送一次請求到 `/door/w2_mode_req_task`。
  3. 只以 `/door/w2_mode_state` 判斷：commit 且 applied_confirmed True、mode_seq 1 ⇒ committed。
  4. 保持 3.0 s ⇒ DONE（finish、stop_solver ⇒ 既有收尾送 /room/stop）。
- 有界中止條件：

  | 情況 | 中止原因 |
  |---|---|
  | 請求被拒 | `w2_request_rejected:<why>` |
  | 確認不符 | `w2_commit_not_applied` |
  | 請求後 2.0 s 未提交 | `w2_commit_timeout` |
  | 權威中止 | `w2_mode_aborted:<why>` |
  | 權威狀態停更 > 1.5 s | `w2_state_stale` |
  | HELD 有界超時 | `w2_held_timeout`（不算完成） |
  | 監督器離開 HELD | `w2_supervisor_left_held` |
  | 非超時的 DONE | `w2_unexpected_done` |

- **DONE 的意義只到**「同一命令流內轉到測試模式並保持 3 s」，不是釋放或退離完成。
- 時程估計：假世界暫停到 HELD 約 11.7 s，加 1 s、提交約 0.1–0.5 s、再加 3 s，約 16–17 s，小於 pause_total_s 25 s。門檻未改。

## 5. 測試（全過；均不需 Isaac）
| 檔案 | 項數 | 內容 |
|---|---|---|
| `test_door_w2_mode3.py` | 11 | R1–R12：Codex 兩個重現、一般流 stamp、確認缺失（提交前後）、提交步五類不符、提交後非保持形、提交前 mode 不符、序號不新、確認次序、r2 介面停用、斷流以 stamp 計、正常 |
| `test_door_w2_exec_chain.py` | 8 | **真 CmdChainE2＋EnvIntake＋applied_source**，見下 |
| `test_door_w2_check_mode3.py` | 8 | J0–J7：以真鏈 C1 紀錄逐項竄改，各自被對應項拒收 |
| `test_door_k3m_policy_w2.py` | 7 | P1–P10：正常時序（held_wait 10.0、請求 11.0、提交 11.2、DONE 14.2，只在 14.2 finish）與各失敗路徑 |
| `test_w2b_wiring_static.py` | 5 | 生成腳本重跑逐位元一致＋insert-only；ev6 主迴圈次序；v2 回呼先 return；話題 |
| `test_w2_held_mode_ros3.py` | M0–M9＋R0 | 真節點：ADM11＋adapter_w2＋替身端點；執行端＝W2Exec＋真 E2 鏈 |
| 既有 `test_door_w2_mode.py`／`test_door_w2_check_mode.py` | 20＋6 | 回歸，全過 |

**test_door_w2_exec_chain 各情境**
- C1：提交步實際首筆＝first_source_seq，profile 同步切換。
- C2：K2 過濾擋候選 ⇒ commit_not_applied:src_ne_first_source_seq。
- C3：鏈拒收、失效 ⇒ commit_not_applied（鏈模式 3）。
- C4：限幅：權威框寬鬆（0.05 s）時候選過、鏈限幅後 0.0095 m/s ⇒ base_not_hold 中止；框以物理步計 ⇒ accel_box 先拒。
- C5：斷流：鏈 stamp＋0.21 逾時、權威 stamp＋0.26 中止。
- C6：Codex 重現（真鏈）。
- C7：缺確認。
- C8：一般流老化時權威與鏈一致。

**假世界 r3 結果**（`/tmp/claude-1000/w2_mode_ros3`）：
- M1：提交 k＝1869、first＝374，C0–C11 全過。鏈無失效、無入口違規；鏈底盤變化率削減共 8 步（本測試未逐步定位發生時段）。
- M2–M8：全過。
  - M3：stale 1；
  - M5：candidate_timeout 於 8.51、HELD 超時結束；
  - M6：post_commit_source_lost 19.39、mode 仍 1；
  - M7：未提交、reverted；
  - M8：延遲 0.5 s 後 19.19 提交，C0–C11 全過。
- M9、R0：見 §7。第二次實跑（手臂限位改 ±100 rad 後）全部通過；首次實跑 M1 因 ±3 rad 限位假象失敗，隨即停止並修正（§6）。

## 6. 揭露與限制
- **ev6 與任務節點 w2 未端到端執行**：Isaac 未核可，任務節點需完整場景輸入。只做到：
  - 共用類別（W2Exec、ModeAuthority3、K3MW2Policy）的單元與假世界測試；
  - 原始碼次序靜態核對；
  - py_compile 與 pyflakes：ev6 新增碼無新警告，警告與 ev5 同一組既有項。
- 假世界手臂限位 ±100 rad：替身 `/door/sync_state` 手臂角恆為 0，首次實跑在約 6 s 因 joint3 設定點積分越過 ±3 rad 而鏈失效，屬替身假象。本測試不驗限位。
- R8 單元測試直接改權威的候選緩衝，模擬「較舊候選較晚可見」；同情境在真鏈測試中未自然出現。
- 權威加速度框改以物理步 dt（0.01 s）計，與鏈的底盤變化率上限一致，比 r2 預設 0.05 s 嚴。
- 雙流只適用於測試模式（候選與保持命令相同）；不外推到真實退離控制。
- 假世界是測試程序，不是 Isaac 執行端：時序與鏈路延遲不同，只驗證機制，不等於安全保證。

## 7. M9 與 R0 結果
**M9：提交步 K2 過濾擋下候選（真鏈）**
- 過濾恰生效一次：feed filtered＝1。
- 提交步 1869：
  - 選定 375（候選）被過濾；
  - 鏈讀出的實際套用仍是 mode 0 舊快照 src 373、chain_mode 0；
  - 權威判 `commit_not_applied:src_ne_first_source_seq`，applied_confirmed False，mode_seq 仍 1（不回切）。
- 中止後 1331 步權威不再選定任何命令；鏈隨後走逾時減速（n_timeout_decel 1317）。

**R0：ADM11 不帶 --w2-***
- 暫停照常 DONE；stats.w2 為 None；/wgmpc/cmd_env3 無任何訊息。

## 8. 凍結
`results/door/freeze_w2b_dev_r1.sha256`：新檔、生成腳本、版本化複本與其來源、相依模組、本規格。**所有文件改完後最後建立。**
