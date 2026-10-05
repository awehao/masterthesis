# D1 S4：一趟線上 shadow 功能確認——實作計畫（送審版，2026-10-05）

依 Codex reviews/20261005_143455_reply.md 第 C 節。問題：**能否在真實執行期間持續產生可追溯觀測，且不造成訊息積壓或接管控制**。不硬求 5 Hz；DONE 不代替感知驗收。

## 0　現況與缺口
- `--wrist-v0` 是獨立模式（擺位／重播＋每步算圖後 return），**不能與控制同跑**。
- 主迴圈已有錄影相機：`world.step(render=False)` 後到時間才 `world.render()` 一次。S4 沿用此模式，在腕部相機 5 Hz（模擬時間）擷取點才渲染。

## 1　模擬器：`--wrist-live`（預設關；關閉時程式路徑不進入，與現行行為逐位元相同）
- 相機建立與掛載沿用 `wrist_v0_capture` 的掛載矩陣、內參讀回（抽出成共用函式，**V0 模式行為不變**，以既有 V0 重播做回歸）。
- 每 0.2 s（模擬時間）在 `world.step` 後 `world.render()`，讀 `get_current_frame()`；**時間匹配**：影格 rendering_time 必須等於本步模擬時間（容差 0.25·dt），否則拒絕並計數（不以讀取當下姿態補戳）；rendering_frame 不變或缺深度 ⇒ 記 no_new_frame。
- 發布（戳＝擷取時間）：`/wrist/aligned_depth_to_color/image_raw`、`/wrist/color/camera_info`、`/wrist/pose`（PoseStamped，相機光學座標的世界位姿，擷取時刻）。RGB 只存檔供疊圖，不發布（減負載）。
- 真值（把手中心／軸、擷取時刻）只寫 `wrist_live/truth.jsonl` 給離線評估，**不發布**。
- 記 `wrist_live/capture.jsonl`：每次擷取的模擬時間、牆鐘、render 牆鐘耗時、結果（published／rejected 原因）。

## 2　shadow 節點 `d1_shadow_node.py`（獨立程序）
- 訂閱深度＋內參＋相機位姿，以**相同擷取戳**配對（不配對者計 unpaired）。
- 待處理槽最多一張：處理中來新影格 ⇒ 覆蓋舊的待處理影格並計 overwritten；工作執行緒呼叫**凍結版** `d1_handle_detect.detect()`（import，不改程式；freeze_d1 驗證）。
- 發布 `/d1/handle_obs`（JSON String）：擷取戳、層級 L0/L1/L2、拒絕原因、處理 ms、輸出時的模擬年齡與牆鐘年齡；**不發布任何任務目標或速度命令**；不訂閱控制話題。
- 記 `d1_shadow.jsonl`：每則收到／處理／覆蓋／拒絕／輸出事件與計數器。
- 從趟次開始即運行（涵蓋導航接近窗口）；模擬結束時寫摘要。

## 3　執行：一趟 MotM
- `run_nav_handover.sh` 加 `WRIST_LIVE=1`（傳 `--wrist-live` 並起 shadow 節點）；其餘與正式 MotM 相同（控制仍用真值目標）。`run_guarded.sh` 92 °C 熱中止。
- 不排整批；失敗照列。

## 4　稽核 `d1_s4_audit.py`（只讀實錄）
- 逐段對帳：擷取次數 → 發布（拒絕原因分列）→ 節點收到 → 配對 → 處理／覆蓋丟棄 → 偵測拒絕 → 輸出；各段數字必須閉合，不閉合列為缺口。
- 處理耗時分布、實際輸出率（模擬與牆鐘）、模擬／牆鐘年齡、連續有效窗（同 draft-2 定義）。
- 輸出結果以凍結 `evaluate()` 對 truth.jsonl 離線評估（距離分箱、誤認、拒絕正確性），**只作報告，不設 S4 門檻**。
- 隔離與負載：控制週期（求解節點時槽）、漏時槽、任務物理判定 S1–S6、熱紀錄、RTF；與同版本無相機 MotM 趟（phf_0*_M）比較**只作參考**，不以一趟差值宣稱相機純開銷。

## 5　測試（實跑前）
- 待處理槽邏輯單元測試（覆蓋、計數、戳不改寫）。
- `--wrist-live` 關閉時：短跑一趟比對 room_run 與既有同設定（或程式路徑檢查＋V0 回歸）。
- 節點對錄好的 V0 動態影格離線餵入（rosbag 不需要；直接以函式層呼叫）確認輸出與 S3 逐格一致。

## 待審問題
1. 在 5 Hz 擷取點才 `world.render()`（不每步渲染）是否可接受？rendering_time 與步時間不一致的影格一律拒絕。
2. RGB 不發布、只存檔是否可以（偵測只用深度，RGB 僅疊圖）？
3. S4 的「通過」定義：我提「各段對帳閉合、無積壓（待處理槽最多一張、年齡不隨時間增長）、控制話題零交集、任務 S1–S6 仍通過」；感知品質只報告。

---

# 修訂 r2（依 Codex reviews/20261005_144036_reply.md；實作與離線測試完成，待審後才跑）

## 時間匹配
- 模擬器每物理步在 `world.step()` **之後**以實際時間記相機 optical 世界位姿＋把手真值到有界 `PoseHistory`（1.0 s）；擷取排程也用 step 之後的時間。
- 新影格以 `rendering_time` 找**唯一**匹配步（容差 0.25·dt；多步命中＝ambiguous 拒絕；早於歷史＝older_than_history；其餘 no_pose_at_render_time）；發布擷取步的位姿與原始擷取戳。
- 不加暖機步、不瞬移、不呼叫 hold()：相機暖機在正常迴圈中發生，早期影格照實拒絕計數。
- 只在擷取點 `world.render()`；`rendering_frame` 未變記 no_new_frame；實際新影格率由稽核量測（不假設每次 render 都有新影格）。相機不設 frequency。

## 資料路徑界限與輸入契約
| 段 | 界限 | 淘汰／原因 |
|---|---|---|
| ROS 發布（模擬器） | KEEP_LAST 2 | 傳輸端丟棄在節點不可見 ⇒ 稽核以來源影格序號推為 transport_drop |
| ROS 接收（節點，每話題） | KEEP_LAST 2、RELIABLE | 同上 |
| 同戳四段配對 PairBuffer | 3 組、逾時 1.0 s（牆鐘，自第一段到達） | pair_timeout／pair_overflow／duplicate_<part>／shutdown_unpaired |
| 待處理槽 LatestSlot | 1 張 | 處理中來新影格 ⇒ 舊待處理影格 overwritten（記其 n、stamp 與覆蓋者 n） |
| 工作執行緒 | 1 | 收尾時槽內剩餘 ⇒ shutdown_unprocessed |

- 深度：`32FC1`、公尺、光軸深度（distance_to_image_plane 原值），little-endian，640×480，step 2560；非有限或 ≤ 0 無效（detect 只用 [0.1, 3.0] m）。frame_id `camera_color_optical_frame`。
- camera_info：同 frame_id、讀回 K。
- `/wrist/pose`：PoseStamped，**header.frame_id = odom（世界參考框）**，pose＝optical frame 在 odom 的位姿；四元數長度偏離 1 超過 1e-6 視為無效（不默默正規化）；位姿矩陣建構與 S3 `cam_T` 相同。
- `/wrist/capture_meta`：String JSON（n、stamp、rendering_frame、source_wall_t）。四段以完全相同的擷取戳配對。
- 牆鐘年齡＝輸出牆鐘 − 來源擷取匹配牆鐘（source_wall_t）；模擬年齡＝輸出當下 /clock − 擷取戳。
- 重現：影格序號 n、rendering_frame、處理順序（processing_order_n）、seed 0、偵測程式 sha256 都寫入紀錄；深度另存 `wrist_live/frames/fNNNN_depth_m.npz`（float32 m）。

## 通過定義（三部分；d1_s4_audit.py）
1. **通路與對帳**：至少 1 格 processed 並發布；以來源影格序號逐格對帳，每格恰一個去向（processed／overwritten／shutdown_unprocessed／contract_reject／evicted:<原因>／transport_drop），無重複、無無法對應的節點事件；節點有界收尾摘要存在。
2. **有效觀測**：ALIGN 之前至少 1 格 L2；全部拒絕 ⇒ 只能說通路連通。有效率與誤差只報告，不設門檻。
3. **任務與隔離**：S1–S6、求解節點時槽（n_missed_slot、n_dup_skip、n_warm_discard、n_sim_stall）、熱紀錄各自報告；節點只發布 `/d1/handle_obs`（靜態核對）；控制仍用真值。只稱**命令／資料介面隔離**（算圖同步占用模擬主迴圈）。年齡趨勢只作診斷。與 phf_0*_M（無相機，n_missed_slot 3／0／1）比較只作參考。
- 正常結束與 TERM 都有界收尾（工作執行緒最多等 1.5 s，運行器 TERM 後 3 s 才 KILL）。

## 執行
- 一趟：`PLAN=results/vision/d1_s4_schedule.tsv FREEZE=results/vision/freeze_d1_s4.sha256 BATCH=d1s4 OFFSET_MOVING=1 WRIST_LIVE=1 bash evaluation/run_plan_batch.sh`（MOTM、open 0.200、N 5，其餘與 phf 正式 MotM 相同）。
- 事後：motm_physical_check（S1–S6）、horizon_replay_check（重播）、`d1_s4_audit.py`。

## 已完成的離線測試
- `test_d1_shadow.py` 29/29：延遲 2 步且 float32 時間的唯一匹配、步間／早於歷史／缺時間／NaN 拒絕、歷史有界、重複時間 ambiguous；配對任意順序、逾時、溢位、重複段、收尾；待處理槽覆蓋回傳舊影格識別、戳不改寫；32FC1 往返（含 inf／NaN）、16UC1／frame_id／尺寸／endianness／長度／四元數拒絕；**S3 開發集 87 格經線上契約路徑與 d1_detect_rev2 逐格相同**。
- `test_d1_shadow_node.py`（ROS，不起模擬器，連跑兩次皆通過）：端到端配對、連發覆蓋、缺段逾時、16UC1 契約拒絕、TERM 後 3 s 內有界收尾並記錄未處理影格、輸出數＝處理數、觀測話題收齊、戳不改寫、處理順序、來源牆鐘年齡；稽核以來源影格逐格對帳閉合（連發時 ROS 接收佇列丟棄的影格歸為 transport_drop 或配對逾時）。
- 模擬器端 `--wrist-live` 未實跑（Isaac 才能驗）：render 節奏與新影格率、每步取位姿的負載只能在 S4 趟量測。

---

# 修訂 r3（依 Codex reviews/20261005_145312_reply.md 四項必修；偵測器與控制設定不變）

1. **對帳**：節點逐段記 `received` 事件（擷取戳、段名）。稽核去向新增 `received_no_fate`（節點收到但無去向 ⇒ 證據不足、FAIL），`transport_drop` 只用於節點完全沒收到任何一段（推定）。工作執行緒以 try/except 包住偵測與發布，例外記 `worker_error`（n、stage、原因）後繼續；收尾時仍在處理的影格記 `shutdown_in_progress`。通路 PASS 另需：worker_alive_at_exit = False、worker_error = 0、無 in-progress、received 各段／paired／published=processed 計數器與事件閉合。
2. **分母與連續窗**：有效率分「來源新影格（含覆蓋、逾時、傳輸遺失）」與「已處理影格」兩種；擷取端拒絕者距離未知另列。連續有效窗沿來源時序（已發布＋擷取端拒絕），任何非有效影格或間隔 > 0.4 s 中斷。已處理影格缺真值或內參 ⇒ 證據不足，不靜默略過。
3. **內參契約**：K 必須 9 個有限值且 fx、fy > 0（cx=NaN、長度 8、None 皆拒絕）；位置非有限或長度錯 ⇒ bad_pose。
4. **欄位**：溫度按欄名讀 `cpu_c`；`processed_capture_cadence_hz`（已處理影格的擷取時序頻率）與 `output_rate_sim_hz`（以節點輸出當下 /clock 的 out_sim_t）分開；`wall_age_s` 明示為「模擬器讀取匹配後 → 節點輸出，不含算圖延遲」。

測試：`test_d1_s4_audit.py` 18/18（含 Codex 反例 paired=2／processed=1／worker_alive=True ⇒ FAIL、received_no_fate 不補成傳輸遺失、worker_error、in-progress、計數器不閉合、兩種分母、覆蓋中斷連續窗、缺真值證據不足、cpu_c、時間欄名）；`test_d1_shadow.py` 34/34（新增 cx=NaN、K 長度、None、位置 NaN／長度）；`test_d1_shadow_node.py`（ROS 整合）通過；`test_d1_shadow_node_fault.py`（`--fault-inject-n`，僅測試用、預設不注入）9/9。
凍結：`freeze_d1_s4_r2.sha256`（原 `freeze_d1_s4.sha256` 保留不改）。

---

# 修訂 r4（依 Codex 複核；只改稽核判定與執行路徑）

- 通路 PASS 另需：摘要必要欄位存在且型別正確；`worker_alive_at_exit` **明確為 False**（缺值或其他值皆不通過）；
  `counters.worker_error` 為整數且等於 `worker_error` 事件數；**任何 `worker_error` 或 `shutdown_in_progress` 事件都不得 PASS**（不論摘要內容）；
  `fault_inject_n` 必須存在且為空（功能趟不得注入）。
- 測試 `test_d1_s4_audit.py` 25/25：新增摘要缺 worker_alive_at_exit、缺 worker_error 計數、事件有 worker_error 但摘要為 0、
  事件有 shutdown_in_progress 但摘要清空、注入集合非空、缺注入欄位，以及乾淨摘要仍 PASS。
- **執行指令（更正；批次腳本會切到工作區根目錄）**：
  `PLAN=evaluation/results/vision/d1_s4_schedule.tsv FREEZE=evaluation/results/vision/freeze_d1_s4_r3.sha256 BATCH=d1s4 DOMAIN=94 OFFSET_MOVING=1 WRIST_LIVE=1 bash evaluation/run_plan_batch.sh`
- 凍結 `freeze_d1_s4_r3.sha256`；`freeze_d1_s4.sha256`、`freeze_d1_s4_r2.sha256` 保留不改。

---

# 修訂 r5（d1s4_M startup_failure 之後；見 d1s4_M_startup_failure.yaml）

- rendering_frame 正規化（`norm_rendering_frame`）：主迴圈 render-on-demand 時 Isaac 回傳整數。
- **感知故障隔離**：模擬器以 try/except 包住 `wlive.on_step`；擷取例外 ⇒ 記 `capture_exception`、停用腕部擷取、控制照常。稽核見到 capture_exception ⇒ 通路 FAIL。
- 重跑（待 Codex 核可）：`PLAN=evaluation/results/vision/d1_s4_schedule_r2.tsv FREEZE=evaluation/results/vision/freeze_d1_s4_r4.sha256 BATCH=d1s4b DOMAIN=94 OFFSET_MOVING=1 WRIST_LIVE=1 bash evaluation/run_plan_batch.sh`；d1s4_M 保留不覆寫。

---

# 修訂 r6（Codex 複核：例外處置本身也要隔離）

- `wrist_live.guarded_step`：**先停用擷取**（回傳 None 讓主迴圈不再呼叫），再嘗試記錄故障；`fail()` 或記錄函式本身失敗都不再拋出。
- `wrist_live.safe_close`：相機資料寫出失敗只印出，不阻止 `room_run.json` 封存；模擬器只在 `a.wrist_live` 時呼叫。
- `test_wrist_live_guard.py` 12/12（fail 失敗、log 失敗、停用後不再呼叫、close 失敗仍封存、靜態檢查呼叫位置與旗標保護）。
- 重跑（Codex 核可條件：上述反例通過、新凍結吻合）：
  `PLAN=evaluation/results/vision/d1_s4_schedule_r2.tsv FREEZE=evaluation/results/vision/freeze_d1_s4_r5.sha256 BATCH=d1s4b DOMAIN=94 OFFSET_MOVING=1 WRIST_LIVE=1 bash evaluation/run_plan_batch.sh`
