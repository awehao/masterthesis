# 連續協同通行：核心段 — 規格 draft-15（bench7：bench_4 控制配置＋執行端停頓診斷）

draft-14（bench6）保留。依 Codex r29：沿用 bench_4 控制配置，**只加觀測**。控制路徑、保護與所有門檻都不變，0.2 s 監看時效也不放寬。

## 1. bench_4 封存口徑（Codex r29）
> 單趟恢復開門段偏移更新後，在共同門角區間觀察到較小的鉛直偏差與持握漂移，並開至 45.70°、越過前趟中止角度；本趟因觀測資料逾時中止，未完成 48° 及受控收尾。結果支持保留此候選，不構成重複性或因果證明。

- 不以中止後的漂移超限，倒推正常控制期間已失守。
- 未進入暫停，不宣稱暫停凍結已實跑驗證。
- 直接證據只有「任務端約 0.26 s 未取得推進的新觀測」。執行端停頓、發布、傳輸、回呼處理與排程延遲還沒有區分。
- GC 只是假設。

## 2. 診斷（只觀測）
- **模組** `exec_timing_diag.StepTimer`：
  - 預先配置的有界陣列，大小為 sim_limit／dt＋16 列 × 8 個標記，約 4.3 MB。
  - 時鐘用 `time.monotonic`，與任務端 `enq_wall_mono` 是同一個系統時鐘，可跨行程配對。
- **每輪標記與區段**：

  | 標記之間 | 區段 |
  |---|---|
  | loop_start → after_callbacks | 回呼處理（含 /clock 發布） |
  | → before_exec_step | 命令處理 |
  | → after_exec_step | ex.step |
  | → after_reads | 套用後的姿態、門角、接觸與力讀取 |
  | → after_evidence | 取證 |
  | → before_world_step | 紀錄與步前發布 |
  | → after_world_step | world.step |
  | → 下一輪 loop_start | 步後狀態、接觸、掃描發布 |

  每列另記步號與模擬時刻。
- **GC**：只掛 `gc.callbacks`，記 start／stop、generation、collected、當下列號。
  - **不呼叫、不關閉、不手動觸發 GC，也不改門檻。**
- **故障處理**：診斷錯誤一律吞掉計數；溢位只計數。結束時先卸除回呼，再寫 `<out>/exec_timing.npz`，並在 room_run 寫 `exec_timing` 摘要（區段分位數、最慢 20 輪、> 10 ms 的 GC）。
- **開銷**：離線量測每輪 8 個標記約 1.9 µs。
- **限制**：診斷本身可能改變停頓時點；沒重現不等於修好。

## 3. 版本
- **執行端** `isaac_door_k2_sim_ev9r3.py`：對 ev9r2 只做插入。靜態測試確認沒有任何刪改行，新增的非註解行都只涉及診斷。
- **跑法** `run_door_coopen_core_bench7.sh`：對 bench6 只換凍結 r7、執行端與開場字樣。
- **測試**：
  - `test_exec_timing_diag` X1–X7：區段、有界、真 GC 回呼與卸除、故障吞掉、不改 GC 行為、存檔、只插入；
  - `test_door_coopen_core_bench7_static` B1–B2。

## 4. 單趟（Codex r29 條件式核可）
- `door_coopen_core_bench_5` 跑一次；失敗保留，不自動補跑。
- 回報重點：停頓落在哪個區段。
