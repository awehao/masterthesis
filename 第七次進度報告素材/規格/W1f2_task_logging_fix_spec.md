# W1f2：任務端最小紀錄修正——規格＋實作＋反例測試＋凍結 w1f2 r1（**未開跑，待核可**）

依 Codex 20261025_034022：
- 先核可「任務端紀錄方案」的版本化實作與離線／假世界測試，不核可實跑。
- 原 trace_gap＝0.25 s 不變；同步重寫對照只作佐證，不據此回改 w1f_60_1。

## 1. 背景（只描述）
- w1f_60_1：第 1000 筆 trace 之後，任務端出現 0.30 s 更新間隙（來源步 8138→8168、目標序號 905→906）。
- 候選原因：任務節點每 100 筆在同一迴圈同步重寫整份 task.json。百位邊界的間隙隨筆數增大（0.08 → 0.30 s），非百位邊界最大 0.12 s。
- 沒有寫檔耗時或牆鐘紀錄 ⇒ **未確認根因**。

## 2. 唯一差異
| 項 | W1f（w1b 任務節點） | W1f2（w1b2 任務節點） |
|---|---|---|
| 迴圈內紀錄 | 每 100 筆在同一迴圈 `save()` 重寫整份 task.json | **移除**（唯一刪改的 2 行）；改為每列深拷貝後 put_nowait 進有界佇列（容量 4096） |
| 序列化／寫入／flush／心跳 | 在迴圈內（每 100 筆） | 全部在背景執行緒：`<out>_trace.jsonl` 逐行 append、0.5 s flush、1 s 覆寫 `<out>_trace_hb.json` |
| 每列欄位 | — | 另記 `wall_mono`（本輪迴圈起點單調時間）、`loop_seq`、`trace_seq` |
| 迴圈時序 | — | LoopTimer：每輪起迄耗時（最大、p99、> 0.1 s 次數、最差一輪） |
| 收尾 | `save()` 一次 | 先以 5 s 期限關閉寫入端（未寫完、丟筆、寫入失敗 ⇒ complete False，不無限等待），再在迴圈外 `save()` 一次；task.json 另記 trace_log、loop_timing |
| 判定器 | w1e2 | **w1f2**＝w1e2 全部判準（含 trace_gap 0.25 s）＋紀錄完整性 L1–L4 |

**其餘不變**
- 任務邏輯、策略、話題、門檻、登錄（w1f_params_60_r2）、求解端、執行端都不變。
- 任務節點以生成腳本 `results/door/mk_w1b2_patch.py` 產生，核對「原檔除被刪 2 行外，其餘行依序全在新檔中」（544 → 562 行）。

## 3. 判定器 w1f2 加查
- **L1**：trace_log.complete 為嚴格 True；丟筆、寫入失敗、未寫完皆為 0；drain_timeout 為 False；寫入數＝入列數＝trace 筆數。
- **L2**：JSONL 行數＝寫入數；enq_seq 與 trace_seq 自 1 起連續；與 task.json trace 同序同筆（t、state_step、phase）。
- **L3**：state_step 不得倒退；同步號重複只計數。
  - 這與 Codex「嚴格前進」的字面不同。依據：三趟既有 trace 都有 1–4 次同步號重複，都是兩輪之間沒有新同步狀態，沒有倒退；task trace 是低於物理頻率的資料，不要求逐物理步連號。
  - 間隙仍由既有 trace_gap 0.25 s 判。
- **L4**（只報告）：wall_mono 最大間隔與位置、來源模擬時間最大間隔與位置、迴圈耗時統計、寫入端最大佇列深度與入列最大耗時。
  - 用來區分三種情形：來源停更、迴圈延遲、寫入延遲。

## 4. 反例測試（`test_bounded_trace_writer.py` 10 項，全過）
**寫入端**
- 正常：全部寫入、順序、心跳檔、重複 close。
- 不可變：入列後再改 row，已寫內容不變。
- 慢寫入（每筆 20 ms；生產 200 筆，每 2 ms 一筆）：
  - put 最大耗時 < 10 ms，仿主迴圈週期不被拖慢、佇列深度 > 50；
  - 寬限足夠 ⇒ complete；
  - 寬限過短 ⇒ drain_timeout、complete False，且收尾 < 1 s（有期限）。
- 溢出（容量 10）：丟筆有計數、complete False。
- 寫入例外：失敗 1 筆、complete False，其餘照寫。
- 同步重寫對照（佐證，不作根因）：舊做法的單次重寫耗時隨筆數增加；新做法 put 耗時與筆數無關。

**判定器**
- 完整假趟 ⇒ 通過。
- 下列反例各自不通過：缺 trace_log、丟筆、寫入失敗、complete 非嚴格 True、drain_timeout 非 False、trace 少一筆、JSONL 不可讀、JSONL 多一行、步號倒退（含來源倒退）、與 task.json 不符、wall_mono 缺。
- 步號重複只計數。
- 主迴圈卡住 0.5 s ⇒ L4 指出位置與間隔；trace_gap 由既有判準判定。

**界線**：背景執行緒仍可能競爭 GIL／CPU；put_nowait 不等於已證明時序有界。實跑時以 wall_mono 與迴圈時序事後核對。

## 5. 凍結與跑法
- 凍結 `freeze_door_wide_w1f2_dev_r1.sha256`：w1f r2 全部＋新增檔。
- 執行腳本 `run_door_wide_w1f2.sh`：與 run_door_wide_w1f.sh 只差任務節點、凍結、判定器名稱。
- **核可後**才申請單趟 60° 驗證 `door_wide_w1f_60_2`。本輪不跑、不推進角度。
