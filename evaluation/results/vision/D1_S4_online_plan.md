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
