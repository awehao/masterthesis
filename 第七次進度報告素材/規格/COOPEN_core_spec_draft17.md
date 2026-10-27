# 連續協同通行：核心段 — 規格 draft-17（bench9：聯合配置驗證 door_coopen_core_bench_6）

draft-16（bench8）保留。依 Codex r31：
- gc.freeze 候選加上開跑條件；
- 暫停期手指上限維持 5 N，作為單一控制候選。

完成後條件式核可，直接跑一次 `door_coopen_core_bench_6`。

**相對 bench_5 同時有兩項變更**：GC 處置、暫停上限。⇒ 這是**聯合配置驗證**，不作嚴格單變因歸因。

## 1. 分析措辭收窄（Codex r31 §3；修正 draft-16 §3）
- 100 Hz 取樣中約 50 Hz 的成分，只支持「**接近逐步交替**」；不足以定性為接觸數值顫動的唯一機制。
- 去趨勢峰峰值除以 0.5 s **不是**實際窗口速度的貢獻量，不能直接與趨勢相加來解釋超限。判定仍以原始、已登錄的窗口算法為準。

## 2. gc.freeze 啟用失敗拒跑（執行端 ev9r5；ev9r4 保留）
- `exec_gc_freeze_check.verify`：旗標啟用時，下列任一成立 ⇒ 不通過：
  - ok 不為 True；
  - 紀錄缺欄；
  - GC 原本未啟用；
  - freeze 前後 enabled 或 threshold 不一致；
  - freeze 數未增加。
- **不通過 ⇒ 物理釋放前拒跑並封存**：
  - 台架釋放狀態記 aborted（`gc_freeze：理由`），並發布 /bench/state aborted；
  - 凍結等待迴圈一開始就離開（不 spin）；
  - bench_go False ⇒ 不進入物理循環；
  - 照常封存 room_run，`gc_freeze.refused` 記理由。
- **揭露**：拒跑時，既有的「凍結前置」訊息與一次 frozen 狀態仍會先發出，之後才發 aborted。
- **失敗注入測試**（`test_exec_gc_freeze_check` V1–V5）：
  - 直接執行 ev9r5 實際原始碼中自 verify 至 bench_go 的片段，用替身物件跑；
  - 7 種反例都沒有呼叫 spin_once，bench_go 為 False，並發布 aborted；
  - 正例與旗標關閉時，都會進入凍結等待；
  - 靜態檢查：只插入。

## 3. 暫停期手指上限維持 5 N（登錄 r11；r10 保留）
- `executor_change.pause_finger_max_effort` 20 → **5**（＝original_effort）。另記 r10 的值、rule_r11、basis_r11（含 bench_5 兩份分析的 sha256）。
- **執行端照既有 W1e 流程**：暫停切換步 set 5 並讀回、之後每秒讀回；不以 0 關閉驗證；不做 20→5 的卸載流程。
- **不變**：閉爪目標、增益、摩擦、偏移凍結時點、10 s 觀察與所有保護門檻（0.2 s、0.001 rad/s、5 mm 等）。
- **依據**（Codex r31）：
  - 本抓法在 5 N 上限下已完成 HOLD、長段 OPEN 與 SETTLE；
  - bench_5 切到 20 N 後承載與波動改變；
  - STOPPED 後設定點固定，實測仍動。
- **尚未證明 20 N 是根因**；只限目前這個抓法與配置，不撤銷其他抓法的 W1e 結論。
- **限制**：5→5 的讀回只能證明數值等於 5，無法區分 set 呼叫是否生效。測試以替身的呼叫紀錄確認確實有呼叫。
- **測試** `test_door_k3m_finger_effort_5to5` Q1–Q4：
  - 模組正例：仍啟用、呼叫一次、讀回 5／5；
  - 截斷、被改回 20、set 例外 ⇒ 拒；
  - 判定正例無理由；讀回 20、set 20、執行端仍用 20、定期讀回 20、缺 pause 紀錄、bench_5 型 20 N 紀錄 ⇒ 各自拒；
  - 登錄 r11 對 r10 只差此項。

## 4. 跑法 bench9（bench8 保留）
- ＝bench8，只換凍結 r9、登錄 r11、執行端 ev9r5 與開場字樣（`test_door_coopen_core_bench9_static` N1–N3）。

## 5. 單趟 door_coopen_core_bench_6（Codex r31 條件式核可）
- **分報**：
  - GC 處置紀錄與時效（exec_timing、任務端 state_age）；
  - 開門同期出口進展；
  - 末段漂移（43.7–48°，必要觀察重點）；
  - 兩指承載；
  - 完整 10 s 停止觀察。
- 乾淨環境、92 °C 守護、跑前後核對；失敗保留、不自動補跑、不新增影片。
