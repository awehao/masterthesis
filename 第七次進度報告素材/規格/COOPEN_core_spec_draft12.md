# 連續協同通行：核心段 — 規格 draft-12（台架初態啟動，bench4 版）

draft-11（bench3）保留。本版只修正 `door_coopen_core_bench_1` 暴露的啟動順序缺口；**其餘全同 draft-11**。

## 1. bench_1 結果（失敗，保留不補跑）
- 凍結前置正常：模擬時間停在 t0＝0.0200 s；名目初態與寫入設定點照登錄；執行端 9851 輪持續發 t0 資料與台架錨點。
- **求解端一次都沒解**：握手在 t0 瞬間完成（init→armed 都在 0.02 s），預熱 68 ms 後，第一輪即以 `park_mode_refused：沒有停車錨點` 停機。
  - 原因：每次 spin 只處理一個回呼；100 Hz 的關節、里程、設定點訊息尚未處理完，/park/gate 還沒被處理到。
  - 正常流程中，求解端會先等展開節點交出約 8 s，錨點早已收到，所以從未暴露。
- 執行端未收到首解宣告，120 s 逾時、不釋放、中止。**物理從未前進，無接觸、無運動。**
  - 任務端隨執行端回報中止（bench_release_failed）。
  - 判定器 r11 照實不通過：bench_release、bench_first_*、iso_evidence_missing 等。

## 2. 修正：求解端 ADM17r2（ADM17 保留）
- **(C6)** 台架模式且 `--park-hold` 時，預熱之前持續處理回呼，直到收到 bench_anchor；上限 `--bench-anchor-wait-s`（牆鐘，預設 10 s），逾時即拒跑。
- 只改啟動順序，不改閘門、任何判準或保護。等待時間記入 bench.anchor_wait_s。
- 跑法 bench4 ＝ bench3，只換求解端與凍結。替身執行端 r2 只新增 `--gate-delay-wall-s`，r1 保留。

## 3. 測試（test_door_bench_start_ros2，真求解節點）

| 情境 | 設定 | 結果 |
|---|---|---|
| R4 重現 | ADM17（r1）＋ /park/gate 延後 1.5 s | park_mode_refused、未釋放（與 bench_1 相同） |
| R5 修正 | ADM17r2 ＋ 同一延後 | 等到錨點 0.48 s；宣告首解、同 seq 釋放；求解無失敗 |
| R6 逾時 | ADM17r2 ＋ 不發 /park/gate、等待 2 s | 拒跑 |
| R7 一致 | ADM17r2 無延後 | 宣告與同 seq 釋放、d0＝d_init、首列 bias＝α⊙d_init |

## 4. 申請
- 單趟 `door_coopen_core_bench_2`，跑法 bench4，凍結 `freeze_door_coopen_core_bench_dev_r4`。
- 範圍與條件同 draft-11。
