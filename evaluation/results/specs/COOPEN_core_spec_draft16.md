# 連續協同通行：核心段 — 規格 draft-16（bench8：一次性 gc.freeze 候選；暫停緩移對時）

draft-15（bench7）保留。依 Codex r30：本包只準備 GC 候選，並附暫停緩移的短分析。**不安排新實跑**，合併送審。

## 1. bench_5 封存口徑（Codex r30）
> 指定台架初態下，恢復開門段偏移更新的配置，於兩趟共同門角區間重現較小的鉛直偏差與持握漂移。本趟 OPEN 實測達 47.75°、參考達 48°，並量得出口距離指標下降 0.176 m；完成 SETTLE、進入暫停並實際凍結估計，但因停止後手臂緩移中止，未完成 10 秒停止觀察。

- 不宣稱穩健性、完整通行或停止驗證通過。
- 診斷版可能影響執行時序。兩趟相近只是描述性重現，不是嚴格的重複性結論。
- **停頓**：0.238 s 的 generation-2 GC 落在 0.248 s 的 world.step 區段內，約占 96%。
  - 尚不能定位是哪個物件配置觸發，也不是所有延遲的唯一來源。
  - 本趟幾何證據已超時（0.196＋0.023 s），不能因任務端只差 4 ms 沒中止就算時序通過。
- **末段漂移**：43.7–47.75° 最大 4.80 mm，距門檻 0.20 mm，列為下一趟必要觀察重點。不因此另開抓法搜尋、增加角度或放寬門檻。

## 2. 一次性 gc.freeze 候選（執行端 ev9r4；ev9r3 保留）
- **旗標** `--gc-freeze-before-release`：需 `--bench-start`，否則拒跑（62）；預設關閉，此時行為同 ev9r3。
- **時點**：台架凍結前置迴圈開始之前呼叫 `exec_gc_freeze.freeze_once()` **一次**。此時物理尚凍結（不呼叫 world.step）、正式釋放之前、釋放計時開始之前。
- **不做的事**：不關閉自動 GC、不改回收門檻、不 collect、不循環 freeze。ev9r3 的逐步區段與 GC 診斷全部保留。
- **紀錄**（room_run `gc_freeze`）：
  - Python 版本、freeze 耗時；
  - 前後的 gc.get_count、get_freeze_count、受追蹤物件數；
  - isenabled、get_threshold；
  - 行程記憶體（VmRSS、VmHWM、ru_maxrss）；
  - 結束時再記 final_state。
- **揭露**：
  - freeze 把當時受追蹤物件移到永久世代，之後回收忽略它們；
  - 被 freeze 的物件（含當時已是垃圾者）不再被回收；
  - 不保證新物件不再造成停頓（Python 3.12 gc.freeze 文件）；
  - Python 3.12 啟動時 freeze 數已可能不為 0（測試環境實測 375）。
- **跑法 bench8**：＝bench7，只換凍結 r8，執行端改為 ev9r4，並加旗標。所有門檻不變（0.2 s 監看時效、暫停 0.001 rad/s）。
- **測試**：
  - `test_exec_gc_freeze` G1–G5：
    - 子行程實測 freeze 數增加、GC 設定不變；
    - freeze 後新建的循環垃圾仍被自動回收；
    - 故障吞掉；模組不含 collect／disable／enable／set_threshold；
    - ev9r4 對 ev9r3 只插入，插入的 if 區塊不吞掉原有程式碼；freeze 只一次，位於 t_b0 之後、BenchRelease 之前。
  - `test_door_coopen_core_bench8_static` B1–B2。
  - 既有 `test_exec_timing_diag` X1–X7 仍通過。
- 準備過程揭露：ev9r4 初版把拒跑檢查插在 `if a.bench_start:` 區塊中間，會吞掉其後的 `try:`。語法仍合法，pyflakes 未抓到；已改到區塊之前，並新增 G5 的縮排檢查防止再犯。

## 3. 暫停緩移的有界對時（bench_5，暫停切換 103.02 s 至中止 108.04 s；只描述）
- 腳本：`door_coopen_core_bench_5_pause_creep.py`、`door_coopen_core_bench_5_pause_osc.py`。
- **設定點**：切換後的低速段變動最大 0.51 mrad，STOPPED（107.19 s）後**完全固定**（偏差 0）。STOPPED 後 applied_arm 六軸皆 0，套用來源鏈模式 0。
- **實測 j2**：比設定點低約 32 mrad，穩定存在（與 d̂ j2 −0.031 相符）。STOPPED 後到中止變化 +0.49 mrad，線性斜率 +3.4e-4 rad/s（STOPPED 前 2 s 為 −4.9e-5）。其餘軸 < 0.07 mrad。
- **力矩**：τj2 約 16–17 Nm（上限 50），STOPPED 後斜率 −0.23 Nm/s；τj3 約 −6.4 Nm。
- **手指**：
  - 驅動上限 103.02 s 切到 20 N，讀回 20／20，定期讀回皆 ok；
  - 指位固定（0.00165／0.00146）；
  - 兩指合力約 18.09 N（OPEN 期約 10 N），STOPPED 後斜率 +0.016 N/s。
- **持握振盪**：

| 窗口 | Fz std | 主頻 | f1 std | j2 去趨勢峰峰 |
|---|---|---|---|---|
| OPEN 60–65 | 6.9 N | 49.8 Hz | 0.58 N | 2.2 mrad |
| SETTLE | 5.2 N | 33.6 Hz | 1.06 N | 1.5 mrad |
| PAUSE 20 N 起 | 8.7 N | 50 Hz | 1.31 N | 2.5 mrad |
| STOPPED 後 | 8.7 N | 32.9 Hz | 1.15 N | 0.56 mrad |

  - 50 Hz 等於逐步交替，是接觸的數值顫動，OPEN 期就有；上限切到 20 N 後 Fz 與 f1 的波動變大。
  - 兩指 7↔11 N 互換、Fz 在 −5 到 +17 N 間擺動，合力不變。
- **判定窗口**：手臂 0.5 s 窗口在 107.96 s 首次超過 0.001（j2 0.00107），最大 0.00143 rad/s（108.02 s）；求解端 FAIL 107.99 s。
  - STOPPED 後 j2 去趨勢峰峰 0.505 mrad，光此項約相當於 0.5 s 窗口 1.0e-3 rad/s。
- **對照 w1f_60_3**（先前通過暫停 DONE，同為 20 N；**不同抓法與門角，非同條件**）：

| | 0.5 s 窗口最大 | j2 峰峰 | j2 斜率 | f1 std |
|---|---|---|---|---|
| w1f_60_3 | 0.00076 rad/s | 0.041 mrad | ≈ 0 | 0.21 N |
| bench_5 | 0.00107 rad/s | 0.505 mrad | 3.4e-4 rad/s | 1.13 N |

- **界線**：只描述時序。零速率命令不等於沒有位置伺服力矩；不把緩移直接歸因於接觸力提高，也不把 0.001 rad/s 改大。
