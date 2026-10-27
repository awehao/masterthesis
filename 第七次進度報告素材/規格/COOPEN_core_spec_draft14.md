# 連續協同通行：核心段 — 規格 draft-14（台架初態啟動，bench6 版：開門段恢復偏移估計更新）

draft-13（bench5）保留。本版只做 Codex r27 核可準備的**唯一候選**：OPEN 期間恢復既有 interval 觀測器的合格區間更新；**其餘全同 draft-13**。
成果口徑不變：「指定初態、經實際夾持取證後的持握開門與出口繞行核心驗證」；不代表自主接近。

## 1. 依據（door_coopen_core_bench_3，開迴路分析，非閉迴路證據）
- bench_3 在 105.32 s 因持握漂移 5.10 mm＞5 mm 中止；中止前最大門角 43.72°。
- 末段漂移以把手 z 分量為主（−3.9 mm）。
- 末段求解核對（r26）：規劃每輪都把鉛直誤差往 0 修（+0.55／+1.07／+1.20 mm），同期實測幾乎不動；預測起點比實測高 1.1→2.0 mm。
- 偏移估計在 4.72 s 首次切 free 時凍結（`--offset-hold-on-free`），整段 OPEN 未再更新（j2 −0.0234）。
- 因果影子重播（r27）：從同一凍結值起照既有規則逐區間更新，鉛直一區間殘差 RMS 0.776 → 0.045 mm；各門角段系統偏差 −0.62～−1.36 mm → ±0.04 mm；
  影子 d̂ j2 −0.0232 → −0.0307，平順、未觸限、單區間最大變化 2e-4 rad。
- **界線**：重播是開迴路，看不到改用新估計後的命令與接觸力；原凍結是為了避免把接觸負載學成偏差、加重夾持內力，這個風險**未被排除**，只能由實跑分報觀察。

## 2. 變更（單一變因）

### 求解端 ADM17r4（`dl3_wgmpc_wg2_adm17r4_node.py`；r3 保留）
- 新旗標 `--offset-update-in-open`（須同時帶 `--offset-hold-on-free`，否則啟動拒跑）。
- **更新開始**：沿用既有 interval 規則，與 r3 相同（錨點、逐步證據、執行端 mode 0、m ≤ 10、限幅 0.05 rad）；切入 free **不再**凍結。
- **更新結束**：K3-M 暫停生效那一輪（`_pause_active()` 首次為真）在本輪更新之前凍結；之後不再學習（沿用既有保持路徑）。
- **阻擋優先序不變**：停止 > 失效停止 > 閘門非 ARMED > 保持。阻擋時清錨點，解除後要先重設錨點，下一個區間才更新。
- **同步不變**：觀測器內部 d、節點 `_of_d`、延遲預推與 QP 的 bias＝α⊙d̂ 仍由同一函式每輪同步。
- 保持紀錄多記 `trigger`（`free`／`pause`）。
- 旗標未給時行為同 r3（靜態差異：非註解刪改只限凍結觸發條件三行）。

### 登錄 r10（`coopen_core_params_48_r10.json`；r9 保留）
- ＝r9＋`solver_change.offset_update_in_open: true`，node 改 ADM17r4，並附 rule_r10 與影子重播來源 sha256。
- 其餘鍵值逐一相同（靜態測試 S3）。

### 判定器 r12（`door_coopen_check_core12.py`；r11 保留）
- r11 全部判準照用。登錄設此旗標時，**只在鏈判定期間**把 W1c 的 `check_offset_hold`（全程凍結 H1–H6）換成 `check_offset_update_open`（U1–U8）；輸入紀錄不改。

| 被取代的原判準 | 取代為 |
|---|---|
| H2 凍結時刻＝首次套用 free | U3 凍結時刻＝暫停生效：switch_t ≤ 凍結 ≤ ＋0.06 s；未進入暫停 ⇒ 不得有凍結紀錄 |
| H3／H6 free 起每列阻擋、使用值＝凍結值 | U4 每個求解列實際使用值＝該列更新後估計（offset_d_hat＝d_after、arm_bias＝α⊙d_after）；U6 只從暫停起要求阻擋 |
| H5 n_updates＝n_upd_before | U7 報告計數＝逐列 updated 數；凍結前計數＝n_upd_before |

- 新增 U 判準：
  - U1 求解端參數兩旗標皆真；
  - U2 每個求解列都有完整 offset_interval；
  - U5 估計連續（相鄰列首尾相接、保持列不改值）；
  - U6 停止後與凍結後每列都被阻擋、無更新；updated 列不得帶阻擋理由；
  - U8 凍結發生時 first_published 與已發布列標記照 H4。
- **保留**：所有接觸、漂移（5 mm）、幾何、停止判準，與 r11 完全相同。
- **分報（不成判準）**：
  - OPEN 窗口（OPEN 起至下一相位或中止事件）的估計變化；
  - 兩指承載（每指最小／p05／平均／最大、低於 0.5 N 步數、合計與不平衡）；
  - 命令粗糙度（逐物理步 applied_arm、base_cmd_body 一階差分 RMS／p95／最大）。

### 跑法 bench6（`run_door_coopen_core_bench6.sh`；bench5 保留）
- ＝bench5，只換凍結 r6、登錄 r10、求解端 r4＋`--offset-update-in-open`、結束字樣，並加一行登錄旗標拒跑檢查（靜態測試 S1）。

## 3. 測試（全部通過）

| 測試 | 內容 |
|---|---|
| `test_dl3_adm17r4_offset_update_logic` M1–M7 | 真觀測器＋真 StepBuffer，非零 d0：free 中連續 10 區間更新且三處同步；暫停那輪更新前凍結（trigger pause）、其後 15 輪不變；停止／失效／閘門優先且清錨點，閘門恢復後先設錨再更新；暫停與停止同時 ⇒ 凍結照記、理由 stop_rx；旗標關 ⇒ free 即凍結（r3）；啟動檢查；靜態差異 |
| `test_dl3_adm17r4_update_ros` U0–U3 | ROS 假世界（free 3 s、pause 5 s）：free 到暫停之間 36 次更新；凍結 5.04 s（switch 5.01）；其後全阻擋；**判定器 r12 U 核對節點實際輸出無理由**；暫停照常 DONE；旗標關 ⇒ 3.04 s 凍結；暫停中停止 ⇒ 停止優先；缺相依旗標拒跑。假世界無物理，d̂ 為 0，非零值由 M 測試覆蓋 |
| `test_door_bench_start_ros4` C1–C5 | 台架啟動握手（真 ADM17r4＋新旗標、真轉接器 k3m2、替身執行端 r3）照常：同 seq 釋放、首步核對 ok；缺就緒拒跑 |
| `test_door_coopen_check_core12` Y1–Y5 | 合成正例（w1f_60_3 為底、轉成暫停才凍結）⇒ 無理由；回歸：旗標未設時與 r11 的 why 完全相同，r11 對新紀錄必拒；15 個反例各自命中；替換只在 context 內（含例外）；最外層 judge 鏈內確實使用新判準 |
| `test_door_coopen_core_bench6_static` S1–S3 | 跑法差異、求解端參數由 r4 argparse 解析成功、登錄 r10 對 r9 差異 |
| 回歸：bench_3 實跑紀錄 | r12（登錄 r9，未啟用）與 r11 的 why／chain_pass／iso／bench／items 逐項相同 |

## 4. 單趟計畫（待 Codex 核可；不自動重跑）
- RUN_ID `door_coopen_core_bench_4`，跑法 bench6，凍結 r6。
- 只看：
  1. 是否越過 bench_3 的中止點（43.7°、5 mm 漂移）；
  2. OPEN 期間 d̂ 是否實際更新且未觸限；
  3. 兩指承載與命令粗糙度相對 bench_3 的變化（是否出現「學入接觸負載、內力加重」的徵象）。
- bench_3 基準（同一分報函式、OPEN 窗口 4.60–105.32 s）：
  - 每指平均 4.85／5.06 N，p05 4.47／4.65 N，最小 0.0（單步）／1.57 N，合計平均 9.91 N；
  - applied_arm 差分 RMS 0.40–1.54 mrad/s／步；
  - OPEN 期間 2 次更新（4.60–4.72 s 凍結前）。
- 不放寬任何門檻；失敗照實封存。
