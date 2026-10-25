# W2a：HELD 持續保持＋模式權威（同一命令流內的模式轉換機制）——差異表＋實作＋反例測試＋判定＋凍結 w2a r1（**不實跑**）

依 Codex 20261025_201543：
- 下一步選 (a)：W2 HELD／模式權威的版本化實作與假世界測試，**不核可實跑**。
- 限定範圍：
  - HELD 持續發布與監看；
  - 「準備首筆命令 → 執行端驗證 → 物理步提交」；
  - mode_seq 與 source_seq 一起穿過 adapter；
  - 假世界涵蓋無空窗、舊模式延遲命令、ack 遺失、重送／衝突、切換前後失效、中止優先；
  - 測試模式不得驅動真實張爪／退離。
- K3-B 解除清空判準留到受控釋放套件，實作張爪／退離前必須定版。本輪不增加角度、不改停止判準、不啟動 W2 模擬。

## 1. 差異表
| 檔案 | 來源 | 差異 |
|---|---|---|
| `door_w2_mode.py` | 新 | HeldSupervisorP4b（P4b 子類）、ModeAuthority（執行端權威）、SolverModeSource（求解端）。測試模式 RELEASE_PREP_TEST |
| `cmd_envelope_v3.py` | 新 | v2 的 18 欄（套件內 cmd_envelope 不改）＋mode_seq＝19 欄，schema 3.0；前 18 欄沿用 v2 的嚴格檢查；話題 /wgmpc/cmd_env3、/wb_vel_cmd_env3 與 v2 分開 |
| `dl3_wgmpc_wg2_adm11_node.py` | ADM9 只插入（2454 → 2505；`results/door/mk_adm11_patch.py` 核對） | 見下 |
| `arm_vel_adapter_w2.py` | arm_vel_adapter_k3m 版本化（267 → 307；`results/door/mk_adapter_w2_patch.py` 核對） | 見下 |
| `door_w2_check_mode.py` | 新 | 逐步證據 C1–C7（純函式） |

**ADM11 的三個旗標**
- `--w2-held`：
  - 監督器改 HeldSupervisorP4b：觀察通過記 OBS_PASSED，進 HELD，持續發布與監看；
  - 超過 `--w2-held-max-s` ⇒ HELD_TIMEOUT_END 後 DONE。
- `--w2-mode-run-id`：
  - 訂 /door/w2_mode_req（只在 HELD 接受、準備新模式首筆）與 /door/w2_mode_state（權威狀態為準）；
  - 每筆發布另以 v3（同 source_seq＋mode_seq）發 /wgmpc/cmd_env3。
- 預設關 ⇒ 與 ADM9 同行為；v2 封裝與話題不變。

**adapter_w2**
- `--cmd-env-v3` ⇒ 只訂 v3、只發 v3，轉送時 **mode_seq 原樣保留**；輸出前再解碼核對 mode_seq 相同，不同即丟棄並計數。
- 刪改一處：初始化的入口擇一，v3 時不訂 v2、不訂舊九維，維持唯一控制入口。其餘只插入。
- 預設關 ⇒ 與 k3m 版同行為。

**本包不含**
- 執行端 ev6 接線：Isaac 執行端以同一 ModeAuthority 類別接上，下一包。
- 任務端 HELD_WAIT 與請求：下一包。
- 張爪、退離、恢復上限：受控釋放套件，須先定 K3-B。

## 2. 語意
**HELD 監督器**
- HELD 期間每輪 publish／zero。
- 監看照舊：速度不低 ⇒ moved_in_held；量測失效；求解失敗；套用非保持命令或來源早於零速段 ⇒ old_command_in_held。
- 結束只有三條路：交接、中止／失效、有界超時。
  - 有界超時不算任務完成，也不代表撤除保持後安全停止。

**模式權威（執行端）**
- 提交時序：請求 ⇒ pending → 求解端準備新 mode_seq 的首筆 → 執行端收到並驗證 → **在該物理步原子提交**。
  - 驗證項：有限、保持形、相對實際 applied u_prev 的加速度框。
  - 同一處同時切換：mode_seq、監看 profile、套用來源。
  - **不是先切模式再等求解端產生首筆。**
- 提交前：舊 mode_seq 命令照常套用（無空窗）。
- 提交後：舊 mode_seq 命令一律 stale，不套用、計數；不以「最新模式」猜歸屬。
- 失效：
  - 切換前候選逾時或不合格 ⇒ 拒絕、保持；
  - 切換後新模式斷流 ⇒ 中止，**不回切**。
- 中止優先：pending 拒絕、之後一律 stale、新請求 after_abort。
- 權威狀態每 2 物理步發布，**ack 遺失不代表未切換**，求解端以狀態為準。
- 監看依該步實際權威模式判斷：profile 與 mode_seq 同步切換。

**本包只有一個測試模式**：RELEASE_PREP_TEST 的命令與 HELD 相同，不張爪、不退離。兩個 profile 內容相同，只分開標示，用來核對切換步。

## 3. 測試（全過）
**`test_door_w2_mode.py` 17 項**
- HELD：
  - H1 觀察通過後持續發布；
  - H2 HELD 中底盤漂移 ⇒ moved_in_held；
  - H3 非保持命令 ⇒ old_command_in_held；
  - H4 有界超時 ⇒ HELD_TIMEOUT_END → DONE、停發；
  - H5 求解失敗。
- 權威：
  - A1 正常提交（提交前舊序號照套）；
  - A2 提交後舊序號 stale；
  - A3 重送 repeat、不同請求 conflict（pending 中、提交後）；
  - A4 候選手臂非零／底盤超界／非有限／加速度框 ⇒ 拒絕保持；
  - A5 候選逾時；
  - A6 提交後斷流 ⇒ 中止不回切；
  - A7 中止優先；
  - A8 超前序號、缺 mode_seq、錯 run、錯轉換、格式錯。
- 求解端來源：S1 只在 HELD 準備；S2 ack 遺失以狀態得知提交；S3 被拒撤回；S4 中止撤回。

**`test_door_w2_check_mode.py` 5 項**
- 證據核對 C1–C7 各自反例，以及兩次提交。
- v3 封裝：往返；mode_seq 負數、非整數或缺欄、schema 不符 ⇒ 拒絕；v2 身分規則照樣檢查。

**`test_w2_held_mode_ros.py` M0–M7（假世界，全過）**
- 架構：真 ADM11、真 adapter_w2、替身端點；執行端角色＝同一 ModeAuthority。
- M0：參數相依 ⇒ exit 2。
- M1 正常：HELD 後 1 s 請求 ⇒ 提交步 1768、首筆套用 source_seq 353；C1–C7 全過；結束時仍在 HELD（持續發布）。
- M2：提交後權威狀態中斷 1 s ⇒ 求解端仍得知提交、證據全過。
- M3：延遲舊命令判 stale、未套用。
- M4：重送 repeat、衝突 conflict_after_commit、恰一次提交。
- M5：HELD 前請求 ⇒ 求解端 ignored_not_held、候選逾時拒絕、保持 mode 0；HELD 有界超時結束；HELD 期間 575 步無逾時。
- M6：提交後斷流 ⇒ post_commit_source_lost 中止、不回切。
- M7：請求後立即中止 ⇒ 不提交、無 mode 1 命令被套用。求解端在發出候選前即由權威狀態撤回（reverted）。

**未在本包驗證（揭露）**
- ADM11 預設關閉時的完整假世界回歸：ADM11 為 ADM9 只插入，旗標預設關，未另跑。
- Isaac 執行端接線。
- 真實張爪與退離。
- 假世界執行端是測試程序，不是 Isaac 執行端：時序與鏈路延遲不同，逐步證據只驗證機制，不等於安全保證。

## 4. 凍結
`results/door/freeze_w2a_dev_r1.sha256`：新檔、相依模組（暫停監督器、封裝、ADM9、adapter_k3m、替身端點）與生成腳本。所有文件改完後最後建立。

## 5. r2（Codex 20261025_211640：四項必修＋ADM11 關閉旗標回歸；仍不實跑）
**① 準備候選期間，舊模式持續供應**（求解端雙流）
- SolverModeSource.tag_normal()＝已提交序號：一般命令流（v2＋v3）在準備候選期間照常以舊 mode_seq 發布。
- 候選走**旁路**：
  - 同週期在一般發布之後另發一筆 v3，帶獨立 source_seq 與新 mode_seq，並記入 _cmd_hist；
  - 權威狀態顯示提交後，候選流才成為一般流。
- ADM11 r2 仍為 ADM9 只插入（2454 → 2517）。
- 揭露：
  - 測試模式下候選內容與一般命令相同；真實模式之後須另算候選內容與對應的 _cmd_hist。
  - 暫停監督器 note_published 與 rec['pause_source_seq'] 記到的是同週期最後一筆（候選）的序號；序號仍單調。

**② 接收與物理步提交分開；入口驗證完整**（ModeAuthority r2）
- 三個介面分工：
  - on_env：只做入口驗證與緩衝；
  - select(step, t, applied u_prev)：在物理步選擇套用，並在此提交；
  - confirm_applied：核對實際套用來源，提交步不符 ⇒ 中止 commit_not_applied（不回切）。
- 同一步連到多筆，以最後緩衝者為準；commit 記錄的就是實際選定那筆。
- 入口規則：
  - 身分：本趟 run_id、stage（adapter）、kind normal、derived；
  - source_seq 為非負整數，每個 mode 單調；
  - 同序同值重送冪等、不刷新；同序異值、跨 mode 同序 ⇒ 違規；亂序、過期 ⇒ stale；
  - 未來時間戳 ⇒ 違規；
  - 候選須在命令新鮮期（0.2 s）內被選定，候選期限 1 s；
  - **缺或非法的 applied u_prev ⇒ 拒絕候選**。
- adapter_w2 r2：v3 的 stamp_sim_t 沿用求解端的模擬時間戳。adapter 節點時鐘不是模擬時間，所以執行端改以模擬時間核對新鮮度與未來時間。

**③ 判定器核對證據覆蓋**（door_w2_check_mode r2）
- C0：
  - 覆蓋登錄窗口 [OBS_PASSED, 提交步＋window]；
  - 步號唯一且逐步連續；時間有限且嚴格遞增。
- C1'：唯一 commit 事件與 auth.commit 一致；applied_confirmed 為 True。
- C7 移除 OBS_PASSED＋0.3 s 豁免。

**④ HELD 不削弱套用回報驗證**
- 保留 3＋6 維、有限、整數步號與來源序號（排除 bool）、時間有限的檢查。
- 同步同值重送不刷新；同步異值 ⇒ applied_conflict；倒退 ⇒ applied_order。
- 無效資料一律 applied_invalid，不拋例外。

**測試（全過）**
- **`test_door_w2_mode.py` 20 項**，新增：
  - 候選延遲超過命令逾時時舊模式持續套用；
  - 同一步多筆候選；
  - 提交步實際套用不符 ⇒ 中止；
  - 過期候選與候選逾時；
  - 缺或非法 u_prev ⇒ 拒絕；
  - 候選內容不合格；
  - 入口 11 種違規，以及冪等、同序異值、跨 mode、亂序、過期；
  - 提交後舊序號 stale 與斷流中止；
  - 請求規則；
  - HELD 回報 6 種無效輸入與重送、衝突、倒退；
  - 雙流 tag。
- **`test_door_w2_check_mode.py` 6 項**，新增 Codex 的三個誤收反例（刪切換後 40 步、截斷在提交步、交換相鄰列），以及時間 NaN、缺觀察通過起區段、提交未確認、事件不符、觀察通過後 0.1 s 逾時。
- **`test_w2_held_mode_ros.py` M0–M8＋R0**（假世界）：
  - 端點改用 r2 介面：每物理步 select，並以 confirm_applied 回報。
  - **M3 r2**：扣住請求後的一筆 mode 0 命令，提交後 0.3 s 才送達 ⇒ stale、計數、未套用；提交後 1332 步皆為 mode 1。
  - **M8 新增**：候選延遲 0.5 s（丟 10 筆）。期間 51 步舊模式持續套用、無逾時步；請求後 0.51 s 才提交（18.68 → 19.19 s）；C0–C7 全過。
  - **R0 新增**：ADM11 不帶 --w2-*（比照 ADM9，v2 直接作套用來源）⇒ 暫停照常 DONE；stats.w2 為 None；/wgmpc/cmd_env3 無任何訊息。
  - 測試時序：連續多情境時 DDS 探索較慢，改為替身 3.0 s 就緒、free 4.0 s、pause 6.0 s。
    - 第一次全跑時，M7、M8、R0 因 ADM9 既有保護 park_mode_refused（就緒時尚未收到 /park/gate）而提前結束。這屬於測試時序，與 W2 無關；M7 單獨跑可通過。

**仍未驗證**：Isaac 執行端（ev6）接線、任務端 HELD_WAIT 與請求、真實張爪／退離（K3-B 須先定版）。

**凍結**：`results/door/freeze_w2a_dev_r2.sha256`，所有文件改完後最後建立；r1 保留。
