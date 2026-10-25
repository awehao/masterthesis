# W2c r2：判定器三個誤收反例修正（Codex 20261025_223632；條件式核可單趟 door_wide_w2c_30_1）

W2c r1 規格 `W2c_run_pack_spec.md` 已凍結於 w2c r1，**不改**；本文件為其增補。

## Codex 裁定（20261025_223632）
- 控制接線兩項必修可接受；只剩整體判定器必修。完成修正、反例、回歸與新凍結核對後，**條件式核可單趟**。
- M10 的 0.45 s：接受分段語意。
  - 前段是鏈內有效套用，後段才是無有效套用的期限。
  - 不能稱「新命令被擋後 0.25 s 即中止」，也不是安全停止保證。
  - 本輪不另加來源序號更新期限。
- 只替換兩項舊 DONE 理由：可接受。
- 父類替身範圍：足以驗證路由，不證明整段串接。**首趟列為整合驗證。**

## 1. 差異
| 檔案 | 來源 | 內容 |
|---|---|---|
| `door_k3m_check_w2c2.py` | 新（重用 w2c.check_w2；w2c 不改） | 固定登錄值、趟別一致、事件時間交叉核對 |
| `isaac_door_k2_sim_ev6r3.py` | ev5 只插入（1687 → 1798；`results/door/mk_ev6r3_patch.py`）；ev6r2 不改 | 只加紀錄 room_run.json `w2.config`：w2 run_id、run_id、run_id_num、權威 run_id 與參數。控制不變（U2 核對） |
| `run_door_wide_w2c_r2.sh` | `run_door_wide_w2c.sh` 的複本 | 只換凍結名稱、執行端 ev6r3、結尾判定器名稱（U3 核對） |
| `test_door_k3m_check_w2c2.py`／`test_w2c2_wiring_static.py` | 新 | §3 |

## 2. 判定器 w2c2
**一、固定登錄值**
- G 項以常數 REG（保持 3 s、提交期限 2 s）呼叫 w2c.check_w2，不讀 task 自報 params。
- 各端紀錄須等於 REG，缺即不通過：
  - task args 與 task.w2.params：req_delay 1.0、commit_timeout 2.0、test_hold 3.0、state_stale 1.5；
  - 執行端 w2.config：cand_timeout 1.0、cmd_max_age 0.2、post_commit 0.25、applied 0.25、dt 0.01；
  - 求解端 args：w2_held True、w2_held_max_s 60。

**二、趟別與請求一致**（RUN＝task args run_id）
- task 的 w2_mode_run_id 與請求內容；
- 執行端 config 三個 run_id；
- 執行端登記的請求＝task 請求，結果 pending；
- 恰轉送一次；
- 求解端 args 的 run_id 與 w2_mode_run_id；
- 求解端恰一筆 prepare 與一筆 committed_seen。

**三、事件時間交叉核對**
- 容差（明列）：
  - 同一模擬時鐘；CLOCK_TOL 0.05 s（50 Hz 同步狀態兩筆間隔加餘量）；
  - DELIVER_MAX 0.5 s（任務請求到執行端登記的送達上限）。
- 先後與時限：
  - OBS_PASSED ≤ held_wait；
  - requested − held_wait ≥ 1 s；
  - requested ≤ 執行端登記 ≤ requested＋0.5；
  - 登記 ≤ 執行端提交 ≤ 登記＋1；
  - 執行端提交 ≤ task committed ≤ 執行端提交＋1.5；
  - done − committed ≥ 3；
  - done ≤ room 最後時刻。
- trace 對齊：w2 狀態序只允許 none→held_wait→requested→committed→done；各狀態第一列時刻＝task 自報值。
- **保持 3 s 不靠 task 自報**：執行端逐步列自執行端提交步起，連續有效套用到執行端提交＋3 s − tol 以後。

## 3. 測試（全過）
**`test_door_k3m_check_w2c2.py`**（4 組，以 60_3 為底的合成資料）
- Q0：合成正例通過。
- Q1–Q3：Codex 三個誤收反例全部拒收。
  - 保持改 0.1 s ⇒ 登錄值不符；
  - 他趟請求 ⇒ task_request；
  - committed 提前 5 s ⇒ task_commit_window。
- Q4：七項缺證據或不一致都拒收。
  - 缺 config、執行端登記請求不同、求解端無 prepare；
  - 執行端保持證據不足 3 s、trace 跳序；
  - 求解端 held_max 改 30、執行端 cand_timeout 改 2。
- Q5：r1 判定器對 Q1–Q3 均誤收，重現 Codex 反例，作對照。
- 合成 trace 的延伸列沿用 PAUSE 期間「持續發布最後一筆目標」的父類行為，pub_seq 遞增；JSONL 另寫暫存檔。這是測試用資料，不是實跑結果。

**`test_w2c2_wiring_static.py`**（U1–U4）
- ev6r3 重生成一致；與 r2 只差紀錄；跑法只差三個名稱；判定器常數固定。

**回歸**
- 既有 W2 純測試、真鏈測試、靜態核對全過。
- 執行端控制碼與 r1 包相同（U2），所以假世界與路由測試不重跑。

## 4. 凍結與單趟
- `results/door/freeze_door_wide_w2c_dev_r2.sha256`：w2c r1 全部＋本增補新檔與本文件。最後建立；起跑前跑法自動核對。
- 單趟條件（Codex）：
  - 30°＋HELD＋測試模式保持 3 s，不張爪、不退離；
  - 乾淨環境、ROS_DOMAIN_ID 97、92 °C 守護、起跑前後凍結核對；
  - 初始化或任務失敗均保留，不補跑、不覆寫；
  - 分報既有任務、停止、保護、紀錄與 W2 各項；
  - 本趟不新增影片。
- 首趟屬**整合驗證**：開門→暫停→HELD→W2 串接第一次在 Isaac 出現。
