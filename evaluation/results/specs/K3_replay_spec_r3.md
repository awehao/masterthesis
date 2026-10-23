# 門場景實錄重播 r3（第一次四支重播失敗後的修正；送短複核申請重跑）

r2（`K3_replay_spec_r2.md`）的其餘內容不變。本文件只列這次的失敗、修正，以及兩項離線修正。

## 1. 第一次四支重播（依 Codex 20261024_002150 核可，各一次，nostep）：**全部失敗，0 幀**
| RUN_ID | 結果 |
|---|---|
| door_k3ab_lw_1_replay_side／close | 約 10 s 結束；frames.csv 0 列、0 張影格；replay_check：`frame_count_0_vs_expected_317` |
| door_k3ab_lw_ev2r_1_replay_side／close | 同上；`frame_count_0_vs_expected_320` |

- **熱保護**：CPU 峰值 71–72 °C，熱中止 0 次。
- **來源**：完整雜湊已記錄，來源實錄未改動。
- **失敗輸出**：全部保留在各自的 runs 目錄。原核對檔改名為 `replay_check_r1tool.json`，新核對器的結果寫在 `replay_check.json`。

**原因**（sim.log 原文）：
- 重播分支在 `world.reset()` **之後**才建立夾爪 `RigidPrim` 視圖（預設 reset_xform_properties=True）。
- 隨後 log 出現 `prim '.../uflite_finger2/grip_col_blade' was deleted while being used by a shape in a tensor view` 與 `Simulation view object is invalidated`。
- 接著第一次 `robot.get_joint_positions()` 就拋出 `Failed to get DOF positions from backend`，在第一幀之前退出。
- 與主程式既有註記「視圖必須在 world.reset() 之前建」一致。
- 這是重播工具本身的錯誤，不是姿態同步核對失敗，所以不適用 step 後備；改為修正工具後申請重跑。

## 2. 修正（ev2rp r2）
- 夾爪視圖改為**只在 `--replay` 時、於 `world.reset()` 之前建立**，並設 `reset_xform_properties=False`；重播分支內只呼叫 `initialize()`。
- 非重播路徑不變：該段以 `if a.replay` 包住。pyflakes 警告與 ev2r 相同。
- 尚未在 Isaac 中驗證這項修正；重跑時第一幀的姿態核對與影格就是驗證。

## 3. 兩項離線修正（Codex 20261024_002150 第 3 節；測試通過）
- **核對器獨立檢查數值**（`door_replay_check.py`）：
  - 不採信 pose_ok／img_ok 旗標；逐幀獨立核對 sim_t 有限，以及八類偏差欄位存在、有限、≤ 登錄容差（與 ev2rp 相同）。
  - 旗標與獨立結果不一致時，另列 `flag_inconsistent`。
  - 你的反例（sim_t＝NaN、dev_base_mm＝999 但 pose_ok＝1）現在判為不完整。另加欄位缺失、空字串、NaN、img_ok 與檔案不一致。
- **缺圖當下標明**（`door_replay_video.py`）：
  - 缺圖的那一刻，以前一張原始影格為底，疊上當下的時間、相位、門角等標註，再加「缺圖（影格 N）：沿用前幀」橫條。
  - 首幀就缺圖時，用明示的灰底佔位畫面。
  - 每列都有自己的畫面，時長照實保留。
- **測試**：`test_door_replay.py` 3 項。首幀與中段各缺一張時，影片總長不縮短、畫面數 ≥ 列數、keymoments 記缺圖 2。

## 4. 凍結
- `freeze_door_replay_dev_r2.sha256`：replay r1 清單，ev2rp、核對器、影片、測試換新雜湊，並加入本文件。
- 舊取證 r2 清單（`freeze_door_k3ab_lw_ev2r_dev_r2`）目前因 `run_door_replay.sh`、`door_replay_video.py` 改版而不符。歷史版本依原 commit（cda9b8355）保留，不覆寫舊清單。replay r1 清單同理（commit 2338dbccf）。

## 5. 申請
重跑同四支：lw_1、lw_ev2r_1 各 side／close。
- 新 RUN_ID，加 `_r2` 後綴；nostep、92 °C 守護、新凍結核對、完整來源雜湊、不改來源實錄。
- 姿態同步核對失敗時，可另存一次 step 後備。
- 取景不當時，只改相機另存一次。
- 第一次的失敗輸出全部保留。

## 6. 增補（10/24，第一支 r2 重播後）
**`door_k3ab_lw_1_replay_side_r2`**（Codex 20261024_002630 核可的第一支；nostep、92 °C 守護）：
- 工具修正有效：317 幀中產出 315 張；tensor view 未失效、無例外。
- 八類姿態偏差全部為 0（關節最大 1e-4 mrad；USD 端根、門扇、夾爪皆 0）。
- 抽看收攏、對準、夾持、OPEN、中止後的畫面：非空白、非停格，姿態隨時間變化。
- **缺圖 2 張：第 1、2 幀**（模擬 0.00 與 0.10 s，相機剛建立）⇒ `replay_check.complete=False`（`missing_images_2`）。
- 依放行條件（complete=True），**其餘三支未放行**。
- 已用這支產出**預覽影片**：明標「重播不完整（缺圖 2）」，首幀用佔位畫面；不作正式版本。片長 15.88 s，等於 31.6 s ÷ 2 倍速。

**修正（ev2rp r3，只影響重播）**：
- 某幀 `cam.get_rgba()` 取不到影像時，對同一姿態重新 `world.render()` 最多 3 次（不推進物理），次數記入 frames.csv 的 `render_retry` 欄。
- 3 次後仍無圖，照舊記 img_ok=0，不靜默。

**凍結**：`freeze_door_replay_dev_r3.sha256`。

**申請**：用 r3 重跑 lw_1 side，RUN_ID `door_k3ab_lw_1_replay_side_r3`；通過 complete=True 後，再依序跑其餘三支。side_r2 的輸出與預覽影片保留。

## 7. 增補 r4（Codex 20261024_003051）
- 姿態回讀移到**最後一次 render 之後、存影格之前**：擷取順序抽成純函式 `door_replay_capture.capture()`（取影像 →（無圖）同一姿態重 render ≤3 次、不推進物理 → 回讀姿態），ev2rp 呼叫它。
- `test_door_replay_capture.py` 3 項：姿態回讀一定在最後一次 render 之後；重試上限 3；有圖不重試；仍無圖回傳 None 並記 retry=3。
- 凍結 `freeze_door_replay_dev_r4.sha256`。
- 放行順序：side_r3 → complete=True 且抽看正常 → 立即產正式影片並更新 README → 依序跑其餘三支。再缺圖或出現工具例外就停批，不自行增加重試次數。
