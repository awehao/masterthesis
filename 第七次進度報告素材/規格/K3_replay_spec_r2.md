# 門場景實錄重播 r2（Codex 20261023_235904 第 2 節修正；已實作與離線測試；**重播未開跑，送短複核**）

r1（ev2r 的 `--replay` 分支，見 `K3_root_evidence_spec_draft2.md` §5）沒有用過。r2 另立版本化檔案 `isaac_door_k2_sim_ev2rp.py`：
- ev2r 的複本，**只改 `--replay` 分支、新增 `--replay-mode`、檔頭說明**；
- diff 範圍限 ev2r 第 829–888 行的重播區塊＋兩處插入；非重播路徑與 ev2r 逐字相同；
- pyflakes 警告與 ev2r 相同。

ev2r 本身在 door_k3ab_lw_ev2r_1 跑時的版本已由 git（commit cda9b8355／b0f8e179f）與凍結 r2 保存，不再改動。
`run_door_replay.sh`、`door_replay_video.py` 的 r1 版本隨該凍結保存於 git；r2 是新版本，有新的凍結。

## 1. 姿態與畫面的對應（必修一）
**每幀的設定**：
- 底盤：`robot.set_world_pose`；
- 手臂與手指：`set_joint_positions`，加上 **`set_joint_velocities(0)`**，以及 **`apply_action(joint_positions=本幀姿態)`**（驅動目標與本幀相同，避免初始化目標殘留）；
- 門扇：繞鉸鏈轉到紀錄門角，並 `set_velocities(0)`；
- 根：線速度與角速度歸零。

**預設 `--replay-mode nostep`：不推進物理**。
- 做法：`omni.physx` 的 `update_transformations(updateToFastCache=True, updateToUsd=True)` 把 PhysX 狀態寫到渲染端，再呼叫 `world.render()`。
- 依本機安裝的 SimulationContext.render 原始碼：它在 `app.update()` 期間把 `/app/player/playSimulations` 設為 False，並強制更新 fabric，所以不會推進物理。
- 能否確實更新畫面，以 §2 的渲染端核對量測。

**後備 `--replay-mode step`**：推一個物理步後擷取。驅動目標已對齊、速度已歸零，偏差同樣量測。
- 是否改用 step，依 nostep 的核對結果決定，不自動切換。

## 2. 擷取前回讀實際姿態，對來源逐項計偏差（逐幀寫入 frames.csv）
| 欄 | 來源 | 容差（事前登錄） |
|---|---|---|
| dev_base_mm、dev_yaw_deg | PhysX tensor 根位姿 vs 紀錄 base_xyth | 1 mm、0.1° |
| dev_joint_mrad | tensor 關節位置 vs 紀錄 q_arm_meas、q_finger（最大值） | 1 mrad |
| dev_door_deg | tensor 門扇姿態推得門角 vs 紀錄 door_q_rad | 0.05° |
| usd_base_mm、usd_yaw_deg | **渲染端** USD（XformCache）根 prim 世界變換 vs 紀錄 | 1 mm、0.1° |
| usd_door_deg | 渲染端門扇 prim vs 紀錄門角 | 0.05° |
| usd_grip_mm | 渲染端夾爪連桿 prim vs 同連桿 PhysX tensor 位姿（手臂連桿是否跟上） | 1 mm |

- 任一超差或讀不到（NaN） ⇒ `pose_ok=0`。數值保留，不隱藏、不剔除。
- **界線**：渲染器使用 fabric；USD 端對應是可量到的最接近渲染的層。手臂只核夾爪連桿一處，其餘連桿以關節回讀代表。

## 3. 跑法與完整性（必修二）
**`run_door_replay.sh` r2**：
- 起跑前核對凍結 `freeze_door_replay_dev_r1.sha256`；
- run.log 記**完整**來源 sha256，ev2rp 在 sim.log 另記一次；
- `frames/` 已有影格 ⇒ 拒跑（runner 與 ev2rp 都檢查）；
- 結束後執行 `door_replay_check.py`。

**`door_replay_check.py`**：
- **預期幀**＝來源步每 round((1/hz)/dt) 取一，並落在 [t0, t1] 內。
- **一一核對**：frames.csv 的 (frame, step, sim_t) 與預期序列逐一相同（同序、無缺、無多）；img_ok＝1 的幀須有 PNG；pose_ok＝1。
- **來源雜湊**：sim.log 所記與實際相同。
- **輸出**：寫 `replay_check.json`。complete＝False 時列原因：缺圖、超差、錯位、幀數、雜湊未記或不符。
- 缺圖逐幀記錄（img_ok＝0），**不靜默略過**。

## 4. 影片時間（必修三）
`door_replay_video.py` r2：
- **每幀時長**＝（下一幀模擬時間 − 本幀模擬時間）÷ 倍速，以 ffmpeg concat 依**實際影格**編碼。
- **缺圖**：沿用前一幀，時長照實保留，頂欄標「缺圖 N」。不把缺幀壓縮成固定倍速。
- **關鍵時刻的影片時間**＝（模擬時間 − 首幀模擬時間）÷ 倍速，與上述編碼一致。
- **不完整時**：replay_check 為 False ⇒ 頂欄標「重播不完整（原因）」，keymoments.json 記 `replay_complete=False`、缺圖數、原因。

## 5. 測試（離線）
`test_door_replay.py` 3 項：
- 一一對應時 complete；
- 缺圖（img_ok＝0、檔案缺）、姿態超差、步號錯位、少幀、多幀、雜湊未記、雜湊不符 ⇒ 各自原因；
- 合成影格缺一幀時，影片總長仍＝模擬時間跨度 ÷ 倍速（缺幀不縮短影片），keymoments 記缺圖 1、replay_complete=False。

## 6. 申請（短複核）
- lw_1 與 lw_ev2r_1 各 side／close 一支：
  - `REPLAY_MODE=nostep`，由 run_guarded_v2 包 92 °C 熱中止；只渲染、不起 ROS；
  - 新 RUN_ID（例 `door_k3ab_lw_1_replay_side`），不寫入來源趟目錄。
- **若 nostep 的 USD 核對顯示畫面沒有跟上**（pose_ok 大量為 0）：照實保留該輸出，不交付為正式影片；改以 `REPLAY_MODE=step` 重做一次並另存。
- 若相機取景不當：只調相機參數，另存一版重渲染一次，保留調整紀錄。
- 不改來源實錄，不挑掉失敗片段。

## 7. 凍結
`freeze_door_replay_dev_r1.sha256`＝lw_ev2r r2 的清單（其中 run_door_replay.sh、door_replay_video.py 改為 r2 雜湊），加上 ev2rp、door_replay_check.py、test_door_replay.py 與本文件。
