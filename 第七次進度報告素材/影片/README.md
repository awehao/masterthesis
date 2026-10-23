# 第七次進度報告：影片

規則（Howard／Codex 20261023_234952）：
- 重要動作實驗都要交影片，成功與失敗都保留。
- 每支影片標明「即時錄影」或「該趟實錄重播」。
- 若即時錄影會影響控制，就用該趟實錄重播，不拿別趟代替。

產生方式：
- `evaluation/run_door_replay.sh`：只渲染，不跑控制、不起 ROS。
- `evaluation/door_replay_video.py`：疊上標註並編碼。
- 關鍵時刻同時寫在影片橫幅與 `*.keymoments.json`。

| RUN_ID | 目的 | 結果 | 影片 | 種類 | 狀態 |
|---|---|---|---|---|---|
| door_k3ab_lw_1 | K2 閂鎖後長窗停止診斷（固定底盤 K3-A-B） | 任務未通過（finger_contact_lost）；閂鎖後 10.11 s 未確認停止：門 0.68→3.16°、底盤約 29 mm；殼—橫桿取樣最小 4.39 mm、非設計接觸 0 | side／close（待產出） | 該趟實錄重播 | 第一次重播（10/24）因重播工具錯誤 0 幀，失敗輸出已保留；工具已修正，重跑待核可 |
| door_k3ab_lw_ev2r_1 | 根／門扇三點讀取取證（同 lw 配置，只加紀錄） | 已跑（10/24）：任務未通過（finger_contact_lost）；取證完整；閂鎖後 10.11 s 未確認停止：門 3.12°、底盤 28.4 mm；殼—橫桿取樣最小 4.43 mm | side／close（待產出） | 該趟實錄重播 | 同上 |

第一次重播失敗紀錄：`evaluation/runs/door_k3ab_lw_1_replay_side`、`_close`，`evaluation/runs/door_k3ab_lw_ev2r_1_replay_side`、`_close`。原因與修正見 `規格/K3_replay_spec_r3.md`。

關鍵時刻（產出後填影片時間）：對準開始、夾持開始、夾持保持、OPEN 開始、策略中止（失接觸）、執行端閂鎖、停止後漂移。
