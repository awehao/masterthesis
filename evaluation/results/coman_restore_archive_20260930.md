# 還原後工作版本的封存（2026-09-30）

事故與還原經過見 `coman_handover_geometry_check_20260929.md` §30.4。
本檔只記錄**封存事實**，不重跑模擬、不做新判定。

## 版本

* 提交 `18fa7688`（`Auto-update: 2026-09-30 01:09`）—— 還原後、測試通過的狀態
* 標籤 `coman-restore-20260930`（輕量標籤，指向上述提交）
* 工作樹：除未追蹤的 `src/xarm_ros2` 外**無未提交差異**

## 三個曾受影響的檔案：型態與 sha

| 檔案 | 型態 | 大小 | sha256[:16] |
|---|---|---|---|
| `arm_link_distance.py` | 正常檔案 | 35393 | `ea0c5be4db7b9333` |
| `wholebody_safety_filter.py` | 正常檔案 | 35494 | `276fcbda767de52a` |
| `wholebody_safety_node.py` | 正常檔案 | 31476 | `aec2c310e0da8755` |

## S1 記錄的 15 個 checker sha：逐項核對

| 檔案 | S1 記錄 | 實際 | 相符 |
|---|---|---|---|
| `coman_handover_state.py` | `b1d8f290b4499fbc` | `b1d8f290b4499fbc` | ✓ |
| `coman_couple_link.py` | `3f89f33edc5a7cd7` | `3f89f33edc5a7cd7` | ✓ |
| `coman_pose_reader.py` | `9581efb675d8d361` | `9581efb675d8d361` | ✓ |
| `coman_pull_policy.py` | `6e1492d2f95d5be7` | `6e1492d2f95d5be7` | ✓ |
| `coman_pull_target.py` | `9199c0a1011d1d91` | `9199c0a1011d1d91` | ✓ |
| `coman_pull_solver_node.py` | `a6f05ff9d8dd1104` | `a6f05ff9d8dd1104` | ✓ |
| `isaac_coman_drawer_sim.py` | `90c6121e4a3203e8` | `90c6121e4a3203e8` | ✓ |
| `wholebody_safety_filter.py` | `276fcbda767de52a` | `276fcbda767de52a` | ✓ |
| `arm_link_distance.py` | `ea0c5be4db7b9333` | `ea0c5be4db7b9333` | ✓ |
| `wholebody_safety_node.py` | `aec2c310e0da8755` | `aec2c310e0da8755` | ✓ |
| `test_pair_exceptions.py` | `da31af8647a8f293` | `da31af8647a8f293` | ✓ |
| `test_pipeline_pairs.py` | `819689265ea84de3` | `819689265ea84de3` | ✓ |
| `coman_zero_cmd_feasible.py` | `59b2c2d21c3cc0eb` | `59b2c2d21c3cc0eb` | ✓ |
| `coman_shell_bar_bound.py` | `d1ba12bd7a0a5c38` | `d1ba12bd7a0a5c38` | ✓ |
| `test_relative_velocity.py` | `26369e356ce75c22` | `26369e356ce75c22` | ✓ |

**15/15 相符**

## 規格檔 sha

| 檔案 | sha256[:16] | 狀態 |
|---|---|---|
| `wb_coman_drawer20_criteria_v1.yaml` | `c9f303a8630d4d0e` | frozen |
| `wb_coman_drawer20_supplement_s1.yaml` | `26e2a533e0f5d195` | draft_pending_approval |
| `coman_near_approach_config_p1_proposal.yaml` | `c7f1c9d27c6a1d30` | proposal_pending_approval |

## 測試結果（本封存時重跑）

| 套件 | 結果 |
|---|---|
| `test_relative_velocity.py` | 相對接近速度離線測試：全部通過（exit 0） |
| `test_pipeline_pairs.py` | 管線配對接線測試：全部通過（exit 0） |
| `test_pair_exceptions.py` | 配對例外離線測試：全部通過（exit 0） |
| `coman_handover_state.py` | 離線狀態序列測試：全部通過（exit 0） |
| `coman_couple_link.py` | 連接／解除握手離線測試：全部通過（exit 0） |
| `coman_pose_reader.py` | 位姿讀取離線測試：全部通過（exit 0） |
| `coman_pull_policy.py` | 拉動任務語意離線測試：全部通過（exit 0） |
| `coman_pull_target.py` | 拉動目標離線測試：全部通過（exit 0） |

零命令核對：屏障：最小餘量 -0.169301（列 199）  **零命令不可行** —— 與事故前一致。

## 暫存工作區

事故用的暫存目錄**已停用**，後續腳本不再引用；
內含一個指回 repo 的 `src` 目錄符號連結，就是事故成因。
依指示**不為清理它冒覆寫風險**。
