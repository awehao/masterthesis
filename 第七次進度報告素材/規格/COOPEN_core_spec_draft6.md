# 連續協同通行：核心段接觸驗證 — 規格 draft-6（接近段成本候選 w_a 0.01，g3 版）

draft-1～5 保留。本版依 Codex 20261027_011742（只試單一成本候選）與 r13 裁定（條件式核可單趟 `door_coopen_core_g3_1`）。
**閘控（r2）、門檻、持續窗、抓法、參考、門速、APPROACH 15 s 總逾時、接觸保護、求解模型、其他權重全部不變。** 只列與 draft-5 的差異。

## 1. 背景（離線證據，只在合成條件下成立）

### door_coopen_core_g_1（g2 跑法）
- 第二次閘控暫停期間，垂直誤差每 0.25 s 只降約 0.07 mm，15 s 期限內未達續進條件，以 timeout:APPROACH 中止。

### d̂_b（底盤偏差預測）
- 因果重播使底盤一步殘差下降約 40 倍。
- 但該殘差的來源是停車保持的小命令沒有被實現：保持命令約 (0.24, −0.89 mm/s, −2.38 mrad/s)，底盤 4 s 只移 0.05 mm。
- 摩擦／死區只是推測，未驗證。
- 合成閉迴路只提前 0.25 s ⇒ 保留程式與測試，**不接線**（Codex）。

### 單一候選 w_a 0.05 → 0.01（Codex 指定試驗選值，不是最佳值）
- 合成閉迴路設定：
  - 起點為 g_1 21.50 s 的記錄狀態；
  - 固定 u_hold；
  - 情境「保持未實現」：底盤偏差 ＝ −u_hold。
- 結果：續進 5.45 → 1.95 s，剩餘預算 4.14 s；無求解失敗。
- 入口暫態命令較粗糙：
  - 五點移動平均殘差 10.4 vs 2.2 mrad/s；
  - 命令一階差方向反轉 22 vs 4 次，集中在前 1.2 s；
  - 末 2 s 無方向反轉。
- 候選先由 3.09 升到 3.32 mm，0.5 s 後才持續下降。
- 回歸情境「固定保持命令被實現」：4.14 s 末誤差 0.78 mm，原值為 0.08 mm。這是該合成對照中的退化，照實保留。
- **不據此宣稱 Isaac 已無振盪風險。**

## 2. 變更（僅此一項成本，限接近段）

### 登錄 r7（`coopen_core_params_48_r7.json` ＝ r6 ＋ approach_cost）
```
approach_cost = {w_a: 0.01, phases: ['APPROACH'], latch_after_exit: true, coord_index: 12, launch_w_a: 0.05, restore_deadline_s: 0.55}
```
- r6 其餘鍵值不變；schema 改為 coopen_core_params/4。

### 任務節點 v4（`door_coopen_core_task_node4.py`）
- 相對 v3 只插入 9 行，標記 A1–A4。
- 每輪在 `phase_coord` 之後經 `door_coopen_approach_cost.ApproachCost.apply`：
  - APPROACH 且未閂鎖 ⇒ 協同訊息 coord[12] ＝ 0.01；
  - 其他相位原樣，off_coord 的 [12] 為 NaN ⇒ 求解端用啟動值 0.05。
- 離開 APPROACH 當輪即送 NaN，之後閂鎖，不再啟用。
- 閘控暫停期間相位仍是 APPROACH，因此同樣啟用。
- trace 每列記 acost；task.json 記 approach_cost（設定、模組 sha256、統計）。

### 求解節點 ADM9 不改
- 既有協同話題 v1 可逐筆覆寫 w_a。
- 給 NaN 或超過 `--coord-max-age-s` 0.5 s ⇒ 啟動值。
- 每輪紀錄 coord.{src, w_a}。

### 有界恢復
- 任務相位切換與求解端接收是非同步的，不宣稱物理步原子切換。
- 期限 ＝ 0.5 s ＋ 一個求解週期 0.05 s ＝ **0.55 s**。

## 3. 判定器 r5（`door_coopen_check_core5.py`）＝ r4（J1–J4）＋ 必要判準 J5

J5 判準如下，任一不符即不通過：

| 理由 | 條件 |
|---|---|
| acost_reg_invalid | 登錄不是 r7 值 |
| acost_task_mismatch | task.json 設定與登錄不符 |
| acost_launch_mismatch | 求解啟動 w_a ≠ 0.05，或 max_age＋1/rate ≠ 0.55 |
| acost_rows_missing | 求解紀錄缺列、缺 coord、w_a 非有限；trace 缺 |
| acost_not_reached | 從未進入 APPROACH（候選未受測） |
| acost_task_conflict | trace acost 與相位不符，或引導衝突 |
| acost_pre_approach | APPROACH 前不是 0.05 |
| acost_not_effective | APPROACH 期間沒有任何候選生效輪 |
| acost_dropout | [t_A0＋0.55, t_A1) 內有輪不是 0.01（候選中途失效） |
| acost_restore_late | t_A1＋0.55 之後仍有輪不是 0.05 |
| acost_open_phase | OPEN／SETTLE／PAUSE 相位的求解輪不是 0.05 |
| acost_restore_unverified | 已離開但之後沒有求解輪，且沒有停止證據 |

- 中止後求解端已停發（stop_rx 有值、停止後發布 0 筆）⇒ restore_status 記 `stopped_no_restore_rows`。不把缺少恢復輪當成正常恢復。

## 4. 測試（全過）

| 測試檔 | 內容 | 結果 |
|---|---|---|
| test_door_coopen_approach_cost | 7 項：APPROACH 只改 [12]、其他相位逐位不變、閂鎖、中止、引導衝突、登錄檢查、r7 實檔 | 全過 |
| test_door_coopen_check_core5 | 5 項：正常、12 種理由各一反例、中止停發、延遲接收窗內／窗外、g_1 實錄 | 全過 |
| test_door_coopen_core_g3_nodes | 4 項：v4 以 r7 啟動的紀錄與 NaN；r6 拒跑；v4 對 v3 只插入 9 行；g3 對 g2 只差 FRZ／PRM／spawn task／結束字樣 | 全過 |
| test_door_coopen_core_wa_switch_ros | ADM9 不改，假世界四情境（見下） | 全過 |
| test_door_coopen_core_wa_switch_offline | g_1 實錄快照，非零 u_prev＋暖啟動，見下 | 全過 |
| 回歸 | gate2 13、check_core4 4、core_sp_db 5、base_bias_est 8 | 全過 |

ROS 四情境：
- normal：≥ 6.55 s 為 0.05（topic）；
- dropout：≥ 6.55 s 為 stale_off 0.05；
- delayed：NaN 延到 6.3 s 才送，≥ 6.85 s 為 0.05；
- abort：5.02 s 收到停止，之後未再發布；
- 各情境：候選前 0.05、候選段 0.01、已發布輪皆 ok。
- 假世界權威 u_prev 為 0，只記錄。

離線切換（g_1 實錄快照，非零 u_prev＋暖啟動）：
- 0.05 與記錄 U_sol 逐位相同；
- 0.01 可行、底盤＝u_hold；
- 切回 0.05 與原值差 9.75e-6；
- 連續三快照 0.05／0.01／0.05 皆可行。

## 5. 實跑（單趟，條件已滿足後）
- `RUN_ID=door_coopen_core_g3_1`，`run_door_coopen_core_g3.sh`，凍結 `freeze_door_coopen_core_g_dev_r3`（最後建；跑前、跑後核對）。
- 乾淨環境、92 °C 守護。
- 失敗保留，不補跑；不新增影片。
- 只驗接近、持握開門與出口繞行核心段；不放手、不收臂、不穿越。
- 不追加權重、不接 d̂_b。
